"""
koi/server.py — FastAPI HTTP service for Koi v2.

Endpoints:
  POST /decide         → agent placement decision
  POST /job/complete   → webhook from Orca on job completion
  GET  /health         → service health
  GET  /jobs           → tracked jobs status

Usage:
  ANTHROPIC_API_KEY=sk-ant-... python -m koi.server
"""

import asyncio
import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from koi.agent import KoiAgent
from koi.contract import ReasonCode
from koi.event_tap import emit_event
from koi.logging_config import setup_logging, get_logger, bind_context, clear_context
from koi.monitor import MonitoringLoop
from koi.resource_ledger import ResourceLedger
from koi.runtime_state import ClaimResult, RuntimeStateStore
from koi.schemas import (
    AgentDecision,
    DataSource,
    EngineConfig,
    JobRequest,
    MonitoringStatus,
    MonitoringTrigger,
    PlacementConfig,
)
from koi.tools.memory import AgenticMemory
from koi.tools.orca_api import OrcaClient
from koi.tools.perfdb import PerfDB
from koi.tools.resources import parse_orca_resources

logger = get_logger("koi.server")

KOI_PORT = int(os.environ.get("KOI_PORT", "8090"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Envelope (Optional during compat window — see koi/contract.py)
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None

    job_request: Dict[str, Any]
    resource_map: Any  # Shape A, B, or C


class JobCompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Envelope (Optional during compat window)
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None

    job_id: str
    # Explicit entity IDs (new; job_id retained as legacy alias for group_id)
    group_id: Optional[str] = None
    decision_id: Optional[str] = None

    status: str
    metrics: Dict[str, Any] = {}

    # Structured failure info (Optional; free-text status/metrics still primary)
    reason_code: Optional[ReasonCode] = None
    reason_detail: Optional[str] = None


class _FixedTestAgent:
    """Deterministic test-only agent for server-backed sim scenarios."""

    def __init__(self, model: str):
        self.model = f"{model}-test-fake"
        self.monitor = None
        self.required_gpus = int(os.environ.get("KOI_TEST_REQUIRED_GPUS", "8"))
        self.preferred_gpu = os.environ.get("KOI_TEST_GPU_TYPE", "L40S")
        self.decide_delay = float(os.environ.get("KOI_TEST_DECIDE_DELAY_SEC", "0.05"))

    async def decide(self, job_request: JobRequest, resource_map) -> AgentDecision:
        await asyncio.sleep(self.decide_delay)

        resource = resource_map.get_resource(self.preferred_gpu)
        if resource is None:
            resource = next(
                (
                    r
                    for r in resource_map.resources
                    if r.available_gpus >= self.required_gpus
                ),
                None,
            )

        if not resource or resource.available_gpus < self.required_gpus:
            raise RuntimeError("insufficient adjusted resources")

        tp = min(resource.gpus_per_instance, self.required_gpus)
        pp = max(1, self.required_gpus // max(tp, 1))
        num_instances = max(1, self.required_gpus // max(resource.gpus_per_instance, 1))
        cost_per_hour = resource.cost_per_instance_hour_usd * num_instances
        total_tokens = job_request.total_tokens or 0
        predicted_tps = float(os.environ.get("KOI_TEST_PREDICTED_TPS", "1200"))
        runtime_hours = (
            (total_tokens / predicted_tps / 3600)
            if total_tokens and predicted_tps > 0
            else None
        )
        total_cost = (
            cost_per_hour * runtime_hours if runtime_hours is not None else None
        )

        return AgentDecision(
            job_id=job_request.job_id or f"test-job-{uuid.uuid4().hex[:8]}",
            model_name=job_request.model_name,
            config=PlacementConfig(
                gpu_type=resource.gpu_type,
                instance_type=resource.instance_type,
                num_gpus=self.required_gpus,
                num_instances=num_instances,
                tp=tp,
                pp=pp,
                dp=1,
                region=resource.region,
                engine_config=EngineConfig(
                    tensor_parallel_size=tp,
                    pipeline_parallel_size=pp,
                ),
            ),
            predicted_tps=predicted_tps,
            predicted_cost_per_hour=cost_per_hour,
            predicted_total_cost=total_cost,
            predicted_runtime_hours=runtime_hours,
            reasoning=f"[TEST FAKE DECIDE] available_gpus={resource.available_gpus}",
            confidence=0.99,
            data_source=DataSource.ANALYTICAL,
            agent_model=self.model,
        )

    async def handle_trigger(self, trigger: MonitoringTrigger) -> str:
        return f"[TEST FAKE TRIGGER] {trigger.trigger_type.value}"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    perfdb_path = os.environ.get("KOI_PERFDB_PATH", "./perfdb/perfdb_all.csv")
    memory_path = os.environ.get("KOI_MEMORY_PATH", "./data/koi_memory.db")
    runtime_state_path = os.environ.get(
        "KOI_RUNTIME_STATE_PATH", "./data/koi_runtime.db"
    )
    orca_url = os.environ.get("ORCA_URL", "")
    provider = os.environ.get("KOI_LLM_PROVIDER", "openrouter").lower()
    base_url = os.environ.get("KOI_BASE_URL")
    # KOI_LLM_MODEL is a deprecated alias for KOI_AGENT_MODEL.
    legacy_model = os.environ.get("KOI_LLM_MODEL")
    model = os.environ.get("KOI_AGENT_MODEL") or legacy_model
    if legacy_model and not os.environ.get("KOI_AGENT_MODEL"):
        logger.warning(
            "koi_llm_model_deprecated",
            message="KOI_LLM_MODEL is deprecated; use KOI_AGENT_MODEL",
        )
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    else:
        api_key = (
            os.environ.get("KOI_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
        )

    # Validate the API key at startup — but only when the real agent will
    # actually be constructed. Sim / CI / KOI_TEST_FAKE_DECIDE=1 paths
    # bypass the agent entirely and must keep working without a key.
    if os.environ.get("KOI_TEST_FAKE_DECIDE") != "1" and not api_key:
        expected = (
            "ANTHROPIC_API_KEY" if provider == "anthropic"
            else "KOI_API_KEY (or OPENROUTER_API_KEY)"
        )
        raise RuntimeError(
            f"{expected} is required to start Koi with KOI_LLM_PROVIDER="
            f"{provider!r}. Set the env var, or set KOI_TEST_FAKE_DECIDE=1 "
            f"for sim/CI paths that don't need an LLM."
        )

    # PerfDB
    try:
        app.state.perfdb = PerfDB(perfdb_path)
        logger.info(
            "perfdb_loaded",
            records=app.state.perfdb.record_count,
            models=app.state.perfdb.models,
            gpus=app.state.perfdb.gpu_types,
        )
    except Exception as e:
        logger.warning("perfdb_load_failed", error=str(e))
        app.state.perfdb = None

    # Memory
    app.state.memory = AgenticMemory(db_path=memory_path)
    logger.info(
        "memory_loaded",
        decisions=app.state.memory.decision_count(),
        outcomes=app.state.memory.outcome_count(),
    )

    # Resource ledger (pending GPU reservations)
    app.state.runtime_state = RuntimeStateStore(runtime_state_path)
    app.state.ledger = ResourceLedger(runtime_state=app.state.runtime_state)
    app.state.decide_lock = asyncio.Lock()

    # Orca client
    app.state.session = aiohttp.ClientSession()
    app.state.orca = (
        OrcaClient(orca_url, session=app.state.session) if orca_url else None
    )
    if orca_url:
        logger.info("orca_client_ready", url=orca_url)
    else:
        logger.info("orca_not_configured")

    # Agent
    if os.environ.get("KOI_TEST_FAKE_DECIDE") == "1":
        app.state.agent = _FixedTestAgent(model=model or "test")
        logger.warning(
            "agent_ready_test_mode",
            model=app.state.agent.model,
            required_gpus=app.state.agent.required_gpus,
            preferred_gpu=app.state.agent.preferred_gpu,
        )
    else:
        app.state.agent = KoiAgent(
            perfdb=app.state.perfdb,
            memory=app.state.memory,
            orca=app.state.orca,
            ledger=app.state.ledger,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
        )
        logger.info("agent_ready", provider=provider, model=app.state.agent.model)

    # Monitor
    app.state.monitor = MonitoringLoop(
        orca=app.state.orca,
        on_trigger=app.state.agent.handle_trigger,
        runtime_state=app.state.runtime_state,
    )
    app.state.agent.monitor = app.state.monitor

    restored_ledger = app.state.ledger.restore()
    restored_monitor = app.state.monitor.restore_runtime_state()
    logger.info(
        "runtime_state_restored",
        ledger_reservations=restored_ledger,
        tracked_jobs=restored_monitor["tracked_jobs"],
        pending_launches=restored_monitor["pending_launches"],
        pending_replica_decisions=restored_monitor["pending_replica_decisions"],
    )

    if app.state.orca:
        await app.state.monitor.start()
        logger.info("monitor_started")
    else:
        logger.info("monitor_not_started", reason="no_orca_connection")

    yield

    await app.state.monitor.stop()
    if app.state.session and not app.state.session.closed:
        await app.state.session.close()
    logger.info("shutdown_complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Koi Placement Service", version="2.0", lifespan=lifespan)


def _finite_or_none(v: Optional[float], ndigits: Optional[int] = None) -> Optional[float]:
    """Sanitize a float for JSON: replace nan/inf with None, optionally round."""
    if v is None or not math.isfinite(v):
        return None
    return round(v, ndigits) if ndigits is not None else v


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    bind_context(request_id=rid)
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response
    finally:
        clear_context()


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

_FAILURE_PATTERNS = [
    (re.compile(r"spot|preempt", re.I), "spot_preemption"),
    (re.compile(r"insufficient.?capacity|no.?capacity", re.I), "no_capacity"),
    (re.compile(r"oom|out.?of.?memory|cuda.?oom", re.I), "oom"),
    (re.compile(r"quota", re.I), "quota"),
]


def _classify_failure(reason: str) -> str:
    """Map Orca's raw failure reason to a structured category."""
    for pattern, category in _FAILURE_PATTERNS:
        if pattern.search(reason):
            return category
    return "unknown"


# ---------------------------------------------------------------------------
# Inbox wrapping
# ---------------------------------------------------------------------------


def _hash_payload(payload: Dict[str, Any]) -> str:
    """Stable sha256 hash of a payload — used for inbox payload_hash."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _synthesize_legacy_event_id(
    payload: Dict[str, Any], event_type: str, job_id: str
) -> str:
    """Derive a stable event_id from a legacy (pre-envelope) payload.

    Shape: 'legacy:<event_type>:<job_id>:<hash16>' — same payload produces
    the same id, so a legacy Orca retrying the identical request still dedups.
    """
    return f"legacy:{event_type}:{job_id}:{_hash_payload(payload)[:16]}"


async def _run_with_inbox(
    req: BaseModel,
    event_type: str,
    job_id: str,
    handler: Callable[[], Awaitable[Any]],
) -> Any:
    """Claim the event, run the handler, mark processed on success.

    Crash discipline: if the handler raises, the row stays in 'processing'
    with last_error set. Orca's next retry, once the reclaim window elapses,
    sees the claim as stale and re-runs the handler. No events lost.
    """
    runtime: Optional[RuntimeStateStore] = getattr(app.state, "runtime_state", None)
    if runtime is None:
        # Inbox not configured (dev harness / legacy embed) — pass-through.
        return await handler()
    payload = req.model_dump()
    event_id = getattr(req, "event_id", None) or _synthesize_legacy_event_id(
        payload, event_type, job_id
    )
    payload_hash = _hash_payload(payload)

    result = runtime.claim_event(
        event_id=event_id,
        event_type=event_type,
        job_id=job_id,
        payload_hash=payload_hash,
    )
    if result == ClaimResult.ALREADY_PROCESSED:
        logger.info(
            "inbox_duplicate_ignored",
            event_id=event_id,
            event_type=event_type,
            job_id=job_id,
        )
        return {"status": "duplicate_ignored", "event_id": event_id}
    if result == ClaimResult.IN_FLIGHT:
        logger.info(
            "inbox_in_flight",
            event_id=event_id,
            event_type=event_type,
            job_id=job_id,
        )
        # 503 so Orca keeps retrying. If the prior claim crashed, it will
        # age out past reclaim_after_secs and a later retry reclaims it as
        # RECLAIMED_STALE. 200 here would purge the outbox row and lose
        # the event.
        return JSONResponse(
            status_code=503,
            content={"status": "in_flight", "event_id": event_id},
        )
    if result == ClaimResult.RECLAIMED_STALE:
        logger.warning(
            "inbox_reclaimed_stale",
            event_id=event_id,
            event_type=event_type,
            job_id=job_id,
        )
    try:
        response = await handler()
        runtime.mark_processed(event_id)
        return response
    except Exception as exc:
        runtime.mark_failed(event_id, repr(exc))
        raise


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    monitor = app.state.monitor
    runtime: Optional[RuntimeStateStore] = getattr(app.state, "runtime_state", None)

    body = {
        "status": "ok",
        "version": "2.0",
        "perfdb_records": app.state.perfdb.record_count if app.state.perfdb else 0,
        "memory_decisions": app.state.memory.decision_count(),
        "memory_outcomes": app.state.memory.outcome_count(),
        "tracked_jobs": len(monitor.tracked_jobs),
        "agent_model": app.state.agent.model,
        "orca_connected": app.state.orca is not None,
    }
    if runtime is not None:
        body["inbox_processed"] = runtime.inbox_count(status="processed")
        body["inbox_processing"] = runtime.inbox_count(status="processing")
        body["stale_inbox_claims"] = runtime.inbox_stale_count(older_than_secs=300)

    fatal = getattr(monitor, "_fatal", None)
    if fatal:
        body["status"] = "fatal"
        body["fatal"] = fatal
        return JSONResponse(status_code=503, content=body)
    return body


@app.post("/decide")
async def decide(req: DecideRequest):
    """Run the Koi agent to make a placement decision."""
    agent: KoiAgent = app.state.agent

    # Parse job request
    try:
        from koi.schemas import TaskType, Objective

        d = req.job_request
        allowed_job_request_keys = {
            "job_id",
            "model_name",
            "task_type",
            "avg_input_tokens",
            "avg_output_tokens",
            "num_requests",
            "expected_concurrency",
            "slo_deadline_hours",
            "slo_tpot_ms",
            "slo_ttft_ms",
            "objective",
            "preferred_gpu_types",
            "max_total_gpus",
            "cost_roofline_usd",
            "region",
            "preferred_market",
            "quantization",
        }
        unknown_job_request_keys = sorted(set(d.keys()) - allowed_job_request_keys)
        if unknown_job_request_keys:
            raise ValueError(
                "Unknown job_request fields: " + ", ".join(unknown_job_request_keys)
            )
        job_request_payload = {
            "model_name": str(d.get("model_name", "unknown")),
            "task_type": TaskType(d.get("task_type", "batch")),
            "avg_input_tokens": int(d.get("avg_input_tokens", 512)),
            "avg_output_tokens": int(d.get("avg_output_tokens", 256)),
            "num_requests": int(d["num_requests"]) if d.get("num_requests") else None,
            "slo_deadline_hours": float(d["slo_deadline_hours"])
            if d.get("slo_deadline_hours")
            else None,
            "objective": Objective(d.get("objective", "cheapest")),
            "preferred_gpu_types": d.get("preferred_gpu_types"),
            "max_total_gpus": int(d["max_total_gpus"])
            if d.get("max_total_gpus") is not None
            else None,
            "cost_roofline_usd": float(d["cost_roofline_usd"])
            if d.get("cost_roofline_usd") is not None
            else None,
            "region": d.get("region"),
            "preferred_market": d.get("preferred_market"),
            "quantization": d.get("quantization"),
        }
        if d.get("job_id"):
            job_request_payload["job_id"] = str(d["job_id"])
        job_request = JobRequest(**job_request_payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid job_request: {e}")

    # Parse resource map and subtract pending reservations
    try:
        resource_map = parse_orca_resources(req.resource_map)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    decide_lock = getattr(app.state, "decide_lock", None)
    if decide_lock is None:
        decide_lock = asyncio.Lock()
        app.state.decide_lock = decide_lock

    # Serialize resource-adjusted decisions so concurrent requests cannot
    # subtract from the same pre-reservation snapshot and double-book GPUs.
    async with decide_lock:
        resource_map = app.state.ledger.apply_to_resource_map(resource_map)

        # Run agent
        try:
            decision = await agent.decide(job_request, resource_map)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Agent decision timed out")
        except Exception as e:
            logger.error("agent_error", error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent error: {e}")

        # Record decision in memory
        decision_id = app.state.memory.record_decision(
            job_id=decision.job_id,
            model_name=decision.model_name,
            instance_type=decision.config.instance_type,
            gpu_type=decision.config.gpu_type,
            tp=decision.config.tp,
            pp=decision.config.pp,
            dp=decision.config.dp,
            num_gpus=decision.config.num_gpus,
            predicted_tps=decision.predicted_tps,
            predicted_cost_per_hour=decision.predicted_cost_per_hour,
            predicted_total_cost=decision.predicted_total_cost,
            predicted_runtime_hours=decision.predicted_runtime_hours,
            prediction_confidence=decision.confidence,
            prediction_source=decision.data_source.value,
            slo_deadline_hours=job_request.slo_deadline_hours or 0,
            objective=job_request.objective.value,
            avg_input_tokens=job_request.avg_input_tokens,
            avg_output_tokens=job_request.avg_output_tokens,
            num_requests=job_request.num_requests,
            triggered_by="user",
            cost_roofline_usd=job_request.cost_roofline_usd,
            market=decision.planned_market,
        )

        # Reserve GPUs in ledger (pending until /job/started confirms)
        app.state.ledger.reserve(
            decision_id=decision_id,
            gpu_type=decision.config.gpu_type,
            num_gpus=decision.config.num_gpus,
            region=decision.config.region,
            instance_type=decision.config.instance_type,
        )

    # NOTE: Do NOT register in monitor here. The job hasn't launched yet.
    # Orca will call POST /job/started after successful launch.

    # Include decision_id in response so Orca can pass it back
    result = decision.model_dump(mode="json")
    result["_decision_id"] = decision_id
    return result


class JobStartedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Envelope (Optional during compat window)
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None

    job_id: str
    # Explicit entity IDs (new; job_id retained as legacy alias for replica_id)
    replica_id: Optional[str] = None
    scale_request_id: Optional[str] = None

    decision_id: Optional[str] = None
    group_id: Optional[str] = None  # parent job ID for chunked replicas
    gpu_type: str
    instance_type: str
    region: str = "unknown"
    market: str = "unknown"
    tp: int
    pp: int
    dp: int = 1
    slo_deadline_hours: float
    total_tokens: int
    predicted_tps: Optional[float] = 0.0
    is_fallback: bool = False  # True if Orca used a fallback config (not primary)


class JobLaunchingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Envelope (Optional during compat window)
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None

    job_id: str
    replica_id: Optional[str] = None
    scale_request_id: Optional[str] = None

    decision_id: Optional[str] = None
    group_id: Optional[str] = None
    gpu_type: str = "unknown"
    instance_type: str = "unknown"
    tp: int = 1
    pp: int = 1
    region: str = "unknown"
    market: str = "unknown"
    attempt_index: int = 0


class JobLaunchHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Envelope (Optional during compat window)
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None

    job_id: str
    replica_id: Optional[str] = None
    scale_request_id: Optional[str] = None

    decision_id: Optional[str] = None
    group_id: Optional[str] = None
    gpu_type: str = "unknown"
    instance_type: str = "unknown"
    tp: int = 1
    pp: int = 1
    region: str = "unknown"
    market: str = "unknown"
    attempt_index: int = 0
    phase: str
    message: str = ""
    timestamp: Optional[float] = None


@app.post("/job/launching")
async def job_launching(req: JobLaunchingRequest):
    """Called by Orca when a replica is provisioned but not yet serving.

    Gives Koi early visibility into GPU spend before model_ready.
    """

    async def _do() -> Dict[str, Any]:
        monitor: MonitoringLoop = app.state.monitor
        now = time.time()
        if req.decision_id:
            app.state.ledger.touch(req.decision_id)
        monitor.track_pending_launch(
            req.job_id,
            {
                "decision_id": req.decision_id,
                "group_id": req.group_id,
                "gpu_type": req.gpu_type,
                "instance_type": req.instance_type,
                "tp": req.tp,
                "pp": req.pp,
                "region": req.region,
                "market": req.market,
                "attempt_index": req.attempt_index,
                "launch_phase": "waiting_model_ready",
                "launch_message": "Replica provisioned, waiting for model_ready",
                "launched_at": now,
                "last_heartbeat_at": now,
            },
        )
        logger.info(
            "job_launching",
            job_id=req.job_id,
            group_id=req.group_id,
            gpu_type=req.gpu_type,
            instance_type=req.instance_type,
        )
        emit_event(
            "job_launching",
            job_id=req.job_id,
            group_id=req.group_id,
            gpu_type=req.gpu_type,
            instance_type=req.instance_type,
        )
        return {"status": "tracked", "job_id": req.job_id}

    return await _run_with_inbox(req, "job_launching", req.job_id, _do)


@app.post("/job/launch-heartbeat")
async def job_launch_heartbeat(req: JobLaunchHeartbeatRequest):
    """Called by Orca while a replica is still searching/provisioning/bootstrapping."""

    async def _do() -> Dict[str, Any]:
        monitor: MonitoringLoop = app.state.monitor
        heartbeat_at = req.timestamp or time.time()
        refreshed = False
        if req.decision_id:
            refreshed = app.state.ledger.touch(req.decision_id)
        monitor.track_pending_launch(
            req.job_id,
            {
                "decision_id": req.decision_id,
                "group_id": req.group_id,
                "gpu_type": req.gpu_type,
                "instance_type": req.instance_type,
                "tp": req.tp,
                "pp": req.pp,
                "region": req.region,
                "market": req.market,
                "attempt_index": req.attempt_index,
                "launch_phase": req.phase,
                "launch_message": req.message,
                "last_heartbeat_at": heartbeat_at,
            },
        )
        logger.info(
            "job_launch_heartbeat",
            job_id=req.job_id,
            group_id=req.group_id,
            phase=req.phase,
            attempt_index=req.attempt_index,
            refreshed=refreshed,
        )
        emit_event(
            "job_launch_heartbeat",
            job_id=req.job_id,
            group_id=req.group_id,
            phase=req.phase,
            attempt_index=req.attempt_index,
            refreshed=refreshed,
        )
        return {"status": "tracked", "job_id": req.job_id, "lease_refreshed": refreshed}

    return await _run_with_inbox(req, "job_launch_heartbeat", req.job_id, _do)


@app.post("/job/started")
async def job_started(req: JobStartedRequest):
    """Called by Orca AFTER a job successfully launches. Registers in monitor."""

    async def _do() -> Dict[str, Any]:
        return await _job_started_impl(req)

    return await _run_with_inbox(req, "job_started", req.job_id, _do)


async def _job_started_impl(req: JobStartedRequest) -> Dict[str, Any]:
    monitor: MonitoringLoop = app.state.monitor
    memory: AgenticMemory = app.state.memory

    pending_launch = monitor.get_pending_launch(req.job_id)
    resolved_region = (
        req.region
        if req.region != "unknown"
        else pending_launch.get("region", "unknown")
    )
    resolved_market = (
        req.market
        if req.market != "unknown"
        else pending_launch.get("market", "unknown")
    )

    # Clear pending launch tracking — replica reached model_ready
    monitor.clear_pending_launch(req.job_id)

    # Release pending reservation — Orca confirmed, its next GET /resources reflects it
    if req.decision_id:
        app.state.ledger.release(req.decision_id)

    # Link scale-up replicas to the exact decision that produced them.
    # scale_chain_tool registers (replica_id → decision) for every
    # new_replica Orca returned; here we consume it by exact replica_id.
    # No FIFO guesswork, so overlapping scale ops can't cross-attribute.
    replica_id_key = req.replica_id or req.job_id
    pending_replica_decisions = getattr(
        monitor, "_pending_replica_decisions", None
    )
    if isinstance(pending_replica_decisions, dict):
        pending = monitor.consume_pending_replica_decision(replica_id_key)
        if pending:
            req.decision_id = pending["decision_id"]

    # Detect fallback: if Orca used a different config than Koi's primary decision,
    # create a child decision so the outcome links to the ACTUAL config, not the intended one.
    actual_decision_id = req.decision_id
    if req.is_fallback and req.decision_id:
        original = memory.get_decision(req.decision_id)
        if original:
            actual_decision_id = memory.record_decision(
                job_id=req.job_id,
                model_name=original["model_name"],
                instance_type=req.instance_type,
                gpu_type=req.gpu_type,
                tp=req.tp,
                pp=req.pp,
                dp=req.dp,
                num_gpus=req.tp * req.pp * req.dp,
                predicted_tps=req.predicted_tps or 0,
                predicted_cost_per_hour=original.get("predicted_cost_per_hour", 0.0)
                or 0.0,
                slo_deadline_hours=req.slo_deadline_hours,
                objective=original.get("objective", "cheapest"),
                avg_input_tokens=original.get("avg_input_tokens", 0),
                avg_output_tokens=original.get("avg_output_tokens", 0),
                num_requests=original.get("num_requests"),
                triggered_by="fallback",
                parent_decision_id=req.decision_id,
                cost_roofline_usd=original.get("cost_roofline_usd"),
                market=(
                    resolved_market
                    if resolved_market != "unknown"
                    else original.get("market", "unknown")
                ),
            )
            logger.info(
                "fallback_detected",
                original_decision=req.decision_id,
                actual_decision=actual_decision_id,
                gpu_type=req.gpu_type,
                tp=req.tp,
                pp=req.pp,
            )

    config = PlacementConfig(
        gpu_type=req.gpu_type,
        instance_type=req.instance_type,
        num_gpus=req.tp * req.pp * req.dp,
        num_instances=max(1, (req.tp * req.pp * req.dp) // 8),
        tp=req.tp,
        pp=req.pp,
        dp=req.dp,
        region=resolved_region,
        engine_config=EngineConfig(
            tensor_parallel_size=req.tp,
            pipeline_parallel_size=req.pp,
        ),
        market=resolved_market,
    )

    decision_meta = memory.get_decision(actual_decision_id) if actual_decision_id else None

    monitor.register_job(
        job_id=req.job_id,
        config=config,
        slo_deadline_hours=req.slo_deadline_hours,
        total_tokens=req.total_tokens,
        predicted_tps=req.predicted_tps,
        predicted_cost_per_hour=(
            float(decision_meta.get("predicted_cost_per_hour") or 0.0)
            if decision_meta
            else None
        ),
        cost_roofline_usd=(
            float(decision_meta.get("cost_roofline_usd"))
            if decision_meta and decision_meta.get("cost_roofline_usd") is not None
            else None
        ),
        decision_id=actual_decision_id,
        group_id=req.group_id,
    )

    # Unfreeze anti-windup for all trackers in this group (new replica is ready)
    if req.group_id:
        for tracker in monitor.tracked_jobs.values():
            if tracker.group_id == req.group_id and tracker.action_in_progress:
                tracker.action_in_progress = False
                tracker.action_freeze_until = None
                monitor.persist_job(tracker.job_id)
                logger.info(
                    "anti_windup_unfrozen",
                    tracker_job=tracker.job_id,
                    new_replica=req.job_id,
                )

    # Update availability prior (success observation)
    if resolved_region != "unknown" and resolved_market != "unknown":
        memory.update_availability(
            gpu_type=req.gpu_type,
            region=resolved_region,
            market=resolved_market,
            launched=True,
        )
    else:
        logger.warning(
            "job_started_missing_launch_context",
            job_id=req.job_id,
            region=resolved_region,
            market=resolved_market,
        )

    logger.info(
        "job_started",
        job_id=req.job_id,
        gpu_type=req.gpu_type,
        tp=req.tp,
        pp=req.pp,
        group_id=req.group_id,
        is_fallback=req.is_fallback,
        region=resolved_region,
        market=resolved_market,
    )
    return {
        "status": "registered",
        "job_id": req.job_id,
        "group_id": req.group_id,
        "decision_id": actual_decision_id,
    }


@app.post("/job/complete")
async def job_complete(req: JobCompleteRequest):
    """Webhook from Orca when a job completes.

    Two modes:
      1. Single-chain job: req.job_id matches a tracked chain directly
      2. Job group (chunked): req.job_id matches the group_id of multiple chains
         → records PER-CHAIN outcomes across all chains
    """

    async def _do() -> Dict[str, Any]:
        return await _job_complete_impl(req)

    return await _run_with_inbox(req, "job_complete", req.job_id, _do)


async def _job_complete_impl(req: JobCompleteRequest) -> Dict[str, Any]:
    monitor: MonitoringLoop = app.state.monitor
    memory: AgenticMemory = app.state.memory

    # Mode 1: direct chain match (single-cluster job)
    tracker = monitor.tracked_jobs.get(req.job_id)
    if tracker and not tracker.group_id:
        actual_tps = req.metrics.get("avg_generation_throughput_toks_per_s")
        actual_cost_per_hour = req.metrics.get("cost_per_hour")

        if tracker.decision_id:
            outcome_id = memory.record_outcome(
                decision_id=tracker.decision_id,
                job_id=req.job_id,
                status=req.status,
                actual_tps=actual_tps,
                actual_cost_per_hour=actual_cost_per_hour,
                actual_runtime_hours=tracker.elapsed_hours,
                slo_met=req.status == "succeeded",
                slo_headroom_pct=tracker.slo_headroom_pct,
            )
            logger.info(
                "outcome_recorded",
                outcome_id=outcome_id,
                job_id=req.job_id,
                status=req.status,
            )

        monitor.unregister_job(req.job_id)
        return {"status": "recorded", "job_id": req.job_id}

    # Mode 2: job group completion (chunked job)
    group_chains = monitor.get_group_chains(req.job_id)
    if group_chains:
        total_tps = sum(t.smoothed_tps for t in group_chains.values())
        max_elapsed = (
            max(t.elapsed_hours for t in group_chains.values()) if group_chains else 0
        )

        # Record PER-CHAIN outcomes so the learning signal stays clean.
        # For single-chain groups, use Orca's aggregate TPS (true average over whole run).
        # For multi-chain groups, use per-chain EMA (best we have per-replica).
        orca_aggregate_tps = req.metrics.get("throughput_tokens_per_sec")
        use_orca_tps = orca_aggregate_tps and len(group_chains) == 1

        outcomes_recorded = 0
        for chain_id, tracker in group_chains.items():
            if not tracker.decision_id:
                continue
            chain_tps = orca_aggregate_tps if use_orca_tps else tracker.smoothed_tps
            memory.record_outcome(
                decision_id=tracker.decision_id,
                job_id=req.job_id,
                status=req.status,
                actual_tps=chain_tps,
                actual_runtime_hours=tracker.elapsed_hours,
                slo_met=req.status == "succeeded",
                slo_headroom_pct=tracker.slo_headroom_pct,
            )
            outcomes_recorded += 1

        logger.info(
            "group_completed",
            job_id=req.job_id,
            outcomes=outcomes_recorded,
            aggregate_tps=round(total_tps),
            status=req.status,
        )
        emit_event(
            "group_completed",
            job_id=req.job_id,
            outcomes=outcomes_recorded,
            aggregate_tps=round(total_tps),
            status=req.status,
        )

        # Unregister all chains in the group
        monitor.unregister_group(req.job_id)
        return {
            "status": "recorded",
            "job_id": req.job_id,
            "chains_closed": len(group_chains),
            "outcomes_recorded": outcomes_recorded,
            "aggregate_tps": round(total_tps, 1),
        }

    logger.warning("job_complete_unknown", job_id=req.job_id)
    return {"status": "unknown_job", "job_id": req.job_id}


class ReplicaFailedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Envelope (Optional during compat window)
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None

    job_id: str  # replica chain ID
    replica_id: Optional[str] = None  # alias for job_id; preferred going forward
    group_id: str  # parent job ID
    decision_id: Optional[str] = None
    instance_type: str = "unknown"
    region: str = "unknown"
    market: str = "unknown"
    status: str = "failed"
    reason: str = ""

    # Structured reason (preferred over free-text `reason` going forward)
    reason_code: Optional[ReasonCode] = None
    reason_detail: Optional[str] = None


@app.post("/job/replica-failed")
async def replica_failed(req: ReplicaFailedRequest):
    """Called by Orca when a replica dies mid-job. Triggers agent for diagnosis."""

    async def _do() -> Dict[str, Any]:
        return await _replica_failed_impl(req)

    return await _run_with_inbox(req, "replica_failed", req.job_id, _do)


async def _replica_failed_impl(req: ReplicaFailedRequest) -> Dict[str, Any]:
    monitor: MonitoringLoop = app.state.monitor
    tracker = monitor.tracked_jobs.get(req.job_id)
    if not tracker:
        logger.warning("replica_failed_unknown", job_id=req.job_id)
        return {"status": "unknown", "job_id": req.job_id}

    # Dedup: if already FAILED, don't re-process (watchdog + launcher can both fire)
    if tracker.status == MonitoringStatus.FAILED:
        logger.info("replica_failed_dedup", job_id=req.job_id)
        return {"status": "already_failed", "job_id": req.job_id}

    # Check if this was an intentional kill (from scale_chain_tool)
    if req.job_id in monitor._koi_initiated_kills:
        monitor._koi_initiated_kills.discard(req.job_id)
        tracker.status = MonitoringStatus.COMPLETED
        tracker.smoothed_tps = 0
        if req.job_id not in tracker.dead_replicas:
            tracker.dead_replicas.append(req.job_id)
        monitor.persist_job(req.job_id)
        logger.info("intentional_kill_ack", job_id=req.job_id)
        return {"status": "intentional_kill", "job_id": req.job_id}

    # Capture TPS before zeroing — valuable ground truth for learning
    actual_tps_before_death = tracker.smoothed_tps

    tracker.status = MonitoringStatus.FAILED
    tracker.smoothed_tps = 0
    if req.job_id not in tracker.dead_replicas:
        tracker.dead_replicas.append(req.job_id)

    if req.instance_type != "unknown":
        tracker.config.instance_type = req.instance_type
    if req.region != "unknown":
        tracker.config.region = req.region
    if req.market != "unknown":
        tracker.config.market = req.market

    monitor.persist_job(req.job_id)

    failure_category = _classify_failure(req.reason)
    region = req.region if req.region != "unknown" else tracker.config.region
    market = req.market if req.market != "unknown" else tracker.config.market
    if market == "unknown" and failure_category == "spot_preemption":
        market = "spot"
    # Record failure outcome in memory for learning (with actual TPS if available)
    memory: AgenticMemory = app.state.memory
    if tracker.decision_id:
        memory.record_outcome(
            decision_id=tracker.decision_id,
            job_id=req.group_id,
            status="replica_failed",
            actual_tps=actual_tps_before_death if actual_tps_before_death > 0 else None,
            failure_category=failure_category,
            diagnosis=req.reason[:200],
        )
    # Update availability prior (failure observation)
    if region != "unknown" and market != "unknown":
        memory.update_availability(
            gpu_type=tracker.config.gpu_type,
            region=region,
            market=market,
            launched=False,
        )
    else:
        logger.warning(
            "replica_failed_missing_launch_context",
            job_id=req.job_id,
            region=region,
            market=market,
            failure_category=failure_category,
        )
    # Emit FAILED trigger to agent
    trigger = MonitoringTrigger(
        trigger_type=MonitoringStatus.FAILED,
        job_id=req.job_id,
        job_tracker=tracker.model_dump(),
        diagnosis_hint=f"Replica died: {req.reason[:200]}",
    )
    await monitor._trigger_queue.put(trigger)
    logger.info(
        "replica_failed",
        job_id=req.job_id,
        group_id=req.group_id,
        reason=req.reason[:100],
    )
    emit_event(
        "replica_failed",
        job_id=req.job_id,
        group_id=req.group_id,
        reason=req.reason[:100],
    )
    return {"status": "trigger_emitted", "job_id": req.job_id}


class ConfigAttemptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Envelope (Optional during compat window)
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None

    job_id: str
    group_id: Optional[str] = None
    decision_id: Optional[str] = None
    instance_type: str
    gpu_type: str
    region: str
    market: str = "unknown"
    launched: bool
    failure_reason: str = ""
    time_to_launch: float = 0
    attempt_index: int = 0


@app.post("/job/config-attempted")
async def config_attempted(req: ConfigAttemptRequest):
    """Called by Orca for EACH allocation attempt (success or failure)."""

    async def _do() -> Dict[str, Any]:
        return await _config_attempted_impl(req)

    return await _run_with_inbox(req, "config_attempted", req.job_id, _do)


async def _config_attempted_impl(req: ConfigAttemptRequest) -> Dict[str, Any]:
    memory: AgenticMemory = app.state.memory
    memory.record_launch_attempt(
        decision_id=req.decision_id or f"unknown-{req.job_id}",
        job_id=req.job_id,
        instance_type=req.instance_type,
        gpu_type=req.gpu_type,
        region=req.region,
        market=req.market,
        count=1,
        launched=req.launched,
        time_to_launch=req.time_to_launch if req.launched else None,
        failure_reason=req.failure_reason if not req.launched else None,
        failure_category=_classify_failure(req.failure_reason)
        if not req.launched
        else None,
    )
    if req.region != "unknown" and req.market != "unknown":
        memory.update_availability(
            gpu_type=req.gpu_type,
            region=req.region,
            market=req.market,
            launched=req.launched,
        )
    else:
        logger.warning(
            "config_attempt_missing_launch_context",
            job_id=req.job_id,
            region=req.region,
            market=req.market,
            launched=req.launched,
        )
    status = "success" if req.launched else "failed"
    logger.info(
        "config_attempt",
        gpu_type=req.gpu_type,
        market=req.market,
        region=req.region,
        launched=req.launched,
    )
    return {"status": "recorded", "job_id": req.job_id, "launched": req.launched}


class LaunchFailedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Envelope (Optional during compat window)
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None

    job_id: str
    group_id: Optional[str] = None
    decision_id: Optional[str] = None
    configs_tried: List[Dict[str, Any]] = []
    failure_reasons: List[str] = []
    total_time_seconds: float = 0.0

    # Structured reason (preferred over free-text failure_reasons going forward)
    reason_code: Optional[ReasonCode] = None
    reason_detail: Optional[str] = None


@app.post("/job/launch-failed")
async def job_launch_failed(req: LaunchFailedRequest):
    """Called by Orca when ALL alternative configs failed to launch."""

    async def _do() -> Dict[str, Any]:
        return await _job_launch_failed_impl(req)

    return await _run_with_inbox(req, "launch_failed", req.job_id, _do)


async def _job_launch_failed_impl(req: LaunchFailedRequest) -> Dict[str, Any]:
    memory: AgenticMemory = app.state.memory
    monitor: MonitoringLoop = app.state.monitor

    # Record each failed config in launch_attempts table
    tracker = monitor.tracked_jobs.get(req.job_id)
    decision_id = req.decision_id or (tracker.decision_id if tracker else None)

    for i, config in enumerate(req.configs_tried):
        reason = req.failure_reasons[i] if i < len(req.failure_reasons) else "unknown"
        gpu = config.get("gpu_type", "unknown")
        rgn = config.get("region", "unknown")
        mkt = config.get("market", "unknown")
        memory.record_launch_attempt(
            decision_id=decision_id or f"unknown-{req.job_id}",
            job_id=req.job_id,
            instance_type=config.get("instance_type", "unknown"),
            gpu_type=gpu,
            region=rgn,
            market=mkt,
            count=1,
            launched=False,
            failure_reason=reason,
            failure_category=_classify_failure(reason),
        )
        if rgn != "unknown" and mkt != "unknown":
            memory.update_availability(
                gpu_type=gpu,
                region=rgn,
                market=mkt,
                launched=False,
            )

    logger.info(
        "launch_failed",
        job_id=req.job_id,
        configs_tried=len(req.configs_tried),
        total_time_s=round(req.total_time_seconds),
    )
    emit_event(
        "launch_failed",
        job_id=req.job_id,
        configs_tried=len(req.configs_tried),
        total_time_s=round(req.total_time_seconds),
    )

    # Release pending reservation (never launched)
    if decision_id:
        app.state.ledger.release(decision_id)

    monitor.clear_pending_launch(req.job_id)
    if hasattr(monitor, "clear_pending_launches_for_group"):
        monitor.clear_pending_launches_for_group(req.job_id)

    # Unregister from monitor
    if tracker:
        monitor.unregister_job(req.job_id)

    return {
        "status": "recorded",
        "job_id": req.job_id,
        "attempts_recorded": len(req.configs_tried),
    }


@app.get("/jobs")
async def list_jobs():
    """List all tracked jobs with current status."""
    monitor: MonitoringLoop = app.state.monitor
    jobs = []
    for job_id, tracker in monitor.tracked_jobs.items():
        jobs.append(
            {
                "job_id": job_id,
                "status": tracker.status.value,
                "gpu_type": tracker.config.gpu_type,
                "tp": tracker.config.tp,
                "pp": tracker.config.pp,
                "dp": tracker.config.dp,
                "smoothed_tps": round(tracker.smoothed_tps, 1),
                "slo_headroom_pct": round(tracker.slo_headroom_pct, 1),
                "elapsed_hours": round(tracker.elapsed_hours, 2),
                "predicted_cost_per_hour": tracker.predicted_cost_per_hour,
                "projected_remaining_cost_usd": _finite_or_none(
                    tracker.projected_remaining_cost_usd, ndigits=2
                ),
                "projected_total_cost_usd": _finite_or_none(
                    tracker.projected_total_cost_usd, ndigits=2
                ),
                "cost_roofline_usd": _finite_or_none(tracker.cost_roofline_usd),
                "cost_overage_usd": _finite_or_none(
                    tracker.cost_overage_usd, ndigits=2
                ),
                "meets_cost_roofline": tracker.meets_cost_roofline,
                "tokens_completed": tracker.tokens_completed,
                "tokens_remaining": tracker.tokens_remaining,
            }
        )
    # Include pending launches (provisioned but not yet serving)
    for job_id, info in monitor._pending_launches.items():
        if job_id not in monitor.tracked_jobs:  # avoid duplicates
            last_heartbeat_at = info.get(
                "last_heartbeat_at", info.get("launched_at", time.time())
            )
            jobs.append(
                {
                    "job_id": job_id,
                    "status": "launching",
                    "launch_phase": info.get("launch_phase", "launching"),
                    "launch_message": info.get("launch_message", ""),
                    "attempt_index": info.get("attempt_index", 0),
                    "gpu_type": info.get("gpu_type", "unknown"),
                    "tp": info.get("tp", 1),
                    "pp": info.get("pp", 1),
                    "dp": 1,
                    "smoothed_tps": 0,
                    "slo_headroom_pct": 0,
                    "elapsed_hours": round(
                        (time.time() - info.get("launched_at", time.time())) / 3600, 2
                    ),
                    "last_heartbeat_seconds_ago": round(
                        time.time() - last_heartbeat_at, 1
                    ),
                    "tokens_completed": 0,
                    "tokens_remaining": 0,
                }
            )
    return {
        "tracked_jobs": len(monitor.tracked_jobs),
        "pending_launches": len(monitor._pending_launches),
        "jobs": jobs,
    }


@app.get("/resources")
async def get_live_resources():
    """Live resource map: Orca quota usage plus pending Koi reservations."""
    ledger: ResourceLedger = app.state.ledger
    orca: Optional[OrcaClient] = getattr(app.state, "orca", None)
    live_resources: Dict[str, Any] = {}
    orca_error: Optional[str] = None

    if orca is not None:
        try:
            live_resources = await asyncio.wait_for(orca.get_resources(), timeout=2.0)
        except Exception as exc:
            orca_error = str(exc)
            logger.warning("resources_fetch_failed", error=orca_error)

    return {
        "instances": live_resources.get("instances", []),
        "quotas": live_resources.get("quotas", []),
        "pending_reservations": ledger.summary(),
        "pending_gpus": ledger.get_pending_by_type(),
        "pending_count": ledger.pending_count,
        "orca_error": orca_error,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=KOI_PORT)
