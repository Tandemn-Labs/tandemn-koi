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

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from koi.agent import KoiAgent
from koi.monitor import MonitoringLoop
from koi.schemas import EngineConfig, JobRequest, MonitoringStatus, MonitoringTrigger, PlacementConfig
from koi.tools.memory import AgenticMemory
from koi.tools.orca_api import OrcaClient
from koi.tools.perfdb import PerfDB
from koi.tools.resources import parse_orca_resources

logger = logging.getLogger("koi.server")

KOI_PORT = int(os.environ.get("KOI_PORT", "8090"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    job_request: Dict[str, Any]
    resource_map: Any  # Shape A, B, or C


class JobCompleteRequest(BaseModel):
    job_id: str
    status: str
    metrics: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init components
    perfdb_path = os.environ.get("KOI_PERFDB_PATH", "./perfdb/perfdb_all.csv")
    memory_path = os.environ.get("KOI_MEMORY_PATH", "./data/koi_memory.db")
    orca_url = os.environ.get("ORCA_URL", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("KOI_LLM_MODEL", "claude-sonnet-4-6")

    # PerfDB
    try:
        app.state.perfdb = PerfDB(perfdb_path)
        logger.info(f"[Koi] PerfDB loaded: {app.state.perfdb.record_count} records, "
                     f"models={app.state.perfdb.models}, gpus={app.state.perfdb.gpu_types}")
    except Exception as e:
        logger.warning(f"[Koi] PerfDB load failed: {e}. Running without benchmark data.")
        app.state.perfdb = None

    # Memory
    app.state.memory = AgenticMemory(db_path=memory_path)
    logger.info(f"[Koi] Memory: {app.state.memory.decision_count()} decisions, "
                f"{app.state.memory.outcome_count()} outcomes")

    # Orca client
    app.state.session = aiohttp.ClientSession()
    app.state.orca = OrcaClient(orca_url, session=app.state.session) if orca_url else None
    if orca_url:
        logger.info(f"[Koi] Orca client: {orca_url}")
    else:
        logger.info("[Koi] No ORCA_URL set — running without Orca connection")

    # Agent
    app.state.agent = KoiAgent(
        perfdb=app.state.perfdb,
        memory=app.state.memory,
        orca=app.state.orca,
        api_key=api_key,
        model=model,
    )
    logger.info(f"[Koi] Agent ready (model={model})")

    # Monitor
    app.state.monitor = MonitoringLoop(
        orca=app.state.orca,
        on_trigger=app.state.agent.handle_trigger,
    )
    app.state.agent.monitor = app.state.monitor  # for anti-windup in scale_chain_tool
    if app.state.orca:
        await app.state.monitor.start()
        logger.info("[Koi] Monitor started (2 async loops)")
    else:
        logger.info("[Koi] Monitor not started — no Orca connection")

    yield

    # Cleanup
    await app.state.monitor.stop()
    if app.state.session and not app.state.session.closed:
        await app.state.session.close()
    logger.info("[Koi] Shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Koi Placement Service", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0",
        "perfdb_records": app.state.perfdb.record_count if app.state.perfdb else 0,
        "memory_decisions": app.state.memory.decision_count(),
        "memory_outcomes": app.state.memory.outcome_count(),
        "tracked_jobs": len(app.state.monitor.tracked_jobs),
        "agent_model": app.state.agent.model,
        "orca_connected": app.state.orca is not None,
    }


@app.post("/decide")
async def decide(req: DecideRequest):
    """Run the Koi agent to make a placement decision."""
    agent: KoiAgent = app.state.agent
    monitor: MonitoringLoop = app.state.monitor

    # Parse job request
    try:
        from koi.schemas import TaskType, Objective
        d = req.job_request
        job_request = JobRequest(
            model_name=str(d.get("model_name", "unknown")),
            task_type=TaskType(d.get("task_type", "batch")),
            avg_input_tokens=int(d.get("avg_input_tokens", 512)),
            avg_output_tokens=int(d.get("avg_output_tokens", 256)),
            num_requests=int(d["num_requests"]) if d.get("num_requests") else None,
            slo_deadline_hours=float(d["slo_deadline_hours"]) if d.get("slo_deadline_hours") else None,
            objective=Objective(d.get("objective", "cheapest")),
            quantization=d.get("quantization"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid job_request: {e}")

    # Parse resource map
    try:
        resource_map = parse_orca_resources(req.resource_map)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Run agent
    try:
        decision = await agent.decide(job_request, resource_map)
    except Exception as e:
        logger.error(f"[Koi] Agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # Record decision in memory
    decision_id = app.state.memory.record_decision(
        job_id=decision.job_id,
        model_name=decision.model_name,
        instance_type=decision.config.instance_type,
        gpu_type=decision.config.gpu_type,
        tp=decision.config.tp, pp=decision.config.pp, dp=decision.config.dp,
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
    )

    # NOTE: Do NOT register in monitor here. The job hasn't launched yet.
    # Orca will call POST /job/started after successful launch.

    # Include decision_id in response so Orca can pass it back
    result = decision.model_dump(mode="json")
    result["_decision_id"] = decision_id
    return result


class JobStartedRequest(BaseModel):
    job_id: str
    decision_id: Optional[str] = None
    group_id: Optional[str] = None      # parent job ID for chunked replicas
    gpu_type: str
    instance_type: str
    tp: int
    pp: int
    dp: int = 1
    slo_deadline_hours: float
    total_tokens: int
    predicted_tps: float = 0.0
    is_fallback: bool = False           # True if Orca used a fallback config (not primary)


@app.post("/job/started")
async def job_started(req: JobStartedRequest):
    """Called by Orca AFTER a job successfully launches. Registers in monitor."""
    monitor: MonitoringLoop = app.state.monitor
    memory: AgenticMemory = app.state.memory

    # Link scale-up replicas to their pending decision
    if not req.decision_id and hasattr(monitor, '_pending_scale_decision'):
        pending = getattr(monitor, '_pending_scale_decision', None)
        if pending and pending.get("group_id") == req.group_id:
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
                tp=req.tp, pp=req.pp, dp=req.dp,
                num_gpus=req.tp * req.pp * req.dp,
                predicted_tps=0,
                slo_deadline_hours=req.slo_deadline_hours,
                objective=original.get("objective", "cheapest"),
                avg_input_tokens=original.get("avg_input_tokens", 0),
                avg_output_tokens=original.get("avg_output_tokens", 0),
                num_requests=original.get("num_requests"),
                triggered_by="fallback",
                parent_decision_id=req.decision_id,
                market="on_demand",
            )
            logger.info(f"[Koi] Fallback detected: {req.decision_id} → {actual_decision_id} "
                       f"({req.gpu_type} TP={req.tp} PP={req.pp})")

    config = PlacementConfig(
        gpu_type=req.gpu_type,
        instance_type=req.instance_type,
        num_gpus=req.tp * req.pp * req.dp,
        num_instances=max(1, (req.tp * req.pp * req.dp) // 8),
        tp=req.tp, pp=req.pp, dp=req.dp,
        region="unknown",
        engine_config=EngineConfig(
            tensor_parallel_size=req.tp,
            pipeline_parallel_size=req.pp,
        ),
    )

    monitor.register_job(
        job_id=req.job_id,
        config=config,
        slo_deadline_hours=req.slo_deadline_hours,
        total_tokens=req.total_tokens,
        predicted_tps=req.predicted_tps,
        decision_id=actual_decision_id,
        group_id=req.group_id,
    )

    group_str = f" (group={req.group_id})" if req.group_id else ""
    fallback_str = " [FALLBACK]" if req.is_fallback else ""
    logger.info(f"[Koi] Job started: {req.job_id} on {req.gpu_type} TP={req.tp} PP={req.pp}{group_str}{fallback_str}")
    return {"status": "registered", "job_id": req.job_id, "group_id": req.group_id,
            "decision_id": actual_decision_id}


@app.post("/job/complete")
async def job_complete(req: JobCompleteRequest):
    """Webhook from Orca when a job completes.

    Two modes:
      1. Single-chain job: req.job_id matches a tracked chain directly
      2. Job group (chunked): req.job_id matches the group_id of multiple chains
         → aggregates metrics across all chains, records ONE outcome
    """
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
            logger.info(f"[Koi] Outcome recorded: {outcome_id} for {req.job_id} ({req.status})")

        monitor.unregister_job(req.job_id)
        return {"status": "recorded", "job_id": req.job_id}

    # Mode 2: job group completion (chunked job)
    group_chains = monitor.get_group_chains(req.job_id)
    if group_chains:
        total_tps = sum(t.smoothed_tps for t in group_chains.values())
        max_elapsed = max(t.elapsed_hours for t in group_chains.values()) if group_chains else 0

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

        logger.info(f"[Koi] Group completed: {req.job_id} — {outcomes_recorded} chain outcomes, "
                    f"aggregate TPS={total_tps:.0f}, {req.status}")

        # Unregister all chains in the group
        monitor.unregister_group(req.job_id)
        return {
            "status": "recorded",
            "job_id": req.job_id,
            "chains_closed": len(group_chains),
            "outcomes_recorded": outcomes_recorded,
            "aggregate_tps": round(total_tps, 1),
        }

    logger.warning(f"[Koi] Job complete webhook for unknown job: {req.job_id}")
    return {"status": "unknown_job", "job_id": req.job_id}


class ReplicaFailedRequest(BaseModel):
    job_id: str          # replica chain ID
    group_id: str        # parent job ID
    status: str = "failed"
    reason: str = ""


@app.post("/job/replica-failed")
async def replica_failed(req: ReplicaFailedRequest):
    """Called by Orca when a replica dies mid-job. Triggers agent for diagnosis."""
    monitor: MonitoringLoop = app.state.monitor
    tracker = monitor.tracked_jobs.get(req.job_id)
    if not tracker:
        logger.warning(f"[Koi] Replica-failed for unknown chain: {req.job_id}")
        return {"status": "unknown", "job_id": req.job_id}

    tracker.status = MonitoringStatus.FAILED
    tracker.smoothed_tps = 0
    if req.job_id not in tracker.dead_replicas:
        tracker.dead_replicas.append(req.job_id)
    # Record failure outcome in memory for learning
    memory: AgenticMemory = app.state.memory
    if tracker.decision_id:
        memory.record_outcome(
            decision_id=tracker.decision_id,
            job_id=req.group_id,
            status="replica_failed",
            failure_category="infrastructure",
            diagnosis=req.reason[:200],
        )
    # Emit FAILED trigger to agent
    trigger = MonitoringTrigger(
        trigger_type=MonitoringStatus.FAILED,
        job_id=req.job_id,
        job_tracker=tracker.model_dump(),
        diagnosis_hint=f"Replica died: {req.reason[:200]}",
    )
    await monitor._trigger_queue.put(trigger)
    logger.info(f"[Koi] Replica failed: {req.job_id} — {req.reason[:100]}")
    return {"status": "trigger_emitted", "job_id": req.job_id}


class LaunchFailedRequest(BaseModel):
    job_id: str
    configs_tried: List[Dict[str, Any]] = []
    failure_reasons: List[str] = []
    total_time_seconds: float = 0.0


@app.post("/job/launch-failed")
async def job_launch_failed(req: LaunchFailedRequest):
    """Called by Orca when ALL alternative configs failed to launch."""
    memory: AgenticMemory = app.state.memory
    monitor: MonitoringLoop = app.state.monitor

    # Record each failed config in launch_attempts table
    tracker = monitor.tracked_jobs.get(req.job_id)
    decision_id = tracker.decision_id if tracker else None

    for i, config in enumerate(req.configs_tried):
        reason = req.failure_reasons[i] if i < len(req.failure_reasons) else "unknown"
        memory.record_launch_attempt(
            decision_id=decision_id or f"unknown-{req.job_id}",
            job_id=req.job_id,
            instance_type=config.get("instance_type", "unknown"),
            gpu_type=config.get("gpu_type", "unknown"),
            region=config.get("region", "unknown"),
            market=config.get("market", "on_demand"),
            count=1,
            launched=False,
            failure_reason=reason,
        )

    logger.info(
        f"[Koi] Launch failed for {req.job_id}: "
        f"{len(req.configs_tried)} configs tried in {req.total_time_seconds:.0f}s, all failed"
    )

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
        jobs.append({
            "job_id": job_id,
            "status": tracker.status.value,
            "gpu_type": tracker.config.gpu_type,
            "tp": tracker.config.tp,
            "pp": tracker.config.pp,
            "dp": tracker.config.dp,
            "smoothed_tps": round(tracker.smoothed_tps, 1),
            "slo_headroom_pct": round(tracker.slo_headroom_pct, 1),
            "elapsed_hours": round(tracker.elapsed_hours, 2),
            "tokens_completed": tracker.tokens_completed,
            "tokens_remaining": tracker.tokens_remaining,
        })
    return {"tracked_jobs": len(jobs), "jobs": jobs}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Configure logging so [Koi] messages show up
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=KOI_PORT)
