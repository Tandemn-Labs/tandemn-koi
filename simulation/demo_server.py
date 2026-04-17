"""Demo backend for the browser-based Koi + Orca simulator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from koi.event_tap import read_recent_events
from simulation.demo_runtime import DemoSessionManager
from simulation.orca_webhooks import (
    build_complete_payload,
    build_config_attempt_payload,
    build_launch_failed_payload,
    build_launch_heartbeat_payload,
    build_launching_payload,
    build_replica_failed_payload,
    build_started_payload,
    heartbeat_message,
)
from simulation.demo_scenarios import (
    default_quota_overrides,
    get_quota_preset,
    get_scenario,
    list_quota_presets,
    quota_preset_editable_rows,
    quota_preset_to_resource_map,
    serialize_catalog,
)
from simulation.model_registry import resolve_model_spec
from simulation.perf_model import DemoPerfModel

try:
    from simulation.demo_preview import get_preview_snapshot, list_preview_scenes
except Exception:  # pragma: no cover - local-only design tooling
    get_preview_snapshot = None
    list_preview_scenes = None


app = FastAPI(title="Koi Demo Server", version="0.1")
logger = logging.getLogger(__name__)
PERF_MODEL = DemoPerfModel()
SESSION_MANAGER = DemoSessionManager()
REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static" / "demo"
app.mount("/demo/static", StaticFiles(directory=str(STATIC_DIR)), name="demo-static")
DEMO_KOI_URL = os.environ.get("KOI_DEMO_URL", "http://localhost:8090")
DEMO_KOI_EVENT_LOG = os.environ.get(
    "KOI_EVENT_TAP_PATH", str(REPO_ROOT / "data" / "koi_events.jsonl")
)
HEARTBEAT_INTERVAL_S = 3.0
LAUNCH_TASKS: set[asyncio.Task] = set()
QUOTA_OVERRIDES_PATH = Path(
    os.environ.get(
        "KOI_DEMO_QUOTA_OVERRIDES_PATH",
        str(REPO_ROOT / "data" / "quota_overrides.json"),
    )
)


class QuotaOverrideStore:
    """Persistent quota overrides per preset.

    Each preset is mapped to ``{quota_row_key -> baseline_vcpus}``. The store is
    loaded from disk on init and rewritten atomically on every ``set``.
    """

    def __init__(self, path: Path):
        self._path = path
        self._state: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            return
        except OSError as exc:  # pragma: no cover - unusual fs errors
            logger.warning(
                "quota_overrides_load_failed", path=str(self._path), error=str(exc)
            )
            return
        try:
            parsed = json.loads(raw) or {}
        except json.JSONDecodeError as exc:
            logger.warning(
                "quota_overrides_corrupt",
                path=str(self._path),
                error=str(exc),
            )
            return
        if not isinstance(parsed, dict):
            return
        clean: dict[str, dict[str, int]] = {}
        for slug, rows in parsed.items():
            if not isinstance(rows, dict):
                continue
            clean[str(slug)] = {
                str(k): int(v) for k, v in rows.items() if isinstance(v, (int, float))
            }
        self._state = clean

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, sort_keys=True, indent=2))
        tmp.replace(self._path)

    def get(self, slug: str) -> dict[str, int]:
        return dict(self._state.get(slug, {}))

    def all(self) -> dict[str, dict[str, int]]:
        return {slug: dict(rows) for slug, rows in self._state.items()}

    def set(self, slug: str, overrides: dict[str, int]) -> None:
        clean = {str(k): max(0, int(v)) for k, v in overrides.items()}
        if clean:
            self._state[slug] = clean
        else:
            self._state.pop(slug, None)
        self._save()

    def reset(self, slug: str) -> None:
        if slug in self._state:
            self._state.pop(slug)
            self._save()


QUOTA_OVERRIDES = QuotaOverrideStore(QUOTA_OVERRIDES_PATH)


def _active_session_count() -> int:
    """Return the number of sessions that are not already completed/failed."""
    count = 0
    for session in SESSION_MANAGER.sessions.values():
        runtime = session.get("runtime") or {}
        status = runtime.get("status")
        if status in {"completed", "launch_failed"}:
            continue
        count += 1
    return count


def _quota_overrides_locked() -> bool:
    return _active_session_count() > 0


class DemoLaunchRequest(BaseModel):
    model_name: str
    avg_input_tokens: int = Field(ge=1)
    avg_output_tokens: int = Field(ge=1)
    total_chunks: int = Field(default=500, ge=1)
    slo_deadline_hours: float = Field(default=8.0, gt=0)
    quota_preset: str
    scenario: str
    cost_cap_usd: Optional[float] = Field(default=None, gt=0)
    dtype: str = "fp16"
    model_overrides: Optional[dict] = None


class ScaleRequest(BaseModel):
    count: int = Field(default=1, ge=1)
    gpu_type: str = "L40S"
    tp_size: int = 4
    pp_size: int = 1
    on_demand: bool = True


class KillRequest(BaseModel):
    replica_ids: list[str]


class ReplicaTpsRequest(BaseModel):
    target_tps: float = Field(gt=0)


class QuotaOverrideRequest(BaseModel):
    preset_slug: str
    overrides: dict[str, int]


async def _post_koi(path: str, payload: dict, timeout: float = 20.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{DEMO_KOI_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


async def _get_koi_json(path: str) -> dict:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{DEMO_KOI_URL}{path}")
        response.raise_for_status()
        return response.json()


async def _request_koi_decision(
    session_id: str,
    req: DemoLaunchRequest,
    resource_map: dict,
) -> Optional[dict]:
    payload = {
        "job_request": {
            "job_id": session_id,
            "model_name": req.model_name,
            "task_type": "batch",
            "avg_input_tokens": req.avg_input_tokens,
            "avg_output_tokens": req.avg_output_tokens,
            "num_requests": req.total_chunks,
            "slo_deadline_hours": req.slo_deadline_hours,
            "objective": "cheapest",
        },
        "resource_map": resource_map,
    }
    return await _post_koi("/decide", payload, timeout=180.0)


def _track_launch_task(task: asyncio.Task) -> None:
    LAUNCH_TASKS.add(task)

    def _done(completed: asyncio.Task) -> None:
        LAUNCH_TASKS.discard(completed)
        try:
            completed.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("demo_launch_task_failed")

    task.add_done_callback(_done)


def _pick_launch_config(
    *,
    session_id: str,
    quota,
    preferred_gpu: str,
    tp: int,
    pp: int,
    total_tokens: int,
    baseline_tps: float,
    koi_decision: Optional[dict],
) -> dict:
    decision_config = (koi_decision or {}).get("config", {}) or {}
    chosen_instance = next(
        (
            instance
            for instance in quota.instances
            if instance.gpu_type == preferred_gpu
        ),
        quota.instances[0],
    )
    chosen_quota = next(
        (
            item
            for item in quota.quotas
            if item["family"] == chosen_instance.quota_family
        ),
        quota.quotas[0],
    )
    return {
        "job_id": f"{session_id}-r0",
        "group_id": session_id,
        "decision_id": (koi_decision or {}).get("_decision_id"),
        "gpu_type": decision_config.get("gpu_type", preferred_gpu),
        "instance_type": decision_config.get(
            "instance_type", chosen_instance.instance_type
        ),
        "tp": int(decision_config.get("tp", tp)),
        "pp": int(decision_config.get("pp", pp)),
        "dp": int(decision_config.get("dp", 1) or 1),
        "region": decision_config.get("region", chosen_quota["region"]),
        "market": decision_config.get("market", chosen_quota["market"]),
        "total_tokens": int(total_tokens),
        "predicted_tps": float(
            (koi_decision or {}).get("predicted_tps") or baseline_tps
        ),
    }


def _filter_session_koi_jobs(session_id: str, jobs_payload: dict) -> dict:
    jobs = [
        job
        for job in jobs_payload.get("jobs", [])
        if job.get("job_id", "").startswith(f"{session_id}-")
        or job.get("job_id") == session_id
    ]
    return {
        "tracked_jobs": len([job for job in jobs if job.get("status") != "launching"]),
        "pending_launches": len(
            [job for job in jobs if job.get("status") == "launching"]
        ),
        "jobs": jobs,
    }


async def _fetch_koi_live_state(session_id: str) -> dict:
    jobs_payload, resources_payload = await asyncio.gather(
        _get_koi_json("/jobs"),
        _get_koi_json("/resources"),
    )
    return {
        "jobs": _filter_session_koi_jobs(session_id, jobs_payload),
        "resources": resources_payload,
    }


def _read_session_koi_events(session_id: str, limit: int = 120) -> list[dict[str, Any]]:
    if not DEMO_KOI_EVENT_LOG:
        return []
    events = read_recent_events(DEMO_KOI_EVENT_LOG, limit=200)
    filtered = []
    for event in events:
        job_id = str(event.get("job_id", ""))
        group_id = str(event.get("group_id", ""))
        if (
            job_id == session_id
            or job_id.startswith(f"{session_id}-")
            or group_id == session_id
        ):
            filtered.append(event)
    return filtered[-limit:]


def _build_replica_payloads(
    session_id: str, snapshot: dict, replica: dict
) -> tuple[dict, dict, dict, dict]:
    session = SESSION_MANAGER.sessions[session_id]
    launch_config = session["launch_config"]
    decision_id = (
        replica["decision_id"]
        if "decision_id" in replica
        else launch_config.get("decision_id")
    )
    attempt_index = int(
        replica.get("config_index", launch_config.get("candidate_index", 0)) or 0
    )
    time_to_launch = max(
        0.0,
        time.time() - float(session.get("launch_started_at") or session["created_at"]),
    )
    config_attempt_payload = build_config_attempt_payload(
        job_id=session_id,
        decision_id=decision_id,
        instance_type=replica["instance_type"],
        gpu_type=replica["gpu_type"],
        region=replica["region"],
        market=replica["market"],
        launched=True,
        attempt_index=attempt_index,
        time_to_launch=time_to_launch,
    )
    launching_payload = build_launching_payload(
        job_id=replica["replica_id"],
        decision_id=decision_id,
        group_id=session_id,
        gpu_type=replica["gpu_type"],
        instance_type=replica["instance_type"],
        tp=replica["tp"],
        pp=replica["pp"],
        region=replica["region"],
        market=replica["market"],
        attempt_index=attempt_index,
    )
    heartbeat_payload = build_launch_heartbeat_payload(
        job_id=replica["replica_id"],
        decision_id=decision_id,
        group_id=session_id,
        gpu_type=replica["gpu_type"],
        instance_type=replica["instance_type"],
        tp=replica["tp"],
        pp=replica["pp"],
        region=replica["region"],
        market=replica["market"],
        attempt_index=attempt_index,
        phase=replica["launch_phase"],
        message=heartbeat_message(replica["launch_phase"]),
        timestamp=time.time(),
    )
    started_payload = build_started_payload(
        job_id=replica["replica_id"],
        decision_id=decision_id,
        group_id=session_id,
        gpu_type=replica["gpu_type"],
        instance_type=replica["instance_type"],
        tp=replica["tp"],
        pp=replica["pp"],
        dp=1,
        region=replica["region"],
        market=replica["market"],
        slo_deadline_hours=snapshot["request"]["slo_deadline_hours"],
        total_tokens=launch_config["total_tokens"],
        predicted_tps=float(replica.get("base_tps") or launch_config["predicted_tps"]),
        is_fallback=attempt_index > 0,
    )
    return config_attempt_payload, heartbeat_payload, launching_payload, started_payload


async def _sync_session_with_koi(session_id: str, snapshot: dict) -> dict:
    session = SESSION_MANAGER.sessions[session_id]
    if session.get("koi", {}).get("decision_status") == "pending":
        return {"status": "decision_pending"}
    if not session.get("koi", {}).get("decision"):
        return {"status": "no_decision"}

    bridge = session["_bridge"]
    now = time.time()
    attempts_sent = 0
    launches_sent = 0
    starts_sent = 0
    heartbeats_sent = 0
    replica_failures_sent = 0
    completes_sent = 0
    launch_failed_sent = 0
    try:
        while bridge["pending_config_attempts"]:
            item = bridge["pending_config_attempts"].pop(0)
            payload = (
                item["payload"]
                if isinstance(item, dict) and "payload" in item
                else item
            )
            await _post_koi("/job/config-attempted", payload)
            attempts_sent += 1

        if (
            session.get("launch_failure")
            and bridge.get("launch_failed_payload")
            and not bridge.get("launch_failed_sent")
        ):
            await _post_koi(
                "/job/launch-failed",
                build_launch_failed_payload(**bridge["launch_failed_payload"]),
            )
            bridge["launch_failed_sent"] = True
            launch_failed_sent += 1

        for replica in snapshot["runtime"]["replicas"]:
            replica_id = replica["replica_id"]
            (
                config_attempt_payload,
                heartbeat_payload,
                launching_payload,
                started_payload,
            ) = _build_replica_payloads(session_id, snapshot, replica)
            if replica["phase"] in {"launching", "provisioned"}:
                if (
                    replica["phase"] == "provisioned"
                    and replica_id not in bridge["config_attempted_sent"]
                ):
                    await _post_koi("/job/config-attempted", config_attempt_payload)
                    bridge["config_attempted_sent"].add(replica_id)
                    attempts_sent += 1
                if (
                    replica["phase"] == "provisioned"
                    and replica_id not in bridge["launching_sent"]
                ):
                    await _post_koi("/job/launching", launching_payload)
                    bridge["launching_sent"].add(replica_id)
                    launches_sent += 1

                last_phase = bridge["last_heartbeat_phase"].get(replica_id)
                last_at = bridge["last_heartbeat_at"].get(replica_id)
                should_send_heartbeat = (
                    last_phase != replica["launch_phase"]
                    or last_at is None
                    or (now - float(last_at)) >= HEARTBEAT_INTERVAL_S
                )
                if should_send_heartbeat:
                    await _post_koi("/job/launch-heartbeat", heartbeat_payload)
                    bridge["last_heartbeat_phase"][replica_id] = replica["launch_phase"]
                    bridge["last_heartbeat_at"][replica_id] = now
                    heartbeats_sent += 1
            elif replica["phase"] == "running":
                if replica_id not in bridge["config_attempted_sent"]:
                    await _post_koi("/job/config-attempted", config_attempt_payload)
                    bridge["config_attempted_sent"].add(replica_id)
                    attempts_sent += 1
                if replica_id not in bridge["launching_sent"]:
                    await _post_koi("/job/launching", launching_payload)
                    bridge["launching_sent"].add(replica_id)
                    launches_sent += 1
                if replica_id not in bridge["started_sent"]:
                    await _post_koi("/job/started", started_payload)
                    bridge["started_sent"].add(replica_id)
                    starts_sent += 1
            elif replica["phase"] in {"dead", "killed", "failed"}:
                if replica_id not in bridge["replica_failed_sent"]:
                    reason = bridge["replica_failure_reasons"].get(
                        replica_id, f"Replica entered {replica['phase']}"
                    )
                    await _post_koi(
                        "/job/replica-failed",
                        build_replica_failed_payload(
                            job_id=replica_id,
                            group_id=session_id,
                            reason=reason,
                        ),
                    )
                    bridge["replica_failed_sent"].add(replica_id)
                    replica_failures_sent += 1

        if snapshot["runtime"]["status"] == "completed" and not bridge["complete_sent"]:
            throughput_tps = max(
                float(session["_orca"].get("last_nonzero_aggregate_tps", 0.0)),
                float(snapshot["runtime"].get("aggregate_tps", 0.0)),
            )
            await _post_koi(
                "/job/complete",
                build_complete_payload(
                    job_id=session_id, throughput_tps=throughput_tps
                ),
            )
            bridge["complete_sent"] = True
            completes_sent += 1

        session["koi"]["sync_error"] = None
        if starts_sent:
            status = "started_sent"
        elif launches_sent:
            status = "launching_sent"
        elif heartbeats_sent:
            status = "heartbeat_sent"
        elif attempts_sent:
            status = "config_attempted_sent"
        elif replica_failures_sent:
            status = "replica_failed_sent"
        elif completes_sent:
            status = "complete_sent"
        elif launch_failed_sent:
            status = "launch_failed_sent"
        elif session.get("launch_failure"):
            status = "launch_rejected"
        else:
            status = "noop"
        return {
            "status": status,
            "attempts_sent": attempts_sent,
            "launches_sent": launches_sent,
            "starts_sent": starts_sent,
            "heartbeats_sent": heartbeats_sent,
            "replica_failures_sent": replica_failures_sent,
            "completes_sent": completes_sent,
            "launch_failed_sent": launch_failed_sent,
        }
    except Exception as exc:
        session["koi"]["sync_error"] = str(exc)
        return {"status": "error", "error": str(exc)}


async def _attach_live_koi(session_id: str, snapshot: dict) -> dict:
    session = SESSION_MANAGER.sessions[session_id]
    snapshot["koi"]["sync"] = await _sync_session_with_koi(session_id, snapshot)
    try:
        snapshot["koi"]["live"] = await _fetch_koi_live_state(session_id)
        session["koi"]["live_error"] = None
    except Exception as exc:
        error = str(exc)
        snapshot["koi"]["live"] = None
        snapshot["koi"]["live_error"] = error
        session["koi"]["live_error"] = error
    snapshot["koi"]["events"] = _read_session_koi_events(session_id)
    snapshot["koi"]["sync_error"] = session["koi"].get("sync_error")
    return snapshot


def _require_session(session_id: str) -> dict:
    session = SESSION_MANAGER.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="unknown demo session")
    return session


def _resolve_session_id_from_job(job_id: str) -> str:
    if job_id in SESSION_MANAGER.sessions:
        return job_id
    # Accept replica-suffixed job ids (e.g. "<session>-r0") by mapping via replicas.
    for session_id, session in SESSION_MANAGER.sessions.items():
        job = session.get("_orca", {}).get("job")
        if job is not None and job_id in getattr(job, "replicas", {}):
            return session_id
    # Fallback: strip a trailing "-rN" segment if present.
    if "-r" in job_id:
        candidate = job_id.rsplit("-r", 1)[0]
        if candidate in SESSION_MANAGER.sessions:
            return candidate
    raise HTTPException(status_code=404, detail=f"unknown demo job: {job_id}")


def _resolve_session_id_from_replica(replica_id: str) -> str:
    for session_id, session in SESSION_MANAGER.sessions.items():
        job = session["_orca"]["job"]
        if replica_id in job.replicas:
            return session_id
    raise HTTPException(status_code=404, detail=f"unknown replica: {replica_id}")


def _scale_launch_timing_for_session(session: dict, gpu_type: str) -> dict[str, float]:
    capacity_pressure = 0.8 if session["scenario"]["slug"] == "slow_launch" else 0.2
    launch_timing = PERF_MODEL.estimate_launch_timing(
        gpu_type=gpu_type, capacity_pressure=capacity_pressure
    )
    multiplier = session["scenario"]["launch_timing_multiplier"] * 0.75
    return {
        "searching_capacity": round(launch_timing.searching_capacity_s * multiplier, 1),
        "provisioning": round(launch_timing.provisioning_s * multiplier, 1),
        "bootstrapping": round(launch_timing.bootstrapping_s * multiplier, 1),
        "waiting_model_ready": round(
            launch_timing.waiting_model_ready_s * multiplier, 1
        ),
        "total": round(launch_timing.total_seconds * multiplier, 1),
    }


def _assess_replica_config_for_session(session: dict, gpu_type: str, tp: int, pp: int):
    request = session["request"]
    return PERF_MODEL.assess_replica_config(
        model_name=session["model"]["model_name"],
        gpu_type=gpu_type,
        tp=tp,
        pp=pp,
        dtype=request.get("dtype", "fp16"),
        overrides=request.get("model_overrides"),
    )


def _estimate_replica_tps_for_session(
    session: dict, gpu_type: str, tp: int, pp: int
) -> float:
    assessment = _assess_replica_config_for_session(session, gpu_type, tp, pp)
    if not assessment.feasible:
        raise ValueError(assessment.reason or "invalid placement")
    request = session["request"]
    return PERF_MODEL.estimate_replica_tps(
        model_name=session["model"]["model_name"],
        gpu_type=gpu_type,
        tp=tp,
        pp=pp,
        input_tokens=request["avg_input_tokens"],
        output_tokens=request["avg_output_tokens"],
        dtype=request.get("dtype", "fp16"),
        overrides=request.get("model_overrides"),
    )


def _normalize_candidate_pref(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return None
    return text


def _launch_timing_for_candidate(
    req: DemoLaunchRequest, scenario, gpu_type: str
) -> dict[str, float]:
    launch_timing = PERF_MODEL.estimate_launch_timing(
        gpu_type=gpu_type,
        capacity_pressure=0.8 if req.scenario == "slow_launch" else 0.2,
    )
    return {
        "searching_capacity": round(
            launch_timing.searching_capacity_s * scenario.launch_timing_multiplier, 1
        ),
        "provisioning": round(
            launch_timing.provisioning_s * scenario.launch_timing_multiplier, 1
        ),
        "bootstrapping": round(
            launch_timing.bootstrapping_s * scenario.launch_timing_multiplier, 1
        ),
        "waiting_model_ready": round(
            launch_timing.waiting_model_ready_s * scenario.launch_timing_multiplier, 1
        ),
        "total": round(
            launch_timing.total_seconds * scenario.launch_timing_multiplier, 1
        ),
    }


def _build_launch_candidates(
    *,
    req: DemoLaunchRequest,
    quota,
    scenario,
    koi_decision: Optional[dict],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def _append_candidate(
        raw: dict[str, Any],
        *,
        predicted_tps: Optional[float],
        is_fallback: bool,
        source: str,
        candidate_index: int,
        decision_id: Optional[str],
    ) -> None:
        gpu_type = raw.get("gpu_type")
        if not gpu_type:
            return
        tp = int(raw.get("tp", 4) or 4)
        pp = int(raw.get("pp", 1) or 1)
        assessment = PERF_MODEL.assess_replica_config(
            model_name=req.model_name,
            gpu_type=gpu_type,
            tp=tp,
            pp=pp,
            dtype=req.dtype,
            overrides=req.model_overrides,
        )
        if not assessment.feasible:
            return
        resolved_tps = float(
            predicted_tps
            or raw.get("predicted_tps")
            or PERF_MODEL.estimate_replica_tps(
                model_name=req.model_name,
                gpu_type=gpu_type,
                tp=tp,
                pp=pp,
                input_tokens=req.avg_input_tokens,
                output_tokens=req.avg_output_tokens,
                dtype=req.dtype,
                overrides=req.model_overrides,
            )
        )
        candidates.append(
            {
                "gpu_type": gpu_type,
                "instance_type": raw.get("instance_type"),
                "tp": tp,
                "pp": pp,
                "dp": int(raw.get("dp", 1) or 1),
                "region": _normalize_candidate_pref(raw.get("region")),
                "market": _normalize_candidate_pref(raw.get("market")),
                "predicted_tps": resolved_tps,
                "launch_timing_s": _launch_timing_for_candidate(
                    req, scenario, gpu_type
                ),
                "is_fallback": is_fallback,
                "source": source,
                "candidate_index": candidate_index,
                "decision_id": decision_id,
            }
        )

    if koi_decision:
        primary = (koi_decision.get("config") or {}) or {}
        _append_candidate(
            primary,
            predicted_tps=koi_decision.get("predicted_tps"),
            is_fallback=False,
            source="koi_primary",
            candidate_index=0,
            decision_id=koi_decision.get("_decision_id"),
        )
        for idx, alt in enumerate(koi_decision.get("alternatives") or [], start=1):
            _append_candidate(
                alt,
                predicted_tps=alt.get("predicted_tps")
                if isinstance(alt, dict)
                else None,
                is_fallback=True,
                source="koi_alternative",
                candidate_index=idx,
                decision_id=koi_decision.get("_decision_id"),
            )
        return candidates

    for idx, instance in enumerate(quota.instances):
        quota_row = next(
            (item for item in quota.quotas if item["family"] == instance.quota_family),
            None,
        )
        _append_candidate(
            {
                "gpu_type": instance.gpu_type,
                "instance_type": instance.instance_type,
                "tp": min(4, int(instance.gpus_per_instance)),
                "pp": 1,
                "region": quota_row["region"] if quota_row else None,
                "market": quota_row["market"] if quota_row else None,
            },
            predicted_tps=None,
            is_fallback=False,
            source="demo_fallback",
            candidate_index=idx,
            decision_id=None,
        )
    return candidates


def _compute_launch_artifacts(
    *,
    session_id: str,
    req: DemoLaunchRequest,
    quota,
    scenario,
    koi_decision: Optional[dict],
) -> tuple[dict[str, Any], dict[str, Any]]:
    preferred_gpu = (koi_decision or {}).get("config", {}).get(
        "gpu_type"
    ) or quota.instances[0].gpu_type
    launch_timing = PERF_MODEL.estimate_launch_timing(
        gpu_type=preferred_gpu,
        capacity_pressure=0.8 if req.scenario == "slow_launch" else 0.2,
    )
    baseline_tps = (koi_decision or {}).get(
        "predicted_tps"
    ) or PERF_MODEL.estimate_replica_tps(
        model_name=req.model_name,
        gpu_type=preferred_gpu,
        tp=((koi_decision or {}).get("config", {}) or {}).get("tp", 4),
        pp=((koi_decision or {}).get("config", {}) or {}).get("pp", 1),
        input_tokens=req.avg_input_tokens,
        output_tokens=req.avg_output_tokens,
        dtype=req.dtype,
        overrides=req.model_overrides,
    )

    tp = ((koi_decision or {}).get("config", {}) or {}).get("tp", 4)
    pp = ((koi_decision or {}).get("config", {}) or {}).get("pp", 1)
    launch_config = _pick_launch_config(
        session_id=session_id,
        quota=quota,
        preferred_gpu=preferred_gpu,
        tp=tp,
        pp=pp,
        total_tokens=req.total_chunks * (req.avg_input_tokens + req.avg_output_tokens),
        baseline_tps=baseline_tps,
        koi_decision=koi_decision,
    )
    launch_preview = {
        "baseline_replica_tps": round(baseline_tps, 1),
        "launch_timing_s": {
            "searching_capacity": round(
                launch_timing.searching_capacity_s * scenario.launch_timing_multiplier,
                1,
            ),
            "provisioning": round(
                launch_timing.provisioning_s * scenario.launch_timing_multiplier, 1
            ),
            "bootstrapping": round(
                launch_timing.bootstrapping_s * scenario.launch_timing_multiplier, 1
            ),
            "waiting_model_ready": round(
                launch_timing.waiting_model_ready_s * scenario.launch_timing_multiplier,
                1,
            ),
            "total": round(
                launch_timing.total_seconds * scenario.launch_timing_multiplier, 1
            ),
        },
        "preferred_gpu": preferred_gpu,
        "instance_type": launch_config["instance_type"],
        "region": launch_config["region"],
        "market": launch_config["market"],
        "tp": tp,
        "pp": pp,
    }
    return launch_preview, launch_config


async def _resolve_session_launch(
    *,
    session_id: str,
    req: DemoLaunchRequest,
    quota,
    scenario,
    resource_map: dict[str, Any],
) -> None:
    koi_decision = None
    koi_error = None
    try:
        live_resource_map = SESSION_MANAGER.aggregate_resources()
        koi_decision = await _request_koi_decision(
            session_id,
            req,
            live_resource_map if live_resource_map.get("instances") else resource_map,
        )
    except Exception as exc:
        koi_error = str(exc)

    launch_preview, launch_config = _compute_launch_artifacts(
        session_id=session_id,
        req=req,
        quota=quota,
        scenario=scenario,
        koi_decision=koi_decision,
    )

    session = _require_session(session_id)
    session["koi"]["decision"] = koi_decision
    session["koi"]["error"] = koi_error
    session["koi"]["decision_status"] = "ready" if koi_decision else "fallback"
    session["launch_preview"] = launch_preview
    session["launch_config"] = launch_config
    SESSION_MANAGER.activate_session_with_candidates(
        session_id,
        now=time.time(),
        candidates=_build_launch_candidates(
            req=req,
            quota=quota,
            scenario=scenario,
            koi_decision=koi_decision,
        ),
    )


@app.get("/demo/health")
async def demo_health():
    return {"status": "ok", "sessions": len(SESSION_MANAGER.sessions)}


@app.get("/demo")
async def demo_index():
    SESSION_MANAGER.fail_active_sessions()
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/demo/catalog")
async def demo_catalog():
    payload = serialize_catalog()
    # Augment each quota preset with slider metadata and currently persisted overrides
    # so the "Edit Quota" UI has a single place to read from.
    overrides_map = QUOTA_OVERRIDES.all()
    for preset in payload.get("quota_presets", []):
        slug = preset.get("slug")
        if not slug:
            continue
        overrides = overrides_map.get(slug, {})
        preset["editable_rows"] = quota_preset_editable_rows(slug)
        preset["overrides"] = overrides
        preset["defaults"] = default_quota_overrides(slug)
        # Reflect overrides directly in the quotas list so UI and agent tooling
        # see the effective values in a single place.
        if overrides:
            patched_quotas = []
            for row in preset.get("quotas", []):
                key = (
                    f"{str(row.get('family', '') or '').upper()}|"
                    f"{str(row.get('region', '') or '')}|"
                    f"{str(row.get('market', '') or '')}"
                )
                if key in overrides:
                    patched = dict(row)
                    patched["baseline_vcpus"] = int(overrides[key])
                    patched_quotas.append(patched)
                else:
                    patched_quotas.append(row)
            preset["quotas"] = patched_quotas
    payload["quota_locked"] = _quota_overrides_locked()
    payload["active_sessions"] = _active_session_count()
    return payload


@app.get("/demo/quota/overrides")
async def demo_quota_overrides():
    presets_meta = [
        {
            "slug": preset.slug,
            "title": preset.title,
            "defaults": default_quota_overrides(preset.slug),
            "editable_rows": quota_preset_editable_rows(preset.slug),
        }
        for preset in list_quota_presets()
    ]
    return {
        "locked": _quota_overrides_locked(),
        "active_sessions": _active_session_count(),
        "overrides": QUOTA_OVERRIDES.all(),
        "presets": presets_meta,
    }


@app.post("/demo/quota/overrides")
async def demo_set_quota_overrides(req: QuotaOverrideRequest):
    if _quota_overrides_locked():
        raise HTTPException(
            status_code=409,
            detail=(
                "quota_locked: a demo session is active; wait for it to finish "
                "before saving quota changes."
            ),
        )
    try:
        defaults = default_quota_overrides(req.preset_slug)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown quota preset: {req.preset_slug}"
        ) from exc

    valid_keys = set(defaults.keys())
    cleaned: dict[str, int] = {}
    for key, value in (req.overrides or {}).items():
        if key not in valid_keys:
            continue
        try:
            cleaned[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue

    # Drop entries that match the default so the store stays lean.
    diff = {k: v for k, v in cleaned.items() if v != defaults.get(k, -1)}
    QUOTA_OVERRIDES.set(req.preset_slug, diff)
    return {
        "status": "ok",
        "preset_slug": req.preset_slug,
        "overrides": QUOTA_OVERRIDES.get(req.preset_slug),
        "locked": _quota_overrides_locked(),
        "active_sessions": _active_session_count(),
    }


@app.post("/demo/quota/overrides/{preset_slug}/reset")
async def demo_reset_quota_overrides(preset_slug: str):
    if _quota_overrides_locked():
        raise HTTPException(
            status_code=409,
            detail=(
                "quota_locked: a demo session is active; wait for it to finish "
                "before resetting quota changes."
            ),
        )
    try:
        default_quota_overrides(preset_slug)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown quota preset: {preset_slug}"
        ) from exc
    QUOTA_OVERRIDES.reset(preset_slug)
    return {
        "status": "ok",
        "preset_slug": preset_slug,
        "overrides": {},
        "locked": _quota_overrides_locked(),
        "active_sessions": _active_session_count(),
    }


@app.get("/demo/preview/catalog")
async def demo_preview_catalog():
    payload = serialize_catalog()
    payload["preview_scenes"] = list_preview_scenes() if list_preview_scenes else []
    return payload


@app.get("/demo/preview/scene/{scene_slug}")
async def demo_preview_scene(scene_slug: str):
    if get_preview_snapshot is None:
        raise HTTPException(status_code=404, detail="preview generator unavailable")
    try:
        return get_preview_snapshot(scene_slug)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown preview scene: {scene_slug}"
        ) from exc


@app.post("/demo/launch")
async def demo_launch(req: DemoLaunchRequest):
    preset_overrides = QUOTA_OVERRIDES.get(req.quota_preset)
    try:
        quota = get_quota_preset(req.quota_preset, overrides=preset_overrides)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown quota preset: {req.quota_preset}"
        ) from exc

    try:
        scenario = get_scenario(req.scenario)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown scenario: {req.scenario}"
        ) from exc

    model_spec = resolve_model_spec(
        req.model_name,
        dtype=req.dtype,
        overrides=req.model_overrides,
    )

    session_id = f"demo-{uuid.uuid4().hex[:10]}"
    resource_map = quota_preset_to_resource_map(
        req.quota_preset, overrides=preset_overrides
    )
    launch_preview, launch_config = _compute_launch_artifacts(
        session_id=session_id,
        req=req,
        quota=quota,
        scenario=scenario,
        koi_decision=None,
    )
    payload = {
        "session_id": session_id,
        "status": "created",
        "created_at": time.time(),
        "launch_started_at": None,
        "request": req.model_dump(mode="json"),
        "model": model_spec.__dict__,
        "scenario": {
            "slug": scenario.slug,
            "title": scenario.title,
            "description": scenario.description,
            "initial_replicas": scenario.initial_replicas,
            "launch_timing_multiplier": scenario.launch_timing_multiplier,
        },
        "quota": {
            "slug": quota.slug,
            "title": quota.title,
            "cloud": quota.cloud,
            "notes": quota.notes,
        },
        "resource_map": resource_map,
        "koi": {
            "configured_url": DEMO_KOI_URL,
            "decision": None,
            "decision_status": "pending",
            "error": None,
            "sync_error": None,
            "live": None,
        },
        "launch_preview": launch_preview,
        "launch_config": launch_config,
    }
    created = SESSION_MANAGER.create_session(payload)
    _track_launch_task(
        asyncio.create_task(
            _resolve_session_launch(
                session_id=session_id,
                req=req,
                quota=quota,
                scenario=scenario,
                resource_map=resource_map,
            )
        )
    )
    return await _attach_live_koi(session_id, created)


@app.get("/demo/session/{session_id}")
async def demo_session(session_id: str, now: Optional[float] = None):
    if session_id not in SESSION_MANAGER.sessions:
        raise HTTPException(status_code=404, detail="unknown demo session")
    snapshot = SESSION_MANAGER.snapshot(session_id, now=now)
    return await _attach_live_koi(session_id, snapshot)


@app.get("/demo/stream/{session_id}")
async def demo_stream(session_id: str):
    if session_id not in SESSION_MANAGER.sessions:
        raise HTTPException(status_code=404, detail="unknown demo session")

    async def _events():
        yield ": connected\nretry: 1000\n\n"
        while True:
            snapshot = SESSION_MANAGER.snapshot(session_id)
            snapshot = await _attach_live_koi(session_id, snapshot)
            yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["runtime"]["status"] == "completed":
                break
            await asyncio.sleep(1)

    return StreamingResponse(_events(), media_type="text/event-stream")


@app.get("/demo/orca/resources")
async def demo_orca_resources(now: Optional[float] = None):
    return SESSION_MANAGER.aggregate_resources(now=now)


@app.get("/demo/orca/job/{job_id}")
async def demo_orca_job_status(job_id: str, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    return SESSION_MANAGER.get_orca_job_status(session_id, now=now)


@app.get("/demo/orca/job/{job_id}/metrics")
async def demo_orca_job_metrics(job_id: str, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    return SESSION_MANAGER.get_orca_job_metrics(session_id, now=now)


@app.get("/demo/orca/job/{job_id}/chunks/progress")
async def demo_orca_chunk_progress(job_id: str, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    return SESSION_MANAGER.get_chunk_progress(session_id, now=now)


@app.get("/demo/orca/job/{job_id}/replicas")
async def demo_orca_replicas(job_id: str, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    return {"replicas": SESSION_MANAGER.get_replicas(session_id, now=now)}


@app.get("/demo/orca/job/{job_id}/replicas/{replica_id}/metrics")
async def demo_orca_replica_metrics(
    job_id: str, replica_id: str, now: Optional[float] = None
):
    session_id = _resolve_session_id_from_job(job_id)
    try:
        return SESSION_MANAGER.get_orca_replica_metrics(session_id, replica_id, now=now)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown replica: {replica_id}"
        ) from exc


@app.post("/demo/orca/job/{job_id}/scale")
async def demo_orca_scale(job_id: str, req: ScaleRequest, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    session = _require_session(session_id)
    market = "on_demand" if req.on_demand else "spot"
    try:
        base_tps = _estimate_replica_tps_for_session(
            session, req.gpu_type, req.tp_size, req.pp_size
        )
    except ValueError as exc:
        return {
            "status": "error",
            "reason": "invalid_placement",
            "message": str(exc),
        }
    launch_timing_s = _scale_launch_timing_for_session(session, req.gpu_type)
    request_now = now or time.time()
    result = SESSION_MANAGER.scale_job(
        session_id,
        now=request_now,
        count=req.count,
        gpu_type=req.gpu_type,
        tp=req.tp_size,
        pp=req.pp_size,
        preferred_region=None,
        preferred_market=market,
        preferred_instance_type=None,
        base_tps=base_tps,
        launch_timing_s=launch_timing_s,
    )
    return result


@app.post("/demo/orca/job/{job_id}/kill")
async def demo_orca_kill(job_id: str, req: KillRequest, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    killed = SESSION_MANAGER.kill_replicas(
        session_id,
        req.replica_ids,
        now=now or time.time(),
        reason="Killed by demo Orca",
    )
    return {"killed": killed}


@app.post("/demo/orca/sim/kill-replica/{replica_id}")
async def demo_orca_kill_replica(replica_id: str, now: Optional[float] = None):
    session_id = _resolve_session_id_from_replica(replica_id)
    killed = SESSION_MANAGER.kill_replicas(
        session_id, [replica_id], now=now or time.time(), reason="Simulated failure"
    )
    if not killed:
        raise HTTPException(status_code=404, detail=f"unknown replica: {replica_id}")
    return {"status": "killed", "replica_id": replica_id}


@app.post("/demo/orca/sim/set-tps/{replica_id}")
async def demo_orca_set_tps(
    replica_id: str, req: ReplicaTpsRequest, now: Optional[float] = None
):
    session_id = _resolve_session_id_from_replica(replica_id)
    try:
        value = SESSION_MANAGER.set_replica_tps(
            session_id, replica_id, target_tps=req.target_tps, now=now or time.time()
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown replica: {replica_id}"
        ) from exc
    return {"status": "updated", "replica_id": replica_id, "tps": value}
