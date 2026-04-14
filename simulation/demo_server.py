"""Demo backend for the browser-based Koi + Orca simulator."""

from __future__ import annotations

import asyncio
import json
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

from simulation.demo_runtime import DemoSessionManager
from simulation.demo_scenarios import (
    get_quota_preset,
    get_scenario,
    quota_preset_to_resource_map,
    serialize_catalog,
)
from simulation.model_registry import resolve_model_spec
from simulation.perf_model import DemoPerfModel


app = FastAPI(title="Koi Demo Server", version="0.1")
PERF_MODEL = DemoPerfModel()
SESSION_MANAGER = DemoSessionManager()
STATIC_DIR = Path(__file__).resolve().parent / "static" / "demo"
app.mount("/demo/static", StaticFiles(directory=str(STATIC_DIR)), name="demo-static")
DEMO_KOI_URL = os.environ.get("KOI_DEMO_URL", "http://localhost:8090")
HEARTBEAT_INTERVAL_S = 3.0


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


async def _post_koi(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{DEMO_KOI_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


async def _get_koi_json(path: str) -> dict:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{DEMO_KOI_URL}{path}")
        response.raise_for_status()
        return response.json()


async def _request_koi_decision(
    req: DemoLaunchRequest,
    resource_map: dict,
) -> Optional[dict]:
    payload = {
        "job_request": {
            "model_name": req.model_name,
            "task_type": "batch",
            "avg_input_tokens": req.avg_input_tokens,
            "avg_output_tokens": req.avg_output_tokens,
            "num_requests": req.total_chunks * 10,
            "slo_deadline_hours": req.slo_deadline_hours,
            "objective": "cheapest",
        },
        "resource_map": resource_map,
    }
    return await _post_koi("/decide", payload)


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
        (instance for instance in quota.instances if instance.gpu_type == preferred_gpu),
        quota.instances[0],
    )
    chosen_quota = next(
        (item for item in quota.quotas if item["family"] == chosen_instance.quota_family),
        quota.quotas[0],
    )
    return {
        "job_id": f"{session_id}-r0",
        "group_id": session_id,
        "decision_id": (koi_decision or {}).get("_decision_id"),
        "gpu_type": decision_config.get("gpu_type", preferred_gpu),
        "instance_type": decision_config.get("instance_type", chosen_instance.instance_type),
        "tp": int(decision_config.get("tp", tp)),
        "pp": int(decision_config.get("pp", pp)),
        "dp": int(decision_config.get("dp", 1) or 1),
        "region": decision_config.get("region", chosen_quota["region"]),
        "market": decision_config.get("market", chosen_quota["market"]),
        "total_tokens": int(total_tokens),
        "predicted_tps": float((koi_decision or {}).get("predicted_tps") or baseline_tps),
    }


def _heartbeat_message(phase: str) -> str:
    messages = {
        "searching_capacity": "Searching quota and trying candidate capacity.",
        "provisioning": "Instances requested and provisioning is in progress.",
        "bootstrapping": "Replica booted and is finishing runtime setup.",
        "waiting_model_ready": "Replica provisioned, waiting for model_ready.",
    }
    return messages.get(phase, phase.replace("_", " "))


def _filter_session_koi_jobs(session_id: str, jobs_payload: dict) -> dict:
    jobs = [
        job for job in jobs_payload.get("jobs", [])
        if job.get("job_id", "").startswith(f"{session_id}-") or job.get("job_id") == session_id
    ]
    return {
        "tracked_jobs": len([job for job in jobs if job.get("status") != "launching"]),
        "pending_launches": len([job for job in jobs if job.get("status") == "launching"]),
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


def _build_replica_payloads(session_id: str, snapshot: dict, replica: dict) -> tuple[dict, dict, dict]:
    launch_config = SESSION_MANAGER.sessions[session_id]["launch_config"]
    base = {
        "job_id": replica["replica_id"],
        "decision_id": launch_config["decision_id"],
        "group_id": session_id,
        "gpu_type": replica["gpu_type"],
        "instance_type": replica["instance_type"],
        "tp": replica["tp"],
        "pp": replica["pp"],
        "region": replica["region"],
        "market": replica["market"],
        "attempt_index": 0,
    }
    heartbeat_payload = {
        **base,
        "phase": replica["launch_phase"],
        "message": _heartbeat_message(replica["launch_phase"]),
        "timestamp": time.time(),
    }
    started_payload = {
        **{key: value for key, value in base.items() if key != "attempt_index"},
        "dp": 1,
        "slo_deadline_hours": snapshot["request"]["slo_deadline_hours"],
        "total_tokens": launch_config["total_tokens"],
        "predicted_tps": launch_config["predicted_tps"],
        "is_fallback": False,
    }
    return heartbeat_payload, base, started_payload


async def _sync_session_with_koi(session_id: str, snapshot: dict) -> dict:
    session = SESSION_MANAGER.sessions[session_id]
    if not session.get("koi", {}).get("decision"):
        return {"status": "no_decision"}

    bridge = session["_bridge"]
    now = time.time()
    launches_sent = 0
    starts_sent = 0
    heartbeats_sent = 0
    try:
        for replica in snapshot["runtime"]["replicas"]:
            replica_id = replica["replica_id"]
            heartbeat_payload, launching_payload, started_payload = _build_replica_payloads(session_id, snapshot, replica)
            if replica["phase"] in {"launching", "provisioned"}:
                if replica["phase"] == "provisioned" and replica_id not in bridge["launching_sent"]:
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
                if replica_id not in bridge["launching_sent"]:
                    await _post_koi("/job/launching", launching_payload)
                    bridge["launching_sent"].add(replica_id)
                    launches_sent += 1
                if replica_id not in bridge["started_sent"]:
                    await _post_koi("/job/started", started_payload)
                    bridge["started_sent"].add(replica_id)
                    starts_sent += 1

        session["koi"]["sync_error"] = None
        if starts_sent:
            status = "started_sent"
        elif launches_sent:
            status = "launching_sent"
        elif heartbeats_sent:
            status = "heartbeat_sent"
        else:
            status = "noop"
        return {
            "status": status,
            "launches_sent": launches_sent,
            "starts_sent": starts_sent,
            "heartbeats_sent": heartbeats_sent,
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
    raise HTTPException(status_code=404, detail=f"unknown demo job: {job_id}")


def _resolve_session_id_from_replica(replica_id: str) -> str:
    for session_id, session in SESSION_MANAGER.sessions.items():
        job = session["_orca"]["job"]
        if replica_id in job.replicas:
            return session_id
    raise HTTPException(status_code=404, detail=f"unknown replica: {replica_id}")


def _pick_instance_for_gpu(session: dict, gpu_type: str, market: str) -> tuple[dict[str, Any], dict[str, Any]]:
    instances = session["resource_map"]["instances"]
    instance = next((item for item in instances if item["gpu_type"] == gpu_type), instances[0])
    quotas = session["resource_map"]["quotas"]
    quota = next(
        (
            item for item in quotas
            if item["family"] == instance["quota_family"] and item["market"] == market
        ),
        quotas[0],
    )
    return instance, quota


def _scale_launch_timing_for_session(session: dict, gpu_type: str) -> dict[str, float]:
    capacity_pressure = 0.8 if session["scenario"]["slug"] == "slow_launch" else 0.2
    launch_timing = PERF_MODEL.estimate_launch_timing(gpu_type=gpu_type, capacity_pressure=capacity_pressure)
    multiplier = session["scenario"]["launch_timing_multiplier"] * 0.75
    return {
        "searching_capacity": round(launch_timing.searching_capacity_s * multiplier, 1),
        "provisioning": round(launch_timing.provisioning_s * multiplier, 1),
        "bootstrapping": round(launch_timing.bootstrapping_s * multiplier, 1),
        "waiting_model_ready": round(launch_timing.waiting_model_ready_s * multiplier, 1),
        "total": round(launch_timing.total_seconds * multiplier, 1),
    }


def _estimate_replica_tps_for_session(session: dict, gpu_type: str, tp: int, pp: int) -> float:
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


@app.get("/demo/health")
async def demo_health():
    return {"status": "ok", "sessions": len(SESSION_MANAGER.sessions)}


@app.get("/demo")
async def demo_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/demo/catalog")
async def demo_catalog():
    return serialize_catalog()


@app.post("/demo/launch")
async def demo_launch(req: DemoLaunchRequest):
    try:
        quota = get_quota_preset(req.quota_preset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown quota preset: {req.quota_preset}") from exc

    try:
        scenario = get_scenario(req.scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {req.scenario}") from exc

    model_spec = resolve_model_spec(
        req.model_name,
        dtype=req.dtype,
        overrides=req.model_overrides,
    )

    resource_map = quota_preset_to_resource_map(req.quota_preset)
    koi_decision = None
    koi_error = None
    try:
        koi_decision = await _request_koi_decision(req, resource_map)
    except Exception as exc:
        koi_error = str(exc)

    preferred_gpu = (
        (koi_decision or {}).get("config", {}).get("gpu_type")
        or quota.instances[0].gpu_type
    )
    launch_timing = PERF_MODEL.estimate_launch_timing(
        gpu_type=preferred_gpu,
        capacity_pressure=0.8 if req.scenario == "slow_launch" else 0.2,
    )
    baseline_tps = (
        (koi_decision or {}).get("predicted_tps")
        or PERF_MODEL.estimate_replica_tps(
            model_name=req.model_name,
            gpu_type=preferred_gpu,
            tp=((koi_decision or {}).get("config", {}) or {}).get("tp", 4),
            pp=((koi_decision or {}).get("config", {}) or {}).get("pp", 1),
            input_tokens=req.avg_input_tokens,
            output_tokens=req.avg_output_tokens,
            dtype=req.dtype,
            overrides=req.model_overrides,
        )
    )

    session_id = f"demo-{uuid.uuid4().hex[:10]}"
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
    payload = {
        "session_id": session_id,
        "status": "created",
        "created_at": time.time(),
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
            "decision": koi_decision,
            "error": koi_error,
            "sync_error": None,
            "live": None,
        },
        "launch_preview": {
            "baseline_replica_tps": round(baseline_tps, 1),
            "launch_timing_s": {
                "searching_capacity": round(launch_timing.searching_capacity_s * scenario.launch_timing_multiplier, 1),
                "provisioning": round(launch_timing.provisioning_s * scenario.launch_timing_multiplier, 1),
                "bootstrapping": round(launch_timing.bootstrapping_s * scenario.launch_timing_multiplier, 1),
                "waiting_model_ready": round(launch_timing.waiting_model_ready_s * scenario.launch_timing_multiplier, 1),
                "total": round(launch_timing.total_seconds * scenario.launch_timing_multiplier, 1),
            },
            "preferred_gpu": preferred_gpu,
            "instance_type": launch_config["instance_type"],
            "region": launch_config["region"],
            "market": launch_config["market"],
            "tp": tp,
            "pp": pp,
        },
        "launch_config": launch_config,
    }
    return SESSION_MANAGER.create_session(payload)


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
async def demo_orca_replica_metrics(job_id: str, replica_id: str, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    try:
        return SESSION_MANAGER.get_orca_replica_metrics(session_id, replica_id, now=now)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown replica: {replica_id}") from exc


@app.post("/demo/orca/job/{job_id}/scale")
async def demo_orca_scale(job_id: str, req: ScaleRequest, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    session = _require_session(session_id)
    market = "on_demand" if req.on_demand else "spot"
    instance, quota = _pick_instance_for_gpu(session, req.gpu_type, market)
    launch_timing_s = _scale_launch_timing_for_session(session, req.gpu_type)
    request_now = now or time.time()
    new_replicas = SESSION_MANAGER.scale_job(
        session_id,
        now=request_now,
        count=req.count,
        gpu_type=req.gpu_type,
        instance_type=instance["instance_type"],
        tp=req.tp_size,
        pp=req.pp_size,
        region=quota["region"],
        market=market,
        base_tps=_estimate_replica_tps_for_session(session, req.gpu_type, req.tp_size, req.pp_size),
        launch_timing_s=launch_timing_s,
    )
    return {"status": "scaling", "new_replicas": new_replicas}


@app.post("/demo/orca/job/{job_id}/kill")
async def demo_orca_kill(job_id: str, req: KillRequest, now: Optional[float] = None):
    session_id = _resolve_session_id_from_job(job_id)
    killed = SESSION_MANAGER.kill_replicas(session_id, req.replica_ids, now=now or time.time(), reason="Killed by demo Orca")
    return {"killed": killed}


@app.post("/demo/orca/sim/kill-replica/{replica_id}")
async def demo_orca_kill_replica(replica_id: str, now: Optional[float] = None):
    session_id = _resolve_session_id_from_replica(replica_id)
    killed = SESSION_MANAGER.kill_replicas(session_id, [replica_id], now=now or time.time(), reason="Simulated failure")
    if not killed:
        raise HTTPException(status_code=404, detail=f"unknown replica: {replica_id}")
    return {"status": "killed", "replica_id": replica_id}


@app.post("/demo/orca/sim/set-tps/{replica_id}")
async def demo_orca_set_tps(replica_id: str, req: ReplicaTpsRequest, now: Optional[float] = None):
    session_id = _resolve_session_id_from_replica(replica_id)
    try:
        value = SESSION_MANAGER.set_replica_tps(session_id, replica_id, target_tps=req.target_tps, now=now or time.time())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown replica: {replica_id}") from exc
    return {"status": "updated", "replica_id": replica_id, "tps": value}
