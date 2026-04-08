"""
simulation/mock_orca.py — Mock Orca server for testing Koi without GPUs.

Mocks every Orca endpoint that Koi calls, plus control endpoints to simulate
replica kills, TPS changes, and job completion.

Usage:
    # Terminal 1: Start mock Orca (port 26336, simulates 4 replicas of Qwen3-32B)
    python simulation/mock_orca.py --replicas 4 --tps 1200 --model Qwen/Qwen3-32B \
        --total-chunks 500 --slo 8

    # Terminal 2: Start Koi pointing at mock Orca
    ORCA_URL=http://localhost:26336 KOI_PORT=8090 python -m koi.server

    # Terminal 3: Trigger events
    curl -X POST localhost:26336/sim/kill-replica/mo-qwen32b-sim-r0
    curl -X POST localhost:26336/sim/set-tps/mo-qwen32b-sim-r1 -d '{"tps": 500}'
    curl -X POST localhost:26336/sim/complete

Control endpoints (prefixed /sim/):
    POST /sim/kill-replica/{replica_id}     — simulate replica death
    POST /sim/set-tps/{replica_id}          — change replica TPS  {"tps": 800}
    POST /sim/complete                      — force job completion
    POST /sim/add-replica                   — simulate scale-up completing
    GET  /sim/state                         — dump full simulator state
"""

import argparse
import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("mock_orca")

app = FastAPI(title="Mock Orca", version="0.1")


# ---------------------------------------------------------------------------
# Simulated state
# ---------------------------------------------------------------------------

@dataclass
class SimReplica:
    replica_id: str
    phase: str = "running"          # launching, provisioned, model_ready, running, dead, failed, killed, completed
    base_tps: float = 1200.0        # baseline tokens/sec (wobble applied on read)
    gpu_type: str = "L40S"
    instance_type: str = "g6e.12xlarge"
    tp: int = 4
    pp: int = 2
    region: str = "us-east-1"
    market: str = "on_demand"
    config_index: int = 0
    started_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    warmup_seconds: float = 30.0    # ramp from 0 to base_tps over this period
    wobble_pct: float = 0.10        # ±10% random noise on each read

    @property
    def tps(self) -> float:
        """Current TPS with warmup ramp + realistic wobble."""
        if self.phase != "running":
            return 0.0
        elapsed = time.time() - self.started_at
        # Warmup: ramp linearly from 0 → base_tps
        if elapsed < self.warmup_seconds:
            ramp = elapsed / self.warmup_seconds
        else:
            ramp = 1.0
        base = self.base_tps * ramp
        # Wobble: ±wobble_pct gaussian noise
        noise = random.gauss(1.0, self.wobble_pct)
        return max(0, base * noise)


@dataclass
class SimJob:
    job_id: str
    model_name: str
    replicas: Dict[str, SimReplica] = field(default_factory=dict)
    total_chunks: int = 500
    completed_chunks: int = 0
    failed_chunks: int = 0
    status: str = "running"         # running, succeeded, failed
    slo_deadline_hours: float = 8.0
    decision_id: Optional[str] = None
    # Tokens per chunk (approximation)
    tokens_per_chunk: int = 12000   # ~6M tokens / 500 chunks
    deploy_timestamp: float = field(default_factory=time.time)


class SimState:
    """Global simulator state."""
    def __init__(self):
        self.jobs: Dict[str, SimJob] = {}
        self.koi_url: str = "http://localhost:8090"
        self._chunk_task: Optional[asyncio.Task] = None

    @property
    def primary_job(self) -> Optional[SimJob]:
        return next(iter(self.jobs.values()), None)

SIM = SimState()


# ---------------------------------------------------------------------------
# Chunk progress background task
# ---------------------------------------------------------------------------

async def _advance_chunks():
    """Background loop: advance completed chunks based on aggregate TPS."""
    while True:
        await asyncio.sleep(5)  # tick every 5s
        for job in list(SIM.jobs.values()):
            if job.status != "running":
                continue
            # Aggregate TPS from alive replicas
            agg_tps = sum(r.tps for r in job.replicas.values()
                          if r.phase == "running")
            if agg_tps <= 0:
                continue
            # Tokens generated in 5s
            tokens_this_tick = agg_tps * 5
            chunks_this_tick = tokens_this_tick / max(job.tokens_per_chunk, 1)
            job.completed_chunks = min(
                job.total_chunks,
                job.completed_chunks + int(chunks_this_tick),
            )
            # Update heartbeats for alive replicas
            now = time.time()
            for r in job.replicas.values():
                if r.phase == "running":
                    r.last_heartbeat = now

            # Check completion
            if job.completed_chunks >= job.total_chunks:
                job.status = "succeeded"
                job.completed_chunks = job.total_chunks
                logger.info(f"[Sim] Job {job.job_id} completed all chunks!")
                await _notify_koi_complete(job)


async def _notify_koi_complete(job: SimJob):
    """POST /job/complete to Koi."""
    agg_tps = sum(r.tps for r in job.replicas.values() if r.phase in ("running", "completed"))
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{SIM.koi_url}/job/complete", json={
                "job_id": job.job_id,
                "status": "succeeded",
                "metrics": {
                    "avg_generation_throughput_toks_per_s": agg_tps,
                    "throughput_tokens_per_sec": agg_tps,
                },
            }, timeout=5)
        logger.info(f"[Sim] Notified Koi: job {job.job_id} complete (TPS={agg_tps:.0f})")
    except Exception as e:
        logger.warning(f"[Sim] Failed to notify Koi of completion: {e}")


async def _notify_koi_replica_started(job: SimJob, replica: SimReplica):
    """POST /job/started to Koi (replica ready)."""
    elapsed_since_deploy = time.time() - job.deploy_timestamp
    adjusted_slo = max(0.1, job.slo_deadline_hours - elapsed_since_deploy / 3600)
    total_tokens = job.total_chunks * job.tokens_per_chunk
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{SIM.koi_url}/job/started", json={
                "job_id": replica.replica_id,
                "decision_id": job.decision_id,
                "group_id": job.job_id,
                "gpu_type": replica.gpu_type,
                "instance_type": replica.instance_type,
                "tp": replica.tp,
                "pp": replica.pp,
                "dp": 1,
                "slo_deadline_hours": adjusted_slo,
                "total_tokens": total_tokens,
                "predicted_tps": 0.0,
                "is_fallback": replica.config_index > 0,
            }, timeout=5)
            logger.info(f"[Sim] Notified Koi: replica {replica.replica_id} started (resp={resp.status_code})")
    except Exception as e:
        logger.warning(f"[Sim] Failed to notify Koi of replica start: {e}")


async def _notify_koi_replica_failed(job: SimJob, replica: SimReplica, reason: str):
    """POST /job/replica-failed to Koi."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{SIM.koi_url}/job/replica-failed", json={
                "job_id": replica.replica_id,
                "group_id": job.job_id,
                "status": "failed",
                "reason": reason,
            }, timeout=5)
        logger.info(f"[Sim] Notified Koi: replica {replica.replica_id} failed ({reason})")
    except Exception as e:
        logger.warning(f"[Sim] Failed to notify Koi of replica failure: {e}")


# ---------------------------------------------------------------------------
# Orca API endpoints (what Koi calls)
# ---------------------------------------------------------------------------

@app.get("/resources")
async def get_resources():
    """Mock resource discovery — matches Orca Shape C format."""
    return {
        "instances": [
            {
                "instance_type": "g6e.12xlarge",
                "gpu_type": "L40S",
                "gpus_per_instance": 4,
                "gpu_memory_gb": 48,
                "vcpus": 48,
                "quota_family": "G6E",
                "cost_per_instance_hour_usd": 7.35,
            },
            {
                "instance_type": "g6e.48xlarge",
                "gpu_type": "L40S",
                "gpus_per_instance": 8,
                "gpu_memory_gb": 48,
                "vcpus": 192,
                "quota_family": "G6E",
                "cost_per_instance_hour_usd": 14.69,
            },
        ],
        "quotas": [
            {"family": "G6E", "region": "us-east-1", "market": "on_demand",
             "baseline_vcpus": 384, "used_vcpus": 0},
            {"family": "G6E", "region": "us-east-2", "market": "on_demand",
             "baseline_vcpus": 192, "used_vcpus": 0},
        ],
    }


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Job status."""
    job = SIM.jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "model_name": job.model_name,
        "num_replicas": len(job.replicas),
        "active_replicas": sum(1 for r in job.replicas.values() if r.phase == "running"),
    }


@app.get("/job/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Aggregate metrics for the job."""
    job = SIM.jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    agg_tps = sum(r.tps for r in job.replicas.values() if r.phase == "running")
    return {
        "avg_generation_throughput_toks_per_s": agg_tps,
        "gpu_cache_usage_perc": 0.65,
        "num_requests_running": 32,
        "num_requests_waiting": 0,
        "gpu_sm_util_pct": 78,
        "gpu_mem_bw_util_pct": 45,
    }


@app.get("/job/{job_id}/replicas/{replica_id}/metrics")
async def get_replica_metrics(job_id: str, replica_id: str):
    """Per-replica metrics."""
    job = SIM.jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    replica = job.replicas.get(replica_id)
    if not replica:
        raise HTTPException(404, f"Replica {replica_id} not found")
    if replica.phase not in ("running",):
        return {
            "avg_generation_throughput_toks_per_s": 0,
            "gpu_cache_usage_perc": 0,
            "gpu_sm_util_pct": 0,
            "gpu_mem_bw_util_pct": 0,
        }
    return {
        "avg_generation_throughput_toks_per_s": replica.tps,
        "gpu_cache_usage_perc": max(0, min(1, random.gauss(0.65, 0.05))),
        "num_requests_running": random.randint(6, 12),
        "num_requests_waiting": random.randint(0, 3),
        "gpu_sm_util_pct": max(0, random.gauss(78, 5)),
        "gpu_mem_bw_util_pct": max(0, random.gauss(45, 4)),
    }


@app.get("/job/{job_id}/chunks/progress")
async def get_chunk_progress(job_id: str):
    """Chunk progress."""
    job = SIM.jobs.get(job_id)
    if not job:
        return {"total": 0, "pending": 0, "inflight": 0, "completed": 0, "failed": 0, "all_done": False}
    pending = job.total_chunks - job.completed_chunks - job.failed_chunks
    return {
        "total": job.total_chunks,
        "pending": max(0, pending),
        "inflight": 0,
        "completed": job.completed_chunks,
        "failed": job.failed_chunks,
        "all_done": job.completed_chunks + job.failed_chunks >= job.total_chunks,
    }


@app.get("/job/{job_id}/replicas")
async def get_replicas(job_id: str):
    """List replicas with phases."""
    job = SIM.jobs.get(job_id)
    if not job:
        return {"replicas": []}
    return {
        "replicas": [
            {
                "replica_id": r.replica_id,
                "phase": r.phase,
                "region": r.region,
                "market": r.market,
                "instance_type": r.instance_type,
                "has_metrics": r.phase == "running",
            }
            for r in job.replicas.values()
        ]
    }


@app.post("/submit/batch")
async def submit_batch(payload: Dict[str, Any]):
    """Mock batch submission. Creates simulated job + replicas."""
    job = SIM.primary_job
    if not job:
        raise HTTPException(500, "No simulated job configured")
    return {"job_id": job.job_id, "status": "launching", "replicas": len(job.replicas)}


class ScaleRequest(BaseModel):
    count: int = 1
    gpu_type: str = "L40S"
    tp_size: int = 4
    pp_size: int = 2
    on_demand: bool = True


@app.post("/job/{job_id}/scale")
async def scale_job(job_id: str, req: ScaleRequest):
    """Mock scale-up. Adds new replicas after a simulated delay."""
    job = SIM.jobs.get(job_id)
    if not job:
        raise HTTPException(404)

    new_replicas = []
    for i in range(req.count):
        idx = len(job.replicas)
        rid = f"{job.job_id}-r{idx}"
        replica = SimReplica(
            replica_id=rid,
            phase="launching",
            base_tps=0,
            gpu_type=req.gpu_type,
            instance_type="g6e.12xlarge",
            tp=req.tp_size,
            pp=req.pp_size,
        )
        job.replicas[rid] = replica
        new_replicas.append(rid)
        logger.info(f"[Sim] Scale-up: replica {rid} launching (TP={req.tp_size} PP={req.pp_size})")

    # Simulate launch delay, then notify Koi
    asyncio.create_task(_delayed_replica_ready(job, new_replicas))

    return {"status": "scaling", "new_replicas": new_replicas}


async def _delayed_replica_ready(job: SimJob, replica_ids: List[str], delay: float = 15.0):
    """Simulate provisioning + model loading delay, then mark running and notify Koi."""
    await asyncio.sleep(delay)
    for rid in replica_ids:
        replica = job.replicas.get(rid)
        if replica and replica.phase == "launching":
            replica.phase = "running"
            replica.base_tps = 1200.0  # default TPS for new replica
            replica.started_at = time.time()
            replica.last_heartbeat = time.time()
            logger.info(f"[Sim] Replica {rid} now running (TPS={replica.tps})")
            await _notify_koi_replica_started(job, replica)


class KillRequest(BaseModel):
    replica_ids: List[str]


@app.post("/job/{job_id}/kill")
async def kill_replicas(job_id: str, req: KillRequest):
    """Mock kill replicas."""
    job = SIM.jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    killed = []
    for rid in req.replica_ids:
        replica = job.replicas.get(rid)
        if replica and replica.phase not in ("dead", "killed", "failed", "completed"):
            replica.phase = "killed"
            replica.base_tps = 0
            killed.append(rid)
            logger.info(f"[Sim] Killed replica {rid}")
    return {"killed": killed}


class SwapRequest(BaseModel):
    gpu_type: str = "L40S"
    tp_size: int = 4
    pp_size: int = 2
    num_replicas: Optional[int] = None
    ready_threshold: int = 1
    on_demand: bool = True


@app.post("/job/{job_id}/swap")
async def swap_replicas(job_id: str, req: SwapRequest):
    """Mock hot-swap. Kills old replicas, launches new ones."""
    job = SIM.jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    return {"status": "swapping", "message": "Mock swap — not implemented in sim"}


# ---------------------------------------------------------------------------
# Control endpoints (/sim/*)
# ---------------------------------------------------------------------------

@app.post("/sim/kill-replica/{replica_id}")
async def sim_kill_replica(replica_id: str):
    """Simulate a replica dying (EC2 termination, spot preemption, OOM, etc.)."""
    for job in SIM.jobs.values():
        replica = job.replicas.get(replica_id)
        if replica:
            replica.phase = "dead"
            replica.base_tps = 0
            logger.info(f"[Sim] Killed replica {replica_id} (simulated death)")
            await _notify_koi_replica_failed(job, replica, "Simulated EC2 termination")
            return {"status": "killed", "replica_id": replica_id}
    raise HTTPException(404, f"Replica {replica_id} not found")


class SetTpsRequest(BaseModel):
    tps: float


@app.post("/sim/set-tps/{replica_id}")
async def sim_set_tps(replica_id: str, req: SetTpsRequest):
    """Change a replica's TPS (simulate throttling, degradation, etc.)."""
    for job in SIM.jobs.values():
        replica = job.replicas.get(replica_id)
        if replica:
            old_tps = replica.base_tps
            replica.base_tps = req.tps
            logger.info(f"[Sim] Set TPS for {replica_id}: {old_tps:.0f} → {req.tps:.0f}")
            return {"status": "updated", "replica_id": replica_id, "tps": req.tps}
    raise HTTPException(404, f"Replica {replica_id} not found")


@app.post("/sim/complete")
async def sim_force_complete():
    """Force-complete the primary job."""
    job = SIM.primary_job
    if not job:
        raise HTTPException(404, "No job running")
    job.completed_chunks = job.total_chunks
    job.status = "succeeded"
    await _notify_koi_complete(job)
    return {"status": "completed", "job_id": job.job_id}


class DegradeRequest(BaseModel):
    target_tps: float = 50
    over_seconds: float = 60


@app.post("/sim/degrade/{replica_id}")
async def sim_degrade_replica(replica_id: str, req: DegradeRequest):
    """Gradually reduce a replica's TPS over time (simulates KV cache thrashing, mem pressure)."""
    for job in SIM.jobs.values():
        replica = job.replicas.get(replica_id)
        if replica:
            start_tps = replica.base_tps
            logger.info(f"[Sim] Degrading {replica_id}: {start_tps:.0f} → {req.target_tps:.0f} over {req.over_seconds}s")
            asyncio.create_task(_gradual_degrade(replica, start_tps, req.target_tps, req.over_seconds))
            return {"status": "degrading", "replica_id": replica_id,
                    "from_tps": start_tps, "to_tps": req.target_tps, "seconds": req.over_seconds}
    raise HTTPException(404, f"Replica {replica_id} not found")


async def _gradual_degrade(replica: SimReplica, start_tps: float, target_tps: float, seconds: float):
    """Linearly ramp TPS from start to target over N seconds."""
    steps = int(seconds / 2)  # update every 2s
    for i in range(steps + 1):
        frac = i / max(steps, 1)
        replica.base_tps = start_tps + (target_tps - start_tps) * frac
        await asyncio.sleep(2)
    replica.base_tps = target_tps


@app.post("/sim/add-replica")
async def sim_add_replica():
    """Manually add a replica that's immediately running (no launch delay)."""
    job = SIM.primary_job
    if not job:
        raise HTTPException(404, "No job running")
    idx = len(job.replicas)
    rid = f"{job.job_id}-r{idx}"
    replica = SimReplica(replica_id=rid, phase="running", base_tps=1200.0)
    job.replicas[rid] = replica
    await _notify_koi_replica_started(job, replica)
    return {"status": "added", "replica_id": rid}


@app.get("/sim/state")
async def sim_state():
    """Dump full simulator state."""
    result = {}
    for jid, job in SIM.jobs.items():
        agg_tps = sum(r.tps for r in job.replicas.values() if r.phase == "running")
        alive = sum(1 for r in job.replicas.values() if r.phase == "running")
        pct = (job.completed_chunks / max(job.total_chunks, 1)) * 100
        result[jid] = {
            "status": job.status,
            "model": job.model_name,
            "chunks": f"{job.completed_chunks}/{job.total_chunks} ({pct:.0f}%)",
            "aggregate_tps": round(agg_tps, 1),
            "replicas_alive": alive,
            "replicas_total": len(job.replicas),
            "replicas": {
                rid: {"phase": r.phase, "tps": r.tps, "gpu": r.gpu_type,
                       "tp": r.tp, "pp": r.pp}
                for rid, r in job.replicas.items()
            },
        }
    return result


# ---------------------------------------------------------------------------
# Startup: create initial job + replicas, ask Koi for /decide, send /job/started
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Start chunk advancement loop."""
    SIM._chunk_task = asyncio.create_task(_advance_chunks())
    logger.info("[Sim] Chunk advancement loop started")


async def init_scenario(
    model_name: str, num_replicas: int, tps_per_replica: float,
    total_chunks: int, slo_hours: float, gpu_type: str,
    tp: int, pp: int, koi_url: str,
):
    """Initialize the simulated scenario: create job, ask Koi for decision, register replicas."""
    SIM.koi_url = koi_url
    short = model_name.split("/")[-1].lower()[:10]
    job_id = f"mo-{short}-sim"

    job = SimJob(
        job_id=job_id,
        model_name=model_name,
        total_chunks=total_chunks,
        slo_deadline_hours=slo_hours,
        tokens_per_chunk=12000,
    )

    # Create replicas
    for i in range(num_replicas):
        rid = f"{job_id}-r{i}"
        job.replicas[rid] = SimReplica(
            replica_id=rid,
            base_tps=tps_per_replica,
            gpu_type=gpu_type,
            tp=tp, pp=pp,
            config_index=0,
        )

    SIM.jobs[job_id] = job
    logger.info(f"[Sim] Job {job_id}: {num_replicas} replicas × {tps_per_replica} TPS, "
                f"{total_chunks} chunks, SLO={slo_hours}h")

    # Step 1: Ask Koi for a decision
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{koi_url}/decide", json={
                "job_request": {
                    "model_name": model_name,
                    "task_type": "batch",
                    "avg_input_tokens": 800,
                    "avg_output_tokens": 200,
                    "num_requests": total_chunks * 10,
                    "slo_deadline_hours": slo_hours,
                    "objective": "cheapest",
                },
                "resource_map": await get_resources(),
            }, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                job.decision_id = data.get("_decision_id")
                # Use agent's decided config for replicas
                cfg = data.get("config", {})
                if cfg.get("tp"):
                    for r in job.replicas.values():
                        r.tp = cfg["tp"]
                        r.pp = cfg.get("pp", r.pp)
                        r.gpu_type = cfg.get("gpu_type", r.gpu_type)
                        r.instance_type = cfg.get("instance_type", r.instance_type)
                logger.info(f"[Sim] Koi decision: {cfg.get('gpu_type', '?')} "
                           f"TP={cfg.get('tp', '?')} PP={cfg.get('pp', '?')} "
                           f"(decision_id={job.decision_id})")
            else:
                logger.warning(f"[Sim] Koi /decide returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"[Sim] Could not reach Koi at {koi_url}/decide: {e}")
        logger.info("[Sim] Continuing without Koi decision — replicas will register without decision_id")

    # Step 2: Simulate launch delay, then notify Koi that replicas are ready
    await asyncio.sleep(2)
    for rid, replica in job.replicas.items():
        replica.phase = "running"
        await _notify_koi_replica_started(job, replica)
        await asyncio.sleep(0.5)

    logger.info(f"[Sim] All {num_replicas} replicas registered with Koi. Chunks advancing.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mock Orca server for Koi testing")
    parser.add_argument("--port", type=int, default=26336, help="Port to listen on")
    parser.add_argument("--koi-url", default="http://localhost:8090", help="Koi service URL")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Model name to simulate")
    parser.add_argument("--replicas", type=int, default=4, help="Number of replicas")
    parser.add_argument("--tps", type=float, default=1200, help="TPS per replica")
    parser.add_argument("--total-chunks", type=int, default=500, help="Total chunks in job")
    parser.add_argument("--slo", type=float, default=8.0, help="SLO deadline in hours")
    parser.add_argument("--gpu-type", default="L40S", help="GPU type")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel")
    parser.add_argument("--pp", type=int, default=2, help="Pipeline parallel")
    parser.add_argument("--no-decide", action="store_true", help="Skip calling Koi /decide")
    args = parser.parse_args()

    import uvicorn

    @app.on_event("startup")
    async def init():
        if not args.no_decide:
            # Wait a moment for Koi to be ready, then init scenario
            asyncio.create_task(_init_with_retry(args))
        else:
            # Just create the job without asking Koi
            await init_scenario(
                model_name=args.model, num_replicas=args.replicas,
                tps_per_replica=args.tps, total_chunks=args.total_chunks,
                slo_hours=args.slo, gpu_type=args.gpu_type,
                tp=args.tp, pp=args.pp, koi_url=args.koi_url,
            )

    uvicorn.run(app, host="0.0.0.0", port=args.port)


async def _init_with_retry(args, max_retries: int = 12, delay: float = 5.0):
    """Wait for Koi to be ready, then initialize the scenario."""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{args.koi_url}/health", timeout=3)
                if resp.status_code == 200:
                    logger.info(f"[Sim] Koi is ready (attempt {attempt + 1})")
                    await init_scenario(
                        model_name=args.model, num_replicas=args.replicas,
                        tps_per_replica=args.tps, total_chunks=args.total_chunks,
                        slo_hours=args.slo, gpu_type=args.gpu_type,
                        tp=args.tp, pp=args.pp, koi_url=args.koi_url,
                    )
                    return
        except Exception:
            pass
        logger.info(f"[Sim] Waiting for Koi... (attempt {attempt + 1}/{max_retries})")
        await asyncio.sleep(delay)
    logger.error("[Sim] Koi never became ready. Starting without /decide.")


if __name__ == "__main__":
    main()
