"""
koi/tools/orca_api.py — Orca REST API client.

Wraps all HTTP calls to Orca: submit jobs, scale, kill, fetch metrics.
Agent tools delegate to OrcaClient methods.
"""

from typing import Any, Dict, List, Optional

import aiohttp

from koi.logging_config import get_logger

logger = get_logger("koi.orca_api")


class OrcaClient:
    """Async HTTP client for Orca's REST API."""

    def __init__(self, base_url: str, session: Optional[aiohttp.ClientSession] = None):
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self):
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Resource discovery
    # ------------------------------------------------------------------

    async def get_resources(self) -> Dict[str, Any]:
        """GET /resources → {instances[], quotas[]}"""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/resources", timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    async def submit_batch(
        self,
        model_name: str,
        input_file: str,
        instance_type: str,
        gpu_type: str,
        tp: int,
        pp: int,
        dp: int = 1,
        slo_deadline_hours: float = 8.0,
        avg_input_tokens: int = 512,
        avg_output_tokens: int = 256,
        num_requests: int = 1000,
        max_output_tokens: int = 1024,
        quantization: Optional[str] = None,
        prefer_spot: bool = True,
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """POST /submit/batch → launch a batch inference job."""
        payload = {
            "model_name": model_name,
            "input_file": input_file,
            "task_type": "batch",
            "engine": "vllm",
            "placement": "user_specified",
            "placement_solver": "user_specified",
            "gpu_type": gpu_type,
            "tp_size": tp,
            "pp_size": pp,
            "replicas": dp,
            "slo_deadline_hours": slo_deadline_hours,
            "slo_mode": "cost_first",
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "num_lines": num_requests,
            "max_output_tokens": max_output_tokens,
            "prefer_spot": prefer_spot,
            "chunk_size": chunk_size,
        }
        if quantization:
            payload["quantization_bits"] = quantization

        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/submit/batch",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error(
                    "submit_failed", status=resp.status, response=str(data)[:200]
                )
            return data

    # ------------------------------------------------------------------
    # Job status and metrics
    # ------------------------------------------------------------------

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """GET /job/{id} → job status, progress, metadata."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/job/{job_id}",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """GET /job/{id}/metrics → live throughput, latency, GPU metrics."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/job/{job_id}/metrics",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 404:
                return {}
            resp.raise_for_status()
            return await resp.json()

    async def get_replica_metrics(self, job_id: str, replica_id: str) -> Dict[str, Any]:
        """GET /job/{id}/replicas/{rid}/metrics → per-replica metrics."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/job/{job_id}/replicas/{replica_id}/metrics",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 404:
                return {}
            resp.raise_for_status()
            return await resp.json()

    async def get_chunk_progress(self, job_id: str) -> Dict[str, Any]:
        """GET /job/{id}/chunks/progress → {total, pending, inflight, completed, failed}."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/job/{job_id}/chunks/progress",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 404:
                return {
                    "total": 0,
                    "pending": 0,
                    "inflight": 0,
                    "completed": 0,
                    "failed": 0,
                }
            resp.raise_for_status()
            return await resp.json()

    async def get_replicas(self, job_id: str) -> Dict[str, Any]:
        """GET /job/{id}/replicas → list of replicas with phases."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/job/{job_id}/replicas",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 404:
                return {"replicas": []}
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    async def scale_job(
        self,
        job_id: str,
        gpu_type: str,
        tp: int,
        pp: int,
        count: int,
        on_demand: bool = False,
        planned_market: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """POST /job/{id}/scale → add replicas (can be different GPU type).

        force=True bypasses Orca's feasibility check (used during cold-start
        recovery when the feasibility solver's recommendation OOMed and the
        agent has explicitly chosen a different config).
        """
        payload = {
            "count": count,
            "gpu_type": gpu_type,
            "tp_size": tp,
            "pp_size": pp,
            "on_demand": on_demand,
            "force": force,
        }
        if planned_market:
            payload["planned_market"] = planned_market
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/job/{job_id}/scale",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error(
                    "scale_failed", status=resp.status, response=str(data)[:200]
                )
                resp.raise_for_status()
            return data

    async def kill_replicas(
        self, job_id: str, replica_ids: List[str]
    ) -> Dict[str, Any]:
        """POST /job/{id}/kill → terminate specific replicas, reclaim chunks."""
        payload = {"replica_ids": replica_ids}
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/job/{job_id}/kill",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error(
                    "kill_failed", status=resp.status, response=str(data)[:200]
                )
                resp.raise_for_status()
            return data

    async def swap_replicas(
        self,
        job_id: str,
        gpu_type: str,
        tp: int,
        pp: int,
        num_replicas: Optional[int] = None,
        ready_threshold: int = 1,
        on_demand: bool = False,
        planned_market: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /job/{id}/swap → hot-swap to new GPU config mid-job."""
        payload = {
            "gpu_type": gpu_type,
            "tp_size": tp,
            "pp_size": pp,
            "ready_threshold": ready_threshold,
            "on_demand": on_demand,
        }
        if planned_market:
            payload["planned_market"] = planned_market
        if num_replicas is not None:
            payload["num_replicas"] = num_replicas

        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/job/{job_id}/swap",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error(
                    "swap_failed", status=resp.status, response=str(data)[:200]
                )
                resp.raise_for_status()
            return data


# ---------------------------------------------------------------------------
# Agent tool functions (async — safe to call from tool_runner)
# ---------------------------------------------------------------------------


async def async_launch_chain(
    orca: OrcaClient,
    model_name: str,
    input_file: str,
    instance_type: str,
    gpu_type: str,
    tp: int,
    pp: int,
    dp: int,
    slo_deadline_hours: float,
    avg_input_tokens: int,
    avg_output_tokens: int,
    num_requests: int,
    quantization: Optional[str] = None,
) -> str:
    """Launch a batch job via Orca. Returns job_id and status."""
    result = await orca.submit_batch(
        model_name=model_name,
        input_file=input_file,
        instance_type=instance_type,
        gpu_type=gpu_type,
        tp=tp,
        pp=pp,
        dp=dp,
        slo_deadline_hours=slo_deadline_hours,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        num_requests=num_requests,
        quantization=quantization,
    )
    job_id = result.get("job_id", "unknown")
    status = result.get("status", "unknown")
    return f"Job launched: {job_id} (status={status})"


async def async_scale_chain(
    orca: OrcaClient,
    job_id: str,
    gpu_type: str,
    tp: int,
    pp: int,
    count: int,
    on_demand: bool = False,
) -> str:
    """Scale a running job. Positive count = add replicas. Negative = kill excess."""
    if count > 0:
        planned_market = "on_demand" if on_demand else "spot"
        result = await orca.scale_job(
            job_id,
            gpu_type,
            tp,
            pp,
            count,
            on_demand=on_demand,
            planned_market=planned_market,
        )
        return f"Scaled up: {count} replicas added. {result}"
    elif count < 0:
        replicas_data = await orca.get_replicas(job_id)
        replicas = replicas_data.get("replicas", [])
        active = [
            r["replica_id"]
            for r in replicas
            if r.get("phase") not in ("dead", "killed", "completed", "failed")
        ]
        to_kill = active[: abs(count)]  # kill the first N (oldest)
        if not to_kill:
            return "No active replicas to kill."
        result = await orca.kill_replicas(job_id, to_kill)
        return f"Scaled down: killed {len(to_kill)} replicas. {result}"
    else:
        return "count=0, no action taken."


async def async_get_job_metrics(orca: OrcaClient, job_id: str) -> str:
    """Get live metrics + chunk progress for a running job."""
    metrics = await orca.get_job_metrics(job_id)
    progress = await orca.get_chunk_progress(job_id)

    tps = metrics.get("avg_generation_throughput_toks_per_s", 0)
    cache = metrics.get("gpu_cache_usage_perc", 0)
    running = metrics.get("num_requests_running", 0)
    waiting = metrics.get("num_requests_waiting", 0)
    sm_util = metrics.get("gpu_sm_util_pct", 0)
    mem_bw = metrics.get("gpu_mem_bw_util_pct", 0)

    total = progress.get("total", 0)
    completed = progress.get("completed", 0)
    failed = progress.get("failed", 0)
    pct = (completed / max(total, 1)) * 100

    return "\n".join(
        [
            f"Job {job_id} metrics:",
            f"  Throughput: {tps:.0f} tok/s",
            f"  GPU cache: {cache:.0%} | SM util: {sm_util:.0f}% | Mem BW: {mem_bw:.0f}%",
            f"  Requests: {running} running, {waiting} waiting",
            f"  Chunks: {completed}/{total} done ({pct:.0f}%), {failed} failed",
        ]
    )
