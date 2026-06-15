from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.models import Plan
from tandemn_system_data.clients import JobStore, PostgresClient, ResourceMapStore
from tandemn_system_data.models.job import Job, RunningJob
from tandemn_system_data.models.resource_map import ResourceMap, ResourcePool


@dataclass
class ClusterResourceSnapshot:
    tick: int | None
    resources: dict[str, Any]
    active_jobs: list[dict[str, Any]]
    pending_jobs: list[dict[str, Any]]

    def resources_summary(self) -> dict[str, Any]:
        return self.resources

    def active_jobs_summary(self) -> list[dict[str, Any]]:
        return self.active_jobs

    def pending_jobs_summary(self) -> list[dict[str, Any]]:
        return self.pending_jobs


def _waiting_job_to_dict(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "user_id": job.user_id,
        "kind": job.kind.value,
        "status": job.status.value,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "spec_json": job.spec_json,
        "input_source": job.input_source,
        "output_target": job.output_target,
    }


def _running_job_to_dict(running: RunningJob) -> dict[str, Any]:
    job = running.job
    base = _waiting_job_to_dict(job)
    base["active_chains"] = [
        {
            "plan_id": chain.plan_id,
            "chain_id": chain.chain_id,
            "rank_id": f"{chain.role.value}-{idx}",
            "rank_index": idx,
            "role": chain.role.value,
            "chain_status": chain.status.value,
            "target_node": chain.target_node,
            "shape_json": chain.shape_json,
        }
        for idx, chain in enumerate(running.chains)
    ]
    return base


def _pool_env_key(provider: str, instance_type: str, pool: ResourcePool) -> str:
    meta = pool.metadata
    parts = (
        meta.get("cloud", provider),
        meta.get("region"),
        meta.get("market"),
        meta.get("gpu_type", instance_type),
    )
    if any(p is None for p in parts[1:3]):
        return f"{provider}|{instance_type}"
    return "|".join(str(p) for p in parts)


def _resources_summary_from_map(resource_map: ResourceMap) -> dict[str, Any]:
    """Map store ``ResourceMap.pools`` to agent env_key -> {free, total, gpu_type}."""
    summary: dict[str, Any] = {}
    for provider, by_type in resource_map.pools.items():
        for instance_type, pool in by_type.items():
            env_key = _pool_env_key(provider, instance_type, pool)
            meta = pool.metadata
            summary[env_key] = {
                "free": pool.available,
                "total": pool.total,
                "gpu_type": meta.get("gpu_type", instance_type),
                "cloud": meta.get("cloud", provider),
                "region": meta.get("region"),
                "market": meta.get("market"),
            }
    return summary


class ResourceMapManager:
    """Read cluster jobs and capacity from ``tandemn_system_data`` (Postgres)."""

    def __init__(self, user_id: str, postgres_client: PostgresClient | None = None):
        if not user_id:
            raise ValueError("user_id is required")
        self.user_id = user_id
        self._pg = postgres_client or PostgresClient()
        self._jobs = JobStore(self._pg)
        self._resource_map = ResourceMapStore(self._pg, user_id=user_id)

    def get_running_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        uid = user_id or self.user_id
        return [_running_job_to_dict(r) for r in self._jobs.running_jobs(uid)]

    def get_waiting_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        uid = user_id or self.user_id
        return [_waiting_job_to_dict(j) for j in self._jobs.waiting_jobs(uid)]

    def get_submitted_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_waiting_jobs(user_id)

    def get_running_chains(self, user_id: str | None = None) -> list[dict[str, Any]]:
        chains: list[dict[str, Any]] = []
        for job in self.get_running_jobs(user_id):
            for chain in job.get("active_chains", []):
                chains.append(
                    {
                        "job_id": job["job_id"],
                        "user_id": job["user_id"],
                        **chain,
                    }
                )
        return chains

    def snapshot(self) -> ClusterResourceSnapshot:
        return self.snapshot_cluster_state(tick=None)

    def snapshot_cluster_state(self, tick: int | None) -> ClusterResourceSnapshot:
        return ClusterResourceSnapshot(
            tick=tick,
            resources=self.resources_summary(),
            active_jobs=self.get_running_jobs(),
            pending_jobs=self.get_waiting_jobs(),
        )

    def resources_summary(self, user_id: str | None = None) -> dict[str, Any]:
        if user_id is not None and user_id != self.user_id:
            store = ResourceMapStore(self._pg, user_id=user_id)
            return _resources_summary_from_map(store.get())
        return _resources_summary_from_map(self._resource_map.get())

    def build_keep_all_plan(self, snapshot: ClusterResourceSnapshot) -> dict[str, dict[str, str]]:
        plan: dict[str, dict[str, str]] = {}
        for job in snapshot.active_jobs_summary():
            plan[job["job_id"]] = {"type": "keep"}
        for job in snapshot.pending_jobs_summary():
            plan[job["job_id"]] = {"type": "defer"}
        return plan

    def get_resource_map(self) -> ResourceMap:
        return self._resource_map.get()

    def get_avail_capacity(self, env, gpu_type: str) -> int:
        env_key = self._env_key(env)
        info = self.resources_summary().get(env_key)
        if info is None or info.get("gpu_type") != gpu_type:
            return 0
        return int(info.get("free", 0))

    def check_resource_feasibility(self, plan):
        future = self.simulate_future_resources(plan)
        violations = [
            f"env {env}: requested {-info['free_after']} more GPUs than available"
            for env, info in future.items()
            if info["free_after"] < 0
        ]
        return len(violations) == 0, violations

    def simulate_future_resources(self, plan):
        resources = self.resources_summary()
        requested = self._requested_gpus_by_env(plan)
        out = {}
        for env, info in resources.items():
            free_now = int(info.get("free", 0))
            delta = int(requested.get(env, 0))
            out[env] = {
                **info,
                "free_now": free_now,
                "free_after": free_now - delta,
                "delta": -delta,
            }
        for env, delta in requested.items():
            if env not in out:
                out[env] = {"free_now": 0, "free_after": -delta, "delta": -delta}
        return out

    def simulate_resource_state_after(self, plan):
        return self.simulate_future_resources(plan)

    @staticmethod
    def _requested_gpus_by_env(plan) -> dict[str, int]:
        typed = plan if isinstance(plan, Plan) else Plan.from_raw(plan, tick=0)
        requested: dict[str, int] = {}
        for action in typed.actions:
            for rank in action.ladder or []:
                env = ResourceMapManager._env_key(rank.env)
                requested[env] = requested.get(env, 0) + rank.total_gpus()
        return requested

    @staticmethod
    def _env_key(env) -> str:
        if isinstance(env, (tuple, list)):
            return "|".join(str(part) for part in env)
        return str(env)
