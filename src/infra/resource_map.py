from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from src.core.models import Plan, env_gpu_type
from tandemn_system_data.clients import (  # type: ignore[import-untyped]
    JobStore,
    PostgresClient,
    ResourceMapStore,
)

ACTIVE_CHAIN_STATUSES = ("launching", "running")
WAITING_JOB_STATUSES = ("waiting",)


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


class ResourceMapManager:
    """Read Koi resource/job state from Tandemn Store public clients."""

    def __init__(self, user_id: str | None = None, postgres_client=None):
        self.user_id = user_id
        self._postgres_client = postgres_client

    # ------------------------------------------------------------------
    # Tandemn Store access
    # ------------------------------------------------------------------

    def _client(self):
        if self._postgres_client is None:
            self._postgres_client = PostgresClient()
        return self._postgres_client

    def _effective_user_id(self, user_id: str | None = None) -> str:
        effective = user_id or self.user_id
        if not effective:
            raise ValueError("user_id is required for Tandemn Store resource map access")
        return effective

    def _resource_map_store(self, user_id: str | None = None):
        return ResourceMapStore(self._client(), user_id=self._effective_user_id(user_id))

    def _job_store(self):
        return JobStore(self._client())

    # ------------------------------------------------------------------
    # Jobs and chains
    # ------------------------------------------------------------------

    def get_submitted_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return waiting jobs for compatibility with the old submitted name."""
        return self.get_waiting_jobs(user_id=user_id)

    def get_running_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return running jobs plus active chain allocations."""
        effective_user_id = self._effective_user_id(user_id)
        return [
            self._running_job_to_summary(running_job)
            for running_job in self._job_store().running_jobs(effective_user_id)
        ]

    def get_waiting_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return jobs waiting for placement."""
        effective_user_id = self._effective_user_id(user_id)
        return [
            self._job_to_summary(job) for job in self._job_store().waiting_jobs(effective_user_id)
        ]

    def get_paused_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return paused jobs that may be resumed by a later plan."""
        effective_user_id = self._effective_user_id(user_id)
        return [
            self._job_to_summary(job) for job in self._job_store().paused_jobs(effective_user_id)
        ]

    def get_running_chains(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return active chains with owning job context."""
        chains: list[dict[str, Any]] = []
        for job in self.get_running_jobs(user_id=user_id):
            for chain in job.get("active_chains", []):
                chains.append({"job_id": job["job_id"], "user_id": job["user_id"], **chain})
        return chains

    @staticmethod
    def _model_dump(model) -> dict[str, Any]:
        dump = getattr(model, "model_dump", None)
        if callable(dump):
            return dump(mode="json")
        return dict(model)

    @classmethod
    def _job_to_summary(cls, job) -> dict[str, Any]:
        raw = cls._model_dump(job)
        spec = dict(raw.get("spec_json") or {})
        job_features = dict(spec.get("job_features") or spec.get("features") or spec)
        return {
            "job_id": raw.get("job_id"),
            "user_id": raw.get("user_id"),
            "tenant_id": spec.get("tenant_id", "default"),
            "kind": raw.get("kind"),
            "status": raw.get("status"),
            "created_at": raw.get("created_at"),
            "finished_at": raw.get("finished_at"),
            "finish_reason": raw.get("finish_reason"),
            "job_features": job_features,
            "spec_json": spec,
            "input_source": raw.get("input_source") or {},
            "output_target": raw.get("output_target") or {},
        }

    @classmethod
    def _chain_to_summary(cls, chain) -> dict[str, Any]:
        raw = cls._model_dump(chain)
        return {
            "chain_id": raw.get("chain_id"),
            "plan_id": raw.get("plan_id"),
            "role": raw.get("role"),
            "chain_status": raw.get("status"),
            "shape_json": raw.get("shape_json") or {},
            "target_node": raw.get("target_node"),
        }

    @classmethod
    def _running_job_to_summary(cls, running_job) -> dict[str, Any]:
        job = cls._job_to_summary(running_job.job)
        chains = [cls._chain_to_summary(chain) for chain in running_job.chains]
        job["active_chains"] = chains
        job["current_ladder"] = chains
        return job

    # ------------------------------------------------------------------
    # Koi-facing snapshot API
    # ------------------------------------------------------------------

    def snapshot(self) -> ClusterResourceSnapshot:
        return self.snapshot_cluster_state(tick=None)

    def snapshot_cluster_state(self, tick) -> ClusterResourceSnapshot:
        return ClusterResourceSnapshot(
            tick=tick,
            resources=self.resources_summary(),
            active_jobs=self.get_running_jobs(),
            pending_jobs=self.get_waiting_jobs(),
        )

    def resources_summary(self, user_id: str | None = None) -> dict[str, Any]:
        resource_map = self.get_resource_map(user_id=user_id)
        used = self._used_gpus_by_env(resource_map, user_id=user_id)
        return self._normalized_scheduling_summary(resource_map, used)

    @classmethod
    def _normalized_scheduling_summary(
        cls, resource_map, used: dict[str, int] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Flatten the store ResourceMap to env_key -> capacity info.

        The store map carries total capacity only; ``free`` is derived here
        as ``total`` minus GPUs consumed by running chains (``used``).
        """
        used = used or {}
        raw = dict(resource_map.scheduling_summary())
        market = cls._default_market(resource_map)
        out: dict[str, dict[str, Any]] = {}
        for key, info in raw.items():
            body = dict(info)
            parts = str(key).split("|")
            if len(parts) == 5:
                env_key = str(key)
                body.setdefault("market", parts[0])
            elif len(parts) == 4:
                env_key = "|".join([market, *parts])
                body.setdefault("market", market)
            else:
                env_key = str(key)
            total = int(body.get("total", 0))
            body["free"] = max(0, total - int(used.get(env_key, 0)))
            out[env_key] = body
        return out

    def _used_gpus_by_env(self, resource_map, user_id: str | None = None) -> dict[str, int]:
        """GPUs currently consumed by running chains, bucketed by env_key.

        Free capacity is not stored on the resource map (total-only); it is
        inferred by subtracting this from each env's total. One chain row is
        one launched serving unit, so its GPU footprint is exactly
        ``shape_json["count"]`` (no replica multiplier).

        The chain's placement env is resolved with precedence
        ``target_node`` -> ``shape_json["env"]`` -> ``shape_json["pool_id"]``;
        the store writes the env key into the first-class ``target_node``
        field. A 4-part legacy env is normalized to 5 parts with the map's
        default market.
        """
        default_market = self._default_market(resource_map)
        used: dict[str, int] = {}
        for chain in self.get_running_chains(user_id=user_id):
            shape = chain.get("shape_json") or {}
            raw_env = (
                chain.get("target_node")
                or shape.get("env")
                or shape.get("pool_id")
                or shape.get("target_node")
            )
            if raw_env is None:
                continue
            env_key = self._normalize_env_key(raw_env, default_market)
            # tandemn-store guarantees a positive int 'count' at launch; read
            # it directly with no parallelism-derived fallback.
            count = shape.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(
                    f"chain {chain.get('chain_id')} shape_json missing positive int "
                    f"'count'; got {count!r}"
                )
            used[env_key] = used.get(env_key, 0) + count
        return used

    @classmethod
    def _normalize_env_key(cls, env, default_market: str) -> str:
        """Normalize an env (list/tuple or pipe string) to a 5-part key.

        A 4-part key (``cloud|region|zone|gpu_type``) is prefixed with the
        default market so it matches scheduling_summary's 5-part keys.
        """
        key = cls._env_key(env)
        parts = key.split("|")
        if len(parts) == 4:
            return "|".join([default_market, *parts])
        return key

    @staticmethod
    def _default_market(resource_map) -> str:
        markets = getattr(resource_map, "market", None)
        if markets is None:
            markets = getattr(resource_map, "capacity_type", None)
        if isinstance(markets, (list, tuple)) and markets:
            return str(markets[0])
        if markets:
            return str(markets)
        return "reserved"

    def dynamic_view(self, user_id: str | None = None) -> dict[str, Any]:
        resource_map = self.get_resource_map(user_id=user_id)
        used = self._used_gpus_by_env(resource_map, user_id=user_id)
        resources = self._normalized_scheduling_summary(resource_map, used)
        return {
            "resource_map_version": resource_map.version,
            "updated_at": resource_map.updated_at,
            "resources": resources,
            "running_jobs": self.get_running_jobs(user_id=user_id),
            "waiting_jobs": self.get_waiting_jobs(user_id=user_id),
            "paused_jobs": self.get_paused_jobs(user_id=user_id),
            "running_chains": self.get_running_chains(user_id=user_id),
        }

    def build_keep_all_plan(self, snapshot: ClusterResourceSnapshot) -> dict[str, dict[str, str]]:
        plan: dict[str, dict[str, str]] = {}
        for job in snapshot.active_jobs_summary():
            plan[job["job_id"]] = {"type": "keep"}
        for job in snapshot.pending_jobs_summary():
            plan[job["job_id"]] = {"type": "defer"}
        return plan

    # ------------------------------------------------------------------
    # Resource-map access and simulation
    # ------------------------------------------------------------------

    def get_resource_map(self, user_id: str | None = None):
        return self._resource_map_store(user_id=user_id).get()

    def refresh_resource_map(self, TandemnStore=None):
        return self.get_resource_map()

    def get_avail_capacity(self, env, gpu_type):
        env_key = self._env_key(env)
        requested_gpu = gpu_type or env_gpu_type(env)
        resources = self.resources_summary()
        info = resources.get(env_key)
        if info is None or (requested_gpu is not None and info.get("gpu_type") != requested_gpu):
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
        """GPUs requested per env by a Plan's ladder-bearing actions.

        Accepts a typed Plan or any raw form Plan.from_raw accepts. Counts
        each rank's full GPU footprint (n_replicas * tp * pp), not just its
        replica count.
        """
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


def _smoke_resource_map():
    from tandemn_system_data.models import (  # type: ignore[import-untyped]
        Cloud,
        IntraMachineInterconnect,
        MachinePool,
        NetworkFabric,
        Region,
        ResourceMap,
        Zone,
    )

    pool_fields = getattr(MachinePool, "model_fields", {})
    pool_values = {
        "instance_family": "p4d",
        "gpu_type": "A100",
        "gpu_memory_gb": 40,
        "gpus_per_instance": 8,
        "total_instances": 2,
        "intra_machine_interconnect": IntraMachineInterconnect(type="nvlink_nvswitch"),
    }
    if "price_per_instance_hour" in pool_fields:
        pool_values["price_per_instance_hour"] = 32.77

    resource_fields = getattr(ResourceMap, "model_fields", {})
    resource_values = {
        "clouds": {
            "aws": Cloud(
                regions={
                    "us-east-2": Region(
                        zones={
                            "use2-az3": Zone(
                                network_fabrics={
                                    "efa-cluster-a": NetworkFabric(
                                        fabric_type="efa",
                                        gpu_direct_rdma=True,
                                        machine_pools={
                                            "p4d.24xlarge": MachinePool(**pool_values),
                                        },
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
    }
    if "market" in resource_fields:
        resource_values["market"] = ["reserved"]
    elif "capacity_type" in resource_fields:
        resource_values["capacity_type"] = ["reserved"]
    return ResourceMap(**resource_values)


class _SmokeResourceMapManager(ResourceMapManager):
    def __init__(self):
        super().__init__(user_id="usr_resource_map_smoke")

    def get_resource_map(self, user_id: str | None = None):
        return _smoke_resource_map()

    def get_running_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return []

    def get_waiting_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return []

    def get_paused_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return []

    def get_running_chains(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return []


def _run_smoke(manager: ResourceMapManager, label: str) -> dict[str, Any]:
    resources = manager.resources_summary()
    if not resources:
        raise RuntimeError(f"{label}: resource map is empty")

    env_key, info = next(iter(sorted(resources.items())))
    env_parts = env_key.split("|")
    if len(env_parts) != 5:
        raise AssertionError(f"{label}: env key must have 5 parts, got {env_key!r}")
    for required in ("free", "total", "gpu_type", "cloud", "region", "zone", "market"):
        if required not in info:
            raise AssertionError(f"{label}: env {env_key!r} missing {required!r}")

    free = int(info["free"])
    assert manager.get_avail_capacity(env_parts, info["gpu_type"]) == free

    plan = {
        "actions": [
            {
                "job_id": "job_resource_map_smoke",
                "type": "place",
                "ladder": [
                    {
                        "role": "aggregate",
                        "env": env_parts,
                        "config": {"gpu_count": 1, "gpu_type": info["gpu_type"]},
                        "n_replicas": 1,
                    }
                ],
            }
        ]
    }
    simulated = manager.simulate_future_resources(plan)
    if simulated[env_key]["free_after"] != free - 1:
        raise AssertionError(f"{label}: expected free_after={free - 1}, got {simulated[env_key]}")
    ok, violations = manager.check_resource_feasibility(plan)
    if not ok:
        raise AssertionError(f"{label}: feasible smoke plan failed: {violations}")

    snapshot = manager.snapshot_cluster_state(tick=0)
    dynamic = manager.dynamic_view()
    result = {
        "label": label,
        "env_key": env_key,
        "free": free,
        "total": int(info["total"]),
        "active_jobs": len(snapshot.active_jobs_summary()),
        "pending_jobs": len(snapshot.pending_jobs_summary()),
        "dynamic_view_keys": sorted(dynamic.keys()),
    }
    print(json.dumps(result, indent=2, default=str))
    return result


class _UsedCapacitySmokeManager(ResourceMapManager):
    """80 total A100 GPUs with one running 8-GPU chain placed via target_node.

    Pins the used-capacity contract: free = total - count = 80 - 8 = 72.
    The chain carries its env in the first-class ``target_node`` field (not
    ``shape_json['env']``), exercising the resolution precedence.
    """

    _ENV = "reserved|aws|us-east-2|use2-az3|A100"

    def __init__(self):
        super().__init__(user_id="usr_used_capacity_smoke")

    def get_resource_map(self, user_id: str | None = None):
        from tandemn_system_data.models import (
            Cloud,
            MachinePool,
            NetworkFabric,
            Region,
            ResourceMap,
            Zone,
        )

        pool = MachinePool(
            instance_family="p4d",
            gpu_type="A100",
            gpus_per_instance=8,
            total_instances=10,  # 10 * 8 = 80 total GPUs
        )
        clouds = {
            "aws": Cloud(
                regions={
                    "us-east-2": Region(
                        zones={
                            "use2-az3": Zone(
                                network_fabrics={
                                    "efa-cluster-a": NetworkFabric(
                                        fabric_type="efa",
                                        machine_pools={"p4d.24xlarge": pool},
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
        return ResourceMap(market=["reserved"], clouds=clouds)

    def get_running_chains(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "chain_id": "chain_used_capacity_smoke",
                "target_node": self._ENV,
                "shape_json": {"gpu_count": 8, "count": 8},
            }
        ]


def _run_used_capacity_check() -> dict[str, Any]:
    manager = _UsedCapacitySmokeManager()
    resources = manager.resources_summary()
    env = _UsedCapacitySmokeManager._ENV
    info = resources[env]
    if int(info["total"]) != 80:
        raise AssertionError(f"used-capacity: expected total=80, got {info['total']}")
    if int(info["free"]) != 72:
        raise AssertionError(f"used-capacity: expected free=72 (80-8), got {info['free']}")
    result = {"label": "used-capacity", "env_key": env, "total": 80, "free": int(info["free"])}
    print(json.dumps(result, indent=2, default=str))
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Koi ResourceMapManager")
    parser.add_argument("--user-id", help="Run against Tandemn Store for this user_id")
    args = parser.parse_args()

    if args.user_id:
        _run_smoke(ResourceMapManager(user_id=args.user_id), "tandemn-store")
    else:
        _run_smoke(_SmokeResourceMapManager(), "in-memory")
        _run_used_capacity_check()


if __name__ == "__main__":
    _main()
