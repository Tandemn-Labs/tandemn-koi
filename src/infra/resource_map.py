from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

ACTIVE_CHAIN_STATUSES = ("pending", "launching", "running")
WAITING_JOB_STATUSES = ("submitted", "planning")


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
    """Read Koi resource/job state from Tandemn Store.

    This first slice only implements read-only job/chain queries. Resource
    capacity simulation remains TODO until the resource-map JSON convention
    is finalized.
    """

    def __init__(self, user_id: str | None = None, postgres_client=None):
        self.user_id = user_id
        self._postgres_client = postgres_client

    # ------------------------------------------------------------------
    # Tandemn Store access
    # ------------------------------------------------------------------

    def _client(self):
        if self._postgres_client is None:
            module = importlib.import_module("tandemn_system_data.clients.postgres")
            PostgresClient = module.PostgresClient
            self._postgres_client = PostgresClient()
        return self._postgres_client

    def _execute(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from sqlalchemy import text

        with self._client().session() as session:
            rows = session.execute(text(sql), params or {}).mappings().all()
        return [dict(row) for row in rows]

    @staticmethod
    def _user_clause(user_id: str | None, table_alias: str = "j") -> str:
        return f"and {table_alias}.user_id = :user_id" if user_id else ""

    @staticmethod
    def _user_params(user_id: str | None) -> dict[str, str]:
        return {"user_id": user_id} if user_id else {}

    # ------------------------------------------------------------------
    # Jobs and chains
    # ------------------------------------------------------------------

    def get_submitted_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return jobs whose canonical job status is submitted."""
        effective_user_id = user_id or self.user_id
        sql = f"""
            select
                j.job_id,
                j.user_id,
                j.kind,
                j.status,
                j.created_at,
                j.completed_at,
                j.spec_json,
                j.input_source,
                j.output_target
            from jobs j
            where j.status = 'submitted'
            {self._user_clause(effective_user_id)}
            order by j.created_at desc
        """
        return self._execute(sql, self._user_params(effective_user_id))

    def get_running_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return jobs with at least one active chain."""
        effective_user_id = user_id or self.user_id
        sql = f"""
            select
                j.job_id,
                j.user_id,
                j.kind,
                j.status as job_status,
                j.created_at as job_created_at,
                j.spec_json,
                pj.plan_id,
                r.rank_id,
                r.rank_index,
                r.status as rank_status,
                c.chain_id,
                c.role,
                c.status as chain_status,
                c.target_node,
                c.shape_json,
                c.parallelism_json
            from jobs j
            join plan_jobs pj on pj.job_id = j.job_id
            join ranks r on r.plan_id = pj.plan_id
            join chains c on c.rank_id = r.rank_id
            where c.status in ('pending', 'launching', 'running')
            {self._user_clause(effective_user_id)}
            order by j.created_at desc, r.rank_index, c.created_at
        """
        return self._group_job_chain_rows(self._execute(sql, self._user_params(effective_user_id)))

    def get_waiting_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return submitted/planning jobs that do not have active chains."""
        effective_user_id = user_id or self.user_id
        sql = f"""
            select
                j.job_id,
                j.user_id,
                j.kind,
                j.status,
                j.created_at,
                j.completed_at,
                j.spec_json,
                j.input_source,
                j.output_target
            from jobs j
            where j.status in ('submitted', 'planning')
            {self._user_clause(effective_user_id)}
            and not exists (
                select 1
                from plan_jobs pj
                join ranks r on r.plan_id = pj.plan_id
                join chains c on c.rank_id = r.rank_id
                where pj.job_id = j.job_id
                and c.status in ('pending', 'launching', 'running')
            )
            order by j.created_at desc
        """
        return self._execute(sql, self._user_params(effective_user_id))

    def get_running_chains(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return active chain rows with their owning job/plan/rank context."""
        effective_user_id = user_id or self.user_id
        sql = f"""
            select
                j.job_id,
                j.user_id,
                pj.plan_id,
                r.rank_id,
                r.rank_index,
                c.chain_id,
                c.role,
                c.status,
                c.target_node,
                c.shape_json,
                c.parallelism_json,
                c.created_at
            from jobs j
            join plan_jobs pj on pj.job_id = j.job_id
            join ranks r on r.plan_id = pj.plan_id
            join chains c on c.rank_id = r.rank_id
            where c.status in ('pending', 'launching', 'running')
            {self._user_clause(effective_user_id)}
            order by j.created_at desc, r.rank_index, c.created_at
        """
        return self._execute(sql, self._user_params(effective_user_id))

    @staticmethod
    def _group_job_chain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        jobs: dict[str, dict[str, Any]] = {}
        for row in rows:
            job = jobs.setdefault(
                row["job_id"],
                {
                    "job_id": row["job_id"],
                    "user_id": row["user_id"],
                    "kind": row["kind"],
                    "status": row["job_status"],
                    "created_at": row["job_created_at"],
                    "spec_json": row["spec_json"],
                    "active_chains": [],
                },
            )
            job["active_chains"].append(
                {
                    "plan_id": row["plan_id"],
                    "rank_id": row["rank_id"],
                    "rank_index": row["rank_index"],
                    "rank_status": row["rank_status"],
                    "chain_id": row["chain_id"],
                    "role": row["role"],
                    "chain_status": row["chain_status"],
                    "target_node": row["target_node"],
                    "shape_json": row["shape_json"],
                    "parallelism_json": row["parallelism_json"],
                }
            )
        return list(jobs.values())

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
        snapshot_json = self._latest_resource_map_json(user_id=user_id)
        return dict(snapshot_json.get("capacity_by_env", {}))

    def build_keep_all_plan(self, snapshot: ClusterResourceSnapshot) -> dict[str, dict[str, str]]:
        plan: dict[str, dict[str, str]] = {}
        for job in snapshot.active_jobs_summary():
            plan[job["job_id"]] = {"action": "keep"}
        for job in snapshot.pending_jobs_summary():
            plan[job["job_id"]] = {"action": "defer"}
        return plan

    def _latest_resource_map_json(self, user_id: str | None = None) -> dict[str, Any]:
        effective_user_id = user_id or self.user_id
        sql = f"""
            select rm.snapshot_json
            from resource_maps rm
            where 1 = 1
            {self._user_clause(effective_user_id, table_alias="rm")}
            order by rm.captured_at desc
            limit 1
        """
        rows = self._execute(sql, self._user_params(effective_user_id))
        return rows[0]["snapshot_json"] if rows else {}

    # ------------------------------------------------------------------
    # Resource-map placeholders
    # ------------------------------------------------------------------

    def get_resource_map(self, TandemnStore=None):
        return self._latest_resource_map_json()

    def refresh_resource_map(self, TandemnStore):
        # Placeholder: refresh the resource map in place from TandemnStore
        pass

    def get_avail_capacity(self, env, gpu_type):
        env_key = self._env_key(env)
        resources = self.resources_summary()
        info = resources.get(env_key)
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
        requested: dict[str, int] = {}
        for action in plan.values():
            ladder = action.get("ladder") if isinstance(action, dict) else None
            if not isinstance(ladder, dict):
                continue
            for rank in ladder.get("ranks", []):
                env = ResourceMapManager._env_key(rank.get("env"))
                requested[env] = requested.get(env, 0) + int(rank.get("n_replicas", 1))
        return requested

    @staticmethod
    def _env_key(env) -> str:
        if isinstance(env, (tuple, list)):
            return "|".join(str(part) for part in env)
        return str(env)


# SMOKE_ENV = "aws|us-east-1|on_demand|H100"
# SMOKE_SNAPSHOT_JSON = {
#     "capacity_by_env": {
#         SMOKE_ENV: {
#             "cloud": "aws",
#             "region": "us-east-1",
#             "market": "on_demand",
#             "gpu_type": "H100",
#             "free": 4,
#             "total": 8,
#         }
#     }
# }
# SMOKE_OK_PLAN = {
#     "job_ok": {
#         "action": "place",
#         "ladder": {"ranks": [{"env": SMOKE_ENV, "n_replicas": 2}]},
#     }
# }
# SMOKE_BAD_PLAN = {
#     "job_bad": {
#         "action": "place",
#         "ladder": {"ranks": [{"env": SMOKE_ENV, "n_replicas": 5}]},
#     }
# }


# class _SmokeResourceMapManager(ResourceMapManager):
#     def _latest_resource_map_json(self, user_id: str | None = None) -> dict[str, Any]:
#         return SMOKE_SNAPSHOT_JSON


# def _run_capacity_smoke(manager: ResourceMapManager, label: str) -> None:
#     print(f"[{label}] Initial resources summary:", manager.resources_summary())
#     assert manager.resources_summary()[SMOKE_ENV]["free"] == 4
#     print(f"[{label}] Available capacity (get_avail_capacity):", manager.get_avail_capacity(SMOKE_ENV, "H100"))
#     assert manager.get_avail_capacity(SMOKE_ENV, "H100") == 4
#     print(f"[{label}] Simulating future resources (SMOKE_OK_PLAN)...")
#     simulated = manager.simulate_future_resources(SMOKE_OK_PLAN)
#     print(f"[{label}] simulate_future_resources:", simulated)
#     assert simulated[SMOKE_ENV]["free_after"] == 2

#     print(f"[{label}] Checking resource feasibility for OK plan...")
#     ok, violations = manager.check_resource_feasibility(SMOKE_OK_PLAN)
#     print(f"[{label}] check_resource_feasibility (OK): ok={ok}, violations={violations}")
#     assert ok and not violations

#     print(f"[{label}] Checking resource feasibility for BAD plan...")
#     ok, violations = manager.check_resource_feasibility(SMOKE_BAD_PLAN)
#     print(f"[{label}] check_resource_feasibility (BAD): ok={ok}, violations={violations}")
#     assert not ok and violations
#     print(f"resource_map {label} smoke passed")


# def _run_db_smoke() -> None:
#     from datetime import UTC, datetime
#     from uuid import uuid4

#     from sqlalchemy import text
#     from tandemn_system_data.db import ResourceMapRow, UserRow

#     user_id = f"usr_koi_smoke_{uuid4().hex[:8]}"
#     resource_map_id = f"rmap_koi_smoke_{uuid4().hex[:8]}"
#     manager = ResourceMapManager(user_id=user_id)

#     print(f"[db smoke] Creating test user {user_id} and resource_map {resource_map_id}...")
#     with manager._client().begin() as session:
#         session.add(UserRow(user_id=user_id, name="koi smoke", created_at=datetime.now(UTC)))
#         session.flush()
#         session.add(
#             ResourceMapRow(
#                 resource_map_id=resource_map_id,
#                 user_id=user_id,
#                 snapshot_json=SMOKE_SNAPSHOT_JSON,
#                 captured_at=datetime.now(UTC),
#             )
#         )

#     try:
#         print("[db smoke] Running capacity smoke tests...")
#         _run_capacity_smoke(manager, "sql")
#     finally:
#         print(f"[db smoke] Cleaning up test user {user_id}...")
#         with manager._client().begin() as session:
#             session.execute(
#                 text("delete from users where user_id = :user_id"), {"user_id": user_id}
#             )


# if __name__ == "__main__":
#     import sys

#     print("[main] Running in-memory smoke test...")
#     _run_capacity_smoke(_SmokeResourceMapManager(), "in-memory")
#     if "--db" in sys.argv:
#         print("[main] Running database-backed smoke test...")
#         _run_db_smoke()
