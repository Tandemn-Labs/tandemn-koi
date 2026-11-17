from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.models import LADDER_ACTIONS, Plan, env_gpu_type
from tandemn_system_data.clients import PostgresClient  # type: ignore[import-untyped]

DEFAULT_TICK_INTERVAL_SECONDS = 300


# TODO: Remove with the legacy telemetry chain-index matching.
# def chain_id_for_rank(rank_id: str, chain_index: int) -> str:
#     return f"{rank_id}_chain_{chain_index}"


def merge_hardware_catalogs(catalogs: Sequence[object]) -> dict[str, Any]:
    """Normalize provider catalog JSON into deployment X's region schema."""
    regions = []
    for catalog in catalogs:
        if not isinstance(catalog, dict):
            continue
        source_regions = catalog.get("regions", [])
        if source_regions and isinstance(source_regions[0], dict):
            regions.extend(source_regions)
            continue
        cloud = catalog.get("cloud")
        if not isinstance(cloud, str):
            continue
        for region in source_regions:
            if not isinstance(region, str):
                continue
            regions.append(
                {
                    "cloud": cloud,
                    "region": region,
                    "instance_types": [
                        instance
                        for instance in catalog.get("instance_types", [])
                        if any(
                            offering.get("region") == region
                            for offering in instance.get("offerings", [])
                        )
                    ],
                }
            )
    return {"regions": regions}


@dataclass
class ClusterResourceSnapshot:
    """Tick inputs; PostgreSQL resources retain the raw per-cluster structure."""

    tick: int | None
    resources: dict[str, Any]
    active_jobs: list[dict[str, Any]]
    pending_jobs: list[dict[str, Any]]
    resource_map_snapshot: dict[str, Any] | None = None

    def resources_summary(self) -> dict[str, Any]:
        return self.resources

    def active_jobs_summary(self) -> list[dict[str, Any]]:
        return self.active_jobs

    def pending_jobs_summary(self) -> list[dict[str, Any]]:
        return self.pending_jobs

    def current_ladder(self, job_id: str) -> list[dict[str, Any]]:
        for job in self.active_jobs:
            if job.get("job_id", job.get("id")) == job_id:
                return list(job.get("current_ladder") or job.get("active_chains") or [])
        return []


@dataclass(frozen=True)
class AllocationUnit:
    """What one rank replica reserves and pays for."""

    env_key: str
    allocation_kind: str
    instance_type: str | None
    gpu_type: str | None
    gpus_per_unit: int
    price_per_unit_hour: float | None = None


class ResourceMapManager:
    """Read PostgreSQL inputs and retain one planning view per tick."""

    def __init__(
        self,
        user_id: str | None = None,
        postgres_client=None,
        # Inject the runner-owned pool for direct PostgreSQL reads.
        *,
        pool=None,
    ):
        self.user_id = user_id
        self._postgres_client = postgres_client
        self._pool = pool
        self._tick_snapshot: ClusterResourceSnapshot | None = None

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

    def hardware_catalog(self) -> dict[str, Any]:
        """Read provider catalogs in deployment X's region schema."""
        if self._pool is None:
            raise ValueError("database pool is required for catalog reads")
        with self._pool.connection() as conn:
            catalogs = [
                row["catalog"]
                for row in conn.execute("SELECT catalog FROM hardware_catalog ORDER BY cloud_name")
            ]
        catalog = merge_hardware_catalogs(catalogs)
        if not catalog["regions"]:
            raise ValueError("hardware catalog is required for deployment X")
        return catalog

    def model_catalog(self, model_id: str, engine_name: str) -> dict[str, Any]:
        """Read model architecture and the selected engine's defaults, raising when absent."""
        if self._pool is None:
            raise ValueError("database pool is required for catalog reads")
        with self._pool.connection() as conn:
            value = conn.execute(
                """
                SELECT * FROM model_architecture
                JOIN job_engine_defaults USING (model_id)
                WHERE model_id = %s AND engine_name = %s
                """,
                (model_id, engine_name),
            ).fetchone()
        if value is None:
            raise ValueError(f"model catalog missing {model_id!r} with engine {engine_name!r}")
        # ponytail: local catalog-read alias; centralize GPU identity if aliases expand.
        for entries in value.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and "gpu_type" in entry:
                    name = str(entry["gpu_type"] or "").upper().replace("_", "-")
                    entry["gpu_type"] = {
                        "RTX-PRO-6000": "RTXPRO6000",
                        "RTX-PRO-6000-BSE": "RTXPRO6000",
                        "RTX PRO 6000": "RTXPRO6000",
                    }.get(name, name)
                    # TODO: We need to coalese all aliasing in one dictionary somewhere
        return value

    # ------------------------------------------------------------------
    # Live-state reconciliation (Layer 2): reserve LIVE replicas, not the
    # committed n_replicas. The store tracks status per rank_id, not per
    # replica, so a rank whose second replica never materialized stays
    # RUNNING and koi reserved a whole box that sits physically empty
    # (real100gpubenchmark run-17: one H100 and one A100-80 held by dead
    # replicas of 8YFER0 / 9R10Y5, starving GLM/phi). gpu_metrics carry
    # (rank_id, chain_index), so per-replica liveness IS observable.
    # ------------------------------------------------------------------
    _REPLICA_LIVENESS_WINDOW_SECONDS = 2 * DEFAULT_TICK_INTERVAL_SECONDS
    _REPLICA_GRACE = timedelta(seconds=2 * DEFAULT_TICK_INTERVAL_SECONDS)

    def observed_replica_liveness(
        self,
        user_id: str | None,
        job_ids,
        window_seconds: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Per-rank live replicas + mean SM utilization from recent telemetry.

        Returns ``{rank_id: {"live_indexes": set[int], "avg_util": float|None}}``
        over the last ``window_seconds`` (default two ticks). A replica counts
        as LIVE if it emitted any gpu_metric row in the window - liveness, not
        activity - so an idle-but-alive low-traffic replica is never mistaken
        for dead. Fully defensive: any store failure yields ``{}`` so callers
        fall back to trusting the committed plan (keeps sim inert).
        """
        result: dict[str, dict[str, Any]] = {}
        ids = {str(j) for j in (job_ids or []) if j}
        if not ids:
            return result
        try:
            from tandemn_system_data.clients import GpuMetricStore

            store = GpuMetricStore(self._client())
            effective_user = self._effective_user_id(user_id)
        except Exception:
            return result
        seconds = int(window_seconds or self._REPLICA_LIVENESS_WINDOW_SECONDS)
        end = datetime.now(UTC)
        start = end - timedelta(seconds=seconds)
        for job_id in ids:
            try:
                rows = store.rows_for_job_window(effective_user, job_id, start, end)
            except Exception:
                continue
            for row in rows or []:
                rank_id = getattr(row, "rank_id", None)
                chain_index = getattr(row, "chain_index", None)
                if rank_id is None or chain_index is None:
                    continue
                if isinstance(chain_index, bool) or not isinstance(chain_index, int):
                    continue
                entry = result.setdefault(
                    str(rank_id),
                    {"live_indexes": set(), "_util_sum": 0.0, "_util_n": 0},
                )
                entry["live_indexes"].add(int(chain_index))
                util = getattr(row, "sm_utilization", None)
                if isinstance(util, (int, float)) and not isinstance(util, bool):
                    entry["_util_sum"] += float(util)
                    entry["_util_n"] += 1
        for entry in result.values():
            n = entry.pop("_util_n", 0)
            total = entry.pop("_util_sum", 0.0)
            entry["avg_util"] = (total / n) if n else None
        return result

    @staticmethod
    def _as_utc(value) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return None

    def _phantom_chain_ids(self, chains, user_id: str | None) -> set[str]:
        """Chain ids whose replica is DEAD - reserved but physically absent.

        A chain is reaped only when its rank has at least one live replica
        (positive-contrast: the telemetry channel works for this rank), the
        rank is older than the 2-tick launch grace, and this replica emitted
        nothing in the window. Never reaps a whole silent rank (fail-safe for
        sim, monitoring gaps, and cold starts - a fully-dead rank is handled
        by the store's active-status filter instead).
        """
        by_rank: dict[str, list[dict[str, Any]]] = {}
        job_ids: set[str] = set()
        for chain in chains:
            shape = chain.get("shape_json") or {}
            rank_id = shape.get("rank_id")
            if not rank_id:
                continue
            by_rank.setdefault(str(rank_id), []).append(chain)
            if chain.get("job_id"):
                job_ids.add(str(chain.get("job_id")))
        if not by_rank:
            return set()
        live = self.observed_replica_liveness(user_id, job_ids)
        if not live:
            return set()
        now = datetime.now(UTC)
        reap: set[str] = set()
        for rank_id, members in by_rank.items():
            live_indexes = (live.get(rank_id) or {}).get("live_indexes") or set()
            if not live_indexes:
                continue  # fail-safe: no telemetry for this rank -> trust commit
            for chain in members:
                created = self._as_utc(chain.get("rank_created_at"))
                if created is not None and (now - created) < self._REPLICA_GRACE:
                    continue  # launch grace: young rank, replicas may be cold-starting
                index = chain.get("chain_index")
                if index is None:
                    tail = str(chain.get("chain_id") or "").rsplit("_chain_", 1)
                    index = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None
                if index is None:
                    continue
                if int(index) not in live_indexes:
                    chain_id = chain.get("chain_id")
                    if chain_id:
                        reap.add(str(chain_id))
        return reap

    # ------------------------------------------------------------------
    # Jobs and chains
    # ------------------------------------------------------------------

    def get_submitted_jobs(self) -> list[dict[str, Any]]:
        """Return queued jobs waiting for admission."""
        return self.get_waiting_jobs()

    def get_running_jobs(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return observed jobs enriched with the tick's SQL context."""
        return self.snapshot().active_jobs_summary()

    def get_waiting_jobs(self) -> list[dict[str, Any]]:
        """Read queued jobs and their intent for Koi's planning view."""
        if self._tick_snapshot is not None:
            return self._tick_snapshot.pending_jobs_summary()
        if self._pool is None:
            raise ValueError("database pool is required for waiting-job reads")
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = %s
                ORDER BY job_id
                """,
                ("queued",),
            ).fetchall()
        return [self._job_context_to_summary(row) for row in rows]

    def get_running_chains(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return active chains with owning job context."""
        chains: list[dict[str, Any]] = []
        for job in self.get_running_jobs(user_id=user_id):
            for chain in job.get("active_chains", []):
                chains.append({"job_id": job["job_id"], **chain})
        return chains

    @staticmethod
    def _job_context_to_summary(row: dict[str, Any]) -> dict[str, Any]:
        features = row.copy()
        job_id = features.pop("job_id")
        kind = features.pop("kind")
        status = features.pop("status")
        return {
            "job_id": job_id,
            "kind": kind,
            "status": status,
            "job_features": {"type": kind, **features},
        }

    # ------------------------------------------------------------------
    # Koi-facing snapshot API
    # ------------------------------------------------------------------

    def read_resource_map_snapshot(self, tick: int) -> dict[str, Any]:
        """Read this exact tick's GlobalResourceMap encoded with proto field names."""
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        if self._pool is None:
            raise ValueError("database pool is required for Resource Map snapshot reads")
        with self._pool.connection() as conn:
            snapshot = conn.execute(
                """
                SELECT * FROM resource_map_snapshots WHERE tick = %s
                """,
                (tick,),
            ).fetchone()
            if snapshot is None:
                raise ValueError(f"Resource Map snapshot missing for tick {tick}")
            resource_map = snapshot["resource_map"]
            cluster_maps = resource_map.get("cluster_resource_maps", [])
            vector_clock = resource_map.get("vector_clock", {})
            invalid_map = f"Invalid GlobalResourceMap in snapshot {snapshot['snapshot_id']!r}"
            if (
                any(key not in {"cluster_resource_maps", "vector_clock"} for key in resource_map)
                or not isinstance(cluster_maps, list)
                or not isinstance(vector_clock, dict)
            ):
                raise ValueError(invalid_map)
            # Pop from a copy to catch duplicate IDs without changing the received clock.
            remaining_clock = vector_clock.copy()
            for cluster_map in cluster_maps:
                cluster = cluster_map.get("cluster") if isinstance(cluster_map, dict) else None
                cluster_id = cluster.get("cluster_id") if isinstance(cluster, dict) else None
                if (
                    not isinstance(cluster_id, str)
                    or not cluster_id.strip()
                    or cluster_id not in remaining_clock
                ):
                    raise ValueError(invalid_map)
                counters = (cluster_map.get("logical_counter", 0), remaining_clock.pop(cluster_id))
                if any(
                    isinstance(counter, bool)
                    or not isinstance(counter, (int, str))
                    or not str(counter).isdecimal()
                    or int(counter) >= 2**64
                    for counter in counters
                ) or int(counters[0]) != int(counters[1]):
                    raise ValueError(invalid_map)
            if remaining_clock:
                raise ValueError(invalid_map)
            return snapshot

    def snapshot(self) -> ClusterResourceSnapshot:
        if self._tick_snapshot is None:
            raise ValueError("tick snapshot is required; capture it in S0 first")
        return self._tick_snapshot

    def snapshot_cluster_state(self, tick) -> ClusterResourceSnapshot:
        """Freeze one FSM tick into the in-memory planning view used for the rest of the tick.

        Load the exact ``resource_map_snapshots`` row for ``tick``, validating
        its GlobalResourceMap membership and vector clock. Never fall back to
        another tick's map. Preserve each cluster's resource records, collection
        time, and logical counter; the global map contains independently collected
        cluster observations, not an atomic physical-world snapshot.

        Read intent for every observed job and every queued job from SQL ``jobs``.
        Group observations across clusters under their owning job,
        validate observed job/rank/chain identities, and attach ranks and chains
        with their runtime information and pod requirements. Cluster-local phases
        stay observational: ``jobs.status`` decides ``active_jobs`` (running)
        versus ``pending_jobs`` (queued).

        Enrich explicitly failed ranks with matching saved placements from earlier
        plan ticks, supplying configuration for failure annotation. A placement
        does not establish failure or deployment status. Failure retention follows
        the received map; this method does not reconstruct a failure-history window.

        Cache the completed view on ``self._tick_snapshot`` and return the same
        object when called again for the same tick, without rereading its inputs.
        A different tick clears the cache before rebuilding, so a load or validation
        failure cannot leave an older tick's view active.

        Args:
            tick: Nonnegative FSM tick whose complete Resource Map row must
                already exist in PostgreSQL.

        Returns:
            ClusterResourceSnapshot containing per-cluster maps, enriched
            active/pending jobs, and the raw stored snapshot envelope.

        Raises:
            ValueError: If the tick is invalid, the database pool is missing,
                the snapshot is missing or fails validation, observed identities
                are malformed or duplicated within their scope, or an observed
                job has no SQL job/intent context. Database errors propagate.
        """
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        if self._tick_snapshot is not None and self._tick_snapshot.tick == tick:
            return self._tick_snapshot
        self._tick_snapshot = None
        stored = deepcopy(self.read_resource_map_snapshot(tick))
        cluster_maps = stored["resource_map"].get("cluster_resource_maps", [])
        observations = self._group_job_observations(cluster_maps)
        contexts = self._read_job_contexts(list(observations))
        self._attach_observed_work(contexts, observations)
        self._enrich_failed_rank_placements(contexts, tick)
        # Cluster-local phases are observations; jobs.status remains authoritative.
        snapshot = ClusterResourceSnapshot(
            tick=tick,
            resources={
                cluster_map["cluster"]["cluster_id"]: cluster_map for cluster_map in cluster_maps
            },
            active_jobs=[job for job in contexts.values() if job["status"] == "running"],
            pending_jobs=[job for job in contexts.values() if job["status"] == "queued"],
            resource_map_snapshot=stored,
        )
        self._tick_snapshot = snapshot
        return snapshot

    @staticmethod
    def _group_job_observations(
        cluster_maps: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group cluster observations by job, retaining source metadata."""
        observations: dict[str, list[dict[str, Any]]] = {}
        for cluster_map in cluster_maps:
            cluster_id = cluster_map["cluster"]["cluster_id"]
            jobs = cluster_map.get("jobs", [])
            if not isinstance(jobs, list):
                raise ValueError(f"cluster {cluster_id!r} jobs must be a list")
            seen_jobs = set()
            for observed in jobs:
                job_id = observed.get("job_id") if isinstance(observed, dict) else None
                if not isinstance(job_id, str) or not job_id.strip() or job_id in seen_jobs:
                    raise ValueError(f"cluster {cluster_id!r} has an invalid or duplicate job_id")
                seen_jobs.add(job_id)
                observations.setdefault(job_id, []).append(
                    {
                        **deepcopy(observed),
                        "cluster_id": cluster_id,
                        "collected_at": cluster_map.get("collected_at"),
                        "logical_counter": cluster_map.get("logical_counter", 0),
                    }
                )
        return observations

    def _read_job_contexts(self, observed_job_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Read queued/observed job intent and require context for every observation."""
        with self._pool.connection() as conn:
            contexts = {
                row["job_id"]: self._job_context_to_summary(row)
                for row in conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = %s OR job_id = ANY(%s)
                    ORDER BY job_id
                    """,
                    ("queued", observed_job_ids),
                )
            }
        for job_id in observed_job_ids:
            if job_id not in contexts:
                raise ValueError(f"job context missing {job_id!r}")
        return contexts

    @staticmethod
    def _attach_observed_work(
        contexts: dict[str, dict[str, Any]],
        observations: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Attach validated rank/chain identities and runtime records to each job."""
        for job_id, job in contexts.items():
            job["observations"] = observations.get(job_id, [])
            ranks = []
            chains = []
            for observed in job["observations"]:
                observed_ranks = observed.get("ranks", [])
                if not isinstance(observed_ranks, list):
                    raise ValueError(f"job {job_id!r} ranks must be a list")
                seen_ranks = set()
                for rank in observed_ranks:
                    rank_id = rank.get("rank_id") if isinstance(rank, dict) else None
                    if not isinstance(rank_id, str) or not rank_id.strip() or rank_id in seen_ranks:
                        raise ValueError(f"job {job_id!r} has an invalid or duplicate rank_id")
                    seen_ranks.add(rank_id)
                    ranks.append({**rank, "cluster_id": observed["cluster_id"]})
                    observed_chains = rank.get("chains", [])
                    if not isinstance(observed_chains, list):
                        raise ValueError(f"rank {rank_id!r} chains must be a list")
                    seen_chains = set()
                    for chain in observed_chains:
                        chain_id = chain.get("chain_id") if isinstance(chain, dict) else None
                        if (
                            not isinstance(chain_id, str)
                            or not chain_id.strip()
                            or chain_id in seen_chains
                        ):
                            raise ValueError(
                                f"rank {rank_id!r} has an invalid or duplicate chain_id"
                            )
                        seen_chains.add(chain_id)
                        chains.append(
                            {
                                **chain,
                                "rank_id": rank_id,
                                "cluster_id": observed["cluster_id"],
                                "runtime": rank.get("runtime", {}),
                                "pod_requirements": rank.get("pod_requirements", []),
                            }
                        )
            job["ranks"] = ranks
            job["active_chains"] = chains
            job["current_ladder"] = chains

    def _enrich_failed_rank_placements(
        self, contexts: dict[str, dict[str, Any]], tick: int
    ) -> None:
        """Attach prior saved settings to failures present in this tick's map."""
        # shortcut: failure retention follows the received map; define a historical lookback separately.
        failed_ranks = {}
        for job_id, job in contexts.items():
            for rank in job["ranks"]:
                status = rank.get("status")
                if isinstance(status, dict) and status.get("phase") == "WORKLOAD_PHASE_FAILED":
                    failed_ranks[(job_id, rank["rank_id"], rank["cluster_id"])] = rank
        if failed_ranks:
            with self._pool.connection() as conn:
                placements = conn.execute(
                    """
                    SELECT DISTINCT ON (p.job_id, p.rank_id, p.cluster_id)
                        p.job_id, p.rank_id, p.cluster_id, p.plan_id,
                        p.accelerator_type, p.tensor_parallelism, p.pipeline_parallelism
                    FROM placement AS p JOIN plans USING (plan_id)
                    WHERE p.rank_id = ANY(%s) AND plans.tick < %s
                    ORDER BY p.job_id, p.rank_id, p.cluster_id, plans.tick DESC, p.plan_id DESC
                    """,
                    ([rank_id for _, rank_id, _ in failed_ranks], tick),
                ).fetchall()
            for placement in placements:
                key = (placement["job_id"], placement["rank_id"], placement["cluster_id"])
                if key in failed_ranks:
                    failed_ranks[key]["placement"] = placement

    def resources_summary(self, user_id: str | None = None) -> dict[str, Any]:
        return self.snapshot().resources_summary()

    @classmethod
    def _normalized_scheduling_summary(
        cls,
        resource_map,
        used_gpus: dict[str, int] | None = None,
        used_instances: dict[tuple[str, str], int] | None = None,
        used_pool_gpus: dict[tuple[str, str], int] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Flatten the store ResourceMap to env_key -> capacity info.

        The store map carries total capacity only; ``free`` is derived here
        as ``total`` minus GPUs consumed by running chains (``used``).
        """
        used_gpus = used_gpus or {}
        used_instances = used_instances or {}
        used_pool_gpus = used_pool_gpus or {}
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
            pools_by_instance: dict[str, dict[str, Any]] = {}
            pool_shapes: dict[str, tuple[str, int]] = {}
            for raw_pool in body.get("pools") or []:
                pool = dict(raw_pool)
                instance_type = str(pool.get("instance_type"))
                kind = str(
                    pool.get("allocation_kind") or pool.get("allocation_unit") or "instance"
                ).lower()
                gpus_per_instance = int(pool.get("gpus_per_instance", 1))
                total_instances = int(pool.get("total_instances", 0))
                raw_total = pool.get("total")
                total = (
                    int(raw_total) if raw_total is not None else total_instances * gpus_per_instance
                )
                aggregate = pools_by_instance.get(instance_type)
                if aggregate is None:
                    pool["total_instances"] = total_instances
                    pool["total"] = total
                    pool["merged_pool_count"] = 1
                    pool["merged_pool_sources"] = [
                        {
                            "pool_id": pool.get("pool_id"),
                            "fabric_id": pool.get("fabric_id", pool.get("fabric")),
                            "fabric_type": pool.get("fabric_type"),
                            "price_per_instance_hour": pool.get("price_per_instance_hour"),
                            "total_instances": total_instances,
                            "total": total,
                        }
                    ]
                    pools_by_instance[instance_type] = pool
                    pool_shapes[instance_type] = (kind, gpus_per_instance)
                    continue

                first_kind, first_gpus_per_instance = pool_shapes[instance_type]
                if kind != first_kind:
                    raise ValueError(
                        f"duplicate pool {instance_type!r} in env {env_key!r} has conflicting "
                        f"allocation kind: {first_kind!r} != {kind!r}"
                    )
                if gpus_per_instance != first_gpus_per_instance:
                    raise ValueError(
                        f"duplicate pool {instance_type!r} in env {env_key!r} has conflicting "
                        "gpus_per_instance: "
                        f"{first_gpus_per_instance} != {gpus_per_instance}"
                    )
                aggregate["total_instances"] += total_instances
                aggregate["total"] += total
                aggregate["merged_pool_count"] += 1
                aggregate["merged_pool_sources"].append(
                    {
                        "pool_id": pool.get("pool_id"),
                        "fabric_id": pool.get("fabric_id", pool.get("fabric")),
                        "fabric_type": pool.get("fabric_type"),
                        "price_per_instance_hour": pool.get("price_per_instance_hour"),
                        "total_instances": total_instances,
                        "total": total,
                    }
                )

            pools = []
            for instance_type, pool in pools_by_instance.items():
                kind, gpus_per_instance = pool_shapes[instance_type]
                total_instances = int(pool["total_instances"])
                if kind == "gpu":
                    free_instances = total_instances
                    free_gpus = max(
                        0,
                        int(pool["total"]) - used_pool_gpus.get((env_key, instance_type), 0),
                    )
                else:
                    free_instances = max(
                        0,
                        total_instances - used_instances.get((env_key, instance_type), 0),
                    )
                    free_gpus = free_instances * gpus_per_instance
                pool["free_instances"] = free_instances
                pool["free"] = free_gpus
                pools.append(pool)
            body["pools"] = pools
            body["free"] = (
                sum(int(pool["free"]) for pool in pools)
                if pools
                else max(0, int(body.get("total", 0)) - used_gpus.get(env_key, 0))
            )
            out[env_key] = body
        return out

    def _used_capacity(
        self, resource_map, user_id: str | None = None
    ) -> tuple[dict[str, int], dict[tuple[str, str], int], dict[tuple[str, str], int]]:
        """Return used GPUs by env and used instances by env/pool.

        Free capacity is not stored on the resource map (total-only); it is
        inferred by subtracting this from each env's total. One chain row is
        one launched serving unit. GPU-granular pools consume
        ``shape_json["count"]``; instance-atomic pools consume the full
        instance capacity that row reserved.

        The chain's placement env is resolved with precedence
        ``target_node`` -> ``shape_json["env"]`` -> ``shape_json["pool_id"]``;
        the store writes the env key into the first-class ``target_node``
        field. A 4-part legacy env is normalized to 5 parts with the map's
        default market.
        """
        default_market = self._default_market(resource_map)
        resources = self._normalized_scheduling_summary(resource_map)
        used_gpus: dict[str, int] = {}
        used_instances: dict[tuple[str, str], int] = {}
        used_pool_gpus: dict[tuple[str, str], int] = {}
        running_chains = list(self.get_running_chains(user_id=user_id))
        # Layer 2: drop reservations for dead replicas so a box held by a
        # phantom replica is returned to free capacity.
        phantom_chain_ids = self._phantom_chain_ids(running_chains, user_id)
        for chain in running_chains:
            if chain.get("chain_id") in phantom_chain_ids:
                continue
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
            unit = self.resolve_allocation_unit(env_key, shape, resources)
            footprint = count if unit.allocation_kind == "gpu" else unit.gpus_per_unit
            used_gpus[env_key] = used_gpus.get(env_key, 0) + footprint
            if unit.allocation_kind == "instance" and unit.instance_type:
                key = (env_key, unit.instance_type)
                used_instances[key] = used_instances.get(key, 0) + 1
            elif unit.instance_type:
                key = (env_key, unit.instance_type)
                used_pool_gpus[key] = used_pool_gpus.get(key, 0) + footprint
        return used_gpus, used_instances, used_pool_gpus

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
        snapshot = self.snapshot()
        stored = snapshot.resource_map_snapshot
        return {
            "tick": snapshot.tick,
            "snapshot_id": stored["snapshot_id"],
            "stored_at": stored["stored_at"],
            "resource_map": stored["resource_map"],
            "resources": snapshot.resources_summary(),
            "running_jobs": self.get_running_jobs(),
            "waiting_jobs": self.get_waiting_jobs(),
            "running_chains": self.get_running_chains(),
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
        return self.snapshot().resource_map_snapshot["resource_map"]

    def refresh_resource_map(self):
        return self.get_resource_map()

    def get_avail_capacity(self, env, gpu_type):
        env_key = self._env_key(env)
        requested_gpu = gpu_type or env_gpu_type(env)
        resources = self.resources_summary()
        info = resources.get(env_key)
        if info is None or (requested_gpu is not None and info.get("gpu_type") != requested_gpu):
            return 0
        return int(info.get("free", 0))

    def resolve_allocation_unit(
        self,
        env,
        config: dict[str, Any],
        resources: dict[str, dict[str, Any]] | None = None,
    ) -> AllocationUnit:
        """Resolve one rank replica to the pool Koi reserves.

        Cloud pools are instance-atomic. A config may use fewer GPUs than the
        instance has, but the full instance capacity and price are reserved.
        Pools marked allocation_kind="gpu" remain discrete-GPU pools.
        """
        resources = resources if resources is not None else self.resources_summary()
        env_key = self._env_key(env)
        info = resources.get(env_key)
        if info is None:
            raise ValueError(f"env {env_key!r} is not in the resource map")

        pool = self._select_pool(env_key, info, config)
        if pool is None:
            return AllocationUnit(env_key, "gpu", None, info.get("gpu_type"), 1, None)

        kind = str(pool.get("allocation_kind") or pool.get("allocation_unit") or "instance")
        kind = kind.lower()
        instance_type = pool.get("instance_type")
        gpu_type = pool.get("gpu_type") or info.get("gpu_type")
        if kind == "gpu":
            price = pool.get("price_per_gpu_hour") or pool.get("price_per_unit_hour")
            return AllocationUnit(env_key, "gpu", instance_type, gpu_type, 1, _float_or_none(price))

        gpus = int(pool.get("gpus_per_instance") or pool.get("gpus_per_unit") or 1)
        price = pool.get("price_per_instance_hour") or pool.get("price_per_unit_hour")
        return AllocationUnit(
            env_key, "instance", instance_type, gpu_type, gpus, _float_or_none(price)
        )

    def rank_capacity_per_replica(
        self,
        rank,
        resources: dict[str, dict[str, Any]] | None = None,
    ) -> int:
        return int(self.rank_allocation_summary(rank, resources)["capacity_per_replica"])

    def rank_capacity_footprint(
        self,
        rank,
        resources: dict[str, dict[str, Any]] | None = None,
    ) -> int:
        return int(rank.n_replicas) * self.rank_capacity_per_replica(rank, resources)

    def rank_allocation_summary(
        self,
        rank,
        resources: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return engine demand plus reserved capacity for one rank replica."""
        resources = resources if resources is not None else self.resources_summary()
        engine_gpus = rank.gpus_per_chain()
        unit = self.resolve_allocation_unit(rank.env, rank.config, resources)
        capacity = engine_gpus if unit.allocation_kind == "gpu" else unit.gpus_per_unit
        info = resources[unit.env_key]
        pool = self._select_pool(unit.env_key, info, rank.config)
        return {
            "allocation_kind": unit.allocation_kind,
            "instance_type": unit.instance_type,
            "gpus_per_unit": unit.gpus_per_unit,
            "price_per_unit_hour": unit.price_per_unit_hour,
            "capacity_per_replica": capacity,
            "free_capacity_gpus": int((pool or info).get("free", 0)),
            "engine_gpus": engine_gpus,
        }

    def pool_capacity(
        self,
        resources: dict[str, dict[str, Any]] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Return available allocation units for each instance pool."""
        resources = resources if resources is not None else self.resources_summary()
        capacity = {}
        for env, info in resources.items():
            for pool in info.get("pools") or []:
                instance_type = pool.get("instance_type")
                if not instance_type:
                    continue
                kind = str(
                    pool.get("allocation_kind") or pool.get("allocation_unit") or "instance"
                ).lower()
                gpus_per_unit = (
                    1
                    if kind == "gpu"
                    else int(pool.get("gpus_per_instance") or pool.get("gpus_per_unit") or 1)
                )
                free_gpus = int(pool.get("free", 0))
                capacity[(env, str(instance_type))] = {
                    "allocation_kind": kind,
                    "available_units": free_gpus // gpus_per_unit,
                    "gpus_per_unit": gpus_per_unit,
                    "free_gpus": free_gpus,
                }
        return capacity

    def requested_capacity(
        self,
        plan,
        resources: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, int], dict[tuple[str, str], dict[str, int]]]:
        """Return a plan's reserved GPUs by env and allocation units by pool."""
        resources = resources if resources is not None else self.resources_summary()
        typed = plan if isinstance(plan, Plan) else Plan.from_raw(plan, tick=0)
        by_env: dict[str, int] = {}
        by_pool: dict[tuple[str, str], dict[str, int]] = {}
        for action in typed.actions:
            if action.type not in LADDER_ACTIONS:
                continue
            for rank in action.ladder or []:
                allocation = self.rank_allocation_summary(rank, resources)
                env = self._env_key(rank.env)
                gpus = int(rank.n_replicas) * int(allocation["capacity_per_replica"])
                by_env[env] = by_env.get(env, 0) + gpus

                instance_type = allocation.get("instance_type")
                if instance_type is None:
                    continue
                key = (env, str(instance_type))
                requested = by_pool.setdefault(key, {"units": 0, "gpus": 0})
                requested["units"] += (
                    gpus if allocation.get("allocation_kind") == "gpu" else int(rank.n_replicas)
                )
                requested["gpus"] += gpus
        return by_env, by_pool

    def switch_pricing_map(self, resources: dict[str, dict[str, Any]] | None = None) -> dict:
        resources = resources if resources is not None else self.resources_summary()
        pricing: dict[str, dict[str, Any]] = {}
        for env, info in resources.items():
            by_instance = {}
            prices = []
            for pool in info.get("pools") or []:
                inst = pool.get("instance_type")
                price = pool.get("price_per_instance_hour") or pool.get("price_per_unit_hour")
                if inst and price is not None:
                    by_instance[str(inst)] = float(price)
                    prices.append(float(price))
            if by_instance:
                pricing[env] = {"by_instance_type": by_instance, "default": max(prices)}
        return pricing

    def check_resource_feasibility(self, plan):
        future = self.simulate_future_resources(plan)
        violations = []
        pool_failed_envs = set()
        for env, info in future.items():
            for pool in info.get("pools") or []:
                if pool.get("free_units_after", 0) >= 0:
                    continue
                pool_failed_envs.add(env)
                unit = "GPUs" if pool.get("allocation_kind") == "gpu" else "instances"
                violations.append(
                    f"env {env} pool {pool.get('instance_type')}: requested "
                    f"{pool.get('requested_units', 0)} {unit}, only "
                    f"{pool.get('free_units_now', 0)} free"
                )
        violations.extend(
            f"env {env}: requested {-info['free_after']} more GPUs than available"
            for env, info in future.items()
            if info["free_after"] < 0 and env not in pool_failed_envs
        )
        return len(violations) == 0, violations

    def simulate_future_resources(self, plan):
        resources = self.resources_summary()
        requested, requested_by_pool = self.requested_capacity(plan, resources)
        pool_capacity = self.pool_capacity(resources)
        out = {}
        for env, info in resources.items():
            free_now = int(info.get("free", 0))
            delta = int(requested.get(env, 0))
            pools = []
            for raw_pool in info.get("pools") or []:
                pool = dict(raw_pool)
                key = (env, str(pool.get("instance_type")))
                limit = pool_capacity.get(key, {})
                demand = requested_by_pool.get(key, {})
                pool["free_units_now"] = int(limit.get("available_units", 0))
                pool["requested_units"] = int(demand.get("units", 0))
                pool["free_units_after"] = pool["free_units_now"] - pool["requested_units"]
                pool["free_after"] = int(pool.get("free", 0)) - int(demand.get("gpus", 0))
                pools.append(pool)
            out[env] = {
                **info,
                "pools": pools,
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

    def _requested_gpus_by_env(
        self,
        plan,
        resources: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        """GPUs requested per env by a Plan's ladder-bearing actions.

        Accepts a typed Plan or any raw form Plan.from_raw accepts. Counts
        each rank's full GPU footprint (n_replicas * tp * pp), not just its
        replica count.
        """
        return self.requested_capacity(plan, resources)[0]

    @staticmethod
    def _select_pool(env_key: str, info: dict[str, Any], config: dict[str, Any]) -> dict | None:
        pools = list(info.get("pools") or [])
        if not pools:
            return None
        instance_type = config.get("instance_type")
        if instance_type:
            for pool in pools:
                if pool.get("instance_type") == instance_type:
                    return pool
            raise ValueError(f"instance_type {instance_type!r} is not available in env {env_key}")
        if len(pools) == 1:
            return pools[0]
        choices = ", ".join(str(p.get("instance_type")) for p in pools)
        raise ValueError(f"env {env_key} has multiple pools; choose instance_type: {choices}")

    @staticmethod
    def _env_key(env) -> str:
        if isinstance(env, (tuple, list)):
            return "|".join(str(part) for part in env)
        return str(env)


def _float_or_none(value) -> float | None:
    return None if value is None else float(value)
