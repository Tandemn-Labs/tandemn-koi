"""Store-backed runtime telemetry for Koi's S1 -> S2 evidence path.

Telemetry owns observed runtime V/Y only. Deployment-time X is built from the
frozen Store/catalog snapshot in ``deployment_x.py``; W is intentionally absent
because workload context now lives in X or measured V.

One collection pass is deliberately small:

    1. S1 passes the frozen cluster snapshot into ``collect_telemetry``.
    2. The adapter reads raw ``GpuMetric`` rows from Tandemn Store per active job.
    3. Rows are validated against active snapshot chains by ``chain_id`` and
       ``rank_id``.
    4. GPU/worker rows collapse into rank-level V/Y trajectories.
    5. ``iter_per_rank`` yields ``RankTelemetry`` objects consumed by S2.

Store remains append-only/raw-ish. Koi owns rank aggregation because evidence
semantics live beside CUSUM/ICP and the candidate graph projection.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

DEFAULT_SAMPLE_BUCKET_SEC = 10

DEFAULT_POLICY = "latest_mean"
METRIC_POLICIES = {
    "throughput_token_per_sec": "sum",
    "live_batch_size": "sum",
    "depth_req_q": "sum",
    "prefill_iteration_counts_per_second": "sum",
    "decode_itr_counts_per_second": "sum",
    "p99_ttft_ms": "max",
    "p99_tpot_ms": "max",
    "slo_margin": "min",
    "cost_per_token": "weighted_by_throughput",
    "kv_cache_util": "mean",
    "activation_mem_pressure": "mean",
    "gpu_mem_used_fraction": "mean",
    "vram_headroom_gb": "mean",
    "nvlink_tput_observed": "mean",
    "pcie_tput_observed": "mean",
    "pd_inbalance": "mean",
    "expert_inbalance": "mean",
    "comm_overhead_pct": "mean",
    "pipeline_bubble_fraction": "mean",
    "per_tok_comm_bytes": "mean",
    "kv_pressure_score": "mean",
    "dispatch_overhead_ms": "mean",
    "input_length_observed": "mean",
    "output_length_observed": "mean",
    "kvcache_hit_rate": "mean",
}

RankKey = tuple[str, str]
Bucket = int
ChainID = str
RowsByRank = dict[RankKey, dict[Bucket, dict[ChainID, list[Any]]]]


@dataclass(frozen=True)
class RankTelemetry:
    """S2-facing telemetry bundle for one deployed rank.

    ``v_observed`` and ``y_observed`` are sub-tick trajectories keyed by
    candidate-graph variable name. Predictions are supplied by deployment
    state, not telemetry; this adapter leaves them empty.
    """

    job_id: str
    rank_id: str
    v_observed: dict[str, np.ndarray]
    v_predicted: dict[str, float]
    y_observed: dict[str, np.ndarray]
    y_predicted: dict[str, float]
    committed_mechanism_id: str | None = None
    deploy_timestamp_utc: float | None = None


@dataclass(frozen=True)
class _ChainRef:
    """Snapshot identity for one active chain.

    Raw Store rows are accepted only when their ``chain_id`` and ``rank_id``
    match this frozen S0 view.
    """

    job_id: str
    chain_id: str
    rank_id: str
    shape: dict[str, Any]
    job_features: dict[str, Any]


@dataclass(frozen=True)
class _RankRef:
    """Snapshot identity for one rank and all active chain replicas in it."""

    job_id: str
    rank_id: str
    chains: list[_ChainRef]
    job_features: dict[str, Any]


@dataclass(frozen=True)
class _TelemetryBundle:
    """Raw Store rows plus the snapshot identities needed to aggregate them."""

    start: datetime
    end: datetime
    rows_by_job: dict[str, list[Any]]
    chains_by_id: dict[str, _ChainRef]
    ranks: list[_RankRef]


class StoreTelemetry:
    """Read Store GPU metrics and yield Koi rank-level V/Y telemetry.

    The adapter is strict about rank identity because S2 resolves deployment X
    by ``(job_id, rank_id)``. Missing telemetry rows skip a rank for this tick;
    conflicting rank identity raises instead of manufacturing evidence.
    """

    def __init__(
        self,
        *,
        user_id: str,
        gpu_metric_store=None,
        candidate_graph=None,
        tick_interval_sec: int = 300,
        sample_bucket_sec: int = DEFAULT_SAMPLE_BUCKET_SEC,
        now_fn=None,
    ) -> None:
        """Create a Store-backed telemetry adapter.

        Args:
            user_id: Tandemn Store tenant/user scope for telemetry reads.
            gpu_metric_store: Store client exposing ``rows_for_job_window``.
                Defaults to a real ``GpuMetricStore(PostgresClient())``.
            candidate_graph: Runtime graph; its ``v`` and ``y`` lists decide
                which metric names become evidence variables.
            tick_interval_sec: First-read window size before ``_last_end`` is
                known.
            sample_bucket_sec: Wall-clock grouping width for one collector
                sample. Orca currently writes roughly every 10 seconds.
            now_fn: Test seam for deterministic wall-clock windows.
        """

        self.user_id = user_id
        self.store = gpu_metric_store or self._default_store()
        self.candidate_graph = candidate_graph
        self.tick_interval_sec = int(tick_interval_sec)
        self.sample_bucket_sec = int(sample_bucket_sec)
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self._last_end: datetime | None = None

    @staticmethod
    def _default_store():
        from tandemn_system_data.clients import (  # type: ignore[import-untyped]
            GpuMetricStore,
            PostgresClient,
        )

        return GpuMetricStore(PostgresClient())

    def collect_telemetry(self, tick_start, tick_end, snapshot) -> _TelemetryBundle:
        """Fetch raw Store rows for active jobs in this wall-clock tick window.

        ``tick_start`` and ``tick_end`` are Koi integer ticks, while Store
        telemetry is timestamped. This method therefore maintains its own
        monotonic wall-clock window and uses the provided snapshot only for job
        and chain identity.
        """

        end = self.now_fn()
        start = self._last_end or end - timedelta(seconds=self.tick_interval_sec)
        self._last_end = end

        chains_by_id, ranks = self._active_refs(snapshot)
        rows_by_job = {
            job_id: self.store.rows_for_job_window(self.user_id, job_id, start, end)
            for job_id in sorted({rank.job_id for rank in ranks})
        }
        return _TelemetryBundle(start, end, rows_by_job, chains_by_id, ranks)

    def iter_per_rank(self, bundle: _TelemetryBundle):
        """Yield one ``RankTelemetry`` per active rank with observed rows.

        Ranks with no Store rows in the window are skipped. Metrics not present
        in the candidate graph are ignored before S2 sees them.
        """

        rows_by_rank = self._rows_by_rank(bundle)
        v_names = set(getattr(self.candidate_graph, "v", []) or [])
        y_names = set(getattr(self.candidate_graph, "y", []) or [])

        for rank in bundle.ranks:
            bucket_rows = rows_by_rank.get((rank.job_id, rank.rank_id), {})
            if not bucket_rows:
                continue
            observed = self._rank_observed(bucket_rows, v_names | y_names)
            v_observed = {name: values for name, values in observed.items() if name in v_names}
            y_observed = {name: values for name, values in observed.items() if name in y_names}
            if not v_observed and not y_observed:
                continue
            yield RankTelemetry(
                job_id=rank.job_id,
                rank_id=rank.rank_id,
                v_observed=v_observed,
                v_predicted={},
                y_observed=y_observed,
                y_predicted={},
                committed_mechanism_id=self._committed_mechanism_id(rank),
            )

    @staticmethod
    def _active_jobs(snapshot) -> list[dict[str, Any]]:
        if hasattr(snapshot, "active_jobs_summary"):
            return list(snapshot.active_jobs_summary() or [])
        return list(getattr(snapshot, "active_jobs", []) or [])

    def _active_refs(self, snapshot) -> tuple[dict[str, _ChainRef], list[_RankRef]]:
        """Return active chain/rank identities from the frozen snapshot."""

        chains_by_id: dict[str, _ChainRef] = {}
        rank_groups: dict[tuple[str, str], list[_ChainRef]] = defaultdict(list)
        for job in self._active_jobs(snapshot):
            job_id = str(job["job_id"])
            job_features = dict(job.get("job_features") or {})
            for chain in job.get("active_chains") or job.get("current_ladder") or []:
                chain_id = chain.get("chain_id")
                shape = dict(chain.get("shape_json") or {})
                rank_id = shape.get("rank_id")
                if not chain_id:
                    raise ValueError(f"active chain for job {job_id!r} missing chain_id")
                if not rank_id:
                    raise ValueError(f"chain {chain_id!r} missing rank_id")
                ref = _ChainRef(job_id, str(chain_id), str(rank_id), shape, job_features)
                chains_by_id[ref.chain_id] = ref
                rank_groups[(ref.job_id, ref.rank_id)].append(ref)
        ranks = [
            _RankRef(job_id, rank_id, chains, chains[0].job_features)
            for (job_id, rank_id), chains in sorted(rank_groups.items())
        ]
        return chains_by_id, ranks

    def _rows_by_rank(self, bundle: _TelemetryBundle) -> RowsByRank:
        """Validate raw rows and group them by rank, time bucket, and chain."""

        out: RowsByRank = {}
        for ref, row in self._validated_rows(bundle):
            rank_rows = out.setdefault((ref.job_id, ref.rank_id), {})
            bucket_rows = rank_rows.setdefault(self._bucket(row), {})
            bucket_rows.setdefault(ref.chain_id, []).append(row)
        return out

    def _validated_rows(self, bundle: _TelemetryBundle):
        """Yield Store rows that belong to active snapshot chains."""

        for rows in bundle.rows_by_job.values():
            for row in rows:
                ref = self._chain_ref_for_row(bundle, row)
                if ref is not None:
                    yield ref, row

    @staticmethod
    def _chain_ref_for_row(bundle: _TelemetryBundle, row) -> _ChainRef | None:
        chain_id = getattr(row, "chain_id", None)
        if chain_id is None:
            return None
        ref = bundle.chains_by_id.get(str(chain_id))
        if ref is None:
            return None

        row_rank_id = getattr(row, "rank_id", None)
        if row_rank_id is None:
            raise ValueError(f"metric row for chain {chain_id!r} missing rank_id")
        if str(row_rank_id) != ref.rank_id:
            raise ValueError(
                f"metric row for chain {chain_id!r} has rank_id {row_rank_id!r}, "
                f"expected {ref.rank_id!r}"
            )
        return ref

    def _bucket(self, row) -> int:
        ts = row.ts
        return int(ts.timestamp() // self.sample_bucket_sec)

    def _rank_observed(
        self,
        bucket_rows: dict[Bucket, dict[ChainID, list[Any]]],
        metric_names: set[str],
    ) -> dict[str, np.ndarray]:
        """Collapse bucketed chain rows into rank-level metric trajectories."""

        trajectories: dict[str, list[float]] = defaultdict(list)
        for bucket in sorted(bucket_rows):
            for metric, value in self._rank_sample(bucket_rows[bucket], metric_names).items():
                trajectories[metric].append(value)
        return {name: np.asarray(values, dtype=float) for name, values in trajectories.items()}

    def _rank_sample(
        self,
        chain_rows: dict[ChainID, list[Any]],
        metric_names: set[str],
    ) -> dict[str, float]:
        """Build one rank sample for one time bucket."""

        throughput = self._chain_values("throughput_token_per_sec", chain_rows)
        sample = {}
        for metric in sorted(metric_names):
            values = self._chain_values(metric, chain_rows)
            if not values:
                continue
            rank_value = self._rank_value(metric, values, throughput)
            if rank_value is not None:
                sample[metric] = rank_value
        return sample

    def _chain_values(
        self,
        metric: str,
        chain_rows: dict[ChainID, list[Any]],
    ) -> dict[ChainID, float]:
        """Collapse every chain's rows for one metric."""

        out = {}
        for chain_id, rows in chain_rows.items():
            value = self._chain_value(metric, rows)
            if value is not None:
                out[chain_id] = value
        return out

    def _chain_value(self, metric: str, rows: list[Any]) -> float | None:
        """Collapse one chain's GPU/worker rows into one bucket value."""

        samples = []
        for row in rows:
            value = _metric_value(row, metric)
            if value is not None:
                samples.append((row.ts, value))
        if not samples:
            return None
        values = [value for _, value in samples]
        if _policy(metric) == "mean":
            return _mean(values)
        return max(samples, key=lambda item: item[0])[1]

    @staticmethod
    def _rank_value(
        metric: str,
        chain_values: dict[str, float],
        chain_throughput: dict[str, float],
    ) -> float | None:
        """Collapse chain bucket values into one rank bucket value."""

        policy = _policy(metric)
        values = list(chain_values.values())
        if policy == "sum":
            return float(sum(values))
        if policy == "max":
            return max(values)
        if policy == "min":
            return min(values)
        if policy == "weighted_by_throughput" and chain_throughput:
            weighted = [
                (value, weight)
                for chain_id, value in chain_values.items()
                if (weight := chain_throughput.get(chain_id)) is not None
            ]
            total = sum(weight for _, weight in weighted)
            if total > 0:
                return sum(value * weight for value, weight in weighted) / total
        return _mean(values)

    @staticmethod
    def _committed_mechanism_id(rank: _RankRef) -> str | None:
        for chain in rank.chains:
            mechanism_id = chain.shape.get("mechanism_id")
            if mechanism_id is not None:
                return str(mechanism_id)
        return None


def _metric_value(row: Any, metric: str) -> float | None:
    """Read one finite metric value from a Store model or raw JSON dict."""

    value = getattr(row, metric, None)
    if value is None:
        metrics_json = getattr(row, "metrics_json", None)
        if isinstance(metrics_json, dict):
            value = metrics_json.get(metric)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _policy(metric: str) -> str:
    return METRIC_POLICIES.get(metric, DEFAULT_POLICY)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


# Telemetry = StoreTelemetry
