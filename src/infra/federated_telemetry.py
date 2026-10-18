"""Collect rank-level observations through source adapters and add node costs."""

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
from src.infra.metrics.base import (
    MetricAdapter,
    RankContext,
    RankObservations,
    adapters_for_rank,
    resolve_rank_contexts,
)
from src.infra.telemetry import RankTelemetry


@dataclass(frozen=True)
class _TelemetryBundle:
    """Keep timestamps, auxiliary features, and query messages alongside V/Y."""

    start: datetime
    end: datetime
    contexts: list[RankContext]
    observations: dict[tuple[str, str], RankObservations]


class FederatedTelemetry:
    """Coordinate preloaded adapters using the existing Koi collection interface."""

    def __init__(
        self,
        *,
        plugins: Mapping[tuple[str, str], MetricAdapter],
        candidate_graph=None,
        tick_interval_sec: int = 300,
        sample_bucket_sec: int = 10,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        for name, value in (
            ("tick_interval_sec", tick_interval_sec),
            ("sample_bucket_sec", sample_bucket_sec),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.plugins = plugins
        self.candidate_graph = candidate_graph
        self.tick_interval_sec = tick_interval_sec
        self.sample_bucket_sec = sample_bucket_sec
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self._last_end: datetime | None = None

    def collect_telemetry(self, tick_start, tick_end, snapshot: Any) -> _TelemetryBundle:
        """Query a wall-clock window; tick numbers are the FSM's logical counters."""
        end = self.now_fn()
        start = self._last_end or end - timedelta(seconds=self.tick_interval_sec)
        if end.utcoffset() is None or end < start:
            raise ValueError("collection time must be timezone-aware and monotonic")
        contexts = resolve_rank_contexts(snapshot)
        observations = {(c.job_id, c.rank_id): RankObservations() for c in contexts}
        groups: dict[tuple[str, str], list[RankContext]] = defaultdict(list)
        for context in contexts:
            for adapter in adapters_for_rank(context, self.plugins):
                groups[(adapter.kind, adapter.name)].append(context)
        hourly_costs = _rank_hourly_costs(contexts)

        for source, ranks in groups.items():
            results = self.plugins[source].collect(
                ranks, start=start, end=end, step_seconds=self.sample_bucket_sec
            )
            requested = {(c.job_id, c.rank_id) for c in ranks}
            for key, result in results.items():
                if key not in requested:
                    raise ValueError(f"{source}: unexpected job/rank {key!r}")
                merged = observations[key]
                for category in ("v", "y", "features"):
                    target = getattr(merged, category)
                    for name, values in getattr(result, category).items():
                        if name in merged.timestamps:
                            raise ValueError(f"duplicate observation {name!r} for {key!r}")
                        times = result.timestamps[name]
                        if self._last_end is not None:
                            # Range queries include both endpoints; do not replay the last sample.
                            fresh = times > start.timestamp()
                            times, values = times[fresh], values[fresh]
                        if len(values):
                            target[name] = values
                            merged.timestamps[name] = times
                merged.warnings += result.warnings
                merged.infos += result.infos

        for key, price in hourly_costs.items():
            observed = observations[key]
            throughput = observed.y.get("throughput_token_per_sec")
            if throughput is None:
                continue
            valid = np.isfinite(throughput) & (throughput > 0)
            with np.errstate(over="ignore"):
                costs = (price / 3600) / throughput[valid]
            finite = np.isfinite(costs)
            if finite.any():
                observed.y["cost_per_token"] = costs[finite]
                observed.timestamps["cost_per_token"] = observed.timestamps[
                    "throughput_token_per_sec"
                ][valid][finite]

        # Advance only after every adapter and derived calculation succeeds, so retries keep the window.
        self._last_end = end
        return _TelemetryBundle(start, end, contexts, observations)

    def iter_per_rank(self, bundle: _TelemetryBundle) -> Iterator[RankTelemetry]:
        """Yield observed graph fields with empty predictions; skip unobserved ranks."""
        graph = self.candidate_graph
        for context in bundle.contexts:
            observed = bundle.observations[(context.job_id, context.rank_id)]
            v = cast(dict[str, np.ndarray], observed.v)
            y = cast(dict[str, np.ndarray], observed.y)
            v_observed = {
                name: values for name, values in v.items() if graph is None or name in graph.v
            }
            y_observed = {
                name: values for name, values in y.items() if graph is None or name in graph.y
            }
            if not v_observed and not y_observed:
                continue
            yield RankTelemetry(
                job_id=context.job_id,
                rank_id=context.rank_id,
                v_observed=v_observed,
                y_observed=y_observed,
                v_predicted={},
                y_predicted={},
                committed_mechanism_id=context.committed_mechanism_id,
                deploy_timestamp_utc=context.deploy_timestamp_utc,
                expected_replicas=context.expected_replicas
                if context.expected_replicas is not None
                else 1,
                # shortcut: rank aggregates lack replica coverage; populate verified counts when available.
                observed_replicas=0,
                throughput_observed_replicas=0,
                health_observed_replicas=0,
            )


def _rank_hourly_costs(contexts: list[RankContext]) -> dict[tuple[str, str], float]:
    """Sum unique node prices; shared nodes need an allocation rule before costing."""
    owners: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    nodes_by_rank = {}
    for context in contexts:
        key = (context.job_id, context.rank_id)
        prices: dict[tuple[str, str], float | None] = {}
        for node in context.nodes:
            node_key = (node.cluster_id, node.node_id)
            if node_key in prices and prices[node_key] != node.price_per_node_hour:
                raise ValueError(f"conflicting node price for {node_key!r} on {key!r}")
            prices[node_key] = node.price_per_node_hour
            owners[node_key].add(key)
        nodes_by_rank[key] = prices
    costs = {}
    for key, nodes in nodes_by_rank.items():
        known_prices = [price for price in nodes.values() if price is not None]
        if nodes and len(known_prices) == len(nodes) and all(len(owners[n]) == 1 for n in nodes):
            costs[key] = sum(known_prices)
    return costs
