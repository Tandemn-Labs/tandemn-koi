"""Rank context supplied by the resource map for federated telemetry."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from math import isfinite
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class MetricQueryError(RuntimeError):
    """The metrics database failed or returned data Koi cannot use."""


@dataclass(frozen=True)
class TimeSeries:
    """One metric over time, with its labels stored once.

    ``timestamps`` contains seconds since the Unix epoch. ``values`` contains
    the matching numeric samples. Both arrays have the same length.
    """

    labels: dict[str, str]
    timestamps: NDArray[np.float64]
    values: NDArray[np.float64]


@dataclass(frozen=True)
class QueryResult:
    """All series and messages returned by one metrics query.

    Warnings can mean that the database returned only part of the requested
    data. Keep them with the result so Koi does not treat partial data as whole.
    """

    series: tuple[TimeSeries, ...]
    warnings: tuple[str, ...] = ()
    infos: tuple[str, ...] = ()


# TODO: map node types and hourly prices from the finalized resource-map format.
@dataclass(frozen=True)
class NodePrice:
    """One node assigned to a rank and the node's price for one hour.

    ``cluster_id`` and ``node_id`` together identify the node. A price of
    ``None`` means unknown; ``0`` means the node is free.
    """

    cluster_id: str
    node_id: str
    price_per_node_hour: float | None = None

    def __post_init__(self) -> None:
        _text(self.cluster_id, "cluster_id")
        _text(self.node_id, "node_id")
        if self.price_per_node_hour is not None:
            price = _finite_number(self.price_per_node_hour, "price_per_node_hour")
            if price < 0:
                raise ValueError("price_per_node_hour must be nonnegative")


# TODO - Need to confirm the data structure
@dataclass(frozen=True)
class RankContext:
    """Map facts needed to collect metrics for one rank.

    The engine and device source choose the two metric adapters. ``nodes`` lists
    the nodes charged to this rank, so Koi can add their prices later. Chain IDs
    are not needed to select adapters or identify rank-level query results.
    Optional counts and version data stay unknown when the map does not provide
    them; this class does not guess them.
    """

    job_id: str
    rank_id: str
    cluster_id: str
    engine_name: str
    device_source: str
    engine_version: str | None = None
    nodes: tuple[NodePrice, ...] = ()
    expected_replicas: int | None = None
    expected_devices: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    job_features: Mapping[str, Any] = field(default_factory=dict)
    committed_mechanism_id: str | None = None
    deploy_timestamp_utc: float | None = None

    def __post_init__(self) -> None:
        required = ("job_id", "rank_id", "cluster_id", "engine_name", "device_source")
        for name in (*required, "engine_version", "committed_mechanism_id"):
            value = getattr(self, name)
            if name in required or value is not None:
                _text(value, name)
        for name in ("expected_replicas", "expected_devices"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(node, NodePrice) for node in self.nodes
        ):
            raise ValueError("nodes must be a tuple of NodePrice records")
        _mapping(self.config, "config")
        _mapping(self.job_features, "job_features")
        if self.deploy_timestamp_utc is not None:
            _finite_number(self.deploy_timestamp_utc, "deploy_timestamp_utc")


class MetricAdapter(Protocol):
    """One engine or device source that can collect metrics for a rank.

    ``kind`` is ``"engine"`` or ``"device"``. ``name`` must match the source
    name in ``RankContext``.
    """

    kind: str
    name: str


def load_plugins(adapters: Iterable[MetricAdapter]) -> dict[tuple[str, str], MetricAdapter]:
    """Build the lookup used to select adapters for ranks.

    Each adapter is stored under ``(kind, name)``. For example,
    ``("engine", "vllm")`` identifies the vLLM engine adapter. Call this once
    when Koi starts, not once for every rank.

    Raises:
        ValueError: If an adapter has an unknown kind, a blank name, or shares
            its ``(kind, name)`` key with another adapter.
    """
    plugins = {}
    for adapter in adapters:
        kind = getattr(adapter, "kind", None)
        name = getattr(adapter, "name", None)
        if kind not in {"engine", "device"} or not isinstance(name, str) or not name.strip():
            raise ValueError("metric adapters require kind 'engine' or 'device' and a source name")
        key = (kind, name)
        if key in plugins:
            raise ValueError(f"duplicate metric adapter for {key!r}")
        plugins[key] = adapter
    return plugins


def adapters_for_rank(
    context: RankContext,
    plugins: Mapping[tuple[str, str], MetricAdapter],
) -> tuple[MetricAdapter, MetricAdapter]:
    """Choose the engine adapter and device adapter named by one rank.

    The returned pair is always ``(engine_adapter, device_adapter)``. This
    uses the exact source names in the rank context.

    Raises:
        ValueError: If either source has no registered adapter. This function
            never chooses a different source as a fallback.
    """

    selected = []
    for kind, source in (("engine", context.engine_name), ("device", context.device_source)):
        adapter = plugins.get((kind, source))
        if adapter is None:
            raise ValueError(
                f"no {kind} adapter for {source!r} on job/rank {context.job_id}/{context.rank_id}"
            )
        selected.append(adapter)
    return selected[0], selected[1]


# TODO - Exact schema not available rn, will update when resource map is finalized
def resolve_rank_contexts(snapshot: Any) -> list[RankContext]:
    """Read rank facts from the resource-map snapshot.

    Each active job must have a ``ranks`` list. Each rank record must name its
    cluster, engine, and device source. Node prices and live counts are read
    only when present; this function does not infer them from legacy chain
    records. The job ID and job features come from the parent job record.

    Returns:
        One context for each unique ``(job_id, rank_id)`` pair.

    Raises:
        ValueError: If a job lacks rank records, a required fact is invalid, or
            the snapshot repeats a job/rank pair.
    """

    contexts = {}
    for raw_job in snapshot.active_jobs_summary():
        job = _mapping(raw_job, "job")
        job_id = _text(job.get("job_id"), "job_id")
        if "ranks" not in job:
            raise ValueError(f"job {job_id!r} requires explicit ranks for federated telemetry")
        job_features = _mapping(job.get("job_features", {}), "job_features")
        for raw_rank in _records(job["ranks"], "ranks"):
            rank = _mapping(raw_rank, "rank")
            nodes = []
            for raw_node in _records(rank.get("nodes", ()), "nodes"):
                node = _mapping(raw_node, "node")
                node_values: dict[str, Any] = {
                    item.name: node.get(item.name) for item in fields(NodePrice)
                }
                nodes.append(NodePrice(**node_values))
            rank_values: dict[str, Any] = {
                item.name: rank.get(item.name) for item in fields(RankContext)
            }
            rank_values.update(
                job_id=job_id,
                nodes=tuple(nodes),
                config=dict(_mapping(rank.get("config", {}), "config")),
                job_features=dict(job_features),
            )
            context = RankContext(**rank_values)
            key = (context.job_id, context.rank_id)
            if key in contexts:
                raise ValueError(f"duplicate rank context for job/rank {key!r}")
            contexts[key] = context
    return list(contexts.values())


def _text(value: Any, name: str) -> str:
    """Return a nonblank string or name the invalid field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite_number(value: Any, name: str) -> float:
    """Return a finite number as a float; reject booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Check that a snapshot field contains a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _records(value: Any, name: str) -> list[Any] | tuple[Any, ...]:
    """Check that a snapshot field contains a list or tuple of records."""
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list or tuple of records")
    return value
