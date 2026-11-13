"""Source-independent data passed into and out of metric adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class MetricTarget:
    """A live rank and the labels adapters may use to query its chains."""

    job_id: str
    rank_id: str
    chain_indexes: tuple[int, ...]
    selectors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id or not self.rank_id:
            raise ValueError("metric target requires job_id and rank_id")
        if not self.chain_indexes:
            raise ValueError("metric target requires at least one chain_index")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.chain_indexes
        ):
            raise ValueError("chain_indexes must be nonnegative integers")


@dataclass(frozen=True)
class TimeSeries:
    """One query result series with its original TSDB labels."""

    labels: dict[str, str]
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class MetricObservation:
    """One normalized candidate-graph observation from an adapter."""

    job_id: str
    rank_id: str
    chain_index: int
    timestamp: float
    variable: str
    value: float
    source: str
    labels: dict[str, str] = field(default_factory=dict)


QueryExecutor = Callable[[str, MetricTarget, datetime, datetime, int], list[TimeSeries]]
