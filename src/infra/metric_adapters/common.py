"""Small shared helpers for metric adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

from src.infra.metric_adapters.models import (
    MetricObservation,
    MetricTarget,
    QueryExecutor,
)


def collect_queries(
    *,
    source: str,
    targets: list[MetricTarget],
    queries: Mapping[str, str],
    execute_query: QueryExecutor,
    start: datetime,
    end: datetime,
    step_seconds: int,
    requested_variables: set[str] | None,
) -> list[MetricObservation]:
    """Run source queries and normalize labeled series into observations."""

    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    observations = []
    for target in targets:
        for variable, template in queries.items():
            if requested_variables is not None and variable not in requested_variables:
                continue
            for series in execute_query(template, target, start, end, step_seconds):
                identity = _series_identity(series.labels, target)
                if identity is None:
                    continue
                job_id, rank_id, chain_index = identity
                for timestamp, value in series.points:
                    if math.isfinite(timestamp) and math.isfinite(value):
                        observations.append(
                            MetricObservation(
                                job_id=job_id,
                                rank_id=rank_id,
                                chain_index=chain_index,
                                timestamp=timestamp,
                                variable=variable,
                                value=value,
                                source=source,
                                labels=dict(series.labels),
                            )
                        )
    return observations


def _series_identity(
    labels: Mapping[str, str], target: MetricTarget
) -> tuple[str, str, int] | None:
    job_id = labels.get("job_id")
    rank_id = labels.get("rank_id")
    raw_chain_index = labels.get("chain_index")
    if job_id != target.job_id or rank_id != target.rank_id or raw_chain_index is None:
        return None
    try:
        chain_index = int(raw_chain_index)
    except ValueError:
        return None
    if chain_index not in target.chain_indexes:
        return None
    return job_id, rank_id, chain_index
