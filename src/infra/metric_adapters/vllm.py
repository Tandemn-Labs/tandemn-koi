"""Normalize vLLM query results into candidate-graph observations."""

from __future__ import annotations

from datetime import datetime

from src.infra.metric_adapters.common import collect_queries
from src.infra.metric_adapters.models import MetricObservation, MetricTarget, QueryExecutor
from src.infra.metric_adapters.profiles import MetricQueryProfile


class VllmAdapter:
    """Collect normalized vLLM V/Y observations for resolved live targets."""

    def __init__(
        self,
        profile: MetricQueryProfile,
        *,
        step_seconds: int = 10,
    ) -> None:
        if profile.source != "vllm":
            raise ValueError(f"vLLM adapter requires a vllm profile, got {profile.source!r}")
        self.profile = profile
        self.step_seconds = step_seconds

    def collect(
        self,
        targets: list[MetricTarget],
        start: datetime,
        end: datetime,
        execute_query: QueryExecutor,
        *,
        requested_variables: set[str] | None = None,
    ) -> list[MetricObservation]:
        return collect_queries(
            source="vllm",
            targets=targets,
            queries=self.profile.queries,
            execute_query=execute_query,
            start=start,
            end=end,
            step_seconds=self.step_seconds,
            requested_variables=requested_variables,
        )
