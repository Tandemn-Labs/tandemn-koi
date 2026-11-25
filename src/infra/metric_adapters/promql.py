"""Minimal Prometheus-compatible range-query executor."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

from src.infra.metric_adapters.models import MetricTarget, TimeSeries

_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class MetricQueryError(RuntimeError):
    """A TSDB query failed or returned an invalid response."""


class PromQLExecutor:
    """Execute query expressions through the Prometheus HTTP range API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        bearer_token: str | None = None,
        opener=None,
    ) -> None:
        if not base_url:
            raise ValueError("Prometheus base_url is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.bearer_token = bearer_token
        self.opener = opener or urllib.request.urlopen

    def execute(
        self,
        expression_template: str,
        target: MetricTarget,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[TimeSeries]:
        """Return finite matrix samples for one PromQL expression."""

        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        expression = expression_template.replace("{target}", _selector_for_target(target))
        data = urllib.parse.urlencode(
            {
                "query": expression,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step_seconds,
            }
        ).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = urllib.request.Request(
            f"{self.base_url}/api/v1/query_range",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            raise MetricQueryError(f"Prometheus range query failed: {error}") from error
        return _parse_matrix(payload)


def _selector_for_target(target: MetricTarget) -> str:
    labels = {
        **{str(name): str(value) for name, value in target.selectors.items()},
        "job_id": target.job_id,
        "rank_id": target.rank_id,
    }
    labels.pop("chain_index", None)
    parts = []
    for name, value in sorted(labels.items()):
        if not _LABEL_NAME.fullmatch(name):
            raise ValueError(f"invalid Prometheus label name {name!r}")
        escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        parts.append(f'{name}="{escaped}"')
    chain_pattern = "|".join(str(index) for index in sorted(set(target.chain_indexes)))
    parts.append(f'chain_index=~"{chain_pattern}"')
    return "{" + ",".join(parts) + "}"


def _parse_matrix(payload: Any) -> list[TimeSeries]:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        message = payload.get("error") if isinstance(payload, dict) else None
        raise MetricQueryError(f"Prometheus returned an error{f': {message}' if message else ''}")
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("resultType") != "matrix":
        raise MetricQueryError("Prometheus range query did not return a matrix")

    output = []
    for result in data.get("result") or []:
        if not isinstance(result, dict):
            continue
        raw_labels = result.get("metric")
        raw_values = result.get("values")
        if not isinstance(raw_labels, dict) or not isinstance(raw_values, list):
            continue
        points = []
        for sample in raw_values:
            if not isinstance(sample, list | tuple) or len(sample) != 2:
                continue
            try:
                timestamp = float(sample[0])
                value = float(sample[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(timestamp) and math.isfinite(value):
                points.append((timestamp, value))
        if points:
            output.append(
                TimeSeries(
                    labels={str(name): str(value) for name, value in raw_labels.items()},
                    points=points,
                )
            )
    return output
