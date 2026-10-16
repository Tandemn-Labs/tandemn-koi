"""Read numeric time series through the Prometheus range-query interface."""

import json
import urllib.parse
import urllib.request
from datetime import datetime
from math import isfinite
from typing import Any

import numpy as np
from src.infra.metrics.base import MetricQueryError, QueryResult, TimeSeries


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward authentication headers to another endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PrometheusReader:
    """
    PrometheusReader provides an interface for fetching numeric time series from a Prometheus
    metrics database using arbitrary PromQL queries.

    This class is agnostic to the meaning or semantics of each query; interpretation is the
    responsibility of the source adapters or callers constructing the PromQL expressions.

    Authentication, request validation, and result parsing are handled internally; time-labelled
    results are returned as QueryResult instances.

    Federated (chain) identifiers are not required or used. Any matched Prometheus targets or
    label sets are queried directly. All responsibility for interpreting or transforming
    metric values is delegated to the code that builds and consumes the query results.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        bearer_token: str | None = None,
        opener=None,
    ) -> None:
        url = urllib.parse.urlsplit(base_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("metrics URL must be an HTTP or HTTPS URL")
        if url.username is not None or url.password is not None:
            raise ValueError("metrics URL must not contain credentials; use bearer_token")
        if url.query or url.fragment:
            raise ValueError("metrics URL must not contain query parameters or a fragment")
        if (
            isinstance(timeout_seconds, bool)
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if bearer_token is not None and (
            not isinstance(bearer_token, str)
            or not bearer_token
            or "\r" in bearer_token
            or "\n" in bearer_token
        ):
            raise ValueError("bearer_token must be a nonempty single-line string")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.bearer_token = bearer_token
        self.opener = opener or urllib.request.build_opener(_NoRedirect()).open

    def query_range(
        self,
        query: str,
        *,
        start: datetime,
        end: datetime,
        step_seconds: int = 10,
    ) -> QueryResult:
        """
        Query Prometheus for a given PromQL expression over a time range and return
        time-labelled results as a QueryResult. This method does not require or use
        federated (chain) identifiers; it simply executes the provided query and
        returns the timeseries for any Prometheus targets or labels matched.

        Args:
            query (str): The Prometheus query string (PromQL).
            start (datetime): Start time (timezone-aware) for the query range.
            end (datetime): End time (timezone-aware) for the query range.
            step_seconds (int, optional): Query resolution step width in seconds.

        Returns:
            QueryResult: Contains times (as floats) and corresponding values,
            along with any metadata or warnings from Prometheus.

        Raises:
            ValueError: If any argument is invalid or empty, or timestamps are naive.
            MetricQueryError: If the Prometheus query fails or returns invalid data.

        Example:
            reader.query_range(
                "node_cpu_seconds_total{mode='idle'}",
                start=datetime_obj_1,
                end=datetime_obj_2,
                step_seconds=15
            )
        """

        if not query.strip():
            raise ValueError("query must not be empty")
        if start.utcoffset() is None or end.utcoffset() is None:
            raise ValueError("query times must include a time zone")
        if end < start:
            raise ValueError("query end must not precede start")
        if isinstance(step_seconds, bool) or not isinstance(step_seconds, int) or step_seconds <= 0:
            raise ValueError("step_seconds must be a positive integer")
        data = urllib.parse.urlencode(
            {
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step_seconds,
            }
        ).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if self.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = urllib.request.Request(
            f"{self.base_url}/api/v1/query_range", data=data, headers=headers, method="POST"
        )
        # the url above is coming from the prometheus documentation
        # https://prometheus.io/docs/prometheus/latest/querying/api/#range-queries
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except (OSError, ValueError) as error:
            raise MetricQueryError(f"Prometheus request failed: {error}") from error
        return _parse_matrix(payload, start.timestamp(), end.timestamp())


def _parse_matrix(payload: Any, start: float, end: float) -> QueryResult:
    """
    Prometheus matrix response into a QueryResult.

    Args:
        payload: Decoded JSON response.
        start: Query window start (epoch seconds).
        end: Query window end (epoch seconds).

    Returns:
        QueryResult.

    Raises:
        MetricQueryError: On invalid or unsupported payload.

    """
    if not isinstance(payload, dict) or payload.get("status") != "success":
        reason = (
            payload.get("error", "invalid response")
            if isinstance(payload, dict)
            else "invalid response"
        )
        raise MetricQueryError(f"Prometheus query failed: {reason}")
    data = payload.get("data")
    if (
        not isinstance(data, dict)
        or data.get("resultType") != "matrix"
        or not isinstance(data.get("result"), list)
    ):
        raise MetricQueryError("Prometheus returned an invalid matrix")
    warnings = list(_messages(payload, "warnings"))
    infos = _messages(payload, "infos")
    series = []
    for result in data["result"]:
        if not isinstance(result, dict):
            raise MetricQueryError("Prometheus returned an invalid series")
        if result.get("histograms"):
            raise MetricQueryError("raw histogram samples are unsupported; query a numeric result")
        labels = result.get("metric")
        values = result.get("values")
        if (
            not isinstance(labels, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in labels.items()
            )
            or not isinstance(values, list)
        ):
            raise MetricQueryError("Prometheus returned an invalid series")
        if not values:
            continue
        try:
            if any(
                not isinstance(point, list | tuple)
                or len(point) != 2
                or any(isinstance(value, bool) for value in point)
                for point in values
            ):
                raise ValueError
            samples = np.asarray([[float(value) for value in point] for point in values])
        except (TypeError, ValueError, OverflowError) as error:
            raise MetricQueryError("Prometheus returned invalid numeric samples") from error
        finite = np.isfinite(samples).all(axis=1)
        dropped = int((~finite).sum())
        if dropped:
            warnings.append(f"Dropped {dropped} non-finite samples from a series")
            samples = samples[finite]
        if not len(samples):
            continue
        timestamps = samples[:, 0]
        if (
            np.any(timestamps < start)
            or np.any(timestamps > end)
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise MetricQueryError(
                "Prometheus sample times must be ordered within the query window"
            )
        series.append(TimeSeries(dict(labels), timestamps, samples[:, 1]))
    return QueryResult(tuple(series), tuple(warnings), infos)


def _messages(payload: dict[str, Any], name: str) -> tuple[str, ...]:
    values = payload.get(name, [])
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise MetricQueryError("Prometheus warnings and infos must be lists of strings")
    return tuple(values)
