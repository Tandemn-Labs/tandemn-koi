"""Advisory queue estimates layered on point-capacity predictions."""

import math
from typing import Any


def estimate_queue_shadow(
    *,
    arrival_rate_rps: Any,
    output_tokens_per_request: Any,
    aggregate_capacity_tps: Any,
    replicas: Any,
    base_ttft_ms: Any = None,
    homogeneous: bool = True,
    scenario: str = "mean",
    peak_to_mean_ratio: Any = 1.0,
) -> dict[str, Any]:
    """Return an uncalibrated Erlang-C estimate without making admission decisions."""
    result: dict[str, Any] = {
        "model": "erlang_c_v1",
        "mode": "shadow",
        "status": "unmodeled",
        "confidence": "uncalibrated",
        "affects_selection": False,
        "scenario": scenario,
    }
    if scenario not in {"mean", "peak"}:
        result["reason"] = "queue scenario must be mean or peak"
        return result
    if not homogeneous:
        result["reason"] = "heterogeneous replicas are outside the M/M/c approximation"
        return result
    try:
        arrival = float(arrival_rate_rps)
        output_tokens = float(output_tokens_per_request)
        capacity_tps = float(aggregate_capacity_tps)
        if isinstance(replicas, bool):
            raise ValueError
        servers_float = float(replicas)
        if not servers_float.is_integer():
            raise ValueError
        servers = int(servers_float)
        peak_ratio = float(peak_to_mean_ratio)
    except (TypeError, ValueError, OverflowError):
        result["reason"] = "arrival rate, output length, capacity, and replicas are required"
        return result
    if (
        not all(math.isfinite(value) for value in (arrival, output_tokens, capacity_tps))
        or arrival < 0
        or output_tokens <= 0
        or capacity_tps <= 0
        or servers < 1
        or servers > 4096
        or not math.isfinite(peak_ratio)
        or peak_ratio <= 0
    ):
        result["reason"] = "queue inputs must be finite and positive"
        return result

    if scenario == "peak":
        arrival *= peak_ratio
    if not math.isfinite(arrival):
        result["reason"] = "scenario-adjusted arrival rate is not finite"
        return result
    total_service_rps = capacity_tps / output_tokens
    service_rate_per_replica = total_service_rps / servers
    if not all(
        math.isfinite(value) and value > 0
        for value in (total_service_rps, service_rate_per_replica)
    ):
        result["reason"] = "derived service rate is not finite and positive"
        return result
    utilization = arrival / total_service_rps
    if not math.isfinite(utilization):
        result["reason"] = "derived utilization is not finite"
        return result
    result.update(
        arrival_rate_rps=arrival,
        service_rate_rps=total_service_rps,
        replicas=servers,
        utilization=utilization,
    )
    if arrival == 0:
        try:
            base_ttft = float(base_ttft_ms)
        except (TypeError, ValueError, OverflowError):
            base_ttft = math.nan
        result.update(
            status="stable",
            probability_wait=0.0,
            queue_wait_p99_ms=0.0,
            base_p99_plus_queue_p99_ms=(
                base_ttft if math.isfinite(base_ttft) and base_ttft >= 0 else None
            ),
        )
        return result
    if utilization >= 1.0:
        result.update(
            status="unstable",
            queue_wait_p99_ms=None,
            base_p99_plus_queue_p99_ms=None,
        )
        return result

    offered_load = arrival / service_rate_per_replica
    # Erlang-B recursion avoids factorial/power overflow for large replica pools.
    erlang_b = 1.0
    for n in range(1, servers + 1):
        erlang_b = offered_load * erlang_b / (n + offered_load * erlang_b)
    probability_wait = erlang_b / (1.0 - utilization + utilization * erlang_b)
    queue_wait_p99_s = (
        math.log(probability_wait / 0.01) / (total_service_rps - arrival)
        if probability_wait > 0.01
        else 0.0
    )
    queue_wait_p99_ms = max(0.0, queue_wait_p99_s * 1000.0)
    if not math.isfinite(probability_wait) or not math.isfinite(queue_wait_p99_ms):
        result.update(status="unmodeled", reason="queue calculation was not finite")
        return result
    try:
        base_ttft = float(base_ttft_ms)
    except (TypeError, ValueError, OverflowError):
        base_ttft = math.nan
    result.update(
        status="stable",
        probability_wait=probability_wait,
        queue_wait_p99_ms=queue_wait_p99_ms,
        base_p99_plus_queue_p99_ms=(
            base_ttft + queue_wait_p99_ms if math.isfinite(base_ttft) and base_ttft >= 0 else None
        ),
    )
    return result
