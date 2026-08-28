"""Advisory queue estimates layered on point-capacity predictions."""

import math
from typing import Any


def estimate_queue_shadow(
    *,
    arrival_rate_rps: Any,
    output_tokens_per_request: Any,
    aggregate_capacity_tps: Any,
    replicas: Any,
    input_tokens_per_request: Any = 0.0,
    input_tokens_per_request_max: Any = None,
    output_tokens_per_request_max: Any = None,
    base_ttft_ms: Any = None,
    homogeneous: bool = True,
    scenario: str = "mean",
    peak_to_mean_ratio: Any = 1.0,
    affects_selection: bool = False,
) -> dict[str, Any]:
    """Return an uncalibrated Erlang-C estimate with conservative token-work pressure."""
    result: dict[str, Any] = {
        "model": "erlang_c_token_work_v2",
        "mode": "shadow",
        "status": "unmodeled",
        "confidence": "uncalibrated",
        "affects_selection": bool(affects_selection),
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
        input_tokens = float(input_tokens_per_request or 0.0)
        output_tokens = float(output_tokens_per_request)
        input_tokens_max = (
            input_tokens
            if input_tokens_per_request_max is None
            else float(input_tokens_per_request_max)
        )
        output_tokens_max = (
            output_tokens
            if output_tokens_per_request_max is None
            else float(output_tokens_per_request_max)
        )
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
        not all(
            math.isfinite(value)
            for value in (
                arrival,
                input_tokens,
                output_tokens,
                input_tokens_max,
                output_tokens_max,
                capacity_tps,
            )
        )
        or arrival < 0
        or input_tokens < 0
        or output_tokens <= 0
        or input_tokens_max < 0
        or output_tokens_max <= 0
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
    combined_tokens = input_tokens + output_tokens
    tail_tokens = max(input_tokens, input_tokens_max) + max(output_tokens, output_tokens_max)
    combined_service_rate_rps = capacity_tps / combined_tokens
    service_rate_per_replica = total_service_rps / servers
    if not all(
        math.isfinite(value) and value > 0
        for value in (total_service_rps, combined_service_rate_rps, service_rate_per_replica)
    ):
        result["reason"] = "derived service rate is not finite and positive"
        return result
    decode_utilization = arrival / total_service_rps
    combined_token_work_pressure = arrival * combined_tokens / capacity_tps
    tail_token_work_pressure = arrival * tail_tokens / capacity_tps
    if not all(
        math.isfinite(value)
        for value in (
            decode_utilization,
            combined_token_work_pressure,
            tail_token_work_pressure,
        )
    ):
        result["reason"] = "derived utilization or token-work pressure is not finite"
        return result
    result.update(
        arrival_rate_rps=arrival,
        service_rate_rps=total_service_rps,
        combined_service_rate_rps=combined_service_rate_rps,
        replicas=servers,
        utilization=decode_utilization,
        decode_utilization=decode_utilization,
        combined_tokens_per_request=combined_tokens,
        combined_token_work_pressure=combined_token_work_pressure,
        tail_tokens_per_request=tail_tokens,
        tail_token_work_pressure=tail_token_work_pressure,
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
    if decode_utilization >= 1.0:
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
    probability_wait = erlang_b / (1.0 - decode_utilization + decode_utilization * erlang_b)
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
