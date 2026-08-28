from src.prediction.queue_model import estimate_queue_shadow


def test_queue_shadow_reports_stable_estimate_without_affecting_selection():
    result = estimate_queue_shadow(
        arrival_rate_rps=4.0,
        output_tokens_per_request=100.0,
        aggregate_capacity_tps=1000.0,
        replicas=2,
        base_ttft_ms=20.0,
    )

    assert result["status"] == "stable"
    assert result["mode"] == "shadow"
    assert result["utilization"] == 0.4
    assert result["queue_wait_p99_ms"] >= 0.0
    assert result["base_p99_plus_queue_p99_ms"] >= 20.0
    assert result["affects_selection"] is False


def test_queue_shadow_reports_unstable_and_unmodeled_inputs():
    unstable = estimate_queue_shadow(
        arrival_rate_rps=10.0,
        output_tokens_per_request=100.0,
        aggregate_capacity_tps=500.0,
        replicas=2,
    )
    heterogeneous = estimate_queue_shadow(
        arrival_rate_rps=1.0,
        output_tokens_per_request=100.0,
        aggregate_capacity_tps=500.0,
        replicas=2,
        homogeneous=False,
    )

    assert unstable["status"] == "unstable"
    assert unstable["base_p99_plus_queue_p99_ms"] is None
    assert heterogeneous["status"] == "unmodeled"
    assert heterogeneous["affects_selection"] is False


def test_queue_shadow_is_fail_open_for_edge_inputs():
    zero_arrival = estimate_queue_shadow(
        arrival_rate_rps=0.0,
        output_tokens_per_request=100.0,
        aggregate_capacity_tps=500.0,
        replicas=2,
        base_ttft_ms=20.0,
    )
    underflow = estimate_queue_shadow(
        arrival_rate_rps=1e-300,
        output_tokens_per_request=1e300,
        aggregate_capacity_tps=1e-300,
        replicas=1,
    )
    fractional_replicas = estimate_queue_shadow(
        arrival_rate_rps=1.0,
        output_tokens_per_request=100.0,
        aggregate_capacity_tps=500.0,
        replicas=1.5,
    )

    assert zero_arrival["status"] == "stable"
    assert zero_arrival["queue_wait_p99_ms"] == 0.0
    assert underflow["status"] == "unmodeled"
    assert fractional_replicas["status"] == "unmodeled"


def test_queue_shadow_uses_peak_arrival_rate():
    result = estimate_queue_shadow(
        arrival_rate_rps=6.0,
        output_tokens_per_request=100.0,
        aggregate_capacity_tps=1000.0,
        replicas=2,
        scenario="peak",
        peak_to_mean_ratio=2.0,
    )

    assert result["arrival_rate_rps"] == 12.0
    assert result["utilization"] == 1.2
    assert result["status"] == "unstable"


def test_queue_shadow_reports_decode_and_tail_token_work_pressure():
    result = estimate_queue_shadow(
        arrival_rate_rps=2.0,
        input_tokens_per_request=200.0,
        input_tokens_per_request_max=800.0,
        output_tokens_per_request=100.0,
        output_tokens_per_request_max=300.0,
        aggregate_capacity_tps=1000.0,
        replicas=2,
        affects_selection=True,
    )

    assert result["utilization"] == 0.2
    assert result["decode_utilization"] == 0.2
    assert result["combined_tokens_per_request"] == 300.0
    assert result["combined_token_work_pressure"] == 0.6
    assert result["tail_tokens_per_request"] == 1100.0
    assert result["tail_token_work_pressure"] == 2.2
    assert result["status"] == "stable"
    assert result["affects_selection"] is True
