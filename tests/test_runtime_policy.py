from koi.runtime_policy import (
    RuntimeChainState,
    RuntimeJobState,
    ScaleUpCandidate,
    compute_required_tps,
    filter_dominated_actions,
    rank_falling_behind_suggestions,
    rank_overprovisioned_suggestions,
)


def test_compute_required_tps_handles_deadline_exceeded():
    assert compute_required_tps(1_000_000, 0.0) == float("inf")


def test_filter_dominated_actions_drops_more_expensive_no_better_option():
    job = RuntimeJobState(
        trigger_type="falling_behind",
        elapsed_hours=1.0,
        time_left_hours=1.0,
        tokens_remaining=3_600_000,
        aggregate_tps=500.0,
        cost_roofline_usd=100.0,
    )
    chains = [
        RuntimeChainState(
            replica_id="r0",
            gpu_type="L40S",
            tp=4,
            pp=1,
            smoothed_tps=500.0,
            predicted_tps=500.0,
            cost_per_hour=10.0,
            status="running",
        )
    ]
    candidates = [
        ScaleUpCandidate(
            gpu_type="L40S",
            tp=4,
            pp=1,
            predicted_tps=700.0,
            cost_per_hour=10.0,
            source="current_config",
        ),
        ScaleUpCandidate(
            gpu_type="A100-80GB",
            tp=8,
            pp=1,
            predicted_tps=700.0,
            cost_per_hour=20.0,
            source="best_running",
        ),
    ]

    ranked = rank_falling_behind_suggestions(job, chains, candidates)
    filtered = filter_dominated_actions(ranked)

    assert len(filtered) == 1
    assert filtered[0].gpu_type == "L40S"


def test_rank_falling_behind_prefers_cheapest_valid_scale_up():
    job = RuntimeJobState(
        trigger_type="falling_behind",
        elapsed_hours=0.5,
        time_left_hours=1.0,
        tokens_remaining=3_600_000,
        aggregate_tps=600.0,
        cost_roofline_usd=50.0,
    )
    chains = [
        RuntimeChainState(
            replica_id="r0",
            gpu_type="L40S",
            tp=4,
            pp=1,
            smoothed_tps=600.0,
            predicted_tps=600.0,
            cost_per_hour=10.0,
            status="running",
        )
    ]
    candidates = [
        ScaleUpCandidate(
            gpu_type="A100-80GB",
            tp=8,
            pp=1,
            predicted_tps=1200.0,
            cost_per_hour=20.0,
            source="best_running",
        ),
        ScaleUpCandidate(
            gpu_type="L40S",
            tp=4,
            pp=1,
            predicted_tps=1200.0,
            cost_per_hour=10.0,
            source="current_config",
        ),
    ]

    ranked = rank_falling_behind_suggestions(job, chains, candidates)

    assert ranked[0].gpu_type == "L40S"
    assert ranked[0].meets_slo is True
    assert ranked[1].gpu_type == "A100-80GB"


def test_rank_overprovisioned_picks_one_safe_lowest_tps_chain():
    job = RuntimeJobState(
        trigger_type="over_provisioned",
        elapsed_hours=0.5,
        time_left_hours=1.5,
        tokens_remaining=1_000_000,
        aggregate_tps=3000.0,
        cost_roofline_usd=100.0,
    )
    chains = [
        RuntimeChainState(
            replica_id="r-fast",
            gpu_type="A100-80GB",
            tp=8,
            pp=1,
            smoothed_tps=1500.0,
            predicted_tps=1500.0,
            cost_per_hour=20.0,
            status="running",
        ),
        RuntimeChainState(
            replica_id="r-mid",
            gpu_type="L40S",
            tp=4,
            pp=1,
            smoothed_tps=900.0,
            predicted_tps=900.0,
            cost_per_hour=10.0,
            status="running",
        ),
        RuntimeChainState(
            replica_id="r-slow",
            gpu_type="L40S",
            tp=4,
            pp=1,
            smoothed_tps=600.0,
            predicted_tps=600.0,
            cost_per_hour=10.0,
            status="running",
        ),
    ]

    ranked = rank_overprovisioned_suggestions(job, chains)

    assert ranked[0].replica_id == "r-slow"
    assert ranked[0].meets_slo is True


def test_rank_overprovisioned_returns_no_suggestion_when_no_safe_kill_exists():
    job = RuntimeJobState(
        trigger_type="over_provisioned",
        elapsed_hours=0.5,
        time_left_hours=1.0,
        tokens_remaining=3_600_000,
        aggregate_tps=1200.0,
        cost_roofline_usd=100.0,
    )
    chains = [
        RuntimeChainState(
            replica_id="r0",
            gpu_type="L40S",
            tp=4,
            pp=1,
            smoothed_tps=700.0,
            predicted_tps=700.0,
            cost_per_hour=10.0,
            status="running",
        ),
        RuntimeChainState(
            replica_id="r1",
            gpu_type="L40S",
            tp=4,
            pp=1,
            smoothed_tps=500.0,
            predicted_tps=500.0,
            cost_per_hour=10.0,
            status="running",
        ),
    ]

    ranked = rank_overprovisioned_suggestions(job, chains)

    assert ranked == []
