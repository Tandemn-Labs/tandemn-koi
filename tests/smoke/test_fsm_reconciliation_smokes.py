from src.core.models import Plan, PlanAction
from src.infra.resource_map import ClusterResourceSnapshot
from src.orchestrator.fsm_states import TickContext, TickRunner


def _runner():
    runner = TickRunner.__new__(TickRunner)
    runner._deployment_ledger = {}
    runner._prediction_ledger = {}
    runner._health_state = {}
    return runner


def test_deployment_reconciliation_marks_repeated_missing_action():
    runner = _runner()
    runner._deployment_ledger["job-1"] = {
        "request_tick": 1,
        "first_tick": 1,
        "attempts": 2,
        "action_type": "place",
        "rank_ids": ["rank-1"],
        "shapes": [{"tp": 2}],
    }
    pending = {"job_id": "job-1", "job_features": {}}
    ctx = TickContext(
        tick=3,
        cluster_snapshot=ClusterResourceSnapshot(3, {}, [], [pending]),
    )

    result = runner._reconcile_deployments(ctx)

    assert result[0]["status"] == "deployment_not_materialized"
    assert pending["deployment_status"] == "deployment_not_materialized"
    assert pending["recent_failures"] == 1


def test_deployment_reconciliation_observes_expected_rank():
    runner = _runner()
    runner._deployment_ledger["job-1"] = {
        "request_tick": 1,
        "first_tick": 1,
        "attempts": 1,
        "action_type": "place",
        "rank_ids": ["rank-1"],
        "shapes": [{"tp": 2}],
    }
    active = {
        "job_id": "job-1",
        "active_chains": [{"shape_json": {"rank_id": "rank-1"}}],
    }
    ctx = TickContext(
        tick=2,
        cluster_snapshot=ClusterResourceSnapshot(2, {}, [active], []),
    )

    result = runner._reconcile_deployments(ctx)

    assert result[0]["status"] == "active"
    assert "job-1" not in runner._deployment_ledger


def test_deployment_reconciliation_accepts_any_successful_retry_attempt():
    runner = _runner()
    runner._deployment_ledger["job-1"] = {
        "request_tick": 2,
        "first_tick": 1,
        "attempts": 2,
        "action_type": "place",
        "rank_ids": ["rank-new"],
        "rank_id_attempts": [["rank-old"], ["rank-new"]],
        "shapes": [{"tp": 2}],
    }
    active = {
        "job_id": "job-1",
        "active_chains": [{"shape_json": {"rank_id": "rank-old"}}],
    }
    ctx = TickContext(
        tick=3,
        cluster_snapshot=ClusterResourceSnapshot(3, {}, [active], []),
    )

    result = runner._reconcile_deployments(ctx)

    assert result[0]["status"] == "active"


def test_deployment_reconciliation_requires_requested_shape_for_reused_rank_id():
    runner = _runner()
    requested_shape = {
        "env": ["reserved", "aws", "r1", "z1", "H100"],
        "instance_type": "p5",
        "tp": 2,
        "pp": 1,
        "sp": 1,
        "ep": 1,
        "cp": 1,
        "n_replicas": 1,
    }
    runner._deployment_ledger["job-1"] = {
        "request_tick": 1,
        "first_tick": 1,
        "attempts": 1,
        "action_type": "swap",
        "rank_ids": ["rank-1"],
        "rank_shapes": {"rank-1": runner._deployment_shape_signature(requested_shape)},
        "attempt_details": [
            {
                "rank_ids": ["rank-1"],
                "rank_shapes": {"rank-1": runner._deployment_shape_signature(requested_shape)},
            }
        ],
        "shapes": [requested_shape],
    }
    active = {
        "job_id": "job-1",
        "active_chains": [
            {
                "shape_json": {
                    **requested_shape,
                    "rank_id": "rank-1",
                    "tp": 1,
                }
            }
        ],
    }
    ctx = TickContext(
        tick=2,
        cluster_snapshot=ClusterResourceSnapshot(2, {}, [active], []),
    )

    result = runner._reconcile_deployments(ctx)

    assert result[0]["status"] == "deployment_pending"


def test_deployment_reconciliation_ignores_store_enriched_runtime_fields():
    runner = _runner()
    requested_shape = {
        "env": ["reserved", "aws", "r1", "z1", "H100"],
        "config": {
            "instance_type": "p5",
            "gpu_count": 8,
            "tp": 8,
            "pp": 1,
        },
        "n_replicas": 1,
    }
    signature = runner._deployment_shape_signature(requested_shape)
    runner._deployment_ledger["job-1"] = {
        "request_tick": 0,
        "first_tick": 0,
        "attempts": 1,
        "action_type": "place",
        "rank_ids": ["rank-1"],
        "rank_shapes": {"rank-1": signature},
        "attempt_details": [
            {
                "attempt_number": 1,
                "request_tick": 0,
                "rank_ids": ["rank-1"],
                "rank_shapes": {"rank-1": signature},
                "state": "acknowledged",
            }
        ],
        "shapes": [requested_shape],
    }
    active = {
        "job_id": "job-1",
        "active_chains": [
            {
                "shape_json": {
                    "rank_id": "rank-1",
                    "env": requested_shape["env"],
                    "instance_type": "p5",
                    "count": 8,
                    "tp": 8,
                    "pp": 1,
                    "model_id": "meta-llama/Llama-3.3-70B-Instruct",
                    "engine_name": "vllm",
                    "engine_version": "0.16.0",
                }
            }
        ],
    }

    result = runner._reconcile_deployments(
        TickContext(tick=1, cluster_snapshot=ClusterResourceSnapshot(1, {}, [active], []))
    )

    assert result[0]["status"] == "active"
    assert "job-1" not in runner._deployment_ledger


def test_timed_out_deployment_becomes_retryable_after_backoff():
    runner = _runner()
    identity = (("reserved", "aws", "r1", "z1", "H100"), "p5", 8, 8, 1, 1, 1, 1, 1)
    runner._deployment_ledger["job-1"] = {
        "request_tick": 1,
        "first_tick": 1,
        "attempts": 1,
        "action_type": "place",
        "rank_ids": ["rank-1"],
        "attempt_details": [
            {
                "attempt_number": 1,
                "request_tick": 1,
                "rank_ids": ["rank-1"],
                "rank_shapes": {"rank-1": identity},
                "deployment_identity": (identity,),
                "state": "acknowledged",
            }
        ],
        "shapes": [],
    }
    pending = {"job_id": "job-1", "job_features": {}}

    runner._reconcile_deployments(
        TickContext(tick=3, cluster_snapshot=ClusterResourceSnapshot(3, {}, [], [pending]))
    )
    assert pending["deployment_status"] == "deployment_not_materialized"
    assert pending["deployment_retry_allowed"] is False
    assert pending["deployment_retry_after_tick"] == 4

    runner._reconcile_deployments(
        TickContext(tick=4, cluster_snapshot=ClusterResourceSnapshot(4, {}, [], [pending]))
    )
    assert pending["deployment_retry_allowed"] is True
    assert pending["attempted_deployment_identities"] == [(identity,)]


def test_created_ack_is_recorded_without_claiming_materialization():
    runner = _runner()
    action = PlanAction.from_dict(
        {
            "job_id": "job-1",
            "type": "place",
            "ladder": [
                {
                    "role": "aggregate",
                    "env": ["reserved", "aws", "r1", "z1", "H100"],
                    "config": {"instance_type": "p5", "gpu_count": 8, "tp": 8, "pp": 1},
                    "n_replicas": 1,
                }
            ],
        }
    )
    ctx = TickContext(tick=1, validated_plan=Plan(tick=1, actions=[action]))

    attempts = runner._record_deployment_requests(ctx)
    runner._record_deployment_acks(
        attempts,
        [{"plan_id": "plan-1", "status": "created"}],
        ctx.tick,
    )

    attempt = runner._deployment_ledger["job-1"]["attempt_details"][0]
    assert attempt["state"] == "acknowledged"
    assert attempt["executor_ack"] == [{"plan_id": "plan-1", "status": "created"}]
    assert attempt["terminal_tick"] is None


def test_swap_cooldown_starts_when_swap_materializes():
    runner = _runner()
    action = PlanAction.from_dict(
        {
            "job_id": "job-1",
            "type": "swap",
            "ladder": [
                {
                    "role": "aggregate",
                    "env": ["reserved", "aws", "r1", "z1", "H100"],
                    "config": {"instance_type": "p5", "gpu_count": 2, "tp": 2, "pp": 1},
                    "n_replicas": 1,
                }
            ],
        }
    )
    runner._record_deployment_requests(
        TickContext(tick=1, validated_plan=Plan(tick=1, actions=[action]))
    )
    assert runner._health_state.get("job-1", {}).get("last_swap_tick") is None

    rank = action.ladder[0]
    active = {
        "job_id": "job-1",
        "active_chains": [
            {
                "shape_json": {
                    **rank.config,
                    "env": list(rank.env),
                    "rank_id": rank.rank_id,
                }
            }
        ],
    }
    runner._reconcile_deployments(
        TickContext(tick=2, cluster_snapshot=ClusterResourceSnapshot(2, {}, [active], []))
    )

    assert runner._health_state["job-1"]["last_swap_tick"] == 2


def test_prediction_identity_retains_runtime_configuration():
    runner = _runner()
    base = {
        "env": ["reserved", "aws", "r1", "z1", "H100"],
        "config": {
            "instance_type": "p5",
            "gpu_count": 8,
            "tp": 8,
            "pp": 1,
            "engine_name": "vllm",
            "engine_version": "0.16.0",
        },
        "n_replicas": 1,
    }

    assert runner._deployment_shape_signature(base) == runner._deployment_shape_signature(
        {**base, "config": {**base["config"], "engine_version": "0.17.0"}}
    )
    assert runner._prediction_shape_signature(base) != runner._prediction_shape_signature(
        {**base, "config": {**base["config"], "engine_version": "0.17.0"}}
    )


def test_active_health_requires_repeat_or_critical_signal_for_rehabilitation():
    runner = _runner()
    job = {
        "job_id": "job-1",
        "job_features": {"target_p99_ttft_ms": 100.0, "target_p99_tpot_ms": 20.0},
    }
    snapshot = ClusterResourceSnapshot(1, {}, [job], [])
    samples = {
        "job-1": [
            {
                "throughput_token_per_sec": 10.0,
                "p99_ttft_ms": 150.0,
                "p99_tpot_ms": 10.0,
                "depth_req_q": 1.0,
                "telemetry_complete": True,
            }
        ]
    }

    first = runner._update_active_health(
        TickContext(tick=1, cluster_snapshot=snapshot),
        {"job-1": job["job_features"]},
        samples,
    )
    second = runner._update_active_health(
        TickContext(tick=2, cluster_snapshot=snapshot),
        {"job-1": job["job_features"]},
        samples,
    )

    assert first["job-1"]["rehabilitation_eligible"] is False
    assert second["job-1"]["rehabilitation_eligible"] is True
    assert job["health"]["status"] == "degraded"
    assert second["job-1"] is not runner._health_state["job-1"]


def test_sparse_telemetry_does_not_trigger_rehabilitation():
    runner = _runner()
    job = {
        "job_id": "job-1",
        "job_features": {"target_p99_ttft_ms": 100.0, "target_p99_tpot_ms": 20.0},
    }
    snapshot = ClusterResourceSnapshot(1, {}, [job], [])

    result = runner._update_active_health(
        TickContext(tick=1, cluster_snapshot=snapshot),
        {"job-1": job["job_features"]},
        {
            "job-1": [
                {
                    "p99_ttft_ms": 50.0,
                    "depth_req_q": 1.0,
                    "telemetry_complete": False,
                }
            ]
        },
    )

    assert result["job-1"]["status"] == "unknown"
    assert result["job-1"]["rehabilitation_eligible"] is False


def test_incomplete_telemetry_with_zero_throughput_is_critical():
    runner = _runner()
    job = {
        "job_id": "job-1",
        "job_features": {
            "type": "online",
            "request_arrival_rate": 1.0,
            "osl_token_avg": 100,
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 20.0,
        },
    }
    snapshot = ClusterResourceSnapshot(40, {}, [job], [])

    result = runner._update_active_health(
        TickContext(tick=40, cluster_snapshot=snapshot),
        {"job-1": job["job_features"]},
        {
            "job-1": [
                {
                    "throughput_token_per_sec": 0.0,
                    "depth_req_q": 1276.0,
                    "telemetry_complete": False,
                }
            ]
        },
    )

    health = result["job-1"]
    assert health["status"] == "critical"
    assert health["rehabilitation_eligible"] is True
    assert health["observed_slo_met"] is None
    assert "zero_throughput" in health["reasons"]
    assert "telemetry_incomplete" in health["reasons"]


def test_incomplete_telemetry_with_deep_queue_is_critical():
    runner = _runner()
    job = {
        "job_id": "job-1",
        "job_features": {
            "type": "online",
            "request_arrival_rate": 1.0,
            "osl_token_avg": 100,
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 20.0,
        },
    }
    snapshot = ClusterResourceSnapshot(5, {}, [job], [])

    result = runner._update_active_health(
        TickContext(tick=5, cluster_snapshot=snapshot),
        {"job-1": job["job_features"]},
        {
            "job-1": [
                {
                    "throughput_token_per_sec": 120.0,
                    "depth_req_q": 150.0,
                    "telemetry_complete": False,
                }
            ]
        },
    )

    health = result["job-1"]
    assert health["status"] == "critical"
    assert health["rehabilitation_eligible"] is True
    assert "telemetry_incomplete" in health["reasons"]


def test_online_empty_queue_does_not_treat_low_throughput_as_capacity_failure():
    runner = _runner()
    job = {
        "job_id": "job-1",
        "job_features": {
            "type": "online",
            "request_arrival_rate": 10.0,
            "osl_token_avg": 100,
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 20.0,
        },
    }
    snapshot = ClusterResourceSnapshot(1, {}, [job], [])

    result = runner._update_active_health(
        TickContext(tick=1, cluster_snapshot=snapshot),
        {"job-1": job["job_features"]},
        {
            "job-1": [
                {
                    "throughput_token_per_sec": 0.0,
                    "p99_ttft_ms": 50.0,
                    "p99_tpot_ms": 10.0,
                    "depth_req_q": 0.0,
                    "telemetry_complete": True,
                }
            ]
        },
    )

    assert result["job-1"]["status"] == "healthy"
    assert result["job-1"]["rehabilitation_eligible"] is False


def test_batch_health_uses_deadline_pace_and_ignores_latency_queue():
    runner = _runner()
    job = {
        "job_id": "job-1",
        "kind": "batch",
        "job_features": {
            "type": "batch",
            "total_token_budget": 1000,
            "deadline_hrs": 1,
            "target_p99_ttft_ms": 1.0,
            "target_p99_tpot_ms": 1.0,
        },
    }
    snapshot = ClusterResourceSnapshot(1, {}, [job], [])

    result = runner._update_active_health(
        TickContext(tick=1, cluster_snapshot=snapshot),
        {"job-1": job["job_features"]},
        {
            "job-1": [
                {
                    "throughput_token_per_sec": 0.2,
                    "p99_ttft_ms": 10_000.0,
                    "p99_tpot_ms": 100.0,
                    "depth_req_q": 1000.0,
                    "telemetry_complete": True,
                }
            ]
        },
    )

    health = result["job-1"]
    assert health["status"] == "degraded"
    assert health["reasons"] == ["throughput_shortfall"]
