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
