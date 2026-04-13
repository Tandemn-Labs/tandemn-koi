from koi.runtime_state import RuntimeStateStore
from koi.schemas import EngineConfig, JobTracker, PlacementConfig


def _sample_tracker(job_id: str = "job-1", decision_id: str = "dec-1", group_id: str = "grp-1") -> JobTracker:
    return JobTracker(
        job_id=job_id,
        decision_id=decision_id,
        group_id=group_id,
        config=PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-west-2",
            market="spot",
            engine_config=EngineConfig(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
            ),
        ),
        slo_deadline_hours=2.0,
        total_tokens=12345,
        predicted_tps=900.0,
    )


def test_store_round_trips_tracked_job():
    store = RuntimeStateStore(":memory:")
    tracker = _sample_tracker()

    store.upsert_tracked_job(tracker.job_id, tracker.model_dump(mode="json"))

    loaded = store.load_tracked_jobs()
    assert set(loaded) == {"job-1"}
    assert loaded["job-1"]["group_id"] == "grp-1"
    assert loaded["job-1"]["decision_id"] == "dec-1"
    assert loaded["job-1"]["tracker"]["config"]["gpu_type"] == "L40S"
    assert loaded["job-1"]["tracker"]["total_tokens"] == 12345


def test_store_round_trips_pending_launch():
    store = RuntimeStateStore(":memory:")
    launch = {
        "group_id": "grp-1",
        "gpu_type": "L40S",
        "instance_type": "g6e.12xlarge",
        "region": "us-west-2",
        "market": "spot",
        "launched_at": 123.4,
    }

    store.upsert_pending_launch("replica-1", launch)

    loaded = store.load_pending_launches()
    assert loaded["replica-1"]["launch"] == launch


def test_store_replaces_pending_scale_group_in_order():
    store = RuntimeStateStore(":memory:")
    initial = [
        {"decision_id": "dec-a", "remaining": 1},
        {"decision_id": "dec-b", "remaining": 2},
    ]
    replacement = [{"decision_id": "dec-c", "remaining": 3}]

    store.replace_pending_scale_group("grp-1", initial)
    assert store.load_pending_scale_decisions()["grp-1"] == initial

    store.replace_pending_scale_group("grp-1", replacement)
    assert store.load_pending_scale_decisions()["grp-1"] == replacement


def test_store_round_trips_ledger_reservation():
    store = RuntimeStateStore(":memory:")
    reservation = {
        "gpu_type": "L40S",
        "num_gpus": 8,
        "cloud": "aws",
        "region": "us-west-2",
        "instance_type": "g6e.24xlarge",
        "tenant_id": "default",
        "decision_id": "dec-123",
        "created_at": 111.2,
    }

    store.upsert_ledger_reservation("dec-123", reservation, expires_at=999.9)

    loaded = store.load_ledger_reservations()
    assert loaded["dec-123"]["reservation"] == reservation
    assert loaded["dec-123"]["expires_at"] == 999.9
