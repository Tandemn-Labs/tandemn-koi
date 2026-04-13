"""Tests for koi/resource_ledger.py."""

from datetime import datetime

from koi.resource_ledger import ResourceLedger
from koi.runtime_state import RuntimeStateStore
from koi.schemas import GPUResource, ResourceMap


def test_apply_to_resource_map_scopes_pending_by_region():
    ledger = ResourceLedger()
    ledger.reserve("dec-east", "L40S", 4, region="us-east-1", cloud="aws")

    base = ResourceMap(
        vpc_id="orca-cluster",
        region="multi-region",
        snapshot_time=datetime.utcnow(),
        resources=[
            GPUResource(
                gpu_type="L40S",
                instance_type="g6e.12xlarge",
                gpus_per_instance=4,
                total_gpus=8,
                allocated_gpus=0,
                cost_per_instance_hour_usd=10.49,
                gpu_memory_gb=48.0,
                region="us-east-1",
                interconnect="PCIe",
                cloud="aws",
            ),
            GPUResource(
                gpu_type="L40S",
                instance_type="g6e.12xlarge",
                gpus_per_instance=4,
                total_gpus=8,
                allocated_gpus=0,
                cost_per_instance_hour_usd=10.49,
                gpu_memory_gb=48.0,
                region="us-west-2",
                interconnect="PCIe",
                cloud="aws",
            ),
        ],
    )

    adjusted = ledger.apply_to_resource_map(base)

    east = next(r for r in adjusted.resources if r.region == "us-east-1")
    west = next(r for r in adjusted.resources if r.region == "us-west-2")
    assert east.allocated_gpus == 4
    assert west.allocated_gpus == 0


def test_ledger_reservation_persists_across_restart(tmp_path):
    db_path = tmp_path / "runtime.sqlite"

    store1 = RuntimeStateStore(str(db_path))
    ledger1 = ResourceLedger(runtime_state=store1)
    ledger1.reserve(
        "dec-123",
        "L40S",
        8,
        region="us-west-2",
        cloud="aws",
        instance_type="g6e.24xlarge",
    )

    store2 = RuntimeStateStore(str(db_path))
    ledger2 = ResourceLedger(runtime_state=store2)
    assert ledger2.restore() == 1
    assert ledger2.get_pending_by_type(region="us-west-2", gpu_type="L40S") == {"L40S": 8}

    released = ledger2.release("dec-123")
    assert released is not None

    store3 = RuntimeStateStore(str(db_path))
    assert store3.load_ledger_reservations() == {}


def test_restore_drops_expired_persisted_reservations(tmp_path):
    db_path = tmp_path / "runtime.sqlite"

    store = RuntimeStateStore(str(db_path))
    store.upsert_ledger_reservation(
        "dec-expired",
        {
            "gpu_type": "L40S",
            "num_gpus": 4,
            "cloud": "aws",
            "region": "us-west-2",
            "tenant_id": "default",
            "instance_type": "g6e.12xlarge",
            "decision_id": "dec-expired",
            "created_at": 1.0,
        },
        expires_at=2.0,
    )

    ledger = ResourceLedger(runtime_state=RuntimeStateStore(str(db_path)))
    assert ledger.restore() == 0
    assert ledger.pending_count == 0
    assert RuntimeStateStore(str(db_path)).load_ledger_reservations() == {}
