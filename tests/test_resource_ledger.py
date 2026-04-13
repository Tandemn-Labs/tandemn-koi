"""Tests for koi/resource_ledger.py."""

from datetime import datetime

from koi.resource_ledger import ResourceLedger
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
