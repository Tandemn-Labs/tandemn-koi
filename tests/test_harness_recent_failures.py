import time

from koi.harness.recent_failures import (
    annotate_and_rank_rows,
    recent_failure_for_scope,
    recent_failure_penalty,
)
from koi.schemas import GPUResource, ResourceMap
from koi.tools.memory import AgenticMemory


def _resource_map():
    return ResourceMap(
        vpc_id="test",
        region="us-east-1",
        resources=[
            GPUResource(
                gpu_type="L40S",
                instance_type="g6e.12xlarge",
                gpus_per_instance=4,
                total_gpus=16,
                allocated_gpus=0,
                cost_per_instance_hour_usd=10.49,
                gpu_memory_gb=48.0,
                region="us-east-1",
                interconnect="PCIe",
            ),
            GPUResource(
                gpu_type="A100-80GB",
                instance_type="p4de.24xlarge",
                gpus_per_instance=8,
                total_gpus=16,
                allocated_gpus=0,
                cost_per_instance_hour_usd=40.96,
                gpu_memory_gb=80.0,
                region="us-east-1",
                interconnect="NVLink",
            ),
        ],
    )


def test_recent_failure_for_scope_returns_annotation():
    memory = AgenticMemory(db_path=":memory:")
    now = time.time()
    memory.record_cooloff(
        key="L40S|g6e.12xlarge|us-east-1|spot",
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        region="us-east-1",
        market="spot",
        tp=4,
        pp=1,
        dp=1,
        reason="spot preemption",
        diagnosis_code="spot_preemption",
        avoid_until=now + 600,
        source_event_id="evt-1",
    )

    signal = recent_failure_for_scope(
        memory,
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        region="us-east-1",
        market="spot",
        tp=4,
        pp=1,
        dp=1,
        now=now + 60,
    )

    assert signal is not None
    assert signal["same_scope"] is True
    assert signal["diagnosis_code"] == "spot_preemption"
    assert signal["age_minutes"] == 1.0
    assert "on_demand" in signal["recommendation"]


def test_annotate_and_rank_rows_downranks_recent_failure_but_keeps_valid():
    memory = AgenticMemory(db_path=":memory:")
    now = time.time()
    memory.record_cooloff(
        key="L40S|g6e.12xlarge|us-east-1|on_demand",
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        region="us-east-1",
        market="on_demand",
        tp=4,
        pp=1,
        dp=1,
        reason="recent no capacity",
        diagnosis_code="no_capacity",
        avoid_until=now + 600,
    )
    rows = [
        {
            "gpu_type": "L40S",
            "instance_type": "g6e.12xlarge",
            "tp": 4,
            "pp": 1,
            "dp": 1,
            "planned_market": "on_demand",
            "meets_slo": True,
            "under_cost_roofline": True,
            "total_cost": 1.0,
        },
        {
            "gpu_type": "A100-80GB",
            "instance_type": "p4de.24xlarge",
            "tp": 8,
            "pp": 1,
            "dp": 1,
            "planned_market": "on_demand",
            "meets_slo": True,
            "under_cost_roofline": True,
            "total_cost": 9.0,
        },
    ]

    ranked = annotate_and_rank_rows(
        memory,
        rows,
        _resource_map(),
        default_market="on_demand",
        now=now,
    )

    assert ranked[0]["gpu_type"] == "A100-80GB"
    assert ranked[1]["gpu_type"] == "L40S"
    assert ranked[1]["recent_failure"]["diagnosis_code"] == "no_capacity"
    assert recent_failure_penalty(ranked[1]) == 1
