from types import SimpleNamespace
import time

import pytest
from pydantic_ai.models.test import TestModel

from koi.agent import KoiAgent
from koi.harness.p1 import build_p1_packet, run_launch_recovery
from koi.resource_ledger import ResourceLedger
from koi.tools.memory import AgenticMemory


class StubPerfDB:
    _rows = [
        {
            "gpu_type": "L40S",
            "instance_type": "g6e.12xlarge",
            "tp": 4,
            "pp": 1,
            "dp": 1,
            "throughput_tps": 1200.0,
        },
        {
            "gpu_type": "A100-80GB",
            "instance_type": "p4de.24xlarge",
            "tp": 8,
            "pp": 1,
            "dp": 1,
            "throughput_tps": 2400.0,
        },
    ]

    def query(self, **kwargs):
        records = list(self._rows)
        gpu_type = kwargs.get("gpu_type")
        tp = kwargs.get("tp")
        pp = kwargs.get("pp")
        if gpu_type:
            records = [row for row in records if row["gpu_type"] == gpu_type]
        if tp is not None:
            records = [row for row in records if row["tp"] == tp]
        if pp is not None:
            records = [row for row in records if row["pp"] == pp]
        return records[: kwargs.get("limit", 20)]


class FakeOrca:
    async def get_resources(self):
        return {
            "instances": [
                {
                    "instance_type": "g6e.12xlarge",
                    "gpu_type": "L40S",
                    "gpus_per_instance": 4,
                    "vcpus": 48,
                    "quota_family": "G",
                    "gpu_memory_gb": 48.0,
                    "cost_per_instance_hour_usd": 10.49,
                    "interconnect": "PCIe",
                },
                {
                    "instance_type": "p4de.24xlarge",
                    "gpu_type": "A100-80GB",
                    "gpus_per_instance": 8,
                    "vcpus": 96,
                    "quota_family": "P",
                    "gpu_memory_gb": 80.0,
                    "cost_per_instance_hour_usd": 40.96,
                    "interconnect": "NVLink",
                },
            ],
            "quotas": [
                {"family": "G", "region": "us-east-1", "market": "on_demand", "baseline_vcpus": 384, "used_vcpus": 0},
                {"family": "P", "region": "us-east-1", "market": "on_demand", "baseline_vcpus": 384, "used_vcpus": 0},
            ],
        }


@pytest.fixture
def memory():
    return AgenticMemory(db_path=":memory:")


@pytest.fixture
def agent(memory):
    agent = KoiAgent(perfdb=StubPerfDB(), memory=memory, orca=FakeOrca(), api_key="test-key")
    agent._model = TestModel()
    agent.model = "test-model"
    return agent


def _record_parent(memory: AgenticMemory) -> str:
    return memory.record_decision(
        job_id="job-p1",
        model_name="Qwen/Qwen3-32B",
        instance_type="g6e.12xlarge",
        gpu_type="L40S",
        tp=4,
        pp=1,
        dp=1,
        num_gpus=4,
        predicted_tps=1200.0,
        predicted_cost_per_hour=10.49,
        slo_deadline_hours=1.0,
        objective="cheapest",
        avg_input_tokens=1024,
        avg_output_tokens=1024,
        num_requests=1500,
        market="spot",
        cost_roofline_usd=100.0,
    )


def _launch_failed(decision_id: str):
    return SimpleNamespace(
        job_id="job-p1",
        decision_id=decision_id,
        configs_tried=[
            {
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "dp": 1,
                "region": "us-east-1",
                "market": "spot",
            }
        ],
        failure_reasons=["InsufficientCapacity"],
        total_time_seconds=60.0,
    )


@pytest.mark.asyncio
async def test_p1_packet_builds_market_and_gpu_recovery_menu(agent, memory):
    decision_id = _record_parent(memory)

    packet = await build_p1_packet(agent, _launch_failed(decision_id), memory)

    assert packet.transition_type.value == "launch_recovery"
    assert packet.state.value == "launch_failed"
    assert packet.policy_context["retry_budget_remaining_before_choice"] == 2
    sources = [option.evidence.get("source") for option in packet.action_options]
    assert "market_alternate" in sources
    assert "gpu_family_alternate" in sources
    assert packet.action_options[-1].action_type == "abort_launch"
    first = packet.action_options[0]
    assert f"failure:{first.action_id}" in first.detail_refs
    assert f"executor_payload:{first.action_id}" in packet.detail_sections


@pytest.mark.asyncio
async def test_p1_recent_failure_downranks_same_scope_recovery(agent, memory):
    decision_id = _record_parent(memory)
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

    packet = await build_p1_packet(agent, _launch_failed(decision_id), memory)

    assert packet.action_options[0].evidence["source"] == "gpu_family_alternate"
    l40s = next(
        option
        for option in packet.action_options
        if option.evidence.get("source") == "market_alternate"
    )
    assert l40s.valid is True
    assert l40s.risk["recent_failure"]["diagnosis_code"] == "no_capacity"
    assert packet.detail_sections[f"recent_failures:{l40s.action_id}"]["recent_failure"]["same_scope"] is True


@pytest.mark.asyncio
async def test_p1_retry_budget_exhausted_emits_only_abort(agent, memory):
    parent = _record_parent(memory)
    child = memory.record_decision(
        job_id="job-p1",
        model_name="Qwen/Qwen3-32B",
        instance_type="p4de.24xlarge",
        gpu_type="A100-80GB",
        tp=8,
        pp=1,
        dp=1,
        num_gpus=8,
        predicted_tps=2400.0,
        predicted_cost_per_hour=40.96,
        slo_deadline_hours=1.0,
        objective="cheapest",
        avg_input_tokens=1024,
        avg_output_tokens=1024,
        num_requests=1500,
        triggered_by="launch_recovery",
        parent_decision_id=parent,
        market="on_demand",
    )

    packet = await build_p1_packet(
        agent,
        _launch_failed(child),
        memory,
        retry_budget=1,
    )

    assert packet.evidence_summary["valid_recovery_count"] == 0
    assert [option.action_type for option in packet.valid_actions()] == ["abort_launch"]
    assert "retry budget exhausted" in packet.action_options[0].summary


@pytest.mark.asyncio
async def test_run_launch_recovery_records_child_decision_and_reserves(agent, memory):
    decision_id = _record_parent(memory)
    ledger = ResourceLedger()

    plan = await run_launch_recovery(
        agent,
        _launch_failed(decision_id),
        memory,
        ledger=ledger,
    )

    assert plan["action"] == "retry_launch"
    assert plan["decision_id"] != decision_id
    child = memory.get_decision(plan["decision_id"])
    assert child["parent_decision_id"] == decision_id
    assert child["triggered_by"] == "launch_recovery"
    assert plan["config"]["market"] == "on_demand"
    assert ledger.pending_count == 1
