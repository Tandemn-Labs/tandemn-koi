from types import SimpleNamespace

import pytest
from pydantic_ai.models.test import TestModel

from koi.agent import KoiAgent
from koi.harness.pscale import (
    _execute_validated_action,
    _packet_tools,
    build_pscale_packet,
    render_pscale_prompt,
    run_runtime_scale,
)
from koi.schemas import EngineConfig, MonitoringStatus, MonitoringTrigger, PlacementConfig
from koi.tools.memory import AgenticMemory


@pytest.fixture
def memory():
    return AgenticMemory(db_path=":memory:")


class FakeScaleOrca:
    def __init__(self):
        self.scale_calls = []

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
                }
            ],
            "quotas": [
                {
                    "family": "G",
                    "region": "us-east-1",
                    "market": "on_demand",
                    "baseline_vcpus": 384,
                    "used_vcpus": 0,
                }
            ],
        }

    async def scale_job(self, job_id, gpu_type, tp, pp, count, **kwargs):
        self.scale_calls.append(
            {
                "job_id": job_id,
                "gpu_type": gpu_type,
                "tp": tp,
                "pp": pp,
                "count": count,
                **kwargs,
            }
        )
        return {
            "status": "scaling",
            "new_replicas": [f"{job_id}-r-new"],
            "scale_request_id": "scale-1",
        }


class FakeKillOrca:
    def __init__(self):
        self.kill_calls = []

    async def kill_replicas(self, job_id, replica_ids):
        self.kill_calls.append({"job_id": job_id, "replica_ids": replica_ids})
        return {"status": "killed"}


class FakeMonitor:
    def __init__(self, chains=None):
        self._chains = chains or {}
        self._koi_initiated_kills = set()
        self.tracked_jobs = dict(self._chains)
        self.pending_decisions = []

    def get_group_chains(self, group_id):
        return self._chains

    def persist_job(self, job_id):
        return None

    def register_pending_replica_decision(self, **kwargs):
        self.pending_decisions.append(kwargs)


def _config(gpu_type="L40S", tp=4, pp=1):
    return PlacementConfig(
        gpu_type=gpu_type,
        instance_type="g6e.12xlarge",
        num_gpus=tp * pp,
        num_instances=1,
        tp=tp,
        pp=pp,
        dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        market="on_demand",
    )


def _agent(memory, orca=None, monitor=None):
    agent = KoiAgent(perfdb=None, memory=memory, orca=orca, api_key="test-key")
    agent._model = TestModel()
    agent.model = "test-model"
    agent.monitor = monitor
    return agent


def _falling_trigger():
    return MonitoringTrigger(
        trigger_type=MonitoringStatus.FALLING_BEHIND,
        job_id="group-job",
        job_tracker={
            "job_id": "group-job",
            "group_id": "group-job",
            "config": _config().model_dump(mode="json"),
            "predicted_tps": 600.0,
            "predicted_cost_per_hour": 10.49,
            "smoothed_tps": 300.0,
            "slo_deadline_hours": 1.0,
            "elapsed_hours": 0.5,
            "tokens_remaining": 1_500_000,
            "slo_headroom_pct": -10.0,
        },
        diagnosis_hint="Behind SLO",
    )


def _overprov_trigger():
    return MonitoringTrigger(
        trigger_type=MonitoringStatus.OVER_PROVISIONED,
        job_id="r-slow",
        job_tracker={
            "job_id": "r-slow",
            "group_id": "group-job",
            "config": _config().model_dump(mode="json"),
            "predicted_tps": 600.0,
            "predicted_cost_per_hour": 10.49,
            "smoothed_tps": 600.0,
            "slo_deadline_hours": 2.0,
            "elapsed_hours": 0.5,
            "tokens_remaining": 500_000,
            "slo_headroom_pct": 80.0,
        },
        diagnosis_hint="Can shed replicas",
    )


@pytest.mark.asyncio
async def test_pscale_packet_builds_scale_up_menu(memory):
    agent = _agent(memory)
    packet = await build_pscale_packet(agent, _falling_trigger())

    assert packet.transition_type.value == "scale"
    assert packet.state.value == "degraded"
    assert packet.action_options[0].action_type == "scale_up"
    assert packet.action_options[0].valid is True
    assert packet.action_options[-1].action_type == "noop"
    assert packet.action_options[-1].valid is False


@pytest.mark.asyncio
async def test_pscale_packet_exposes_granular_detail_sections(memory):
    agent = _agent(memory)
    packet = await build_pscale_packet(agent, _falling_trigger())

    scale_option = packet.action_options[0]
    expected = {
        f"physics:{scale_option.action_id}",
        f"perfdb_exact:{scale_option.action_id}",
        f"perfdb_proxy:{scale_option.action_id}",
        f"memory_success:{scale_option.action_id}",
        f"memory_failure:{scale_option.action_id}",
        f"quota:{scale_option.action_id}",
        f"recent_failures:{scale_option.action_id}",
        f"runtime_metrics:{scale_option.action_id}",
        f"executor_payload:{scale_option.action_id}",
        f"suggestion:{scale_option.action_id}",
    }
    assert set(scale_option.detail_refs) == expected
    assert all(ref in packet.detail_sections for ref in expected)
    physics = packet.detail_sections[f"physics:{scale_option.action_id}"]
    assert physics["gpu_type"] == "L40S"
    assert physics["tp"] == 4

    tools = _packet_tools(agent, packet)
    listing = await tools["list_detail_sections"](scale_option.action_id)
    assert "physics:" in listing
    by_section = await tools["read_option_detail"](
        scale_option.action_id, section="physics"
    )
    assert "physics:" in by_section
    assert "L40S" in by_section
    invalid = await tools["read_option_detail"](
        scale_option.action_id, section="bogus"
    )
    assert "unknown section" in invalid


@pytest.mark.asyncio
async def test_pscale_exposes_read_tools_and_custom_scale_option(memory):
    orca = FakeScaleOrca()
    monitor = FakeMonitor()
    agent = _agent(memory, orca=orca, monitor=monitor)
    packet = await build_pscale_packet(agent, _falling_trigger())
    tools = _packet_tools(agent, packet)

    assert "read_option_detail" in tools
    assert "get_resources_tool" in tools
    assert "request_custom_scale_option" in tools
    assert "scale_chain_tool" not in tools
    assert "kill_replica_tool" not in tools

    result = await tools["request_custom_scale_option"](
        gpu_type="A100-80GB",
        tp=8,
        pp=1,
        count=1,
        on_demand=True,
        reason="explored faster GPU family",
    )

    assert "x1" in result
    custom = packet.get_action("x1")
    assert custom is not None
    assert custom.action_type == "scale_up"
    assert custom.evidence["source"] == "llm_exploration"

    exec_result = await _execute_validated_action(agent, packet, "x1")
    assert "Scaled up" in exec_result
    assert orca.scale_calls[0]["gpu_type"] == "A100-80GB"
    assert orca.scale_calls[0]["on_demand"] is True


@pytest.mark.asyncio
async def test_pscale_prompt_explicitly_allows_bounded_exploration(memory):
    packet = await build_pscale_packet(_agent(memory), _falling_trigger())
    prompt = render_pscale_prompt(packet)

    assert "explore alternatives" in prompt
    assert "request_custom_scale_option" in prompt
    assert "Do not execute raw cluster mutations directly" in prompt


@pytest.mark.asyncio
async def test_pscale_executes_scale_up_choice(memory):
    orca = FakeScaleOrca()
    monitor = FakeMonitor()
    agent = _agent(memory, orca=orca, monitor=monitor)

    result = await run_runtime_scale(agent, _falling_trigger())

    assert "Scaled up" in result
    assert orca.scale_calls[0]["job_id"] == "group-job"
    assert orca.scale_calls[0]["gpu_type"] == "L40S"
    assert orca.scale_calls[0]["count"] == 1


@pytest.mark.asyncio
async def test_pscale_executes_safe_kill_choice(memory):
    orca = FakeKillOrca()
    chains = {
        "r-fast": SimpleNamespace(
            job_id="r-fast",
            group_id="group-job",
            config=_config(),
            status=MonitoringStatus.ON_TRACK,
            smoothed_tps=1800.0,
            predicted_tps=1800.0,
            predicted_cost_per_hour=10.49,
            action_in_progress=False,
            action_freeze_until=None,
        ),
        "r-slow": SimpleNamespace(
            job_id="r-slow",
            group_id="group-job",
            config=_config(),
            status=MonitoringStatus.ON_TRACK,
            smoothed_tps=300.0,
            predicted_tps=300.0,
            predicted_cost_per_hour=10.49,
            action_in_progress=False,
            action_freeze_until=None,
        ),
    }
    monitor = FakeMonitor(chains)
    agent = _agent(memory, orca=orca, monitor=monitor)

    result = await run_runtime_scale(agent, _overprov_trigger())

    assert "Killed 1 replicas" in result
    assert orca.kill_calls[0] == {"job_id": "group-job", "replica_ids": ["r-slow"]}
    assert "r-slow" in monitor._koi_initiated_kills
    scale_downs = [
        dec
        for dec in memory.query_decisions(limit=10)
        if dec.get("triggered_by") == "scale_down"
    ]
    assert len(scale_downs) == 1


@pytest.mark.asyncio
async def test_handle_trigger_uses_pscale_when_enabled(monkeypatch, memory):
    orca = FakeScaleOrca()
    agent = _agent(memory, orca=orca, monitor=FakeMonitor())
    monkeypatch.setenv("KOI_HARNESS", "1")
    monkeypatch.setenv("KOI_HARNESS_PROMPTS", "pscale")

    result = await agent.handle_trigger(_falling_trigger())

    assert "Scaled up" in result
    assert orca.scale_calls
