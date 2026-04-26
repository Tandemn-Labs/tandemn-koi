import pytest
from pydantic_ai.models.test import TestModel

from koi.agent import KoiAgent
from koi.harness.config import fail_open_enabled, harness_enabled, prompt_enabled
from koi.harness.p0 import build_p0_packet
from koi.schemas import GPUResource, JobRequest, ResourceMap
from koi.tools.memory import AgenticMemory


class StubPerfDB:
    def query(self, **kwargs):
        return [
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
                "throughput_tps": 2000.0,
            },
        ]


@pytest.fixture
def memory():
    return AgenticMemory(db_path=":memory:")


@pytest.fixture
def resource_map():
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


@pytest.fixture
def job_request():
    return JobRequest(
        model_name="Qwen/Qwen2.5-32B",
        avg_input_tokens=1024,
        avg_output_tokens=1024,
        num_requests=1000,
        slo_deadline_hours=2.0,
        preferred_market="on_demand",
        cost_roofline_usd=100.0,
    )


@pytest.fixture
def agent(memory):
    agent = KoiAgent(perfdb=StubPerfDB(), memory=memory, api_key="test-key")
    agent._model = TestModel()
    agent.model = "test-model"
    return agent


def test_harness_flag_helpers(monkeypatch):
    monkeypatch.delenv("KOI_HARNESS", raising=False)
    assert harness_enabled() is False
    assert prompt_enabled("p0") is False

    monkeypatch.setenv("KOI_HARNESS", "1")
    assert harness_enabled() is True
    assert prompt_enabled("p0") is True

    monkeypatch.setenv("KOI_HARNESS_PROMPTS", "pscale,p4")
    assert prompt_enabled("p0") is False
    assert prompt_enabled("p4") is True

    monkeypatch.setenv("KOI_HARNESS_FAIL_OPEN", "0")
    assert fail_open_enabled() is False


def test_p0_packet_builds_valid_physics_annotated_menu(agent, job_request, resource_map):
    packet = build_p0_packet(agent, job_request, resource_map)

    assert packet.transition_type.value == "initial_placement"
    assert len(packet.action_options) == 2
    first = packet.action_options[0]
    assert first.action_id == "a"
    assert first.valid is True
    assert first.hard_feasibility["vram_fit"] is True
    assert first.hard_feasibility["tp_heads_valid"] is True
    assert first.hard_feasibility["pp_layers_valid"] is True
    assert "vram_headroom_gb" in first.hard_feasibility
    assert "bandwidth_per_param" in first.physics
    assert "row:a" in packet.detail_sections


def test_absolute_fallback_decision_has_required_cost_field(
    agent, job_request, resource_map
):
    agent._last_cost_rows = []

    decision = agent._fallback_decision(job_request, resource_map, elapsed=0.1)

    assert decision.predicted_cost_per_hour == 0
    assert decision.predicted_tps == 0


@pytest.mark.asyncio
async def test_decide_uses_p0_harness_when_enabled(
    monkeypatch, agent, job_request, resource_map
):
    monkeypatch.setenv("KOI_HARNESS", "1")
    monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p0")

    decision = await agent.decide(job_request, resource_map)

    assert decision.config.gpu_type == "L40S"
    assert decision.config.tp == 4
    assert decision.config.pp == 1
    assert decision.planned_market == "on_demand"
    assert decision.predicted_tps == 1200.0
    assert decision.agent_model == "test-model"
    assert decision.tool_calls_made >= 0


@pytest.mark.asyncio
async def test_decide_harness_fail_closed_raises(
    monkeypatch, agent, job_request, resource_map
):
    import koi.harness.p0 as p0

    async def boom(*args, **kwargs):
        raise RuntimeError("p0 exploded")

    monkeypatch.setattr(p0, "run_initial_placement", boom)
    monkeypatch.setenv("KOI_HARNESS", "1")
    monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p0")
    monkeypatch.setenv("KOI_HARNESS_FAIL_OPEN", "0")

    with pytest.raises(RuntimeError, match="p0 exploded"):
        await agent.decide(job_request, resource_map)
