"""Tests for koi/agent.py — prompt building, tool wiring, response parsing."""

import pytest
from koi.agent import KoiAgent, KOI_SYSTEM_PROMPT
from koi.schemas import (
    JobRequest, ResourceMap, GPUResource, EngineConfig, PlacementConfig,
    MonitoringStatus, MonitoringTrigger, DataSource,
)
from koi.tools.perfdb import PerfDB
from koi.tools.memory import AgenticMemory

CSV_PATH = "perfdb/perfdb_all.csv"


@pytest.fixture
def perfdb():
    return PerfDB(CSV_PATH)


@pytest.fixture
def memory():
    return AgenticMemory(db_path=":memory:")


@pytest.fixture
def resource_map():
    return ResourceMap(
        vpc_id="test", region="us-east-1",
        resources=[
            GPUResource(gpu_type="L40S", instance_type="g6e.12xlarge",
                        gpus_per_instance=4, total_gpus=80, allocated_gpus=0,
                        cost_per_instance_hour_usd=10.49, gpu_memory_gb=48.0,
                        region="us-east-1", interconnect="PCIe"),
            GPUResource(gpu_type="A100-80GB", instance_type="p4de.24xlarge",
                        gpus_per_instance=8, total_gpus=32, allocated_gpus=0,
                        cost_per_instance_hour_usd=40.96, gpu_memory_gb=80.0,
                        region="us-west-2", interconnect="NVLink"),
        ],
    )


@pytest.fixture
def job_request():
    return JobRequest(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        avg_input_tokens=953, avg_output_tokens=1024,
        num_requests=5000, slo_deadline_hours=8.0,
        objective="cheapest",
    )


@pytest.fixture
def agent(perfdb, memory):
    return KoiAgent(perfdb=perfdb, memory=memory, api_key="test-key")


class TestSystemPrompt:
    def test_has_key_instructions(self):
        assert "CHECK MEMORY FIRST" in KOI_SYSTEM_PROMPT
        assert "throughput_tokens_per_sec" in KOI_SYSTEM_PROMPT
        assert "A100-40GB" in KOI_SYSTEM_PROMPT
        assert "FALLING_BEHIND" in KOI_SYSTEM_PROMPT


class TestToolWiring:
    def test_build_tools_count(self, agent, resource_map):
        tools = agent._build_tools(resource_map=resource_map)
        assert len(tools) == 8  # query_perfdb, query_memory, gpu_physics, model_arch, similar_models, resources, record_outcome

    def test_build_tools_without_resource_map(self, agent):
        tools = agent._build_tools()
        assert len(tools) == 8


class TestPromptBuilding:
    def test_decide_prompt(self, agent, job_request, resource_map):
        prompt = agent._build_decide_prompt(job_request, resource_map)
        assert "Qwen/Qwen2.5-72B-Instruct" in prompt
        assert "5,000" in prompt or "5000" in prompt
        assert "8.0h" in prompt
        assert "cheapest" in prompt
        assert "L40S" in prompt
        assert "A100-80GB" in prompt
        assert "gpu_type" in prompt  # JSON schema

    def test_decide_prompt_has_required_tps(self, agent, job_request, resource_map):
        prompt = agent._build_decide_prompt(job_request, resource_map)
        # 5000 * (953+1024) / (8*3600) ≈ 343
        assert "343" in prompt or "Required TPS" in prompt

    def test_trigger_prompt_falling_behind(self, agent):
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.FALLING_BEHIND,
            job_id="job-test",
            job_tracker={"smoothed_tps": 200, "slo_headroom_pct": 5.0,
                         "elapsed_hours": 2.0, "tokens_remaining": 5_000_000,
                         "gpu_cache_usage": 0.85, "gpu_sm_util": 90, "gpu_mem_bw_util": 95},
            diagnosis_hint="Throughput dropped below SLO requirement",
        )
        prompt = agent._build_trigger_prompt(trigger)
        assert "falling_behind" in prompt
        assert "scale_chain_tool" in prompt
        assert "Throughput dropped" in prompt

    def test_trigger_prompt_completed(self, agent):
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.COMPLETED,
            job_id="job-done",
            job_tracker={"smoothed_tps": 1500, "slo_headroom_pct": 80.0,
                         "elapsed_hours": 1.5, "tokens_remaining": 0},
        )
        prompt = agent._build_trigger_prompt(trigger)
        assert "completed" in prompt
        assert "record_outcome" in prompt.lower() or "Record the outcome" in prompt


class TestParseDecision:
    def test_parse_json_block(self, agent, job_request, resource_map):
        text = """Based on my analysis, here's the placement:

```json
{
  "gpu_type": "A100-80GB",
  "instance_type": "p4de.24xlarge",
  "tp": 4,
  "pp": 2,
  "dp": 1,
  "predicted_tps": 2590.0,
  "predicted_cost_per_hour": 40.96,
  "reasoning": "PerfDB shows A100-80GB TP=4 PP=2 gets 2590 TPS",
  "confidence": 0.88,
  "data_source": "perfdb_exact"
}
```"""
        decision = agent._parse_decision(text, job_request, resource_map, tool_calls=5, elapsed=12.3)
        assert decision.config.gpu_type == "A100-80GB"
        assert decision.config.tp == 4
        assert decision.config.pp == 2
        assert decision.predicted_tps == 2590.0
        assert decision.confidence == 0.88
        assert decision.data_source == DataSource.EXACT_MATCH
        assert decision.tool_calls_made == 5
        assert decision.latency_seconds == 12.3

    def test_parse_raw_json(self, agent, job_request, resource_map):
        text = 'I recommend {"gpu_type": "L40S", "instance_type": "g6e.12xlarge", "tp": 4, "pp": 2, "dp": 1, "predicted_tps": 833.0, "confidence": 0.75, "reasoning": "cheapest option"}'
        decision = agent._parse_decision(text, job_request, resource_map, tool_calls=3, elapsed=8.0)
        assert decision.config.gpu_type == "L40S"
        assert decision.predicted_tps == 833.0

    def test_parse_fallback_on_no_json(self, agent, job_request, resource_map):
        text = "I suggest using L40S with TP=4 but I couldn't format JSON properly."
        decision = agent._parse_decision(text, job_request, resource_map, tool_calls=2, elapsed=5.0)
        # Should fall back to defaults without crashing
        assert decision.config.gpu_type == "L40S"  # default
        assert decision.confidence == 0.5  # default
