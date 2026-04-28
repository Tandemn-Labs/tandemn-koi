"""Tests for koi/agent.py — prompt building, tool wiring, response parsing."""

import pytest
from types import SimpleNamespace
from koi.agent import KoiAgent, KOI_SYSTEM_PROMPT


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    # Keep LLM env vars off so KoiAgent construction in tests relies purely on
    # the explicit `api_key="test-key"` kwarg + default "openrouter" provider.
    for var in (
        "KOI_LLM_PROVIDER",
        "KOI_BASE_URL",
        "KOI_AGENT_MODEL",
        "KOI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
from koi.schemas import (
    JobRequest,
    JobTracker,
    ResourceMap,
    GPUResource,
    EngineConfig,
    PlacementConfig,
    MonitoringStatus,
    MonitoringTrigger,
    DataSource,
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
        vpc_id="test",
        region="us-east-1",
        resources=[
            GPUResource(
                gpu_type="L40S",
                instance_type="g6e.12xlarge",
                gpus_per_instance=4,
                total_gpus=80,
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
                total_gpus=32,
                allocated_gpus=0,
                cost_per_instance_hour_usd=40.96,
                gpu_memory_gb=80.0,
                region="us-west-2",
                interconnect="NVLink",
            ),
        ],
    )


@pytest.fixture
def job_request():
    return JobRequest(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        avg_input_tokens=953,
        avg_output_tokens=1024,
        num_requests=5000,
        slo_deadline_hours=8.0,
        objective="cheapest",
        preferred_market="on_demand",
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
        assert (
            len(tools) == 8
        )  # query_perfdb, query_memory, gpu_physics, model_arch, similar_models, resources, record_outcome

    def test_build_tools_without_resource_map(self, agent):
        tools = agent._build_tools()
        assert len(tools) == 8


class TestActionTools:
    @pytest.mark.asyncio
    async def test_get_quota_status_tool_treats_missing_rows_as_zero(
        self, perfdb, memory
    ):
        class FakeOrca:
            async def get_resources(self):
                return {
                    "instances": [
                        {
                            "instance_type": "g6.2xlarge",
                            "gpu_type": "L4",
                            "gpus_per_instance": 1,
                            "vcpus": 8,
                            "quota_family": "G",
                            "gpu_memory_gb": 24.0,
                            "cost_per_instance_hour_usd": 1.0,
                            "interconnect": "PCIe",
                        },
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 96,
                            "used_vcpus": 0,
                        },
                    ],
                }

        agent = KoiAgent(
            perfdb=perfdb, memory=memory, orca=FakeOrca(), api_key="test-key"
        )
        tools = agent._build_tools(monitor=None)
        quota_tool = tools["get_quota_status_tool"]

        result = await quota_tool(gpu_type="L4")

        assert "on_demand" in result
        assert "no rows returned -> treat all unlisted spot quota as ZERO" in result

    @pytest.mark.asyncio
    async def test_scale_chain_tool_does_not_freeze_or_record_on_confirm(
        self, perfdb, memory, resource_map
    ):
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
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 96,
                            "used_vcpus": 0,
                        },
                    ],
                }

            async def scale_job(self, *args, **kwargs):
                return {"status": "confirm", "message": "Config may not be feasible"}

        agent = KoiAgent(
            perfdb=perfdb, memory=memory, orca=FakeOrca(), api_key="test-key"
        )
        parent_decision = memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=10.49,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            market="on_demand",
        )
        tracker = SimpleNamespace(
            group_id="parent-job",
            job_id="parent-job-r0",
            decision_id=parent_decision,
            action_in_progress=False,
            action_freeze_until=None,
        )
        monitor = SimpleNamespace(
            tracked_jobs={"parent-job-r0": tracker},
            _koi_initiated_kills=set(),
            _pending_replica_decisions={},
        )
        tools = agent._build_tools(resource_map=resource_map, monitor=monitor)
        scale_tool = tools["scale_chain_tool"]

        result = await scale_tool(
            job_id="parent-job", gpu_type="L40S", tp=4, pp=1, count=1
        )

        assert "Scale not started" in result
        assert memory.decision_count() == 1
        assert tracker.action_in_progress is False
        assert tracker.action_freeze_until is None
        # Failed scale must not register phantom replica_id mappings.
        assert monitor._pending_replica_decisions == {}

    @pytest.mark.asyncio
    async def test_scale_chain_recovery_mode_passes_force_true_to_orca(
        self, perfdb, memory, resource_map
    ):
        """recovery_mode=True is the boundary signal that this tool-set is
        being used for cold-start failure recovery. scale_chain_tool must
        thread force=True down to orca.scale_job so Orca's feasibility
        check is bypassed (the agent has already overridden the solver's
        recommendation that just OOMed)."""
        captured = {}

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
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 96,
                            "used_vcpus": 0,
                        },
                    ],
                }

            async def scale_job(self, *args, **kwargs):
                captured.update(kwargs)
                return {"status": "scaling", "new_replicas": ["new-r0"]}

        agent = KoiAgent(
            perfdb=perfdb, memory=memory, orca=FakeOrca(), api_key="test-key"
        )
        parent_decision = memory.record_decision(
            job_id="parent-rec",
            model_name="Qwen/Qwen3-4B",
            instance_type="g6.xlarge",
            gpu_type="L4",
            tp=1,
            pp=1,
            dp=1,
            num_gpus=1,
            predicted_tps=1400.0,
            predicted_cost_per_hour=2.62,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            market="on_demand",
        )
        tracker = SimpleNamespace(
            group_id="parent-rec",
            job_id="parent-rec-r0",
            decision_id=parent_decision,
            action_in_progress=False,
            action_freeze_until=None,
        )
        monitor = SimpleNamespace(
            tracked_jobs={"parent-rec-r0": tracker},
            _koi_initiated_kills=set(),
            _pending_replica_decisions={},
            persist_job=lambda *a, **k: None,
            register_pending_replica_decision=lambda *a, **k: None,
        )
        # recovery_mode=True is the boundary flag set by
        # recover_from_startup_failure when invoking the agent.
        tools = agent._build_tools(
            resource_map=resource_map, monitor=monitor, recovery_mode=True
        )
        scale_tool = tools["scale_chain_tool"]

        result = await scale_tool(
            job_id="parent-rec", gpu_type="L40S", tp=1, pp=1, count=1
        )

        assert "Scaled up" in result
        # The boundary signal must reach Orca so its feasibility check is skipped.
        assert captured.get("force") is True
        assert captured.get("planned_market") in ("on_demand", "spot")

    @pytest.mark.asyncio
    async def test_scale_chain_default_mode_passes_force_false(
        self, perfdb, memory, resource_map
    ):
        """Default behavior (no recovery_mode flag) keeps force=False so
        the runtime trigger path retains its feasibility safety net."""
        captured = {}

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
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 96,
                            "used_vcpus": 0,
                        },
                    ],
                }

            async def scale_job(self, *args, **kwargs):
                captured.update(kwargs)
                return {"status": "scaling"}

        agent = KoiAgent(
            perfdb=perfdb, memory=memory, orca=FakeOrca(), api_key="test-key"
        )
        parent_decision = memory.record_decision(
            job_id="parent-norm",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=10.49,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            market="on_demand",
        )
        tracker = SimpleNamespace(
            group_id="parent-norm",
            job_id="parent-norm-r0",
            decision_id=parent_decision,
            action_in_progress=False,
            action_freeze_until=None,
        )
        monitor = SimpleNamespace(
            tracked_jobs={"parent-norm-r0": tracker},
            _koi_initiated_kills=set(),
            _pending_replica_decisions={},
            persist_job=lambda *a, **k: None,
            register_pending_replica_decision=lambda *a, **k: None,
        )
        # Default tools — no recovery_mode flag
        tools = agent._build_tools(resource_map=resource_map, monitor=monitor)
        scale_tool = tools["scale_chain_tool"]

        await scale_tool(
            job_id="parent-norm", gpu_type="L40S", tp=4, pp=1, count=1
        )

        assert captured.get("force") is False

    @pytest.mark.asyncio
    async def test_scale_chain_defaults_on_demand_and_avoids_spot_without_quota(
        self, perfdb, memory
    ):
        class FakeOrca:
            def __init__(self):
                self.calls = []

            async def get_resources(self):
                return {
                    "instances": [
                        {
                            "instance_type": "g6.2xlarge",
                            "gpu_type": "L4",
                            "gpus_per_instance": 1,
                            "vcpus": 8,
                            "quota_family": "G",
                            "gpu_memory_gb": 24.0,
                            "cost_per_instance_hour_usd": 1.0,
                            "interconnect": "PCIe",
                        },
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 96,
                            "used_vcpus": 0,
                        },
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "spot",
                            "baseline_vcpus": 0,
                            "used_vcpus": 0,
                        },
                    ],
                }

            async def scale_job(self, *args, **kwargs):
                self.calls.append(kwargs)
                return {"status": "scaling"}

        fake_orca = FakeOrca()
        agent = KoiAgent(
            perfdb=perfdb, memory=memory, orca=fake_orca, api_key="test-key"
        )
        tools = agent._build_tools(monitor=None)
        scale_tool = tools["scale_chain_tool"]

        # No parent decision + no override should default to on-demand.
        await scale_tool(job_id="job-default", gpu_type="L4", tp=1, pp=1, count=1)
        assert fake_orca.calls[-1]["on_demand"] is True

        # Explicit spot request should still switch to on-demand when spot quota is zero.
        result = await scale_tool(
            job_id="job-spot", gpu_type="L4", tp=1, pp=1, count=1, on_demand=False
        )
        assert fake_orca.calls[-1]["on_demand"] is True
        assert "forced on-demand" in result


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
        assert "planned_market" in prompt
        assert "Preferred market: on_demand" in prompt


class TestCostTableRanking:
    def test_build_cost_table_marks_cost_roofline_metadata(self, memory, resource_map):
        class StubPerfDB:
            def query(self, **kwargs):
                return [
                    {
                        "gpu_type": "L40S",
                        "tp": 4,
                        "pp": 1,
                        "dp": 1,
                        "throughput_tps": 1200.0,
                    }
                ]

        agent = KoiAgent(perfdb=StubPerfDB(), memory=memory, api_key="test-key")
        req = JobRequest(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            slo_deadline_hours=8.0,
            cost_roofline_usd=20.0,
            preferred_market="on_demand",
        )

        agent._build_cost_table(req, resource_map)

        row = agent._last_cost_rows[0]
        assert row["under_cost_roofline"] is False
        assert row["cost_overage_usd"] > 0

    def test_build_cost_table_uses_none_without_cost_roofline(self, memory, resource_map):
        class StubPerfDB:
            def query(self, **kwargs):
                return [
                    {
                        "gpu_type": "L40S",
                        "tp": 4,
                        "pp": 1,
                        "dp": 1,
                        "throughput_tps": 1200.0,
                    }
                ]

        agent = KoiAgent(perfdb=StubPerfDB(), memory=memory, api_key="test-key")
        req = JobRequest(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            slo_deadline_hours=8.0,
            preferred_market="on_demand",
        )

        agent._build_cost_table(req, resource_map)

        row = agent._last_cost_rows[0]
        assert row["under_cost_roofline"] is None
        assert row["cost_overage_usd"] is None

    def test_build_cost_table_sorts_slo_meeting_before_cheaper_slo_miss(
        self, memory, resource_map
    ):
        class StubPerfDB:
            def query(self, **kwargs):
                return [
                    {
                        "gpu_type": "L40S",
                        "tp": 4,
                        "pp": 1,
                        "dp": 1,
                        "throughput_tps": 200.0,
                    },
                    {
                        "gpu_type": "A100-80GB",
                        "tp": 8,
                        "pp": 1,
                        "dp": 1,
                        "throughput_tps": 500.0,
                    },
                ]

        agent = KoiAgent(perfdb=StubPerfDB(), memory=memory, api_key="test-key")
        req = JobRequest(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            slo_deadline_hours=8.0,
            preferred_market="on_demand",
        )

        agent._build_cost_table(req, resource_map)

        assert len(agent._last_cost_rows) == 2
        assert agent._last_cost_rows[0]["gpu_type"] == "A100-80GB"
        assert agent._last_cost_rows[0]["meets_slo"] is True
        assert agent._last_cost_rows[1]["gpu_type"] == "L40S"
        assert agent._last_cost_rows[1]["meets_slo"] is False

    def test_decide_prompt_has_required_tps(self, agent, job_request, resource_map):
        prompt = agent._build_decide_prompt(job_request, resource_map)
        # 5000 * (953+1024) / (8*3600) ≈ 343
        assert "343" in prompt or "Required TPS" in prompt

    def test_trigger_prompt_falling_behind(self, agent):
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.FALLING_BEHIND,
            job_id="job-test",
            job_tracker={
                "smoothed_tps": 200,
                "slo_headroom_pct": 5.0,
                "elapsed_hours": 2.0,
                "tokens_remaining": 5_000_000,
                "projected_total_cost_usd": 148.0,
                "cost_roofline_usd": 120.0,
                "cost_overage_usd": 28.0,
                "gpu_cache_usage": 0.85,
                "gpu_sm_util": 90,
                "gpu_mem_bw_util": 95,
            },
            diagnosis_hint="Throughput dropped below SLO requirement",
        )
        prompt = agent._build_trigger_prompt(trigger)
        assert "falling_behind" in prompt
        assert "scale_chain_tool" in prompt
        assert "Throughput dropped" in prompt
        assert "Projected total cost: $148.00" in prompt
        assert "Cost roofline: $120.00" in prompt
        assert "Projected overage: $28.00" in prompt
        assert "SCALE UP FIRST" in prompt

    def test_trigger_prompt_completed(self, agent):
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.COMPLETED,
            job_id="job-done",
            job_tracker={
                "smoothed_tps": 1500,
                "slo_headroom_pct": 80.0,
                "elapsed_hours": 1.5,
                "tokens_remaining": 0,
            },
        )
        prompt = agent._build_trigger_prompt(trigger)
        assert "completed" in prompt
        assert "record_outcome" in prompt.lower() or "Record the outcome" in prompt

    # ----- Area 2 tests: job-headroom-first trigger prompt ------------------

    def _make_chain_tracker(
        self,
        job_id,
        group_id,
        gpu,
        tp,
        pp,
        tps,
        status,
        predicted_tps=None,
        predicted_cost_per_hour=10.0,
    ):
        """Build a live JobTracker for a mocked chain."""
        return JobTracker(
            job_id=job_id,
            group_id=group_id,
            config=PlacementConfig(
                gpu_type=gpu,
                instance_type="dummy",
                num_gpus=tp * pp,
                num_instances=1,
                tp=tp,
                pp=pp,
                dp=1,
                region="us-east-1",
                engine_config=EngineConfig(
                    tensor_parallel_size=tp, pipeline_parallel_size=pp
                ),
                market="on_demand",
            ),
            slo_deadline_hours=1.0,
            total_tokens=12_000_000,
            predicted_tps=tps if predicted_tps is None else predicted_tps,
            predicted_cost_per_hour=predicted_cost_per_hour,
            smoothed_tps=tps,
            status=status,
        )

    def test_trigger_prompt_falling_behind_puts_job_headroom_before_chains(self, agent):
        """Area 2 rule: JOB-LEVEL HEADROOM block must render before CHAINS."""
        group_id = "demo-job-abc"
        chains = {
            "demo-job-abc-r0": self._make_chain_tracker(
                "demo-job-abc-r0",
                group_id,
                "A100-80GB",
                8,
                1,
                1140.0,
                MonitoringStatus.ON_TRACK,
            ),
            "demo-job-abc-r1": self._make_chain_tracker(
                "demo-job-abc-r1",
                group_id,
                "L40S",
                2,
                4,
                210.0,
                MonitoringStatus.ON_TRACK,
            ),
            "demo-job-abc-r2": self._make_chain_tracker(
                "demo-job-abc-r2",
                group_id,
                "L40S",
                2,
                4,
                205.0,
                MonitoringStatus.ON_TRACK,
            ),
        }

        fake_monitor = SimpleNamespace(get_group_chains=lambda _gid: chains)
        agent.monitor = fake_monitor
        try:
            trigger = MonitoringTrigger(
                trigger_type=MonitoringStatus.FALLING_BEHIND,
                job_id="demo-job-abc-r0",
                job_tracker={
                    "group_id": group_id,
                    "config": {"gpu_type": "A100-80GB", "tp": 8, "pp": 1, "dp": 1},
                    "slo_deadline_hours": 1.0,
                    "elapsed_hours": 0.1,
                    "tokens_remaining": 10_000_000,
                    "total_tokens": 12_000_000,
                    "smoothed_tps": 1140.0,
                    "predicted_tps": 1140.0,
                    "slo_headroom_pct": -50.0,
                },
                diagnosis_hint="Headroom=-50%, TPS=1555",
            )
            prompt = agent._build_trigger_prompt(trigger)
        finally:
            agent.monitor = None

        assert "JOB-LEVEL HEADROOM" in prompt
        assert "CHAINS (informational" in prompt
        # Order check: job-level block must come first
        assert prompt.index("JOB-LEVEL HEADROOM") < prompt.index(
            "CHAINS (informational"
        )
        # Aggregate TPS derived from the 3 chains
        assert "Aggregate TPS: 1,555" in prompt
        # Required TPS computed from remaining / time_left (tokens=10e6, time=0.9h)
        # 10_000_000 / (0.9 * 3600) ≈ 3086
        assert "Required TPS: 3,086" in prompt or "Required TPS: 3,087" in prompt
        # Median of [210, 205, 1140] sorted = [205, 210, 1140] → median 210
        assert "Median TPS: 210" in prompt

    def test_trigger_prompt_falling_behind_forbids_killing_above_median(self, agent):
        """Rule 3: never kill a chain with TPS >= fleet median."""
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.FALLING_BEHIND,
            job_id="job-test",
            job_tracker={
                "smoothed_tps": 500,
                "slo_headroom_pct": 5.0,
                "elapsed_hours": 0.5,
                "slo_deadline_hours": 1.0,
                "tokens_remaining": 1_000_000,
            },
            diagnosis_hint="Headroom=5%, TPS=500",
        )
        prompt = agent._build_trigger_prompt(trigger)
        assert "SCALE UP FIRST" in prompt
        assert "NEVER kill a chain whose TPS is >= the fleet median" in prompt
        assert "scale_chain_tool" in prompt

    def test_trigger_prompt_falling_behind_handles_deadline_exceeded(self, agent):
        """When elapsed >= slo, Required TPS must render as 'unattainable'."""
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.FALLING_BEHIND,
            job_id="job-late",
            job_tracker={
                "smoothed_tps": 500,
                "slo_headroom_pct": -100.0,
                "elapsed_hours": 2.0,
                "slo_deadline_hours": 1.0,
                "tokens_remaining": 4_000_000,
            },
            diagnosis_hint="Headroom=-100%, TPS=500",
        )
        prompt = agent._build_trigger_prompt(trigger)
        assert "unattainable" in prompt
        assert "Time left: 0.00h" in prompt

    def test_trigger_prompt_over_provisioned_has_strict_action_framework(self, agent):
        """OVER_PROVISIONED must forbid scale_up and limit to one kill per trigger."""
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.OVER_PROVISIONED,
            job_id="job-over",
            job_tracker={
                "smoothed_tps": 4800,
                "slo_headroom_pct": 95.0,
                "elapsed_hours": 0.5,
                "slo_deadline_hours": 4.0,
                "tokens_remaining": 2_000_000,
                "projected_total_cost_usd": 148.0,
                "cost_roofline_usd": 120.0,
                "cost_overage_usd": 28.0,
            },
            diagnosis_hint="Headroom=95%, can shed replicas",
        )
        prompt = agent._build_trigger_prompt(trigger)
        assert "OVER_PROVISIONED" in prompt or "over_provisioned" in prompt
        assert "LEAST productive chain" in prompt
        assert "AT MOST one chain per trigger" in prompt
        # Explicit ban on positive-count scale
        assert "Do NOT call scale_chain_tool with a positive count" in prompt
        assert "Projected total cost: $148.00" in prompt
        assert "highest-ranked POLICY RANKING removal option" in prompt

    def test_trigger_prompt_ranks_cheaper_valid_scale_up_first(self, agent):
        group_id = "cost-job"
        chains = {
            "cost-job-r0": self._make_chain_tracker(
                "cost-job-r0",
                group_id,
                "L40S",
                4,
                1,
                600.0,
                MonitoringStatus.ON_TRACK,
                predicted_tps=1200.0,
                predicted_cost_per_hour=10.0,
            ),
            "cost-job-r1": self._make_chain_tracker(
                "cost-job-r1",
                group_id,
                "A100-80GB",
                8,
                1,
                900.0,
                MonitoringStatus.ON_TRACK,
                predicted_tps=1200.0,
                predicted_cost_per_hour=20.0,
            ),
        }
        fake_monitor = SimpleNamespace(get_group_chains=lambda _gid: chains)
        agent.monitor = fake_monitor
        try:
            trigger = MonitoringTrigger(
                trigger_type=MonitoringStatus.FALLING_BEHIND,
                job_id="cost-job-r0",
                job_tracker={
                    "group_id": group_id,
                    "config": {"gpu_type": "L40S", "tp": 4, "pp": 1, "dp": 1},
                    "slo_deadline_hours": 1.0,
                    "elapsed_hours": 0.5,
                    "tokens_remaining": 3_600_000,
                    "smoothed_tps": 600.0,
                    "predicted_tps": 1200.0,
                    "predicted_cost_per_hour": 10.0,
                    "slo_headroom_pct": -10.0,
                },
                diagnosis_hint="Headroom=-10%, TPS=1500",
            )
            prompt = agent._build_trigger_prompt(trigger)
        finally:
            agent.monitor = None

        assert "POLICY RANKING" in prompt
        assert "scale_up L40S TP=4 PP=1 count=1 [current_config]" in prompt
        assert "scale_up A100-80GB TP=8 PP=1 count=1 [running:cost-job-r1]" in prompt
        assert prompt.index("scale_up L40S TP=4 PP=1 count=1") < prompt.index(
            "scale_up A100-80GB TP=8 PP=1 count=1"
        )

    def test_trigger_prompt_ranks_safe_lowest_tps_removal_first(self, agent):
        group_id = "over-job"
        chains = {
            "over-job-r-fast": self._make_chain_tracker(
                "over-job-r-fast",
                group_id,
                "A100-80GB",
                8,
                1,
                1500.0,
                MonitoringStatus.ON_TRACK,
                predicted_cost_per_hour=20.0,
            ),
            "over-job-r-mid": self._make_chain_tracker(
                "over-job-r-mid",
                group_id,
                "L40S",
                4,
                1,
                900.0,
                MonitoringStatus.ON_TRACK,
                predicted_cost_per_hour=10.0,
            ),
            "over-job-r-slow": self._make_chain_tracker(
                "over-job-r-slow",
                group_id,
                "L40S",
                4,
                1,
                600.0,
                MonitoringStatus.ON_TRACK,
                predicted_cost_per_hour=10.0,
            ),
        }
        fake_monitor = SimpleNamespace(get_group_chains=lambda _gid: chains)
        agent.monitor = fake_monitor
        try:
            trigger = MonitoringTrigger(
                trigger_type=MonitoringStatus.OVER_PROVISIONED,
                job_id="over-job-r-fast",
                job_tracker={
                    "group_id": group_id,
                    "config": {"gpu_type": "A100-80GB", "tp": 8, "pp": 1, "dp": 1},
                    "slo_deadline_hours": 4.0,
                    "elapsed_hours": 0.5,
                    "tokens_remaining": 2_000_000,
                    "smoothed_tps": 1500.0,
                    "predicted_tps": 1500.0,
                    "predicted_cost_per_hour": 20.0,
                    "slo_headroom_pct": 95.0,
                },
                diagnosis_hint="Headroom=95%, can shed replicas",
            )
            prompt = agent._build_trigger_prompt(trigger)
        finally:
            agent.monitor = None

        assert "POLICY RANKING" in prompt
        assert "kill_replica over-job-r-slow" in prompt
        assert "kill_replica over-job-r-mid" in prompt
        assert prompt.index("kill_replica over-job-r-slow") < prompt.index(
            "kill_replica over-job-r-mid"
        )


class TestRedecideCandidates:
    @pytest.mark.asyncio
    async def test_build_redecide_surfaces_fresh_configs(self, perfdb, memory):
        """Re-decide pulls fresh ResourceMap from Orca and returns viable configs."""

        async def fake_get_resources():
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
                    },
                ],
                "quotas": [
                    {
                        "family": "G",
                        "market": "on_demand",
                        "region": "us-east-1",
                        "baseline_vcpus": 10000,
                        "used_vcpus": 0,
                    }
                ],
                "vpc_id": "vpc-test",
                "region": "us-east-1",
            }

        orca_stub = SimpleNamespace(get_resources=fake_get_resources)
        agent = KoiAgent(perfdb=perfdb, memory=memory, orca=orca_stub, api_key="test-key")
        dec_id = memory.record_decision(
            job_id="job-re",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=10.49,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            market="on_demand",
        )

        candidates = await agent._build_redecide_candidates(
            {"decision_id": dec_id, "job_id": "job-re"}
        )

        assert candidates, "expected at least one redecide candidate"
        assert all(c.source == "redecide" for c in candidates)
        assert any(c.gpu_type == "L40S" for c in candidates)

    @pytest.mark.asyncio
    async def test_build_redecide_returns_empty_when_orca_unreachable(
        self, perfdb, memory
    ):
        """If Orca raises, we fall back silently — trigger path uses narrow set."""

        async def raising_get_resources():
            raise RuntimeError("orca is down")

        orca_stub = SimpleNamespace(get_resources=raising_get_resources)
        agent = KoiAgent(perfdb=perfdb, memory=memory, orca=orca_stub, api_key="test-key")
        dec_id = memory.record_decision(
            job_id="job-re",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=10.49,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
        )

        candidates = await agent._build_redecide_candidates(
            {"decision_id": dec_id, "job_id": "job-re"}
        )
        assert candidates == []

    @pytest.mark.asyncio
    async def test_build_redecide_returns_empty_without_decision(
        self, perfdb, memory
    ):
        """Missing decision_id or unknown decision → empty list, no crash."""
        agent = KoiAgent(perfdb=perfdb, memory=memory, api_key="test-key")
        assert await agent._build_redecide_candidates({}) == []
        assert await agent._build_redecide_candidates(
            {"decision_id": "dec-nonexistent"}
        ) == []

    def test_trigger_prompt_surfaces_redecide_candidates(self, agent):
        """Precomputed [redecide] candidates appear in the POLICY RANKING block."""
        from koi.runtime_policy import ScaleUpCandidate

        precomputed = [
            ScaleUpCandidate(
                gpu_type="H100",
                tp=2,
                pp=1,
                predicted_tps=2500.0,
                cost_per_hour=15.0,
                source="redecide",
            )
        ]
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.FALLING_BEHIND,
            job_id="job-re",
            job_tracker={
                "config": {"gpu_type": "L40S", "tp": 4, "pp": 1, "dp": 1},
                "slo_deadline_hours": 1.0,
                "elapsed_hours": 0.5,
                "tokens_remaining": 3_600_000,
                "smoothed_tps": 600.0,
                "predicted_tps": 1200.0,
                "predicted_cost_per_hour": 10.0,
                "slo_headroom_pct": -10.0,
            },
            diagnosis_hint="Headroom=-10%",
        )
        prompt = agent._build_trigger_prompt(
            trigger, precomputed_candidates=precomputed
        )

        assert "[redecide]" in prompt
        assert "scale_up H100 TP=2 PP=1 count=1 [redecide]" in prompt


class TestParseDecision:
    def test_parse_decision_sets_cost_roofline_warning_fields(
        self, agent, resource_map
    ):
        req = JobRequest(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            slo_deadline_hours=8.0,
            cost_roofline_usd=20.0,
            preferred_market="on_demand",
        )
        text = 'I recommend {"gpu_type": "L40S", "instance_type": "g6e.12xlarge", "tp": 4, "pp": 1, "dp": 1, "predicted_tps": 1200.0, "predicted_cost_per_hour": 10.49, "confidence": 0.75, "reasoning": "cheapest option"}'
        decision = agent._parse_decision(text, req, resource_map, tool_calls=3, elapsed=8.0)
        assert decision.meets_cost_roofline is False
        assert decision.cost_roofline_usd == 20.0
        assert decision.projected_cost_overage_usd > 0
        assert "Projected cost exceeds roofline" in decision.cost_warning

    def test_parse_json_block(self, agent, job_request, resource_map):
        text = """Based on my analysis, here's the placement:

```json
{
  "gpu_type": "A100-80GB",
  "instance_type": "p4de.24xlarge",
  "tp": 4,
  "pp": 2,
  "dp": 1,
  "planned_market": "spot",
  "predicted_tps": 2590.0,
  "predicted_cost_per_hour": 40.96,
  "reasoning": "PerfDB shows A100-80GB TP=4 PP=2 gets 2590 TPS",
  "confidence": 0.88,
  "data_source": "perfdb_exact"
}
```"""
        decision = agent._parse_decision(
            text, job_request, resource_map, tool_calls=5, elapsed=12.3
        )
        assert decision.config.gpu_type == "A100-80GB"
        assert decision.config.tp == 4
        assert decision.config.pp == 2
        assert decision.config.market == "spot"
        assert decision.planned_market == "spot"
        assert decision.predicted_tps == 2590.0
        assert decision.confidence == 0.88
        assert decision.data_source == DataSource.EXACT_MATCH
        assert decision.tool_calls_made == 5
        assert decision.latency_seconds == 12.3

    def test_parse_raw_json(self, agent, job_request, resource_map):
        text = 'I recommend {"gpu_type": "L40S", "instance_type": "g6e.12xlarge", "tp": 4, "pp": 2, "dp": 1, "predicted_tps": 833.0, "confidence": 0.75, "reasoning": "cheapest option"}'
        decision = agent._parse_decision(
            text, job_request, resource_map, tool_calls=3, elapsed=8.0
        )
        assert decision.config.gpu_type == "L40S"
        assert decision.predicted_tps == 833.0
        assert decision.planned_market == "on_demand"

    def test_parse_fallback_on_no_json(self, agent, job_request, resource_map):
        text = "I suggest using L40S with TP=4 but I couldn't format JSON properly."
        decision = agent._parse_decision(
            text, job_request, resource_map, tool_calls=2, elapsed=5.0
        )
        # Should fall back to defaults without crashing
        assert decision.config.gpu_type == "L40S"  # default
        assert decision.planned_market == "on_demand"
        assert decision.confidence == 0.5  # default
