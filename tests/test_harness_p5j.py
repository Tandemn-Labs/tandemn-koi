"""Phase 6: tests for P5j job post-mortem harness."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic_ai.models.test import TestModel

from koi.agent import KoiAgent
from koi.harness.p5c import P5cDiagnosis
from koi.harness.p5j import (
    P5jDiagnosis,
    build_p5j_packet,
    collect_chain_diagnoses,
    deterministic_job_diagnosis,
    render_p5j_prompt,
    run_job_postmortem,
)
from koi.schemas import EngineConfig, JobTracker, MonitoringStatus, PlacementConfig
from koi.tools.memory import AgenticMemory


@pytest.fixture
def memory() -> AgenticMemory:
    return AgenticMemory(db_path=":memory:")


def _req(status: str = "failed") -> SimpleNamespace:
    return SimpleNamespace(
        job_id="job-group",
        group_id=None,
        decision_id=None,
        status=status,
        metrics={"throughput_tokens_per_sec": 0.0},
        reason_code=None,
        reason_detail="all chains failed",
    )


def _tracker(decision_id: str = "dec-parent") -> JobTracker:
    config = PlacementConfig(
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        num_gpus=4,
        num_instances=1,
        tp=4,
        pp=1,
        dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=1),
        market="spot",
    )
    tracker = JobTracker(
        job_id="r0",
        decision_id=decision_id,
        group_id="job-group",
        config=config,
        slo_deadline_hours=8.0,
        total_tokens=6_000_000,
        predicted_tps=1200.0,
        tokens_remaining=4_000_000,
    )
    tracker.status = MonitoringStatus.FAILED
    tracker.smoothed_tps = 0.0
    tracker.elapsed_hours = 0.5
    tracker.slo_headroom_pct = -100.0
    return tracker


def _record_decision(memory: AgenticMemory, decision_id_ref: str = "job-group") -> str:
    return memory.record_decision(
        job_id=decision_id_ref,
        model_name="Qwen/Qwen3-32B",
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
        avg_input_tokens=1024,
        avg_output_tokens=1024,
        num_requests=1500,
        market="spot",
    )


class TestP5jDeterministic:
    def test_collects_existing_p5c_diagnosis_from_outcome(self, memory):
        decision_id = _record_decision(memory)
        p5c = P5cDiagnosis(
            diagnosis_code="spot_preemption",
            bottleneck="market_capacity",
            next_fix="retry_same_topology_on_demand",
            failure_scope="L40S|g6e.12xlarge|us-east-1|spot",
            rationale="spot interrupted",
        )
        memory.record_outcome(
            decision_id=decision_id,
            job_id="job-group",
            status="replica_failed",
            failure_category="spot_preemption",
            diagnosis="spot_preemption: interrupted",
            bottleneck="market_capacity",
            diff_from_parent=p5c.model_dump_json(),
        )

        diagnoses = collect_chain_diagnoses(
            req=_req(),
            memory=memory,
            group_chains={"r0": _tracker(decision_id)},
        )

        assert len(diagnoses) == 1
        assert diagnoses[0]["diagnosis_code"] == "spot_preemption"
        assert diagnoses[0]["source"] == "p5c_outcome"

    def test_synthesizes_job_level_capacity_failure(self):
        diag = deterministic_job_diagnosis(
            req=_req(),
            chain_diagnoses=[
                {"diagnosis_code": "spot_preemption"},
                {"diagnosis_code": "spot_preemption"},
            ],
            launch_failures=[],
            chain_count=2,
            now=1000.0,
        )

        assert diag.diagnosis_code == "job_capacity_exhausted"
        assert diag.bottleneck == "market_capacity"
        assert diag.next_fix == "retry_same_topology_on_demand"
        assert diag.failed_chains == 2
        assert diag.diagnosed_chains == 2
        assert diag.event_at == 1000.0

    def test_packet_uses_terminal_failed_job_postmortem(self, memory):
        decision_id = _record_decision(memory)
        packet = build_p5j_packet(
            req=_req(),
            memory=memory,
            group_chains={"r0": _tracker(decision_id)},
        )

        assert packet.state.value == "terminal_failed"
        assert packet.transition_type.value == "job_postmortem"
        assert "chain_diagnoses:all" in packet.detail_sections
        assert "launch_failures:job" in packet.detail_sections
        prompt = render_p5j_prompt(packet)
        assert "P5J JOB POST-MORTEM" in prompt
        assert "JOB CONTEXT" in prompt


class TestRunJobPostmortem:
    @pytest.mark.asyncio
    async def test_returns_deterministic_diagnosis_on_timeout(self, memory):
        decision_id = _record_decision(memory)
        agent = KoiAgent(perfdb=None, memory=memory, api_key="test-key")
        agent._model = TestModel()

        from koi.harness import p5j as p5j_module

        async def _raise_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        original = p5j_module.KoiToolRunner.run_typed
        p5j_module.KoiToolRunner.run_typed = _raise_timeout
        try:
            diag = await run_job_postmortem(
                agent=agent,
                req=_req(),
                memory=memory,
                group_chains={"r0": _tracker(decision_id)},
            )
        finally:
            p5j_module.KoiToolRunner.run_typed = original

        assert isinstance(diag, P5jDiagnosis)
        assert diag.diagnosis_code == "job_failed_unknown"
        assert diag.failed_chains == 1

    @pytest.mark.asyncio
    async def test_uses_llm_diagnosis_when_available(self, memory):
        decision_id = _record_decision(memory)
        agent = KoiAgent(perfdb=None, memory=memory, api_key="test-key")
        agent._model = TestModel()

        from koi.harness import p5j as p5j_module

        async def _stub_run_typed(self, prompt, **kwargs):  # noqa: ARG002
            return 0, P5jDiagnosis(
                diagnosis_code="job_runtime_unhealthy",
                bottleneck="runtime_unhealthy",
                next_fix="review_runtime_logs",
                failure_scope="job-group",
                rationale="LLM synthesized terminal diagnosis",
            )

        original = p5j_module.KoiToolRunner.run_typed
        p5j_module.KoiToolRunner.run_typed = _stub_run_typed
        try:
            diag = await run_job_postmortem(
                agent=agent,
                req=_req(),
                memory=memory,
                group_chains={"r0": _tracker(decision_id)},
            )
        finally:
            p5j_module.KoiToolRunner.run_typed = original

        assert diag.diagnosis_code == "job_runtime_unhealthy"
        assert diag.chain_diagnoses
        assert diag.diagnosed_chains == 1
