"""Phase 2.5: tests for koi.harness.pscale_menu and Pscale fast-path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai.models.test import TestModel

from koi.agent import KoiAgent
from koi.harness.pscale import (
    build_pscale_packet,
    render_pscale_prompt,
    run_runtime_scale,
)
from koi.harness.pscale_menu import (
    ExclusionRecord,
    MenuCandidate,
    _apply_source_caps,
    _dominance_filter,
    build_menu,
)
from koi.schemas import (
    EngineConfig,
    GPUResource,
    MonitoringStatus,
    MonitoringTrigger,
    PlacementConfig,
    ResourceMap,
)
from koi.tools.memory import AgenticMemory


@pytest.fixture
def memory():
    return AgenticMemory(db_path=":memory:")


class StubPerfDB:
    """Provides exact + cross-family rows for Qwen/Qwen3-32B."""

    _rows = [
        {"gpu_type": "L40S", "instance_type": "g6e.12xlarge", "tp": 4, "pp": 1, "dp": 1, "throughput_tps": 1200.0, "num_params_billions": 32, "num_layers": 64, "num_attention_heads": 64, "num_kv_heads": 8, "is_moe": 0},
        {"gpu_type": "L40S", "instance_type": "g6e.12xlarge", "tp": 4, "pp": 2, "dp": 1, "throughput_tps": 1828.0, "num_params_billions": 32, "num_layers": 64, "num_attention_heads": 64, "num_kv_heads": 8, "is_moe": 0},
        {"gpu_type": "A100-80GB", "instance_type": "p4de.24xlarge", "tp": 4, "pp": 2, "dp": 1, "throughput_tps": 4119.0, "num_params_billions": 32, "num_layers": 64, "num_attention_heads": 64, "num_kv_heads": 8, "is_moe": 0},
        {"gpu_type": "A100-80GB", "instance_type": "p4de.24xlarge", "tp": 8, "pp": 1, "dp": 1, "throughput_tps": 4619.0, "num_params_billions": 32, "num_layers": 64, "num_attention_heads": 64, "num_kv_heads": 8, "is_moe": 0},
    ]

    def query(self, model_name=None, gpu_type=None, tp=None, pp=None, sort_by="throughput_tps", limit=20, **kwargs):
        records = list(self._rows)
        if gpu_type:
            records = [r for r in records if r["gpu_type"] == gpu_type]
        if tp is not None:
            records = [r for r in records if r["tp"] == tp]
        if pp is not None:
            records = [r for r in records if r["pp"] == pp]
        records.sort(key=lambda r: r.get("throughput_tps", 0), reverse=True)
        return records[:limit]

    def get_distinct_models(self):
        return [
            {
                "model_name": "Qwen/Qwen3-32B",
                "num_params_billions": 32,
                "num_layers": 64,
                "hidden_dim": 5120,
                "num_attention_heads": 64,
                "num_kv_heads": 8,
                "is_moe": False,
            }
        ]


class FakeOrca:
    def __init__(self, scale_calls=None, kill_calls=None):
        self.scale_calls = scale_calls if scale_calls is not None else []
        self.kill_calls = kill_calls if kill_calls is not None else []

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

    async def scale_job(self, job_id, gpu_type, tp, pp, count, **kwargs):
        self.scale_calls.append(
            {"job_id": job_id, "gpu_type": gpu_type, "tp": tp, "pp": pp, "count": count, **kwargs}
        )
        return {"status": "scaling", "new_replicas": [f"{job_id}-r-new"], "scale_request_id": "scale-1"}

    async def kill_replicas(self, job_id, replica_ids):
        self.kill_calls.append({"job_id": job_id, "replica_ids": list(replica_ids)})
        return {"status": "killed"}


class FakeMonitor:
    def __init__(self, chains=None):
        self._chains = chains or {}
        self._koi_initiated_kills = set()
        self.tracked_jobs = dict(self._chains)

    def get_group_chains(self, group_id):
        return self._chains

    def persist_job(self, job_id):
        return None

    def register_pending_replica_decision(self, **kwargs):
        return None


def _config(gpu_type="L40S", tp=4, pp=1, instance_type="g6e.12xlarge"):
    return PlacementConfig(
        gpu_type=gpu_type,
        instance_type=instance_type,
        num_gpus=tp * pp,
        num_instances=1,
        tp=tp,
        pp=pp,
        dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        market="on_demand",
    )


def _agent(memory, orca=None, monitor=None, perfdb=None) -> KoiAgent:
    agent = KoiAgent(perfdb=perfdb, memory=memory, orca=orca, api_key="test-key")
    agent._model = TestModel()
    agent.model = "test-model"
    agent.monitor = monitor
    return agent


def _prime_parent_decision(memory: AgenticMemory) -> str:
    return memory.record_decision(
        job_id="group-job",
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
        market="on_demand",
        cost_roofline_usd=100.0,
    )


def _falling_trigger(decision_id):
    return MonitoringTrigger(
        trigger_type=MonitoringStatus.FALLING_BEHIND,
        job_id="group-job-r0",
        job_tracker={
            "job_id": "group-job-r0",
            "group_id": "group-job",
            "decision_id": decision_id,
            "config": _config().model_dump(mode="json"),
            "predicted_tps": 1200.0,
            "predicted_cost_per_hour": 10.49,
            "smoothed_tps": 370.0,
            "slo_deadline_hours": 1.0,
            "elapsed_hours": 0.5,
            "tokens_remaining": 1_500_000,
            "slo_headroom_pct": -100.0,
            "cost_roofline_usd": 100.0,
        },
        diagnosis_hint="Headroom=-100%, TPS=370",
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_apply_source_caps_reserves_one_per_source_then_fills_globally():
    candidates = [
        MenuCandidate(
            kind="scale_up", source="current_config", gpu_type="L40S",
            tp=4, pp=1, dp=1, market="on_demand", instance_type="g6e.12xlarge",
            predicted_tps=600.0, cost_per_hour=10.49,
            prediction_source="memory_verified", prediction_confidence=0.9,
        ),
        MenuCandidate(
            kind="scale_up", source="tp_pp_alternate", gpu_type="L40S",
            tp=4, pp=2, dp=1, market="on_demand", instance_type="g6e.12xlarge",
            predicted_tps=1828.0, cost_per_hour=20.98,
            prediction_source="perfdb_exact", prediction_confidence=0.85,
        ),
        MenuCandidate(
            kind="scale_up", source="tp_pp_alternate", gpu_type="L40S",
            tp=4, pp=4, dp=1, market="on_demand", instance_type="g6e.12xlarge",
            predicted_tps=3156.0, cost_per_hour=41.96,
            prediction_source="perfdb_exact", prediction_confidence=0.8,
        ),
        MenuCandidate(
            kind="scale_up", source="gpu_family_alternate", gpu_type="A100-80GB",
            tp=4, pp=2, dp=1, market="on_demand", instance_type="p4de.24xlarge",
            predicted_tps=4119.0, cost_per_hour=40.96,
            prediction_source="perfdb_exact", prediction_confidence=0.85,
        ),
    ]
    capped = _apply_source_caps(candidates, current_aggregate=370.0, max_total=3)
    sources = [c.source for c in capped]
    assert sources.count("current_config") == 1
    assert sources.count("gpu_family_alternate") == 1
    assert "tp_pp_alternate" in sources
    assert len(capped) == 3


def test_dominance_filter_drops_strictly_dominated_options():
    cheap_strong = MenuCandidate(
        kind="scale_up", source="tp_pp_alternate", gpu_type="L40S",
        tp=4, pp=2, dp=1, market="on_demand", instance_type="g6e.12xlarge",
        predicted_tps=1828.0, cost_per_hour=20.0,
        prediction_source="perfdb_exact", prediction_confidence=0.85,
    )
    expensive_weaker = MenuCandidate(
        kind="scale_up", source="redecide_proxy", gpu_type="L40S",
        tp=4, pp=2, dp=1, market="on_demand", instance_type="g6e.12xlarge",
        predicted_tps=1500.0, cost_per_hour=25.0,
        prediction_source="physics_proxy", prediction_confidence=0.7,
    )
    kept, excluded = _dominance_filter(
        [cheap_strong, expensive_weaker],
        required_tps=300.0,
        current_aggregate=370.0,
    )
    assert kept == [cheap_strong]
    assert any("dominated_by" in rec.reason for rec in excluded)


def test_recent_failure_candidate_does_not_dominate_safer_option():
    risky_strong = MenuCandidate(
        kind="scale_up", source="tp_pp_alternate", gpu_type="L40S",
        tp=4, pp=2, dp=1, market="on_demand", instance_type="g6e.12xlarge",
        predicted_tps=2000.0, cost_per_hour=20.0,
        prediction_source="perfdb_exact", prediction_confidence=0.85,
        recent_failure={"same_scope": True, "diagnosis_code": "no_capacity"},
    )
    safer_weaker = MenuCandidate(
        kind="scale_up", source="gpu_family_alternate", gpu_type="A100-80GB",
        tp=4, pp=2, dp=1, market="on_demand", instance_type="p4de.24xlarge",
        predicted_tps=1800.0, cost_per_hour=25.0,
        prediction_source="perfdb_exact", prediction_confidence=0.85,
    )

    kept, excluded = _dominance_filter(
        [risky_strong, safer_weaker],
        required_tps=300.0,
        current_aggregate=370.0,
    )

    assert risky_strong in kept
    assert safer_weaker in kept
    assert excluded == []


def test_source_caps_downrank_recent_failure_within_same_source():
    risky = MenuCandidate(
        kind="scale_up", source="tp_pp_alternate", gpu_type="L40S",
        tp=4, pp=2, dp=1, market="on_demand", instance_type="g6e.12xlarge",
        predicted_tps=2000.0, cost_per_hour=20.0,
        prediction_source="perfdb_exact", prediction_confidence=0.85,
        recent_failure={"same_scope": True, "diagnosis_code": "no_capacity"},
    )
    safer = MenuCandidate(
        kind="scale_up", source="tp_pp_alternate", gpu_type="A100-80GB",
        tp=4, pp=2, dp=1, market="on_demand", instance_type="p4de.24xlarge",
        predicted_tps=1800.0, cost_per_hour=25.0,
        prediction_source="perfdb_exact", prediction_confidence=0.85,
    )

    capped = _apply_source_caps([risky, safer], current_aggregate=370.0, max_total=1)

    assert capped == [safer]


# ---------------------------------------------------------------------------
# Menu builder integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_menu_surfaces_heterogeneous_sources(memory):
    orca = FakeOrca()
    monitor = FakeMonitor()
    agent = _agent(memory, orca=orca, monitor=monitor, perfdb=StubPerfDB())
    decision_id = _prime_parent_decision(memory)
    trigger = _falling_trigger(decision_id)

    result = await build_menu(agent, trigger, max_options=7)

    sources = {c.source for c in result.candidates_by_action.values()}
    assert "current_config" in sources
    assert "tp_pp_alternate" in sources or "gpu_family_alternate" in sources
    # We always expect at least 2 distinct sources for a feasible cluster.
    assert len(sources) >= 2


@pytest.mark.asyncio
async def test_pscale_packet_carries_excluded_block_when_present(memory):
    orca = FakeOrca()
    monitor = FakeMonitor()
    agent = _agent(memory, orca=orca, monitor=monitor, perfdb=StubPerfDB())
    decision_id = _prime_parent_decision(memory)
    trigger = _falling_trigger(decision_id)

    packet = await build_pscale_packet(agent, trigger)

    summary = packet.evidence_summary
    assert summary["source"] == "pscale_menu"
    assert "counts_by_source" in summary
    # excluded list is allowed to be empty in this scenario; just assert shape
    assert "excluded" in summary
    assert isinstance(summary["excluded"], list)


@pytest.mark.asyncio
async def test_pscale_packet_keeps_legacy_single_source_when_no_redecide(memory):
    """Packet must still build when Orca/perfdb are unavailable."""

    agent = _agent(memory, orca=None, monitor=FakeMonitor(), perfdb=None)
    trigger = _falling_trigger(decision_id=None)
    trigger.job_tracker.pop("decision_id", None)

    packet = await build_pscale_packet(agent, trigger)

    # current_config still emitted; menu is degenerate.
    assert any(opt.action_type == "scale_up" for opt in packet.action_options)
    assert packet.runtime_context.get("menu_degenerate") is True


# ---------------------------------------------------------------------------
# Render prompt: source tags + excluded block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_prompt_shows_evidence_summary_and_source_tags(memory):
    orca = FakeOrca()
    monitor = FakeMonitor()
    agent = _agent(memory, orca=orca, monitor=monitor, perfdb=StubPerfDB())
    decision_id = _prime_parent_decision(memory)
    trigger = _falling_trigger(decision_id)

    packet = await build_pscale_packet(agent, trigger)
    prompt = render_pscale_prompt(packet)

    assert "EVIDENCE SUMMARY:" in prompt
    assert "counts_by_source" in prompt
    assert "source=" in prompt or "source: " in prompt


# ---------------------------------------------------------------------------
# Fast path (Option A)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_path_skips_llm_when_only_one_valid_option(monkeypatch, memory):
    orca = FakeOrca()
    monitor = FakeMonitor()
    agent = _agent(memory, orca=orca, monitor=monitor, perfdb=None)
    trigger = _falling_trigger(decision_id=None)
    trigger.job_tracker.pop("decision_id", None)

    sentinel = {"called": False}

    class ExplodingReasoner:
        def __init__(self, *args, **kwargs):
            sentinel["called"] = True

        async def choose(self, *args, **kwargs):
            raise AssertionError("Reasoner must not run on fast path")

    with patch("koi.harness.pscale.HarnessReasoner", ExplodingReasoner):
        result = await run_runtime_scale(agent, trigger)

    assert sentinel["called"] is False
    assert "Scaled up" in result
    assert orca.scale_calls and orca.scale_calls[0]["gpu_type"] == "L40S"


@pytest.mark.asyncio
async def test_fast_path_returns_fallback_text_with_zero_valid_options(monkeypatch, memory):
    """Build a contrived packet with no valid options and assert fast-path bails out."""

    orca = FakeOrca()
    monitor = FakeMonitor()
    agent = _agent(memory, orca=orca, monitor=monitor, perfdb=None)
    trigger = _falling_trigger(decision_id=None)
    trigger.job_tracker.pop("decision_id", None)

    async def empty_packet(*_args, **_kwargs):
        from koi.harness.schemas import (
            HarnessState,
            TransitionPacket,
            TransitionType,
        )

        return TransitionPacket(
            packet_id="pscale-empty",
            job_id=trigger.job_id,
            state=HarnessState.DEGRADED,
            transition_type=TransitionType.SCALE,
            runtime_context={"menu_degenerate": True},
            evidence_summary={"valid_action_count": 0, "source": "pscale_menu"},
            action_options=[],
            detail_sections={},
            guards={},
        )

    with patch("koi.harness.pscale.build_pscale_packet", empty_packet):
        result = await run_runtime_scale(agent, trigger)

    assert "HARNESS FALLBACK" in result
