"""Tests for koi/tools/memory.py"""

import pytest
from koi.tools.memory import AgenticMemory, query_memory, record_outcome_tool


@pytest.fixture
def memory():
    """In-memory SQLite for testing."""
    return AgenticMemory(db_path=":memory:")


class TestDecisions:
    def test_record_and_query(self, memory):
        dec_id = memory.record_decision(
            job_id="job-test1", model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="p4de.24xlarge", gpu_type="A100-80GB",
            tp=4, pp=2, dp=1, num_gpus=8,
            predicted_tps=2590.0, predicted_cost_per_hour=40.96,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=1200, avg_output_tokens=300,
            num_requests=5000,
        )
        assert dec_id.startswith("dec-")

        results = memory.query_decisions(model_name="Qwen")
        assert len(results) == 1
        assert results[0]["gpu_type"] == "A100-80GB"

    def test_multiple_decisions(self, memory):
        for i in range(3):
            memory.record_decision(
                job_id=f"job-{i}", model_name="Qwen/Qwen2.5-72B-Instruct",
                instance_type="p4de.24xlarge", gpu_type="A100-80GB",
                tp=4, pp=2, dp=1, num_gpus=8,
                predicted_tps=2590.0, predicted_cost_per_hour=40.96,
                slo_deadline_hours=8.0, objective="cheapest",
                avg_input_tokens=1200, avg_output_tokens=300,
            )
        results = memory.query_decisions(model_name="Qwen")
        assert len(results) == 3


class TestOutcomes:
    def test_record_outcome_with_delta(self, memory):
        dec_id = memory.record_decision(
            job_id="job-1", model_name="test-model",
            instance_type="g6e.12xlarge", gpu_type="L40S",
            tp=4, pp=2, dp=1, num_gpus=8,
            predicted_tps=1000.0, predicted_cost_per_hour=13.35,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=500, avg_output_tokens=200,
        )

        out_id = memory.record_outcome(
            decision_id=dec_id, job_id="job-1", status="succeeded",
            actual_tps=850.0, actual_cost_per_hour=13.35,
        )
        assert out_id.startswith("out-")

        outcomes = memory.query_outcomes(model_name="test-model")
        assert len(outcomes) == 1
        assert outcomes[0]["delta_tps_pct"] == pytest.approx(-15.0, abs=0.5)

    def test_query_by_status(self, memory):
        dec_id = memory.record_decision(
            job_id="job-fail", model_name="test",
            instance_type="g6e", gpu_type="L40S",
            tp=4, pp=2, dp=1, num_gpus=8,
            predicted_tps=1000.0, predicted_cost_per_hour=10.0,
            slo_deadline_hours=4.0, objective="cheapest",
            avg_input_tokens=500, avg_output_tokens=200,
        )
        memory.record_outcome(
            decision_id=dec_id, job_id="job-fail", status="failed",
            failure_reason="OOM", failure_category="oom",
        )

        failed = memory.query_outcomes(status="failed")
        assert len(failed) == 1
        assert failed[0]["failure_reason"] == "OOM"


class TestRules:
    def test_add_and_query(self, memory):
        rule_id = memory.add_rule(
            rule_text="Qwen-72B on A100-40GB requires TP>=8 (OOM at TP=4)",
            rule_type="constraint",
            confidence=0.9, evidence_count=3,
            model_pattern="Qwen*72B*", gpu_pattern="A100-40GB",
        )
        assert rule_id.startswith("rule-")

        rules = memory.query_rules(model_pattern="Qwen")
        assert len(rules) == 1
        assert "TP>=8" in rules[0]["rule_text"]

    def test_rule_count(self, memory):
        assert memory.rule_count() == 0
        memory.add_rule("test rule", "preference", 0.5)
        assert memory.rule_count() == 1


class TestLaunchAttempts:
    def test_record_success(self, memory):
        att_id = memory.record_launch_attempt(
            decision_id="dec-test", job_id="job-1",
            instance_type="p4de.24xlarge", gpu_type="A100-80GB",
            region="us-west-2", market="on_demand", count=1,
            launched=True, time_to_launch=45.0,
        )
        assert att_id.startswith("att-")

    def test_record_failure(self, memory):
        memory.record_launch_attempt(
            decision_id="dec-test", job_id="job-1",
            instance_type="p5.48xlarge", gpu_type="H100",
            region="us-west-2", market="on_demand", count=1,
            launched=False, failure_reason="InsufficientCapacity",
        )

    def test_success_rate(self, memory):
        for i in range(4):
            memory.record_launch_attempt(
                decision_id=f"dec-{i}", job_id=f"job-{i}",
                instance_type="p5.48xlarge", gpu_type="H100",
                region="us-west-2", market="on_demand", count=1,
                launched=(i < 1),  # 1 success, 3 failures
            )
        rate = memory.get_launch_success_rate("p5.48xlarge", region="us-west-2")
        assert rate["attempts"] == 4
        assert rate["succeeded"] == 1
        assert rate["rate"] == pytest.approx(0.25)


class TestChainSnapshots:
    def test_record_and_query(self, memory):
        snap_id = memory.record_chain_snapshot(
            decision_id="dec-test", job_id="job-1",
            throughput_tps=833.0, tokens_completed=2_000_000,
            tokens_remaining=5_500_000, elapsed_hours=0.5,
            slo_headroom_pct=85.0, gpu_cache_usage_pct=0.62,
            gpu_sm_util_pct=78.0, gpu_mem_bw_util_pct=85.0,
        )
        assert snap_id.startswith("snap-")

        snaps = memory.query_chain_snapshots("dec-test")
        assert len(snaps) == 1
        assert snaps[0]["throughput_tps"] == 833.0
        assert snaps[0]["slo_headroom_pct"] == 85.0

    def test_time_series(self, memory):
        """Multiple snapshots create a time series."""
        for i in range(5):
            memory.record_chain_snapshot(
                decision_id="dec-ts", job_id="job-1",
                throughput_tps=800.0 - i * 50,  # degrading
                tokens_completed=i * 1_000_000,
                tokens_remaining=5_000_000 - i * 1_000_000,
                elapsed_hours=i * 0.5,
                slo_headroom_pct=90.0 - i * 10,
            )
        snaps = memory.query_chain_snapshots("dec-ts")
        assert len(snaps) == 5
        # Should be ordered by timestamp (ascending)
        tps_values = [s["throughput_tps"] for s in snaps]
        assert tps_values[0] > tps_values[-1]  # degrading over time

    def test_query_nonexistent(self, memory):
        snaps = memory.query_chain_snapshots("dec-nonexistent")
        assert len(snaps) == 0


class TestCounts:
    def test_initial_counts(self, memory):
        assert memory.decision_count() == 0
        assert memory.outcome_count() == 0
        assert memory.rule_count() == 0


class TestToolFunctions:
    def test_query_memory_empty(self, memory):
        result = query_memory(memory, model_name="Qwen")
        assert "No memory found" in result

    def test_query_memory_with_data(self, memory):
        dec_id = memory.record_decision(
            job_id="job-1", model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge", gpu_type="L40S",
            tp=4, pp=2, dp=1, num_gpus=8,
            predicted_tps=833.0, predicted_cost_per_hour=13.35,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=1200, avg_output_tokens=300,
        )
        memory.record_outcome(
            decision_id=dec_id, job_id="job-1", status="succeeded",
            actual_tps=800.0,
        )
        result = query_memory(memory, model_name="Qwen")
        assert "PAST OUTCOMES" in result
        assert "L40S" in result

    def test_record_outcome_tool(self, memory):
        dec_id = memory.record_decision(
            job_id="job-1", model_name="test",
            instance_type="g6e", gpu_type="L40S",
            tp=4, pp=2, dp=1, num_gpus=8,
            predicted_tps=1000.0, predicted_cost_per_hour=10.0,
            slo_deadline_hours=4.0, objective="cheapest",
            avg_input_tokens=500, avg_output_tokens=200,
        )
        result = record_outcome_tool(
            memory, decision_id=dec_id, job_id="job-1",
            status="succeeded", actual_tps=950.0,
        )
        assert "Outcome recorded" in result
