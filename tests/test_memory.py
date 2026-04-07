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

    def test_new_fields(self, memory):
        dec_id = memory.record_decision(
            job_id="job-retry", model_name="test-model",
            instance_type="g6e.12xlarge", gpu_type="L40S",
            tp=4, pp=2, dp=1, num_gpus=8,
            predicted_tps=833.0, predicted_cost_per_hour=13.35,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=500, avg_output_tokens=200,
            triggered_by="slo_violation",
            parent_decision_id="dec-original",
            market="spot",
        )
        results = memory.query_decisions(model_name="test-model")
        assert results[0]["triggered_by"] == "slo_violation"
        assert results[0]["parent_decision_id"] == "dec-original"
        assert results[0]["market"] == "spot"


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
            diagnosis="OOM at batch_size=256, KV cache exceeded 40GB VRAM",
            failure_category="oom",
            bottleneck="memory_bound",
        )

        failed = memory.query_outcomes(status="failed")
        assert len(failed) == 1
        assert failed[0]["diagnosis"] == "OOM at batch_size=256, KV cache exceeded 40GB VRAM"
        assert failed[0]["failure_category"] == "oom"
        assert failed[0]["bottleneck"] == "memory_bound"

    def test_outcome_with_diagnosis_and_bottleneck(self, memory):
        dec_id = memory.record_decision(
            job_id="job-diag", model_name="test",
            instance_type="g6e", gpu_type="L40S",
            tp=4, pp=2, dp=1, num_gpus=8,
            predicted_tps=800.0, predicted_cost_per_hour=13.35,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=500, avg_output_tokens=200,
        )
        memory.record_outcome(
            decision_id=dec_id, job_id="job-diag", status="succeeded",
            actual_tps=833.0, actual_cost_per_hour=13.35,
            diagnosis="Ran smoothly, slight overperformance vs prediction",
            bottleneck="memory_bound",
            diff_from_parent='{"tp": {"old": 2, "new": 4}}',
        )
        outcomes = memory.query_outcomes(model_name="test")
        assert outcomes[0]["diagnosis"] is not None
        assert outcomes[0]["bottleneck"] == "memory_bound"
        assert "tp" in outcomes[0]["diff_from_parent"]

    def test_query_memory_shows_diagnosis(self, memory):
        dec_id = memory.record_decision(
            job_id="job-fail2", model_name="Qwen-72B",
            instance_type="p4d.24xlarge", gpu_type="A100-40GB",
            tp=4, pp=1, dp=1, num_gpus=8,
            predicted_tps=900.0, predicted_cost_per_hour=32.0,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=500, avg_output_tokens=200,
        )
        memory.record_outcome(
            decision_id=dec_id, job_id="job-fail2", status="failed",
            diagnosis="A100-40GB TP=4: 144GB/4=36GB > 40GB VRAM, OOM",
            failure_category="oom",
            bottleneck="memory_bound",
        )
        result = query_memory(memory, model_name="Qwen-72B")
        assert "memory_bound" in result
        assert "OOM" in result


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


class TestCounts:
    def test_initial_counts(self, memory):
        assert memory.decision_count() == 0
        assert memory.outcome_count() == 0


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

    def test_decision_lineage(self, memory):
        """Parent→child decision linking for retry chains."""
        dec1 = memory.record_decision(
            job_id="job-abc", model_name="Qwen-72B",
            instance_type="g6e.12xlarge", gpu_type="L40S",
            tp=4, pp=2, dp=1, num_gpus=8,
            predicted_tps=870.0, predicted_cost_per_hour=13.35,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=500, avg_output_tokens=200,
            triggered_by="user",
        )
        dec2 = memory.record_decision(
            job_id="job-abc", model_name="Qwen-72B",
            instance_type="p4de.24xlarge", gpu_type="A100-80GB",
            tp=8, pp=1, dp=1, num_gpus=8,
            predicted_tps=1500.0, predicted_cost_per_hour=40.96,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=500, avg_output_tokens=200,
            triggered_by="slo_violation",
            parent_decision_id=dec1,
        )
        results = memory.query_decisions(model_name="Qwen-72B")
        assert len(results) == 2
        child = [r for r in results if r["parent_decision_id"] is not None][0]
        assert child["parent_decision_id"] == dec1
        assert child["triggered_by"] == "slo_violation"
