"""Tests for koi/schemas.py — v2 data models."""

import pytest
from datetime import datetime

from koi.schemas import (
    TaskType, Objective, DataSource, MonitoringStatus,
    JobRequest, GPUResource, ResourceMap,
    EngineConfig, PlacementConfig, PredictedMetrics,
    RuntimeMetrics, AgentDecision, JobTracker, MonitoringTrigger,
)


class TestEnums:
    def test_task_type(self):
        assert TaskType.BATCH.value == "batch"
        assert TaskType.ONLINE.value == "online"

    def test_objective(self):
        assert Objective.CHEAPEST.value == "cheapest"
        assert Objective.FASTEST.value == "fastest"
        assert Objective.BALANCED.value == "balanced"

    def test_data_source_includes_memory(self):
        assert DataSource.MEMORY.value == "memory"
        assert DataSource.EXACT_MATCH.value == "exact_match"

    def test_monitoring_status(self):
        assert MonitoringStatus.WARMING_UP.value == "warming_up"
        assert MonitoringStatus.OVER_PROVISIONED.value == "over_provisioned"
        assert MonitoringStatus.FALLING_BEHIND.value == "falling_behind"
        assert MonitoringStatus.CHAIN_END.value == "chain_end"
        assert MonitoringStatus.LAUNCH_FAILED.value == "launch_failed"


class TestJobRequest:
    def test_basic(self):
        jr = JobRequest(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            avg_input_tokens=1200,
            avg_output_tokens=300,
            num_requests=5000,
            slo_deadline_hours=8.0,
        )
        assert jr.model_name == "Qwen/Qwen2.5-72B-Instruct"
        assert jr.task_type == TaskType.BATCH
        assert jr.objective == Objective.CHEAPEST
        assert jr.job_id.startswith("job-")

    def test_total_tokens(self):
        jr = JobRequest(
            model_name="test", avg_input_tokens=100, avg_output_tokens=50,
            num_requests=1000,
        )
        assert jr.total_tokens == 150_000

    def test_total_tokens_none_without_requests(self):
        jr = JobRequest(
            model_name="test", avg_input_tokens=100, avg_output_tokens=50,
        )
        assert jr.total_tokens is None

    def test_prefill_decode_ratio(self):
        jr = JobRequest(
            model_name="test", avg_input_tokens=1200, avg_output_tokens=300,
        )
        assert jr.prefill_decode_ratio == 4.0

    def test_required_tps(self):
        jr = JobRequest(
            model_name="test", avg_input_tokens=1000, avg_output_tokens=500,
            num_requests=5000, slo_deadline_hours=8.0,
        )
        expected = 5000 * 1500 / (8.0 * 3600)
        assert abs(jr.required_tps - expected) < 0.1

    def test_required_tps_none_without_slo(self):
        jr = JobRequest(
            model_name="test", avg_input_tokens=100, avg_output_tokens=50,
            num_requests=1000,
        )
        assert jr.required_tps is None

    def test_quantization_field(self):
        jr = JobRequest(
            model_name="test", avg_input_tokens=100, avg_output_tokens=50,
            quantization="fp8",
        )
        assert jr.quantization == "fp8"

    def test_cost_roofline_field(self):
        jr = JobRequest(
            model_name="test",
            avg_input_tokens=100,
            avg_output_tokens=50,
            cost_roofline_usd=42.5,
        )
        assert jr.cost_roofline_usd == 42.5


class TestGPUResource:
    def test_available_gpus(self):
        r = GPUResource(
            gpu_type="L40S", instance_type="g6e.12xlarge",
            gpus_per_instance=4, total_gpus=16, allocated_gpus=4,
            cost_per_instance_hour_usd=10.49, gpu_memory_gb=48.0,
            region="us-east-1", interconnect="PCIe",
        )
        assert r.available_gpus == 12

    def test_cost_per_gpu(self):
        r = GPUResource(
            gpu_type="A100-80GB", instance_type="p4de.24xlarge",
            gpus_per_instance=8, total_gpus=32, allocated_gpus=0,
            cost_per_instance_hour_usd=40.96, gpu_memory_gb=80.0,
            region="us-west-2", interconnect="NVLink",
        )
        assert abs(r.cost_per_gpu_hour_usd - 5.12) < 0.01


class TestResourceMap:
    def test_available_gpu_types(self):
        rm = ResourceMap(
            vpc_id="test", region="us-east-1",
            resources=[
                GPUResource(gpu_type="L40S", instance_type="g6e.12xlarge",
                            gpus_per_instance=4, total_gpus=16, allocated_gpus=0,
                            cost_per_instance_hour_usd=10.49, gpu_memory_gb=48.0,
                            region="us-east-1", interconnect="PCIe"),
                GPUResource(gpu_type="A10G", instance_type="g5.12xlarge",
                            gpus_per_instance=4, total_gpus=8, allocated_gpus=8,
                            cost_per_instance_hour_usd=5.67, gpu_memory_gb=24.0,
                            region="us-east-1", interconnect="PCIe"),
            ],
        )
        assert rm.available_gpu_types() == ["L40S"]
        assert rm.total_available_gpus() == 16

    def test_get_resource(self):
        rm = ResourceMap(
            vpc_id="test", region="us-east-1",
            resources=[
                GPUResource(gpu_type="H100", instance_type="p5.48xlarge",
                            gpus_per_instance=8, total_gpus=16, allocated_gpus=0,
                            cost_per_instance_hour_usd=98.32, gpu_memory_gb=80.0,
                            region="us-west-2", interconnect="NVLink"),
            ],
        )
        assert rm.get_resource("H100") is not None
        assert rm.get_resource("A100") is None


class TestPlacementConfig:
    def test_summary(self):
        cfg = PlacementConfig(
            gpu_type="L40S", instance_type="g6e.12xlarge",
            num_gpus=8, num_instances=2, tp=4, pp=2, dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
        )
        assert "L40S" in cfg.summary
        assert "TP=4" in cfg.summary
        assert cfg.gpus_per_replica == 8


class TestAgentDecision:
    def test_basic(self):
        cfg = PlacementConfig(
            gpu_type="A100-80GB", instance_type="p4de.24xlarge",
            num_gpus=8, num_instances=1, tp=4, pp=2, dp=1,
            region="us-west-2",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
        )
        dec = AgentDecision(
            job_id="job-test123",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            config=cfg,
            predicted_tps=2590.0,
            predicted_cost_per_hour=40.96,
            predicted_total_cost=14.34,
            predicted_runtime_hours=0.35,
            reasoning="PerfDB record shows A100-80GB TP=4 PP=2 gets 2590 TPS",
            confidence=0.88,
            data_source=DataSource.EXACT_MATCH,
            memory_hits=3,
            perfdb_records_used=12,
        )
        assert dec.job_id == "job-test123"
        assert dec.predicted_tps == 2590.0
        assert dec.memory_hits == 3
        assert dec.data_source == DataSource.EXACT_MATCH

    def test_cost_roofline_warning_fields(self):
        cfg = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=1),
        )
        dec = AgentDecision(
            job_id="job-test123",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            config=cfg,
            predicted_tps=1200.0,
            predicted_cost_per_hour=10.49,
            predicted_total_cost=24.28,
            predicted_runtime_hours=2.31,
            reasoning="test",
            confidence=0.8,
            meets_cost_roofline=False,
            cost_roofline_usd=20.0,
            projected_cost_overage_usd=4.28,
            cost_warning="Projected cost exceeds roofline.",
        )
        assert dec.meets_cost_roofline is False
        assert dec.cost_roofline_usd == 20.0
        assert dec.projected_cost_overage_usd == 4.28
        assert dec.cost_warning == "Projected cost exceeds roofline."


class TestJobTracker:
    def test_basic(self):
        cfg = PlacementConfig(
            gpu_type="L40S", instance_type="g6e.12xlarge",
            num_gpus=8, num_instances=2, tp=4, pp=2, dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
        )
        tracker = JobTracker(
            job_id="job-abc",
            config=cfg,
            slo_deadline_hours=8.0,
            total_tokens=7_500_000,
            predicted_tps=2590.0,
        )
        assert tracker.status == MonitoringStatus.WARMING_UP
        assert tracker.slo_headroom_pct == 100.0
        assert tracker.warmup_complete is False
        assert tracker.decision_id is None  # not linked yet

    def test_with_decision_id(self):
        cfg = PlacementConfig(
            gpu_type="L40S", instance_type="g6e.12xlarge",
            num_gpus=8, num_instances=2, tp=4, pp=2, dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
        )
        tracker = JobTracker(
            job_id="job-abc",
            decision_id="dec-12345678",
            config=cfg,
            slo_deadline_hours=8.0,
            total_tokens=7_500_000,
            predicted_tps=2590.0,
        )
        assert tracker.decision_id == "dec-12345678"


class TestMonitoringTrigger:
    def test_basic(self):
        trigger = MonitoringTrigger(
            trigger_type=MonitoringStatus.FALLING_BEHIND,
            job_id="job-abc",
            job_tracker={"slo_headroom_pct": 5.0, "smoothed_tps": 200},
            diagnosis_hint="TPS dropped below SLO threshold",
        )
        assert trigger.trigger_type == MonitoringStatus.FALLING_BEHIND
        assert trigger.diagnosis_hint != ""


class TestEngineConfig:
    def test_vllm_args(self):
        ec = EngineConfig(
            tensor_parallel_size=4, pipeline_parallel_size=2,
            quantization="fp8",
        )
        args = ec.to_vllm_args()
        assert "--tensor-parallel-size 4" in args
        assert "--quantization fp8" in args
