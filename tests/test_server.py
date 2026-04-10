"""Tests for koi/server.py — FastAPI endpoints."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from koi.schemas import (
    AgentDecision, PlacementConfig, EngineConfig, DataSource, MonitoringStatus,
)
from koi.server import app
from koi.resource_ledger import ResourceLedger
from koi.tools.memory import AgenticMemory
from koi.tools.perfdb import PerfDB


def _mock_decision(job_id="job-test123"):
    config = PlacementConfig(
        gpu_type="L40S", instance_type="g6e.12xlarge",
        num_gpus=8, num_instances=2, tp=4, pp=2, dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
    )
    return AgentDecision(
        job_id=job_id, model_name="Qwen/Qwen2.5-72B-Instruct",
        config=config,
        predicted_tps=833.0, predicted_cost_per_hour=13.35,
        predicted_total_cost=33.38, predicted_runtime_hours=2.5,
        reasoning="PerfDB shows L40S TP=4 PP=2 gets 833 TPS",
        confidence=0.85, data_source=DataSource.EXACT_MATCH,
    )


@pytest_asyncio.fixture
async def client():
    """Set up app state manually to avoid needing real API keys."""
    memory = AgenticMemory(db_path=":memory:")

    app.state.perfdb = MagicMock()
    app.state.perfdb.record_count = 307
    app.state.memory = memory
    app.state.ledger = ResourceLedger()
    app.state.orca = None
    app.state.agent = MagicMock()
    app.state.agent.model = "claude-sonnet-4-6"
    app.state.agent.decide = AsyncMock(return_value=_mock_decision())
    app.state.agent.handle_trigger = AsyncMock(return_value="ok")
    app.state.monitor = MagicMock()
    app.state.monitor.tracked_jobs = {}
    app.state.monitor.register_job = MagicMock()
    app.state.monitor.get_group_chains = MagicMock(return_value={})
    app.state.monitor.unregister_group = MagicMock(return_value=[])

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "2.0"
        assert "tracked_jobs" in body
        assert "memory_decisions" in body


class TestDecide:
    @pytest.mark.asyncio
    async def test_valid_decide(self, client):
        resp = await client.post("/decide", json={
            "job_request": {
                "model_name": "Qwen/Qwen2.5-72B-Instruct",
                "task_type": "batch",
                "avg_input_tokens": 953,
                "avg_output_tokens": 1024,
                "num_requests": 5000,
                "slo_deadline_hours": 8.0,
                "objective": "cheapest",
            },
            "resource_map": {
                "instances": [
                    {"instance_type": "g6e.12xlarge", "gpu_type": "L40S",
                     "gpus_per_instance": 4, "vcpus": 48, "quota_family": "G",
                     "gpu_memory_gb": 48.0, "cost_per_instance_hour_usd": 10.49}
                ],
                "quotas": [
                    {"family": "G", "region": "us-east-1", "market": "on_demand",
                     "baseline_vcpus": 192, "used_vcpus": 0}
                ],
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "job-test123"
        assert body["config"]["gpu_type"] == "L40S"

    @pytest.mark.asyncio
    async def test_empty_resources_422(self, client):
        resp = await client.post("/decide", json={
            "job_request": {
                "model_name": "test", "avg_input_tokens": 512, "avg_output_tokens": 256,
            },
            "resource_map": {"instances": [], "quotas": []},
        })
        assert resp.status_code == 422


class TestJobComplete:
    @pytest.mark.asyncio
    async def test_webhook(self, client):
        # Register a job first
        app.state.monitor.tracked_jobs = {
            "job-done": MagicMock(
                decision_id="dec-123",
                group_id=None,
                elapsed_hours=2.5,
                slo_headroom_pct=80.0,
            ),
        }

        resp = await client.post("/job/complete", json={
            "job_id": "job-done",
            "status": "succeeded",
            "metrics": {"avg_generation_throughput_toks_per_s": 1500.0},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"

    @pytest.mark.asyncio
    async def test_unknown_job(self, client):
        app.state.monitor.tracked_jobs = {}
        resp = await client.post("/job/complete", json={
            "job_id": "job-unknown",
            "status": "succeeded",
            "metrics": {},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown_job"


class TestLaunchFailed:
    @pytest.mark.asyncio
    async def test_records_failures(self, client):
        app.state.monitor.tracked_jobs = {}
        resp = await client.post("/job/launch-failed", json={
            "job_id": "job-fail1",
            "configs_tried": [
                {"gpu_type": "A100-80GB", "instance_type": "p4de.24xlarge", "region": "us-west-2"},
                {"gpu_type": "L40S", "instance_type": "g6e.12xlarge", "region": "us-east-1"},
            ],
            "failure_reasons": ["InsufficientCapacity", "QuotaExceeded"],
            "total_time_seconds": 180.0,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["attempts_recorded"] == 2


class TestClassifyFailure:
    def test_spot_preemption(self):
        from koi.server import _classify_failure
        assert _classify_failure("SpotInstanceInterruption") == "spot_preemption"
        assert _classify_failure("spot preempted by EC2") == "spot_preemption"

    def test_no_capacity(self):
        from koi.server import _classify_failure
        assert _classify_failure("InsufficientCapacity in us-east-1") == "no_capacity"

    def test_oom(self):
        from koi.server import _classify_failure
        assert _classify_failure("CUDA OOM: tried to allocate 40GB") == "oom"
        assert _classify_failure("OutOfMemoryError") == "oom"

    def test_quota(self):
        from koi.server import _classify_failure
        assert _classify_failure("QuotaExceeded for p5.48xlarge") == "quota"

    def test_unknown(self):
        from koi.server import _classify_failure
        assert _classify_failure("some random error") == "unknown"
        assert _classify_failure("") == "unknown"


class TestListJobs:
    @pytest.mark.asyncio
    async def test_empty(self, client):
        app.state.monitor.tracked_jobs = {}
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json()["tracked_jobs"] == 0
