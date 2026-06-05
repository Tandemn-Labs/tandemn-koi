"""Tests for koi/server.py — FastAPI endpoints."""

import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from contextlib import asynccontextmanager

from koi.schemas import (
    AgentDecision,
    PlacementConfig,
    EngineConfig,
    DataSource,
    MonitoringStatus,
)
from koi.server import app
from koi.resource_ledger import ResourceLedger
from koi.runtime_state import RuntimeStateStore
from koi.tools.memory import AgenticMemory
from koi.tools.perfdb import PerfDB


def _mock_decision(job_id="job-test123"):
    config = PlacementConfig(
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        num_gpus=8,
        num_instances=2,
        tp=4,
        pp=2,
        dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
        market="on_demand",
    )
    return AgentDecision(
        job_id=job_id,
        model_name="Qwen/Qwen2.5-72B-Instruct",
        config=config,
        planned_market="on_demand",
        predicted_tps=833.0,
        predicted_cost_per_hour=13.35,
        predicted_total_cost=33.38,
        predicted_runtime_hours=2.5,
        meets_cost_roofline=False,
        cost_roofline_usd=30.0,
        projected_cost_overage_usd=3.38,
        cost_warning="Projected cost exceeds roofline, but this is the cheapest SLO-meeting plan.",
        reasoning="PerfDB shows L40S TP=4 PP=2 gets 833 TPS",
        confidence=0.85,
        data_source=DataSource.EXACT_MATCH,
    )


@pytest_asyncio.fixture
async def client():
    """Set up app state manually to avoid needing real API keys."""
    import asyncio

    memory = AgenticMemory(db_path=":memory:")

    app.state.perfdb = MagicMock()
    app.state.perfdb.record_count = 307
    app.state.memory = memory
    app.state.runtime_state = RuntimeStateStore(":memory:")
    app.state.ledger = ResourceLedger()
    app.state.decide_lock = asyncio.Lock()
    app.state.orca = None
    app.state.agent = MagicMock()
    app.state.agent.model = "claude-sonnet-4-6"
    app.state.agent.decide = AsyncMock(return_value=_mock_decision())
    app.state.agent.handle_trigger = AsyncMock(return_value="ok")
    # Cold-start failure recovery — default to a no-op response. Tests that
    # exercise the recovery path override this with their own AsyncMock.
    app.state.agent.recover_from_startup_failure = AsyncMock(return_value="ok")
    monitor = MagicMock()
    monitor.tracked_jobs = {}
    monitor._pending_launches = {}
    monitor._pending_replica_decisions = {}
    monitor._fatal = None

    def _track_pending_launch(job_id, launch_info):
        merged = dict(monitor._pending_launches.get(job_id, {}))
        merged.update(launch_info)
        monitor._pending_launches[job_id] = merged

    def _get_pending_launch(job_id):
        return monitor._pending_launches.get(job_id, {})

    def _clear_pending_launch(job_id):
        return monitor._pending_launches.pop(job_id, None)

    def _clear_pending_launches_for_group(group_id):
        removed = [
            job_id
            for job_id, launch in list(monitor._pending_launches.items())
            if launch.get("group_id") == group_id
        ]
        for job_id in removed:
            monitor._pending_launches.pop(job_id, None)
        return len(removed)

    def _consume_pending_replica_decision(replica_id):
        return monitor._pending_replica_decisions.pop(replica_id, None)

    def _register_pending_replica_decision(
        replica_id, decision_id, scale_request_id=None, decision=None
    ):
        monitor._pending_replica_decisions[replica_id] = {
            "decision_id": decision_id,
            "scale_request_id": scale_request_id,
            "decision": dict(decision or {}),
        }

    monitor.track_pending_launch = MagicMock(side_effect=_track_pending_launch)
    monitor.get_pending_launch = MagicMock(side_effect=_get_pending_launch)
    monitor.clear_pending_launch = MagicMock(side_effect=_clear_pending_launch)
    monitor.clear_pending_launches_for_group = MagicMock(
        side_effect=_clear_pending_launches_for_group
    )
    monitor.consume_pending_replica_decision = MagicMock(
        side_effect=_consume_pending_replica_decision
    )
    monitor.register_pending_replica_decision = MagicMock(
        side_effect=_register_pending_replica_decision
    )
    monitor.persist_job = MagicMock()
    monitor.register_job = MagicMock()
    monitor.unregister_job = MagicMock(
        side_effect=lambda job_id: monitor.tracked_jobs.pop(job_id, None)
    )
    monitor.get_group_chains = MagicMock(return_value={})
    monitor.unregister_group = MagicMock(return_value=[])
    app.state.monitor = monitor

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@asynccontextmanager
async def _lifespan_client():
    from koi.server import lifespan

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with lifespan(app):
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
        resp = await client.post(
            "/decide",
            json={
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
                        {
                            "instance_type": "g6e.12xlarge",
                            "gpu_type": "L40S",
                            "gpus_per_instance": 4,
                            "vcpus": 48,
                            "quota_family": "G",
                            "gpu_memory_gb": 48.0,
                            "cost_per_instance_hour_usd": 10.49,
                        }
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 192,
                            "used_vcpus": 0,
                        }
                    ],
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "job-test123"
        assert body["config"]["gpu_type"] == "L40S"
        assert body["planned_market"] == "on_demand"
        pending = app.state.ledger.summary()
        assert len(pending) == 1
        assert pending[0]["region"] == "us-east-1"

    @pytest.mark.asyncio
    async def test_decide_accepts_cost_roofline_field(self, client):
        captured = {}

        async def decide_capture(job_request, resource_map):
            captured["cost_roofline_usd"] = job_request.cost_roofline_usd
            return _mock_decision(job_id=job_request.job_id)

        app.state.agent.decide = decide_capture

        resp = await client.post(
            "/decide",
            json={
                "job_request": {
                    "model_name": "Qwen/Qwen2.5-72B-Instruct",
                    "task_type": "batch",
                    "avg_input_tokens": 953,
                    "avg_output_tokens": 1024,
                    "num_requests": 5000,
                    "slo_deadline_hours": 8.0,
                    "objective": "cheapest",
                    "cost_roofline_usd": 120.0,
                },
                "resource_map": {
                    "instances": [
                        {
                            "instance_type": "g6e.12xlarge",
                            "gpu_type": "L40S",
                            "gpus_per_instance": 4,
                            "vcpus": 48,
                            "quota_family": "G",
                            "gpu_memory_gb": 48.0,
                            "cost_per_instance_hour_usd": 10.49,
                        }
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 192,
                            "used_vcpus": 0,
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 200
        assert captured["cost_roofline_usd"] == 120.0

    @pytest.mark.asyncio
    async def test_decide_returns_cost_roofline_warning_fields(self, client):
        resp = await client.post(
            "/decide",
            json={
                "job_request": {
                    "model_name": "Qwen/Qwen2.5-72B-Instruct",
                    "task_type": "batch",
                    "avg_input_tokens": 953,
                    "avg_output_tokens": 1024,
                    "num_requests": 5000,
                    "slo_deadline_hours": 8.0,
                    "objective": "cheapest",
                    "cost_roofline_usd": 30.0,
                },
                "resource_map": {
                    "instances": [
                        {
                            "instance_type": "g6e.12xlarge",
                            "gpu_type": "L40S",
                            "gpus_per_instance": 4,
                            "vcpus": 48,
                            "quota_family": "G",
                            "gpu_memory_gb": 48.0,
                            "cost_per_instance_hour_usd": 10.49,
                        }
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 192,
                            "used_vcpus": 0,
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["meets_cost_roofline"] is False
        assert body["cost_roofline_usd"] == 30.0
        assert body["projected_cost_overage_usd"] == 3.38
        assert "Projected cost exceeds roofline" in body["cost_warning"]

    @pytest.mark.asyncio
    async def test_decide_preserves_incoming_job_id(self, client):
        captured = {}

        async def decide_capture(job_request, resource_map):
            captured["job_id"] = job_request.job_id
            return _mock_decision(job_id=job_request.job_id)

        app.state.agent.decide = decide_capture

        resp = await client.post(
            "/decide",
            json={
                "job_request": {
                    "job_id": "demo-session-123",
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
                        {
                            "instance_type": "g6e.12xlarge",
                            "gpu_type": "L40S",
                            "gpus_per_instance": 4,
                            "vcpus": 48,
                            "quota_family": "G",
                            "gpu_memory_gb": 48.0,
                            "cost_per_instance_hour_usd": 10.49,
                        }
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 192,
                            "used_vcpus": 0,
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 200
        assert captured["job_id"] == "demo-session-123"
        assert resp.json()["job_id"] == "demo-session-123"

    @pytest.mark.asyncio
    async def test_decide_rejects_unknown_job_request_fields(self, client):
        resp = await client.post(
            "/decide",
            json={
                "job_request": {
                    "model_name": "Qwen/Qwen2.5-72B-Instruct",
                    "avg_input_tokens": 953,
                    "avg_output_tokens": 1024,
                    "num_requests": 5000,
                    "mystery_field": 123,
                },
                "resource_map": {
                    "instances": [
                        {
                            "instance_type": "g6e.12xlarge",
                            "gpu_type": "L40S",
                            "gpus_per_instance": 4,
                            "vcpus": 48,
                            "quota_family": "G",
                            "gpu_memory_gb": 48.0,
                            "cost_per_instance_hour_usd": 10.49,
                        }
                    ],
                    "quotas": [
                        {
                            "family": "G",
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 192,
                            "used_vcpus": 0,
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 400
        assert "mystery_field" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_empty_resources_422(self, client):
        resp = await client.post(
            "/decide",
            json={
                "job_request": {
                    "model_name": "test",
                    "avg_input_tokens": 512,
                    "avg_output_tokens": 256,
                },
                "resource_map": {"instances": [], "quotas": []},
            },
        )
        assert resp.status_code == 422


class TestDecideConcurrency:
    @pytest.mark.asyncio
    async def test_parallel_decide_requests_do_not_double_book(self, client):
        """Concurrent /decide calls should not see the same 8 GPUs as free."""
        import asyncio

        seen_available = []

        async def decide_once(job_request, resource_map):
            resource = resource_map.get_resource("L40S")
            seen_available.append(resource.available_gpus if resource else None)
            await asyncio.sleep(0.05)
            if not resource or resource.available_gpus < 8:
                raise RuntimeError("insufficient adjusted resources")
            return _mock_decision(job_id=f"job-{len(seen_available)}")

        app.state.agent.decide = decide_once

        payload = {
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
                        "region": "us-east-1",
                        "market": "on_demand",
                        "baseline_vcpus": 96,
                        "used_vcpus": 0,
                    },
                ],
            },
        }

        resp1, resp2 = await asyncio.gather(
            client.post("/decide", json=payload),
            client.post("/decide", json=payload),
        )

        status_codes = sorted([resp1.status_code, resp2.status_code])
        assert status_codes == [200, 500]
        assert seen_available == [8, 0]
        assert app.state.ledger.pending_count == 1


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

        resp = await client.post(
            "/job/complete",
            json={
                "job_id": "job-done",
                "status": "succeeded",
                "metrics": {"avg_generation_throughput_toks_per_s": 1500.0},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"

    @pytest.mark.asyncio
    async def test_unknown_job(self, client):
        app.state.monitor.tracked_jobs = {}
        resp = await client.post(
            "/job/complete",
            json={
                "job_id": "job-unknown",
                "status": "succeeded",
                "metrics": {},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown_job"


class TestLaunchFailed:
    @pytest.mark.asyncio
    async def test_records_failures(self, client):
        app.state.monitor.tracked_jobs = {}
        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "job-fail1",
                "configs_tried": [
                    {
                        "gpu_type": "A100-80GB",
                        "instance_type": "p4de.24xlarge",
                        "region": "us-west-2",
                    },
                    {
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "region": "us-east-1",
                    },
                ],
                "failure_reasons": ["InsufficientCapacity", "QuotaExceeded"],
                "total_time_seconds": 180.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["attempts_recorded"] == 2

    @pytest.mark.asyncio
    async def test_releases_ledger_and_unregisters_job(self, client):
        """All-failed launches should release reservations and unregister the job."""
        app.state.ledger.reserve("dec-launchfail", "L40S", 8, region="us-east-1")
        app.state.monitor.tracked_jobs = {
            "job-fail2": MagicMock(decision_id="dec-launchfail"),
        }
        app.state.monitor.unregister_job = MagicMock()

        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "job-fail2",
                "configs_tried": [
                    {
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "region": "us-east-1",
                        "market": "spot",
                    },
                    {
                        "gpu_type": "A100-80GB",
                        "instance_type": "p4de.24xlarge",
                        "region": "us-west-2",
                        "market": "on_demand",
                    },
                ],
                # Use unrecoverable categories so this test exercises the
                # legacy "recorded" path, not the new agent-driven retry.
                # Recovery is covered separately in TestLaunchFailedRecovery.
                "failure_reasons": ["QuotaExceeded", "QuotaExceeded"],
                "total_time_seconds": 240.0,
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["attempts_recorded"] == 2
        assert app.state.ledger.pending_count == 0
        app.state.monitor.unregister_job.assert_called_once_with("job-fail2")

        spot = app.state.memory.get_failure_summary(
            "L40S",
            region="us-east-1",
            market="spot",
        )
        on_demand = app.state.memory.get_failure_summary(
            "A100-80GB",
            region="us-west-2",
            market="on_demand",
        )
        assert spot["effective_observations"] == 1
        assert spot["availability_pct"] < 50.0
        assert on_demand["effective_observations"] == 1
        assert on_demand["availability_pct"] < 50.0

    @pytest.mark.asyncio
    async def test_releases_ledger_without_tracked_job_when_decision_id_provided(
        self, client
    ):
        """decision_id should release pending GPUs before the job is registered."""
        app.state.ledger.reserve("dec-launchfail-direct", "L40S", 8, region="us-east-1")
        app.state.monitor.tracked_jobs = {}

        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "job-fail3",
                "decision_id": "dec-launchfail-direct",
                "configs_tried": [
                    {
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "region": "us-east-1",
                        "market": "spot",
                    },
                ],
                "failure_reasons": ["InsufficientCapacity"],
                "total_time_seconds": 45.0,
            },
        )

        assert resp.status_code == 200
        assert resp.json()["attempts_recorded"] == 1
        assert app.state.ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_clears_pending_launches_for_group(self, client):
        """All-failed chunked launches should clear any pending launch rows for the group."""
        app.state.monitor._pending_launches = {
            "replica-r0": {
                "group_id": "job-fail-group",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
            },
            "replica-r1": {
                "group_id": "job-fail-group",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
            },
            "replica-other": {
                "group_id": "other-job",
                "gpu_type": "A100-80GB",
                "instance_type": "p4de.24xlarge",
            },
        }

        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "job-fail-group",
                "configs_tried": [
                    {
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "region": "us-east-1",
                        "market": "spot",
                    },
                ],
                "failure_reasons": ["InsufficientCapacity"],
                "total_time_seconds": 45.0,
            },
        )

        assert resp.status_code == 200
        assert "replica-r0" not in app.state.monitor._pending_launches
        assert "replica-r1" not in app.state.monitor._pending_launches
        assert "replica-other" in app.state.monitor._pending_launches
        app.state.monitor.clear_pending_launches_for_group.assert_called_once_with(
            "job-fail-group"
        )

    @pytest.mark.asyncio
    async def test_p1_harness_appends_recovery_when_enabled(self, client, monkeypatch):
        import koi.harness.p1 as p1

        dec_id = app.state.memory.record_decision(
            job_id="job-p1-server",
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
            market="spot",
        )

        async def fake_recovery(agent, req, memory, **kwargs):
            assert req.decision_id == dec_id
            return {"action": "abort", "reasoning": "test recovery"}

        monkeypatch.setenv("KOI_HARNESS", "1")
        monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p1")
        monkeypatch.setattr(p1, "run_launch_recovery", fake_recovery)

        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "job-p1-server",
                "decision_id": dec_id,
                "configs_tried": [
                    {
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "region": "us-east-1",
                        "market": "spot",
                    }
                ],
                "failure_reasons": ["InsufficientCapacity"],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["recovery"] == {"action": "abort", "reasoning": "test recovery"}

    @pytest.mark.asyncio
    async def test_p1_harness_fail_open_preserves_legacy_response(self, client, monkeypatch):
        import koi.harness.p1 as p1

        async def boom(*args, **kwargs):
            raise RuntimeError("p1 exploded")

        monkeypatch.setenv("KOI_HARNESS", "1")
        monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p1")
        monkeypatch.setenv("KOI_HARNESS_FAIL_OPEN", "1")
        monkeypatch.setattr(p1, "run_launch_recovery", boom)

        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "job-p1-open",
                "configs_tried": [
                    {
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "region": "us-east-1",
                        "market": "spot",
                    }
                ],
                "failure_reasons": ["InsufficientCapacity"],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["attempts_recorded"] == 1
        assert "recovery" not in body


class TestReplicaFailed:
    @pytest.mark.asyncio
    async def test_captures_tps_before_zeroing(self, client):
        """actual_tps should be recorded from smoothed_tps BEFORE it's set to 0."""
        import asyncio
        from koi.schemas import JobTracker, MonitoringStatus

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
        )
        tracker = JobTracker(
            job_id="r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
            tokens_remaining=5_000_000,
        )
        tracker.smoothed_tps = 1850.0  # was running at 1850 TPS before death
        # Record a decision so the outcome JOIN works
        dec_id = app.state.memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen3-32B",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
        )
        tracker.decision_id = dec_id

        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r0",
                "group_id": "parent-job",
                "status": "failed",
                "reason": "Heartbeat timeout (45s)",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "trigger_emitted"

        # Verify tracker is zeroed
        assert tracker.smoothed_tps == 0
        assert tracker.status == MonitoringStatus.FAILED

        # Verify outcome was recorded with actual_tps=1850
        outcomes = app.state.memory.query_outcomes(status="replica_failed")
        assert len(outcomes) >= 1
        last = outcomes[0]
        assert last["actual_tps"] == 1850.0

    @pytest.mark.asyncio
    async def test_dedup_already_failed(self, client):
        """Second /job/replica-failed for same ID returns already_failed."""
        import asyncio
        from koi.schemas import JobTracker, MonitoringStatus

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
        )
        tracker = JobTracker(
            job_id="r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
        )
        tracker.status = MonitoringStatus.FAILED  # already dead

        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r0",
                "group_id": "parent-job",
                "status": "failed",
                "reason": "Heartbeat timeout (45s)",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "already_failed"

    @pytest.mark.asyncio
    async def test_dedup_real_orca_launcher_then_watchdog(self, client):
        """Launcher and watchdog can both report the same dead replica; Koi should process it once."""
        import asyncio
        from koi.schemas import JobTracker, MonitoringStatus

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
        )
        tracker = JobTracker(
            job_id="r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
        )
        tracker.smoothed_tps = 900.0

        dec_id = app.state.memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen3-32B",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
        )
        tracker.decision_id = dec_id

        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        launcher_resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r0",
                "group_id": "parent-job",
                "status": "failed",
                "reason": "Clean exit with pending chunks (likely killed)",
            },
        )
        watchdog_resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r0",
                "group_id": "parent-job",
                "status": "failed",
                "reason": "Heartbeat timeout (45s)",
            },
        )

        assert launcher_resp.status_code == 200
        assert launcher_resp.json()["status"] == "trigger_emitted"
        assert watchdog_resp.status_code == 200
        assert watchdog_resp.json()["status"] == "already_failed"

        assert tracker.status == MonitoringStatus.FAILED
        assert app.state.monitor._trigger_queue.qsize() == 1

        outcomes = app.state.memory.query_outcomes(status="replica_failed")
        assert len(outcomes) == 1
        assert outcomes[0]["job_id"] == "parent-job"
        assert outcomes[0]["actual_tps"] == 900.0
        assert (
            outcomes[0]["diagnosis"] == "Clean exit with pending chunks (likely killed)"
        )

    @pytest.mark.asyncio
    async def test_real_monitor_poll_then_webhook_no_self_fight(self):
        """End-to-end production race: drive both observers — the real
        MonitoringLoop._poll_job AND the real /job/replica-failed handler —
        against the same MonitoringLoop. Whichever runs first must consume
        the marker without leaving the other in a FAILED-trigger state.

        This mirrors the actual production sequence: real Orca returns
        phase='killed' to the poll AND emits /job/replica-failed; whichever
        Koi observes first must keep the system out of self-fight."""
        from koi.monitor import MonitoringLoop
        from koi.schemas import MonitoringStatus
        from koi.tools.memory import AgenticMemory
        from koi.server import _replica_failed_impl, ReplicaFailedRequest

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
            market="on_demand",
        )

        orca = MagicMock()
        orca.get_replicas = AsyncMock(
            return_value={
                "replicas": [
                    {"replica_id": "prod-r0", "phase": "killed"},
                ]
            }
        )

        memory = AgenticMemory(db_path=":memory:")
        monitor = MonitoringLoop(orca=orca)
        decision_id = memory.record_decision(
            job_id="prod-group",
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
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            market="on_demand",
        )
        monitor.register_job(
            job_id="prod-r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
            decision_id=decision_id,
            group_id="prod-group",
        )
        monitor.tracked_jobs["prod-r0"].smoothed_tps = 1200.0

        # Pscale just decided to scale-down: register the marker.
        monitor._koi_initiated_kills.add("prod-r0")

        # Wire app.state for the real /job/replica-failed handler.
        original_monitor = app.state.monitor
        original_memory = app.state.memory
        app.state.monitor = monitor
        app.state.memory = memory
        try:
            # Observer #1 wins: monitor poll observes the killed replica
            # FIRST and consumes the marker.
            await monitor._poll_job("prod-r0")
            tracker = monitor.tracked_jobs["prod-r0"]
            assert tracker.status == MonitoringStatus.COMPLETED
            assert "prod-r0" not in monitor._koi_initiated_kills
            assert monitor._trigger_queue.empty()

            # Observer #2 arrives: real Orca's /job/replica-failed webhook.
            req = ReplicaFailedRequest(
                job_id="prod-r0",
                group_id="prod-group",
                status="failed",
                reason="Orca observed replica killed",
            )
            response = await _replica_failed_impl(req)

            # Webhook must recognize it as the same intentional kill and
            # NOT re-emit a FAILED trigger that would scale-up.
            assert response["status"] == "intentional_kill"
            assert monitor._trigger_queue.empty()
            assert tracker.status == MonitoringStatus.COMPLETED
        finally:
            app.state.monitor = original_monitor
            app.state.memory = original_memory

    @pytest.mark.asyncio
    async def test_real_webhook_then_monitor_poll_no_double_trigger(self):
        """Reverse order: webhook arrives first, then the 10s poll runs.
        Poll must short-circuit on the COMPLETED status and not re-emit."""
        from koi.monitor import MonitoringLoop
        from koi.schemas import MonitoringStatus
        from koi.tools.memory import AgenticMemory
        from koi.server import _replica_failed_impl, ReplicaFailedRequest

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
            market="on_demand",
        )
        orca = MagicMock()
        orca.get_replicas = AsyncMock(
            return_value={
                "replicas": [
                    {"replica_id": "prod-r0", "phase": "killed"},
                ]
            }
        )
        memory = AgenticMemory(db_path=":memory:")
        monitor = MonitoringLoop(orca=orca)
        decision_id = memory.record_decision(
            job_id="prod-group",
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
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            market="on_demand",
        )
        monitor.register_job(
            job_id="prod-r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
            decision_id=decision_id,
            group_id="prod-group",
        )
        monitor.tracked_jobs["prod-r0"].smoothed_tps = 1200.0
        monitor._koi_initiated_kills.add("prod-r0")

        original_monitor = app.state.monitor
        original_memory = app.state.memory
        app.state.monitor = monitor
        app.state.memory = memory
        try:
            req = ReplicaFailedRequest(
                job_id="prod-r0",
                group_id="prod-group",
                status="failed",
                reason="Orca observed replica killed",
            )
            response = await _replica_failed_impl(req)
            assert response["status"] == "intentional_kill"
            assert monitor._trigger_queue.empty()

            # Now the 10s poll runs. Tracker is COMPLETED and replica is in
            # dead_replicas; the poll must NOT re-emit anything.
            await monitor._poll_job("prod-r0")
            assert monitor._trigger_queue.empty()
        finally:
            app.state.monitor = original_monitor
            app.state.memory = original_memory

    @pytest.mark.asyncio
    async def test_intentional_kill_after_monitor_consumed_marker(self, client):
        """Regression: the monitor poll can consume `_koi_initiated_kills` first
        and mark the tracker COMPLETED. A subsequent /job/replica-failed for
        the same replica must still be treated as an intentional kill, not
        re-trigger a FAILED scale-up (the "self-fight" race)."""
        import asyncio
        from koi.schemas import JobTracker, MonitoringStatus

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
            market="on_demand",
        )
        tracker = JobTracker(
            job_id="r-killed",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
        )
        # Monitor poll already saw r-killed dead and applied the koi-kill
        # cleanup: status = COMPLETED, replica added to dead_replicas, marker
        # discarded. /job/replica-failed must NOT re-emit a FAILED trigger.
        tracker.status = MonitoringStatus.COMPLETED
        tracker.dead_replicas.append("r-killed")
        tracker.smoothed_tps = 0

        app.state.monitor.tracked_jobs = {"r-killed": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r-killed",
                "group_id": "parent-job",
                "status": "failed",
                "reason": "Orca observed replica killed",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "intentional_kill"
        assert app.state.monitor._trigger_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_p5c_harness_records_postmortem_and_cooloff(self, client, monkeypatch):
        """When the p5c flag is on, replica-failed should attach a postmortem
        and write an active cooloff for the failed scope."""
        import asyncio
        from koi.harness import p5c as p5c_module
        from koi.harness.p5c import P5cDiagnosis
        from koi.schemas import JobTracker, MonitoringStatus

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
            market="spot",
        )
        tracker = JobTracker(
            job_id="r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
        )
        tracker.smoothed_tps = 1180.0

        dec_id = app.state.memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen3-32B",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            market="spot",
        )
        tracker.decision_id = dec_id

        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        import time as _time

        now = _time.time()

        async def _stub_postmortem(**kwargs):
            return P5cDiagnosis(
                diagnosis_code="spot_preemption",
                bottleneck="market_capacity",
                next_fix="retry_same_topology_on_demand",
                failure_scope="L40S|g6e.12xlarge|us-east-1|spot",
                event_at=now,
                avoid_until=now + 30 * 60,
                hard_until=now + 10 * 60,
                cooloff_key="L40S|g6e.12xlarge|us-east-1|spot",
                cooloff_minutes=30,
                rationale="seeded postmortem",
            )

        monkeypatch.setattr(p5c_module, "run_chain_postmortem", _stub_postmortem)
        monkeypatch.setenv("KOI_HARNESS", "1")
        monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p5c")

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r0",
                "group_id": "parent-job",
                "status": "failed",
                "reason": "SpotInstanceInterruption",
                "market": "spot",
                "region": "us-east-1",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "trigger_emitted"
        assert body["postmortem"]["diagnosis_code"] == "spot_preemption"
        assert body["postmortem"]["cooloff_minutes"] == 30
        assert tracker.status == MonitoringStatus.FAILED

        outcomes = app.state.memory.query_outcomes(status="replica_failed")
        assert outcomes
        assert outcomes[0]["bottleneck"] == "market_capacity"
        assert outcomes[0]["diff_from_parent"] is not None
        assert "spot_preemption" in outcomes[0]["diagnosis"]

        cooloffs = app.state.memory.get_active_cooloffs(
            gpu_type="L40S", region="us-east-1", market="spot"
        )
        assert len(cooloffs) == 1
        assert cooloffs[0]["diagnosis_code"] == "spot_preemption"

    @pytest.mark.asyncio
    async def test_p4_recovery_appends_plan_when_p4_flag_enabled(self, client, monkeypatch):
        """When p4 flag is on, the response should include a recovery plan
        produced by run_replica_recovery."""
        import asyncio
        from koi.harness import p4 as p4_module
        from koi.schemas import JobTracker, MonitoringStatus

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
            market="spot",
        )
        tracker = JobTracker(
            job_id="r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
        )
        tracker.smoothed_tps = 1180.0

        dec_id = app.state.memory.record_decision(
            job_id="parent-job-p4",
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
            avg_input_tokens=512,
            avg_output_tokens=512,
            num_requests=5000,
            market="spot",
        )
        tracker.decision_id = dec_id

        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        async def _stub_recovery(**kwargs):
            assert kwargs["region"] == "us-east-1"
            assert kwargs["market"] == "spot"
            return {
                "action": "replace_market",
                "decision_id": "dec-recovery-1",
                "parent_decision_id": dec_id,
                "config": {
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 1,
                    "dp": 1,
                    "market": "on_demand",
                    "region": "us-east-1",
                },
                "reasoning": "p4 stub plan",
                "confidence": 0.7,
                "retry_budget_remaining": 2,
            }

        monkeypatch.setattr(p4_module, "run_replica_recovery", _stub_recovery)
        monkeypatch.setenv("KOI_HARNESS", "1")
        monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p4")

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r0",
                "group_id": "parent-job-p4",
                "status": "failed",
                "reason": "SpotInstanceInterruption",
                "region": "us-east-1",
                "market": "spot",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "trigger_emitted"
        assert body["recovery"]["action"] == "replace_market"
        assert body["recovery"]["parent_decision_id"] == dec_id
        assert tracker.status == MonitoringStatus.FAILED

    @pytest.mark.asyncio
    async def test_p4_recovery_fail_open_preserves_legacy_behavior(self, client, monkeypatch):
        """A P4 reasoner crash with fail-open must not break the webhook contract."""
        import asyncio
        from koi.harness import p4 as p4_module
        from koi.schemas import JobTracker

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
        )
        tracker = JobTracker(
            job_id="r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
        )
        tracker.smoothed_tps = 1100.0
        dec_id = app.state.memory.record_decision(
            job_id="parent-job-p4",
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
            avg_input_tokens=512,
            avg_output_tokens=512,
        )
        tracker.decision_id = dec_id
        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        async def _boom(**kwargs):
            raise RuntimeError("p4 exploded")

        monkeypatch.setattr(p4_module, "run_replica_recovery", _boom)
        monkeypatch.setenv("KOI_HARNESS", "1")
        monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p4")
        monkeypatch.setenv("KOI_HARNESS_FAIL_OPEN", "1")

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r0",
                "group_id": "parent-job-p4",
                "status": "failed",
                "reason": "Heartbeat timeout (45s)",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "trigger_emitted"
        assert "recovery" not in body

    @pytest.mark.asyncio
    async def test_p5c_harness_fail_open_preserves_legacy_behavior(self, client, monkeypatch):
        """Harness exception should not break legacy replica-failed handling."""
        import asyncio
        from koi.harness import p5c as p5c_module
        from koi.schemas import JobTracker

        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
        )
        tracker = JobTracker(
            job_id="r0",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1200.0,
        )
        tracker.smoothed_tps = 1100.0
        dec_id = app.state.memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen3-32B",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
        )
        tracker.decision_id = dec_id
        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        async def _boom(**kwargs):
            raise RuntimeError("p5c exploded")

        monkeypatch.setattr(p5c_module, "run_chain_postmortem", _boom)
        monkeypatch.setenv("KOI_HARNESS", "1")
        monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p5c")
        monkeypatch.setenv("KOI_HARNESS_FAIL_OPEN", "1")

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "r0",
                "group_id": "parent-job",
                "status": "failed",
                "reason": "Heartbeat timeout (45s)",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "trigger_emitted"
        assert "postmortem" not in body


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


class TestJobLaunching:
    @pytest.mark.asyncio
    async def test_launching_tracked(self, client):
        """POST /job/launching stores in _pending_launches."""
        resp = await client.post(
            "/job/launching",
            json={
                "job_id": "r0",
                "decision_id": "dec-r0",
                "group_id": "parent-job",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "region": "us-east-1",
                "market": "on_demand",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "tracked"
        assert "r0" in app.state.monitor._pending_launches
        launch = app.state.monitor._pending_launches["r0"]
        assert launch["decision_id"] == "dec-r0"
        assert launch["launch_phase"] == "waiting_model_ready"
        assert (
            launch["launch_message"] == "Replica provisioned, waiting for model_ready"
        )

    @pytest.mark.asyncio
    async def test_launching_visible_in_jobs(self, client):
        """Pending launches appear in /jobs with status=launching."""
        import time

        app.state.monitor._pending_launches = {
            "r0": {
                "group_id": "parent",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "region": "us-east-1",
                "market": "on_demand",
                "launch_phase": "provisioning",
                "launch_message": "Still provisioning",
                "attempt_index": 1,
                "launched_at": time.time(),
                "last_heartbeat_at": time.time(),
            },
        }
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pending_launches"] == 1
        launching = [j for j in data["jobs"] if j["status"] == "launching"]
        assert len(launching) == 1
        assert launching[0]["gpu_type"] == "L40S"
        assert launching[0]["launch_phase"] == "provisioning"
        assert launching[0]["launch_message"] == "Still provisioning"
        assert launching[0]["attempt_index"] == 1

    @pytest.mark.asyncio
    async def test_launch_heartbeat_refreshes_lease_and_phase(self, client):
        """Heartbeats should refresh the reservation lease and merge launch progress."""
        app.state.ledger.reserve("dec-heartbeat", "L40S", 4, region="us-east-1")
        app.state.ledger.touch = MagicMock(return_value=True)
        app.state.monitor._pending_launches = {
            "r-heartbeat": {
                "group_id": "parent-job",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "region": "us-east-1",
                "market": "spot",
                "launched_at": 123.0,
            },
        }

        resp = await client.post(
            "/job/launch-heartbeat",
            json={
                "job_id": "r-heartbeat",
                "decision_id": "dec-heartbeat",
                "group_id": "parent-job",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "region": "us-east-1",
                "market": "spot",
                "attempt_index": 2,
                "phase": "provisioning",
                "message": "still searching for capacity",
                "timestamp": 456.0,
            },
        )

        assert resp.status_code == 200
        assert resp.json()["lease_refreshed"] is True
        app.state.ledger.touch.assert_called_once_with("dec-heartbeat")

        launch = app.state.monitor._pending_launches["r-heartbeat"]
        assert launch["decision_id"] == "dec-heartbeat"
        assert launch["launch_phase"] == "provisioning"
        assert launch["launch_message"] == "still searching for capacity"
        assert launch["attempt_index"] == 2
        assert launch["last_heartbeat_at"] == 456.0
        assert launch["launched_at"] == 123.0


class TestConfigAttempted:
    @pytest.mark.asyncio
    async def test_config_attempted_separates_spot_and_on_demand_priors(self, client):
        """Spot failures and on-demand successes should update different priors."""
        spot_resp = await client.post(
            "/job/config-attempted",
            json={
                "job_id": "job-market",
                "decision_id": "dec-market",
                "instance_type": "g6e.12xlarge",
                "gpu_type": "L40S",
                "region": "us-east-1",
                "market": "spot",
                "launched": False,
                "failure_reason": "InsufficientCapacity",
                "attempt_index": 0,
            },
        )
        assert spot_resp.status_code == 200
        assert spot_resp.json()["launched"] is False

        on_demand_resp = await client.post(
            "/job/config-attempted",
            json={
                "job_id": "job-market",
                "decision_id": "dec-market",
                "instance_type": "g6e.12xlarge",
                "gpu_type": "L40S",
                "region": "us-east-1",
                "market": "on_demand",
                "launched": True,
                "time_to_launch": 45.0,
                "attempt_index": 1,
            },
        )
        assert on_demand_resp.status_code == 200
        assert on_demand_resp.json()["launched"] is True

        spot = app.state.memory.get_failure_summary(
            "L40S",
            region="us-east-1",
            market="spot",
        )
        on_demand = app.state.memory.get_failure_summary(
            "L40S",
            region="us-east-1",
            market="on_demand",
        )
        assert spot["effective_observations"] == 1
        assert spot["availability_pct"] < 50.0
        assert on_demand["effective_observations"] == 1
        assert on_demand["availability_pct"] > 50.0


class TestJobStarted:
    @pytest.mark.asyncio
    async def test_started_clears_pending_and_registers_job(self, client):
        """POST /job/started should clear pending state and register the job."""
        app.state.ledger.reserve("dec-started", "L40S", 4, region="unknown")
        app.state.monitor._pending_launches = {
            "r0": {
                "group_id": "parent-job",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "region": "us-east-1",
                "market": "spot",
                "launched_at": 0,
            },
        }
        existing = MagicMock(
            group_id="parent-job",
            action_in_progress=True,
            action_freeze_until=123.0,
        )
        app.state.monitor.tracked_jobs = {"existing-r1": existing}

        resp = await client.post(
            "/job/started",
            json={
                "job_id": "r0",
                "decision_id": "dec-started",
                "group_id": "parent-job",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "dp": 1,
                "slo_deadline_hours": 8.0,
                "total_tokens": 6_000_000,
                "predicted_tps": 1200.0,
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "registered"
        assert body["decision_id"] == "dec-started"
        assert "r0" not in app.state.monitor._pending_launches
        assert app.state.ledger.pending_count == 0

        kwargs = app.state.monitor.register_job.call_args.kwargs
        assert kwargs["job_id"] == "r0"
        assert kwargs["decision_id"] == "dec-started"
        assert kwargs["group_id"] == "parent-job"
        assert kwargs["predicted_tps"] == 1200.0
        assert kwargs["config"].gpu_type == "L40S"
        assert kwargs["config"].instance_type == "g6e.12xlarge"
        assert kwargs["config"].region == "us-east-1"
        assert kwargs["config"].market == "spot"

        assert existing.action_in_progress is False
        assert existing.action_freeze_until is None

        summary = app.state.memory.get_failure_summary(
            "L40S",
            region="us-east-1",
            market="spot",
        )
        assert summary["effective_observations"] == 1
        assert summary["availability_pct"] > 50.0

    @pytest.mark.asyncio
    async def test_started_fallback_creates_child_decision(self, client):
        """Fallback launches should create a child decision for the actual config."""
        original_decision_id = app.state.memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            market="spot",
        )
        app.state.ledger.reserve(original_decision_id, "L40S", 4, region="unknown")

        resp = await client.post(
            "/job/started",
            json={
                "job_id": "r1",
                "decision_id": original_decision_id,
                "group_id": "parent-job",
                "gpu_type": "A100-80GB",
                "instance_type": "p4de.24xlarge",
                "region": "us-west-2",
                "market": "on_demand",
                "tp": 8,
                "pp": 1,
                "dp": 1,
                "slo_deadline_hours": 8.0,
                "total_tokens": 8_000_000,
                "predicted_tps": 0.0,
                "is_fallback": True,
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "registered"
        assert body["decision_id"] != original_decision_id
        assert app.state.ledger.pending_count == 0

        kwargs = app.state.monitor.register_job.call_args.kwargs
        assert kwargs["job_id"] == "r1"
        assert kwargs["decision_id"] == body["decision_id"]
        assert kwargs["config"].gpu_type == "A100-80GB"
        assert kwargs["config"].instance_type == "p4de.24xlarge"
        assert kwargs["config"].tp == 8
        assert kwargs["config"].region == "us-west-2"
        assert kwargs["config"].market == "on_demand"

        decisions = app.state.memory.query_decisions(
            model_name="Qwen/Qwen2.5-72B-Instruct"
        )
        child = next(d for d in decisions if d["decision_id"] == body["decision_id"])
        assert child["parent_decision_id"] == original_decision_id
        assert child["triggered_by"] == "fallback"
        assert child["gpu_type"] == "A100-80GB"
        assert child["instance_type"] == "p4de.24xlarge"
        assert child["market"] == "on_demand"

        on_demand = app.state.memory.get_failure_summary(
            "A100-80GB",
            region="us-west-2",
            market="on_demand",
        )
        spot = app.state.memory.get_failure_summary(
            "A100-80GB",
            region="us-west-2",
            market="spot",
        )
        assert on_demand["effective_observations"] == 1
        assert on_demand["availability_pct"] > 50.0
        assert spot["effective_observations"] == 0
        assert spot["availability_pct"] == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_started_consumes_pending_replica_decision_by_replica_id(
        self, client
    ):
        """/job/started looks up decisions by exact replica_id, not FIFO order.

        Two scale ops overlap: decision A produced r-scale-0 and r-scale-1,
        decision B produced r-other. Even if r-other arrives first (out of
        launch order), each replica gets its correct decision_id.
        """
        app.state.monitor._pending_replica_decisions = {
            "r-scale-0": {
                "decision_id": "dec-scale",
                "scale_request_id": "sr-1",
                "decision": {"gpu_type": "L40S"},
            },
            "r-scale-1": {
                "decision_id": "dec-scale",
                "scale_request_id": "sr-1",
                "decision": {"gpu_type": "L40S"},
            },
            "r-other": {
                "decision_id": "dec-other",
                "scale_request_id": "sr-2",
                "decision": {"gpu_type": "L40S"},
            },
        }

        # Out-of-order arrival: r-other, then r-scale-0, then r-scale-1.
        expected = [
            ("r-other", "dec-other"),
            ("r-scale-0", "dec-scale"),
            ("r-scale-1", "dec-scale"),
        ]
        for replica_id, expected_decision in expected:
            resp = await client.post(
                "/job/started",
                json={
                    "job_id": replica_id,
                    "decision_id": "dec-parent",
                    "group_id": "parent-job",
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "region": "us-east-1",
                    "market": "on_demand",
                    "tp": 4,
                    "pp": 1,
                    "dp": 1,
                    "slo_deadline_hours": 8.0,
                    "total_tokens": 6_000_000,
                    "predicted_tps": 1200.0,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["decision_id"] == expected_decision

        # All three pending decisions consumed exactly.
        assert app.state.monitor._pending_replica_decisions == {}


class TestStartupRestore:
    @pytest.mark.asyncio
    async def test_lifespan_restores_runtime_state(self, tmp_path):
        from koi.runtime_state import RuntimeStateStore
        from koi.server import lifespan

        runtime_path = tmp_path / "runtime.sqlite"
        memory_path = tmp_path / "memory.sqlite"
        store = RuntimeStateStore(str(runtime_path))

        tracker = _mock_decision(job_id="restored-job")
        tracker_state = {
            "job_id": "restored-job",
            "decision_id": "dec-restored",
            "group_id": "grp-restored",
            "config": tracker.config.model_dump(mode="json"),
            "slo_deadline_hours": 8.0,
            "total_tokens": 6000000,
            "predicted_tps": 833.0,
            "started_at": "2026-04-13T00:00:00",
            "tokens_completed": 1000,
            "tokens_remaining": 5999000,
            "elapsed_hours": 0.5,
            "smoothed_tps": 800.0,
            "projected_eta_hours": 2.0,
            "slo_headroom_pct": 50.0,
            "status": "on_track",
            "warmup_complete": True,
            "gpu_cache_usage": 0.0,
            "gpu_sm_util": 0.0,
            "gpu_mem_bw_util": 0.0,
            "last_positive_tps_at": None,
            "action_in_progress": True,
            "action_freeze_until": 999999.0,
            "consecutive_fetch_failures": 3,
            "last_metrics_update": None,
            "replica_ids": [],
            "dead_replicas": [],
        }
        store.upsert_tracked_job("restored-job", tracker_state)
        store.upsert_pending_launch(
            "replica-restore",
            {
                "group_id": "grp-restored",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "region": "us-west-2",
                "market": "spot",
                "launched_at": 123.4,
            },
        )
        store.upsert_pending_replica_decision(
            replica_id="grp-restored-v2-r0",
            decision_id="dec-scale",
            decision={"gpu_type": "L40S"},
        )
        store.upsert_ledger_reservation(
            "dec-ledger",
            {
                "gpu_type": "L40S",
                "num_gpus": 4,
                "cloud": "aws",
                "region": "us-west-2",
                "tenant_id": "default",
                "instance_type": "g6e.12xlarge",
                "decision_id": "dec-ledger",
                "created_at": 123.4,
            },
            expires_at=9999999999.0,
        )

        env = {
            "KOI_TEST_FAKE_DECIDE": "1",
            "KOI_RUNTIME_STATE_PATH": str(runtime_path),
            "KOI_MEMORY_PATH": str(memory_path),
        }
        with patch.dict(os.environ, env, clear=False):
            async with lifespan(app):
                assert "restored-job" in app.state.monitor.tracked_jobs
                restored = app.state.monitor.tracked_jobs["restored-job"]
                assert restored.action_in_progress is False
                assert restored.action_freeze_until is None
                assert (
                    app.state.monitor._pending_launches["replica-restore"]["market"]
                    == "spot"
                )
                assert (
                    app.state.monitor._pending_replica_decisions[
                        "grp-restored-v2-r0"
                    ]["decision_id"]
                    == "dec-scale"
                )
                assert app.state.ledger.pending_count == 1


class TestRestartPersistenceFlows:
    @pytest.mark.asyncio
    async def test_decide_launching_started_state_survives_restarts(self, tmp_path):
        runtime_path = tmp_path / "runtime.sqlite"
        memory_path = tmp_path / "memory.sqlite"
        env = {
            "KOI_TEST_FAKE_DECIDE": "1",
            "KOI_RUNTIME_STATE_PATH": str(runtime_path),
            "KOI_MEMORY_PATH": str(memory_path),
        }

        with patch.dict(os.environ, env, clear=False):
            async with _lifespan_client() as c1:
                decide_resp = await c1.post(
                    "/decide",
                    json={
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
                                {
                                    "instance_type": "g6e.12xlarge",
                                    "gpu_type": "L40S",
                                    "gpus_per_instance": 4,
                                    "vcpus": 48,
                                    "quota_family": "G",
                                    "gpu_memory_gb": 48.0,
                                    "cost_per_instance_hour_usd": 10.49,
                                    "region": "us-west-2",
                                },
                            ],
                            "quotas": [
                                {
                                    "family": "G",
                                    "region": "us-west-2",
                                    "market": "on_demand",
                                    "baseline_vcpus": 192,
                                    "used_vcpus": 0,
                                },
                            ],
                        },
                    },
                )
                assert decide_resp.status_code == 200
                decision_id = decide_resp.json()["_decision_id"]
                assert app.state.ledger.pending_count == 1

                launching_resp = await c1.post(
                    "/job/launching",
                    json={
                        "job_id": "replica-r0",
                        "group_id": "group-1",
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "tp": 4,
                        "pp": 1,
                        "region": "us-west-2",
                        "market": "on_demand",
                    },
                )
                assert launching_resp.status_code == 200
                assert "replica-r0" in app.state.monitor._pending_launches

            async with _lifespan_client() as c2:
                assert app.state.ledger.pending_count == 1
                assert "replica-r0" in app.state.monitor._pending_launches
                assert app.state.monitor.tracked_jobs == {}

                started_resp = await c2.post(
                    "/job/started",
                    json={
                        "job_id": "replica-r0",
                        "decision_id": decision_id,
                        "group_id": "group-1",
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "region": "us-west-2",
                        "market": "on_demand",
                        "tp": 4,
                        "pp": 1,
                        "dp": 1,
                        "slo_deadline_hours": 8.0,
                        "total_tokens": 6_000_000,
                        "predicted_tps": 1200.0,
                    },
                )
                assert started_resp.status_code == 200
                assert app.state.ledger.pending_count == 0
                assert "replica-r0" not in app.state.monitor._pending_launches
                assert "replica-r0" in app.state.monitor.tracked_jobs

            async with _lifespan_client() as c3:
                assert app.state.ledger.pending_count == 0
                assert app.state.monitor._pending_launches == {}
                assert "replica-r0" in app.state.monitor.tracked_jobs
                restored = app.state.monitor.tracked_jobs["replica-r0"]
                assert restored.decision_id == decision_id
                assert restored.config.region == "us-west-2"

    @pytest.mark.asyncio
    async def test_pending_replica_decision_survives_restart_and_is_consumed(
        self, tmp_path
    ):
        runtime_path = tmp_path / "runtime.sqlite"
        memory_path = tmp_path / "memory.sqlite"
        env = {
            "KOI_TEST_FAKE_DECIDE": "1",
            "KOI_RUNTIME_STATE_PATH": str(runtime_path),
            "KOI_MEMORY_PATH": str(memory_path),
        }

        with patch.dict(os.environ, env, clear=False):
            async with _lifespan_client():
                app.state.monitor.register_pending_replica_decision(
                    replica_id="replica-r1",
                    decision_id="dec-scale",
                    scale_request_id="sr-1",
                    decision={"gpu_type": "L40S"},
                )
                assert (
                    app.state.monitor._pending_replica_decisions["replica-r1"][
                        "decision_id"
                    ]
                    == "dec-scale"
                )

            async with _lifespan_client() as c2:
                assert (
                    app.state.monitor._pending_replica_decisions["replica-r1"][
                        "decision_id"
                    ]
                    == "dec-scale"
                )
                started_resp = await c2.post(
                    "/job/started",
                    json={
                        "job_id": "replica-r1",
                        "decision_id": "dec-parent",
                        "group_id": "group-1",
                        "gpu_type": "L40S",
                        "instance_type": "g6e.12xlarge",
                        "region": "us-west-2",
                        "market": "on_demand",
                        "tp": 4,
                        "pp": 1,
                        "dp": 1,
                        "slo_deadline_hours": 8.0,
                        "total_tokens": 6_000_000,
                        "predicted_tps": 1200.0,
                    },
                )
                assert started_resp.status_code == 200
                assert started_resp.json()["decision_id"] == "dec-scale"
                assert (
                    "replica-r1"
                    not in app.state.monitor._pending_replica_decisions
                )

            async with _lifespan_client():
                assert app.state.monitor._pending_replica_decisions == {}


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_decide_started_complete_flow(self, client):
        """Decision, start, and complete should preserve one clean learning record."""

        def register_job_side_effect(**kwargs):
            app.state.monitor.tracked_jobs[kwargs["job_id"]] = MagicMock(
                decision_id=kwargs["decision_id"],
                group_id=None,
                elapsed_hours=2.5,
                slo_headroom_pct=70.0,
            )

        app.state.monitor.register_job.side_effect = register_job_side_effect

        decide_resp = await client.post(
            "/decide",
            json={
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
                            "region": "us-east-1",
                            "market": "on_demand",
                            "baseline_vcpus": 192,
                            "used_vcpus": 0,
                        },
                    ],
                },
            },
        )
        assert decide_resp.status_code == 200
        decide_body = decide_resp.json()
        decision_id = decide_body["_decision_id"]
        assert app.state.ledger.pending_count == 1
        assert app.state.memory.decision_count() == 1

        started_resp = await client.post(
            "/job/started",
            json={
                "job_id": decide_body["job_id"],
                "decision_id": decision_id,
                "gpu_type": decide_body["config"]["gpu_type"],
                "instance_type": decide_body["config"]["instance_type"],
                "tp": decide_body["config"]["tp"],
                "pp": decide_body["config"]["pp"],
                "dp": decide_body["config"]["dp"],
                "slo_deadline_hours": 8.0,
                "total_tokens": 6_000_000,
                "predicted_tps": decide_body["predicted_tps"],
            },
        )
        assert started_resp.status_code == 200
        assert started_resp.json()["decision_id"] == decision_id
        assert app.state.ledger.pending_count == 0
        assert decide_body["job_id"] in app.state.monitor.tracked_jobs

        complete_resp = await client.post(
            "/job/complete",
            json={
                "job_id": decide_body["job_id"],
                "status": "succeeded",
                "metrics": {
                    "avg_generation_throughput_toks_per_s": 1500.0,
                    "cost_per_hour": 10.49,
                },
            },
        )
        assert complete_resp.status_code == 200
        assert complete_resp.json()["status"] == "recorded"
        assert app.state.memory.outcome_count() == 1

        outcomes = app.state.memory.query_outcomes(status="succeeded")
        assert len(outcomes) == 1
        assert outcomes[0]["decision_id"] == decision_id
        assert outcomes[0]["actual_tps"] == 1500.0
        assert outcomes[0]["actual_cost_per_hour"] == 10.49


class TestGroupedCompletion:
    @pytest.mark.asyncio
    async def test_group_complete_records_per_chain_outcomes(self, client):
        """Grouped completion should record one clean outcome per live chain."""
        from koi.schemas import JobTracker

        dec1 = app.state.memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1100.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
        )
        dec2 = app.state.memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1150.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
        )

        tracker1 = JobTracker(
            job_id="r0",
            config=PlacementConfig(
                gpu_type="L40S",
                instance_type="g6e.12xlarge",
                num_gpus=4,
                num_instances=1,
                tp=4,
                pp=1,
                dp=1,
                region="us-east-1",
                engine_config=EngineConfig(
                    tensor_parallel_size=4,
                    pipeline_parallel_size=1,
                ),
            ),
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1100.0,
        )
        tracker1.decision_id = dec1
        tracker1.group_id = "parent-job"
        tracker1.smoothed_tps = 1200.0
        tracker1.elapsed_hours = 2.0
        tracker1.slo_headroom_pct = 35.0

        tracker2 = JobTracker(
            job_id="r1",
            config=PlacementConfig(
                gpu_type="L40S",
                instance_type="g6e.12xlarge",
                num_gpus=4,
                num_instances=1,
                tp=4,
                pp=1,
                dp=1,
                region="us-east-1",
                engine_config=EngineConfig(
                    tensor_parallel_size=4,
                    pipeline_parallel_size=1,
                ),
            ),
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1150.0,
        )
        tracker2.decision_id = dec2
        tracker2.group_id = "parent-job"
        tracker2.smoothed_tps = 1400.0
        tracker2.elapsed_hours = 2.5
        tracker2.slo_headroom_pct = 42.0

        app.state.monitor.get_group_chains = MagicMock(
            return_value={
                "r0": tracker1,
                "r1": tracker2,
            }
        )
        app.state.monitor.unregister_group = MagicMock(return_value=["r0", "r1"])

        resp = await client.post(
            "/job/complete",
            json={
                "job_id": "parent-job",
                "status": "succeeded",
                "metrics": {"throughput_tokens_per_sec": 2600.0},
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["chains_closed"] == 2
        assert body["outcomes_recorded"] == 2
        assert body["aggregate_tps"] == 2600.0
        app.state.monitor.unregister_group.assert_called_once_with("parent-job")

        outcomes = app.state.memory.query_outcomes(status="succeeded")
        group_outcomes = [o for o in outcomes if o["job_id"] == "parent-job"]
        assert len(group_outcomes) == 2
        assert {o["decision_id"] for o in group_outcomes} == {dec1, dec2}
        assert {o["actual_tps"] for o in group_outcomes} == {1200.0, 1400.0}

    @pytest.mark.asyncio
    async def test_group_failed_runs_p5j_and_records_terminal_outcome(
        self, client, monkeypatch
    ):
        """P5j should attach a job-level diagnosis on terminal group failure."""
        from koi.harness.p5j import P5jDiagnosis
        from koi.schemas import JobTracker
        import koi.harness.p5j as p5j_module

        dec1 = app.state.memory.record_decision(
            job_id="parent-failed",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1100.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
        )
        dec2 = app.state.memory.record_decision(
            job_id="parent-failed",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1150.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
        )
        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
            ),
            market="spot",
        )
        tracker1 = JobTracker(
            job_id="r0",
            decision_id=dec1,
            group_id="parent-failed",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1100.0,
        )
        tracker1.status = MonitoringStatus.FAILED
        tracker1.smoothed_tps = 0.0
        tracker1.elapsed_hours = 1.0
        tracker2 = JobTracker(
            job_id="r1",
            decision_id=dec2,
            group_id="parent-failed",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1150.0,
        )
        tracker2.status = MonitoringStatus.FAILED
        tracker2.smoothed_tps = 0.0
        tracker2.elapsed_hours = 1.5
        app.state.monitor.get_group_chains = MagicMock(
            return_value={"r0": tracker1, "r1": tracker2}
        )
        app.state.monitor.unregister_group = MagicMock(return_value=["r0", "r1"])

        async def _stub_postmortem(**kwargs):
            assert set(kwargs["group_chains"]) == {"r0", "r1"}
            return P5jDiagnosis(
                diagnosis_code="job_capacity_exhausted",
                bottleneck="market_capacity",
                next_fix="retry_same_topology_on_demand",
                failure_scope="parent-failed",
                terminal_status="failed",
                failed_chains=2,
                diagnosed_chains=2,
                chain_diagnoses=[{"diagnosis_code": "spot_preemption"}],
                rationale="all chains lost spot capacity",
            )

        monkeypatch.setenv("KOI_HARNESS", "1")
        monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p5j")
        monkeypatch.setattr(p5j_module, "run_job_postmortem", _stub_postmortem)

        resp = await client.post(
            "/job/complete",
            json={
                "job_id": "parent-failed",
                "status": "failed",
                "metrics": {"throughput_tokens_per_sec": 0.0},
                "reason_detail": "all chains failed",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["postmortem"]["diagnosis_code"] == "job_capacity_exhausted"
        assert body["terminal_outcome_id"]

        terminal = app.state.memory.query_outcomes(status="terminal_failed")
        assert len(terminal) == 1
        assert terminal[0]["job_id"] == "parent-failed"
        assert terminal[0]["bottleneck"] == "market_capacity"
        assert "job_capacity_exhausted" in terminal[0]["diagnosis"]

    @pytest.mark.asyncio
    async def test_group_failed_p5j_fail_open_preserves_response(
        self, client, monkeypatch
    ):
        from koi.schemas import JobTracker
        import koi.harness.p5j as p5j_module

        dec_id = app.state.memory.record_decision(
            job_id="parent-p5j-open",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1100.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
        )
        tracker = JobTracker(
            job_id="r0",
            decision_id=dec_id,
            group_id="parent-p5j-open",
            config=PlacementConfig(
                gpu_type="L40S",
                instance_type="g6e.12xlarge",
                num_gpus=4,
                num_instances=1,
                tp=4,
                pp=1,
                dp=1,
                region="us-east-1",
                engine_config=EngineConfig(
                    tensor_parallel_size=4,
                    pipeline_parallel_size=1,
                ),
            ),
            slo_deadline_hours=8.0,
            total_tokens=6_000_000,
            predicted_tps=1100.0,
        )
        app.state.monitor.get_group_chains = MagicMock(return_value={"r0": tracker})
        app.state.monitor.unregister_group = MagicMock(return_value=["r0"])

        async def _boom(**kwargs):
            raise RuntimeError("p5j exploded")

        monkeypatch.setenv("KOI_HARNESS", "1")
        monkeypatch.setenv("KOI_HARNESS_PROMPTS", "p5j")
        monkeypatch.setenv("KOI_HARNESS_FAIL_OPEN", "1")
        monkeypatch.setattr(p5j_module, "run_job_postmortem", _boom)

        resp = await client.post(
            "/job/complete",
            json={
                "job_id": "parent-p5j-open",
                "status": "failed",
                "metrics": {"throughput_tokens_per_sec": 0.0},
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert "postmortem" not in body
        assert app.state.memory.query_outcomes(status="terminal_failed") == []


class TestListJobs:
    @pytest.mark.asyncio
    async def test_empty(self, client):
        app.state.monitor.tracked_jobs = {}
        app.state.monitor._pending_launches = {}
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json()["tracked_jobs"] == 0

    @pytest.mark.asyncio
    async def test_tracked_job_includes_runtime_cost_fields(self, client):
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
            market="on_demand",
        )
        from koi.schemas import JobTracker

        app.state.monitor.tracked_jobs["job-cost"] = JobTracker(
            job_id="job-cost",
            config=config,
            slo_deadline_hours=8.0,
            total_tokens=1_000_000,
            predicted_tps=1200.0,
            predicted_cost_per_hour=10.0,
            cost_roofline_usd=9.0,
            smoothed_tps=1000.0,
            projected_remaining_cost_usd=0.25,
            projected_total_cost_usd=10.25,
            cost_overage_usd=1.25,
            meets_cost_roofline=False,
            tokens_remaining=100_000,
        )

        resp = await client.get("/jobs")

        assert resp.status_code == 200
        body = resp.json()
        job = next(j for j in body["jobs"] if j["job_id"] == "job-cost")
        assert job["predicted_cost_per_hour"] == 10.0
        assert job["projected_remaining_cost_usd"] == 0.25
        assert job["projected_total_cost_usd"] == 10.25
        assert job["cost_roofline_usd"] == 9.0
        assert job["cost_overage_usd"] == 1.25
        assert job["meets_cost_roofline"] is False

    @pytest.mark.asyncio
    async def test_inf_cost_projections_serialize_as_null(self, client):
        """Regression: when smoothed_tps==0 (just after /job/started), monitor
        sets projected_eta_hours=inf, which propagates to projected_remaining_cost,
        projected_total_cost, and cost_overage. Previously /jobs returned a raw
        float('inf'), which fails JSON serialization with 'Out of range float
        values are not JSON compliant'. The endpoint must coerce non-finite
        floats to null instead of 500-ing the whole response."""
        config = PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.xlarge",
            num_gpus=1,
            num_instances=1,
            tp=1,
            pp=1,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
            market="on_demand",
        )
        from koi.schemas import JobTracker

        app.state.monitor.tracked_jobs["job-inf"] = JobTracker(
            job_id="job-inf",
            config=config,
            slo_deadline_hours=2.0,
            total_tokens=1_000_000,
            predicted_tps=1000.0,
            predicted_cost_per_hour=2.62,
            cost_roofline_usd=10.0,
            smoothed_tps=0.0,
            projected_remaining_cost_usd=float("inf"),
            projected_total_cost_usd=float("inf"),
            cost_overage_usd=float("inf"),
            meets_cost_roofline=False,
            tokens_remaining=1_000_000,
        )

        resp = await client.get("/jobs")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        job = next(j for j in body["jobs"] if j["job_id"] == "job-inf")
        assert job["projected_remaining_cost_usd"] is None
        assert job["projected_total_cost_usd"] is None
        assert job["cost_overage_usd"] is None

# ===========================================================================
# Cold-start failure recovery
# ===========================================================================
#
# When a replica dies during vLLM startup (CUDA OOM, missing model arch,
# weight-load failure), Orca fires /job/replica-failed BEFORE the replica
# ever transitioned via /job/started — so it's in monitor._pending_launches,
# not monitor.tracked_jobs. Pre-recovery, _replica_failed_impl returned
# {"status": "unknown"} for these and Koi silently dropped the event.
# These tests exercise the new path that detects pending-launch deaths,
# records the attempt, and asks the agent to pick a different config.


class TestReplicaFailedStartupRecovery:
    """`/job/replica-failed` for a replica still in _pending_launches."""

    @pytest.mark.asyncio
    async def test_pending_launch_oom_triggers_agent_recovery(self, client):
        """OOM during cold-start with budget remaining → agent.recover called,
        scale_chain_tool path drives a new launch, status=retrying."""
        from koi.server import app as _app

        # Pre-register a pending launch that's about to die
        _app.state.monitor.track_pending_launch(
            "mo-test-aaaa-r0",
            {
                "decision_id": "dec-aaaa1234",
                "group_id": "mo-test-aaaa",
                "instance_type": "g6.xlarge",
                "gpu_type": "L4",
                "region": "us-east-1",
                "market": "on_demand",
            },
        )
        # Agent will be called — make it return a non-NO_VIABLE answer
        _app.state.agent.recover_from_startup_failure = AsyncMock(
            return_value="Called scale_chain_tool with L40S TP=1 PP=1 count=1"
        )

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "mo-test-aaaa-r0",
                "group_id": "mo-test-aaaa",
                "decision_id": "dec-aaaa1234",
                "instance_type": "g6.xlarge",
                "region": "us-east-1",
                "market": "on_demand",
                "status": "failed",
                "reason": "vLLM exited with code 1 during startup — CUDA out of memory",
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "retrying"
        assert body["parent_job_id"] == "mo-test-aaaa"
        # Agent was actually invoked with the right context
        _app.state.agent.recover_from_startup_failure.assert_called_once()
        kwargs = _app.state.agent.recover_from_startup_failure.call_args.kwargs
        assert kwargs["parent_job_id"] == "mo-test-aaaa"
        assert kwargs["failure_category"] == "oom"

    @pytest.mark.asyncio
    async def test_pending_launch_unknown_category_records_no_recovery(self, client):
        """Non-recoverable failure (unknown reason) → recorded but no retry."""
        from koi.server import app as _app

        _app.state.monitor.track_pending_launch(
            "mo-test-bbbb-r0",
            {
                "decision_id": "dec-bbbb1234",
                "group_id": "mo-test-bbbb",
                "instance_type": "g6e.xlarge",
                "gpu_type": "L40S",
                "region": "us-east-1",
                "market": "on_demand",
            },
        )
        _app.state.agent.recover_from_startup_failure = AsyncMock()

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "mo-test-bbbb-r0",
                "group_id": "mo-test-bbbb",
                "decision_id": "dec-bbbb1234",
                "instance_type": "g6e.xlarge",
                "region": "us-east-1",
                "market": "on_demand",
                "status": "failed",
                "reason": "weights file corrupted",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "recorded_no_recovery"
        assert body["failure_category"] == "unknown"
        _app.state.agent.recover_from_startup_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_launch_oom_after_budget_exhausted(self, client):
        """If MAX_STARTUP_RETRIES launch attempts already recorded for this
        parent group, return status=exhausted instead of calling the agent."""
        from koi.server import app as _app
        from koi.server import MAX_STARTUP_RETRIES

        # Pre-seed launch_attempts so count_launch_attempts hits the cap
        memory: AgenticMemory = _app.state.memory
        for i in range(MAX_STARTUP_RETRIES):
            memory.record_launch_attempt(
                decision_id="dec-cccc1234",
                job_id=f"mo-test-cccc-r{i}",
                instance_type="g6.xlarge",
                gpu_type="L4",
                region="us-east-1",
                market="on_demand",
                count=1,
                launched=False,
                failure_reason="CUDA out of memory",
                failure_category="oom",
            )

        _app.state.monitor.track_pending_launch(
            "mo-test-cccc-r9",
            {
                "decision_id": "dec-cccc1234",
                "group_id": "mo-test-cccc",
                "instance_type": "g6.xlarge",
                "gpu_type": "L4",
                "region": "us-east-1",
                "market": "on_demand",
            },
        )
        _app.state.agent.recover_from_startup_failure = AsyncMock()

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "mo-test-cccc-r9",
                "group_id": "mo-test-cccc",
                "decision_id": "dec-cccc1234",
                "instance_type": "g6.xlarge",
                "region": "us-east-1",
                "market": "on_demand",
                "status": "failed",
                "reason": "vLLM exited with code 1 — CUDA out of memory",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "exhausted"
        # Agent must NOT be called when budget is exhausted
        _app.state.agent.recover_from_startup_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_launch_oom_agent_returns_no_alternative(self, client):
        """Agent says NO_VIABLE_ALTERNATIVE → status=no_alternative, no retry."""
        from koi.server import app as _app

        _app.state.monitor.track_pending_launch(
            "mo-test-dddd-r0",
            {
                "decision_id": "dec-dddd1234",
                "group_id": "mo-test-dddd",
                "instance_type": "g6e.xlarge",
                "gpu_type": "L40S",
                "region": "us-east-1",
                "market": "on_demand",
            },
        )
        _app.state.agent.recover_from_startup_failure = AsyncMock(
            return_value="NO_VIABLE_ALTERNATIVE — every config in the menu would also OOM"
        )

        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "mo-test-dddd-r0",
                "group_id": "mo-test-dddd",
                "decision_id": "dec-dddd1234",
                "instance_type": "g6e.xlarge",
                "region": "us-east-1",
                "market": "on_demand",
                "status": "failed",
                "reason": "CUDA out of memory",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "no_alternative"

    @pytest.mark.asyncio
    async def test_no_pending_no_tracked_still_unknown(self, client):
        """Nothing tracked, nothing pending → unchanged legacy behavior."""
        from koi.server import app as _app

        _app.state.agent.recover_from_startup_failure = AsyncMock()
        resp = await client.post(
            "/job/replica-failed",
            json={
                "job_id": "mo-ghost-9999-r0",
                "group_id": "mo-ghost-9999",
                "instance_type": "unknown",
                "region": "unknown",
                "market": "unknown",
                "status": "failed",
                "reason": "out of memory",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "unknown"
        _app.state.agent.recover_from_startup_failure.assert_not_called()


class TestLaunchFailedRecovery:
    """`/job/launch-failed` (all replicas in a group died during launch)."""

    @pytest.mark.asyncio
    async def test_all_oom_triggers_agent_retry(self, client):
        """Every config tried hit OOM → agent is called for re-decision."""
        from koi.server import app as _app

        _app.state.agent.recover_from_startup_failure = AsyncMock(
            return_value="Called scale_chain_tool with L40S TP=1 count=1"
        )

        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "mo-test-eeee",
                "decision_id": "dec-eeee1234",
                "configs_tried": [
                    {
                        "instance_type": "g6.xlarge",
                        "gpu_type": "L4",
                        "region": "us-east-1",
                        "market": "on_demand",
                    },
                    {
                        "instance_type": "g6.xlarge",
                        "gpu_type": "L4",
                        "region": "us-west-2",
                        "market": "on_demand",
                    },
                ],
                "failure_reasons": [
                    "CUDA out of memory during sampler warmup",
                    "torch.cuda.OutOfMemoryError",
                ],
                "total_time_seconds": 600,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "retrying"
        _app.state.agent.recover_from_startup_failure.assert_called_once()
        kwargs = _app.state.agent.recover_from_startup_failure.call_args.kwargs
        assert kwargs["failure_category"] == "oom"

    @pytest.mark.asyncio
    async def test_unknown_failures_just_recorded(self, client):
        """Mixed/unknown failure categories → fall back to existing record-only behavior."""
        from koi.server import app as _app

        _app.state.agent.recover_from_startup_failure = AsyncMock()

        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "mo-test-ffff",
                "decision_id": "dec-ffff1234",
                "configs_tried": [
                    {
                        "instance_type": "p4d.24xlarge",
                        "gpu_type": "A100",
                        "region": "us-east-1",
                        "market": "on_demand",
                    }
                ],
                "failure_reasons": ["something nondescript happened"],
                "total_time_seconds": 120,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "recorded"
        _app.state.agent.recover_from_startup_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_oom_but_budget_exhausted(self, client):
        """All-OOM failure but we've already retried MAX_STARTUP_RETRIES times → exhausted."""
        from koi.server import app as _app
        from koi.server import MAX_STARTUP_RETRIES

        memory: AgenticMemory = _app.state.memory
        for i in range(MAX_STARTUP_RETRIES):
            memory.record_launch_attempt(
                decision_id="dec-gggg1234",
                job_id=f"mo-test-gggg-r{i}",
                instance_type="g6.xlarge",
                gpu_type="L4",
                region="us-east-1",
                market="on_demand",
                count=1,
                launched=False,
                failure_reason="CUDA out of memory",
                failure_category="oom",
            )

        _app.state.agent.recover_from_startup_failure = AsyncMock()

        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "mo-test-gggg",
                "decision_id": "dec-gggg1234",
                "configs_tried": [
                    {
                        "instance_type": "g6.xlarge",
                        "gpu_type": "L4",
                        "region": "us-east-1",
                        "market": "on_demand",
                    }
                ],
                "failure_reasons": ["CUDA out of memory"],
                "total_time_seconds": 300,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "exhausted"
        _app.state.agent.recover_from_startup_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_returns_no_alternative(self, client):
        from koi.server import app as _app

        _app.state.agent.recover_from_startup_failure = AsyncMock(
            return_value="NO_VIABLE_ALTERNATIVE"
        )
        resp = await client.post(
            "/job/launch-failed",
            json={
                "job_id": "mo-test-hhhh",
                "decision_id": "dec-hhhh1234",
                "configs_tried": [
                    {
                        "instance_type": "g6.xlarge",
                        "gpu_type": "L4",
                        "region": "us-east-1",
                        "market": "on_demand",
                    }
                ],
                "failure_reasons": ["CUDA out of memory"],
                "total_time_seconds": 300,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "no_alternative"


class TestMemoryQueriesForRecovery:
    """The new memory helpers used by the recovery path."""

    def test_count_launch_attempts_matches_parent_and_replicas(self):
        memory = AgenticMemory(db_path=":memory:")
        # Parent-level row + two replica rows
        memory.record_launch_attempt(
            decision_id="dec-x",
            job_id="parent-1",
            instance_type="g6.xlarge",
            gpu_type="L4",
            region="us-east-1",
            market="on_demand",
            count=1,
            launched=False,
            failure_reason="oom",
            failure_category="oom",
        )
        memory.record_launch_attempt(
            decision_id="dec-x",
            job_id="parent-1-r0",
            instance_type="g6.xlarge",
            gpu_type="L4",
            region="us-east-1",
            market="on_demand",
            count=1,
            launched=False,
            failure_reason="oom",
            failure_category="oom",
        )
        memory.record_launch_attempt(
            decision_id="dec-x",
            job_id="parent-1-r1",
            instance_type="g6.xlarge",
            gpu_type="L4",
            region="us-east-1",
            market="on_demand",
            count=1,
            launched=True,
            failure_category=None,
        )
        # Different parent (must not bleed across)
        memory.record_launch_attempt(
            decision_id="dec-y",
            job_id="parent-2-r0",
            instance_type="g6.xlarge",
            gpu_type="L4",
            region="us-east-1",
            market="on_demand",
            count=1,
            launched=False,
            failure_reason="oom",
            failure_category="oom",
        )

        # 2 failed under parent-1 (parent-level + r0); r1 succeeded
        assert memory.count_launch_attempts("parent-1") == 2
        assert memory.count_launch_attempts("parent-1", only_failed=False) == 3
        # parent-2 has 1 failure
        assert memory.count_launch_attempts("parent-2") == 1

    def test_get_failed_configs_groups_distinct_tuples(self):
        memory = AgenticMemory(db_path=":memory:")
        memory.record_launch_attempt(
            decision_id="dec-a",
            job_id="parent-X-r0",
            instance_type="g6.xlarge",
            gpu_type="L4",
            region="us-east-1",
            market="on_demand",
            count=1,
            launched=False,
            failure_reason="oom",
            failure_category="oom",
        )
        memory.record_launch_attempt(
            decision_id="dec-a",
            job_id="parent-X-r1",
            instance_type="g6.xlarge",
            gpu_type="L4",
            region="us-east-1",
            market="on_demand",
            count=1,
            launched=False,
            failure_reason="oom",
            failure_category="oom",
        )
        memory.record_launch_attempt(
            decision_id="dec-a",
            job_id="parent-X-r2",
            instance_type="g6e.xlarge",
            gpu_type="L40S",
            region="us-east-1",
            market="on_demand",
            count=1,
            launched=False,
            failure_reason="oom",
            failure_category="oom",
        )
        rows = memory.get_failed_configs("parent-X")
        # Two distinct (instance_type, gpu_type, region, market, category) tuples
        assert len(rows) == 2
        types = {r["instance_type"] for r in rows}
        assert types == {"g6.xlarge", "g6e.xlarge"}
        # Aggregated count for the L4 row should be 2
        l4_row = next(r for r in rows if r["instance_type"] == "g6.xlarge")
        assert l4_row["attempts"] == 2
