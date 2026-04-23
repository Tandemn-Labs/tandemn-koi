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
                "failure_reasons": ["InsufficientCapacity", "QuotaExceeded"],
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


class TestListJobs:
    @pytest.mark.asyncio
    async def test_empty(self, client):
        app.state.monitor.tracked_jobs = {}
        app.state.monitor._pending_launches = {}
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json()["tracked_jobs"] == 0
