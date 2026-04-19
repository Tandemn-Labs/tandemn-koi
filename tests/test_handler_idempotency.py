"""Tests that webhook handlers are idempotent under duplicate delivery.

Phase 2c of contract-hardening. The inbox wrapper (`_run_with_inbox` in
koi/server.py) claims each event before running the handler and marks it
processed only after the handler succeeds. The properties proven here:

  1. Same event_id delivered twice → handler runs once (duplicate_ignored).
  2. Legacy payload (no event_id) → synthesized id is stable across retries.
  3. Different event_ids on same logical state → handler runs each time
     (inbox dedups by event_id, not by job_id).
  4. After handler runs, /job/complete outcome rows aren't re-written.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from koi.resource_ledger import ResourceLedger
from koi.runtime_state import RuntimeStateStore
from koi.schemas import JobTracker, PlacementConfig, EngineConfig, MonitoringStatus
from koi.server import app
from koi.tools.memory import AgenticMemory


def _tracker(job_id: str, decision_id: str = "d-1", group_id=None) -> JobTracker:
    return JobTracker(
        job_id=job_id,
        decision_id=decision_id,
        group_id=group_id,
        config=PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=8,
            num_instances=1,
            tp=4,
            pp=2,
            dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
            market="on_demand",
        ),
        slo_deadline_hours=8.0,
        total_tokens=1_000_000,
        predicted_tps=1200.0,
        tokens_remaining=1_000_000,
    )


@pytest_asyncio.fixture
async def client_with_tracker():
    memory = AgenticMemory(db_path=":memory:")
    app.state.perfdb = MagicMock()
    app.state.perfdb.record_count = 0
    app.state.memory = memory
    app.state.runtime_state = RuntimeStateStore(":memory:")
    app.state.ledger = ResourceLedger()
    app.state.decide_lock = asyncio.Lock()
    app.state.orca = None
    app.state.agent = MagicMock()
    app.state.agent.model = "test"
    app.state.agent.decide = AsyncMock()
    app.state.agent.handle_trigger = AsyncMock()

    monitor = MagicMock()
    tracker = _tracker("mo-abc", decision_id="d-1")
    monitor.tracked_jobs = {"mo-abc": tracker}
    monitor._pending_launches = {}
    monitor._pending_scale_decisions = {}
    monitor._koi_initiated_kills = set()
    monitor._trigger_queue = asyncio.Queue()
    monitor.persist_job = MagicMock()
    monitor.unregister_job = MagicMock(
        side_effect=lambda jid: monitor.tracked_jobs.pop(jid, None)
    )
    monitor.unregister_group = MagicMock(return_value=[])
    monitor.get_group_chains = MagicMock(return_value={})
    app.state.monitor = monitor

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, memory


class TestExplicitEventIdDedups:
    @pytest.mark.asyncio
    async def test_same_event_id_processed_once(self, client_with_tracker):
        c, memory = client_with_tracker
        payload = {
            "event_id": "job_complete:mo-abc",
            "job_id": "mo-abc",
            "status": "succeeded",
            "metrics": {"avg_generation_throughput_toks_per_s": 1234.5},
        }
        r1 = await c.post("/job/complete", json=payload)
        r2 = await c.post("/job/complete", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["status"] == "recorded"
        assert r2.json()["status"] == "duplicate_ignored"
        # Handler wrote exactly one outcome (the first call).
        assert memory.outcome_count() == 1


class TestLegacyEventIdSynthesis:
    @pytest.mark.asyncio
    async def test_same_legacy_payload_dedups(self, client_with_tracker):
        c, memory = client_with_tracker
        payload = {
            "job_id": "mo-abc",
            "status": "succeeded",
            "metrics": {"avg_generation_throughput_toks_per_s": 1234.5},
        }
        r1 = await c.post("/job/complete", json=payload)
        r2 = await c.post("/job/complete", json=payload)
        assert r1.json()["status"] == "recorded"
        assert r2.json()["status"] == "duplicate_ignored"
        assert memory.outcome_count() == 1

    @pytest.mark.asyncio
    async def test_different_payload_different_synthesized_id(
        self, client_with_tracker
    ):
        """Two logically distinct legacy calls get distinct synthesized ids."""
        c, memory = client_with_tracker
        # Add a second tracker so the second call has a valid mapping too.
        monitor = app.state.monitor
        monitor.tracked_jobs["mo-xyz"] = _tracker("mo-xyz", decision_id="d-2")
        p1 = {"job_id": "mo-abc", "status": "succeeded", "metrics": {}}
        p2 = {"job_id": "mo-xyz", "status": "succeeded", "metrics": {}}
        r1 = await c.post("/job/complete", json=p1)
        r2 = await c.post("/job/complete", json=p2)
        assert r1.json()["status"] == "recorded"
        assert r2.json()["status"] == "recorded"
        assert memory.outcome_count() == 2


class TestInboxPersistence:
    @pytest.mark.asyncio
    async def test_inbox_row_marked_processed(self, client_with_tracker):
        c, _ = client_with_tracker
        payload = {
            "event_id": "job_complete:mo-abc",
            "job_id": "mo-abc",
            "status": "succeeded",
            "metrics": {},
        }
        await c.post("/job/complete", json=payload)
        assert app.state.runtime_state.inbox_count(status="processed") == 1
        assert app.state.runtime_state.inbox_count(status="processing") == 0

    @pytest.mark.asyncio
    async def test_inbox_tracks_each_distinct_event(self, client_with_tracker):
        c, _ = client_with_tracker
        monitor = app.state.monitor
        # Add a replica tracker so /job/launching handler is happy.
        monitor.track_pending_launch = MagicMock()
        monitor.get_pending_launch = MagicMock(return_value={})
        monitor.clear_pending_launch = MagicMock()

        for i in range(3):
            await c.post(
                "/job/launching",
                json={
                    "event_id": f"job_launching:mo-abc-r{i}",
                    "job_id": f"mo-abc-r{i}",
                    "decision_id": "d-1",
                    "group_id": "mo-abc",
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 2,
                    "region": "us-east-1",
                    "market": "on_demand",
                    "attempt_index": 0,
                },
            )
        assert app.state.runtime_state.inbox_count(status="processed") == 3
