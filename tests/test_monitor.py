"""Tests for koi/monitor.py — two async loops, SLO computation, thresholds."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from koi.schemas import (
    JobTracker, MonitoringStatus, PlacementConfig, EngineConfig, MonitoringTrigger,
)
from koi.monitor import (
    _ema, _classify_status, compute_slo_headroom,
    MonitoringLoop, WARMUP_MINUTES,
)
from koi.runtime_state import RuntimeStateStore


def _make_config():
    return PlacementConfig(
        gpu_type="L40S", instance_type="g6e.12xlarge",
        num_gpus=8, num_instances=2, tp=4, pp=2, dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
    )


def _make_tracker(**overrides):
    defaults = dict(
        job_id="job-test", config=_make_config(),
        slo_deadline_hours=8.0, total_tokens=7_500_000,
        predicted_tps=2590.0, tokens_remaining=7_500_000,
    )
    defaults.update(overrides)
    return JobTracker(**defaults)


class TestEMA:
    def test_first_value(self):
        assert _ema(0, 100.0, 0.3) == 100.0

    def test_smoothing(self):
        val = _ema(100.0, 200.0, 0.3)
        assert val == pytest.approx(130.0)

    def test_low_alpha_slow(self):
        val = _ema(100.0, 200.0, 0.1)
        assert val == pytest.approx(110.0)


class TestSLOHeadroom:
    def test_on_track(self):
        # 7.5M tokens, 2590 TPS → ~0.8h needed. SLO=8h, elapsed=0.5h
        h = compute_slo_headroom(8.0, 0.5, 7_500_000, 2590.0)
        assert h > 80  # tons of headroom

    def test_falling_behind(self):
        # Only 100 TPS, 7.5M tokens remaining, 6h elapsed, SLO=8h
        h = compute_slo_headroom(8.0, 6.0, 7_500_000, 100.0)
        assert h < 0  # behind schedule

    def test_zero_tps(self):
        h = compute_slo_headroom(8.0, 1.0, 5_000_000, 0.0)
        assert h == -100.0  # tokens remaining + zero TPS = deeply behind
        h2 = compute_slo_headroom(8.0, 1.0, 0, 0.0)
        assert h2 == 0.0  # no tokens remaining = done

    def test_just_met(self):
        # Exactly enough TPS to finish in remaining time
        # 3.6M tokens, 1000 TPS → 1h. SLO=8h, elapsed=7h → 1h left. Headroom ~0%
        h = compute_slo_headroom(8.0, 7.0, 3_600_000, 1000.0)
        assert abs(h) < 1.0


class TestClassifyStatus:
    def test_warmup(self):
        tracker = _make_tracker(elapsed_hours=0.01, warmup_complete=False)
        assert _classify_status(tracker) == MonitoringStatus.WARMING_UP

    def test_warmup_complete(self):
        tracker = _make_tracker(elapsed_hours=0.2, warmup_complete=True, slo_headroom_pct=80.0)
        assert _classify_status(tracker) == MonitoringStatus.ON_TRACK

    def test_on_track(self):
        tracker = _make_tracker(elapsed_hours=1.0, warmup_complete=True, slo_headroom_pct=50.0)
        assert _classify_status(tracker) == MonitoringStatus.ON_TRACK

    def test_at_risk(self):
        tracker = _make_tracker(elapsed_hours=5.0, warmup_complete=True, slo_headroom_pct=15.0)
        assert _classify_status(tracker) == MonitoringStatus.AT_RISK

    def test_falling_behind(self):
        tracker = _make_tracker(elapsed_hours=6.0, warmup_complete=True, slo_headroom_pct=5.0)
        assert _classify_status(tracker) == MonitoringStatus.FALLING_BEHIND

    def test_over_provisioned(self):
        # headroom > 70% AND elapsed > 20% of SLO
        tracker = _make_tracker(elapsed_hours=2.0, warmup_complete=True, slo_headroom_pct=85.0)
        assert _classify_status(tracker) == MonitoringStatus.OVER_PROVISIONED

    def test_not_over_provisioned_too_early(self):
        # headroom > 70% BUT elapsed < 20% of SLO → just ON_TRACK
        tracker = _make_tracker(elapsed_hours=0.5, warmup_complete=True, slo_headroom_pct=85.0)
        assert _classify_status(tracker) == MonitoringStatus.ON_TRACK


class TestMonitoringLoopRegistration:
    def test_register_job(self):
        monitor = MonitoringLoop(orca=MagicMock())
        monitor.register_job(
            job_id="job-1", config=_make_config(),
            slo_deadline_hours=8.0, total_tokens=7_500_000,
            predicted_tps=2590.0, decision_id="dec-abc",
        )
        assert "job-1" in monitor.tracked_jobs
        assert monitor.tracked_jobs["job-1"].decision_id == "dec-abc"

    def test_unregister_job(self):
        monitor = MonitoringLoop(orca=MagicMock())
        monitor.register_job(
            job_id="job-1", config=_make_config(),
            slo_deadline_hours=8.0, total_tokens=7_500_000,
            predicted_tps=2590.0,
        )
        monitor.unregister_job("job-1")
        assert "job-1" not in monitor.tracked_jobs


class TestTriggerSuppression:
    @pytest.mark.asyncio
    async def test_falling_behind_suppressed_for_failed_replica(self):
        """FALLING_BEHIND trigger should NOT emit for a replica already marked FAILED."""
        monitor = MonitoringLoop(orca=MagicMock())
        monitor.register_job(
            job_id="job-1", config=_make_config(),
            slo_deadline_hours=8.0, total_tokens=7_500_000,
            predicted_tps=2590.0,
        )
        tracker = monitor.tracked_jobs["job-1"]
        tracker.status = MonitoringStatus.FAILED  # already dead

        await monitor._emit_trigger("job-1", MonitoringStatus.FALLING_BEHIND, "TPS low")

        # Queue should be empty — trigger suppressed
        assert monitor._trigger_queue.empty()

    @pytest.mark.asyncio
    async def test_failed_trigger_still_emits_for_failed_replica(self):
        """FAILED trigger should still emit even if status is already FAILED (idempotent)."""
        monitor = MonitoringLoop(orca=MagicMock())
        monitor.register_job(
            job_id="job-1", config=_make_config(),
            slo_deadline_hours=8.0, total_tokens=7_500_000,
            predicted_tps=2590.0,
        )
        tracker = monitor.tracked_jobs["job-1"]
        tracker.status = MonitoringStatus.FAILED

        await monitor._emit_trigger("job-1", MonitoringStatus.FAILED, "Heartbeat timeout")

        assert not monitor._trigger_queue.empty()


class TestGroupAggregation:
    def test_dead_replicas_excluded_from_aggregate(self):
        """Dead replicas (FAILED/COMPLETED) should not drag down aggregate TPS."""
        t1 = _make_tracker(job_id="r0", smoothed_tps=2000.0)
        t1.status = MonitoringStatus.ON_TRACK

        t2 = _make_tracker(job_id="r1", smoothed_tps=0.0)
        t2.status = MonitoringStatus.FAILED  # dead

        t3 = _make_tracker(job_id="r2", smoothed_tps=1800.0)
        t3.status = MonitoringStatus.ON_TRACK

        group = {"r0": t1, "r1": t2, "r2": t3}

        # Filter like the production code does
        live = {k: v for k, v in group.items()
                if v.status not in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED)}
        aggregate = sum(t.smoothed_tps for t in live.values())

        assert "r1" not in live  # dead replica excluded
        assert aggregate == 3800.0  # 2000 + 1800, not 2000 + 0 + 1800

    def test_all_dead_gives_zero(self):
        """If all replicas are dead, aggregate TPS is 0."""
        t1 = _make_tracker(job_id="r0", smoothed_tps=0.0)
        t1.status = MonitoringStatus.FAILED
        t2 = _make_tracker(job_id="r1", smoothed_tps=0.0)
        t2.status = MonitoringStatus.COMPLETED

        group = {"r0": t1, "r1": t2}
        live = {k: v for k, v in group.items()
                if v.status not in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED)}
        aggregate = sum(t.smoothed_tps for t in live.values()) if live else 0

        assert aggregate == 0


class TestMonitoringLoopPersistence:
    def test_register_and_unregister_job_persist_runtime_state(self, tmp_path):
        store = RuntimeStateStore(str(tmp_path / "runtime.sqlite"))
        monitor = MonitoringLoop(orca=MagicMock(), runtime_state=store)

        monitor.register_job(
            job_id="job-1",
            config=_make_config(),
            slo_deadline_hours=8.0,
            total_tokens=7_500_000,
            predicted_tps=2590.0,
            decision_id="dec-abc",
            group_id="grp-1",
        )

        persisted = store.load_tracked_jobs()
        assert persisted["job-1"]["decision_id"] == "dec-abc"
        assert persisted["job-1"]["tracker"]["group_id"] == "grp-1"

        monitor.unregister_job("job-1")
        assert store.load_tracked_jobs() == {}

    def test_pending_launch_and_scale_queue_persist(self, tmp_path):
        store = RuntimeStateStore(str(tmp_path / "runtime.sqlite"))
        monitor = MonitoringLoop(orca=MagicMock(), runtime_state=store)

        monitor.track_pending_launch("replica-1", {
            "group_id": "grp-1",
            "gpu_type": "L40S",
            "instance_type": "g6e.12xlarge",
            "region": "us-west-2",
            "market": "spot",
        })
        assert store.load_pending_launches()["replica-1"]["launch"]["market"] == "spot"

        monitor.enqueue_pending_scale_decision("grp-1", {"decision_id": "dec-a", "remaining": 2})
        monitor.enqueue_pending_scale_decision("grp-1", {"decision_id": "dec-b", "remaining": 1})
        assert store.load_pending_scale_decisions()["grp-1"] == [
            {"decision_id": "dec-a", "remaining": 2},
            {"decision_id": "dec-b", "remaining": 1},
        ]

        first = monitor.consume_pending_scale_decision("grp-1")
        assert first["decision_id"] == "dec-a"
        assert store.load_pending_scale_decisions()["grp-1"] == [
            {"decision_id": "dec-a", "remaining": 1},
            {"decision_id": "dec-b", "remaining": 1},
        ]

        monitor.clear_pending_launch("replica-1")
        assert store.load_pending_launches() == {}

    @pytest.mark.asyncio
    async def test_poll_job_persists_updated_tracker_state(self, tmp_path):
        store = RuntimeStateStore(str(tmp_path / "runtime.sqlite"))
        orca = MagicMock()
        orca.get_job_metrics = AsyncMock(return_value={
            "avg_generation_throughput_toks_per_s": 1000.0,
            "gpu_cache_usage_perc": 25.0,
            "gpu_sm_util_pct": 50.0,
            "gpu_mem_bw_util_pct": 60.0,
        })
        orca.get_chunk_progress = AsyncMock(return_value={
            "total": 10,
            "completed": 5,
            "failed": 0,
            "all_done": False,
        })
        orca.get_job_status = AsyncMock(return_value={"status": "running"})

        monitor = MonitoringLoop(orca=orca, runtime_state=store)
        monitor.register_job(
            job_id="job-1",
            config=_make_config(),
            slo_deadline_hours=8.0,
            total_tokens=10_000,
            predicted_tps=1200.0,
            decision_id="dec-1",
        )

        await monitor._poll_job("job-1")

        persisted = store.load_tracked_jobs()["job-1"]["tracker"]
        assert persisted["smoothed_tps"] == pytest.approx(1000.0)
        assert persisted["tokens_completed"] == 5000
        assert persisted["gpu_cache_usage"] == 25.0
