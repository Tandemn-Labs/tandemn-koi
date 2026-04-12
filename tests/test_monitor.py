"""Tests for koi/monitor.py — two async loops, SLO computation, thresholds."""

import pytest
from datetime import datetime, timedelta

from koi.schemas import (
    JobTracker, MonitoringStatus, PlacementConfig, EngineConfig, MonitoringTrigger,
)
from koi.monitor import (
    _ema, _classify_status, compute_slo_headroom,
    MonitoringLoop, WARMUP_MINUTES,
)


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
        from unittest.mock import MagicMock
        monitor = MonitoringLoop(orca=MagicMock())
        monitor.register_job(
            job_id="job-1", config=_make_config(),
            slo_deadline_hours=8.0, total_tokens=7_500_000,
            predicted_tps=2590.0, decision_id="dec-abc",
        )
        assert "job-1" in monitor.tracked_jobs
        assert monitor.tracked_jobs["job-1"].decision_id == "dec-abc"

    def test_unregister_job(self):
        from unittest.mock import MagicMock
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
        from unittest.mock import MagicMock
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
        from unittest.mock import MagicMock
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
