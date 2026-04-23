"""Tests for koi/monitor.py — two async loops, SLO computation, thresholds."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from koi.schemas import (
    JobTracker,
    MonitoringStatus,
    PlacementConfig,
    EngineConfig,
    MonitoringTrigger,
)
from koi.monitor import (
    _ema,
    _classify_status,
    _required_overprovision_elapsed_hours,
    compute_slo_headroom,
    MonitoringLoop,
    WARMUP_MINUTES,
)
from koi.runtime_state import RuntimeStateStore


def _make_config():
    return PlacementConfig(
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        num_gpus=8,
        num_instances=2,
        tp=4,
        pp=2,
        dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
    )


def _make_tracker(**overrides):
    defaults = dict(
        job_id="job-test",
        config=_make_config(),
        slo_deadline_hours=8.0,
        total_tokens=7_500_000,
        predicted_tps=2590.0,
        predicted_cost_per_hour=10.49,
        tokens_remaining=7_500_000,
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

    def test_late_job_with_buffer_stays_positive(self):
        # 1.8M tokens, 1000 TPS → 0.5h remaining. SLO=8h, elapsed=7h → 1h left.
        # This should be comfortably on track, not near-falling-behind.
        h = compute_slo_headroom(8.0, 7.0, 1_800_000, 1000.0)
        assert h == pytest.approx(50.0)

    def test_overdue_job_is_clamped(self):
        h = compute_slo_headroom(8.0, 8.5, 1_000_000, 1200.0)
        assert h == -100.0


class TestClassifyStatus:
    def test_warmup(self):
        tracker = _make_tracker(elapsed_hours=0.01, warmup_complete=False)
        assert _classify_status(tracker) == MonitoringStatus.WARMING_UP

    def test_warmup_complete(self):
        tracker = _make_tracker(
            elapsed_hours=0.2, warmup_complete=True, slo_headroom_pct=60.0
        )
        assert _classify_status(tracker) == MonitoringStatus.ON_TRACK

    def test_on_track(self):
        tracker = _make_tracker(
            elapsed_hours=1.0, warmup_complete=True, slo_headroom_pct=50.0
        )
        assert _classify_status(tracker) == MonitoringStatus.ON_TRACK

    def test_at_risk(self):
        tracker = _make_tracker(
            elapsed_hours=5.0, warmup_complete=True, slo_headroom_pct=15.0
        )
        assert _classify_status(tracker) == MonitoringStatus.AT_RISK

    def test_falling_behind(self):
        tracker = _make_tracker(
            elapsed_hours=6.0, warmup_complete=True, slo_headroom_pct=5.0
        )
        assert _classify_status(tracker) == MonitoringStatus.FALLING_BEHIND

    def test_over_provisioned(self):
        # headroom > 70% AND elapsed exceeds the capped wait threshold
        tracker = _make_tracker(
            elapsed_hours=2.0, warmup_complete=True, slo_headroom_pct=85.0
        )
        assert _classify_status(tracker) == MonitoringStatus.OVER_PROVISIONED

    def test_not_over_provisioned_too_early(self):
        # headroom > 70% BUT elapsed < capped wait threshold → just ON_TRACK
        tracker = _make_tracker(
            elapsed_hours=0.03, warmup_complete=True, slo_headroom_pct=85.0
        )
        assert _classify_status(tracker) == MonitoringStatus.ON_TRACK

    def test_single_live_replica_can_mark_over_provisioned_by_default(self):
        tracker = _make_tracker(
            elapsed_hours=1.0, warmup_complete=True, slo_headroom_pct=85.0
        )
        assert (
            _classify_status(tracker, active_replicas=1)
            == MonitoringStatus.OVER_PROVISIONED
        )


class TestOverprovisionWindow:
    def test_long_slo_wait_is_capped(self):
        required = _required_overprovision_elapsed_hours(8.0)
        assert required == pytest.approx(max(WARMUP_MINUTES / 60, 0.05))

    def test_short_slo_still_respects_warmup_floor(self):
        required = _required_overprovision_elapsed_hours(0.02)
        assert required >= WARMUP_MINUTES / 60


class TestMonitoringLoopRegistration:
    def test_job_tracker_can_carry_cost_roofline(self):
        tracker = _make_tracker(cost_roofline_usd=120.0)
        assert tracker.cost_roofline_usd == 120.0

    def test_register_job(self):
        monitor = MonitoringLoop(orca=MagicMock())
        monitor.register_job(
            job_id="job-1",
            config=_make_config(),
            slo_deadline_hours=8.0,
            total_tokens=7_500_000,
            predicted_tps=2590.0,
            predicted_cost_per_hour=10.49,
            decision_id="dec-abc",
        )
        assert "job-1" in monitor.tracked_jobs
        assert monitor.tracked_jobs["job-1"].decision_id == "dec-abc"

    def test_unregister_job(self):
        monitor = MonitoringLoop(orca=MagicMock())
        monitor.register_job(
            job_id="job-1",
            config=_make_config(),
            slo_deadline_hours=8.0,
            total_tokens=7_500_000,
            predicted_tps=2590.0,
            predicted_cost_per_hour=10.49,
        )
        monitor.unregister_job("job-1")
        assert "job-1" not in monitor.tracked_jobs


class TestTriggerSuppression:
    @pytest.mark.asyncio
    async def test_falling_behind_suppressed_for_failed_replica(self):
        """FALLING_BEHIND trigger should NOT emit for a replica already marked FAILED."""
        monitor = MonitoringLoop(orca=MagicMock())
        monitor.register_job(
            job_id="job-1",
            config=_make_config(),
            slo_deadline_hours=8.0,
            total_tokens=7_500_000,
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
            job_id="job-1",
            config=_make_config(),
            slo_deadline_hours=8.0,
            total_tokens=7_500_000,
            predicted_tps=2590.0,
        )
        tracker = monitor.tracked_jobs["job-1"]
        tracker.status = MonitoringStatus.FAILED

        await monitor._emit_trigger(
            "job-1", MonitoringStatus.FAILED, "Heartbeat timeout"
        )

        assert not monitor._trigger_queue.empty()

    @pytest.mark.asyncio
    async def test_group_health_triggers_dedupe_within_cooldown(self):
        """
        Area 1: only ONE FALLING_BEHIND trigger should fire per group within
        the 60s cooldown window, no matter how many chains report unhealthy.
        """
        monitor = MonitoringLoop(orca=MagicMock())
        group_id = "group-x"
        for rid in ("group-x-r0", "group-x-r1", "group-x-r2"):
            monitor.register_job(
                job_id=rid,
                config=_make_config(),
                slo_deadline_hours=8.0,
                total_tokens=7_500_000,
                predicted_tps=2590.0,
                predicted_cost_per_hour=10.49,
                group_id=group_id,
            )

        await monitor._emit_trigger(
            "group-x-r0", MonitoringStatus.FALLING_BEHIND, "Headroom=-50%"
        )
        await monitor._emit_trigger(
            "group-x-r1", MonitoringStatus.FALLING_BEHIND, "Headroom=-50%"
        )
        await monitor._emit_trigger(
            "group-x-r2", MonitoringStatus.FALLING_BEHIND, "Headroom=-50%"
        )

        drained: list[MonitoringTrigger] = []
        while not monitor._trigger_queue.empty():
            drained.append(monitor._trigger_queue.get_nowait())

        assert len(drained) == 1
        assert drained[0].trigger_type == MonitoringStatus.FALLING_BEHIND

    @pytest.mark.asyncio
    async def test_group_health_triggers_share_cooldown_across_status(self):
        """
        Area 1: cooldown key is group:health, not group:status — rapid alternations
        between FALLING_BEHIND and OVER_PROVISIONED should all dedupe together.
        """
        monitor = MonitoringLoop(orca=MagicMock())
        group_id = "group-y"
        for rid in ("group-y-r0", "group-y-r1"):
            monitor.register_job(
                job_id=rid,
                config=_make_config(),
                slo_deadline_hours=8.0,
                total_tokens=7_500_000,
                predicted_tps=2590.0,
                predicted_cost_per_hour=10.49,
                group_id=group_id,
            )

        await monitor._emit_trigger(
            "group-y-r0", MonitoringStatus.FALLING_BEHIND, "Headroom=-50%"
        )
        # Within 60s: OVER_PROVISIONED should be suppressed too (health-family
        # dedup means rapid flapping doesn't spam the agent).
        await monitor._emit_trigger(
            "group-y-r1", MonitoringStatus.OVER_PROVISIONED, "Headroom=95%"
        )

        drained: list[MonitoringTrigger] = []
        while not monitor._trigger_queue.empty():
            drained.append(monitor._trigger_queue.get_nowait())

        assert len(drained) == 1
        assert drained[0].trigger_type == MonitoringStatus.FALLING_BEHIND

    @pytest.mark.asyncio
    async def test_group_trigger_suppressed_while_sibling_action_in_flight(self):
        """
        Area 1: if a sibling chain in the same group is mid-action (anti-windup
        freeze), no new health trigger should fire for the group.
        """
        import time as _time

        monitor = MonitoringLoop(orca=MagicMock())
        group_id = "group-z"
        for rid in ("group-z-r0", "group-z-r1"):
            monitor.register_job(
                job_id=rid,
                config=_make_config(),
                slo_deadline_hours=8.0,
                total_tokens=7_500_000,
                predicted_tps=2590.0,
                predicted_cost_per_hour=10.49,
                group_id=group_id,
            )
        # Freeze r0 (simulating scale_chain_tool / kill_replica_tool firing)
        sibling = monitor.tracked_jobs["group-z-r0"]
        sibling.action_in_progress = True
        sibling.action_freeze_until = _time.time() + 120

        # A trigger for r1 should be suppressed even though r1 itself isn't frozen.
        await monitor._emit_trigger(
            "group-z-r1", MonitoringStatus.FALLING_BEHIND, "Headroom=-20%"
        )
        assert monitor._trigger_queue.empty()

    @pytest.mark.asyncio
    async def test_per_chain_failed_trigger_not_deduped_by_group(self):
        """
        Area 1: group dedup only applies to health triggers. FAILED triggers
        still fire per chain so the agent can react to a specific replica's
        fate.
        """
        monitor = MonitoringLoop(orca=MagicMock())
        group_id = "group-a"
        for rid in ("group-a-r0", "group-a-r1"):
            monitor.register_job(
                job_id=rid,
                config=_make_config(),
                slo_deadline_hours=8.0,
                total_tokens=7_500_000,
                predicted_tps=2590.0,
                predicted_cost_per_hour=10.49,
                group_id=group_id,
            )

        await monitor._emit_trigger(
            "group-a-r0", MonitoringStatus.FAILED, "Orca reports dead"
        )
        await monitor._emit_trigger(
            "group-a-r1", MonitoringStatus.FAILED, "Orca reports dead"
        )

        drained: list[MonitoringTrigger] = []
        while not monitor._trigger_queue.empty():
            drained.append(monitor._trigger_queue.get_nowait())

        assert len(drained) == 2
        assert all(t.trigger_type == MonitoringStatus.FAILED for t in drained)


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
        live = {
            k: v
            for k, v in group.items()
            if v.status not in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED)
        }
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
        live = {
            k: v
            for k, v in group.items()
            if v.status not in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED)
        }
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
            predicted_cost_per_hour=10.49,
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

        monitor.track_pending_launch(
            "replica-1",
            {
                "group_id": "grp-1",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "region": "us-west-2",
                "market": "spot",
            },
        )
        assert store.load_pending_launches()["replica-1"]["launch"]["market"] == "spot"

        monitor.register_pending_replica_decision(
            replica_id="grp-1-v2-r0",
            decision_id="dec-a",
            scale_request_id="sr-1",
            decision={"gpu_type": "L40S"},
        )
        monitor.register_pending_replica_decision(
            replica_id="grp-1-v2-r1",
            decision_id="dec-a",
            scale_request_id="sr-1",
            decision={"gpu_type": "L40S"},
        )
        loaded = store.load_pending_replica_decisions()
        assert set(loaded.keys()) == {"grp-1-v2-r0", "grp-1-v2-r1"}
        assert loaded["grp-1-v2-r0"]["decision_id"] == "dec-a"

        first = monitor.consume_pending_replica_decision("grp-1-v2-r0")
        assert first["decision_id"] == "dec-a"
        # Sibling replica's mapping untouched by the consume.
        loaded = store.load_pending_replica_decisions()
        assert "grp-1-v2-r0" not in loaded
        assert loaded["grp-1-v2-r1"]["decision_id"] == "dec-a"

        monitor.clear_pending_launch("replica-1")
        assert store.load_pending_launches() == {}

    @pytest.mark.asyncio
    async def test_poll_job_persists_updated_tracker_state(self, tmp_path):
        store = RuntimeStateStore(str(tmp_path / "runtime.sqlite"))
        orca = MagicMock()
        orca.get_job_metrics = AsyncMock(
            return_value={
                "avg_generation_throughput_toks_per_s": 1000.0,
                "gpu_cache_usage_perc": 25.0,
                "gpu_sm_util_pct": 50.0,
                "gpu_mem_bw_util_pct": 60.0,
            }
        )
        orca.get_chunk_progress = AsyncMock(
            return_value={
                "total": 10,
                "completed": 5,
                "failed": 0,
                "all_done": False,
            }
        )
        orca.get_job_status = AsyncMock(return_value={"status": "running"})

        monitor = MonitoringLoop(orca=orca, runtime_state=store)
        monitor.register_job(
            job_id="job-1",
            config=_make_config(),
            slo_deadline_hours=8.0,
            total_tokens=10_000,
            predicted_tps=1200.0,
            predicted_cost_per_hour=10.0,
            cost_roofline_usd=19.5,
            decision_id="dec-1",
        )
        monitor.tracked_jobs["job-1"].started_at = datetime.utcnow() - timedelta(hours=2)

        await monitor._poll_job("job-1")

        persisted = store.load_tracked_jobs()["job-1"]["tracker"]
        assert persisted["smoothed_tps"] == pytest.approx(1000.0)
        assert persisted["tokens_completed"] == 5000
        assert persisted["gpu_cache_usage"] == 25.0
        assert persisted["projected_total_cost_usd"] == pytest.approx(20.01, abs=0.05)
        assert persisted["projected_remaining_cost_usd"] == pytest.approx(0.01, abs=0.02)
        assert persisted["meets_cost_roofline"] is False
        assert persisted["cost_overage_usd"] == pytest.approx(0.51, abs=0.05)

    def test_restore_runtime_state_rebuilds_monitor_and_clears_freeze(self, tmp_path):
        db_path = str(tmp_path / "runtime.sqlite")
        store = RuntimeStateStore(db_path)

        tracker = _make_tracker(
            job_id="job-restore",
            decision_id="dec-restore",
            group_id="grp-restore",
            action_in_progress=True,
            action_freeze_until=999999.0,
            warmup_complete=True,
            smoothed_tps=850.0,
        )
        store.upsert_tracked_job("job-restore", tracker.model_dump(mode="json"))
        store.upsert_pending_launch(
            "replica-restore",
            {
                "group_id": "grp-restore",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "region": "us-west-2",
                "market": "spot",
            },
        )
        store.upsert_pending_replica_decision(
            replica_id="grp-restore-v2-r0",
            decision_id="dec-scale",
            decision={"gpu_type": "L40S", "tp": 4, "pp": 2},
        )

        restored = MonitoringLoop(
            orca=MagicMock(),
            runtime_state=RuntimeStateStore(db_path),
        )
        summary = restored.restore_runtime_state()

        assert summary == {
            "tracked_jobs": 1,
            "pending_launches": 1,
            "pending_replica_decisions": 1,
        }
        assert "job-restore" in restored.tracked_jobs
        assert restored.tracked_jobs["job-restore"].action_in_progress is False
        assert restored.tracked_jobs["job-restore"].action_freeze_until is None
        assert restored._pending_launches["replica-restore"]["market"] == "spot"
        assert (
            restored._pending_replica_decisions["grp-restore-v2-r0"]["decision_id"]
            == "dec-scale"
        )
