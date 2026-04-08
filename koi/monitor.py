"""
koi/monitor.py — Two async loops for job monitoring.

Loop 1 (TelemetryLoop): 10s polling, pure code, updates JobTracker, checks thresholds
Loop 2 (TriggerDispatcher): event-driven, fires agent on triggers

These are independent asyncio.Tasks. They do NOT share a timer.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Callable, Coroutine, Dict, List, Optional

from koi.schemas import (
    JobTracker, MonitoringStatus, MonitoringTrigger, PlacementConfig,
)
from koi.tools.orca_api import OrcaClient

logger = logging.getLogger("koi.monitor")

# Thresholds
WARMUP_MINUTES = float(os.environ.get("KOI_WARMUP_MINUTES", "5.0"))
EMA_ALPHA = 0.3                   # exponential moving average smoothing

# Hysteresis thresholds (enter/exit pairs prevent oscillation)
FALLING_BEHIND_ENTER = 10.0       # headroom < 10% → enter FALLING_BEHIND
FALLING_BEHIND_EXIT = 20.0        # headroom > 20% → exit FALLING_BEHIND
ON_TRACK_THRESHOLD = 30.0         # headroom > 30% → ON_TRACK
OVER_PROVISIONED_ENTER = 70.0     # headroom > 70% → enter OVER_PROVISIONED
OVER_PROVISIONED_EXIT = 50.0      # headroom < 50% → exit OVER_PROVISIONED
OVER_PROVISIONED_MIN_ELAPSED = 0.20  # 20% of SLO elapsed before considering scale-down


class MonitoringLoop:
    """
    Manages all three async loops for tracked jobs.

    Usage:
        monitor = MonitoringLoop(orca=orca_client, memory=memory, on_trigger=agent.handle_trigger)
        monitor.register_job(job_id, config, slo, total_tokens, predicted_tps, decision_id)
        await monitor.start()  # starts all 3 loops as background tasks
        await monitor.stop()   # cancels all tasks
    """

    def __init__(
        self,
        orca: OrcaClient,
        on_trigger: Optional[Callable[[MonitoringTrigger], Coroutine]] = None,
        telemetry_interval: float = 10.0,
    ):
        self.orca = orca
        self.on_trigger = on_trigger
        self.telemetry_interval = telemetry_interval

        self.tracked_jobs: Dict[str, JobTracker] = {}
        self._trigger_queue: asyncio.Queue = asyncio.Queue()
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._group_trigger_cooldown: Dict[str, float] = {}  # "group_id:status" → last_emit_time

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def register_job(
        self,
        job_id: str,
        config: PlacementConfig,
        slo_deadline_hours: float,
        total_tokens: int,
        predicted_tps: float,
        decision_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        tracker = JobTracker(
            job_id=job_id,
            decision_id=decision_id,
            group_id=group_id,
            config=config,
            slo_deadline_hours=slo_deadline_hours,
            total_tokens=total_tokens,
            predicted_tps=predicted_tps,
            tokens_remaining=total_tokens,
        )
        self.tracked_jobs[job_id] = tracker
        group_str = f", group={group_id}" if group_id else ""
        logger.info(f"[Monitor] Registered job {job_id} (SLO={slo_deadline_hours}h, {total_tokens:,} tokens{group_str})")

    def unregister_job(self, job_id: str):
        self.tracked_jobs.pop(job_id, None)
        logger.info(f"[Monitor] Unregistered job {job_id}")

    def get_group_chains(self, group_id: str) -> Dict[str, JobTracker]:
        """Get all tracked chains that belong to a job group."""
        return {jid: t for jid, t in self.tracked_jobs.items() if t.group_id == group_id}

    def unregister_group(self, group_id: str) -> List[JobTracker]:
        """Unregister all chains in a group. Returns the trackers for aggregation."""
        chains = self.get_group_chains(group_id)
        for jid in chains:
            self.tracked_jobs.pop(jid, None)
        logger.info(f"[Monitor] Unregistered group {group_id} ({len(chains)} chains)")
        return list(chains.values())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start all 3 loops as background tasks."""
        self._running = True
        self._trigger_queue = asyncio.Queue()
        self._tasks = [
            asyncio.create_task(self._telemetry_loop(), name="telemetry"),
            asyncio.create_task(self._trigger_dispatcher(), name="triggers"),
        ]
        logger.info(f"[Monitor] Started 2 async loops (telemetry={self.telemetry_interval}s, triggers=event-driven)")

    async def stop(self):
        """Cancel all loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        logger.info("[Monitor] Stopped all loops")

    # ------------------------------------------------------------------
    # Loop 1: Telemetry polling (10s, pure code, no LLM)
    # ------------------------------------------------------------------

    async def _telemetry_loop(self):
        """Poll Orca every N seconds, update trackers, check thresholds."""
        while self._running:
            try:
                for job_id in list(self.tracked_jobs.keys()):
                    await self._poll_job(job_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Monitor/L1] Telemetry error: {e}")
            # Clean up dead/completed trackers after grace period
            now = time.time()
            for jid in list(self.tracked_jobs.keys()):
                t = self.tracked_jobs[jid]
                if t.status in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED):
                    dead_since = t.last_positive_tps_at or (now - 120)
                    if now - dead_since > 60:
                        self.tracked_jobs.pop(jid, None)
                        logger.info(f"[Monitor/L1] Cleaned up dead tracker: {jid}")
            await asyncio.sleep(self.telemetry_interval)

    async def _poll_job(self, job_id: str):
        """Single poll iteration for one job."""
        tracker = self.tracked_jobs.get(job_id)
        if not tracker:
            return

        # Skip dead/completed replicas — no point polling them
        if tracker.status in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED):
            return

        # For grouped chains, Orca indexes by parent job_id — not replica_id
        orca_job_id = tracker.group_id or job_id

        # Check replica liveness from Orca (detect dead/failed/killed replicas)
        if tracker.group_id and job_id not in tracker.dead_replicas:
            try:
                replicas_resp = await self.orca.get_replicas(tracker.group_id)
                for r in replicas_resp.get("replicas", []):
                    if r["replica_id"] == job_id and r.get("phase") in ("dead", "failed", "killed"):
                        logger.warning(f"[Monitor/L1] {job_id}: Orca reports phase={r['phase']}, zeroing TPS")
                        tracker.smoothed_tps = 0
                        tracker.status = MonitoringStatus.FAILED
                        tracker.dead_replicas.append(job_id)
                        await self._emit_trigger(job_id, MonitoringStatus.FAILED,
                                                 f"Orca reports replica {r['phase']}")
                        return
            except Exception:
                pass  # fallback to EMA decay below

        # Fetch metrics from Orca
        try:
            # Per-replica throughput (individual chain), job-level chunk progress
            if tracker.group_id:
                metrics = await self.orca.get_replica_metrics(tracker.group_id, job_id)
            else:
                metrics = await self.orca.get_job_metrics(orca_job_id)
            progress = await self.orca.get_chunk_progress(orca_job_id)
        except Exception as e:
            logger.warning(f"[Monitor/L1] Failed to fetch metrics for {job_id}: {e}")
            return

        # Update throughput with EMA
        tps = metrics.get("avg_generation_throughput_toks_per_s", 0)
        if tps > 0:
            tracker.smoothed_tps = _ema(tracker.smoothed_tps, tps, EMA_ALPHA)
            tracker.last_positive_tps_at = time.time()
        elif tracker.last_positive_tps_at and tracker.smoothed_tps > 0:
            stale_seconds = time.time() - tracker.last_positive_tps_at
            if stale_seconds > 60:
                decay_factor = 0.5 ** (stale_seconds / 60)
                tracker.smoothed_tps *= decay_factor

        # Estimate tokens from chunk progress
        total_chunks = progress.get("total", 0)
        completed_chunks = progress.get("completed", 0)
        failed_chunks = progress.get("failed", 0)
        if total_chunks > 0:
            completion_frac = (completed_chunks + failed_chunks) / total_chunks
            tracker.tokens_completed = int(tracker.total_tokens * completion_frac)
            tracker.tokens_remaining = tracker.total_tokens - tracker.tokens_completed

        # Time tracking
        elapsed_s = (datetime.utcnow() - tracker.started_at).total_seconds()
        tracker.elapsed_hours = elapsed_s / 3600

        # GPU health
        tracker.gpu_cache_usage = metrics.get("gpu_cache_usage_perc", 0)
        tracker.gpu_sm_util = metrics.get("gpu_sm_util_pct", 0)
        tracker.gpu_mem_bw_util = metrics.get("gpu_mem_bw_util_pct", 0)

        # SLO projection
        tracker.slo_headroom_pct = compute_slo_headroom(
            tracker.slo_deadline_hours, tracker.elapsed_hours,
            tracker.tokens_remaining, tracker.smoothed_tps,
        )
        if tracker.smoothed_tps > 0:
            tracker.projected_eta_hours = tracker.tokens_remaining / tracker.smoothed_tps / 3600
        else:
            tracker.projected_eta_hours = float("inf")

        # Check for completion — don't emit trigger, /job/complete webhook handles outcome recording
        all_done = progress.get("all_done", False)
        if all_done or (total_chunks > 0 and (completed_chunks + failed_chunks) >= total_chunks):
            tracker.status = MonitoringStatus.COMPLETED
            logger.info(f"[Monitor/L1] {job_id}: all chunks completed, awaiting /job/complete webhook")
            return

        # Check Orca job status (use parent job_id for grouped chains)
        try:
            job_status = await self.orca.get_job_status(orca_job_id)
            if job_status.get("status") in ("failed", "cancelled"):
                tracker.status = MonitoringStatus.FAILED
                await self._emit_trigger(job_id, MonitoringStatus.FAILED,
                                         f"Orca reports status={job_status.get('status')}")
                return
        except Exception:
            pass

        # Classify status — for grouped chains, use aggregate TPS for SLO check
        if tracker.group_id:
            group_chains = self.get_group_chains(tracker.group_id)
            aggregate_tps = sum(t.smoothed_tps for t in group_chains.values())
            # Recompute headroom using aggregate throughput and full job tokens
            # All replicas share the same chunk pool — use max, not sum
            total_job_tokens = max(t.total_tokens for t in group_chains.values())
            total_remaining = max(0, total_job_tokens - int(
                total_job_tokens * ((completed_chunks + failed_chunks) / max(total_chunks, 1))
            )) if total_chunks > 0 else total_job_tokens
            tracker.slo_headroom_pct = compute_slo_headroom(
                tracker.slo_deadline_hours, tracker.elapsed_hours,
                total_remaining, aggregate_tps,
            )

        prev_status = tracker.status
        new_status = _classify_status(tracker)
        tracker.status = new_status

        # Handle warmup transition
        if not tracker.warmup_complete and new_status != MonitoringStatus.WARMING_UP:
            tracker.warmup_complete = True
            logger.info(f"[Monitor/L1] {job_id}: warmup complete, TPS={tracker.smoothed_tps:.0f}")

        # Emit triggers on state transitions (anti-windup: skip if action in progress)
        if new_status != prev_status:
            frozen = (tracker.action_in_progress and
                      tracker.action_freeze_until and
                      time.time() < tracker.action_freeze_until)
            if frozen:
                logger.debug(f"[Monitor/L1] {job_id}: anti-windup active, skipping trigger")
            elif new_status == MonitoringStatus.FALLING_BEHIND:
                await self._emit_trigger(job_id, MonitoringStatus.FALLING_BEHIND,
                                         f"Headroom={tracker.slo_headroom_pct:.1f}%, TPS={tracker.smoothed_tps:.0f}")
            elif new_status == MonitoringStatus.OVER_PROVISIONED:
                await self._emit_trigger(job_id, MonitoringStatus.OVER_PROVISIONED,
                                         f"Headroom={tracker.slo_headroom_pct:.0f}%, can shed replicas")

    async def _emit_trigger(self, job_id: str, status: MonitoringStatus, hint: str):
        """Push a trigger event to Loop 3's queue."""
        tracker = self.tracked_jobs.get(job_id)
        if not tracker:
            return
        # Group-level dedup: at most one trigger per group per status per 30s
        if tracker.group_id:
            key = f"{tracker.group_id}:{status.value}"
            last = self._group_trigger_cooldown.get(key, 0)
            if time.time() - last < 30:
                logger.debug(f"[Monitor/L1] Dedup: skipping {status.value} for {job_id} (group cooldown)")
                return
            self._group_trigger_cooldown[key] = time.time()
        trigger = MonitoringTrigger(
            trigger_type=status,
            job_id=job_id,
            job_tracker=tracker.model_dump(),
            diagnosis_hint=hint,
        )
        await self._trigger_queue.put(trigger)
        logger.info(f"[Monitor/L1] Trigger: {status.value} for {job_id} — {hint}")

    # ------------------------------------------------------------------
    # Loop 2: Trigger dispatcher (event-driven, fires agent)
    # ------------------------------------------------------------------

    async def _trigger_dispatcher(self):
        """Wait for trigger events and dispatch to agent callback."""
        while self._running:
            try:
                trigger = await asyncio.wait_for(
                    self._trigger_queue.get(), timeout=5.0
                )
                if self.on_trigger:
                    logger.info(f"[Monitor/L3] Dispatching {trigger.trigger_type.value} for {trigger.job_id}")
                    try:
                        result = await self.on_trigger(trigger)
                        logger.info(f"[Monitor/L3] Agent response: {str(result)[:200]}")
                    except Exception as e:
                        logger.error(f"[Monitor/L3] Agent handler error: {e}")
                else:
                    logger.warning(f"[Monitor/L3] No trigger callback, ignoring: {trigger.trigger_type.value}")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Monitor/L3] Dispatcher error: {e}")


# ---------------------------------------------------------------------------
# Pure functions (no state, testable)
# ---------------------------------------------------------------------------

def _ema(prev: float, new: float, alpha: float) -> float:
    """Exponential moving average."""
    if prev == 0:
        return new
    return alpha * new + (1 - alpha) * prev


def _classify_status(tracker: JobTracker) -> MonitoringStatus:
    """Classify job status with hysteresis to prevent oscillation.

    Enter/exit thresholds differ: entering FALLING_BEHIND requires headroom < 10%,
    but exiting requires headroom > 20%. This dead band prevents rapid flipping
    when headroom hovers near a boundary.
    """
    # Respect terminal states — don't reclassify dead/completed replicas
    if tracker.status in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED):
        return tracker.status

    if tracker.elapsed_hours < (WARMUP_MINUTES / 60) and not tracker.warmup_complete:
        return MonitoringStatus.WARMING_UP

    prev = tracker.status
    h = tracker.slo_headroom_pct

    # OVER_PROVISIONED: enter at 70%, exit at 50%
    if prev == MonitoringStatus.OVER_PROVISIONED:
        if h < OVER_PROVISIONED_EXIT:
            pass  # fall through to normal classification
        else:
            return MonitoringStatus.OVER_PROVISIONED
    elif (h > OVER_PROVISIONED_ENTER and
          tracker.elapsed_hours > tracker.slo_deadline_hours * OVER_PROVISIONED_MIN_ELAPSED):
        return MonitoringStatus.OVER_PROVISIONED

    # FALLING_BEHIND: enter at 10%, exit at 20%
    if prev == MonitoringStatus.FALLING_BEHIND:
        if h > FALLING_BEHIND_EXIT:
            pass  # fall through to normal classification
        else:
            return MonitoringStatus.FALLING_BEHIND
    elif h < FALLING_BEHIND_ENTER:
        return MonitoringStatus.FALLING_BEHIND

    # Normal classification (no hysteresis needed)
    if h > ON_TRACK_THRESHOLD:
        return MonitoringStatus.ON_TRACK

    return MonitoringStatus.AT_RISK


def compute_slo_headroom(
    slo_deadline_hours: float,
    elapsed_hours: float,
    tokens_remaining: int,
    smoothed_tps: float,
) -> float:
    """Compute SLO headroom percentage. >0 means on track, <0 means behind."""
    if smoothed_tps <= 0:
        return 0.0
    remaining_hours = tokens_remaining / smoothed_tps / 3600
    time_left = slo_deadline_hours - elapsed_hours
    return ((time_left - remaining_hours) / max(slo_deadline_hours, 0.01)) * 100
