"""
koi/monitor.py — Two async loops for job monitoring.

Loop 1 (TelemetryLoop): 10s polling, pure code, updates JobTracker, checks thresholds
Loop 2 (TriggerDispatcher): event-driven, fires agent on triggers

These are independent asyncio.Tasks. They do NOT share a timer.
"""

import asyncio
import math
import os
import time
from datetime import datetime
from typing import Callable, Coroutine, Dict, List, Optional

from koi.costing import evaluate_cost_roofline, project_total_cost
from koi.logging_config import get_logger, bind_context, clear_context
from koi.runtime_state import RuntimeStateStore
from koi.schemas import (
    JobTracker,
    MonitoringStatus,
    MonitoringTrigger,
    PlacementConfig,
)
from koi.tools.orca_api import OrcaClient

logger = get_logger("koi.monitor")

# Thresholds
WARMUP_MINUTES = float(os.environ.get("KOI_WARMUP_MINUTES", "5.0"))
EMA_ALPHA = 0.3  # exponential moving average smoothing

# Hysteresis thresholds (enter/exit pairs prevent oscillation)
FALLING_BEHIND_ENTER = 10.0  # headroom < 10% → enter FALLING_BEHIND
FALLING_BEHIND_EXIT = 20.0  # headroom > 20% → exit FALLING_BEHIND
ON_TRACK_THRESHOLD = 30.0  # headroom > 30% → ON_TRACK
OVER_PROVISIONED_ENTER = 70.0  # headroom > 70% → enter OVER_PROVISIONED
OVER_PROVISIONED_EXIT = 50.0  # headroom < 50% → exit OVER_PROVISIONED
OVER_PROVISIONED_MIN_ELAPSED = float(
    os.environ.get("KOI_OVERPROV_MIN_ELAPSED", "0.20")
)  # fraction of SLO
OVER_PROVISIONED_MAX_WAIT_HOURS = float(
    os.environ.get("KOI_OVERPROV_MAX_WAIT_HOURS", "0.05")
)
OVER_PROVISIONED_MIN_LIVE_REPLICAS = int(
    os.environ.get("KOI_OVERPROV_MIN_LIVE_REPLICAS", "1")
)
# Group-level health-trigger cooldown window. One FALLING_BEHIND or
# OVER_PROVISIONED event can leave the monitor's dedup filter per group per
# this many seconds. Default 30s preserves historical behaviour; bump to
# ~60s for a live customer demo if the monitor is chatty.
GROUP_HEALTH_COOLDOWN_S = float(os.environ.get("KOI_GROUP_HEALTH_COOLDOWN_S", "30"))


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
        runtime_state: Optional[RuntimeStateStore] = None,
    ):
        self.orca = orca
        self.on_trigger = on_trigger
        self.telemetry_interval = telemetry_interval
        self.runtime_state = runtime_state

        self.tracked_jobs: Dict[str, JobTracker] = {}
        self._trigger_queue: asyncio.Queue = asyncio.Queue()
        self._tasks: List[asyncio.Task] = []
        self._running = False
        # Fatal signal set by _on_task_done when a background loop dies
        # unexpectedly. Read by /health in koi/server.py.
        self._fatal: Optional[str] = None
        self._group_trigger_cooldown: Dict[
            str, float
        ] = {}  # "group_id:status" → last_emit_time
        self._koi_initiated_kills: set = set()  # replica IDs killed by scale_chain_tool
        self._pending_launches: Dict[
            str, dict
        ] = {}  # replica_id → launch info (pre-model_ready)
        # Per-replica scale-up decision mapping (replaces the old
        # group_id-keyed FIFO queue, which misattributed decisions under
        # overlapping scale ops with out-of-order replica arrivals).
        # Populated by scale_chain_tool with exactly the replica_ids Orca's
        # /job/{id}/scale response reported; consumed by /job/started.
        self._pending_replica_decisions: Dict[str, dict] = {}

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
        predicted_cost_per_hour: Optional[float] = None,
        cost_roofline_usd: Optional[float] = None,
        decision_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        if job_id in self.tracked_jobs:
            logger.info("job_already_registered", job_id=job_id)
            return
        tracker = JobTracker(
            job_id=job_id,
            decision_id=decision_id,
            group_id=group_id,
            config=config,
            slo_deadline_hours=slo_deadline_hours,
            total_tokens=total_tokens,
            predicted_tps=predicted_tps,
            predicted_cost_per_hour=predicted_cost_per_hour,
            cost_roofline_usd=cost_roofline_usd,
            tokens_remaining=total_tokens,
        )
        self.tracked_jobs[job_id] = tracker
        self.persist_job(job_id)
        logger.info(
            "job_registered",
            job_id=job_id,
            slo_hours=slo_deadline_hours,
            total_tokens=total_tokens,
            group_id=group_id,
        )

    def unregister_job(self, job_id: str):
        self.tracked_jobs.pop(job_id, None)
        if self.runtime_state:
            self.runtime_state.delete_tracked_job(job_id)
        logger.info("job_unregistered", job_id=job_id)

    def get_group_chains(self, group_id: str) -> Dict[str, JobTracker]:
        """Get all tracked chains that belong to a job group."""
        return {
            jid: t for jid, t in self.tracked_jobs.items() if t.group_id == group_id
        }

    def unregister_group(self, group_id: str) -> List[JobTracker]:
        """Unregister all chains in a group. Returns the trackers for aggregation."""
        chains = self.get_group_chains(group_id)
        for jid in chains:
            self.tracked_jobs.pop(jid, None)
            if self.runtime_state:
                self.runtime_state.delete_tracked_job(jid)
        logger.info("group_unregistered", group_id=group_id, chains=len(chains))
        return list(chains.values())

    def persist_job(self, job_id: str) -> None:
        tracker = self.tracked_jobs.get(job_id)
        if tracker and self.runtime_state:
            self.runtime_state.upsert_tracked_job(
                job_id=job_id,
                tracker=tracker.model_dump(mode="json"),
            )

    def persist_group(self, group_id: str) -> None:
        for tracker_job_id, tracker in self.tracked_jobs.items():
            if tracker.group_id == group_id:
                self.persist_job(tracker_job_id)

    def track_pending_launch(self, job_id: str, launch_info: dict) -> None:
        merged = dict(self._pending_launches.get(job_id, {}))
        merged.update(launch_info)
        merged.setdefault("launched_at", time.time())
        self._pending_launches[job_id] = merged
        if self.runtime_state:
            self.runtime_state.upsert_pending_launch(job_id, merged)

    def get_pending_launch(self, job_id: str) -> dict:
        return self._pending_launches.get(job_id, {})

    def clear_pending_launch(self, job_id: str) -> Optional[dict]:
        launch = self._pending_launches.pop(job_id, None)
        if launch and self.runtime_state:
            self.runtime_state.delete_pending_launch(job_id)
        return launch

    def clear_pending_launches_for_group(self, group_id: str) -> int:
        removed_job_ids = [
            job_id
            for job_id, launch in self._pending_launches.items()
            if launch.get("group_id") == group_id
        ]
        for job_id in removed_job_ids:
            self._pending_launches.pop(job_id, None)
            if self.runtime_state:
                self.runtime_state.delete_pending_launch(job_id)
        return len(removed_job_ids)

    def register_pending_replica_decision(
        self,
        replica_id: str,
        decision_id: str,
        scale_request_id: Optional[str] = None,
        decision: Optional[dict] = None,
    ) -> None:
        """Remember the decision that produced this replica.

        Called by scale_chain_tool for each replica_id returned by Orca's
        scale response. When the replica later fires /job/started, the
        handler does an exact-match lookup via consume_pending_replica_decision.
        """
        record = {
            "decision_id": decision_id,
            "scale_request_id": scale_request_id,
            "decision": dict(decision or {}),
            "registered_at": time.time(),
        }
        self._pending_replica_decisions[replica_id] = record
        if self.runtime_state:
            self.runtime_state.upsert_pending_replica_decision(
                replica_id=replica_id,
                decision_id=decision_id,
                decision=record["decision"],
                scale_request_id=scale_request_id,
            )

    def consume_pending_replica_decision(self, replica_id: str) -> Optional[dict]:
        """Pop and return the pending decision for this exact replica_id."""
        record = self._pending_replica_decisions.pop(replica_id, None)
        if record is None:
            return None
        if self.runtime_state:
            self.runtime_state.delete_pending_replica_decision(replica_id)
        return record

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start all 3 loops as background tasks."""
        self._running = True
        self._fatal = None
        self._trigger_queue = asyncio.Queue()
        self._tasks = [
            asyncio.create_task(self._telemetry_loop(), name="telemetry"),
            asyncio.create_task(self._trigger_dispatcher(), name="triggers"),
        ]
        for task in self._tasks:
            task.add_done_callback(self._on_task_done)
        logger.info("loops_started", telemetry_interval=self.telemetry_interval)

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Monitor a background task for unexpected termination.

        Catches TWO failure modes operator would otherwise miss:
          1. Task raised an exception.
          2. Task returned cleanly while `_running` is still True — a silent
             loop exit is just as fatal as a crash; /health must show 503.

        Intentional cancellation during stop() is NOT fatal.
        """
        if not self._running:
            return  # expected shutdown path
        if task.cancelled():
            return  # expected cancellation
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            self._fatal = f"{task.get_name()} raised: {exc!r}"
        else:
            self._fatal = f"{task.get_name()} exited unexpectedly (clean return)"
        logger.error(
            "monitor_task_died",
            task=task.get_name(),
            fatal=self._fatal,
        )

    async def stop(self):
        """Cancel all loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        logger.info("loops_stopped")

    def restore_runtime_state(self) -> dict:
        """Load persisted monitor state from the runtime store."""
        if not self.runtime_state:
            return {
                "tracked_jobs": 0,
                "pending_launches": 0,
                "pending_replica_decisions": 0,
            }

        restored_trackers: Dict[str, JobTracker] = {}
        for job_id, entry in self.runtime_state.load_tracked_jobs().items():
            tracker = JobTracker.model_validate(entry["tracker"])
            # Do not preserve in-flight freeze state across process restart.
            tracker.action_in_progress = False
            tracker.action_freeze_until = None
            tracker.consecutive_fetch_failures = 0
            restored_trackers[job_id] = tracker

        self.tracked_jobs = restored_trackers
        self._pending_launches = {
            job_id: entry["launch"]
            for job_id, entry in self.runtime_state.load_pending_launches().items()
        }
        self._pending_replica_decisions = (
            self.runtime_state.load_pending_replica_decisions()
        )

        summary = {
            "tracked_jobs": len(self.tracked_jobs),
            "pending_launches": len(self._pending_launches),
            "pending_replica_decisions": len(self._pending_replica_decisions),
        }
        if any(summary.values()):
            logger.info("monitor_restored", **summary)
        return summary

    # ------------------------------------------------------------------
    # Loop 1: Telemetry polling (10s, pure code, no LLM)
    # ------------------------------------------------------------------

    async def _telemetry_loop(self):
        """Poll Orca every N seconds, update trackers, check thresholds."""
        while self._running:
            try:
                for job_id in list(self.tracked_jobs.keys()):
                    try:
                        await asyncio.wait_for(self._poll_job(job_id), timeout=30.0)
                    except asyncio.TimeoutError:
                        tracker = self.tracked_jobs.get(job_id)
                        if tracker:
                            tracker.consecutive_fetch_failures += 1
                            logger.warning(
                                "poll_job_timeout",
                                job_id=job_id,
                                failures=tracker.consecutive_fetch_failures,
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("telemetry_error", error=str(e))
            # Clean up dead/completed trackers after grace period
            now = time.time()
            for jid in list(self.tracked_jobs.keys()):
                t = self.tracked_jobs[jid]
                if t.status in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED):
                    dead_since = t.last_positive_tps_at or (now - 120)
                    if now - dead_since > 60:
                        self.tracked_jobs.pop(jid, None)
                        logger.info("tracker_cleaned_up", job_id=jid)
            await asyncio.sleep(self.telemetry_interval)

    async def _poll_job(self, job_id: str):
        """Single poll iteration for one job."""
        tracker = self.tracked_jobs.get(job_id)
        if not tracker:
            return

        try:
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
                        if r["replica_id"] == job_id and r.get("phase") in (
                            "dead",
                            "failed",
                            "killed",
                        ):
                            tracker.smoothed_tps = 0
                            tracker.dead_replicas.append(job_id)
                            # Intentional kills (from scale_chain_tool) don't trigger FAILED
                            if job_id in self._koi_initiated_kills:
                                tracker.status = (
                                    MonitoringStatus.COMPLETED
                                )  # clean exit
                                self._koi_initiated_kills.discard(job_id)
                                logger.info("intentional_kill_cleanup", job_id=job_id)
                            else:
                                tracker.status = MonitoringStatus.FAILED
                                logger.warning(
                                    "replica_dead", job_id=job_id, phase=r["phase"]
                                )
                                await self._emit_trigger(
                                    job_id,
                                    MonitoringStatus.FAILED,
                                    f"Orca reports replica {r['phase']}",
                                )
                            return
                except Exception as e:
                    tracker.consecutive_fetch_failures += 1
                    logger.warning(
                        "replica_check_failed",
                        job_id=job_id,
                        error=str(e),
                        failures=tracker.consecutive_fetch_failures,
                    )

            # Fetch metrics from Orca
            try:
                # Per-replica throughput (individual chain), job-level chunk progress
                if tracker.group_id:
                    metrics = await self.orca.get_replica_metrics(
                        tracker.group_id, job_id
                    )
                else:
                    metrics = await self.orca.get_job_metrics(orca_job_id)
                progress = await self.orca.get_chunk_progress(orca_job_id)
            except Exception as e:
                tracker.consecutive_fetch_failures += 1
                logger.warning(
                    "metrics_fetch_failed",
                    job_id=job_id,
                    error=str(e),
                    failures=tracker.consecutive_fetch_failures,
                )
                return

            # Metrics fetched successfully — reset failure counter
            tracker.consecutive_fetch_failures = 0
            tracker.last_metrics_update = time.time()

            # Update throughput with EMA
            tps = metrics.get("avg_generation_throughput_toks_per_s", 0)
            if tps > 0 and math.isfinite(tps):
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
                new_completed = int(tracker.total_tokens * completion_frac)
                tracker.tokens_completed = max(tracker.tokens_completed, new_completed)
                tracker.tokens_remaining = (
                    tracker.total_tokens - tracker.tokens_completed
                )

            # Time tracking
            elapsed_s = (datetime.utcnow() - tracker.started_at).total_seconds()
            tracker.elapsed_hours = elapsed_s / 3600

            # GPU health
            tracker.gpu_cache_usage = metrics.get("gpu_cache_usage_perc", 0)
            tracker.gpu_sm_util = metrics.get("gpu_sm_util_pct", 0)
            tracker.gpu_mem_bw_util = metrics.get("gpu_mem_bw_util_pct", 0)

            # SLO projection
            tracker.slo_headroom_pct = compute_slo_headroom(
                tracker.slo_deadline_hours,
                tracker.elapsed_hours,
                tracker.tokens_remaining,
                tracker.smoothed_tps,
            )
            if tracker.smoothed_tps > 0:
                tracker.projected_eta_hours = (
                    tracker.tokens_remaining / tracker.smoothed_tps / 3600
                )
            else:
                tracker.projected_eta_hours = float("inf")

            # Cost projection
            (
                tracker.projected_remaining_cost_usd,
                tracker.projected_total_cost_usd,
            ) = project_total_cost(
                tracker.predicted_cost_per_hour,
                tracker.elapsed_hours,
                tracker.projected_eta_hours,
            )
            (
                tracker.meets_cost_roofline,
                tracker.cost_overage_usd,
            ) = evaluate_cost_roofline(
                tracker.projected_total_cost_usd,
                tracker.cost_roofline_usd,
            )

            # Check for completion — don't emit trigger, /job/complete webhook handles outcome recording
            all_done = progress.get("all_done", False)
            if all_done or (
                total_chunks > 0 and (completed_chunks + failed_chunks) >= total_chunks
            ):
                tracker.status = MonitoringStatus.COMPLETED
                logger.info("chunks_completed", job_id=job_id)
                return

            # Check Orca job status (use parent job_id for grouped chains)
            try:
                job_status = await self.orca.get_job_status(orca_job_id)
                if job_status.get("status") in ("failed", "cancelled"):
                    tracker.status = MonitoringStatus.FAILED
                    await self._emit_trigger(
                        job_id,
                        MonitoringStatus.FAILED,
                        f"Orca reports status={job_status.get('status')}",
                    )
                    return
            except Exception as e:
                tracker.consecutive_fetch_failures += 1
                logger.warning(
                    "job_status_check_failed",
                    job_id=job_id,
                    error=str(e),
                    failures=tracker.consecutive_fetch_failures,
                )

            # Classify status — for grouped chains, use aggregate TPS for SLO check
            live_replica_count = None
            if tracker.group_id:
                group_chains = self.get_group_chains(tracker.group_id)
                if not group_chains:
                    # Race: all trackers removed mid-poll (e.g., /job/complete webhook)
                    return
                # Exclude dead/completed replicas from aggregate — they contribute 0 TPS
                # and drag down headroom calculation unnecessarily
                live_chains = {
                    k: v
                    for k, v in group_chains.items()
                    if v.status
                    not in (MonitoringStatus.FAILED, MonitoringStatus.COMPLETED)
                }
                live_replica_count = len(live_chains)
                aggregate_tps = (
                    sum(t.smoothed_tps for t in live_chains.values())
                    if live_chains
                    else 0
                )
                # Recompute headroom using aggregate throughput and full job tokens
                # All replicas share the same chunk pool — use max, not sum
                total_job_tokens = max(t.total_tokens for t in group_chains.values())
                total_remaining = (
                    max(
                        0,
                        total_job_tokens
                        - int(
                            total_job_tokens
                            * (
                                (completed_chunks + failed_chunks)
                                / max(total_chunks, 1)
                            )
                        ),
                    )
                    if total_chunks > 0
                    else total_job_tokens
                )
                tracker.slo_headroom_pct = compute_slo_headroom(
                    tracker.slo_deadline_hours,
                    tracker.elapsed_hours,
                    total_remaining,
                    aggregate_tps,
                )

            prev_status = tracker.status
            new_status = _classify_status(tracker, active_replicas=live_replica_count)
            tracker.status = new_status

            # Handle warmup transition
            if (
                not tracker.warmup_complete
                and new_status != MonitoringStatus.WARMING_UP
            ):
                tracker.warmup_complete = True
                logger.info(
                    "warmup_complete", job_id=job_id, tps=round(tracker.smoothed_tps)
                )

            # Emit triggers on state transitions (anti-windup: skip if action in progress)
            if new_status != prev_status:
                frozen = (
                    tracker.action_in_progress
                    and tracker.action_freeze_until
                    and time.time() < tracker.action_freeze_until
                )
                if frozen:
                    logger.debug("anti_windup_skip", job_id=job_id)
                elif new_status == MonitoringStatus.FALLING_BEHIND:
                    await self._emit_trigger(
                        job_id,
                        MonitoringStatus.FALLING_BEHIND,
                        f"Headroom={tracker.slo_headroom_pct:.1f}%, TPS={tracker.smoothed_tps:.0f}",
                    )
                elif new_status == MonitoringStatus.OVER_PROVISIONED:
                    await self._emit_trigger(
                        job_id,
                        MonitoringStatus.OVER_PROVISIONED,
                        f"Headroom={tracker.slo_headroom_pct:.0f}%, can shed replicas",
                    )
        finally:
            if job_id in self.tracked_jobs:
                self.persist_job(job_id)

    async def _emit_trigger(self, job_id: str, status: MonitoringStatus, hint: str):
        """Push a trigger event to Loop 3's queue.

        For grouped jobs (multi-replica), we treat FALLING_BEHIND and
        OVER_PROVISIONED as *job-level* events and deliberately emit at most
        one per group per ``KOI_GROUP_HEALTH_COOLDOWN_S`` seconds (default
        30s; bump to ~60s for live customer demos where chatter is costly).
        Per-chain FAILED triggers are still emitted individually because
        they describe a specific replica's fate, not the job's health. The
        cooldown is also suppressed when any sibling chain in the same
        group already has an action in flight (scale/kill), to stop us
        from piling on top of an in-progress remediation.
        """
        tracker = self.tracked_jobs.get(job_id)
        if not tracker:
            return
        # Don't emit non-FAILED triggers for replicas already marked FAILED
        # (prevents FALLING_BEHIND racing ahead of FAILED webhook)
        if (
            tracker.status == MonitoringStatus.FAILED
            and status != MonitoringStatus.FAILED
        ):
            return

        # Group-level dedup only applies to health triggers (not FAILED).
        is_health_trigger = status in (
            MonitoringStatus.FALLING_BEHIND,
            MonitoringStatus.OVER_PROVISIONED,
        )
        if tracker.group_id and is_health_trigger:
            now = time.time()

            # Suppress if any chain in the group is mid-action.
            group_chains = self.get_group_chains(tracker.group_id)
            for sibling in group_chains.values():
                if (
                    sibling.action_in_progress
                    and sibling.action_freeze_until
                    and now < sibling.action_freeze_until
                ):
                    logger.debug(
                        "trigger_dedup_group_action_in_flight",
                        job_id=job_id,
                        status=status.value,
                        sibling=sibling.job_id,
                    )
                    return

            # Per-group health cooldown, status-agnostic: one health trigger
            # per group per window (configurable via KOI_GROUP_HEALTH_COOLDOWN_S,
            # default 30s).
            key = f"{tracker.group_id}:health"
            last = self._group_trigger_cooldown.get(key, 0)
            if now - last < GROUP_HEALTH_COOLDOWN_S:
                logger.debug(
                    "trigger_dedup_group_cooldown",
                    job_id=job_id,
                    status=status.value,
                    cooldown_s=GROUP_HEALTH_COOLDOWN_S,
                )
                return
            self._group_trigger_cooldown[key] = now

        trigger = MonitoringTrigger(
            trigger_type=status,
            job_id=job_id,
            job_tracker=tracker.model_dump(),
            diagnosis_hint=hint,
        )
        await self._trigger_queue.put(trigger)
        logger.info("trigger_emitted", job_id=job_id, status=status.value, hint=hint)

    # ------------------------------------------------------------------
    # Loop 2: Trigger dispatcher (event-driven, fires agent)
    # ------------------------------------------------------------------

    async def _trigger_dispatcher(self):
        """Wait for trigger events and dispatch to agent callback."""
        while self._running:
            try:
                trigger = await asyncio.wait_for(self._trigger_queue.get(), timeout=5.0)
                if self.on_trigger:
                    logger.info(
                        "trigger_dispatching",
                        job_id=trigger.job_id,
                        trigger_type=trigger.trigger_type.value,
                    )
                    try:
                        result = await self.on_trigger(trigger)
                        logger.info(
                            "trigger_handled",
                            job_id=trigger.job_id,
                            response=str(result)[:200],
                        )
                    except Exception as e:
                        logger.error(
                            "trigger_handler_error", job_id=trigger.job_id, error=str(e)
                        )
                else:
                    logger.warning(
                        "no_trigger_callback", trigger_type=trigger.trigger_type.value
                    )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("dispatcher_error", error=str(e))


# ---------------------------------------------------------------------------
# Pure functions (no state, testable)
# ---------------------------------------------------------------------------


def _ema(prev: float, new: float, alpha: float) -> float:
    """Exponential moving average."""
    if prev == 0:
        return new
    return alpha * new + (1 - alpha) * prev


def _required_overprovision_elapsed_hours(slo_deadline_hours: float) -> float:
    """How long to wait before considering a job safely overprovisioned.

    The old 20%-of-SLO gate was far too conservative for long-deadline jobs.
    Cap it to a small absolute maximum so downscale triggers can still happen on
    practical timescales.
    """
    return max(
        WARMUP_MINUTES / 60,
        min(
            slo_deadline_hours * OVER_PROVISIONED_MIN_ELAPSED,
            OVER_PROVISIONED_MAX_WAIT_HOURS,
        ),
    )


def _classify_status(
    tracker: JobTracker, active_replicas: Optional[int] = None
) -> MonitoringStatus:
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
    can_shed_replicas = (
        active_replicas is None or active_replicas >= OVER_PROVISIONED_MIN_LIVE_REPLICAS
    )

    # OVER_PROVISIONED: enter at 70%, exit at 50%
    if prev == MonitoringStatus.OVER_PROVISIONED:
        if not can_shed_replicas or h < OVER_PROVISIONED_EXIT:
            pass  # fall through to normal classification
        else:
            return MonitoringStatus.OVER_PROVISIONED
    elif (
        can_shed_replicas
        and h > OVER_PROVISIONED_ENTER
        and tracker.elapsed_hours
        >= _required_overprovision_elapsed_hours(tracker.slo_deadline_hours)
    ):
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
    """Compute SLO headroom as a percentage of time left.

    0% means the current throughput is exactly enough to finish on time.
    Positive means slack remains; negative means the job is projected to miss.
    """
    if tokens_remaining <= 0:
        return 0.0

    time_left = slo_deadline_hours - elapsed_hours
    if time_left <= 0:
        return -100.0
    if smoothed_tps <= 0:
        return -100.0

    remaining_hours = tokens_remaining / smoothed_tps / 3600
    headroom = ((time_left - remaining_hours) / max(time_left, 0.01)) * 100
    return max(-100.0, headroom)
