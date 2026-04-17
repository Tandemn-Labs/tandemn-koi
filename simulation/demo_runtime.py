"""Runtime state and streaming snapshots for demo sessions."""

from __future__ import annotations

import math
import random
import threading
import time
from copy import deepcopy
from typing import Any, Optional

from simulation.demo_scenarios import due_scenario_events
from simulation.sim_engine import SimJob, SimReplica

TERMINAL_REPLICA_PHASES = {"dead", "killed", "failed", "completed"}

# Display-only Gaussian noise tuning. We only wobble the rendered TPS — underlying
# base_tps used for chunk progression, SLO headroom, and agent decisions is untouched
# so the simulator stays deterministic under the hood while the UI feels alive.
#
# Tests that assert exact TPS values can call ``set_tps_noise_sigma(0.0)`` to disable
# the jitter. The default of 4.5% matches light realistic noise on a well-tuned
# continuous-batching runtime.
TPS_NOISE_SIGMA = 0.045  # 4.5% standard deviation on the rendered value
TPS_NOISE_SLOT_SEC = 2.0  # jitter bucket: same value for ~2 seconds per replica


def set_tps_noise_sigma(sigma: float) -> None:
    """Adjust (or disable with 0.0) the display-time Gaussian noise sigma at runtime."""
    global TPS_NOISE_SIGMA
    TPS_NOISE_SIGMA = max(0.0, float(sigma))


class DemoSessionManager:
    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self.sessions.clear()

    def fail_active_sessions(self, now: Optional[float] = None) -> int:
        """Mark all non-terminal sessions as failed. Frees quota and stops progress."""
        now = float(now or time.time())
        failed = 0
        with self._lock:
            for session in self.sessions.values():
                job: SimJob = session["_orca"]["job"]
                if job.status in {"succeeded", "failed"}:
                    continue
                for replica in job.replicas.values():
                    if replica.phase not in TERMINAL_REPLICA_PHASES:
                        replica.phase = "failed"
                job.status = "failed"
                runtime = session["runtime"]
                runtime["status"] = "launch_failed"
                runtime["error"] = "Session abandoned on page reload"
                session["launch_failure"] = {
                    "reason": "abandoned",
                    "message": "Session abandoned on page reload",
                }
                failed += 1
        return failed

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = deepcopy(payload)
            session_id = session["session_id"]
            created_at = float(session["created_at"])
            launch_started_at = session.get("launch_started_at", created_at)
            model_name = (
                ((session.get("model") or {}).get("model_name"))
                or session.get("request", {}).get("model_name")
                or "unknown"
            )
            request = session["request"]
            launch_config = session.get("launch_config") or {
                "gpu_type": session.get("launch_preview", {}).get(
                    "preferred_gpu", "L40S"
                ),
                "instance_type": session.get("launch_preview", {}).get(
                    "instance_type", "g6e.12xlarge"
                ),
                "tp": session.get("launch_preview", {}).get("tp", 4),
                "pp": session.get("launch_preview", {}).get("pp", 1),
                "dp": 1,
                "region": session.get("launch_preview", {}).get("region", "us-east-1"),
                "market": session.get("launch_preview", {}).get("market", "on_demand"),
                "decision_id": ((session.get("koi") or {}).get("decision") or {}).get(
                    "_decision_id"
                ),
                "predicted_tps": float(
                    session.get("launch_preview", {}).get("baseline_replica_tps", 0.0)
                ),
                "total_tokens": 0,
                "is_fallback": False,
                "num_instances_per_replica": 1,
            }
            session["launch_config"] = launch_config
            session["launch_failure"] = session.get("launch_failure")
            total_tokens = session["request"]["total_chunks"] * (
                session["request"]["avg_input_tokens"]
                + session["request"]["avg_output_tokens"]
            )
            launch_config["total_tokens"] = int(total_tokens)
            tokens_per_chunk = max(
                session["request"]["avg_input_tokens"]
                + session["request"]["avg_output_tokens"],
                1,
            )

            session["_bridge"] = {
                "group_id": session_id,
                "launching_sent": set(),
                "started_sent": set(),
                "config_attempted_sent": set(),
                "last_heartbeat_phase": {},
                "last_heartbeat_at": {},
                "pending_config_attempts": [],
                "launch_failed_payload": None,
                "launch_failed_sent": False,
                "replica_failed_sent": set(),
                "replica_failure_reasons": {},
                "complete_sent": False,
            }

            job = SimJob(
                job_id=session_id,
                model_name=model_name,
                total_chunks=request["total_chunks"],
                status="launching" if launch_started_at is not None else "queued",
                slo_deadline_hours=request.get("slo_deadline_hours", 8.0),
                decision_id=((session.get("koi") or {}).get("decision") or {}).get(
                    "_decision_id"
                ),
                tokens_per_chunk=tokens_per_chunk,
                deploy_timestamp=launch_started_at or created_at,
            )

            baseline_tps = float(session["launch_preview"]["baseline_replica_tps"])
            launch_timing = session["launch_preview"]["launch_timing_s"]
            replica_schedules: dict[str, dict[str, float]] = {}

            session["_orca"] = {
                "job": job,
                "replica_schedules": replica_schedules,
                "manual_tps_overrides": {},
                "scenario_tps_overrides": {},
                "next_replica_index": 0,
                "last_refreshed_at": created_at,
                "progress_updated_at": created_at,
                "partial_tokens": 0.0,
                "last_manual_event_at": created_at,
                "last_nonzero_aggregate_tps": 0.0,
            }
            session["launch_started_at"] = launch_started_at
            session["runtime"] = {
                "active_replicas": 0,
                "baseline_replica_tps": baseline_tps,
                "events": [],
                "emitted_event_ids": [],
                "status": "launching"
                if launch_started_at is not None
                else "koi_deciding",
                "launch_phase": "searching_capacity"
                if launch_started_at is not None
                else "waiting_for_koi",
                "aggregate_tps": 0.0,
                "progress_pct": 0.0,
                "tokens_completed": 0,
                "tokens_total": total_tokens,
                "eta_seconds": None,
                "slo_headroom_pct": None,
                "replicas": [],
                "error": None,
            }
            if (
                launch_started_at is not None
                and session["scenario"]["initial_replicas"] > 0
            ):
                self._schedule_replicas(
                    session,
                    now=float(launch_started_at),
                    count=session["scenario"]["initial_replicas"],
                    gpu_type=launch_config["gpu_type"],
                    instance_type=launch_config["instance_type"],
                    num_instances_per_replica=int(
                        launch_config.get("num_instances_per_replica", 1) or 1
                    ),
                    tp=launch_config["tp"],
                    pp=launch_config["pp"],
                    region=launch_config["region"],
                    market=launch_config["market"],
                    base_tps=baseline_tps,
                    launch_timing_s=launch_timing,
                    decision_id=launch_config.get("decision_id"),
                    config_index=int(launch_config.get("candidate_index", 0) or 0),
                )
            self.sessions[session_id] = session
            return self.snapshot(session_id, now=created_at)

    def activate_session(
        self,
        session_id: str,
        *,
        now: float,
        launch_config: dict[str, Any],
        baseline_tps: float,
        launch_timing_s: dict[str, float],
    ) -> dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            if session.get("launch_started_at") is not None:
                return self.snapshot(session_id, now=now)

            self._prepare_session_for_launch_locked(
                session,
                now=float(now),
                launch_config=launch_config,
                baseline_tps=float(baseline_tps),
                launch_timing_s=launch_timing_s,
            )
            self._schedule_replicas(
                session,
                now=float(now),
                count=session["scenario"]["initial_replicas"],
                gpu_type=launch_config["gpu_type"],
                instance_type=launch_config["instance_type"],
                num_instances_per_replica=int(
                    launch_config.get("num_instances_per_replica", 1) or 1
                ),
                tp=launch_config["tp"],
                pp=launch_config["pp"],
                region=launch_config["region"],
                market=launch_config["market"],
                base_tps=float(baseline_tps),
                launch_timing_s=launch_timing_s,
                decision_id=launch_config.get("decision_id"),
                config_index=int(launch_config.get("candidate_index", 0) or 0),
            )
            return self.snapshot(session_id, now=now)

    def activate_session_with_candidates(
        self,
        session_id: str,
        *,
        now: float,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            bridge = session["_bridge"]
            if session.get("launch_started_at") is not None:
                return self.snapshot(session_id, now=now)

            last_rejection: Optional[dict[str, Any]] = None
            configs_tried: list[dict[str, Any]] = []
            failure_reasons: list[str] = []
            for candidate in candidates:
                admission = self._admit_replicas_locked(
                    session,
                    now=float(now),
                    count=session["scenario"]["initial_replicas"],
                    gpu_type=str(candidate["gpu_type"]),
                    preferred_instance_type=candidate.get("instance_type"),
                    tp=int(candidate["tp"]),
                    pp=int(candidate["pp"]),
                    preferred_region=candidate.get("region"),
                    preferred_market=candidate.get("market"),
                    base_tps=float(candidate["predicted_tps"]),
                    launch_timing_s=deepcopy(candidate["launch_timing_s"]),
                    commit=False,
                )
                if admission["status"] != "accepted":
                    attempted_config = {
                        "instance_type": candidate.get("instance_type")
                        or admission.get("instance_type", "unknown"),
                        "gpu_type": str(candidate["gpu_type"]),
                        "region": str(
                            candidate.get("region")
                            or admission.get("region")
                            or "unknown"
                        ),
                        "market": str(
                            candidate.get("market")
                            or admission.get("market")
                            or "on_demand"
                        ),
                    }
                    bridge["pending_config_attempts"].append(
                        {
                            "config": attempted_config,
                            "payload": {
                                "job_id": session_id,
                                "decision_id": candidate.get("decision_id"),
                                "instance_type": attempted_config["instance_type"],
                                "gpu_type": attempted_config["gpu_type"],
                                "region": attempted_config["region"],
                                "market": attempted_config["market"],
                                "launched": False,
                                "attempt_index": int(
                                    candidate.get("candidate_index", 0) or 0
                                ),
                                "failure_reason": admission.get(
                                    "message", "candidate rejected"
                                ),
                                "time_to_launch": 0.0,
                            },
                        }
                    )
                    configs_tried.append(attempted_config)
                    failure_reasons.append(
                        admission.get("message", "candidate rejected")
                    )
                    session["runtime"]["events"].append(
                        {
                            "event_id": f"launch-reject-{session_id}-{candidate.get('candidate_index', 0)}-{int(now)}",
                            "at_seconds": round(float(now) - session["created_at"], 1),
                            "action": "launch_candidate_rejected",
                            "label": "Candidate rejected",
                            "description": admission.get(
                                "message", "Candidate rejected."
                            ),
                            "params": {
                                "gpu_type": attempted_config["gpu_type"],
                                "instance_type": attempted_config["instance_type"],
                                "attempt_index": int(
                                    candidate.get("candidate_index", 0) or 0
                                ),
                            },
                        }
                    )
                    last_rejection = admission
                    continue

                launch_config = {
                    "job_id": f"{session_id}-r0",
                    "group_id": session_id,
                    "decision_id": candidate.get("decision_id"),
                    "gpu_type": admission["gpu_type"],
                    "instance_type": admission["instance_type"],
                    "tp": admission["tp"],
                    "pp": admission["pp"],
                    "dp": int(candidate.get("dp", 1) or 1),
                    "region": admission["region"],
                    "market": admission["market"],
                    "total_tokens": session["launch_config"]["total_tokens"],
                    "predicted_tps": float(candidate["predicted_tps"]),
                    "is_fallback": bool(candidate.get("is_fallback", False)),
                    "num_instances_per_replica": admission["instances_per_replica"],
                    "candidate_index": int(candidate.get("candidate_index", 0) or 0),
                    "candidate_source": candidate.get("source", "unknown"),
                }
                self._prepare_session_for_launch_locked(
                    session,
                    now=float(now),
                    launch_config=launch_config,
                    baseline_tps=float(candidate["predicted_tps"]),
                    launch_timing_s=candidate["launch_timing_s"],
                )
                self._schedule_replicas(
                    session,
                    now=float(now),
                    count=session["scenario"]["initial_replicas"],
                    gpu_type=admission["gpu_type"],
                    instance_type=admission["instance_type"],
                    num_instances_per_replica=admission["instances_per_replica"],
                    tp=admission["tp"],
                    pp=admission["pp"],
                    region=admission["region"],
                    market=admission["market"],
                    base_tps=float(candidate["predicted_tps"]),
                    launch_timing_s=candidate["launch_timing_s"],
                    decision_id=candidate.get("decision_id"),
                    config_index=int(candidate.get("candidate_index", 0) or 0),
                )
                session["runtime"]["events"].append(
                    {
                        "event_id": f"launch-accepted-{session_id}-{int(now)}",
                        "at_seconds": round(float(now) - session["created_at"], 1),
                        "action": "launch_selected",
                        "label": "Launch config admitted",
                        "description": (
                            f"{admission['gpu_type']} TP {admission['tp']} PP {admission['pp']} "
                            f"on {admission['instance_type']} in {admission['region']} "
                            f"({admission['market']})."
                        ),
                        "params": {
                            "instance_type": admission["instance_type"],
                            "region": admission["region"],
                            "market": admission["market"],
                            "is_fallback": launch_config["is_fallback"],
                            "attempt_index": launch_config["candidate_index"],
                        },
                    }
                )
                return self.snapshot(session_id, now=now)

            session["launch_failure"] = (
                deepcopy(last_rejection)
                if last_rejection
                else {
                    "status": "error",
                    "reason": "no_candidate",
                    "message": "No launch candidate was provided.",
                }
            )
            runtime = session["runtime"]
            runtime["status"] = "launch_failed"
            runtime["launch_phase"] = "rejected"
            runtime["aggregate_tps"] = 0.0
            runtime["progress_pct"] = 0.0
            runtime["tokens_completed"] = 0
            runtime["eta_seconds"] = None
            runtime["slo_headroom_pct"] = None
            runtime["replicas"] = []
            runtime["error"] = session["launch_failure"].get("message")
            runtime["events"].append(
                {
                    "event_id": f"launch-rejected-{session_id}-{int(now)}",
                    "at_seconds": round(float(now) - session["created_at"], 1),
                    "action": "launch_rejected",
                    "label": "Launch rejected",
                    "description": runtime["error"],
                    "params": {
                        "reason": session["launch_failure"].get("reason"),
                    },
                }
            )
            if configs_tried:
                first_decision_id = next(
                    (
                        candidate.get("decision_id")
                        for candidate in candidates
                        if candidate.get("decision_id")
                    ),
                    None,
                )
                bridge["launch_failed_payload"] = {
                    "job_id": session_id,
                    "decision_id": first_decision_id,
                    "configs_tried": configs_tried,
                    "failure_reasons": failure_reasons,
                    "total_time_seconds": 0.0,
                }
            return self.snapshot(session_id, now=now)

    def _prepare_session_for_launch_locked(
        self,
        session: dict[str, Any],
        *,
        now: float,
        launch_config: dict[str, Any],
        baseline_tps: float,
        launch_timing_s: dict[str, float],
    ) -> None:
        session["launch_started_at"] = float(now)
        session["launch_failure"] = None
        session["launch_config"] = deepcopy(launch_config)
        session["launch_preview"]["baseline_replica_tps"] = float(baseline_tps)
        session["launch_preview"]["launch_timing_s"] = deepcopy(launch_timing_s)
        session["launch_preview"]["preferred_gpu"] = launch_config["gpu_type"]
        session["launch_preview"]["instance_type"] = launch_config["instance_type"]
        session["launch_preview"]["region"] = launch_config["region"]
        session["launch_preview"]["market"] = launch_config["market"]
        session["launch_preview"]["tp"] = launch_config["tp"]
        session["launch_preview"]["pp"] = launch_config["pp"]

        state = session["_orca"]
        job: SimJob = state["job"]
        job.status = "launching"
        job.deploy_timestamp = float(now)
        job.decision_id = launch_config.get("decision_id")
        state["replica_schedules"].clear()
        state["manual_tps_overrides"].clear()
        state["scenario_tps_overrides"].clear()
        state["next_replica_index"] = 0
        state["progress_updated_at"] = float(now)
        state["partial_tokens"] = 0.0
        state["last_nonzero_aggregate_tps"] = 0.0
        job.replicas.clear()

        runtime = session["runtime"]
        runtime["status"] = "launching"
        runtime["launch_phase"] = "searching_capacity"
        runtime["aggregate_tps"] = 0.0
        runtime["progress_pct"] = 0.0
        runtime["tokens_completed"] = 0
        runtime["eta_seconds"] = None
        runtime["slo_headroom_pct"] = None
        runtime["replicas"] = []
        runtime["error"] = None
        bridge = session["_bridge"]
        bridge["launching_sent"].clear()
        bridge["started_sent"].clear()
        bridge["config_attempted_sent"].clear()
        bridge["last_heartbeat_phase"].clear()
        bridge["last_heartbeat_at"].clear()
        bridge["pending_config_attempts"].clear()
        bridge["replica_failed_sent"].clear()
        bridge["replica_failure_reasons"].clear()
        bridge["complete_sent"] = False
        bridge["launch_failed_payload"] = None
        bridge["launch_failed_sent"] = False

    def snapshot(self, session_id: str, now: Optional[float] = None) -> dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            now = float(now or time.time())
            self._refresh_orca_state(session, now)

            runtime = session["runtime"]
            job: SimJob = session["_orca"]["job"]
            launch_started_at = session.get("launch_started_at")
            if launch_started_at is None:
                runtime["aggregate_tps"] = 0.0
                runtime["active_replicas"] = 0
                runtime["tokens_completed"] = 0
                runtime["progress_pct"] = 0.0
                runtime["eta_seconds"] = None
                runtime["slo_headroom_pct"] = None
                runtime["replicas"] = []
                if session.get("launch_failure"):
                    runtime["status"] = "launch_failed"
                    runtime["launch_phase"] = "rejected"
                    runtime["error"] = session["launch_failure"].get("message")
                else:
                    runtime["launch_phase"] = "waiting_for_koi"
                    runtime["status"] = "koi_deciding"
                    runtime["error"] = None

                snapshot = deepcopy(session)
                for key in list(snapshot.keys()):
                    if key.startswith("_"):
                        snapshot.pop(key)
                snapshot["runtime"] = deepcopy(runtime)
                snapshot["runtime"]["elapsed_seconds"] = round(
                    now - session["created_at"], 1
                )
                snapshot["runtime"]["launch_elapsed_seconds"] = 0.0
                return snapshot

            aggregate_tps = self._aggregate_job_tps(session, now)
            if aggregate_tps > 0:
                session["_orca"]["last_nonzero_aggregate_tps"] = float(aggregate_tps)
            running_replicas = self._running_replicas(session)
            launching_phase = self._session_launch_phase(session, now)

            # Display aggregate uses jittered per-replica values so the UI wobbles
            # coherently with the per-replica TPS, while progress accounting uses the
            # deterministic aggregate.
            display_aggregate_tps = sum(
                self._replica_display_tps(session, replica, now)
                for replica in job.replicas.values()
            )
            runtime["aggregate_tps"] = round(display_aggregate_tps, 1)
            runtime["active_replicas"] = len(running_replicas)
            runtime["launch_phase"] = launching_phase
            runtime["tokens_completed"] = min(
                runtime["tokens_total"],
                job.completed_chunks * max(job.tokens_per_chunk, 1),
            )
            runtime["progress_pct"] = round(
                (runtime["tokens_completed"] / max(runtime["tokens_total"], 1)) * 100, 2
            )
            remaining_tokens = max(
                0, runtime["tokens_total"] - runtime["tokens_completed"]
            )
            if job.status == "succeeded" or remaining_tokens == 0:
                runtime["status"] = "completed"
                runtime["eta_seconds"] = 0.0
            elif running_replicas:
                runtime["status"] = "running"
                runtime["eta_seconds"] = round(
                    remaining_tokens / max(aggregate_tps, 1), 1
                )
            else:
                ready_times = [
                    schedule["ready_at"]
                    for replica_id, schedule in session["_orca"][
                        "replica_schedules"
                    ].items()
                    if job.replicas.get(replica_id)
                    and job.replicas[replica_id].phase not in TERMINAL_REPLICA_PHASES
                ]
                if ready_times:
                    runtime["status"] = "launching"
                    runtime["eta_seconds"] = round(max(0.0, min(ready_times) - now), 1)
                else:
                    runtime["status"] = "stalled"
                    runtime["eta_seconds"] = None

            deadline_seconds = (
                float(session["request"].get("slo_deadline_hours", 8.0)) * 3600.0
            )
            elapsed_seconds = now - session["created_at"]
            if deadline_seconds <= 0:
                runtime["slo_headroom_pct"] = None
            elif remaining_tokens <= 0:
                runtime["slo_headroom_pct"] = 0.0
            elif aggregate_tps <= 0 and remaining_tokens > 0 and not running_replicas:
                runtime["slo_headroom_pct"] = -100.0
            elif runtime["eta_seconds"] is None:
                runtime["slo_headroom_pct"] = None
            else:
                time_left_seconds = deadline_seconds - elapsed_seconds
                if time_left_seconds <= 0:
                    runtime["slo_headroom_pct"] = -100.0
                else:
                    runtime["slo_headroom_pct"] = round(
                        max(
                            -100.0,
                            (
                                (time_left_seconds - float(runtime["eta_seconds"]))
                                / max(time_left_seconds, 1.0)
                            )
                            * 100.0,
                        ),
                        1,
                    )
            runtime["error"] = None

            runtime["replicas"] = [
                self._public_replica(session, replica, now)
                for replica in job.replicas.values()
            ]

            snapshot = deepcopy(session)
            for key in list(snapshot.keys()):
                if key.startswith("_"):
                    snapshot.pop(key)
            snapshot["runtime"] = deepcopy(runtime)
            snapshot["runtime"]["elapsed_seconds"] = round(
                now - session["created_at"], 1
            )
            snapshot["runtime"]["launch_elapsed_seconds"] = round(
                now - float(launch_started_at), 1
            )
            return snapshot

    def get_orca_job_status(
        self, session_id: str, now: Optional[float] = None
    ) -> dict[str, Any]:
        session = self.snapshot(session_id, now=now)
        runtime = session["runtime"]
        return {
            "job_id": session_id,
            "status": runtime["status"],
            "model_name": session["model"]["model_name"],
            "num_replicas": len(runtime["replicas"]),
            "active_replicas": runtime["active_replicas"],
        }

    def get_orca_job_metrics(
        self, session_id: str, now: Optional[float] = None
    ) -> dict[str, Any]:
        session = self.snapshot(session_id, now=now)
        runtime = session["runtime"]
        active = runtime["active_replicas"]
        return {
            "avg_generation_throughput_toks_per_s": runtime["aggregate_tps"],
            "gpu_cache_usage_perc": 0.55 + min(active, 8) * 0.03 if active else 0.0,
            "num_requests_running": 16 * max(active, 1) if active else 0,
            "num_requests_waiting": 2 if runtime["status"] == "launching" else 0,
            "gpu_sm_util_pct": 62 + min(active, 8) * 4 if active else 0,
            "gpu_mem_bw_util_pct": 35 + min(active, 8) * 3 if active else 0,
        }

    def get_orca_replica_metrics(
        self, session_id: str, replica_id: str, now: Optional[float] = None
    ) -> dict[str, Any]:
        session = self.snapshot(session_id, now=now)
        replica = next(
            (
                item
                for item in session["runtime"]["replicas"]
                if item["replica_id"] == replica_id
            ),
            None,
        )
        if not replica:
            raise KeyError(replica_id)
        if replica["phase"] != "running":
            return {
                "avg_generation_throughput_toks_per_s": 0.0,
                "gpu_cache_usage_perc": 0.0,
                "num_requests_running": 0,
                "num_requests_waiting": 0,
                "gpu_sm_util_pct": 0.0,
                "gpu_mem_bw_util_pct": 0.0,
            }
        return {
            "avg_generation_throughput_toks_per_s": replica["tps"],
            "gpu_cache_usage_perc": 0.64,
            "num_requests_running": 8,
            "num_requests_waiting": 0,
            "gpu_sm_util_pct": 76.0,
            "gpu_mem_bw_util_pct": 48.0,
        }

    def get_chunk_progress(
        self, session_id: str, now: Optional[float] = None
    ) -> dict[str, Any]:
        session = self.snapshot(session_id, now=now)
        runtime = session["runtime"]
        job: SimJob = self.sessions[session_id]["_orca"]["job"]
        pending = max(0, job.total_chunks - job.completed_chunks - job.failed_chunks)
        return {
            "total": job.total_chunks,
            "pending": pending,
            "inflight": 0,
            "completed": job.completed_chunks,
            "failed": job.failed_chunks,
            "all_done": runtime["status"] == "completed",
        }

    def get_replicas(
        self, session_id: str, now: Optional[float] = None
    ) -> list[dict[str, Any]]:
        session = self.snapshot(session_id, now=now)
        return session["runtime"]["replicas"]

    def _base_quota_usage_locked(
        self, session: dict[str, Any]
    ) -> dict[tuple[str, str, str], int]:
        usage: dict[tuple[str, str, str], int] = {}
        for quota in session.get("resource_map", {}).get("quotas", []):
            key = (quota["family"], quota["region"], quota["market"])
            usage[key] = int(quota.get("used_vcpus", 0))
        return usage

    def _replica_quota_usage_locked(
        self, session: dict[str, Any]
    ) -> dict[tuple[str, str, str], int]:
        usage: dict[tuple[str, str, str], int] = {}
        instances_by_type = {
            instance["instance_type"]: instance
            for instance in session.get("resource_map", {}).get("instances", [])
        }
        job: SimJob = session["_orca"]["job"]
        if job.status in {"succeeded", "failed"}:
            return usage
        for replica in job.replicas.values():
            if replica.phase in TERMINAL_REPLICA_PHASES:
                continue
            instance = instances_by_type.get(replica.instance_type)
            if not instance:
                continue
            key = (instance["quota_family"], replica.region, replica.market)
            usage[key] = usage.get(key, 0) + (
                int(instance["vcpus"])
                * max(1, int(getattr(replica, "num_instances", 1)))
            )
        return usage

    def _quota_usage_locked(
        self, session: dict[str, Any]
    ) -> dict[tuple[str, str, str], int]:
        usage = self._base_quota_usage_locked(session)
        for key, used in self._replica_quota_usage_locked(session).items():
            usage[key] = usage.get(key, 0) + used
        return usage

    def _cluster_quota_usage_locked(
        self, now: float
    ) -> dict[tuple[str, str, str], int]:
        base_usage: dict[tuple[str, str, str], int] = {}
        replica_usage: dict[tuple[str, str, str], int] = {}
        for session in self.sessions.values():
            self._refresh_orca_state(session, now)
            for key, used in self._base_quota_usage_locked(session).items():
                base_usage[key] = max(base_usage.get(key, 0), int(used))
            for key, used in self._replica_quota_usage_locked(session).items():
                replica_usage[key] = replica_usage.get(key, 0) + int(used)
        combined = dict(base_usage)
        for key, used in replica_usage.items():
            combined[key] = combined.get(key, 0) + used
        return combined

    def _admit_replicas_locked(
        self,
        session: dict[str, Any],
        *,
        now: float,
        count: int,
        gpu_type: str,
        preferred_instance_type: Optional[str],
        tp: int,
        pp: int,
        preferred_region: Optional[str],
        preferred_market: Optional[str],
        base_tps: float,
        launch_timing_s: dict[str, float],
        commit: bool = True,
    ) -> dict[str, Any]:
        self._refresh_orca_state(session, now)
        gpus_per_replica = int(tp) * int(pp)
        if count <= 0 or gpus_per_replica <= 0:
            return {
                "status": "error",
                "reason": "invalid_config",
                "message": f"Invalid launch request for {gpu_type}: TP={tp}, PP={pp}, count={count}.",
            }

        instances = [
            item
            for item in session.get("resource_map", {}).get("instances", [])
            if item.get("gpu_type") == gpu_type
        ]
        if not instances:
            return {
                "status": "error",
                "reason": "unknown_gpu",
                "message": f"No instance mapping exists for GPU type {gpu_type}.",
            }

        def _instance_sort_key(instance: dict[str, Any]) -> tuple[int, int, int, str]:
            gpus_per_instance = max(1, int(instance.get("gpus_per_instance", 1)))
            instances_per_replica = max(
                1, math.ceil(gpus_per_replica / gpus_per_instance)
            )
            required_vcpus = instances_per_replica * int(instance.get("vcpus", 0))
            exact_match = (
                0
                if preferred_instance_type
                and instance.get("instance_type") == preferred_instance_type
                else 1
            )
            return (
                exact_match,
                required_vcpus,
                instances_per_replica,
                str(instance.get("instance_type", "")),
            )

        instances.sort(key=_instance_sort_key)
        usage = self._cluster_quota_usage_locked(now)
        best_rejection: Optional[dict[str, Any]] = None

        for instance in instances:
            family = instance["quota_family"]
            gpus_per_instance = max(1, int(instance.get("gpus_per_instance", 1)))
            instances_per_replica = max(
                1, math.ceil(gpus_per_replica / gpus_per_instance)
            )
            required_instances = int(count) * instances_per_replica
            required_vcpus = required_instances * int(instance.get("vcpus", 0))
            quota_rows = [
                quota
                for quota in session.get("resource_map", {}).get("quotas", [])
                if quota.get("family") == family
                and (
                    preferred_region is None or quota.get("region") == preferred_region
                )
                and (
                    preferred_market is None or quota.get("market") == preferred_market
                )
            ]
            if not quota_rows:
                best_rejection = {
                    "status": "error",
                    "reason": "no_quota_row",
                    "message": (
                        f"No quota row matched {family}"
                        + (f" in {preferred_region}" if preferred_region else "")
                        + (f" ({preferred_market})" if preferred_market else "")
                        + "."
                    ),
                }
                continue

            def _quota_sort_key(quota: dict[str, Any]) -> tuple[int, int, str]:
                key = (family, quota["region"], quota["market"])
                available = int(quota.get("baseline_vcpus", 0)) - usage.get(key, 0)
                preferred_market_rank = 0 if quota.get("market") == "on_demand" else 1
                return (-available, preferred_market_rank, str(quota.get("region", "")))

            quota_rows.sort(key=_quota_sort_key)
            for quota in quota_rows:
                key = (family, quota["region"], quota["market"])
                baseline_vcpus = int(quota.get("baseline_vcpus", 0))
                available_vcpus = max(0, baseline_vcpus - usage.get(key, 0))
                if required_vcpus > available_vcpus:
                    rejection = {
                        "status": "error",
                        "reason": "insufficient_quota",
                        "message": (
                            f"{gpu_type} TP={tp} PP={pp} x{count} needs {required_vcpus} vCPUs "
                            f"on {family} in {quota['region']}/{quota['market']}, but only "
                            f"{available_vcpus} remain."
                        ),
                        "gpu_type": gpu_type,
                        "instance_type": instance["instance_type"],
                        "required_vcpus": required_vcpus,
                        "available_vcpus": available_vcpus,
                        "region": quota["region"],
                        "market": quota["market"],
                    }
                    if best_rejection is None or available_vcpus > int(
                        best_rejection.get("available_vcpus", -1)
                    ):
                        best_rejection = rejection
                    continue

                new_replica_ids: list[str] = []
                if commit:
                    new_replica_ids = self._schedule_replicas(
                        session,
                        now=now,
                        count=count,
                        gpu_type=gpu_type,
                        instance_type=instance["instance_type"],
                        num_instances_per_replica=instances_per_replica,
                        tp=tp,
                        pp=pp,
                        region=quota["region"],
                        market=quota["market"],
                        base_tps=base_tps,
                        launch_timing_s=launch_timing_s,
                    )
                return {
                    "status": "accepted",
                    "gpu_type": gpu_type,
                    "instance_type": instance["instance_type"],
                    "tp": int(tp),
                    "pp": int(pp),
                    "region": quota["region"],
                    "market": quota["market"],
                    "required_instances": required_instances,
                    "instances_per_replica": instances_per_replica,
                    "required_vcpus": required_vcpus,
                    "available_vcpus": available_vcpus,
                    "new_replicas": new_replica_ids,
                    "launch_timing_s": deepcopy(launch_timing_s),
                }

        return best_rejection or {
            "status": "error",
            "reason": "insufficient_quota",
            "message": f"No quota-backed placement exists for {gpu_type} TP={tp} PP={pp} x{count}.",
        }

    def _schedule_replicas(
        self,
        session: dict[str, Any],
        *,
        now: float,
        count: int,
        gpu_type: str,
        instance_type: str,
        num_instances_per_replica: int,
        tp: int,
        pp: int,
        region: str,
        market: str,
        base_tps: float,
        launch_timing_s: dict[str, float],
        decision_id: Optional[str] = None,
        config_index: int = 0,
    ) -> list[str]:
        state = session["_orca"]
        job: SimJob = state["job"]
        provisioned_at = (
            now
            + float(launch_timing_s["searching_capacity"])
            + float(launch_timing_s["provisioning"])
            + float(launch_timing_s["bootstrapping"])
        )
        ready_at = now + float(launch_timing_s["total"])
        new_replica_ids: list[str] = []
        for _ in range(count):
            idx = state["next_replica_index"]
            replica_id = f"{session['session_id']}-r{idx}"
            state["next_replica_index"] += 1
            job.replicas[replica_id] = SimReplica(
                replica_id=replica_id,
                phase="launching",
                base_tps=base_tps,
                gpu_type=gpu_type,
                instance_type=instance_type,
                num_instances=max(1, int(num_instances_per_replica)),
                tp=tp,
                pp=pp,
                region=region,
                market=market,
                config_index=int(config_index),
                decision_id=decision_id,
                started_at=ready_at,
                warmup_seconds=0.0,
                wobble_pct=0.0,
            )
            state["replica_schedules"][replica_id] = {
                "launch_started_at": now,
                "provisioned_at": provisioned_at,
                "ready_at": ready_at,
            }
            new_replica_ids.append(replica_id)
        return new_replica_ids

    def scale_job(
        self,
        session_id: str,
        *,
        now: Optional[float],
        count: int,
        gpu_type: str,
        tp: int,
        pp: int,
        preferred_market: Optional[str],
        preferred_region: Optional[str],
        preferred_instance_type: Optional[str],
        base_tps: float,
        launch_timing_s: dict[str, float],
    ) -> dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            now = float(now or time.time())
            admission = self._admit_replicas_locked(
                session,
                now=now,
                count=count,
                gpu_type=gpu_type,
                preferred_instance_type=preferred_instance_type,
                tp=tp,
                pp=pp,
                preferred_region=preferred_region,
                preferred_market=preferred_market,
                base_tps=base_tps,
                launch_timing_s=launch_timing_s,
            )
            if admission["status"] != "accepted":
                return admission

            for replica_id in admission["new_replicas"]:
                session["runtime"]["events"].append(
                    {
                        "event_id": f"manual-scale-{replica_id}",
                        "at_seconds": round(now - session["created_at"], 1),
                        "action": "scale_up",
                        "label": "Scale up requested",
                        "description": (
                            f"Launching {replica_id} on {gpu_type} (TP {tp}, PP {pp}) "
                            f"in {admission['region']} ({admission['market']})."
                        ),
                        "params": {
                            "replica_id": replica_id,
                            "gpu_type": gpu_type,
                            "tp": tp,
                            "pp": pp,
                            "region": admission["region"],
                            "market": admission["market"],
                        },
                    }
                )
            return {
                "status": "scaling",
                "new_replicas": admission["new_replicas"],
                "instance_type": admission["instance_type"],
                "region": admission["region"],
                "market": admission["market"],
                "required_vcpus": admission["required_vcpus"],
                "available_vcpus": admission["available_vcpus"],
            }

    def kill_replicas(
        self,
        session_id: str,
        replica_ids: list[str],
        *,
        now: Optional[float] = None,
        reason: str = "Killed",
    ) -> list[str]:
        with self._lock:
            session = self.sessions[session_id]
            now = float(now or time.time())
            self._refresh_orca_state(session, now)
            job: SimJob = session["_orca"]["job"]
            killed: list[str] = []
            for replica_id in replica_ids:
                replica = job.replicas.get(replica_id)
                if not replica or replica.phase in TERMINAL_REPLICA_PHASES:
                    continue
                replica.phase = "killed"
                killed.append(replica_id)
                session["_bridge"]["replica_failure_reasons"][replica_id] = reason
                session["runtime"]["events"].append(
                    {
                        "event_id": f"manual-kill-{replica_id}-{int(now)}",
                        "at_seconds": round(now - session["created_at"], 1),
                        "action": "kill_replica",
                        "label": "Replica removed",
                        "description": f"{replica_id} was removed. {reason}",
                        "params": {"replica_id": replica_id, "reason": reason},
                    }
                )
            return killed

    def set_replica_tps(
        self,
        session_id: str,
        replica_id: str,
        *,
        target_tps: float,
        now: Optional[float] = None,
    ) -> float:
        session = self.sessions[session_id]
        now = float(now or time.time())
        self._refresh_orca_state(session, now)
        job: SimJob = session["_orca"]["job"]
        replica = job.replicas.get(replica_id)
        if not replica:
            raise KeyError(replica_id)
        session["_orca"]["manual_tps_overrides"][replica_id] = target_tps
        session["runtime"]["events"].append(
            {
                "event_id": f"manual-tps-{replica_id}-{int(now)}",
                "at_seconds": round(now - session["created_at"], 1),
                "action": "set_replica_tps",
                "label": "Replica TPS adjusted",
                "description": f"{replica_id} now targets {target_tps:.0f} tok/s.",
                "params": {"replica_id": replica_id, "target_tps": target_tps},
            }
        )
        return target_tps

    def aggregate_resources(self, now: Optional[float] = None) -> dict[str, Any]:
        with self._lock:
            now = float(now or time.time())
            instances_by_type: dict[str, dict[str, Any]] = {}
            quotas_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
            allocated_gpus: dict[str, int] = {}

            for session in self.sessions.values():
                self._refresh_orca_state(session, now)
                for instance in session.get("resource_map", {}).get("instances", []):
                    instances_by_type.setdefault(
                        instance["instance_type"], deepcopy(instance)
                    )
                for quota in session.get("resource_map", {}).get("quotas", []):
                    key = (quota["family"], quota["region"], quota["market"])
                    existing = quotas_by_key.get(key)
                    if (
                        existing is None
                        or quota["baseline_vcpus"] > existing["baseline_vcpus"]
                    ):
                        quotas_by_key[key] = deepcopy(quota)
                job: SimJob = session["_orca"]["job"]
                if job.status not in {"succeeded", "failed"}:
                    for replica in job.replicas.values():
                        if replica.phase in TERMINAL_REPLICA_PHASES:
                            continue
                        allocated_gpus[replica.gpu_type] = allocated_gpus.get(
                            replica.gpu_type, 0
                        ) + max(1, int(replica.tp) * int(replica.pp))

            used_vcpus_by_key = self._cluster_quota_usage_locked(now)

            quotas = []
            for key, quota in quotas_by_key.items():
                merged = deepcopy(quota)
                merged["used_vcpus"] = used_vcpus_by_key.get(
                    key, int(quota.get("used_vcpus", 0))
                )
                quotas.append(merged)

            return {
                "instances": list(instances_by_type.values()),
                "quotas": quotas,
                "allocated_gpus": allocated_gpus,
            }

    def _refresh_orca_state(self, session: dict[str, Any], now: float) -> None:
        state = session["_orca"]
        if session.get("launch_started_at") is None:
            state["last_refreshed_at"] = now
            return
        if now < state["last_refreshed_at"]:
            return
        self._apply_launch_transitions(session, now)
        self._emit_due_scenario_events(
            session, now - float(session["launch_started_at"])
        )
        self._advance_progress(session, now)
        state["last_refreshed_at"] = now

    def _emit_due_scenario_events(
        self, session: dict[str, Any], elapsed_seconds: float
    ) -> None:
        runtime = session["runtime"]
        seen = runtime["emitted_event_ids"]
        due = due_scenario_events(
            session["scenario"]["slug"],
            elapsed_seconds=elapsed_seconds,
            completed_event_ids=seen,
        )
        for event in due:
            runtime["emitted_event_ids"].append(event.event_id)
            payload = {
                "event_id": event.event_id,
                "at_seconds": event.at_seconds,
                "action": event.action,
                "label": event.label,
                "description": event.description,
                "params": dict(event.params),
            }
            runtime["events"].append(payload)
            self._apply_scenario_event(session, payload)

    def _apply_scenario_event(
        self, session: dict[str, Any], event: dict[str, Any]
    ) -> None:
        state = session["_orca"]
        job: SimJob = state["job"]
        params = event["params"]
        action = event["action"]

        if action == "degrade_replica":
            target_replica = self._oldest_running_replica(job)
            if target_replica:
                state["scenario_tps_overrides"][target_replica.replica_id] = float(
                    params.get("target_tps", target_replica.base_tps)
                )
                params["replica_id"] = target_replica.replica_id
        elif action == "restore_cluster_tps":
            state["scenario_tps_overrides"].clear()
        elif action == "kill_oldest_running":
            target_replica = self._oldest_running_replica(job)
            if target_replica:
                target_replica.phase = "dead"
                params["replica_id"] = target_replica.replica_id
                session["_bridge"]["replica_failure_reasons"][
                    target_replica.replica_id
                ] = event["description"]
        elif action == "capacity_pressure":
            state["capacity_pressure"] = float(params.get("pressure", 0.0))

    def _apply_launch_transitions(self, session: dict[str, Any], now: float) -> None:
        state = session["_orca"]
        job: SimJob = state["job"]
        any_running = False
        any_launching = False
        for replica_id, replica in job.replicas.items():
            if replica.phase in TERMINAL_REPLICA_PHASES:
                continue
            schedule = state["replica_schedules"].get(replica_id)
            if not schedule:
                if replica.phase == "running":
                    any_running = True
                continue
            if now >= schedule["ready_at"]:
                if replica.phase != "running":
                    replica.phase = "running"
                    replica.started_at = schedule["ready_at"]
                    replica.last_heartbeat = now
                any_running = True
            elif now >= schedule["provisioned_at"]:
                replica.phase = "provisioned"
                any_launching = True
            else:
                replica.phase = "launching"
                any_launching = True
        if job.status != "succeeded":
            if any_running:
                job.status = "running"
            elif any_launching:
                job.status = "launching"

    def _advance_progress(self, session: dict[str, Any], now: float) -> None:
        state = session["_orca"]
        job: SimJob = state["job"]
        if job.status == "succeeded":
            return

        progress_from = float(state["progress_updated_at"])
        ready_cutoff = float(session["launch_started_at"]) + float(
            session["launch_preview"]["launch_timing_s"]["total"]
        )
        if now <= progress_from:
            return
        effective_from = max(progress_from, ready_cutoff)
        if now <= effective_from:
            state["progress_updated_at"] = now
            return

        aggregate_tps = self._aggregate_job_tps(session, now)
        if aggregate_tps <= 0:
            state["progress_updated_at"] = now
            return
        state["last_nonzero_aggregate_tps"] = float(aggregate_tps)

        delta_seconds = now - effective_from
        carried_tokens = state["partial_tokens"] + (aggregate_tps * delta_seconds)
        chunks_this_tick = int(carried_tokens / max(job.tokens_per_chunk, 1))
        state["partial_tokens"] = carried_tokens - (
            chunks_this_tick * job.tokens_per_chunk
        )
        if chunks_this_tick > 0:
            job.completed_chunks = min(
                job.total_chunks, job.completed_chunks + chunks_this_tick
            )
        for replica in job.replicas.values():
            if replica.phase == "running":
                replica.last_heartbeat = now
        if job.completed_chunks >= job.total_chunks:
            job.completed_chunks = job.total_chunks
            job.status = "succeeded"
            for replica in job.replicas.values():
                if replica.phase not in TERMINAL_REPLICA_PHASES:
                    replica.phase = "completed"
        state["progress_updated_at"] = now

    def _aggregate_job_tps(self, session: dict[str, Any], now: float) -> float:
        job: SimJob = session["_orca"]["job"]
        return sum(
            self._replica_tps(session, replica, now)
            for replica in job.replicas.values()
        )

    def _replica_tps(
        self, session: dict[str, Any], replica: SimReplica, now: float
    ) -> float:
        if replica.phase != "running":
            return 0.0
        manual_override = session["_orca"]["manual_tps_overrides"].get(
            replica.replica_id
        )
        scenario_override = session["_orca"]["scenario_tps_overrides"].get(
            replica.replica_id
        )
        override = manual_override if manual_override is not None else scenario_override
        base_tps = float(override if override is not None else replica.base_tps)
        if replica.warmup_seconds and replica.started_at < now:
            ramp = min(
                1.0, max(0.0, (now - replica.started_at) / replica.warmup_seconds)
            )
        else:
            ramp = 1.0
        return max(0.0, round(base_tps * ramp, 1))

    def _replica_display_tps(
        self, session: dict[str, Any], replica: SimReplica, now: float
    ) -> float:
        """Return the per-replica TPS the UI should render, with light Gaussian jitter.

        Only the *display* path uses this. Progress accounting stays on
        `_replica_tps` so chunks progress deterministically and SLO headroom math
        doesn't bounce every snapshot.
        """
        deterministic = self._replica_tps(session, replica, now)
        if deterministic <= 0:
            return deterministic
        # Seed per replica per 2-second slot so identical snapshots fetched
        # close together agree, while value wobbles naturally over time.
        slot = int(now / max(TPS_NOISE_SLOT_SEC, 0.25))
        seed_material = f"{replica.replica_id}:{slot}"
        rng = random.Random(seed_material)
        noise = rng.gauss(0.0, TPS_NOISE_SIGMA)
        # Clamp so the UI never flips sign or collapses below a believable floor.
        noise = max(-0.18, min(0.18, noise))
        return max(0.0, round(deterministic * (1.0 + noise), 1))

    def _running_replicas(self, session: dict[str, Any]) -> list[SimReplica]:
        job: SimJob = session["_orca"]["job"]
        return [
            replica for replica in job.replicas.values() if replica.phase == "running"
        ]

    def _session_launch_phase(self, session: dict[str, Any], now: float) -> str:
        if session.get("launch_started_at") is None:
            return "waiting_for_koi"
        job: SimJob = session["_orca"]["job"]
        schedules = session["_orca"]["replica_schedules"]
        active_schedules = [
            schedules[replica_id]
            for replica_id, replica in job.replicas.items()
            if replica.phase not in TERMINAL_REPLICA_PHASES
            and replica.phase != "running"
            and replica_id in schedules
        ]
        if not active_schedules:
            return "running"
        next_schedule = min(active_schedules, key=lambda item: item["ready_at"])
        if now < next_schedule["provisioned_at"]:
            launch = session["launch_preview"]["launch_timing_s"]
            launch_started_at = float(session["launch_started_at"])
            searching_end = launch_started_at + float(launch["searching_capacity"])
            provisioning_end = searching_end + float(launch["provisioning"])
            bootstrapping_end = provisioning_end + float(launch["bootstrapping"])
            if now < searching_end:
                return "searching_capacity"
            if now < provisioning_end:
                return "provisioning"
            if now < bootstrapping_end:
                return "bootstrapping"
        return "waiting_model_ready"

    def _public_replica(
        self, session: dict[str, Any], replica: SimReplica, now: float
    ) -> dict[str, Any]:
        return {
            "replica_id": replica.replica_id,
            "phase": replica.phase,
            "launch_phase": self._replica_launch_phase(
                session, replica.replica_id, now
            ),
            "region": replica.region,
            "market": replica.market,
            "instance_type": replica.instance_type,
            "num_instances": max(1, int(getattr(replica, "num_instances", 1))),
            "gpu_type": replica.gpu_type,
            "decision_id": getattr(replica, "decision_id", None),
            "config_index": int(getattr(replica, "config_index", 0)),
            "tp": replica.tp,
            "pp": replica.pp,
            "has_metrics": replica.phase == "running",
            "base_tps": float(replica.base_tps),
            # UI-facing TPS with light Gaussian jitter so demo feels alive.
            "tps": self._replica_display_tps(session, replica, now),
        }

    @staticmethod
    def _oldest_running_replica(job: SimJob) -> Optional[SimReplica]:
        running = [
            replica for replica in job.replicas.values() if replica.phase == "running"
        ]
        if not running:
            return None
        return min(running, key=lambda replica: replica.started_at)

    def _replica_launch_phase(
        self, session: dict[str, Any], replica_id: str, now: float
    ) -> str:
        job: SimJob = session["_orca"]["job"]
        replica = job.replicas.get(replica_id)
        if not replica:
            return "unknown"
        if replica.phase == "running":
            return "running"
        if replica.phase == "provisioned":
            return "waiting_model_ready"
        if replica.phase in TERMINAL_REPLICA_PHASES:
            return replica.phase

        schedule = session["_orca"]["replica_schedules"].get(replica_id)
        if not schedule:
            return replica.phase
        launch_started_at = schedule.get(
            "launch_started_at", session.get("launch_started_at", session["created_at"])
        )
        total = max(schedule["ready_at"] - launch_started_at, 0.1)
        searching = total * 0.25
        provisioning = total * 0.45
        bootstrapping = total * 0.20
        searching_end = launch_started_at + searching
        provisioning_end = searching_end + provisioning
        bootstrapping_end = provisioning_end + bootstrapping
        if now < searching_end:
            return "searching_capacity"
        if now < provisioning_end:
            return "provisioning"
        if now < bootstrapping_end:
            return "bootstrapping"
        return "waiting_model_ready"
