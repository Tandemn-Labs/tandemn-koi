"""Runtime state and streaming snapshots for demo sessions."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Optional

from simulation.demo_scenarios import due_scenario_events
from simulation.sim_engine import SimJob, SimReplica

TERMINAL_REPLICA_PHASES = {"dead", "killed", "failed", "completed"}


class DemoSessionManager:
    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}

    def clear(self):
        self.sessions.clear()

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = deepcopy(payload)
        session_id = session["session_id"]
        created_at = float(session["created_at"])
        model_name = ((session.get("model") or {}).get("model_name")) or session.get("request", {}).get("model_name") or "unknown"
        request = session["request"]
        launch_config = session.get("launch_config") or {
            "gpu_type": session.get("launch_preview", {}).get("preferred_gpu", "L40S"),
            "instance_type": session.get("launch_preview", {}).get("instance_type", "g6e.12xlarge"),
            "tp": session.get("launch_preview", {}).get("tp", 4),
            "pp": session.get("launch_preview", {}).get("pp", 1),
            "dp": 1,
            "region": session.get("launch_preview", {}).get("region", "us-east-1"),
            "market": session.get("launch_preview", {}).get("market", "on_demand"),
            "decision_id": ((session.get("koi") or {}).get("decision") or {}).get("_decision_id"),
            "predicted_tps": float(session.get("launch_preview", {}).get("baseline_replica_tps", 0.0)),
            "total_tokens": 0,
        }
        session["launch_config"] = launch_config
        total_tokens = session["request"]["total_chunks"] * (
            session["request"]["avg_input_tokens"] + session["request"]["avg_output_tokens"]
        )
        launch_config["total_tokens"] = int(total_tokens)
        tokens_per_chunk = max(
            session["request"]["avg_input_tokens"] + session["request"]["avg_output_tokens"],
            1,
        )

        session["_bridge"] = {
            "group_id": session_id,
            "launching_sent": set(),
            "started_sent": set(),
            "last_heartbeat_phase": {},
            "last_heartbeat_at": {},
        }

        job = SimJob(
            job_id=session_id,
            model_name=model_name,
            total_chunks=request["total_chunks"],
            status="launching",
            slo_deadline_hours=request.get("slo_deadline_hours", 8.0),
            decision_id=((session.get("koi") or {}).get("decision") or {}).get("_decision_id"),
            tokens_per_chunk=tokens_per_chunk,
            deploy_timestamp=created_at,
        )

        baseline_tps = float(session["launch_preview"]["baseline_replica_tps"])
        launch_timing = session["launch_preview"]["launch_timing_s"]
        provisioned_at = (
            created_at
            + float(launch_timing["searching_capacity"])
            + float(launch_timing["provisioning"])
            + float(launch_timing["bootstrapping"])
        )
        ready_at = created_at + float(launch_timing["total"])

        replica_schedules: dict[str, dict[str, float]] = {}
        for index in range(session["scenario"]["initial_replicas"]):
            replica_id = f"{session_id}-r{index}"
            job.replicas[replica_id] = SimReplica(
                replica_id=replica_id,
                phase="launching",
                base_tps=baseline_tps,
                gpu_type=launch_config["gpu_type"],
                instance_type=launch_config["instance_type"],
                tp=launch_config["tp"],
                pp=launch_config["pp"],
                region=launch_config["region"],
                market=launch_config["market"],
                started_at=ready_at,
                warmup_seconds=0.0,
                wobble_pct=0.0,
            )
            replica_schedules[replica_id] = {
                "launch_started_at": created_at,
                "provisioned_at": provisioned_at,
                "ready_at": ready_at,
            }

        session["_orca"] = {
            "job": job,
            "replica_schedules": replica_schedules,
            "tps_overrides": {},
            "next_replica_index": len(job.replicas),
            "last_refreshed_at": created_at,
            "progress_updated_at": created_at,
            "partial_tokens": 0.0,
            "last_manual_event_at": created_at,
        }
        session["runtime"] = {
            "active_replicas": 0,
            "baseline_replica_tps": baseline_tps,
            "events": [],
            "emitted_event_ids": [],
            "status": "launching",
            "launch_phase": "searching_capacity",
            "aggregate_tps": 0.0,
            "progress_pct": 0.0,
            "tokens_completed": 0,
            "tokens_total": total_tokens,
            "eta_seconds": None,
            "slo_headroom_pct": None,
            "replicas": [],
        }
        self.sessions[session_id] = session
        return self.snapshot(session_id, now=created_at)

    def snapshot(self, session_id: str, now: Optional[float] = None) -> dict[str, Any]:
        session = self.sessions[session_id]
        now = float(now or time.time())
        self._refresh_orca_state(session, now)

        runtime = session["runtime"]
        job: SimJob = session["_orca"]["job"]
        aggregate_tps = self._aggregate_job_tps(session, now)
        running_replicas = self._running_replicas(session)
        launching_phase = self._session_launch_phase(session, now)

        runtime["aggregate_tps"] = round(aggregate_tps, 1)
        runtime["active_replicas"] = len(running_replicas)
        runtime["launch_phase"] = launching_phase
        runtime["tokens_completed"] = min(
            runtime["tokens_total"],
            job.completed_chunks * max(job.tokens_per_chunk, 1),
        )
        runtime["progress_pct"] = round(
            (runtime["tokens_completed"] / max(runtime["tokens_total"], 1)) * 100, 2
        )
        remaining_tokens = max(0, runtime["tokens_total"] - runtime["tokens_completed"])
        if job.status == "succeeded" or remaining_tokens == 0:
            runtime["status"] = "completed"
            runtime["eta_seconds"] = 0.0
        elif running_replicas:
            runtime["status"] = "running"
            runtime["eta_seconds"] = round(remaining_tokens / max(aggregate_tps, 1), 1)
        else:
            runtime["status"] = "launching"
            ready_times = [
                schedule["ready_at"]
                for replica_id, schedule in session["_orca"]["replica_schedules"].items()
                if job.replicas.get(replica_id) and job.replicas[replica_id].phase not in TERMINAL_REPLICA_PHASES
            ]
            runtime["eta_seconds"] = round(max(0.0, min(ready_times or [now]) - now), 1)

        deadline_seconds = float(session["request"].get("slo_deadline_hours", 8.0)) * 3600.0
        if runtime["eta_seconds"] is None or deadline_seconds <= 0:
            runtime["slo_headroom_pct"] = None
        else:
            projected_completion_seconds = (now - session["created_at"]) + float(runtime["eta_seconds"])
            runtime["slo_headroom_pct"] = round(
                ((deadline_seconds - projected_completion_seconds) / deadline_seconds) * 100.0,
                1,
            )

        runtime["replicas"] = [
            self._public_replica(session, replica, now)
            for replica in job.replicas.values()
        ]

        snapshot = deepcopy(session)
        for key in list(snapshot.keys()):
            if key.startswith("_"):
                snapshot.pop(key)
        snapshot["runtime"] = deepcopy(runtime)
        snapshot["runtime"]["elapsed_seconds"] = round(now - session["created_at"], 1)
        return snapshot

    def get_orca_job_status(self, session_id: str, now: Optional[float] = None) -> dict[str, Any]:
        session = self.snapshot(session_id, now=now)
        runtime = session["runtime"]
        return {
            "job_id": session_id,
            "status": runtime["status"],
            "model_name": session["model"]["model_name"],
            "num_replicas": len(runtime["replicas"]),
            "active_replicas": runtime["active_replicas"],
        }

    def get_orca_job_metrics(self, session_id: str, now: Optional[float] = None) -> dict[str, Any]:
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

    def get_orca_replica_metrics(self, session_id: str, replica_id: str, now: Optional[float] = None) -> dict[str, Any]:
        session = self.snapshot(session_id, now=now)
        replica = next(
            (item for item in session["runtime"]["replicas"] if item["replica_id"] == replica_id),
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

    def get_chunk_progress(self, session_id: str, now: Optional[float] = None) -> dict[str, Any]:
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

    def get_replicas(self, session_id: str, now: Optional[float] = None) -> list[dict[str, Any]]:
        session = self.snapshot(session_id, now=now)
        return session["runtime"]["replicas"]

    def scale_job(
        self,
        session_id: str,
        *,
        now: Optional[float],
        count: int,
        gpu_type: str,
        instance_type: str,
        tp: int,
        pp: int,
        region: str,
        market: str,
        base_tps: float,
        launch_timing_s: dict[str, float],
    ) -> list[str]:
        session = self.sessions[session_id]
        now = float(now or time.time())
        self._refresh_orca_state(session, now)
        state = session["_orca"]
        job: SimJob = state["job"]
        new_replica_ids: list[str] = []
        provisioned_at = (
            now
            + float(launch_timing_s["searching_capacity"])
            + float(launch_timing_s["provisioning"])
            + float(launch_timing_s["bootstrapping"])
        )
        ready_at = now + float(launch_timing_s["total"])
        for _ in range(count):
            idx = state["next_replica_index"]
            replica_id = f"{session_id}-r{idx}"
            state["next_replica_index"] += 1
            job.replicas[replica_id] = SimReplica(
                replica_id=replica_id,
                phase="launching",
                base_tps=base_tps,
                gpu_type=gpu_type,
                instance_type=instance_type,
                tp=tp,
                pp=pp,
                region=region,
                market=market,
                started_at=ready_at,
                warmup_seconds=0.0,
                wobble_pct=0.0,
            )
            state["replica_schedules"][replica_id] = {
                "launch_started_at": now,
                "provisioned_at": provisioned_at,
                "ready_at": ready_at,
            }
            session["runtime"]["events"].append(
                {
                    "event_id": f"manual-scale-{replica_id}",
                    "at_seconds": round(now - session["created_at"], 1),
                    "action": "scale_up",
                    "label": "Scale up requested",
                    "description": f"Launching {replica_id} on {gpu_type} (TP {tp}, PP {pp}).",
                    "params": {"replica_id": replica_id, "gpu_type": gpu_type, "tp": tp, "pp": pp},
                }
            )
            new_replica_ids.append(replica_id)
        return new_replica_ids

    def kill_replicas(self, session_id: str, replica_ids: list[str], *, now: Optional[float] = None, reason: str = "Killed") -> list[str]:
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

    def set_replica_tps(self, session_id: str, replica_id: str, *, target_tps: float, now: Optional[float] = None) -> float:
        session = self.sessions[session_id]
        now = float(now or time.time())
        self._refresh_orca_state(session, now)
        job: SimJob = session["_orca"]["job"]
        replica = job.replicas.get(replica_id)
        if not replica:
            raise KeyError(replica_id)
        session["_orca"]["tps_overrides"][replica_id] = target_tps
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
        now = float(now or time.time())
        instances_by_type: dict[str, dict[str, Any]] = {}
        quotas_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        used_vcpus_by_key: dict[tuple[str, str, str], int] = {}

        for session_id, session in self.sessions.items():
            self._refresh_orca_state(session, now)
            for instance in session["resource_map"]["instances"]:
                instances_by_type.setdefault(instance["instance_type"], deepcopy(instance))
            for quota in session["resource_map"]["quotas"]:
                key = (quota["family"], quota["region"], quota["market"])
                existing = quotas_by_key.get(key)
                if existing is None or quota["baseline_vcpus"] > existing["baseline_vcpus"]:
                    quotas_by_key[key] = deepcopy(quota)
                used_vcpus_by_key.setdefault(key, 0)

            job: SimJob = session["_orca"]["job"]
            for replica in job.replicas.values():
                if replica.phase in TERMINAL_REPLICA_PHASES:
                    continue
                instance = instances_by_type.get(replica.instance_type)
                if not instance:
                    continue
                key = (instance["quota_family"], replica.region, replica.market)
                used_vcpus_by_key[key] = used_vcpus_by_key.get(key, 0) + int(instance["vcpus"])

        quotas = []
        for key, quota in quotas_by_key.items():
            merged = deepcopy(quota)
            merged["used_vcpus"] = used_vcpus_by_key.get(key, quota.get("used_vcpus", 0))
            quotas.append(merged)

        return {
            "instances": list(instances_by_type.values()),
            "quotas": quotas,
        }

    def _refresh_orca_state(self, session: dict[str, Any], now: float) -> None:
        state = session["_orca"]
        if now < state["last_refreshed_at"]:
            return
        self._apply_launch_transitions(session, now)
        self._emit_due_scenario_events(session, now - session["created_at"])
        self._advance_progress(session, now)
        state["last_refreshed_at"] = now

    def _emit_due_scenario_events(self, session: dict[str, Any], elapsed_seconds: float) -> None:
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

    def _apply_scenario_event(self, session: dict[str, Any], event: dict[str, Any]) -> None:
        state = session["_orca"]
        job: SimJob = state["job"]
        params = event["params"]
        action = event["action"]

        if action == "degrade_replica":
            target_replica = self._oldest_running_replica(job)
            if target_replica:
                state["tps_overrides"][target_replica.replica_id] = float(params.get("target_tps", target_replica.base_tps))
                params["replica_id"] = target_replica.replica_id
        elif action == "restore_cluster_tps":
            state["tps_overrides"].clear()
        elif action == "kill_oldest_running":
            target_replica = self._oldest_running_replica(job)
            if target_replica:
                target_replica.phase = "dead"
                params["replica_id"] = target_replica.replica_id
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
        ready_cutoff = session["created_at"] + float(session["launch_preview"]["launch_timing_s"]["total"])
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

        delta_seconds = now - effective_from
        carried_tokens = state["partial_tokens"] + (aggregate_tps * delta_seconds)
        chunks_this_tick = int(carried_tokens / max(job.tokens_per_chunk, 1))
        state["partial_tokens"] = carried_tokens - (chunks_this_tick * job.tokens_per_chunk)
        if chunks_this_tick > 0:
            job.completed_chunks = min(job.total_chunks, job.completed_chunks + chunks_this_tick)
        for replica in job.replicas.values():
            if replica.phase == "running":
                replica.last_heartbeat = now
        if job.completed_chunks >= job.total_chunks:
            job.completed_chunks = job.total_chunks
            job.status = "succeeded"
        state["progress_updated_at"] = now

    def _aggregate_job_tps(self, session: dict[str, Any], now: float) -> float:
        job: SimJob = session["_orca"]["job"]
        return sum(self._replica_tps(session, replica, now) for replica in job.replicas.values())

    def _replica_tps(self, session: dict[str, Any], replica: SimReplica, now: float) -> float:
        if replica.phase != "running":
            return 0.0
        override = session["_orca"]["tps_overrides"].get(replica.replica_id)
        base_tps = float(override if override is not None else replica.base_tps)
        if replica.warmup_seconds and replica.started_at < now:
            ramp = min(1.0, max(0.0, (now - replica.started_at) / replica.warmup_seconds))
        else:
            ramp = 1.0
        return max(0.0, round(base_tps * ramp, 1))

    def _running_replicas(self, session: dict[str, Any]) -> list[SimReplica]:
        job: SimJob = session["_orca"]["job"]
        return [replica for replica in job.replicas.values() if replica.phase == "running"]

    def _session_launch_phase(self, session: dict[str, Any], now: float) -> str:
        job: SimJob = session["_orca"]["job"]
        schedules = session["_orca"]["replica_schedules"]
        active_schedules = [
            schedules[replica_id]
            for replica_id, replica in job.replicas.items()
            if replica.phase not in TERMINAL_REPLICA_PHASES and replica.phase != "running" and replica_id in schedules
        ]
        if not active_schedules:
            return "running"
        next_schedule = min(active_schedules, key=lambda item: item["ready_at"])
        if now < next_schedule["provisioned_at"]:
            launch = session["launch_preview"]["launch_timing_s"]
            created_at = session["created_at"]
            searching_end = created_at + float(launch["searching_capacity"])
            provisioning_end = searching_end + float(launch["provisioning"])
            bootstrapping_end = provisioning_end + float(launch["bootstrapping"])
            if now < searching_end:
                return "searching_capacity"
            if now < provisioning_end:
                return "provisioning"
            if now < bootstrapping_end:
                return "bootstrapping"
        return "waiting_model_ready"

    def _public_replica(self, session: dict[str, Any], replica: SimReplica, now: float) -> dict[str, Any]:
        return {
            "replica_id": replica.replica_id,
            "phase": replica.phase,
            "launch_phase": self._replica_launch_phase(session, replica.replica_id, now),
            "region": replica.region,
            "market": replica.market,
            "instance_type": replica.instance_type,
            "gpu_type": replica.gpu_type,
            "tp": replica.tp,
            "pp": replica.pp,
            "has_metrics": replica.phase == "running",
            "tps": self._replica_tps(session, replica, now),
        }

    @staticmethod
    def _oldest_running_replica(job: SimJob) -> Optional[SimReplica]:
        running = [replica for replica in job.replicas.values() if replica.phase == "running"]
        if not running:
            return None
        return min(running, key=lambda replica: replica.started_at)

    def _replica_launch_phase(self, session: dict[str, Any], replica_id: str, now: float) -> str:
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
        launch_started_at = schedule.get("launch_started_at", session["created_at"])
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
