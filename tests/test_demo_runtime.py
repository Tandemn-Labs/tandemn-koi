"""Tests for demo session runtime progression."""

import pytest

from simulation.demo_runtime import DemoSessionManager
from simulation import demo_runtime as _demo_runtime


@pytest.fixture(autouse=True)
def _disable_tps_noise():
    """Run every test in this file with deterministic TPS (no display jitter)."""
    original = _demo_runtime.TPS_NOISE_SIGMA
    _demo_runtime.set_tps_noise_sigma(0.0)
    try:
        yield
    finally:
        _demo_runtime.set_tps_noise_sigma(original)


def _session_payload():
    return {
        "session_id": "demo-123",
        "created_at": 1000.0,
        "request": {
            "total_chunks": 500,
            "avg_input_tokens": 800,
            "avg_output_tokens": 200,
        },
        "scenario": {
            "slug": "kill_and_recover",
            "title": "Kill And Recover",
            "description": "demo",
            "initial_replicas": 2,
            "launch_timing_multiplier": 1.0,
        },
        "launch_preview": {
            "baseline_replica_tps": 1000.0,
            "launch_timing_s": {
                "searching_capacity": 1.0,
                "provisioning": 2.0,
                "bootstrapping": 1.0,
                "waiting_model_ready": 1.0,
                "total": 5.0,
            },
            "preferred_gpu": "L40S",
        },
    }


def _pending_session_payload():
    payload = _session_payload()
    payload["launch_started_at"] = None
    payload["koi"] = {
        "decision": None,
        "decision_status": "pending",
    }
    return payload


def _resource_map(*, used_vcpus: int = 0):
    return {
        "instances": [
            {
                "instance_type": "g6e.12xlarge",
                "gpu_type": "L40S",
                "gpus_per_instance": 4,
                "gpu_memory_gb": 48.0,
                "vcpus": 48,
                "quota_family": "G6E",
                "cost_per_instance_hour_usd": 7.35,
            }
        ],
        "quotas": [
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": used_vcpus,
            }
        ],
    }


class TestDemoRuntime:
    def test_pending_session_waits_for_koi_until_activated(self):
        mgr = DemoSessionManager()
        created = mgr.create_session(_pending_session_payload())

        assert created["runtime"]["status"] == "koi_deciding"
        assert created["runtime"]["launch_phase"] == "waiting_for_koi"
        assert created["runtime"]["replicas"] == []

        activated = mgr.activate_session(
            "demo-123",
            now=1005.0,
            launch_config={
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "region": "us-east-1",
                "market": "on_demand",
                "decision_id": "dec-demo",
                "predicted_tps": 1000.0,
                "total_tokens": 500000,
            },
            baseline_tps=1000.0,
            launch_timing_s={
                "searching_capacity": 1.0,
                "provisioning": 2.0,
                "bootstrapping": 1.0,
                "waiting_model_ready": 1.0,
                "total": 5.0,
            },
        )
        assert activated["runtime"]["status"] == "launching"
        assert len(activated["runtime"]["replicas"]) == 2

    def test_launching_phase_progresses_into_running(self):
        mgr = DemoSessionManager()
        mgr.create_session(_session_payload())

        launching = mgr.snapshot("demo-123", now=1001.0)
        assert launching["runtime"]["status"] == "launching"
        assert launching["runtime"]["launch_phase"] == "provisioning"

        running = mgr.snapshot("demo-123", now=1015.0)
        assert running["runtime"]["status"] == "running"
        assert running["runtime"]["aggregate_tps"] == 2000.0
        assert running["runtime"]["tokens_completed"] > 0
        assert running["runtime"]["slo_headroom_pct"] is not None

    def test_kill_event_reduces_active_replicas(self):
        mgr = DemoSessionManager()
        mgr.create_session(_session_payload())

        snapshot = mgr.snapshot("demo-123", now=1035.0)
        assert snapshot["runtime"]["active_replicas"] == 1
        assert snapshot["runtime"]["aggregate_tps"] == 1000.0
        assert {event["event_id"] for event in snapshot["runtime"]["events"]} == {
            "kill-primary"
        }

    def test_manual_throttle_survives_scenario_restore(self):
        mgr = DemoSessionManager()
        payload = _session_payload()
        payload["scenario"] = {
            "slug": "hero_elastic",
            "title": "Hero Elastic",
            "description": "demo",
            "initial_replicas": 1,
            "launch_timing_multiplier": 1.0,
        }
        mgr.create_session(payload)

        throttled = mgr.set_replica_tps(
            "demo-123", "demo-123-r0", target_tps=250.0, now=1090.0
        )
        assert throttled == 250.0

        snapshot = mgr.snapshot("demo-123", now=1120.0)
        replica = next(
            item
            for item in snapshot["runtime"]["replicas"]
            if item["replica_id"] == "demo-123-r0"
        )
        assert replica["tps"] == 250.0

    def test_aggregate_resources_preserves_preset_used_vcpus(self):
        mgr = DemoSessionManager()
        payload = _pending_session_payload()
        payload["resource_map"] = _resource_map(used_vcpus=48)
        mgr.create_session(payload)

        resources = mgr.aggregate_resources(now=1000.0)
        quota = next(item for item in resources["quotas"] if item["family"] == "G6E")
        assert quota["used_vcpus"] == 48

    def test_completed_job_releases_reserved_quota(self):
        mgr = DemoSessionManager()
        payload = _session_payload()
        payload["request"]["total_chunks"] = 1
        payload["resource_map"] = _resource_map()
        mgr.create_session(payload)

        during_launch = mgr.aggregate_resources(now=1001.0)
        quota_during = next(
            item for item in during_launch["quotas"] if item["family"] == "G6E"
        )
        assert quota_during["used_vcpus"] == 96

        after_completion = mgr.aggregate_resources(now=1100.0)
        quota_after = next(
            item for item in after_completion["quotas"] if item["family"] == "G6E"
        )
        assert quota_after["used_vcpus"] == 0

        snapshot = mgr.snapshot("demo-123", now=1100.0)
        assert snapshot["runtime"]["status"] == "completed"
        assert all(
            replica["phase"] in {"dead", "killed", "completed"}
            for replica in snapshot["runtime"]["replicas"]
        )
        assert any(
            replica["phase"] == "completed"
            for replica in snapshot["runtime"]["replicas"]
        )

    def test_killing_all_replicas_sets_negative_headroom(self):
        mgr = DemoSessionManager()
        mgr.create_session(_session_payload())

        running = mgr.snapshot("demo-123", now=1015.0)
        replica_ids = [
            replica["replica_id"] for replica in running["runtime"]["replicas"]
        ]
        assert replica_ids

        killed = mgr.kill_replicas("demo-123", replica_ids, now=1016.0)
        assert set(killed) == set(replica_ids)

        snapshot = mgr.snapshot("demo-123", now=1017.0)
        assert snapshot["runtime"]["active_replicas"] == 0
        assert snapshot["runtime"]["aggregate_tps"] == 0.0
        assert snapshot["runtime"]["status"] == "stalled"
        assert snapshot["runtime"]["slo_headroom_pct"] == -100.0
