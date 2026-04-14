"""Tests for demo session runtime progression."""

from simulation.demo_runtime import DemoSessionManager


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


class TestDemoRuntime:
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
        assert {event["event_id"] for event in snapshot["runtime"]["events"]} == {"kill-primary"}
