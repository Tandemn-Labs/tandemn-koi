"""Tests for the demo backend API."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import simulation.demo_server as demo_server
from simulation.demo_server import SESSION_MANAGER, app


@pytest_asyncio.fixture
async def client():
    SESSION_MANAGER.clear()

    async def _no_koi(*args, **kwargs):
        return None

    async def _noop_post(*args, **kwargs):
        return {"status": "ok"}

    async def _empty_get(*args, **kwargs):
        if args and args[0] == "/jobs":
            return {"tracked_jobs": 0, "pending_launches": 0, "jobs": []}
        return {"pending_reservations": [], "pending_gpus": {}, "pending_count": 0}

    demo_server._request_koi_decision = _no_koi
    demo_server._post_koi = _noop_post
    demo_server._get_koi_json = _empty_get
    transport = ASGITransport(app=app, raise_app_exceptions=True)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    SESSION_MANAGER.clear()


class TestDemoCatalog:
    @pytest.mark.asyncio
    async def test_demo_index_serves_html(self, client):
        resp = await client.get("/demo")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Koi Demo Simulator" in resp.text

    @pytest.mark.asyncio
    async def test_demo_static_assets_are_served(self, client):
        resp = await client.get("/demo/static/app.js")
        assert resp.status_code == 200
        assert "application/javascript" in resp.headers["content-type"] or "text/javascript" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_catalog_endpoint_returns_controls(self, client):
        resp = await client.get("/demo/catalog")
        assert resp.status_code == 200
        body = resp.json()
        assert {"models", "quota_presets", "scenarios"} <= set(body.keys())
        assert body["models"]
        assert body["quota_presets"]
        assert body["scenarios"]


class TestDemoLaunch:
    @pytest.mark.asyncio
    async def test_launch_creates_session_with_preview(self, client):
        resp = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "total_chunks": 500,
                "slo_deadline_hours": 8.0,
                "quota_preset": "aws_mixed_demo",
                "scenario": "hero_elastic",
                "cost_cap_usd": 120.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "created"
        assert body["session_id"].startswith("demo-")
        assert body["model"]["model_name"] == "Qwen/Qwen3-32B"
        assert body["quota"]["slug"] == "aws_mixed_demo"
        assert body["scenario"]["slug"] == "hero_elastic"
        assert body["launch_preview"]["baseline_replica_tps"] > 0
        assert body["launch_preview"]["launch_timing_s"]["total"] > 0
        assert body["resource_map"]["instances"]
        assert body["koi"]["decision"] is None
        assert body["koi"]["live"] is None

        session = await client.get(f"/demo/session/{body['session_id']}")
        assert session.status_code == 200
        session_body = session.json()
        assert session_body["session_id"] == body["session_id"]
        assert session_body["koi"]["live"]["jobs"]["jobs"] == []

    @pytest.mark.asyncio
    async def test_launch_rejects_unknown_quota(self, client):
        resp = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "nope",
                "scenario": "hero_elastic",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_session_endpoint_returns_dynamic_runtime_state(self, client):
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "total_chunks": 500,
                "slo_deadline_hours": 8.0,
                "quota_preset": "aws_mixed_demo",
                "scenario": "hero_elastic",
            },
        )
        body = launch.json()
        session_id = body["session_id"]
        created_at = body["created_at"]
        launch_total = body["launch_preview"]["launch_timing_s"]["total"]

        launching = await client.get(f"/demo/session/{session_id}", params={"now": created_at + 0.5})
        assert launching.status_code == 200
        assert launching.json()["runtime"]["status"] == "launching"

        running = await client.get(
            f"/demo/session/{session_id}",
            params={"now": created_at + launch_total + 15},
        )
        assert running.status_code == 200
        running_body = running.json()
        assert running_body["runtime"]["status"] in {"running", "completed"}
        assert running_body["runtime"]["aggregate_tps"] > 0
        assert running_body["runtime"]["tokens_completed"] > 0

        after_pressure = await client.get(
            f"/demo/session/{session_id}",
            params={"now": created_at + launch_total + 25},
        )
        after_pressure_body = after_pressure.json()
        labels = {event["label"] for event in after_pressure_body["runtime"]["events"]}
        assert "Input spike" in labels

    @pytest.mark.asyncio
    async def test_launch_can_include_live_koi_decision_summary(self, client):
        async def _fake_koi(*args, **kwargs):
            return {
                "_decision_id": "dec-demo-1",
                "predicted_tps": 1875.0,
                "confidence": 0.91,
                "config": {
                    "gpu_type": "A100-80GB",
                    "instance_type": "p4de.24xlarge",
                    "tp": 8,
                    "pp": 1,
                },
            }

        demo_server._request_koi_decision = _fake_koi
        resp = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_mixed_demo",
                "scenario": "hero_elastic",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["koi"]["decision"]["_decision_id"] == "dec-demo-1"
        assert body["launch_preview"]["preferred_gpu"] == "A100-80GB"
        assert body["launch_preview"]["baseline_replica_tps"] == 1875.0
        assert body["launch_preview"]["tp"] == 8
        assert body["launch_preview"]["pp"] == 1

    @pytest.mark.asyncio
    async def test_session_endpoint_attaches_filtered_live_koi_state(self, client):
        async def _fake_koi(*args, **kwargs):
            return {
                "_decision_id": "dec-demo-2",
                "predicted_tps": 900.0,
                "config": {
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 1,
                },
            }

        async def _fake_post(*args, **kwargs):
            return {"status": "tracked"}

        async def _fake_get(path, *args, **kwargs):
            if path == "/jobs":
                return {
                    "tracked_jobs": 2,
                    "pending_launches": 1,
                    "jobs": [
                        {"job_id": "demo-other-r0", "status": "running"},
                        {"job_id": "demo-placeholder-r0", "status": "launching"},
                    ],
                }
            return {
                "pending_reservations": [{"decision_id": "dec-demo-2"}],
                "pending_gpus": {"L40S": 4},
                "pending_count": 1,
            }

        demo_server._request_koi_decision = _fake_koi
        demo_server._post_koi = _fake_post
        demo_server._get_koi_json = _fake_get

        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_mixed_demo",
                "scenario": "overprovisioned",
            },
        )
        session_id = launch.json()["session_id"]
        running = await client.get(f"/demo/session/{session_id}", params={"now": launch.json()["created_at"] + 20})
        assert running.status_code == 200
        body = running.json()
        assert body["koi"]["sync"]["status"] in {"launching_sent", "heartbeat_sent", "started_sent", "noop"}
        assert body["koi"]["live"]["resources"]["pending_count"] == 1
        assert body["koi"]["live"]["jobs"]["jobs"] == []

    @pytest.mark.asyncio
    async def test_koi_sync_sends_started_once_when_session_turns_running(self, client):
        recorded = []

        async def _fake_post(path, payload):
            recorded.append((path, payload))
            return {"status": "ok"}

        session = SESSION_MANAGER.create_session(
            {
                "session_id": "demo-test-sync",
                "created_at": 1000.0,
                "request": {
                    "model_name": "Qwen/Qwen3-32B",
                    "avg_input_tokens": 800,
                    "avg_output_tokens": 200,
                    "total_chunks": 10,
                    "slo_deadline_hours": 8.0,
                },
                "model": {"model_name": "Qwen/Qwen3-32B"},
                "scenario": {
                    "slug": "hero_elastic",
                    "title": "Hero Elastic",
                    "description": "demo",
                    "initial_replicas": 1,
                    "launch_timing_multiplier": 1.0,
                },
                "quota": {"slug": "aws_mixed_demo", "title": "AWS Mixed Demo", "cloud": "aws", "notes": ""},
                "resource_map": {"instances": [], "quotas": []},
                "koi": {
                    "configured_url": "http://localhost:8090",
                    "decision": {
                        "_decision_id": "dec-sync-1",
                        "predicted_tps": 1100.0,
                        "config": {"gpu_type": "L40S", "instance_type": "g6e.12xlarge", "tp": 4, "pp": 1},
                    },
                    "error": None,
                    "sync_error": None,
                    "live": None,
                },
                "launch_preview": {
                    "baseline_replica_tps": 1100.0,
                    "launch_timing_s": {
                        "searching_capacity": 1.0,
                        "provisioning": 1.0,
                        "bootstrapping": 1.0,
                        "waiting_model_ready": 1.0,
                        "total": 4.0,
                    },
                    "preferred_gpu": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "region": "us-east-1",
                    "market": "on_demand",
                    "tp": 4,
                    "pp": 1,
                },
                "launch_config": {
                    "job_id": "demo-test-sync-r0",
                    "group_id": "demo-test-sync",
                    "decision_id": "dec-sync-1",
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 1,
                    "dp": 1,
                    "region": "us-east-1",
                    "market": "on_demand",
                    "total_tokens": 10000,
                    "predicted_tps": 1100.0,
                },
            }
        )
        demo_server._post_koi = _fake_post

        snapshot = SESSION_MANAGER.snapshot(session["session_id"], now=1010.0)
        result = await demo_server._sync_session_with_koi(session["session_id"], snapshot)

        assert result["status"] == "started_sent"
        assert [path for path, _ in recorded] == ["/job/launching", "/job/started"]
        assert recorded[-1][1]["decision_id"] == "dec-sync-1"


class TestDemoOrcaApi:
    @pytest.mark.asyncio
    async def test_orca_endpoints_expose_job_metrics_and_replicas(self, client):
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_mixed_demo",
                "scenario": "overprovisioned",
            },
        )
        body = launch.json()
        session_id = body["session_id"]
        running_now = body["created_at"] + body["launch_preview"]["launch_timing_s"]["total"] + 15

        resources = await client.get("/demo/orca/resources", params={"now": running_now})
        assert resources.status_code == 200
        assert resources.json()["instances"]

        status = await client.get(f"/demo/orca/job/{session_id}", params={"now": running_now})
        assert status.status_code == 200
        assert status.json()["job_id"] == session_id
        assert status.json()["active_replicas"] >= 1

        metrics = await client.get(f"/demo/orca/job/{session_id}/metrics", params={"now": running_now})
        assert metrics.status_code == 200
        assert metrics.json()["avg_generation_throughput_toks_per_s"] > 0

        replicas = await client.get(f"/demo/orca/job/{session_id}/replicas", params={"now": running_now})
        assert replicas.status_code == 200
        replica_list = replicas.json()["replicas"]
        assert replica_list
        assert replica_list[0]["has_metrics"] is True

        progress = await client.get(f"/demo/orca/job/{session_id}/chunks/progress", params={"now": running_now})
        assert progress.status_code == 200
        assert progress.json()["completed"] > 0

        rid = replica_list[0]["replica_id"]
        replica_metrics = await client.get(
            f"/demo/orca/job/{session_id}/replicas/{rid}/metrics",
            params={"now": running_now},
        )
        assert replica_metrics.status_code == 200
        assert replica_metrics.json()["avg_generation_throughput_toks_per_s"] > 0

    @pytest.mark.asyncio
    async def test_orca_scale_adds_new_launching_replicas(self, client):
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_mixed_demo",
                "scenario": "hero_elastic",
            },
        )
        body = launch.json()
        session_id = body["session_id"]
        running_now = body["created_at"] + body["launch_preview"]["launch_timing_s"]["total"] + 5

        scale = await client.post(
            f"/demo/orca/job/{session_id}/scale",
            params={"now": running_now},
            json={"count": 2, "gpu_type": "A100-80GB", "tp_size": 8, "pp_size": 1, "on_demand": True},
        )
        assert scale.status_code == 200
        scale_body = scale.json()
        assert scale_body["status"] == "scaling"
        assert len(scale_body["new_replicas"]) == 2

        replicas = await client.get(f"/demo/orca/job/{session_id}/replicas", params={"now": running_now + 0.1})
        phases = {replica["replica_id"]: replica["phase"] for replica in replicas.json()["replicas"]}
        for replica_id in scale_body["new_replicas"]:
            assert phases[replica_id] == "launching"

    @pytest.mark.asyncio
    async def test_orca_kill_and_set_tps_mutate_replica_state(self, client):
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_mixed_demo",
                "scenario": "overprovisioned",
            },
        )
        body = launch.json()
        session_id = body["session_id"]
        running_now = body["created_at"] + body["launch_preview"]["launch_timing_s"]["total"] + 10

        replicas_resp = await client.get(f"/demo/orca/job/{session_id}/replicas", params={"now": running_now})
        replica_id = replicas_resp.json()["replicas"][0]["replica_id"]

        set_tps = await client.post(
            f"/demo/orca/sim/set-tps/{replica_id}",
            params={"now": running_now},
            json={"target_tps": 250.0},
        )
        assert set_tps.status_code == 200
        assert set_tps.json()["tps"] == 250.0

        replica_metrics = await client.get(
            f"/demo/orca/job/{session_id}/replicas/{replica_id}/metrics",
            params={"now": running_now + 1},
        )
        assert replica_metrics.status_code == 200
        assert replica_metrics.json()["avg_generation_throughput_toks_per_s"] == 250.0

        kill = await client.post(
            f"/demo/orca/job/{session_id}/kill",
            params={"now": running_now + 1},
            json={"replica_ids": [replica_id]},
        )
        assert kill.status_code == 200
        assert kill.json()["killed"] == [replica_id]

        replicas_after = await client.get(f"/demo/orca/job/{session_id}/replicas", params={"now": running_now + 1.1})
        target = next(replica for replica in replicas_after.json()["replicas"] if replica["replica_id"] == replica_id)
        assert target["phase"] == "killed"
