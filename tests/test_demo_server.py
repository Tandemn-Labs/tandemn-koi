"""Tests for the demo backend API."""

import asyncio
import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import simulation.demo_server as demo_server
from simulation import demo_runtime as _demo_runtime
from simulation.demo_server import SESSION_MANAGER, app


@pytest.fixture(autouse=True)
def _disable_tps_noise():
    """Keep TPS deterministic across assertions by disabling display jitter."""
    original = _demo_runtime.TPS_NOISE_SIGMA
    _demo_runtime.set_tps_noise_sigma(0.0)
    try:
        yield
    finally:
        _demo_runtime.set_tps_noise_sigma(original)


@pytest_asyncio.fixture
async def client():
    SESSION_MANAGER.clear()
    old_request = demo_server._request_koi_decision
    old_post = demo_server._post_koi
    old_get = demo_server._get_koi_json

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
    pending_tasks = list(demo_server.LAUNCH_TASKS)
    for task in pending_tasks:
        task.cancel()
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)
    demo_server.LAUNCH_TASKS.clear()
    demo_server._request_koi_decision = old_request
    demo_server._post_koi = old_post
    demo_server._get_koi_json = old_get
    SESSION_MANAGER.clear()


async def _wait_for_session_activation(
    client: AsyncClient, session_id: str, attempts: int = 20
):
    for _ in range(attempts):
        resp = await client.get(f"/demo/session/{session_id}")
        body = resp.json()
        if body["runtime"]["status"] != "koi_deciding":
            return body
        await asyncio.sleep(0.01)
    return body


class TestDemoCatalog:
    @pytest.mark.asyncio
    async def test_demo_index_serves_html(self, client):
        resp = await client.get("/demo")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Koi Demo Simulator" in resp.text
        assert 'id="manual-replica-count"' in resp.text
        assert "Live Event Tap" not in resp.text
        assert "Koi Reasoning" in resp.text

    @pytest.mark.asyncio
    async def test_demo_static_assets_are_served(self, client):
        resp = await client.get("/demo/static/app.js")
        assert resp.status_code == 200
        assert (
            "application/javascript" in resp.headers["content-type"]
            or "text/javascript" in resp.headers["content-type"]
        )

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
    async def test_request_koi_decision_uses_extended_timeout(self):
        captured = {}

        async def _fake_post(path, payload, timeout=20.0):
            captured["path"] = path
            captured["timeout"] = timeout
            captured["payload"] = payload
            return {"config": {"gpu_type": "L40S"}}

        old_post = demo_server._post_koi
        demo_server._post_koi = _fake_post
        try:
            req = demo_server.DemoLaunchRequest(
                model_name="Qwen/Qwen3-32B",
                avg_input_tokens=800,
                avg_output_tokens=200,
                total_chunks=500,
                slo_deadline_hours=8.0,
                quota_preset="aws_mixed_demo",
                scenario="hero_elastic",
            )
            result = await demo_server._request_koi_decision(
                "demo-timeout-check",
                req,
                {"instances": [], "quotas": []},
            )
        finally:
            demo_server._post_koi = old_post

        assert result["config"]["gpu_type"] == "L40S"
        assert captured["path"] == "/decide"
        assert captured["timeout"] == 180.0
        assert captured["payload"]["job_request"]["job_id"] == "demo-timeout-check"
        assert captured["payload"]["job_request"]["num_requests"] == 500

    @pytest.mark.asyncio
    async def test_launch_returns_before_background_decision_resolves(self, client):
        async def _slow_koi(*args, **kwargs):
            await asyncio.sleep(0.05)
            return {
                "_decision_id": "dec-async-1",
                "predicted_tps": 1400.0,
                "config": {
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 1,
                },
            }

        demo_server._request_koi_decision = _slow_koi
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
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["runtime"]["status"] == "koi_deciding"
        assert body["koi"]["decision_status"] == "pending"
        assert body["koi"]["decision"] is None

        await asyncio.sleep(0.08)
        activated = await _wait_for_session_activation(client, body["session_id"])
        assert activated["runtime"]["status"] in {"launching", "running", "completed"}
        assert activated["koi"]["decision"]["_decision_id"] == "dec-async-1"

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
        assert body["koi"]["decision_status"] == "pending"
        assert body["koi"]["sync"]["status"] == "decision_pending"
        assert body["runtime"]["status"] == "koi_deciding"
        assert body["koi"]["live"]["jobs"]["jobs"] == []

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
        await _wait_for_session_activation(client, session_id)

        launching = await client.get(
            f"/demo/session/{session_id}", params={"now": created_at + 0.5}
        )
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
        launch_body = resp.json()
        assert launch_body["koi"]["decision"] is None
        await asyncio.sleep(0.01)
        body = await _wait_for_session_activation(client, launch_body["session_id"])
        assert body["koi"]["decision"]["_decision_id"] == "dec-demo-1"
        assert body["launch_preview"]["preferred_gpu"] == "A100-80GB"
        assert body["launch_preview"]["baseline_replica_tps"] == 1875.0
        assert body["launch_preview"]["tp"] == 8
        assert body["launch_preview"]["pp"] == 1

    @pytest.mark.asyncio
    async def test_launch_sends_live_orca_resource_usage_to_koi(self, client):
        base_now = demo_server.time.time()
        resource_map = demo_server.quota_preset_to_resource_map("aws_a100_tight")
        SESSION_MANAGER.create_session(
            {
                "session_id": "demo-existing-usage",
                "created_at": base_now - 2.0,
                "launch_started_at": base_now - 1.0,
                "request": {
                    "model_name": "Qwen/Qwen3-32B",
                    "avg_input_tokens": 800,
                    "avg_output_tokens": 200,
                    "total_chunks": 500,
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
                "quota": {
                    "slug": "aws_a100_tight",
                    "title": "AWS A100 Tight",
                    "cloud": "aws",
                    "notes": "",
                },
                "resource_map": resource_map,
                "koi": {
                    "configured_url": "http://localhost:8090",
                    "decision": {
                        "_decision_id": "dec-existing",
                        "predicted_tps": 1100.0,
                        "config": {
                            "gpu_type": "L40S",
                            "instance_type": "g6e.12xlarge",
                            "tp": 4,
                            "pp": 1,
                        },
                    },
                    "decision_status": "ready",
                    "error": None,
                    "sync_error": None,
                    "live": None,
                },
                "launch_preview": {
                    "baseline_replica_tps": 1100.0,
                    "launch_timing_s": {
                        "searching_capacity": 0.0,
                        "provisioning": 0.0,
                        "bootstrapping": 0.0,
                        "waiting_model_ready": 0.0,
                        "total": 0.0,
                    },
                    "preferred_gpu": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "region": "us-east-1",
                    "market": "on_demand",
                    "tp": 4,
                    "pp": 1,
                },
                "launch_config": {
                    "job_id": "demo-existing-usage-r0",
                    "group_id": "demo-existing-usage",
                    "decision_id": "dec-existing",
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 1,
                    "dp": 1,
                    "region": "us-east-1",
                    "market": "on_demand",
                    "total_tokens": 500000,
                    "predicted_tps": 1100.0,
                },
            }
        )

        captured = {}

        async def _fake_koi(session_id, req, resource_map):
            captured["session_id"] = session_id
            captured["resource_map"] = resource_map
            return {
                "_decision_id": "dec-live-1",
                "predicted_tps": 900.0,
                "config": {
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 1,
                },
            }

        demo_server._request_koi_decision = _fake_koi
        SESSION_MANAGER.snapshot("demo-existing-usage", now=base_now)

        resp = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "total_chunks": 500,
                "slo_deadline_hours": 8.0,
                "quota_preset": "aws_a100_tight",
                "scenario": "hero_elastic",
            },
        )
        assert resp.status_code == 200
        await asyncio.sleep(0.02)

        g6e_quota = next(
            quota
            for quota in captured["resource_map"]["quotas"]
            if quota["family"] == "G6E"
            and quota["region"] == "us-east-1"
            and quota["market"] == "on_demand"
        )
        assert g6e_quota["used_vcpus"] == 48

    @pytest.mark.asyncio
    async def test_launch_retries_koi_alternative_when_primary_exceeds_quota(
        self, client
    ):
        async def _fake_koi(*args, **kwargs):
            return {
                "_decision_id": "dec-alt-1",
                "predicted_tps": 1800.0,
                "config": {
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 4,
                },
                "alternatives": [
                    {
                        "gpu_type": "A10G",
                        "instance_type": "g5.12xlarge",
                        "tp": 4,
                        "pp": 1,
                        "predicted_tps": 850.0,
                    }
                ],
            }

        demo_server._request_koi_decision = _fake_koi
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_a100_tight",
                "scenario": "hero_elastic",
            },
        )
        assert launch.status_code == 200
        launch_body = launch.json()
        await asyncio.sleep(0.01)
        body = await _wait_for_session_activation(client, launch_body["session_id"])
        assert body["runtime"]["status"] in {"launching", "running", "completed"}
        assert body["launch_preview"]["preferred_gpu"] == "A10G"
        assert body["launch_preview"]["instance_type"] == "g5.12xlarge"
        assert body["launch_config"]["is_fallback"] is True
        assert body["launch_config"]["decision_id"] == "dec-alt-1"

    @pytest.mark.asyncio
    async def test_alternative_launch_marks_started_payload_as_fallback(self, client):
        recorded = []

        async def _fake_koi(*args, **kwargs):
            return {
                "_decision_id": "dec-alt-2",
                "predicted_tps": 1800.0,
                "config": {
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 4,
                },
                "alternatives": [
                    {
                        "gpu_type": "A10G",
                        "instance_type": "g5.12xlarge",
                        "tp": 4,
                        "pp": 1,
                        "predicted_tps": 825.0,
                    }
                ],
            }

        async def _fake_post(path, payload):
            recorded.append((path, payload))
            return {"status": "ok"}

        demo_server._request_koi_decision = _fake_koi
        demo_server._post_koi = _fake_post

        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_a100_tight",
                "scenario": "hero_elastic",
            },
        )
        assert launch.status_code == 200
        launch_body = launch.json()
        await asyncio.sleep(0.01)
        body = await _wait_for_session_activation(client, launch_body["session_id"])
        running = await client.get(
            f"/demo/session/{launch_body['session_id']}",
            params={
                "now": launch_body["created_at"]
                + body["launch_preview"]["launch_timing_s"]["total"]
                + 10
            },
        )
        assert running.status_code == 200
        started_payloads = [
            payload for path, payload in recorded if path == "/job/started"
        ]
        assert started_payloads
        assert started_payloads[0]["is_fallback"] is True
        assert started_payloads[0]["decision_id"] == "dec-alt-2"
        assert started_payloads[0]["gpu_type"] == "A10G"

    def test_read_session_koi_events_filters_by_session(self, tmp_path):
        path = tmp_path / "koi-events.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"event": "agent_deciding", "job_id": "demo-a"}),
                    json.dumps({"event": "tool_call", "job_id": "demo-a-r0"}),
                    json.dumps({"event": "tool_call", "job_id": "demo-b"}),
                    json.dumps({"event": "job_launching", "group_id": "demo-a"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        old_path = demo_server.DEMO_KOI_EVENT_LOG
        demo_server.DEMO_KOI_EVENT_LOG = str(path)
        try:
            events = demo_server._read_session_koi_events("demo-a")
        finally:
            demo_server.DEMO_KOI_EVENT_LOG = old_path
        assert [event["event"] for event in events] == [
            "agent_deciding",
            "tool_call",
            "job_launching",
        ]

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
        launch_body = launch.json()
        session_id = launch_body["session_id"]
        await asyncio.sleep(0.01)
        await _wait_for_session_activation(client, session_id)
        running = await client.get(
            f"/demo/session/{session_id}",
            params={"now": launch_body["created_at"] + 20},
        )
        assert running.status_code == 200
        body = running.json()
        assert body["koi"]["sync"]["status"] in {
            "launching_sent",
            "heartbeat_sent",
            "started_sent",
            "noop",
        }
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
                "quota": {
                    "slug": "aws_mixed_demo",
                    "title": "AWS Mixed Demo",
                    "cloud": "aws",
                    "notes": "",
                },
                "resource_map": {"instances": [], "quotas": []},
                "koi": {
                    "configured_url": "http://localhost:8090",
                    "decision": {
                        "_decision_id": "dec-sync-1",
                        "predicted_tps": 1100.0,
                        "config": {
                            "gpu_type": "L40S",
                            "instance_type": "g6e.12xlarge",
                            "tp": 4,
                            "pp": 1,
                        },
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
        result = await demo_server._sync_session_with_koi(
            session["session_id"], snapshot
        )

        assert result["status"] == "started_sent"
        assert [path for path, _ in recorded] == [
            "/job/config-attempted",
            "/job/launching",
            "/job/started",
        ]
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
        await _wait_for_session_activation(client, session_id)
        running_now = (
            body["created_at"] + body["launch_preview"]["launch_timing_s"]["total"] + 15
        )

        resources = await client.get(
            "/demo/orca/resources", params={"now": running_now}
        )
        assert resources.status_code == 200
        assert resources.json()["instances"]

        status = await client.get(
            f"/demo/orca/job/{session_id}", params={"now": running_now}
        )
        assert status.status_code == 200
        assert status.json()["job_id"] == session_id
        assert status.json()["active_replicas"] >= 1

        metrics = await client.get(
            f"/demo/orca/job/{session_id}/metrics", params={"now": running_now}
        )
        assert metrics.status_code == 200
        assert metrics.json()["avg_generation_throughput_toks_per_s"] > 0

        replicas = await client.get(
            f"/demo/orca/job/{session_id}/replicas", params={"now": running_now}
        )
        assert replicas.status_code == 200
        replica_list = replicas.json()["replicas"]
        assert replica_list
        assert replica_list[0]["has_metrics"] is True

        progress = await client.get(
            f"/demo/orca/job/{session_id}/chunks/progress", params={"now": running_now}
        )
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
        await _wait_for_session_activation(client, session_id)
        running_now = (
            body["created_at"] + body["launch_preview"]["launch_timing_s"]["total"] + 5
        )

        scale = await client.post(
            f"/demo/orca/job/{session_id}/scale",
            params={"now": running_now},
            json={
                "count": 2,
                "gpu_type": "L40S",
                "tp_size": 4,
                "pp_size": 1,
                "on_demand": True,
            },
        )
        assert scale.status_code == 200
        scale_body = scale.json()
        assert scale_body["status"] == "scaling"
        assert len(scale_body["new_replicas"]) == 2

        replicas = await client.get(
            f"/demo/orca/job/{session_id}/replicas", params={"now": running_now + 0.1}
        )
        phases = {
            replica["replica_id"]: replica["phase"]
            for replica in replicas.json()["replicas"]
        }
        for replica_id in scale_body["new_replicas"]:
            assert phases[replica_id] == "launching"

    @pytest.mark.asyncio
    async def test_orca_scale_enforces_quota_until_kill_releases_capacity(self, client):
        async def _fake_koi(*args, **kwargs):
            return {
                "_decision_id": "dec-scale-tight",
                "predicted_tps": 1100.0,
                "config": {
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "tp": 4,
                    "pp": 1,
                },
            }

        demo_server._request_koi_decision = _fake_koi
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_a100_tight",
                "scenario": "hero_elastic",
            },
        )
        assert launch.status_code == 200
        launch_body = launch.json()
        await asyncio.sleep(0.01)
        activated = await _wait_for_session_activation(
            client, launch_body["session_id"]
        )
        running_now = (
            launch_body["created_at"]
            + activated["launch_preview"]["launch_timing_s"]["total"]
            + 5
        )

        first_scale = await client.post(
            f"/demo/orca/job/{launch_body['session_id']}/scale",
            params={"now": running_now},
            json={
                "count": 1,
                "gpu_type": "L40S",
                "tp_size": 4,
                "pp_size": 1,
                "on_demand": True,
            },
        )
        assert first_scale.status_code == 200
        assert first_scale.json()["status"] == "scaling"

        second_scale = await client.post(
            f"/demo/orca/job/{launch_body['session_id']}/scale",
            params={"now": running_now + 0.1},
            json={
                "count": 1,
                "gpu_type": "L40S",
                "tp_size": 4,
                "pp_size": 1,
                "on_demand": True,
            },
        )
        assert second_scale.status_code == 200
        second_scale_body = second_scale.json()
        assert second_scale_body["status"] == "error"
        assert second_scale_body["reason"] == "insufficient_quota"

        resources_full = await client.get(
            "/demo/orca/resources", params={"now": running_now + 0.1}
        )
        g6e_full = next(
            quota
            for quota in resources_full.json()["quotas"]
            if quota["family"] == "G6E" and quota["market"] == "on_demand"
        )
        assert g6e_full["used_vcpus"] == 96

        added_replica = first_scale.json()["new_replicas"][0]
        kill = await client.post(
            f"/demo/orca/job/{launch_body['session_id']}/kill",
            params={"now": running_now + 0.2},
            json={"replica_ids": [added_replica]},
        )
        assert kill.status_code == 200
        assert kill.json()["killed"] == [added_replica]

        resources_after_kill = await client.get(
            "/demo/orca/resources", params={"now": running_now + 0.3}
        )
        g6e_after_kill = next(
            quota
            for quota in resources_after_kill.json()["quotas"]
            if quota["family"] == "G6E" and quota["market"] == "on_demand"
        )
        assert g6e_after_kill["used_vcpus"] == 48

        third_scale = await client.post(
            f"/demo/orca/job/{launch_body['session_id']}/scale",
            params={"now": running_now + 0.4},
            json={
                "count": 1,
                "gpu_type": "L40S",
                "tp_size": 4,
                "pp_size": 1,
                "on_demand": True,
            },
        )
        assert third_scale.status_code == 200
        assert third_scale.json()["status"] == "scaling"

    @pytest.mark.asyncio
    async def test_orca_scale_rejects_invalid_model_fit(self, client):
        async def _fake_koi(*args, **kwargs):
            return {
                "_decision_id": "dec-huge-model",
                "predicted_tps": 126.0,
                "config": {
                    "gpu_type": "A100-80GB",
                    "instance_type": "p4de.24xlarge",
                    "tp": 8,
                    "pp": 1,
                },
            }

        demo_server._request_koi_decision = _fake_koi
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "acme/HugeMoE",
                "avg_input_tokens": 800,
                "avg_output_tokens": 2000,
                "quota_preset": "aws_l40s_roomy",
                "scenario": "hero_elastic",
                "model_overrides": {
                    "num_params_billions": 227.27,
                    "num_layers": 62,
                    "hidden_dim": 3072,
                    "num_attention_heads": 48,
                    "num_kv_heads": 8,
                    "vocab_size": 200064,
                    "is_moe": True,
                    "num_experts": 256,
                    "active_experts": 8,
                    "architecture_family": "unknown",
                },
            },
        )
        assert launch.status_code == 200
        launch_body = launch.json()
        await asyncio.sleep(0.01)
        activated = await _wait_for_session_activation(
            client, launch_body["session_id"]
        )
        running_now = (
            launch_body["created_at"]
            + activated["launch_preview"]["launch_timing_s"]["total"]
            + 5
        )

        scale = await client.post(
            f"/demo/orca/job/{launch_body['session_id']}/scale",
            params={"now": running_now},
            json={
                "count": 1,
                "gpu_type": "L40S",
                "tp_size": 4,
                "pp_size": 1,
                "on_demand": True,
            },
        )
        assert scale.status_code == 200
        scale_body = scale.json()
        assert scale_body["status"] == "error"
        assert scale_body["reason"] == "invalid_placement"
        assert "not feasible" in scale_body["message"]

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
        await _wait_for_session_activation(client, session_id)
        running_now = (
            body["created_at"] + body["launch_preview"]["launch_timing_s"]["total"] + 10
        )

        replicas_resp = await client.get(
            f"/demo/orca/job/{session_id}/replicas", params={"now": running_now}
        )
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

        replicas_after = await client.get(
            f"/demo/orca/job/{session_id}/replicas", params={"now": running_now + 1.1}
        )
        target = next(
            replica
            for replica in replicas_after.json()["replicas"]
            if replica["replica_id"] == replica_id
        )
        assert target["phase"] == "killed"


class TestQuotaOverrides:
    @pytest_asyncio.fixture(autouse=True)
    async def _reset_overrides(self, client):
        demo_server.QUOTA_OVERRIDES._state.clear()
        if demo_server.QUOTA_OVERRIDES_PATH.exists():
            demo_server.QUOTA_OVERRIDES_PATH.unlink()
        yield
        demo_server.QUOTA_OVERRIDES._state.clear()
        if demo_server.QUOTA_OVERRIDES_PATH.exists():
            demo_server.QUOTA_OVERRIDES_PATH.unlink()

    @pytest.mark.asyncio
    async def test_overrides_endpoint_exposes_defaults_and_lock_state(self, client):
        resp = await client.get("/demo/quota/overrides")
        assert resp.status_code == 200
        body = resp.json()
        assert body["locked"] is False
        assert body["active_sessions"] == 0
        slugs = [preset["slug"] for preset in body["presets"]]
        assert "aws_l40s_roomy" in slugs
        aws = next(p for p in body["presets"] if p["slug"] == "aws_l40s_roomy")
        assert "G6E|us-east-1|on_demand" in aws["defaults"]
        assert any(row["family"] == "G6E" for row in aws["editable_rows"])

    @pytest.mark.asyncio
    async def test_save_override_persists_and_applies_to_launch(self, client):
        save = await client.post(
            "/demo/quota/overrides",
            json={
                "preset_slug": "aws_l40s_roomy",
                "overrides": {"G6E|us-east-1|on_demand": 512},
            },
        )
        assert save.status_code == 200
        assert save.json()["overrides"]["G6E|us-east-1|on_demand"] == 512

        # Disk persistence check
        assert demo_server.QUOTA_OVERRIDES_PATH.exists()
        stored = json.loads(demo_server.QUOTA_OVERRIDES_PATH.read_text())
        assert stored["aws_l40s_roomy"]["G6E|us-east-1|on_demand"] == 512

        # Catalog reflects overrides in both the per-preset map and quotas list.
        catalog_resp = await client.get("/demo/catalog")
        catalog = catalog_resp.json()
        aws = next(p for p in catalog["quota_presets"] if p["slug"] == "aws_l40s_roomy")
        assert aws["overrides"]["G6E|us-east-1|on_demand"] == 512
        g6e_row = next(q for q in aws["quotas"] if q["family"] == "G6E")
        assert g6e_row["baseline_vcpus"] == 512

        # Launch path should see the overridden quota in resource_map.
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_l40s_roomy",
                "scenario": "hero_elastic",
            },
        )
        assert launch.status_code == 200
        body = launch.json()
        g6e = next(
            row
            for row in body["resource_map"]["quotas"]
            if row["family"] == "G6E"
        )
        assert g6e["baseline_vcpus"] == 512

    @pytest.mark.asyncio
    async def test_lock_blocks_save_while_session_active(self, client):
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_l40s_roomy",
                "scenario": "hero_elastic",
            },
        )
        assert launch.status_code == 200

        save = await client.post(
            "/demo/quota/overrides",
            json={
                "preset_slug": "aws_l40s_roomy",
                "overrides": {"G6E|us-east-1|on_demand": 512},
            },
        )
        assert save.status_code == 409
        assert "quota_locked" in save.json()["detail"]

        reset = await client.post("/demo/quota/overrides/aws_l40s_roomy/reset")
        assert reset.status_code == 409

    @pytest.mark.asyncio
    async def test_reset_clears_overrides(self, client):
        await client.post(
            "/demo/quota/overrides",
            json={
                "preset_slug": "aws_l40s_roomy",
                "overrides": {"G6E|us-east-1|on_demand": 256},
            },
        )
        reset = await client.post("/demo/quota/overrides/aws_l40s_roomy/reset")
        assert reset.status_code == 200
        assert reset.json()["overrides"] == {}
        assert not demo_server.QUOTA_OVERRIDES.get("aws_l40s_roomy")
