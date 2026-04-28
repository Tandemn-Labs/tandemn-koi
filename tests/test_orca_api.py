"""Tests for koi/tools/orca_api.py"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from koi.tools.orca_api import OrcaClient


class MockResponse:
    """Mock aiohttp response."""
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status = status

    async def json(self):
        return self._json

    def raise_for_status(self):
        if self.status >= 400:
            raise Exception(f"HTTP {self.status}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSession:
    """Mock aiohttp.ClientSession."""
    def __init__(self):
        self.closed = False
        self._responses = {}

    def set_response(self, method, url_suffix, json_data, status=200):
        self._responses[(method, url_suffix)] = MockResponse(json_data, status)

    def get(self, url, **kwargs):
        for (method, suffix), resp in self._responses.items():
            if method == "get" and url.endswith(suffix):
                return resp
        return MockResponse({}, 404)

    def post(self, url, **kwargs):
        for (method, suffix), resp in self._responses.items():
            if method == "post" and url.endswith(suffix):
                return resp
        return MockResponse({}, 404)


@pytest.fixture
def mock_session():
    return MockSession()


@pytest.fixture
def orca(mock_session):
    client = OrcaClient("http://localhost:26336", session=mock_session)
    return client


class TestGetResources:
    @pytest.mark.asyncio
    async def test_returns_shape_c(self, orca, mock_session):
        mock_session.set_response("get", "/resources", {
            "instances": [{"instance_type": "g6e.12xlarge", "gpu_type": "L40S"}],
            "quotas": [{"family": "G", "region": "us-east-1"}],
        })
        result = await orca.get_resources()
        assert "instances" in result
        assert len(result["instances"]) == 1


class TestGetJobMetrics:
    @pytest.mark.asyncio
    async def test_returns_metrics(self, orca, mock_session):
        mock_session.set_response("get", "/job/job-123/metrics", {
            "avg_generation_throughput_toks_per_s": 1500.0,
            "gpu_cache_usage_perc": 0.65,
            "num_requests_running": 12,
        })
        result = await orca.get_job_metrics("job-123")
        assert result["avg_generation_throughput_toks_per_s"] == 1500.0

    @pytest.mark.asyncio
    async def test_404_returns_empty(self, orca, mock_session):
        # Default mock returns 404
        result = await orca.get_job_metrics("nonexistent")
        assert result == {}


class TestGetChunkProgress:
    @pytest.mark.asyncio
    async def test_returns_progress(self, orca, mock_session):
        mock_session.set_response("get", "/job/job-123/chunks/progress", {
            "total": 50, "pending": 10, "inflight": 5, "completed": 35, "failed": 0,
        })
        result = await orca.get_chunk_progress("job-123")
        assert result["completed"] == 35
        assert result["total"] == 50


class TestScale:
    @pytest.mark.asyncio
    async def test_scale_up(self, orca, mock_session):
        mock_session.set_response("post", "/job/job-123/scale", {
            "status": "scaling",
            "new_replicas": ["job-123-v1-r0", "job-123-v1-r1"],
        })
        result = await orca.scale_job("job-123", "A100-80GB", 4, 2, 2)
        assert result["status"] == "scaling"
        assert len(result["new_replicas"]) == 2

    @pytest.mark.asyncio
    async def test_scale_job_raises_on_http_error(self, orca, mock_session):
        mock_session.set_response("post", "/job/job-123/scale", {
            "status": "error",
            "detail": "no capacity",
        }, status=409)
        with pytest.raises(Exception, match="HTTP 409"):
            await orca.scale_job("job-123", "A100-80GB", 4, 2, 2)

    @pytest.mark.asyncio
    async def test_scale_job_default_force_false(self, orca, mock_session):
        """Default behavior: force=False is sent so Orca runs the
        feasibility check. This is the normal scale-up path used by
        runtime triggers."""
        captured = {}

        original_post = mock_session.post

        def _capture_post(url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            return original_post(url, json=json, **kwargs)

        mock_session.post = _capture_post
        mock_session.set_response("post", "/job/job-1/scale", {"status": "scaling"})

        await orca.scale_job("job-1", "L40S", 1, 1, 1)
        assert captured["json"]["force"] is False

    @pytest.mark.asyncio
    async def test_scale_job_force_true_passed_through(self, orca, mock_session):
        """Recovery path: force=True must reach Orca's payload so the
        feasibility check is skipped (the agent has already overridden
        the solver's recommendation with a deliberate alternative)."""
        captured = {}

        original_post = mock_session.post

        def _capture_post(url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            return original_post(url, json=json, **kwargs)

        mock_session.post = _capture_post
        mock_session.set_response("post", "/job/job-1/scale", {"status": "scaling"})

        await orca.scale_job("job-1", "L40S", 1, 1, 1, force=True)
        assert captured["json"]["force"] is True
        assert captured["json"]["gpu_type"] == "L40S"
        assert captured["json"]["count"] == 1


class TestKill:
    @pytest.mark.asyncio
    async def test_kill_replicas(self, orca, mock_session):
        mock_session.set_response("post", "/job/job-123/kill", {
            "status": "killing",
            "killed": ["job-123-v0-r2", "job-123-v0-r3"],
            "reclaimed": 8,
        })
        result = await orca.kill_replicas("job-123", ["job-123-v0-r2", "job-123-v0-r3"])
        assert result["reclaimed"] == 8

    @pytest.mark.asyncio
    async def test_kill_replicas_raises_on_http_error(self, orca, mock_session):
        mock_session.set_response("post", "/job/job-123/kill", {
            "status": "error",
            "detail": "cannot kill",
        }, status=500)
        with pytest.raises(Exception, match="HTTP 500"):
            await orca.kill_replicas("job-123", ["job-123-v0-r2"])


class TestSubmitBatch:
    @pytest.mark.asyncio
    async def test_submit(self, orca, mock_session):
        mock_session.set_response("post", "/submit/batch", {
            "job_id": "job-new123",
            "status": "launching",
        })
        result = await orca.submit_batch(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            input_file="s3://bucket/input.jsonl",
            instance_type="p4de.24xlarge",
            gpu_type="A100-80GB",
            tp=4, pp=2, dp=1,
        )
        assert result["job_id"] == "job-new123"
        assert result["status"] == "launching"


class TestSwap:
    @pytest.mark.asyncio
    async def test_swap(self, orca, mock_session):
        mock_session.set_response("post", "/job/job-123/swap", {
            "status": "swapping",
            "old_replicas": ["job-123-v0-r0"],
            "new_replicas": ["job-123-v1-r0"],
        })
        result = await orca.swap_replicas("job-123", "H100", 8, 1)
        assert result["status"] == "swapping"

    @pytest.mark.asyncio
    async def test_swap_replicas_raises_on_http_error(self, orca, mock_session):
        mock_session.set_response("post", "/job/job-123/swap", {
            "status": "error",
            "detail": "swap rejected",
        }, status=400)
        with pytest.raises(Exception, match="HTTP 400"):
            await orca.swap_replicas("job-123", "H100", 8, 1)
