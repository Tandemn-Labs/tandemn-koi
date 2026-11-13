"""Smoke tests for Koi HTTP endpoints and job-submission schemas."""

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.api.app import app
from src.api.schemas import JobSubmission


class ApiSmokeTests(unittest.TestCase):
    """Verify process health and unimplemented endpoint responses."""

    def setUp(self):
        app.state.runner = None
        app.state.shutting_down = False
        self.client = self.enterContext(TestClient(app))

    def tearDown(self):
        app.state.runner = None
        app.state.shutting_down = False

    def test_health_without_runner_returns_503(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "unhealthy"})

    def test_health_with_running_process_returns_200(self):
        app.state.runner = SimpleNamespace(returncode=None)

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_health_with_exited_process_returns_503(self):
        for returncode in (0, 1, -15):
            with self.subTest(returncode=returncode):
                app.state.runner = SimpleNamespace(returncode=returncode)

                response = self.client.get("/health")

                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json(), {"status": "unhealthy"})

    def test_health_reflects_process_exit(self):
        process = SimpleNamespace(returncode=None)
        app.state.runner = process
        self.assertEqual(self.client.get("/health").status_code, 200)

        process.returncode = 0

        self.assertEqual(self.client.get("/health").status_code, 503)

    def test_health_during_shutdown_returns_503(self):
        app.state.runner = SimpleNamespace(returncode=None)
        app.state.shutting_down = True

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "unhealthy"})

    def test_job_returns_501_without_request_validation(self):
        for payload in (None, {"kind": "online", "spec": {"model_id": "test"}}, {"invalid": True}):
            with self.subTest(payload=payload):
                response = self.client.post("/job", json=payload)

                self.assertEqual(response.status_code, 501)
                self.assertEqual(response.json(), {"detail": "Not implemented"})

    def test_resource_map_returns_501(self):
        response = self.client.get("/resource_map")

        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.json(), {"detail": "Not implemented"})


class EngineConfigSchemaTests(unittest.TestCase):
    """Verify the optional engine_config schema parses and rejects bad input."""

    def test_job_spec_stores_user_kubernetes_labels(self):
        submission = JobSubmission.model_validate(
            {
                "kind": "online",
                "spec": {
                    "model_id": "test",
                    "kubernetes_labels": {"team": "inference", "app": "chat"},
                },
            }
        )

        self.assertEqual(
            submission.spec.kubernetes_labels,
            {"team": "inference", "app": "chat"},
        )

    def test_job_spec_without_engine_config_still_parses(self):
        submission = JobSubmission.model_validate({"kind": "online", "spec": {"model_id": "test"}})

        self.assertIsNone(submission.spec.engine_config)
        self.assertEqual(submission.spec.kubernetes_labels, {})

    def test_partial_engine_config_leaves_unset_fields_unset(self):
        submission = JobSubmission.model_validate(
            {
                "kind": "batch",
                "spec": {
                    "model_id": "test",
                    "engine_config": {
                        "gpu_mem_util": 0.92,
                        "prefix_cache_enabled": False,
                        "chunked_prefill_enabled": False,
                    },
                },
            }
        )

        engine = submission.spec.engine_config
        self.assertEqual(engine.gpu_mem_util, 0.92)
        self.assertFalse(engine.prefix_cache_enabled)
        self.assertFalse(engine.chunked_prefill_enabled)
        self.assertIsNone(engine.max_model_len)
        self.assertIsNone(engine.max_num_seq)
        self.assertEqual(engine.extra_args, [])

    def test_per_gpu_overrides_and_extra_args_parse(self):
        submission = JobSubmission.model_validate(
            {
                "kind": "online",
                "spec": {
                    "model_id": "test",
                    "engine_config": {
                        "max_num_seq": [{"gpu_type": "H100", "value": 256}],
                        "kv_cache_dtype": [{"gpu_type": "MI300X", "value": "fp8"}],
                        "extra_args": ["--enable-prefix-caching", "--swap-space", "4"],
                    },
                },
            }
        )

        engine = submission.spec.engine_config
        self.assertEqual(engine.max_num_seq[0].value, 256)
        self.assertEqual(engine.kv_cache_dtype[0].gpu_type, "MI300X")
        self.assertEqual(engine.kv_cache_dtype[0].value, "fp8")
        self.assertEqual(engine.extra_args, ["--enable-prefix-caching", "--swap-space", "4"])

    def test_scalar_overrides_parse_without_per_gpu_entries(self):
        submission = JobSubmission.model_validate(
            {
                "kind": "online",
                "spec": {
                    "model_id": "test",
                    "engine_config": {"max_num_seq": 256, "gpu_mem_util": 0.9},
                },
            }
        )

        engine = submission.spec.engine_config
        self.assertEqual(engine.max_num_seq, 256)
        self.assertEqual(engine.gpu_mem_util, 0.9)

    def test_invalid_engine_config_is_rejected(self):
        bad_engine_configs = [
            {"gpu_mem_util": 0},
            {"gpu_mem_util": 1.5},
            {"max_model_len": 0},
            {"max_num_seq": -1},
            {"block_size": [{"gpu_type": "", "value": 16}]},
            {"chunk_size": 0.5},
            {"scheduling_policy": "wspt-lowercase"},
            {"router_policy": "kv_router"},
            {"spec_acceptance_threshold": -0.1},
            {"unknown_engine_field": True},
        ]
        for engine_config in bad_engine_configs:
            with (
                self.subTest(engine_config=engine_config),
                self.assertRaises(ValidationError),
            ):
                JobSubmission.model_validate(
                    {
                        "kind": "online",
                        "spec": {"model_id": "test", "engine_config": engine_config},
                    }
                )


if __name__ == "__main__":
    unittest.main()
