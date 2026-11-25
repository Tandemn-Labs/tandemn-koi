"""Smoke tests for Koi HTTP endpoints and job-submission schemas."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.api.app import app
from src.api.schemas import JobSubmission
from tandemn_system_data.models.job import Job, RankAllocation, RunningJob


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

    def test_list_jobs_returns_scoped_summaries(self):
        self.enterContext(patch.object(app.state, "user_id", "usr_test"))
        store = Mock()
        self.enterContext(patch.object(app.state, "job_store", store))
        job = Job(
            user_id="usr_test",
            kind="online",
            status="running",
            spec_json={
                "model_id": "test-model",
                "job_intent": {"priority_class": "HIGH", "target_p99_ttft_ms": 125},
                "job_features": {"target_p99_tpot_ms": "40"},
                "internal_config": "private",
            },
        )
        store.list_jobs.return_value = [job]
        ranks = [
            RankAllocation(
                rank_id=f"rank_{index}",
                plan_id=None,
                role="aggregate",
                status="running",
                shape_json={"env": env},
                n_replicas=replicas,
            )
            for index, (env, replicas) in enumerate(
                [
                    (["on_demand", "aws", "us-east-1", "a", "H100"], 2),
                    ("on_demand|aws|us-east-1|b|H100", 1),
                    (None, 1),
                ]
            )
        ]
        store.running_jobs.return_value = [RunningJob(job=job, ranks=ranks)]

        response = self.client.get("/jobs", params={"user_id": "other"})

        store.list_jobs.assert_called_once_with("usr_test")
        store.running_jobs.assert_called_once_with("usr_test")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [
                {
                    "job_id": job.job_id,
                    "kind": "online",
                    "status": "running",
                    "owner_user_id": "usr_test",
                    "model_id": "test-model",
                    "priority_class": "HIGH",
                    "target_p99_ttft_ms": 125,
                    "target_p99_tpot_ms": 40,
                    "deadline_hrs": None,
                    "allocated_replicas": 4,
                    "placements": [{"cloud": "aws", "region": "us-east-1"}],
                    "created_at": job.model_dump(mode="json")["created_at"],
                    "finished_at": None,
                    "finish_reason": None,
                }
            ],
        )

        store.list_jobs.return_value = []
        self.assertEqual(self.client.get("/jobs").json(), [])

    def test_list_jobs_reads_legacy_batch_deadline(self):
        self.enterContext(patch.object(app.state, "user_id", "usr_test"))
        store = Mock()
        self.enterContext(patch.object(app.state, "job_store", store))
        for primary_deadline, expected in ((None, 2), ("", 2), ("invalid", 2), (0, 0), (3, 3)):
            with self.subTest(primary_deadline=primary_deadline):
                store.list_jobs.return_value = [
                    Job(
                        user_id="usr_test",
                        kind="batch",
                        spec_json={
                            "features": {"deadline_hrs": primary_deadline, "deadline_hours": 2}
                        },
                    )
                ]
                response = self.client.get("/jobs")

                self.assertEqual(response.status_code, 200)
                summary = response.json()[0]
                self.assertEqual(summary["deadline_hrs"], expected)
                self.assertIsNone(summary["allocated_replicas"])
                self.assertEqual(summary["placements"], [])
        store.running_jobs.assert_not_called()

    def test_list_jobs_without_scope_returns_503(self):
        self.enterContext(patch.object(app.state, "user_id", None))
        store = Mock()
        self.enterContext(patch.object(app.state, "job_store", store))

        self.assertEqual(self.client.get("/jobs").status_code, 503)
        store.list_jobs.assert_not_called()

    def test_get_job_queries_one_id_and_returns_the_database_row(self):
        pool = MagicMock()
        self.enterContext(patch.object(app.state, "postgres_pool", pool))
        conn = pool.connection.return_value.__enter__.return_value
        row = {"job_id": "job_1", "status": "queued", "deadline_hrs": 0, "priority_class": None}
        conn.execute.return_value.fetchone.return_value = row

        response = self.client.get("/jobs/job_1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), row)
        conn.execute.assert_called_once()
        query, params = conn.execute.call_args.args
        self.assertTrue(query.strip().endswith("WHERE job_id = %s"))
        self.assertEqual(params, ("job_1",))

        conn.execute.return_value.fetchone.return_value = None
        self.assertEqual(self.client.get("/jobs/missing").status_code, 404)

    def test_get_job_without_pool_returns_503(self):
        self.assertEqual(self.client.get("/jobs/job_1").status_code, 503)

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
