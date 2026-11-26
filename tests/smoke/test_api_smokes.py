"""Smoke tests for Koi HTTP endpoints and job-submission schemas."""

import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

    def test_jobs_endpoints_return_the_same_database_fields(self):
        pool = MagicMock()
        self.enterContext(patch.object(app.state, "postgres_pool", pool))
        conn = pool.connection.return_value.__enter__.return_value
        row = {"job_id": "job_1", "status": "queued", "deadline_hrs": 0, "priority_class": None}
        conn.execute.return_value.fetchall.return_value = [row]
        conn.execute.return_value.fetchone.return_value = row

        listing = self.client.get("/jobs")
        detail = self.client.get("/jobs/job_1")

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json(), row)
        self.assertEqual(listing.json(), [detail.json()])
        list_call, detail_call = conn.execute.call_args_list
        list_query = list_call.args[0]
        detail_query, params = detail_call.args
        self.assertIn("FROM jobs", list_query)
        self.assertEqual(
            list_query.partition(" ORDER BY ")[0],
            detail_query.removesuffix(" WHERE job_id = %s"),
        )
        self.assertEqual(params, ("job_1",))

        conn.execute.return_value.fetchone.return_value = None
        self.assertEqual(self.client.get("/jobs/missing").status_code, 404)

    def test_job_configuration_returns_stored_values_for_one_rank(self):
        pool = MagicMock()
        self.enterContext(patch.object(app.state, "postgres_pool", pool))
        conn = pool.connection.return_value.__enter__.return_value
        row = {
            "job": {"job_id": "job_1", "model_id": "model_1", "deadline_hrs": 0},
            "model_architecture": {"model_id": "model_1", "num_hidden_layers": 80},
            "placement": {"job_id": "job_1", "rank_id": "rank_2", "engine_name": "sglang"},
            "job_engine_defaults": None,
        }
        conn.execute.return_value.fetchone.return_value = row

        response = self.client.get("/jobs/job_1/configuration", params={"rank_id": "rank_2"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), row)
        conn.execute.assert_called_once()
        query, bindings = conn.execute.call_args.args
        self.assertIn("WHERE placement.job_id = %s AND placement.rank_id = %s", query)
        self.assertEqual(bindings, ("job_1", "rank_2"))

        conn.execute.return_value.fetchone.return_value = None
        response = self.client.get("/jobs/job_1/configuration", params={"rank_id": "missing"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/jobs/job_1/configuration").status_code, 422)

    def test_job_replicas_distinguishes_unplaced_and_missing_jobs(self):
        pool = MagicMock()
        self.enterContext(patch.object(app.state, "postgres_pool", pool))
        conn = pool.connection.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = []
        conn.execute.return_value.fetchone.return_value = {"exists": 1}
        response = self.client.get("/jobs/job_1/replicas")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])
        conn.execute.assert_called_with("SELECT 1 FROM jobs WHERE job_id = %s", ("job_1",))

        conn.execute.return_value.fetchone.return_value = None
        self.assertEqual(self.client.get("/jobs/missing/replicas").status_code, 404)

    def test_job_plan_reports_unavailable_historical_placements(self):
        pool = MagicMock()
        self.enterContext(patch.object(app.state, "postgres_pool", pool))
        conn = pool.connection.return_value.__enter__.return_value
        decision = {
            "job_id": "job_1",
            "tick": 0,
            "action": "place",
            "rationale": "Capacity available",
            "status": None,
            "created_at": None,
            "tick_rationale": None,
            "placement": None,
        }
        conn.execute.return_value.fetchone.return_value = decision

        response = self.client.get("/jobs/job_1/plan/0")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), decision)
        conn.execute.assert_called_once()
        query, params = conn.execute.call_args.args
        self.assertIn("WHERE plan_actions.tick = %s AND plan_actions.job_id = %s", query)
        self.assertIn("NULL AS placement", query)
        self.assertNotIn("FROM placement", query)
        self.assertEqual(params, (0, "job_1"))

        conn.execute.return_value.fetchone.return_value = None
        self.assertEqual(self.client.get("/jobs/job_1/plan/1").status_code, 404)
        self.assertEqual(self.client.get("/jobs/job_1/plan/-1").status_code, 422)
        self.assertEqual(conn.execute.call_count, 2)

    def test_job_replicas_groups_observed_replica_ids_by_rank(self):
        pool = MagicMock()
        self.enterContext(patch.object(app.state, "postgres_pool", pool))
        conn = pool.connection.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = [
            {"rank_id": "rank_1"},
            {"rank_id": "rank_2"},
        ]
        store = MagicMock()
        self.enterContext(patch.object(app.state, "telemetry_client", store))
        self.enterContext(patch.object(app.state, "telemetry_user_id", "usr_test"))
        session = store.session.return_value.__enter__.return_value
        session.execute.return_value.mappings.return_value = [
            {"rank_id": "rank_1", "chain_index": 0},
            {"rank_id": "rank_1", "chain_index": 2},
        ]

        response = self.client.get("/jobs/job_1/replicas")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [
                {
                    "rank_id": "rank_1",
                    "replicas": [{"chain_index": 0}, {"chain_index": 2}],
                },
                {"rank_id": "rank_2", "replicas": []},
            ],
        )
        conn.execute.assert_called_once()
        placement_query, placement_bindings = conn.execute.call_args.args
        self.assertIn(
            "FROM placement WHERE job_id = %s ORDER BY rank_id", " ".join(placement_query.split())
        )
        self.assertEqual(placement_bindings, ("job_1",))
        session.execute.assert_called_once()
        query, bindings = session.execute.call_args.args
        self.assertIn("SELECT DISTINCT gpu_metrics.rank_id, gpu_metrics.chain_index", str(query))
        self.assertIn("INTERVAL '1 hour'", str(query))
        self.assertEqual(
            bindings, {"user_id": "usr_test", "job_id": "job_1", "rank_ids": ["rank_1", "rank_2"]}
        )

    def test_job_replicas_reports_telemetry_unavailable(self):
        pool = MagicMock()
        self.enterContext(patch.object(app.state, "postgres_pool", pool))
        self.enterContext(patch.object(app.state, "telemetry_client", None))
        conn = pool.connection.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = [{"rank_id": "rank_1"}]

        response = self.client.get("/jobs/job_1/replicas")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Telemetry Store is unavailable"})

    def test_jobs_endpoints_without_pool_return_503(self):
        for path in (
            "/jobs",
            "/jobs/job_1",
            "/jobs/job_1/configuration?rank_id=rank_1",
            "/jobs/job_1/replicas",
            "/jobs/job_1/plan/0",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 503)

    def test_job_metrics_queries_one_replica_in_a_time_window(self):
        store = MagicMock()
        self.enterContext(patch.object(app.state, "telemetry_client", store))
        self.enterContext(patch.object(app.state, "telemetry_user_id", "usr_test"))
        session = store.session.return_value.__enter__.return_value
        start = "2026-11-25T10:00:00Z"
        end = "2026-11-25T11:00:00Z"
        sample = {
            "job_id": "job_1",
            "rank_id": "rank_1",
            "chain_index": 0,
            "ts": start,
            "metrics_json": {"p99_ttft_ms": 125, "sm_utilization": 0.65},
        }
        session.execute.return_value.mappings.return_value = [sample]
        params = {"rank_id": "rank_1", "chain_index": 0, "start": start, "end": end}

        response = self.client.get("/jobs/job_1/metrics", params=params)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [sample])
        statement, bindings = session.execute.call_args.args
        self.assertIn("jobs.user_id = :user_id", str(statement))
        self.assertIn("gpu_metrics.job_id = :job_id", str(statement))
        self.assertIn("gpu_metrics.rank_id = :rank_id", str(statement))
        self.assertIn("gpu_metrics.chain_index = :chain_index", str(statement))
        self.assertIn("gpu_metrics.ts >= :start AND gpu_metrics.ts <= :end", str(statement))
        self.assertEqual(
            bindings,
            {
                **params,
                "user_id": "usr_test",
                "job_id": "job_1",
                "start": datetime.fromisoformat(start),
                "end": datetime.fromisoformat(end),
            },
        )

        session.execute.return_value.mappings.return_value = []
        self.assertEqual(self.client.get("/jobs/job_1/metrics", params=params).json(), [])

    def test_job_metrics_rejects_invalid_queries_before_reading_store(self):
        store = MagicMock()
        self.enterContext(patch.object(app.state, "telemetry_client", store))
        self.enterContext(patch.object(app.state, "telemetry_user_id", "usr_test"))
        params = {
            "rank_id": "rank_1",
            "chain_index": 0,
            "start": "2026-11-25T10:00:00Z",
            "end": "2026-11-25T11:00:00Z",
        }
        for invalid in (
            {"chain_index": -1},
            {"start": "2026-11-25T10:00:00"},
            {"end": "2026-11-25T09:00:00Z"},
        ):
            with self.subTest(invalid=invalid):
                response = self.client.get("/jobs/job_1/metrics", params={**params, **invalid})
                self.assertEqual(response.status_code, 422)
        store.session.assert_not_called()


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
