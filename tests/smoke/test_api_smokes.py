"""Smoke tests for Koi HTTP endpoints without Store or LLM access."""

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient
from src.api.app import app


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


if __name__ == "__main__":
    unittest.main()
