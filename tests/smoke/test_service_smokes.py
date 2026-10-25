"""Smoke tests for service supervision without Store or LLM access."""

import asyncio
import signal
import sys
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from src import service


class _Process:
    """Runner double with an asynchronous exit notification."""

    def __init__(self):
        self.pid = 12345
        self.returncode = None
        self.exited = asyncio.Event()
        self.honor_terminate = True
        self.terminate_calls = 0

    async def wait(self):
        await self.exited.wait()
        return self.returncode

    def finish(self, returncode):
        self.returncode = returncode
        self.exited.set()

    def terminate(self):
        self.terminate_calls += 1
        if self.honor_terminate:
            self.finish(0)


class _Server:
    """HTTP double that exits when the supervisor requests shutdown."""

    def __init__(self):
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.error = None
        self._should_exit = False

    @property
    def should_exit(self):
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value):
        self._should_exit = value
        if value:
            self.finished.set()

    async def serve(self):
        self.started.set()
        await self.finished.wait()
        if self.error is not None:
            raise self.error


class ServiceSmokeTests(unittest.IsolatedAsyncioTestCase):
    """Verify argument forwarding, health, and coupled lifecycle behavior."""

    async def asyncSetUp(self):
        self.process = _Process()
        self.server = _Server()
        self.handlers = {}
        self.task = None
        self.real_server_class = service._HTTPServer
        loop = asyncio.get_running_loop()
        self.enterContext(
            patch.object(loop, "add_signal_handler", side_effect=self.handlers.__setitem__)
        )
        self.remove_handler = self.enterContext(patch.object(loop, "remove_signal_handler"))
        self.spawn = self.enterContext(
            patch.object(
                service.asyncio, "create_subprocess_exec", new=AsyncMock(return_value=self.process)
            )
        )
        self.server_factory = self.enterContext(
            patch.object(service, "_HTTPServer", return_value=self.server)
        )
        self.config = self.enterContext(patch.object(service.uvicorn, "Config"))

    async def asyncTearDown(self):
        self.process.finish(0)
        self.server.finished.set()
        if self.task is not None and not self.task.done():
            await asyncio.wait_for(self.task, timeout=2)

    async def start_service(self, argv=None):
        self.task = asyncio.create_task(service.run_service([] if argv is None else argv))
        await asyncio.wait_for(self.server.started.wait(), timeout=2)

    async def test_forwards_arguments_and_inherits_environment_and_output(self):
        argv = ["--ticks", "0", "--log-dir", "logs/with spaces", "--model", "test-model"]
        await self.start_service(argv)

        self.spawn.assert_awaited_once_with(
            sys.executable, "-m", "src.orchestrator.runner", *argv, start_new_session=True
        )
        self.config.assert_called_once_with(
            service.app,
            host="0.0.0.0",
            port=8080,
            workers=1,
            reload=False,
            timeout_graceful_shutdown=service.HTTP_SHUTDOWN_TIMEOUT_SEC,
        )
        self.server_factory.assert_called_once()
        self.assertIs(service.app.state.runner, self.process)
        self.process.finish(0)
        self.assertEqual(await asyncio.wait_for(self.task, timeout=2), 0)

    async def test_http_remains_responsive_while_runner_is_waiting(self):
        await self.start_service()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.app), base_url="http://koi"
        ) as client:
            response = await asyncio.wait_for(client.get("/health"), timeout=1)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.process.returncode)
        self.assertFalse(self.task.done())

    async def test_runner_failure_stops_http_and_preserves_exit_status(self):
        await self.start_service()
        self.process.finish(7)

        self.assertEqual(await asyncio.wait_for(self.task, timeout=2), 7)
        self.assertTrue(self.server.should_exit)
        self.assertEqual(self.process.terminate_calls, 0)
        self.assertIsNone(service.app.state.runner)
        self.spawn.assert_awaited_once()

    async def test_runner_signal_exit_maps_to_shell_status(self):
        await self.start_service()
        self.process.finish(-9)

        self.assertEqual(await asyncio.wait_for(self.task, timeout=2), 137)

    async def test_http_exit_stops_runner_and_returns_failure(self):
        await self.start_service()
        with self.assertLogs("koi.service", level="ERROR"):
            self.server.finished.set()
            self.assertEqual(await asyncio.wait_for(self.task, timeout=2), 1)

        self.assertEqual(self.process.terminate_calls, 1)
        self.assertEqual(self.process.returncode, 0)

    async def test_http_failure_stops_runner(self):
        await self.start_service()
        self.server.error = RuntimeError("bind failed")
        with self.assertLogs("koi.service", level="ERROR"):
            self.server.finished.set()
            self.assertEqual(await asyncio.wait_for(self.task, timeout=2), 1)

        self.assertEqual(self.process.terminate_calls, 1)

    async def test_sigterm_marks_unhealthy_and_drains_runner(self):
        previous_handler = signal.getsignal(signal.SIGTERM)
        await self.start_service()
        self.handlers[signal.SIGTERM]()
        self.assertTrue(service.app.state.shutting_down)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.app), base_url="http://koi"
        ) as client:
            self.assertEqual((await client.get("/health")).status_code, 503)

        self.assertEqual(await asyncio.wait_for(self.task, timeout=2), 0)
        self.assertEqual(self.process.terminate_calls, 1)
        self.assertTrue(self.server.should_exit)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)
        self.remove_handler.assert_any_call(signal.SIGTERM)

    async def test_sigint_also_drains_runner(self):
        await self.start_service()
        self.handlers[signal.SIGINT]()

        self.assertEqual(await asyncio.wait_for(self.task, timeout=2), 0)
        self.assertEqual(self.process.terminate_calls, 1)

    async def test_shutdown_timeout_kills_runner_group_and_returns_failure(self):
        self.process.honor_terminate = False
        await self.start_service()
        with (
            patch.object(service, "RUNNER_SHUTDOWN_TIMEOUT_SEC", 0.01),
            patch.object(
                service.os, "killpg", side_effect=lambda pid, sig: self.process.finish(-9)
            ) as kill,
            self.assertLogs("koi.service", level="ERROR"),
        ):
            self.handlers[signal.SIGTERM]()
            self.assertEqual(await asyncio.wait_for(self.task, timeout=2), 1)

        kill.assert_called_once_with(self.process.pid, signal.SIGKILL)
        self.assertEqual(self.process.returncode, -9)

    async def test_spawn_failure_does_not_start_http_or_retry(self):
        self.spawn.side_effect = OSError("cannot start runner")
        with self.assertLogs("koi.service", level="ERROR"):
            self.assertEqual(await service.run_service([]), 1)

        self.spawn.assert_awaited_once()
        self.assertFalse(self.server.started.is_set())
        self.assertIsNone(service.app.state.runner)

    async def test_cancellation_still_cleans_up_runner(self):
        await self.start_service()
        self.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self.task

        self.assertEqual(self.process.terminate_calls, 1)
        self.assertTrue(self.server.should_exit)
        self.assertIsNone(service.app.state.runner)

    async def test_uvicorn_startup_system_exit_is_contained(self):
        server = self.real_server_class(self.config.return_value)
        with (
            patch.object(service.uvicorn.Server, "serve", new=AsyncMock(side_effect=SystemExit(1))),
            self.assertRaisesRegex(RuntimeError, "HTTP server failed to start"),
        ):
            await server.serve()

    async def test_uvicorn_does_not_replace_supervisor_signal_handlers(self):
        server = self.real_server_class(self.config.return_value)
        previous_handler = signal.getsignal(signal.SIGTERM)
        with server.capture_signals():
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)


if __name__ == "__main__":
    unittest.main()
