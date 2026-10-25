"""Run Koi's HTTP server and one independently supervised runner process."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections.abc import Generator
from contextlib import contextmanager, suppress

import uvicorn

from src.api.app import app

log = logging.getLogger("koi.service")

HTTP_PORT = 8080
HTTP_SHUTDOWN_TIMEOUT_SEC = 30
# Kubernetes termination grace must exceed the runner's drain budget.
RUNNER_SHUTDOWN_TIMEOUT_SEC = 300


class _HTTPServer(uvicorn.Server):
    """
    Keep signal ownership and startup failures inside the supervisor.
    Supervisor is the current script that you are seeing.
    """

    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield

    async def serve(self, sockets=None) -> None:
        try:
            await super().serve(sockets=sockets)
        except SystemExit as exc:
            raise RuntimeError("HTTP server failed to start") from exc


async def _stop_runner(
    koi_loop_process: asyncio.subprocess.Process | None,
    koi_loop_exit_watcher: asyncio.Task[int] | None,
) -> bool:
    """Drain the runner, returning False if it required a forced kill."""
    if koi_loop_process is None or koi_loop_exit_watcher is None:
        return True

    if koi_loop_process.returncode is None:
        with suppress(ProcessLookupError):
            koi_loop_process.terminate()

    try:
        # Keep the exit watcher alive so we can reap the process after a timeout.
        await asyncio.wait_for(
            asyncio.shield(koi_loop_exit_watcher), timeout=RUNNER_SHUTDOWN_TIMEOUT_SEC
        )
        return True
    except TimeoutError:
        log.error("runner shutdown timed out; killing its process group")
        with suppress(ProcessLookupError):
            os.killpg(koi_loop_process.pid, signal.SIGKILL)
        await koi_loop_exit_watcher
        return False


async def _wait_for_http_shutdown(http_server_task: asyncio.Task[None] | None) -> bool:
    """Wait for HTTP to stop, returning False on failure or timeout."""
    if http_server_task is None:
        return True

    try:
        # Collect server failures without interrupting the remaining cleanup.
        await asyncio.wait_for(
            asyncio.gather(http_server_task, return_exceptions=True),
            timeout=HTTP_SHUTDOWN_TIMEOUT_SEC + 1,
        )
    except TimeoutError:
        log.error("HTTP shutdown timed out")
        return False

    if http_server_task.cancelled():
        return False
    return http_server_task.exception() is None


async def run_service(argv: list[str]) -> int:
    """Forward runner arguments and stop both components when either exits."""
    server = _HTTPServer(
        uvicorn.Config(
            app,
            host="0.0.0.0",
            port=HTTP_PORT,
            workers=1,
            reload=False,
            timeout_graceful_shutdown=HTTP_SHUTDOWN_TIMEOUT_SEC,
        )
    )
    app.state.runner = None
    app.state.shutting_down = False
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    koi_loop_process = None
    koi_loop_exit_watcher: asyncio.Task[int] | None = None
    http_server_task: asyncio.Task[None] | None = None
    shutdown_signal_task: asyncio.Task[bool] | None = None
    exit_code = 1

    def request_shutdown() -> None:
        app.state.shutting_down = True
        shutdown.set()

    async def cleanup_supervisor() -> None:
        tasks = [
            task
            for task in (koi_loop_exit_watcher, http_server_task, shutdown_signal_task)
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        app.state.runner = None
        for sig, handler in previous_handlers.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)

    try:
        for sig in previous_handlers:
            loop.add_signal_handler(sig, request_shutdown)
        koi_loop_process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "src.orchestrator.runner",
            *argv,
            start_new_session=True,
        )
        app.state.runner = koi_loop_process
        log.info("started runner pid=%d", koi_loop_process.pid)
        koi_loop_exit_watcher = asyncio.create_task(koi_loop_process.wait())
        http_server_task = asyncio.create_task(server.serve())
        shutdown_signal_task = asyncio.create_task(shutdown.wait())
        done, _ = await asyncio.wait(
            (koi_loop_exit_watcher, http_server_task, shutdown_signal_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if koi_loop_exit_watcher in done:
            returncode = koi_loop_exit_watcher.result()
            log.info("runner exited with status %d", returncode)
            exit_code = returncode if returncode >= 0 else 128 - returncode
        elif shutdown_signal_task in done:
            log.info("shutdown requested")
            exit_code = 0
        else:
            http_server_task.result()
            log.error("HTTP server exited unexpectedly")
    except Exception:
        log.exception("Koi service failed")
    finally:
        app.state.shutting_down = True
        server.should_exit = True
        try:
            if not await _stop_runner(koi_loop_process, koi_loop_exit_watcher):
                exit_code = 1
            if not await _wait_for_http_shutdown(http_server_task):
                exit_code = 1
        finally:
            await cleanup_supervisor()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Start the service; all command-line arguments belong to the runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return asyncio.run(run_service(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
