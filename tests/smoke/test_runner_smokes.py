"""Smoke tests for the Koi runner entry point."""

import json
import tempfile
import types
import unittest

from src.orchestrator import runner
from src.orchestrator.debug_logging import DebugLogger
from src.orchestrator.fsm_states import FSMState, TickRunner


class RunnerSmokeTests(unittest.TestCase):
    """Verify runner control flow without touching Tandemn Store."""

    def test_main_runs_next_persisted_tick(self):
        """Without --tick, the runner starts at evidence current_tick + 1."""
        captured = {}
        evidence = _Evidence(current_tick=41)

        def build_runner(args):
            captured["args"] = args
            return object(), evidence, object(), object()

        def run_tick(tick):
            captured["tick"] = tick
            return _Context(tick)

        with tempfile.TemporaryDirectory() as log_dir:
            code = runner.main(
                [
                    "--user-id",
                    "user_1",
                    "--api-key",
                    "key",
                    "--tick-interval-sec",
                    "0",
                    "--log-level",
                    "CRITICAL",
                    "--log-dir",
                    log_dir,
                    "--run-id",
                    "test-main-next",
                ],
                build_runner_fn=build_runner,
                run_tick_fn=run_tick,
            )

        self.assertEqual(code, 0)
        self.assertEqual(captured["args"].user_id, "user_1")
        self.assertEqual(captured["tick"], 42)

    def test_main_respects_explicit_tick(self):
        """An explicit --tick overrides persisted evidence state."""
        captured = {}

        def build_runner(args):
            return object(), _Evidence(current_tick=41), object(), object()

        def run_tick(tick):
            captured["tick"] = tick
            return _Context(tick)

        with tempfile.TemporaryDirectory() as log_dir:
            code = runner.main(
                [
                    "--user-id",
                    "user_1",
                    "--api-key",
                    "key",
                    "--tick",
                    "7",
                    "--log-level",
                    "CRITICAL",
                    "--log-dir",
                    log_dir,
                    "--run-id",
                    "test-main-explicit",
                ],
                build_runner_fn=build_runner,
                run_tick_fn=run_tick,
            )

        self.assertEqual(code, 0)
        self.assertEqual(captured["tick"], 7)

    def test_run_ticks_runs_requested_count_and_clears_buffers(self):
        """A bounded loop increments ticks and clears debug buffers."""
        seen = []
        agent = _Agent()
        llm = _LLM()

        def run_tick(tick):
            seen.append(tick)
            agent.trace.events.append({"tick": tick})
            llm.calls.append({"tick": tick})
            return _Context(tick)

        code = runner.run_ticks(
            evidence_store=_Evidence(current_tick=10),
            agent=agent,
            llm=llm,
            requested_tick=None,
            ticks=3,
            run_tick_fn=run_tick,
        )

        self.assertEqual(code, 0)
        self.assertEqual(seen, [11, 12, 13])
        self.assertEqual(agent.trace.events, [])
        self.assertEqual(llm.calls, [])

    def test_run_ticks_writes_debug_logs_before_clearing_buffers(self):
        """Full trace mode persists LLM calls and agent events."""
        agent = _Agent()
        llm = _LLM()

        def run_tick(tick):
            agent.trace.events.append({"kind": "repl_exec", "tick": tick})
            llm.calls.append({"elapsed_sec": 0.1, "messages": [], "response": "ok"})
            return _Context(tick)

        with tempfile.TemporaryDirectory() as log_dir:
            debug_logger = DebugLogger(log_dir, trace="full", run_id="test-run")
            code = runner.run_ticks(
                evidence_store=_Evidence(current_tick=0),
                agent=agent,
                llm=llm,
                requested_tick=None,
                ticks=1,
                run_tick_fn=run_tick,
                debug_logger=debug_logger,
            )
            events = [
                json.loads(line) for line in debug_logger.events_path.read_text().splitlines()
            ]

        self.assertEqual(code, 0)
        self.assertEqual(agent.trace.events, [])
        self.assertEqual(llm.calls, [])
        self.assertEqual(
            [event["kind"] for event in events],
            ["tick_summary", "llm_summary", "agent_summary", "llm_calls", "agent_trace"],
        )
        self.assertEqual(events[3]["payload"]["calls"][0]["response"], "ok")
        self.assertEqual(events[4]["payload"]["events"][0]["kind"], "repl_exec")

    def test_debug_logger_writes_state_snapshot(self):
        """State logging captures Store snapshot counts and resources."""
        ctx = _Context(3)
        ctx.cluster_snapshot = types.SimpleNamespace(
            resources={
                "reserved|aws|us-east-1|use1-az1|H100": {
                    "gpu_type": "H100",
                    "free": 8,
                    "total": 8,
                    "pools": [{"instance_type": "p5.48xlarge", "free_instances": 1}],
                }
            },
            active_jobs=[
                {
                    "job_id": "job_running",
                    "status": "running",
                    "spec_json": {"model_id": "model_a"},
                    "active_chains": [{"chain_id": "chain_1"}],
                }
            ],
            pending_jobs=[{"job_id": "job_waiting", "status": "waiting"}],
        )

        with tempfile.TemporaryDirectory() as log_dir:
            debug_logger = DebugLogger(log_dir, run_id="state-test")
            debug_logger.persist_state(FSMState.S0_ENTER_TICK, ctx)
            event = json.loads(debug_logger.events_path.read_text().splitlines()[0])

        self.assertEqual(event["kind"], "state")
        self.assertEqual(event["payload"]["state"], "S0_ENTER_TICK")
        snapshot = event["payload"]["cluster_snapshot"]
        self.assertEqual(
            snapshot["resources"]["reserved|aws|us-east-1|use1-az1|H100"]["gpu_type"], "H100"
        )
        self.assertEqual(snapshot["active_jobs"][0]["job_id"], "job_running")

    def test_tick_runner_persist_state_uses_trace_logger(self):
        """TickRunner forwards state events to trace_logger.persist_state."""
        trace = _TraceRecorder()
        tick_runner = object.__new__(TickRunner)
        tick_runner.trace = trace
        ctx = _Context(9)

        tick_runner._persist_state(FSMState.S1_OBSERVE, ctx)

        self.assertEqual(trace.states, [(FSMState.S1_OBSERVE, ctx)])

    def test_build_runner_requires_api_key_before_store_access(self):
        """A missing API key fails before constructing real dependencies."""
        args = types.SimpleNamespace(user_id="user_1", api_key=None)

        with self.assertRaisesRegex(SystemExit, "OPENAI_API_KEY"):
            runner.build_runner(args)


class _Evidence:
    """Evidence-store test double exposing only current_tick."""

    def __init__(self, current_tick: int):
        self._current_tick = current_tick

    def current_tick(self) -> int:
        """Return the configured latest tick."""
        return self._current_tick


class _Context:
    """TickContext-shaped test double for summary logging."""

    def __init__(self, tick: int):
        self.tick = tick
        self.state_history = [FSMState.S0_ENTER_TICK, FSMState.S7_EXIT_TICK]
        self.evidence_rows = []
        self.validated_plan = types.SimpleNamespace(actions=[])
        self.deploy_acks = []
        self.error = None


class _Agent:
    """Agent test double with trace events."""

    def __init__(self):
        self.trace = types.SimpleNamespace(events=[])


class _LLM:
    """LLM test double with recorded calls."""

    def __init__(self):
        self.calls = []


class _TraceRecorder:
    """Trace logger test double that records state callbacks."""

    def __init__(self):
        self.states = []

    def persist_state(self, state, ctx) -> None:
        """Record one state callback."""
        self.states.append((state, ctx))


if __name__ == "__main__":
    unittest.main()
