"""Smoke tests for the Koi runner entry point."""

import json
import tempfile
import types
import unittest
from unittest.mock import DEFAULT, patch

from src.orchestrator import runner
from src.orchestrator.debug_logging import DebugLogger
from src.orchestrator.fsm_states import FSMState, TickRunner


class RunnerSmokeTests(unittest.TestCase):
    """Verify runner control flow without touching Tandemn Store."""

    def test_parse_args_accepts_surrogate_configuration(self):
        args = runner.parse_args(
            [
                "--surrogate-lower-quantile",
                "0.1",
                "--surrogate-peer-mode",
                "enabled",
                "--partial-online-admission",
                "advisory",
            ]
        )

        self.assertEqual(args.surrogate_lower_quantile, 0.1)
        self.assertEqual(args.surrogate_peer_mode, "enabled")
        self.assertEqual(args.partial_online_admission, "advisory")

    def test_build_runner_wires_partial_online_admission_into_validator(self):
        with patch.dict(runner.os.environ, {}, clear=True):
            args = runner.parse_args(
                [
                    "--user-id",
                    "user_1",
                    "--api-key",
                    "test-key",
                    "--partial-online-admission",
                    "advisory",
                ]
            )
        candidate_graph = types.SimpleNamespace(y=[], v=[])
        mechanism_registry = object()
        confidence_service = object()
        resource_map = object()

        with (
            patch.object(runner.agent_tools, "configure_surrogate_call_budget"),
            patch.object(runner.agent_tools, "configure_partial_online_admission"),
            patch.multiple(
                runner,
                PostgresClient=DEFAULT,
                init_causal_graph=DEFAULT,
                EvidenceService=DEFAULT,
                DRO=DEFAULT,
                RegretCalculator=DEFAULT,
                Cusum=DEFAULT,
                ICP=DEFAULT,
                QuadrantValidator=DEFAULT,
                SlowLoop=DEFAULT,
                ResourceMapManager=DEFAULT,
                GpuMetricStore=DEFAULT,
                StoreTelemetry=DEFAULT,
                Validator=DEFAULT,
            ) as dependencies,
            self.assertRaisesRegex(RuntimeError, "validator wired"),
        ):
            dependencies["init_causal_graph"].return_value = (
                candidate_graph,
                mechanism_registry,
                confidence_service,
            )
            dependencies["ResourceMapManager"].return_value = resource_map
            dependencies["Validator"].side_effect = RuntimeError("validator wired")
            runner.build_runner(args)

        dependencies["Validator"].assert_called_once_with(
            candidate_graph=candidate_graph,
            mechanism_registry=mechanism_registry,
            resource_map=resource_map,
            partial_online_admission_mode="advisory",
        )

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

    def test_run_ticks_all_trace_persists_llm_calls_and_agent_events(self):
        """All trace mode persists LLM calls and agent events."""
        agent = _Agent()
        llm = _LLM()

        def run_tick(tick):
            agent.trace.events.append({"kind": "repl_exec", "tick": tick})
            llm.calls.append({"elapsed_sec": 0.1, "messages": [], "response": "ok"})
            return _Context(tick)

        with tempfile.TemporaryDirectory() as log_dir:
            debug_logger = DebugLogger(log_dir, trace="all", run_id="test-run")
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

    def test_run_ticks_no_llm_trace_omits_llm_output(self):
        """No-LLM trace mode retains decision metadata without transcripts."""
        agent = _Agent()
        llm = _LLM()

        def run_tick(tick):
            agent.trace.events.append({"kind": "repl_exec", "tick": tick})
            llm.calls.append({"elapsed_sec": 0.1, "messages": [], "response": "secret"})
            return _Context(tick)

        with tempfile.TemporaryDirectory() as log_dir:
            debug_logger = DebugLogger(log_dir, trace="no-llm", run_id="no-llm-test")
            runner.run_ticks(
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

        self.assertEqual(
            [event["kind"] for event in events], ["tick_summary", "llm_summary", "agent_summary"]
        )
        self.assertNotIn("responses", events[1]["payload"])

    def test_errors_trace_writes_failed_llm_metadata_only(self):
        """Error trace mode excludes prompts and responses from failed calls."""
        llm = types.SimpleNamespace(
            calls=[
                {
                    "call_index": 0,
                    "elapsed_sec": 0.1,
                    "error": "RuntimeError('boom')",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as log_dir:
            debug_logger = DebugLogger(log_dir, trace="errors", run_id="errors-test")
            debug_logger.persist_runner_tick(_Context(1), _Agent(), llm)
            events = [
                json.loads(line) for line in debug_logger.events_path.read_text().splitlines()
            ]

        self.assertEqual(
            [event["kind"] for event in events],
            ["tick_summary", "llm_summary", "agent_summary", "llm_errors"],
        )
        self.assertEqual(events[3]["payload"]["calls"][0]["error"], "RuntimeError('boom')")

    def test_surrogate_logging_respects_trace_detail(self):
        trace = {
            "schema_version": 3,
            "scenario": "peak",
            "composite_version": "koi-surrogate-v3:test",
            "normalized_candidate": {"job_config": {"model_id": "secret-model"}},
            "components": {"primary": {"status": "success", "version": "aic-v1"}},
            "backends": {"primary": {"status": "success", "version": "aic-v1"}},
            "compatibility": {
                "primary": {
                    "gpu": {
                        "requested": "A10G",
                        "resolved": "A30",
                        "kind": "nearest",
                        "confidence": 0.5,
                    }
                }
            },
            "fusion": {"status": "insufficient_evidence", "lower_quantile": 0.05},
            "calibration": {"status": "insufficient_evidence", "offsets_y": {}},
            "timings_ms": {"total": 1.0},
        }

        with tempfile.TemporaryDirectory() as log_dir:
            compact = DebugLogger(log_dir, trace="no-llm", run_id="surrogate-compact")
            compact.persist_surrogate_prediction(trace, tick=4)
            compact_event = json.loads(compact.events_path.read_text().splitlines()[0])

            full = DebugLogger(log_dir, trace="all", run_id="surrogate-full")
            full.persist_surrogate_prediction(trace, tick=4)
            full_event = json.loads(full.events_path.read_text().splitlines()[0])

        self.assertEqual(compact_event["kind"], "surrogate_prediction")
        self.assertEqual(compact_event["tick"], 4)
        self.assertNotIn("normalized_candidate", compact_event["payload"])
        self.assertEqual(
            compact_event["payload"]["compatibility"]["primary"]["gpu"]["resolved"],
            "A30",
        )
        self.assertEqual(
            full_event["payload"]["normalized_candidate"]["job_config"]["model_id"],
            "secret-model",
        )

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
            debug_logger = DebugLogger(log_dir, trace="all", run_id="state-test")
            debug_logger.persist_state(FSMState.S0_ENTER_TICK, ctx)
            event = json.loads(debug_logger.events_path.read_text().splitlines()[0])

        self.assertEqual(event["kind"], "state")
        self.assertEqual(event["payload"]["state"], "S0_ENTER_TICK")
        snapshot = event["payload"]["cluster_snapshot"]
        self.assertEqual(
            snapshot["resources"]["reserved|aws|us-east-1|use1-az1|H100"]["gpu_type"], "H100"
        )
        self.assertEqual(snapshot["active_jobs"][0]["job_id"], "job_running")

    def test_debug_logger_all_trace_preserves_raw_state_and_plan(self):
        """All trace mode does not compact state fields or plan ladders."""
        ctx = _Context(3)
        ctx.cluster_snapshot = types.SimpleNamespace(
            samples=list(range(81)),
            long_value="x" * 5001,
        )
        ctx.candidate_plan = types.SimpleNamespace(
            actions=[
                types.SimpleNamespace(
                    job_id="job_1",
                    type="place",
                    ladder=[
                        types.SimpleNamespace(
                            env=[
                                "reserved",
                                "aws",
                                "us-east-1",
                                "us-east-1a",
                                "L40S",
                            ],
                            config={"gpu_count": 1, "tp": 1},
                            n_replicas=2,
                        )
                    ],
                )
            ],
            tick_rationale="place job_1",
        )

        with tempfile.TemporaryDirectory() as log_dir:
            debug_logger = DebugLogger(log_dir, trace="all", run_id="all-test")
            debug_logger.persist_state(FSMState.S0_ENTER_TICK, ctx)
            debug_logger.persist_state(FSMState.S4_AGENTIC_PLAN, ctx)
            events = [
                json.loads(line) for line in debug_logger.events_path.read_text().splitlines()
            ]

        self.assertEqual(events[0]["payload"]["cluster_snapshot"]["samples"], list(range(81)))
        self.assertEqual(len(events[0]["payload"]["cluster_snapshot"]["long_value"]), 5001)
        ladder = events[1]["payload"]["candidate_plan"]["actions"][0]["ladder"][0]
        self.assertEqual(ladder["config"], {"gpu_count": 1, "tp": 1})
        self.assertEqual(ladder["n_replicas"], 2)

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
