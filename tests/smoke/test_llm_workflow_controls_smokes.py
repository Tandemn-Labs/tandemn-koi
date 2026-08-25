"""Focused smoke tests for bounded LLM workflow and trace controls."""

import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

from src.agent.agent import AgentTrace, KoiAgentHarness, SpecialistRunner
from src.agent.tools import agent_tools
from src.core.models import ActionType, Plan, PlanAction, RankSpec
from src.orchestrator import debug_logging, runner
from src.orchestrator.debug_logging import DebugLogger


class LLMWorkflowControlsSmokeTests(unittest.TestCase):
    """Exercise workflow controls without external services."""

    def test_specialist_empty_response_consumes_one_of_two_total_attempts(self):
        class ScriptedLLM:
            def __init__(self):
                self.responses = ["", "not-json", '{"job_id":"job_1"}']
                self.calls = 0
                self.messages = []

            def complete(self, messages):
                self.messages.append([dict(message) for message in messages])
                response = self.responses[self.calls]
                self.calls += 1
                return response

        llm = ScriptedLLM()
        with patch.object(
            agent_tools,
            "get_job_brief",
            return_value={"job_id": "job_1", "current_ladder": None},
        ):
            result = SpecialistRunner(llm).run_one(
                "job_1",
                {"user_id": "user_1", "env_budget": {}},
            )

        self.assertEqual(llm.calls, 2)
        self.assertIn("response was empty", llm.messages[1][-1]["content"])
        self.assertEqual(result["type"], "defer")
        self.assertEqual(result["fitness"], "blocked")
        self.assertNotIn("predicted_y", result)
        self.assertNotIn("predicted_sigma", result)

    def test_specialist_accepts_valid_response_without_prediction_fields(self):
        proposal = {
            "job_id": "job_1",
            "user_id": "user_1",
            "type": "defer",
            "ladder": [],
            "budget_utilization": {},
            "used_capacity": {},
            "fitness": "blocked",
            "marginal_value_of_more": {},
            "unused_capacity": {},
            "mechanism_ids": [],
            "new_mechanism_proposals": [],
            "reasoning": "No feasible placement in the slice.",
        }
        llm = types.SimpleNamespace(complete=lambda _messages: json.dumps(proposal))
        with patch.object(
            agent_tools,
            "get_job_brief",
            return_value={"job_id": "job_1", "current_ladder": None},
        ):
            result = SpecialistRunner(llm).run_one(
                "job_1",
                {"user_id": "user_1", "env_budget": {}},
            )

        self.assertEqual(result, proposal)
        self.assertEqual(
            SpecialistRunner._validate(
                result,
                "job_1",
                {"user_id": "user_1", "env_budget": {}},
            ),
            [],
        )
        self.assertNotIn("predicted_y", result)
        self.assertNotIn("predicted_sigma", result)

    def test_unscorable_plan_is_excluded_before_negative_valid_plan_selection(self):
        unscorable = Plan(tick=7, tick_rationale="unscorable")
        valid_negative = Plan(tick=7, tick_rationale="valid-negative")
        plans = iter((unscorable, valid_negative))
        harness = KoiAgentHarness.__new__(KoiAgentHarness)
        harness.k_p = 2
        harness.trace = AgentTrace()
        harness.plan_validator = None
        harness._pending_violations = []
        harness._current_tick = 0
        harness.one_trajectory = lambda **_kwargs: next(plans)

        def compute_sigma(plan):
            if plan is unscorable:
                raise RuntimeError("cannot score")
            return {"aggregate_sigma": -4.5}

        with (
            patch.object(agent_tools, "bind_tools"),
            patch.object(agent_tools, "assert_planning_ready"),
            patch.object(agent_tools, "compute_sigma", side_effect=compute_sigma),
            patch.object(
                agent_tools,
                "stamp_plan_predictions",
                side_effect=lambda plan, _snapshot: plan,
            ),
        ):
            selected = harness.run_agent_loop(None, None, None, None, tick=7)

        self.assertIs(selected, valid_negative)
        scored_events = [
            event for event in harness.trace.events if event["kind"] == "kp_candidate_scored"
        ]
        unscorable_events = [
            event for event in harness.trace.events if event["kind"] == "kp_candidate_unscorable"
        ]
        self.assertEqual([event["score"] for event in scored_events], [-4.5])
        self.assertEqual([event["score"] for event in unscorable_events], [None])
        self.assertEqual([event["k_idx"] for event in scored_events], [1])
        self.assertEqual([event["k_idx"] for event in unscorable_events], [0])
        self.assertNotIn("Infinity", json.dumps(harness.trace.events))

    def test_plan_with_unscored_ladder_action_is_unscorable(self):
        harness = KoiAgentHarness.__new__(KoiAgentHarness)
        plan = Plan(
            tick=7,
            actions=[
                PlanAction(
                    job_id="job_1",
                    type=ActionType.PLACE,
                    ladder=[
                        RankSpec(
                            role="aggregate",
                            env=("reserved", "aws", "r1", "z1", "H100"),
                            config={"instance_type": "p5", "gpu_count": 1, "tp": 1, "pp": 1},
                        )
                    ],
                )
            ],
        )
        with patch.object(
            agent_tools,
            "compute_sigma",
            return_value={"aggregate_sigma": 0.0, "per_job": {}},
        ):
            self.assertIsNone(harness._score_plan(plan))

    def test_placement_floor_materializes_recommended_launch_config(self):
        harness = KoiAgentHarness.__new__(KoiAgentHarness)
        harness._current_tick = 7
        plan = Plan(
            tick=7,
            actions=[PlanAction(job_id="job_1", type=ActionType.DEFER)],
        )
        recommendation = {
            "chosen": [
                {
                    "job_id": "job_1",
                    "type": "place",
                    "ladder": [
                        {
                            "role": "aggregate",
                            "env": ["reserved", "aws", "r1", "z1", "H100"],
                            "config": {
                                "instance_type": "p5",
                                "gpu_count": 1,
                                "tp": 1,
                                "pp": 1,
                            },
                            "n_replicas": 1,
                            "mechanism_id": "M_test",
                        }
                    ],
                }
            ]
        }
        with (
            patch.object(harness, "_materialize_launch_configs") as materialize,
            patch.object(harness, "_validate_ladder"),
        ):
            harness._apply_placement_floor(
                plan,
                {"job_1": "waiting"},
                recommendation,
                object(),
            )

        materialize.assert_called_once()
        self.assertEqual(plan.actions[0].type, ActionType.PLACE)

    def test_cli_defaults_environment_validation_and_build_wiring(self):
        with patch.dict(os.environ, {}, clear=True):
            defaults = runner.parse_args([])

        self.assertEqual(defaults.temperature, 0.2)
        self.assertEqual(defaults.wall_clock_sec, 240.0)
        self.assertEqual(defaults.k_p, 1)
        self.assertEqual(defaults.k_max, 4)
        self.assertEqual(defaults.surrogate_call_budget, 100)
        self.assertEqual(defaults.surrogate_lower_quantile, 0.05)

        with patch.dict(
            os.environ,
            {
                "KOI_SURROGATE_CALL_BUDGET": "37",
                "KOI_SURROGATE_LOWER_QUANTILE": "0.2",
            },
            clear=True,
        ):
            environment = runner.parse_args([])
        self.assertEqual(environment.surrogate_call_budget, 37)
        self.assertEqual(environment.surrogate_lower_quantile, 0.2)

        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            runner.parse_args(["--surrogate-call-budget", "0"])

        with patch.dict(os.environ, {}, clear=True):
            args = runner.parse_args(
                [
                    "--user-id",
                    "user_1",
                    "--api-key",
                    "test-key",
                    "--surrogate-call-budget",
                    "17",
                ]
            )
        with (
            patch.object(agent_tools, "configure_surrogate_call_budget") as configure,
            patch.object(runner, "PostgresClient", side_effect=RuntimeError("stop after wiring")),
            self.assertRaisesRegex(RuntimeError, "stop after wiring"),
        ):
            runner.build_runner(args)
        configure.assert_called_once_with(17)

    def test_root_prompt_prescribes_one_pipeline_or_one_shortcut(self):
        harness = KoiAgentHarness.__new__(KoiAgentHarness)
        harness.k_max = 4
        harness.wall_clock_sec = 240.0

        prompt = harness.build_root_prompt(tick=9)
        specialist_prompt = SpecialistRunner._default_prompt("job_1", {}, {})

        self.assertEqual(prompt.count("run_job_specialists()"), 1)
        self.assertEqual(prompt.count("build_scored_candidates("), 1)
        self.assertEqual(prompt.count("jointly_select_placements("), 1)
        self.assertEqual(prompt.count("plan_tick()"), 1)
        self.assertIn("You are the final decision-maker", prompt)
        self.assertIn("recommendation for you to inspect", prompt)
        self.assertIn("do not call any primary workflow stage before or after it", prompt)
        self.assertIn("Inspect `priority`", prompt)
        self.assertIn("scored['diagnostics']", prompt)
        self.assertIn("scored['budget_limited']", prompt)
        self.assertIn("Online candidates are SLO-complete, full-service", prompt)
        self.assertIn("Any allowed batch partial candidate", prompt)
        self.assertIn("meets_target and served_fraction", prompt)
        self.assertIn("include an explicit DEFER action", prompt)
        self.assertNotIn("solver owns the pick", prompt)
        self.assertNotIn("FINAL_VAR(plan_tick())", prompt)
        self.assertNotIn("predicted_y", specialist_prompt)
        self.assertNotIn("predicted_sigma", specialist_prompt)

    def test_tick_summary_and_surrogate_trace_keep_compact_provenance(self):
        ctx = types.SimpleNamespace(
            tick=5,
            state_history=[],
            evidence_rows=[],
            deploy_acks=[],
            error=None,
            state_durations_ms={},
            validated_plan=None,
        )
        budget_status = {
            "limit": 23,
            "calls_executed": 7,
            "cache_hits": 2,
            "budget_rejections": 0,
            "remaining": 16,
        }
        with tempfile.TemporaryDirectory() as log_dir:
            logger = DebugLogger(log_dir, trace="no-llm", run_id="compact-controls")
            with patch.object(
                debug_logging.agent_tools,
                "get_surrogate_budget_status",
                return_value=budget_status,
            ):
                logger.persist_runner_tick(
                    ctx,
                    types.SimpleNamespace(trace=types.SimpleNamespace(events=[])),
                    types.SimpleNamespace(calls=[]),
                )
            logger.persist_surrogate_prediction(
                {
                    "schema_version": 3,
                    "scenario": "mean",
                    "method": ["AIC_Direct"],
                    "components": {},
                    "backends": {},
                },
                tick=5,
            )
            events = [json.loads(line) for line in logger.events_path.read_text().splitlines()]

        self.assertEqual(events[0]["kind"], "tick_summary")
        self.assertEqual(events[0]["payload"]["surrogate_budget"], budget_status)
        self.assertEqual(events[-1]["kind"], "surrogate_prediction")
        self.assertEqual(events[-1]["payload"]["method"], ["AIC_Direct"])

    def test_run_manifest_is_non_secret_and_records_budget_and_revision(self):
        with patch.dict(os.environ, {}, clear=True):
            args = runner.parse_args(["--api-key", "do-not-log"])
        budget_status = {"limit": 100, "calls_executed": 0, "remaining": 100}
        with tempfile.TemporaryDirectory() as log_dir:
            logger = DebugLogger(log_dir, trace="no-llm", run_id="manifest-controls")
            with (
                patch.object(
                    runner.agent_tools,
                    "get_surrogate_budget_status",
                    return_value=budget_status,
                ),
                patch.object(runner, "_source_revision", return_value="466d69f"),
                patch.object(runner, "_source_dirty", return_value=True),
            ):
                runner.emit_run_manifest(logger, args)
            event = json.loads(logger.events_path.read_text().splitlines()[0])

        self.assertEqual(event["kind"], "run_manifest")
        self.assertEqual(event["payload"]["source_revision"], "466d69f")
        self.assertIs(event["payload"]["source_dirty"], True)
        self.assertEqual(event["payload"]["surrogate_budget"], budget_status)
        self.assertEqual(event["payload"]["config"]["surrogate_call_budget"], 100)
        self.assertEqual(event["payload"]["config"]["surrogate_lower_quantile"], 0.05)
        self.assertNotIn("do-not-log", json.dumps(event))
        self.assertNotIn("api_key", json.dumps(event))

    def test_source_dirty_is_bounded_and_fails_open(self):
        completed = types.SimpleNamespace(returncode=0, stdout=" M src/agent/agent.py\n")
        with patch.object(runner.subprocess, "run", return_value=completed) as run:
            self.assertIs(runner._source_dirty(), True)
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        self.assertNotIn("shell", run.call_args.kwargs)

        with patch.object(runner.subprocess, "run", side_effect=OSError):
            self.assertIsNone(runner._source_revision())
            self.assertIsNone(runner._source_dirty())


if __name__ == "__main__":
    unittest.main()
