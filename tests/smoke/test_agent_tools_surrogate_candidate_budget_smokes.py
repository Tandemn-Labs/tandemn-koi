import unittest
from contextlib import ExitStack
from unittest.mock import patch

from src.agent.tools import agent_tools


class _DRO:
    @staticmethod
    def compute_dro_band(y_hat):
        return {
            objective: {
                "point": float(value),
                "lower": float(value) - 1.0,
                "upper": float(value) + 1.0,
            }
            for objective, value in y_hat.items()
        }


class _RecordingSurrogate:
    def __init__(self):
        self.calls = []

    def compose_prediction_with_trace(self, **kwargs):
        self.calls.append(kwargs)
        y_hat = {
            "p99_ttft_ms": 20.0,
            "p99_tpot_ms": 2.0,
            "throughput_token_per_sec": 200.0,
        }
        return (
            y_hat,
            {"queue": {"depth": 1.0}},
            {
                "schema_version": 3,
                "method": list(kwargs["method"]),
                "scenario": kwargs["scenario"],
                "raw": {"y_hat": y_hat},
            },
        )


class _SlowLoop:
    @staticmethod
    def get_sss_wt(*_args, **_kwargs):
        return {"cost_per_token": 0.0}


def _rank(env, instance_type, *, n_replicas=1):
    return {
        "role": "aggregate",
        "env": env.split("|") if isinstance(env, str) else list(env),
        "config": {
            "instance_type": instance_type,
            "gpu_count": 1,
            "tp": 1,
            "pp": 1,
        },
        "n_replicas": n_replicas,
    }


class SurrogateCandidateBudgetSmokeTests(unittest.TestCase):
    def setUp(self):
        self.saved_budget = agent_tools.SURROGATE_CALL_BUDGET
        self.saved_context = {
            name: getattr(agent_tools._CTX, name)
            for name in (
                "candidate_graph",
                "cluster_snapshot",
                "dro",
                "resource_map",
                "slow_loop",
                "specialist_runner",
                "surrogate",
                "validated_budget_book",
            )
        }
        agent_tools.configure_surrogate_call_budget(100)
        agent_tools.reset_tick_caches()

    def tearDown(self):
        agent_tools.reset_tick_caches()
        agent_tools.configure_surrogate_call_budget(self.saved_budget)
        for name, value in self.saved_context.items():
            setattr(agent_tools._CTX, name, value)

    def _bind_prediction_fakes(self, surrogate=None):
        surrogate = surrogate or _RecordingSurrogate()
        agent_tools._CTX.candidate_graph = object()
        agent_tools._CTX.dro = _DRO()
        agent_tools._CTX.surrogate = surrogate
        agent_tools._CTX.resource_map = None
        return surrogate

    def test_budget_configuration_cache_hits_and_rejections_are_accounted(self):
        surrogate = self._bind_prediction_fakes()
        agent_tools.configure_surrogate_call_budget(1)

        first = agent_tools._predict_outcome_core({"model_id": "model-a"}, {"type": "online"})
        first["y_hat"]["p99_ttft_ms"] = 999.0
        first["v_hat"]["queue"]["depth"] = 999.0
        cached = agent_tools._predict_outcome_core({"model_id": "model-a"}, {"type": "online"})
        with self.assertRaises(agent_tools.SurrogateBudgetExceeded):
            agent_tools._predict_outcome_core({"model_id": "model-b"}, {"type": "online"})

        status = agent_tools.get_surrogate_budget_status()
        self.assertEqual(cached["y_hat"]["p99_ttft_ms"], 20.0)
        self.assertEqual(cached["v_hat"]["queue"]["depth"], 1.0)
        self.assertEqual(len(surrogate.calls), 1)
        self.assertEqual(status["limit"], 1)
        self.assertEqual(status["calls_executed"], 1)
        self.assertEqual(status["cache_hits"], 1)
        self.assertEqual(status["budget_rejections"], 1)
        self.assertEqual(status["remaining"], 0)

        agent_tools.reset_tick_caches()
        reset_status = agent_tools.get_surrogate_budget_status()
        self.assertEqual(reset_status["limit"], 1)
        self.assertEqual(reset_status["calls_executed"], 0)
        self.assertEqual(reset_status["cache_hits"], 0)
        self.assertEqual(reset_status["budget_rejections"], 0)

    def test_finalization_helper_is_not_exposed_to_the_llm(self):
        self.assertNotIn("stamp_plan_predictions", agent_tools.all_callables())

    def test_finalization_bypasses_search_cap_and_stamps_decision_metadata(self):
        surrogate = self._bind_prediction_fakes()
        agent_tools.configure_surrogate_call_budget(1)
        agent_tools._predict_outcome_core({"model_id": "search"}, {"type": "online"})
        with self.assertRaises(agent_tools.SurrogateBudgetExceeded):
            agent_tools._predict_outcome_core({"model_id": "rejected"}, {"type": "online"})

        snapshot = type(
            "Snapshot",
            (),
            {
                "pending_jobs_summary": lambda self: [
                    {
                        "job_id": "job-final",
                        "job_features": {
                            "model_id": "final",
                            "type": "online",
                            "request_arrival_rate": 2.0,
                            "output_len_tokens_avg": 50.0,
                            "target_p99_ttft_ms": 100.0,
                            "target_p99_tpot_ms": 10.0,
                        },
                    }
                ],
                "slo_thresholds": lambda self, _job_id: {
                    "p99_ttft_ms": 100.0,
                    "p99_tpot_ms": 10.0,
                },
            },
        )()
        final_rank = _rank("reserved|aws|r1|z1|H100", "p5")
        final_rank["config"]["_arrival_share_rps"] = 0.75
        typed_plan = agent_tools.Plan(
            tick=0,
            actions=[
                agent_tools.PlanAction(
                    job_id="job-final",
                    type=agent_tools.ActionType.PLACE,
                    ladder=[agent_tools.RankSpec.from_dict(final_rank)],
                    target_tps=100.0,
                )
            ],
        )
        plan = agent_tools.stamp_plan_predictions(typed_plan, snapshot)

        status = agent_tools.get_surrogate_budget_status()
        lineage = plan.actions[0].ladder[0].prediction_lineage
        self.assertEqual(status["calls_executed"], 1)
        self.assertEqual(status["finalization_calls"], 1)
        self.assertEqual(status["budget_rejections"], 1)
        self.assertNotIn("dro_band", lineage)
        self.assertEqual(
            lineage["decision_dro_band"]["p99_ttft_ms"],
            {"point": 20.0, "lower": 19.0, "upper": 21.0},
        )
        self.assertEqual(
            lineage["decision_required_objectives"],
            ["p99_tpot_ms", "p99_ttft_ms", "throughput_token_per_sec"],
        )
        self.assertEqual(
            [call["method"] for call in surrogate.calls],
            [("AIC_Direct",), ("AIC_DynoSim",)],
        )
        self.assertEqual(surrogate.calls[-1]["job_features"]["request_arrival_rate"], 0.75)

    def test_public_prediction_defaults_direct_and_queue_verification_is_explicit(self):
        surrogate = self._bind_prediction_fakes()

        online = {"job_config": {"model_id": "online"}, "job_features": {"type": "online"}}
        agent_tools.predict_outcome(online)
        agent_tools.predict_outcome(online, queue_aware=True)
        agent_tools.predict_outcome(
            {"job_config": {"model_id": "batch"}, "job_features": {"type": "batch"}}
        )

        self.assertEqual(
            [call["method"] for call in surrogate.calls],
            [("AIC_Direct",), ("AIC_DynoSim",), ("AIC_Direct",)],
        )

    def test_plan_tick_rationale_separates_candidate_and_budget_limits(self):
        with (
            patch.object(agent_tools, "build_user_envelopes", return_value={}),
            patch.object(agent_tools, "get_priority", return_value=[]),
            patch.object(agent_tools, "allocate_budget_book", return_value={}),
            patch.object(agent_tools, "validate_budget_book", return_value={"ok": True}),
            patch.object(agent_tools, "run_job_specialists", return_value={}),
            patch.object(
                agent_tools,
                "build_scored_candidates",
                return_value={
                    "candidates": [],
                    "exhausted": {"physical-job": "no fit"},
                    "budget_limited": {"limited-job": "cap"},
                    "diagnostics": {},
                },
            ),
            patch.object(
                agent_tools,
                "jointly_select_placements",
                return_value={"chosen": [], "deferred": [], "objective": 0.0},
            ),
            patch.object(agent_tools, "check_feasibility", return_value={"feasible": True}),
        ):
            plan = agent_tools.plan_tick()

        self.assertIn("candidate_exhausted=['physical-job']", plan["tick_rationale"])
        self.assertIn("budget_limited=['limited-job']", plan["tick_rationale"])

    def test_frame_and_composite_report_budget_exhaustion_distinctly(self):
        env = "reserved|aws|r1|z1|H100"
        ranks = [_rank(env, "p5-a"), _rank(env, "p5-b")]
        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(
                agent_tools,
                "size_ladder",
                side_effect=agent_tools.SurrogateBudgetExceeded("search cap reached"),
            ),
        ):
            frame = agent_tools._score_one_frame("job", "user", "slice", ranks[0], {})
            composite = agent_tools._score_composite("job", "user", "slice", ranks, {})

        self.assertEqual(frame["diag"]["status"], "budget_exhausted")
        self.assertEqual(composite["diag"]["status"], "budget_exhausted")

        def sized(scored_ranks, _features):
            return {
                "ranks": scored_ranks,
                "meets_target": True,
                "target_tps": 1.0,
                "achieved_tps": 1.0,
                "per_rank": [],
            }

        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(agent_tools, "size_ladder", side_effect=sized),
            patch.object(agent_tools, "check_feasibility", return_value={"feasible": True}),
            patch.object(
                agent_tools,
                "compute_sigma",
                side_effect=agent_tools.SurrogateBudgetExceeded("score cap reached"),
            ),
        ):
            frame = agent_tools._score_one_frame("job", "user", "slice", ranks[0], {})
            composite = agent_tools._score_composite("job", "user", "slice", ranks, {})

        self.assertEqual(frame["diag"]["status"], "budget_exhausted")
        self.assertEqual(composite["diag"]["status"], "budget_exhausted")

    def test_online_under_slo_and_incomplete_predictions_are_not_candidates(self):
        env = "reserved|aws|r1|z1|H100"
        rank = _rank(env, "p5")
        features = {
            "type": "online",
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 10.0,
        }
        base = {
            "ranks": [rank],
            "meets_target": False,
            "target_tps": 100.0,
            "achieved_tps": 100.0,
        }
        cases = (
            (
                {
                    "prediction_received": True,
                    "prediction_complete": False,
                    "slo_ok": False,
                },
                "prediction_incomplete",
            ),
            (
                {
                    "prediction_received": True,
                    "prediction_complete": True,
                    "slo_ok": False,
                },
                "under_slo",
            ),
            (
                {
                    "prediction_received": True,
                    "prediction_complete": True,
                    "slo_ok": True,
                },
                "under_target",
            ),
        )

        with patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"):
            for per_rank, expected in cases:
                with self.subTest(expected=expected):
                    sized = {**base, "per_rank": [per_rank]}
                    with patch.object(agent_tools, "size_ladder", return_value=sized):
                        result = agent_tools._score_one_frame(
                            "job", "user", "slice", rank, features
                        )
                    self.assertIsNone(result["candidate"])
                    self.assertEqual(result["diag"]["status"], expected)
                    self.assertEqual(
                        result.get("composite_eligible", False), expected == "under_target"
                    )

    def test_successful_candidate_does_not_run_automatic_stress_diagnostic(self):
        rank = _rank("reserved|aws|r1|z1|H100", "p5")
        sized = {
            "ranks": [rank],
            "meets_target": True,
            "target_tps": 100.0,
            "achieved_tps": 100.0,
            "per_rank": [],
        }
        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(agent_tools, "size_ladder", return_value=sized),
            patch.object(agent_tools, "check_feasibility", return_value={"feasible": True}),
            patch.object(
                agent_tools,
                "compute_sigma",
                return_value={"per_job": {"job": {"sigma": 1.0}}},
            ),
            patch.object(agent_tools, "_attach_peak_multiturn_stress") as stress,
        ):
            result = agent_tools._score_one_frame(
                "job", "user", "slice", rank, {"type": "batch", "multi_turn_ratio": 1.0}
            )

        self.assertIsNotNone(result["candidate"])
        stress.assert_not_called()

    def test_specialist_results_are_cached_and_returned_as_deep_copies(self):
        class Runner:
            def __init__(self):
                self.calls = []

            def run_many(self, jobs, budget_book, max_workers):
                self.calls.append(list(jobs))
                return [
                    {"job_id": job_id, "reasoning": "original", "ladder": []} for job_id in jobs
                ]

        runner = Runner()
        agent_tools._CTX.specialist_runner = runner
        agent_tools._CTX.validated_budget_book = {
            "job_budgets": {
                "job-1": {"slice_id": "job-1"},
                "job-2": {"slice_id": "job-2"},
            }
        }

        with (
            patch.object(agent_tools, "get_pending_jobs", return_value=[{"job_id": "job-1"}]),
            patch.object(agent_tools, "get_active_jobs", return_value=[{"job_id": "job-2"}]),
        ):
            first = agent_tools.run_job_specialists()
            first["job-1"]["reasoning"] = "mutated"
            second = agent_tools.run_job_specialists()
            with_active = agent_tools.run_job_specialists(include_active=True)
            with_active["job-2"]["reasoning"] = "mutated"
            with_active_again = agent_tools.run_job_specialists(include_active=True)

        self.assertEqual(runner.calls, [["job-1"], ["job-1", "job-2"]])
        self.assertEqual(second["job-1"]["reasoning"], "original")
        self.assertEqual(with_active_again["job-2"]["reasoning"], "original")

    def _candidate_patches(self, resources, specs, pending, scorer, *, job_type="batch"):
        def job_features(_snapshot, jid):
            features = {"type": job_type, "model_id": f"model-{jid}"}
            if job_type == "online":
                features.update(
                    {
                        "target_p99_ttft_ms": 100.0,
                        "target_p99_tpot_ms": 10.0,
                    }
                )
            return features

        stack = ExitStack()
        stack.enter_context(patch.object(agent_tools, "get_resource_map", return_value=resources))
        stack.enter_context(patch.object(agent_tools, "instance_catalog", return_value=specs))
        stack.enter_context(patch.object(agent_tools, "get_pending_jobs", return_value=pending))
        stack.enter_context(patch.object(agent_tools, "_snapshot", return_value=object()))
        stack.enter_context(
            patch.object(
                agent_tools,
                "_job_features_for",
                side_effect=job_features,
            )
        )
        stack.enter_context(patch.object(agent_tools, "_model_num_heads", return_value=8))
        stack.enter_context(patch.object(agent_tools, "_score_one_frame", side_effect=scorer))
        return stack

    def test_round_robin_preserves_completed_candidates_and_marks_budget_skips(self):
        env_a = "reserved|aws|r1|a|H100"
        env_b = "reserved|aws|r1|b|H100"
        resources = {
            env_b: {"free": 1, "gpu_type": "H100"},
            env_a: {"free": 1, "gpu_type": "H100"},
        }
        specs = {
            env_a: {"pool-a": {"gpus_per_instance": 1, "free_instances": 1}},
            env_b: {"pool-b": {"gpus_per_instance": 1, "free_instances": 1}},
        }
        pending = [{"job_id": "job-2"}, {"job_id": "job-1"}]
        specialist_results = {
            "job-1": {"ladder": [_rank(env_a, "llm-1")]},
            "job-2": {"ladder": [_rank(env_b, "llm-2")]},
        }
        calls = []

        def score(jid, _user_id, _slice_id, rank, _features):
            marker = rank["config"]["instance_type"]
            calls.append((jid, marker))
            if len(calls) == 4:
                return {
                    "candidate": None,
                    "meets_target": False,
                    "diag": {
                        "status": "budget_exhausted",
                        "reason": "cap",
                        "instance_type": marker,
                    },
                }
            return {
                "candidate": {"job_id": jid, "marker": marker},
                "meets_target": True,
                "diag": {"status": "ok", "reason": None, "instance_type": marker},
            }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with self._candidate_patches(resources, specs, pending, score):
            result = agent_tools.build_scored_candidates({}, specialist_results)

        self.assertEqual(
            calls,
            [
                ("job-1", "llm-1"),
                ("job-2", "llm-2"),
                ("job-1", "pool-a"),
                ("job-2", "pool-a"),
            ],
        )
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(set(result["budget_limited"]), {"job-1", "job-2"})
        self.assertEqual(result["exhausted"], {})
        self.assertEqual(
            [diag["status"] for diag in result["diagnostics"]["job-1"]],
            ["ok", "ok", "budget_skipped"],
        )
        self.assertEqual(
            [diag["status"] for diag in result["diagnostics"]["job-2"]],
            ["ok", "budget_exhausted", "budget_skipped"],
        )

    def test_all_same_gpu_environments_are_stable_and_candidate_results_are_cached(self):
        env_a = "reserved|aws|r1|a|H100"
        env_b = "reserved|aws|r1|b|H100"
        resources = {
            env_b: {"free": 1, "gpu_type": "H100"},
            env_a: {"free": 2, "gpu_type": "H100"},
        }
        specs = {
            env_a: {
                "z-pool": {"gpus_per_instance": 1, "free_instances": 1},
                "a-pool": {"gpus_per_instance": 1, "free_instances": 1},
            },
            env_b: {"b-pool": {"gpus_per_instance": 1, "free_instances": 1}},
        }
        calls = []

        def score(jid, _user_id, _slice_id, rank, _features):
            marker = ("|".join(rank["env"]), rank["config"]["instance_type"])
            calls.append(marker)
            return {
                "candidate": {"job_id": jid, "marker": list(marker)},
                "meets_target": True,
                "diag": {"status": "ok", "reason": None},
            }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with self._candidate_patches(resources, specs, [{"job_id": "job-1"}], score):
            first = agent_tools.build_scored_candidates({}, {})
            first["candidates"][0]["marker"] = ["mutated"]
            second = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(
            calls,
            [(env_a, "a-pool"), (env_a, "z-pool"), (env_b, "b-pool")],
        )
        self.assertEqual(len(second["candidates"]), 3)
        self.assertNotEqual(second["candidates"][0]["marker"], ["mutated"])

    def test_slo_safe_online_partial_frames_remain_available_to_composites(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 2, "gpu_type": "H100"}}
        specs = {
            env: {
                "pool-a": {"gpus_per_instance": 1, "free_instances": 1},
                "pool-b": {"gpus_per_instance": 1, "free_instances": 1},
            }
        }
        composite_orders = []

        def score(jid, _user_id, _slice_id, rank, _features):
            marker = rank["config"]["instance_type"]
            return {
                "candidate": None,
                "composite_eligible": True,
                "meets_target": False,
                "diag": {
                    "status": "under_target",
                    "reason": "safe partial",
                    "achieved_tps": 60.0 if marker == "pool-a" else 40.0,
                },
            }

        def score_composite(jid, _user_id, _slice_id, ranks, _features):
            composite_orders.append([rank["config"]["instance_type"] for rank in ranks])
            return {
                "candidate": {"job_id": jid, "marker": "composite"},
                "meets_target": True,
                "diag": {"status": "ok", "reason": None},
            }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(
                resources,
                specs,
                [{"job_id": "job-1"}],
                score,
                job_type="online",
            ),
            patch.object(agent_tools, "_score_composite", side_effect=score_composite),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(composite_orders, [["pool-a", "pool-b"]])
        self.assertEqual(result["candidates"], [{"job_id": "job-1", "marker": "composite"}])
        self.assertEqual(result["exhausted"], {})

    def test_composite_budget_exhaustion_marks_unattempted_composite_jobs_limited(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 2, "gpu_type": "H100"}}
        specs = {
            env: {
                "pool-a": {"gpus_per_instance": 1, "free_instances": 1},
                "pool-b": {"gpus_per_instance": 1, "free_instances": 1},
            }
        }

        def score(jid, _user_id, _slice_id, _rank, _features):
            return {
                "candidate": None,
                "composite_eligible": True,
                "meets_target": False,
                "diag": {
                    "status": "under_target",
                    "reason": "safe partial",
                    "achieved_tps": 50.0,
                },
            }

        def exhaust_composite(*_args, **_kwargs):
            return {
                "candidate": None,
                "meets_target": False,
                "diag": {"status": "budget_exhausted", "reason": "cap"},
            }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(
                resources,
                specs,
                [{"job_id": "job-1"}, {"job_id": "job-2"}],
                score,
                job_type="online",
            ),
            patch.object(agent_tools, "_score_composite", side_effect=exhaust_composite),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(result["exhausted"], {})
        self.assertEqual(set(result["budget_limited"]), {"job-1", "job-2"})
        self.assertIn(
            "budget_skipped",
            [diag["status"] for diag in result["diagnostics"]["job-2"]],
        )

    def test_skipped_full_service_composite_marks_batch_job_with_partial_candidate_limited(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 2, "gpu_type": "H100"}}
        specs = {
            env: {
                "pool-a": {"gpus_per_instance": 1, "free_instances": 1},
                "pool-b": {"gpus_per_instance": 1, "free_instances": 1},
            }
        }

        def score(jid, _user_id, _slice_id, rank, _features):
            return {
                "candidate": {
                    "job_id": jid,
                    "marker": rank["config"]["instance_type"],
                },
                "meets_target": False,
                "diag": {
                    "status": "ok",
                    "reason": None,
                    "achieved_tps": 50.0,
                },
            }

        def exhaust_composite(*_args, **_kwargs):
            return {
                "candidate": None,
                "meets_target": False,
                "diag": {"env": "composite", "status": "budget_exhausted", "reason": "cap"},
            }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(
                resources,
                specs,
                [{"job_id": "job-1"}, {"job_id": "job-2"}],
                score,
                job_type="batch",
            ),
            patch.object(agent_tools, "_score_composite", side_effect=exhaust_composite),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(set(result["budget_limited"]), {"job-1", "job-2"})
        self.assertEqual(len(result["candidates"]), 4)

    def test_job_prediction_requires_every_rank_to_succeed(self):
        action = agent_tools.PlanAction(
            job_id="job",
            type=agent_tools.ActionType.PLACE,
            ladder=[
                agent_tools.RankSpec.from_dict(_rank("reserved|aws|r1|z1|H100", "pool-a")),
                agent_tools.RankSpec.from_dict(_rank("reserved|aws|r1|z1|H100", "pool-b")),
            ],
        )
        predictions = iter(
            (
                {"y_hat": {"throughput_token_per_sec": 100.0}},
                {"y_hat": {}},
            )
        )
        with (
            patch.object(
                agent_tools,
                "_rank_prediction_payload",
                return_value={"job_config": {}, "job_features": {"type": "batch"}},
            ),
            patch.object(
                agent_tools,
                "_predict_outcome_core",
                side_effect=lambda *_a, **_k: next(predictions),
            ),
        ):
            self.assertEqual(agent_tools._compose_job_y_hat(action, {"type": "batch"}), {})

    def test_size_ladder_tries_specialist_replica_count_first(self):
        env = "reserved|aws|r1|z1|H100"
        self.assertNotEqual(
            agent_tools._rank_shape_key(_rank(env, "p5", n_replicas=1)),
            agent_tools._rank_shape_key(_rank(env, "p5", n_replicas=3)),
        )

        class ResourceMap:
            @staticmethod
            def resources_summary():
                return {env: {"free": 4, "gpu_type": "H100"}}

            @staticmethod
            def rank_allocation_summary(rank, resources=None):
                return {
                    "allocation_kind": "gpu",
                    "instance_type": rank.config["instance_type"],
                    "gpus_per_unit": 1,
                    "price_per_unit_hour": None,
                    "capacity_per_replica": 1,
                    "free_capacity_gpus": 4,
                    "engine_gpus": 1,
                }

        tried = []

        def predict(config, _features, **kwargs):
            tried.append((kwargs["method"], config["dp"]))
            return {
                "y_hat": {
                    "p99_ttft_ms": 20.0,
                    "p99_tpot_ms": 2.0,
                    "throughput_token_per_sec": 200.0,
                }
            }

        agent_tools._CTX.resource_map = ResourceMap()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.candidate_graph = object()
        agent_tools._CTX.dro = _DRO()
        with (
            patch.object(
                agent_tools,
                "_rank_prediction_payload",
                side_effect=lambda rank, features: {
                    "job_config": {"dp": rank.n_replicas},
                    "job_features": features,
                },
            ),
            patch.object(agent_tools, "_predict_outcome_core", side_effect=predict),
        ):
            result = agent_tools.size_ladder(
                [_rank(env, "p5", n_replicas=3)],
                {
                    "type": "online",
                    "output_len_tokens_avg": 100.0,
                    "target_p99_ttft_ms": 100.0,
                    "target_p99_tpot_ms": 10.0,
                },
                target_tps=100.0,
            )

        self.assertTrue(result["meets_target"])
        self.assertEqual(
            tried,
            [
                (("AIC_Direct",), 3),
                (("AIC_DynoSim",), 3),
            ],
        )

    def test_size_ladder_direct_screens_before_dynosim_and_can_find_smaller_dp(self):
        env = "reserved|aws|r1|z1|H100"

        class ResourceMap:
            @staticmethod
            def resources_summary():
                return {env: {"free": 4, "gpu_type": "H100"}}

            @staticmethod
            def rank_allocation_summary(rank, resources=None):
                return {
                    "allocation_kind": "gpu",
                    "instance_type": rank.config["instance_type"],
                    "gpus_per_unit": 1,
                    "price_per_unit_hour": None,
                    "capacity_per_replica": 1,
                    "free_capacity_gpus": 4,
                    "engine_gpus": 1,
                }

        calls = []
        direct_throughput = {1: 20.0, 2: 120.0, 3: 60.0, 4: 80.0}

        def predict(config, _features, **kwargs):
            method = kwargs["method"]
            dp = config["dp"]
            calls.append((method, dp))
            if method == ("AIC_Direct",):
                return {"y_hat": {"throughput_token_per_sec": direct_throughput[dp]}}
            return {
                "y_hat": {
                    "p99_ttft_ms": 20.0 if dp == 2 else 200.0,
                    "p99_tpot_ms": 2.0 if dp == 2 else 20.0,
                    "throughput_token_per_sec": direct_throughput[dp],
                }
            }

        agent_tools._CTX.resource_map = ResourceMap()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.candidate_graph = object()
        agent_tools._CTX.dro = _DRO()
        with (
            patch.object(
                agent_tools,
                "_rank_prediction_payload",
                side_effect=lambda rank, features: {
                    "job_config": {"dp": rank.n_replicas},
                    "job_features": features,
                },
            ),
            patch.object(agent_tools, "_predict_outcome_core", side_effect=predict),
        ):
            result = agent_tools.size_ladder(
                [_rank(env, "p5", n_replicas=3)],
                {
                    "type": "online",
                    "output_len_tokens_avg": 100.0,
                    "target_p99_ttft_ms": 100.0,
                    "target_p99_tpot_ms": 10.0,
                },
                target_tps=100.0,
            )

        self.assertTrue(result["meets_target"])
        self.assertEqual(result["ranks"][0]["n_replicas"], 2)
        self.assertEqual(
            calls,
            [
                (("AIC_Direct",), 3),
                (("AIC_Direct",), 4),
                (("AIC_DynoSim",), 4),
                (("AIC_Direct",), 1),
                (("AIC_Direct",), 2),
                (("AIC_DynoSim",), 2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
