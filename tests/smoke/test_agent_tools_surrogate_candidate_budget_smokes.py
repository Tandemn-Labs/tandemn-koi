import math
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

    @staticmethod
    def dro_chance_constraint(**_kwargs):
        return {"_any_violated": 0.0}


class _RecordingSurrogate:
    def __init__(self, y_hats=None):
        self.calls = []
        self.y_hats = y_hats

    def compose_prediction_with_trace(self, **kwargs):
        self.calls.append(kwargs)
        y_hat = dict(
            self.y_hats[len(self.calls) - 1]
            if self.y_hats is not None
            else {
                "p99_ttft_ms": 20.0,
                "p99_tpot_ms": 2.0,
                "throughput_token_per_sec": 200.0,
            }
        )
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


class _SigmaSlowLoop:
    def __init__(self):
        self.typical_ranges = dict(agent_tools.DEFAULT_TYPICAL_RANGES)

    @staticmethod
    def get_sss_wt(*_args, **_kwargs):
        return {"throughput_token_per_sec": 1.0, "cost_per_token": 0.0}

    @staticmethod
    def get_sss_z_star_t(*_args, **_kwargs):
        return dict(agent_tools.DEFAULT_COLD_START_Z_STAR)

    @staticmethod
    def get_sss_eig_incentive_t():
        return 0.0

    @staticmethod
    def get_sss_lambda_switch():
        return 0.0

    @staticmethod
    def get_sss_radius_dro():
        return 0.0


class _Tchebycheff:
    @staticmethod
    def compute_tchebycheff(**_kwargs):
        return -0.25


class _EIG:
    @staticmethod
    def compute_eig(**_kwargs):
        return 0.0


class _SwitchCost:
    @staticmethod
    def compute_switch_cost(**_kwargs):
        return type("Bundle", (), {"total": 0.0})()


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
        self.saved_admission_mode = agent_tools.PARTIAL_ONLINE_ADMISSION_MODE
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
        agent_tools.configure_partial_online_admission("off")
        agent_tools.reset_tick_caches()

    def tearDown(self):
        agent_tools.reset_tick_caches()
        agent_tools.configure_surrogate_call_budget(self.saved_budget)
        agent_tools.configure_partial_online_admission(self.saved_admission_mode)
        for name, value in self.saved_context.items():
            setattr(agent_tools._CTX, name, value)

    def test_sizing_queue_assessment_reports_selection_usage_and_token_pressure(self):
        assessment = agent_tools._prediction_assessment(
            {
                "point_capacity_tps": 1000.0,
                "per_rank": [{"n_replicas": 2, "base_p99_ttft_ms": 20.0}],
            },
            {
                "type": "online",
                "request_arrival_rate": 2.0,
                "isl_token_avg": 200.0,
                "isl_token_max": 800.0,
                "osl_token_avg": 100.0,
                "osl_token_max": 300.0,
            },
        )

        self.assertEqual(assessment["queue_utilization"], 0.2)
        self.assertEqual(assessment["queue_token_work_pressure"], 0.6)
        self.assertEqual(assessment["queue_tail_token_work_pressure"], 2.2)
        self.assertEqual(assessment["queue_state"], "stable")
        self.assertTrue(assessment["queue_affects_selection"])

    def _bind_prediction_fakes(self, surrogate=None):
        surrogate = surrogate or _RecordingSurrogate()
        agent_tools._CTX.candidate_graph = object()
        agent_tools._CTX.dro = _DRO()
        agent_tools._CTX.surrogate = surrogate
        agent_tools._CTX.resource_map = None
        return surrogate

    def _size_online(self, predict, *, target_tps=100.0, include_tpot_target=True):
        env = "reserved|aws|r1|z1|H100"

        class ResourceMap:
            @staticmethod
            def resources_summary():
                return {env: {"free": 1, "gpu_type": "H100"}}

            @staticmethod
            def rank_allocation_summary(rank, resources=None):
                return {
                    "allocation_kind": "gpu",
                    "instance_type": rank.config["instance_type"],
                    "gpus_per_unit": 1,
                    "price_per_unit_hour": None,
                    "capacity_per_replica": 1,
                    "free_capacity_gpus": 1,
                    "engine_gpus": 1,
                }

        features = {
            "type": "online",
            "request_arrival_rate": 10.0,
            "output_len_tokens_avg": 1.0,
            "target_p99_ttft_ms": 100.0,
        }
        if include_tpot_target:
            features["target_p99_tpot_ms"] = 10.0
        agent_tools._CTX.resource_map = ResourceMap()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.candidate_graph = object()
        agent_tools._CTX.dro = _DRO()
        with (
            patch.object(
                agent_tools,
                "_rank_prediction_payload",
                side_effect=lambda rank, rank_features, **_kwargs: {
                    "job_config": {"dp": rank.n_replicas},
                    "job_features": dict(rank_features),
                },
            ),
            patch.object(agent_tools, "_predict_outcome_core", side_effect=predict),
        ):
            result = agent_tools.size_ladder(
                [_rank(env, "p5")],
                features,
                target_tps=target_tps,
            )
        return env, features, result

    def _sigma_plan(self, *instance_types):
        ranks = []
        for instance_type in instance_types or ("p5",):
            rank = _rank("reserved|aws|r1|z1|H100", instance_type)
            rank["mechanism_id"] = "M_test"
            ranks.append(agent_tools.RankSpec.from_dict(rank))
        return agent_tools.Plan(
            tick=0,
            actions=[
                agent_tools.PlanAction(
                    job_id="job-final",
                    type=agent_tools.ActionType.PLACE,
                    ladder=ranks,
                    target_tps=1.0,
                )
            ],
        )

    def _sigma_patches(self):
        snapshot = type(
            "Snapshot",
            (),
            {
                "active_jobs_summary": lambda self: [],
                "pending_jobs_summary": lambda self: [
                    {
                        "job_id": "job-final",
                        "job_features": {
                            "type": "batch",
                            "total_token_budget": 3600.0,
                            "deadline_hours": 1.0,
                            "headroom_factor": 1.0,
                        },
                    }
                ],
            },
        )()
        stack = ExitStack()
        stack.enter_context(patch.object(agent_tools, "_snapshot", return_value=snapshot))
        stack.enter_context(patch.object(agent_tools, "get_pending_jobs", return_value=[]))
        stack.enter_context(patch.object(agent_tools, "get_priority", return_value=[]))
        stack.enter_context(patch.object(agent_tools, "_materialize_ladder", return_value=object()))
        stack.enter_context(patch.object(agent_tools, "_materialize_chain_list", return_value=[]))
        stack.enter_context(patch.object(agent_tools._CTX, "slow_loop", _SigmaSlowLoop()))
        stack.enter_context(patch.object(agent_tools._CTX, "tchebycheff_module", _Tchebycheff()))
        stack.enter_context(patch.object(agent_tools._CTX, "eig_module", _EIG()))
        stack.enter_context(patch.object(agent_tools._CTX, "switchcost_module", _SwitchCost()))
        stack.enter_context(patch.object(agent_tools._CTX, "mechanism_registry", object()))
        stack.enter_context(patch.object(agent_tools._CTX, "confidence_service", object()))
        stack.enter_context(patch.object(agent_tools._CTX, "evidence_store", object()))
        return stack

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

    def test_partial_admission_mode_is_guarded_hidden_and_survives_tick_reset(self):
        self.assertNotIn("configure_partial_online_admission", agent_tools.all_callables())
        self.assertIn("get_partial_online_admission_status", agent_tools.all_callables())
        with self.assertRaises(TypeError):
            agent_tools.configure_partial_online_admission(True)
        with self.assertRaises(ValueError):
            agent_tools.configure_partial_online_admission("ADVISORY")

        agent_tools.configure_partial_online_admission("advisory")
        agent_tools.reset_tick_caches()

        self.assertEqual(
            agent_tools.get_partial_online_admission_status(),
            {
                "mode": "advisory",
                "prediction_basis": "aic_direct_point",
                "queue_model": "none",
                "queue_slo_verified": False,
                "searches": 0,
                "queue_aware_probes": 0,
                "safe_probes": 0,
                "admissions": 0,
                "truncated_searches": 0,
            },
        )

    def test_finalization_helper_is_not_exposed_to_the_llm(self):
        self.assertNotIn("stamp_plan_predictions", agent_tools.all_callables())
        self.assertNotIn("compute_sigma_for_commit", agent_tools.all_callables())

    def test_public_compute_sigma_uses_search_budget(self):
        surrogate = self._bind_prediction_fakes()
        agent_tools.configure_surrogate_call_budget(1)

        with self._sigma_patches():
            result = agent_tools.compute_sigma(self._sigma_plan("p5"))
            with self.assertRaises(agent_tools.SurrogateBudgetExceeded):
                agent_tools.compute_sigma(self._sigma_plan("p4"))

        status = agent_tools.get_surrogate_budget_status()
        self.assertEqual(
            set(result),
            {"per_job", "aggregate_sigma", "swap_count", "unserved_penalty"},
        )
        self.assertIn("job-final", result["per_job"])
        self.assertEqual(len(surrogate.calls), 1)
        self.assertEqual(status["calls_executed"], 1)
        self.assertEqual(status["finalization_calls"], 0)
        self.assertEqual(status["budget_rejections"], 1)
        self.assertEqual(status["remaining"], 0)

    def test_commit_sigma_reuses_same_tick_direct_point_with_finalization_accounting(self):
        surrogate = self._bind_prediction_fakes()
        agent_tools.configure_surrogate_call_budget(1)
        plan = self._sigma_plan("p5")

        with self._sigma_patches():
            search_result = agent_tools.compute_sigma(plan)
            commit_result = agent_tools.compute_sigma_for_commit(plan)

        status = agent_tools.get_surrogate_budget_status()
        self.assertEqual(set(commit_result), set(search_result))
        self.assertEqual(set(commit_result["per_job"]), {"job-final"})
        self.assertEqual(len(surrogate.calls), 1)
        self.assertEqual(status["calls_executed"], 1)
        self.assertEqual(status["cache_hits"], 1)
        self.assertEqual(status["finalization_calls"], 1)
        self.assertEqual(status["budget_rejections"], 0)
        self.assertEqual(status["remaining"], 0)

    def test_commit_sigma_rejects_a_partial_rank_prediction_with_the_same_schema(self):
        full_prediction = {
            "p99_ttft_ms": 20.0,
            "p99_tpot_ms": 2.0,
            "throughput_token_per_sec": 200.0,
        }
        surrogate = self._bind_prediction_fakes(_RecordingSurrogate([full_prediction, {}]))
        agent_tools.configure_surrogate_call_budget(0)

        with self._sigma_patches():
            result = agent_tools.compute_sigma_for_commit(self._sigma_plan("p5-a", "p5-b"))

        status = agent_tools.get_surrogate_budget_status()
        self.assertEqual(
            set(result),
            {"per_job", "aggregate_sigma", "swap_count", "unserved_penalty"},
        )
        self.assertEqual(result["per_job"], {})
        self.assertEqual(len(surrogate.calls), 2)
        self.assertEqual(status["calls_executed"], 0)
        self.assertEqual(status["finalization_calls"], 2)
        self.assertEqual(status["remaining"], 0)

    def test_final_score_charges_only_residual_unserved_fraction(self):
        self._bind_prediction_fakes()
        partial = self._sigma_plan("p5")
        partial.actions[0].served_fraction = 0.4
        pending = [{"job_id": "job-final"}]
        priority = [{"job_id": "job-final", "priority_score": 10.0}]

        with (
            self._sigma_patches(),
            patch.object(agent_tools, "get_pending_jobs", return_value=pending),
            patch.object(agent_tools, "get_priority", return_value=priority),
        ):
            result = agent_tools.compute_sigma_for_commit(partial)
            legacy = agent_tools.compute_sigma_for_commit(self._sigma_plan("p5"))
            malformed = self._sigma_plan("p5")
            malformed.actions[0].served_fraction = float("nan")
            with self.assertRaisesRegex(ValueError, "served_fraction"):
                agent_tools.compute_sigma_for_commit(malformed)

        self.assertEqual(result["unserved_penalty"], 6.0)
        self.assertAlmostEqual(
            result["aggregate_sigma"],
            result["per_job"]["job-final"]["sigma"] - 6.0,
        )
        self.assertEqual(legacy["unserved_penalty"], 0.0)

    def test_final_score_rederives_point_capacity_coverage_for_exact_ladder(self):
        prediction = {
            "p99_ttft_ms": 20.0,
            "p99_tpot_ms": 2.0,
            "throughput_token_per_sec": 0.25,
        }
        self._bind_prediction_fakes(_RecordingSurrogate([prediction]))
        plan = self._sigma_plan("p5")
        plan.actions[0].served_fraction = 1.0  # stale candidate metadata
        plan.actions[0].prediction_assessment = {
            "basis": "aic_direct_point",
            "kind": "point",
            "status": "success",
            "queue_slo_verified": False,
        }

        with (
            self._sigma_patches(),
            patch.object(agent_tools, "get_pending_jobs", return_value=[{"job_id": "job-final"}]),
            patch.object(
                agent_tools,
                "get_priority",
                return_value=[{"job_id": "job-final", "priority_score": 10.0}],
            ),
        ):
            result = agent_tools.compute_sigma_for_commit(plan)

        self.assertEqual(result["unserved_penalty"], 7.5)

    def test_final_score_reclassifies_exploration_when_prediction_recovers(self):
        surrogate = self._bind_prediction_fakes()
        plan = self._sigma_plan("p5")
        plan.actions[0].prediction_assessment = {
            "basis": "aic_direct_point",
            "kind": "exploratory",
            "status": "unsupported_prediction",
            "queue_slo_verified": False,
        }
        pending = [{"job_id": "job-final"}]
        priority = [{"job_id": "job-final", "priority_score": 10.0}]

        with (
            self._sigma_patches(),
            patch.object(agent_tools, "get_pending_jobs", return_value=pending),
            patch.object(agent_tools, "get_priority", return_value=priority),
        ):
            result = agent_tools.compute_sigma_for_commit(plan)

        score = result["per_job"]["job-final"]
        self.assertNotIn("scoring_mode", score)
        self.assertEqual(plan.actions[0].prediction_assessment["kind"], "point")
        self.assertEqual(plan.actions[0].prediction_assessment["status"], "success")
        self.assertEqual(plan.actions[0].service_class, "supported")
        self.assertEqual(result["unserved_penalty"], 0.0)
        self.assertEqual(len(surrogate.calls), 1)

    def test_final_score_downgrades_a_point_candidate_when_prediction_disappears(self):
        self._bind_prediction_fakes()
        plan = self._sigma_plan("p5")
        plan.actions[0].prediction_assessment = {
            "basis": "aic_direct_point",
            "kind": "point",
            "status": "success",
            "queue_slo_verified": False,
        }

        with (
            self._sigma_patches(),
            patch.object(agent_tools, "_compose_job_y_hat", return_value={}),
        ):
            result = agent_tools.compute_sigma_for_commit(plan)

        self.assertEqual(result["per_job"]["job-final"]["scoring_mode"], "exploration_only")
        self.assertEqual(
            plan.actions[0].prediction_assessment["status"],
            "prediction_incomplete",
        )
        self.assertIsNone(plan.actions[0].served_fraction)
        self.assertIsNone(plan.actions[0].achieved_tps)
        self.assertEqual(plan.actions[0].service_class, "exploratory")

    def test_emergency_recovery_commit_score_keeps_work_conserving_floor(self):
        self._bind_prediction_fakes()
        plan = self._sigma_plan("p5")
        action = plan.actions[0]
        action.type = agent_tools.ActionType.SWAP
        action.queue_state = "unstable"
        action.rehabilitation_status = "critical"
        action.rehabilitation_reasons = ["zero_throughput", "queue_critical"]
        action.service_class = "partial"
        action.prediction_assessment = {
            "basis": "aic_direct_point",
            "kind": "point",
            "status": "success",
            "queue_slo_verified": False,
            "selection_mode": "emergency_recovery",
        }

        with self._sigma_patches():
            result = agent_tools.compute_sigma_for_commit(plan)

        self.assertEqual(result["per_job"]["job-final"]["sigma"], 1e-9)
        self.assertEqual(action.swap_gain_over_keep, 1e-9)
        self.assertEqual(action.service_class, "partial")

    def test_swap_score_uses_observed_keep_baseline(self):
        self._bind_prediction_fakes()
        plan = self._sigma_plan("p5")
        action = plan.actions[0]
        action.type = agent_tools.ActionType.SWAP
        action.rehabilitation_status = "degraded"
        action.rehabilitation_reasons = ["throughput_shortfall"]

        class ThroughputScore:
            @staticmethod
            def compute_tchebycheff(**kwargs):
                return float(kwargs["y_hat"]["throughput_token_per_sec"]) - 1.0

        snapshot = type(
            "Snapshot",
            (),
            {
                "active_jobs_summary": lambda self: [
                    {
                        "job_id": "job-final",
                        "job_features": {
                            "type": "batch",
                            "total_token_budget": 3600.0,
                            "deadline_hours": 1.0,
                            "headroom_factor": 1.0,
                        },
                        "health": {
                            "observed": {"throughput_token_per_sec": 0.25},
                        },
                    }
                ],
                "pending_jobs_summary": lambda self: [],
            },
        )()

        with (
            self._sigma_patches(),
            patch.object(agent_tools, "_snapshot", return_value=snapshot),
            patch.object(agent_tools._CTX, "tchebycheff_module", ThroughputScore()),
        ):
            result = agent_tools.compute_sigma_for_commit(plan)

        score = result["per_job"]["job-final"]
        self.assertEqual(score["keep_baseline_sigma"], -0.75)
        self.assertEqual(score["swap_gain_over_keep"], score["sigma"] + 0.75)
        self.assertEqual(result["aggregate_sigma"], score["swap_gain_over_keep"])

    def test_final_score_never_downgrades_physical_no_fit_to_exploration(self):
        self._bind_prediction_fakes()
        plan = self._sigma_plan("p5")
        plan.actions[0].prediction_assessment = {
            "basis": "aic_direct_point",
            "kind": "exploratory",
            "status": "unsupported_prediction",
            "queue_slo_verified": False,
        }

        with (
            self._sigma_patches(),
            patch.object(
                agent_tools,
                "_compose_job_y_hat",
                side_effect=agent_tools.SurrogateMemoryNoFit("requested model does not fit"),
            ),
            self.assertRaisesRegex(agent_tools.SurrogateMemoryNoFit, "does not fit"),
        ):
            agent_tools.compute_sigma_for_commit(plan)

    def test_online_sigma_uses_peak_scenario_for_search_and_finalization(self):
        self._bind_prediction_fakes()
        calls = []

        def compose(_action, _features, **kwargs):
            calls.append(kwargs)
            return {
                "p99_ttft_ms": 20.0,
                "p99_tpot_ms": 2.0,
                "throughput_token_per_sec": 100.0,
            }

        online_features = {
            "type": "online",
            "request_arrival_rate": 1.0,
            "output_len_tokens_avg": 1.0,
            "headroom_factor": 1.0,
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 10.0,
        }
        with (
            self._sigma_patches(),
            patch.object(agent_tools, "_job_features_for", return_value=online_features),
            patch.object(agent_tools, "_compose_job_y_hat", side_effect=compose),
        ):
            agent_tools.compute_sigma(self._sigma_plan("p5"))
            agent_tools.compute_sigma_for_commit(self._sigma_plan("p5"))

        self.assertEqual(
            [(call["scenario"], call["finalization"]) for call in calls],
            [("peak", False), ("peak", True)],
        )
        self.assertEqual(
            [call["method"] for call in calls],
            [("AIC_Direct",), ("AIC_Direct",)],
        )

    def test_composed_online_score_preserves_original_request_rate(self):
        rank = _rank("reserved|aws|r1|z1|H100", "p5")
        rank["config"]["_arrival_share_rps"] = 0.375
        rank["rank_traffic_share"] = 0.25
        action = agent_tools.PlanAction(
            job_id="job",
            type=agent_tools.ActionType.PLACE,
            ladder=[agent_tools.RankSpec.from_dict(rank)],
            target_tps=100.0,
        )
        calls = []

        def predict(job_config, job_features, **kwargs):
            calls.append((job_config, job_features, kwargs))
            return {
                "y_hat": {
                    "p99_ttft_ms": 20.0,
                    "p99_tpot_ms": 2.0,
                    "throughput_token_per_sec": 37.5,
                }
            }

        agent_tools._CTX.resource_map = None
        with patch.object(agent_tools, "_predict_outcome_core", side_effect=predict):
            agent_tools._compose_job_y_hat(
                action,
                {
                    "type": "online",
                    "request_arrival_rate": 9.0,
                    "output_len_tokens_avg": 50.0,
                },
                method=("AIC_DynoSim",),
                scenario="peak",
            )

        config, features, kwargs = calls[0]
        self.assertNotIn("_arrival_share_rps", config)
        self.assertEqual(features["request_arrival_rate"], 9.0)
        self.assertEqual(kwargs["method"], ("AIC_Direct",))
        self.assertEqual(kwargs["scenario"], "peak")

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
        final_rank["rank_traffic_share"] = 0.6
        typed_plan = agent_tools.Plan(
            tick=0,
            actions=[
                agent_tools.PlanAction(
                    job_id="job-final",
                    type=agent_tools.ActionType.PLACE,
                    ladder=[agent_tools.RankSpec.from_dict(final_rank)],
                    target_tps=100.0,
                    admitted_tps=60.0,
                    achieved_tps=60.0,
                    unmet_tps=40.0,
                    meets_target=False,
                    served_fraction=0.6,
                    admission_mode="advisory",
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
            ["throughput_token_per_sec"],
        )
        self.assertEqual(lineage["prediction_semantics"]["basis"], "aic_direct_point")
        self.assertIs(lineage["prediction_semantics"]["queue_slo_verified"], False)
        self.assertEqual(lineage["queue_shadow"]["status"], "stable")
        self.assertFalse(lineage["queue_shadow"]["affects_selection"])
        self.assertEqual(
            [call["method"] for call in surrogate.calls],
            [("AIC_Direct",), ("AIC_Direct",)],
        )
        self.assertEqual(surrogate.calls[-1]["job_features"]["request_arrival_rate"], 2.0)
        self.assertEqual(
            lineage["partial_admission"],
            {
                "mode": "advisory",
                "requested_tps": 100.0,
                "admitted_tps": 60.0,
                "served_fraction": 0.6,
                "enforced": False,
            },
        )

    def test_public_prediction_ignores_queue_aware_and_always_uses_direct(self):
        surrogate = self._bind_prediction_fakes()

        online = {"job_config": {"model_id": "online"}, "job_features": {"type": "online"}}
        agent_tools.predict_outcome(online)
        agent_tools.predict_outcome(
            {"job_config": {"model_id": "online-queue"}, "job_features": {"type": "online"}},
            queue_aware=True,
        )
        agent_tools.predict_outcome(
            {"job_config": {"model_id": "batch"}, "job_features": {"type": "batch"}}
        )

        self.assertEqual(
            [call["method"] for call in surrogate.calls],
            [("AIC_Direct",), ("AIC_Direct",), ("AIC_Direct",)],
        )

    def test_priority_uses_nested_class_and_workload_signals(self):
        jobs = [
            {
                "job_id": "job-low",
                "kind": "online",
                "job_features": {
                    "user_priority": 1.0,
                    "priority_class": "LOW",
                    "type": "batch",
                    "deadline_pressure": 0.1,
                    "slo_margin": 0.5,
                    "queue_age_ticks": 2,
                    "recent_failures": 0,
                },
            },
            {
                "job_id": "job-high",
                "kind": "batch",
                "job_features": {
                    "user_priority": 2.0,
                    "priority_class": "HIGH",
                    "workload_type": "online",
                    "deadline_pressure": 0.5,
                    "slo_margin": -0.25,
                    "queue_age_ticks": 4,
                    "recent_failures": 2,
                },
            },
        ]
        with (
            patch.object(agent_tools, "get_pending_jobs", return_value=jobs),
            patch.object(agent_tools, "get_active_jobs", return_value=[]),
        ):
            priorities = agent_tools.get_priority()

        by_id = {entry["job_id"]: entry for entry in priorities}
        self.assertEqual([entry["job_id"] for entry in priorities], ["job-high", "job-low"])
        self.assertEqual(
            by_id["job-high"]["signals"],
            {
                "user_priority": 2.0,
                "priority_class": 2.0,
                "is_online": 1.0,
                "deadline_pressure": 0.5,
                "slo_margin_deficit": 0.25,
                "queue_age_ticks": 4.0,
                "recent_failures": 2.0,
            },
        )
        self.assertEqual(by_id["job-high"]["priority_score"], 53.5)
        self.assertEqual(by_id["job-low"]["signals"]["priority_class"], 0.0)
        self.assertEqual(by_id["job-low"]["signals"]["is_online"], 0.0)
        self.assertEqual(by_id["job-low"]["priority_score"], 11.5)

    def test_priority_handles_malformed_values_and_ties_stably(self):
        malformed_features = {
            "user_priority": "not-a-number",
            "priority_class": "URGENT",
            "workload_type": "batch",
            "deadline_pressure": {},
            "slo_margin": [],
            "queue_age_ticks": "NaN",
            "recent_failures": None,
        }
        jobs = [
            {"job_id": "job-b", "job_features": dict(malformed_features)},
            {"job_id": "job-a", "job_features": dict(malformed_features)},
            {
                "job_id": "job-numeric",
                "job_features": {
                    "user_priority": 0,
                    "priority_class": 2.5,
                    "workload_type": "batch",
                },
            },
        ]
        with (
            patch.object(agent_tools, "get_pending_jobs", return_value=jobs),
            patch.object(agent_tools, "get_active_jobs", return_value=[]),
        ):
            priorities = agent_tools.get_priority()

        self.assertEqual(
            [entry["job_id"] for entry in priorities],
            ["job-numeric", "job-a", "job-b"],
        )
        by_id = {entry["job_id"]: entry for entry in priorities}
        self.assertEqual(by_id["job-numeric"]["signals"]["priority_class"], 2.5)
        self.assertEqual(by_id["job-numeric"]["priority_score"], 25.0)
        self.assertEqual(by_id["job-a"]["priority_score"], 10.0)
        self.assertTrue(all(math.isfinite(entry["priority_score"]) for entry in priorities))

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

    def test_resource_budget_rejects_before_surrogate_sizing(self):
        rank = _rank("reserved|aws|r1|z1|H100", "p5")
        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(
                agent_tools,
                "_candidate_budget_errors",
                return_value=["slice capacity exceeded"],
            ),
            patch.object(agent_tools, "size_ladder") as size_ladder,
        ):
            result = agent_tools._score_one_frame("job", "user", "slice", rank, {})

        self.assertIsNone(result["candidate"])
        self.assertEqual(result["diag"]["status"], "resource_budget")
        size_ladder.assert_not_called()

    def test_point_estimate_conditions_are_not_hard_online_rejections(self):
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

        agent_tools.configure_partial_online_admission("off")
        for per_rank, previous_status in cases:
            with self.subTest(previous_status=previous_status):
                sized = {**base, "per_rank": [per_rank]}
                self.assertIsNone(agent_tools._online_sizing_rejection(sized, features))

    def test_point_capacity_candidate_retains_base_latency_miss(self):
        env = "reserved|aws|r1|z1|H100"
        rank = _rank(env, "p5")
        features = {
            "type": "online",
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 10.0,
        }
        sized = {
            "ranks": [rank],
            "meets_target": False,
            "target_tps": 100.0,
            "achieved_tps": 100.0,
            "per_rank": [
                {
                    "n_replicas": 1,
                    "prediction_received": True,
                    "prediction_complete": True,
                    "slo_ok": False,
                }
            ],
        }

        agent_tools.configure_partial_online_admission("advisory")
        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(agent_tools, "size_ladder", return_value=sized),
            patch.object(agent_tools, "check_feasibility", return_value={"feasible": True}),
            patch.object(
                agent_tools,
                "compute_sigma",
                return_value={"per_job": {"job": {"sigma": -1.0, "pr_slo_dro": 1.0}}},
            ),
        ):
            result = agent_tools._score_one_frame("job", "user", "slice", rank, features)

        candidate = result["candidate"]
        self.assertIsNotNone(candidate)
        self.assertFalse(candidate["meets_target"])
        self.assertTrue(result["diag"]["slo_risk"])
        self.assertNotIn("admitted_tps", candidate)
        self.assertNotIn("admission_mode", candidate)
        self.assertIn("base-latency target missed", result["diag"]["reason"])

    def test_swap_scores_gain_before_feasibility_and_uses_observed_baseline(self):
        env = "reserved|aws|r1|z1|H100"
        rank = _rank(env, "p5")
        features = {
            "type": "online",
            "request_arrival_rate": 0.01,
            "osl_token_avg": 10,
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 10.0,
            "_active_health": {
                "status": "degraded",
                "reasons": ["ttft_breach", "throughput_shortfall"],
                "observed": {
                    "throughput_token_per_sec": 10.0,
                    "p99_ttft_ms": 200.0,
                    "p99_tpot_ms": 5.0,
                },
            },
        }
        sized = {
            "ranks": [rank],
            "meets_target": False,
            "target_tps": 100.0,
            "achieved_tps": 100.0,
            "point_capacity_tps": 100.0,
            "base_latency_within_target": False,
            "per_rank": [
                {
                    "n_replicas": 1,
                    "prediction_received": True,
                    "prediction_complete": True,
                    "base_p99_ttft_ms": 150.0,
                    "base_p99_tpot_ms": 5.0,
                    "slo_ok": False,
                }
            ],
        }

        def feasibility(plan):
            action = plan["actions"][0]
            self.assertEqual(action["swap_gain_over_keep"], 2.0)
            return {"feasible": True}

        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(agent_tools, "size_ladder", return_value=sized),
            patch.object(agent_tools, "check_feasibility", side_effect=feasibility),
            patch.object(
                agent_tools,
                "compute_sigma",
                return_value={
                    "per_job": {
                        "job": {
                            "sigma": 1.0,
                            "keep_baseline_sigma": -1.0,
                            "swap_gain_over_keep": 2.0,
                        }
                    }
                },
            ),
        ):
            result = agent_tools._score_one_frame(
                "job", "user", "slice", rank, features, action_type="swap"
            )

        self.assertIsNotNone(result["candidate"])
        self.assertEqual(result["candidate"]["swap_gain_over_keep"], 2.0)

    def test_point_capacity_partial_frame_and_composite_are_candidates(self):
        env = "reserved|aws|r1|z1|H100"
        ranks = [_rank(env, "p5-a"), _rank(env, "p5-b")]
        features = {
            "type": "online",
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 10.0,
        }

        def sized(scored_ranks, _features):
            per_rank = [
                {
                    "n_replicas": rank.get("n_replicas", 1),
                    "served_tps": 50.0 / len(scored_ranks),
                    "prediction_received": True,
                    "prediction_complete": True,
                    "slo_ok": True,
                    "partial_admission": True,
                }
                for rank in scored_ranks
            ]
            return {
                "ranks": scored_ranks,
                "meets_target": False,
                "target_tps": 100.0,
                "achieved_tps": 50.0,
                "unmet_tps": 50.0,
                "partial_online_admission": True,
                "admission_mode": "advisory",
                "partial_search_truncated": True,
                "per_rank": per_rank,
            }

        agent_tools.configure_partial_online_admission("advisory")
        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(agent_tools, "size_ladder", side_effect=sized),
            patch.object(agent_tools, "check_feasibility", return_value={"feasible": True}),
            patch.object(
                agent_tools,
                "compute_sigma",
                return_value={"per_job": {"job": {"sigma": -0.25}}},
            ),
        ):
            frame = agent_tools._score_one_frame("job", "user", "slice", ranks[0], features)
            composite = agent_tools._score_composite(
                "job",
                "user",
                "slice",
                ranks,
                features,
            )

        for result in (frame, composite):
            candidate = result["candidate"]
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate["achieved_tps"], 50.0)
            self.assertEqual(candidate["unmet_tps"], 50.0)
            self.assertFalse(candidate["meets_target"])
            self.assertEqual(candidate["served_fraction"], 0.5)
            self.assertNotIn("admitted_tps", candidate)
            self.assertNotIn("admission_mode", candidate)
            self.assertEqual(candidate["prediction_assessment"]["basis"], "aic_direct_point")
            self.assertIs(candidate["prediction_assessment"]["queue_slo_verified"], False)
            self.assertTrue(result["diag"]["partial_search_truncated"])

    def test_point_capacity_under_target_without_latency_target_is_a_candidate(self):
        rank = _rank("reserved|aws|r1|z1|H100", "p5")
        sized = {
            "ranks": [rank],
            "meets_target": False,
            "target_tps": 100.0,
            "achieved_tps": 50.0,
            "partial_online_admission": True,
            "per_rank": [
                {
                    "n_replicas": 1,
                    "served_tps": 50.0,
                    "prediction_received": True,
                    "prediction_complete": True,
                    "slo_ok": True,
                }
            ],
        }
        agent_tools.configure_partial_online_admission("advisory")
        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(agent_tools, "size_ladder", return_value=sized),
            patch.object(agent_tools, "check_feasibility", return_value={"feasible": True}),
            patch.object(
                agent_tools,
                "compute_sigma",
                return_value={"per_job": {"job": {"sigma": -0.5}}},
            ),
        ):
            result = agent_tools._score_one_frame(
                "job",
                "user",
                "slice",
                rank,
                {"type": "online"},
            )

        self.assertIsNotNone(result["candidate"])
        self.assertNotIn("admitted_tps", result["candidate"])
        self.assertNotIn("admission_mode", result["candidate"])
        self.assertEqual(
            result["candidate"]["prediction_assessment"]["basis"],
            "aic_direct_point",
        )
        self.assertEqual(result["diag"]["status"], "ok")

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
                "job",
                "user",
                "slice",
                rank,
                {
                    "type": "batch",
                    "multi_turn_ratio": 1.0,
                    "target_p99_ttft_ms": 100.0,
                    "target_p99_tpot_ms": 10.0,
                },
            )

        self.assertIsNotNone(result["candidate"])
        self.assertIsNone(result["candidate"]["target_p99_ttft_ms"])
        self.assertIsNone(result["candidate"]["target_p99_tpot_ms"])
        self.assertNotIn("admission_mode", result["candidate"])
        stress.assert_not_called()

    def test_batch_partial_with_latency_benchmark_fields_is_not_online_admission(self):
        rank = _rank("reserved|aws|r1|z1|H100", "p5")
        sized = {
            "ranks": [rank],
            "meets_target": False,
            "target_tps": 100.0,
            "achieved_tps": 50.0,
            "per_rank": [
                {
                    "n_replicas": 1,
                    "prediction_received": True,
                    "prediction_complete": True,
                    "slo_ok": True,
                }
            ],
        }
        features = {
            "type": "batch",
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 10.0,
        }
        with (
            patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
            patch.object(agent_tools, "size_ladder", return_value=sized),
            patch.object(agent_tools, "check_feasibility", return_value={"feasible": True}),
            patch.object(
                agent_tools,
                "compute_sigma",
                return_value={"per_job": {"job": {"sigma": -0.5}}},
            ),
        ):
            result = agent_tools._score_one_frame("job", "user", "slice", rank, features)

        candidate = result["candidate"]
        self.assertIsNotNone(candidate)
        self.assertIsNone(candidate["target_p99_ttft_ms"])
        self.assertIsNone(candidate["target_p99_tpot_ms"])
        self.assertNotIn("admitted_tps", candidate)
        self.assertNotIn("admission_mode", candidate)
        self.assertFalse(result["diag"]["slo_risk"])

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
            patch.object(
                agent_tools,
                "get_active_jobs",
                return_value=[{"job_id": "job-2", "health": {"rehabilitation_eligible": True}}],
            ),
        ):
            first = agent_tools.run_job_specialists(include_active=False)
            first["job-1"]["reasoning"] = "mutated"
            second = agent_tools.run_job_specialists(include_active=False)
            with_active = agent_tools.run_job_specialists()
            with_active["job-2"]["reasoning"] = "mutated"
            with_active_again = agent_tools.run_job_specialists()

        self.assertEqual(runner.calls, [["job-1"], ["job-1", "job-2"]])
        self.assertEqual(second["job-1"]["reasoning"], "original")
        self.assertEqual(with_active_again["job-2"]["reasoning"], "original")

    def test_specialists_skip_deployment_blocked_jobs(self):
        class Runner:
            def __init__(self):
                self.jobs = []

            def run_many(self, jobs, budget_book, max_workers):
                self.jobs = list(jobs)
                return [{"job_id": job_id, "ladder": []} for job_id in jobs]

        runner = Runner()
        agent_tools._CTX.specialist_runner = runner
        job_ids = ["fresh", "pending", "exhausted", "retry-ready", "active-ready"]
        agent_tools._CTX.validated_budget_book = {
            "job_budgets": {job_id: {"slice_id": job_id} for job_id in job_ids}
        }
        pending = [
            {"job_id": "fresh"},
            {"job_id": "pending", "deployment_status": "deployment_pending"},
            {
                "job_id": "exhausted",
                "deployment_status": "deployment_not_materialized",
                "deployment_retry_exhausted": True,
            },
            {
                "job_id": "retry-ready",
                "deployment_status": "deployment_not_materialized",
                "deployment_retry_allowed": True,
            },
        ]
        active = [
            {
                "job_id": "active-ready",
                "health": {"rehabilitation_eligible": True},
            }
        ]

        with (
            patch.object(agent_tools, "get_pending_jobs", return_value=pending),
            patch.object(agent_tools, "get_active_jobs", return_value=active),
        ):
            agent_tools.run_job_specialists()

        self.assertEqual(runner.jobs, ["fresh", "retry-ready", "active-ready"])

    def test_specialist_mechanism_survives_normalization_and_applicability(self):
        raw = {
            **_rank("reserved|aws|r1|a|H100", "p5"),
            "mechanism_id": "M_specialist",
        }

        normalized = agent_tools._normalize_candidate_rank(raw)
        with patch.object(
            agent_tools,
            "get_applicable_mechanisms",
            return_value=[
                {"mechanism_id": "M_default"},
                {"mechanism_id": "M_specialist"},
            ],
        ):
            selected = agent_tools._applicable_mechanism_id(normalized, {})

        self.assertEqual(normalized["mechanism_id"], "M_specialist")
        self.assertEqual(selected, "M_specialist")

    def test_specialist_shape_suppresses_duplicate_generated_frame(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 1}}}
        specialist_rank = {
            **_rank(env, "p5"),
            "mechanism_id": "M_specialist",
        }
        frames = []

        def score(jid, _user_id, _slice_id, rank, _features):
            frames.append(rank)
            return {
                "candidate": {"job_id": jid},
                "meets_target": True,
                "diag": {"status": "ok", "reason": None},
            }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with self._candidate_patches(resources, specs, [{"job_id": "job-1"}], score):
            agent_tools.build_scored_candidates(
                {},
                {"job-1": {"ladder": [specialist_rank]}},
            )

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["mechanism_id"], "M_specialist")
        self.assertEqual(frames[0]["proposal_source"], "specialist")

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

    def test_candidate_round_robin_starts_with_highest_priority_job(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 1}}}
        pending = [{"job_id": "job-low"}, {"job_id": "job-high"}]
        calls = []

        def score(jid, _user_id, _slice_id, _rank, _features):
            calls.append(jid)
            return {
                "candidate": {"job_id": jid},
                "meets_target": True,
                "diag": {"status": "ok", "reason": None},
            }

        budget = {
            "job_budgets": {
                "job-low": {
                    "slice_id": "low",
                    "priority_score": 1.0,
                    "env_budget": {env: 1},
                    "pool_budget": {env: {"p5": 1}},
                },
                "job-high": {
                    "slice_id": "high",
                    "priority_score": 100.0,
                    "env_budget": {env: 1},
                    "pool_budget": {env: {"p5": 1}},
                },
            }
        }
        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with self._candidate_patches(resources, specs, pending, score):
            agent_tools.build_scored_candidates(budget, {})

        self.assertEqual(calls, ["job-high", "job-low"])

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

    def test_generated_frames_enumerate_fixed_replica_options_within_budget(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 4, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 4}}}
        replicas = []

        def score(jid, _user_id, _slice_id, rank, _features):
            replicas.append(rank["n_replicas"])
            return {
                "candidate": {"job_id": jid, "marker": rank["n_replicas"]},
                "meets_target": True,
                "diag": {"status": "ok", "reason": None},
            }

        budget = {
            "job_budgets": {
                "job-1": {
                    "slice_id": "slice-1",
                    "env_budget": {env: 3},
                    "pool_budget": {env: {"p5": 3}},
                }
            }
        }
        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with self._candidate_patches(resources, specs, [{"job_id": "job-1"}], score):
            agent_tools.build_scored_candidates(budget, {})

        self.assertEqual(replicas, [1, 2, 3])

    def test_generated_frames_cover_each_pool_before_replica_expansion(self):
        env_a = "reserved|aws|r1|a|H100"
        env_b = "ondemand|azure|r2|b|MI300X"
        resources = {
            env_a: {"free": 4, "gpu_type": "H100"},
            env_b: {"free": 1, "gpu_type": "MI300X"},
        }
        specs = {
            env_a: {"p5": {"gpus_per_instance": 1, "free_instances": 4}},
            env_b: {"nd-mi300": {"gpus_per_instance": 1, "free_instances": 1}},
        }
        frames = []

        def score(jid, _user_id, _slice_id, rank, _features):
            marker = (rank["config"]["instance_type"], rank["n_replicas"])
            frames.append(marker)
            return {
                "candidate": {"job_id": jid, "marker": marker},
                "meets_target": True,
                "diag": {"status": "ok", "reason": None},
            }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        budget = {
            "job_budgets": {
                "job-1": {
                    "slice_id": "slice-1",
                    "env_budget": {env_a: 4, env_b: 1},
                    "pool_budget": {env_a: {"p5": 4}, env_b: {"nd-mi300": 1}},
                }
            }
        }
        with self._candidate_patches(resources, specs, [{"job_id": "job-1"}], score):
            agent_tools.build_scored_candidates(budget, {})

        self.assertEqual(set(frames[:2]), {("p5", 1), ("nd-mi300", 1)})
        self.assertEqual(frames[2], ("p5", 2))

    def test_pending_deployment_capacity_reserves_whole_instances(self):
        env = "reserved|aws|r1|a|H100"
        specs = {
            env: {"p5": {"gpus_per_instance": 8, "free_instances": 4}},
        }
        pending = {
            "job_id": "active-job",
            "deployment_status": "deployment_pending",
            "last_requested_shapes": [
                {
                    "env": env.split("|"),
                    "instance_type": "p5",
                    "gpu_count": 4,
                    "tp": 4,
                    "pp": 1,
                    "n_replicas": 2,
                }
            ],
        }

        with (
            patch.object(agent_tools, "get_pending_jobs", return_value=[]),
            patch.object(agent_tools, "get_active_jobs", return_value=[pending]),
        ):
            reserved = agent_tools._pending_deployment_capacity(specs)

        self.assertEqual(reserved[("gpu", env)], 16)
        self.assertEqual(reserved[("pool", env, "p5")], 2)

    def test_pending_deployment_capacity_removes_unmaterialized_pool_from_search(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 8, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 8, "free_instances": 1}}}
        pending_job = {"job_id": "new-job"}
        active_job = {
            "job_id": "active-job",
            "deployment_status": "deployment_pending",
            "deployment_action_type": "swap",
            "last_requested_shapes": [
                {
                    "env": env.split("|"),
                    "instance_type": "p5",
                    "gpu_count": 4,
                    "tp": 4,
                    "pp": 1,
                    "n_replicas": 1,
                }
            ],
        }
        calls = []

        def score(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("reserved pool must not reach candidate scoring")

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(resources, specs, [pending_job], score),
            patch.object(agent_tools, "get_active_jobs", return_value=[active_job]),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(calls, [])
        self.assertIn("no policy-valid candidate frame", result["exhausted"]["new-job"])

    def test_unhealthy_active_job_generates_swap_candidates(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"new-pool": {"gpus_per_instance": 1, "free_instances": 1}}}
        active = {
            "job_id": "active-job",
            "deployment_status": "deployment_not_materialized",
            "health": {"rehabilitation_eligible": True},
            "current_ladder": [
                {
                    "shape_json": {
                        "env": env.split("|"),
                        "instance_type": "old-pool",
                        "tp": 1,
                        "pp": 1,
                        "ep": 1,
                        "n_replicas": 1,
                    }
                }
            ],
        }
        action_types = []

        def score(jid, _user_id, _slice_id, rank, _features, action_type="place"):
            action_types.append(action_type)
            return {
                "candidate": {"job_id": jid, "type": action_type, "ladder": [rank]},
                "meets_target": True,
                "diag": {"status": "ok", "reason": None},
            }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(resources, specs, [], score),
            patch.object(agent_tools, "get_active_jobs", return_value=[active]),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(action_types, ["swap"])
        self.assertEqual(result["candidates"][0]["type"], "swap")

    def test_failed_swap_respects_retry_backoff(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"new-pool": {"gpus_per_instance": 1, "free_instances": 1}}}
        active = {
            "job_id": "active-job",
            "deployment_status": "deployment_not_materialized",
            "deployment_action_type": "swap",
            "deployment_retry_allowed": False,
            "deployment_retry_after_tick": 5,
            "health": {"rehabilitation_eligible": True},
        }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(resources, specs, [], lambda *_args: self.fail()),
            patch.object(agent_tools, "get_active_jobs", return_value=[active]),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(result["candidates"], [])
        self.assertEqual(
            result["diagnostics"]["active-job"][0]["status"],
            "deployment_retry_backoff",
        )
        self.assertNotIn("active-job", result["exhausted"])

    def test_failed_swap_retry_filters_previously_attempted_shape(self):
        env = "reserved|aws|r1|a|H100"
        rank = _rank(env, "new-pool")
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"new-pool": {"gpus_per_instance": 1, "free_instances": 1}}}
        active = {
            "job_id": "active-job",
            "deployment_status": "deployment_not_materialized",
            "deployment_action_type": "swap",
            "deployment_retry_allowed": True,
            "attempted_deployment_identities": [agent_tools.deployment_ladder_identity([rank])],
            "health": {"rehabilitation_eligible": True},
            "current_ladder": [],
        }

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(resources, specs, [], lambda *_args: self.fail()),
            patch.object(agent_tools, "get_active_jobs", return_value=[active]),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(result["candidates"], [])
        self.assertEqual(
            result["diagnostics"]["active-job"][0]["status"],
            "deployment_shape_already_attempted",
        )

    def test_deployment_retry_backoff_has_typed_diagnostic(self):
        env = "reserved|aws|r1|a|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 1}}}
        pending = [
            {
                "job_id": "job-1",
                "deployment_status": "deployment_not_materialized",
                "deployment_attempts": 1,
                "deployment_retry_after_tick": 4,
                "deployment_retry_allowed": False,
            }
        ]

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(
                resources,
                specs,
                pending,
                lambda *_args, **_kwargs: self.fail("retry backoff must not score frames"),
            ),
            patch.object(agent_tools, "get_active_jobs", return_value=[]),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(
            result["diagnostics"]["job-1"][0]["status"],
            "deployment_retry_backoff",
        )
        self.assertEqual(result["diagnostics"]["job-1"][0]["retry_after_tick"], 4)

    def test_deployment_retry_filters_previously_attempted_shape(self):
        env = "reserved|aws|r1|a|H100"
        rank = _rank(env, "p5")
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 1}}}
        pending = [
            {
                "job_id": "job-1",
                "deployment_status": "deployment_not_materialized",
                "deployment_attempts": 1,
                "deployment_retry_allowed": True,
                "attempted_deployment_identities": [agent_tools.deployment_ladder_identity([rank])],
            }
        ]

        agent_tools._CTX.resource_map = object()
        agent_tools._CTX.surrogate = object()
        agent_tools._CTX.slow_loop = _SlowLoop()
        with (
            self._candidate_patches(
                resources,
                specs,
                pending,
                lambda *_args, **_kwargs: self.fail("attempted shape must not be scored"),
            ),
            patch.object(agent_tools, "get_active_jobs", return_value=[]),
        ):
            result = agent_tools.build_scored_candidates({}, {})

        self.assertEqual(
            result["diagnostics"]["job-1"][0]["status"],
            "deployment_shape_already_attempted",
        )

    def test_under_target_online_frames_remain_available_to_composites(self):
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

    def test_selector_derives_proportional_credit_and_rejects_malformed_fractions(self):
        env = "reserved|aws|r1|z1|H100"
        resources = {env: {"free": 2, "gpu_type": "H100"}}
        specs = {
            env: {
                "p5": {
                    "gpus_per_instance": 1,
                    "free_instances": 2,
                }
            }
        }
        base = {
            "type": "place",
            "ladder": [_rank(env, "p5")],
            "target_tps": 100.0,
            "achieved_tps": 25.0,
            "served_fraction": 0.25,
        }
        candidates = [
            {**base, "job_id": "valid", "sigma": -4.0},
            {
                **base,
                "job_id": "inconsistent",
                "sigma": 100.0,
                "served_fraction": 0.9,
            },
            {
                **base,
                "job_id": "nonfinite",
                "sigma": 100.0,
                "served_fraction": float("nan"),
            },
            {
                **base,
                "job_id": "out-of-range",
                "sigma": 100.0,
                "served_fraction": 1.1,
            },
            {
                **{key: value for key, value in base.items() if key != "served_fraction"},
                "job_id": "incomplete",
                "sigma": 100.0,
            },
            {
                **base,
                "job_id": "zero-target",
                "target_tps": 0.0,
                "achieved_tps": 0.0,
                "served_fraction": 1.0,
                "sigma": 100.0,
            },
            {
                **base,
                "job_id": "bad-admitted",
                "admitted_tps": 50.0,
                "sigma": 100.0,
            },
            {
                "type": "place",
                "ladder": [_rank(env, "p5")],
                "job_id": "legacy",
                "sigma": -4.0,
            },
        ]
        pending = [{"job_id": candidate["job_id"]} for candidate in candidates]
        with (
            patch.object(agent_tools._CTX, "resource_map", object()),
            patch.object(agent_tools, "get_resource_map", return_value=resources),
            patch.object(agent_tools, "instance_catalog", return_value=specs),
            patch.object(agent_tools, "get_pending_jobs", return_value=pending),
            patch.object(
                agent_tools,
                "get_priority",
                return_value=[
                    {"job_id": candidate["job_id"], "priority_score": 20.0}
                    for candidate in candidates
                ],
            ),
        ):
            selected = agent_tools.jointly_select_placements(candidates)

        self.assertEqual(
            {candidate["job_id"] for candidate in selected["chosen"]},
            {"valid"},
        )
        chosen = next(
            candidate for candidate in selected["chosen"] if candidate["job_id"] == "valid"
        )
        self.assertEqual(chosen["served_fraction"], 0.25)
        self.assertEqual(chosen["served_credit"], 5.0)
        self.assertEqual(chosen["solver_gain"], 1.0)
        self.assertEqual(selected["objective"], 1.0)
        self.assertEqual(
            set(selected["deferred"]),
            {
                "inconsistent",
                "nonfinite",
                "out-of-range",
                "incomplete",
                "zero-target",
                "bad-admitted",
                "legacy",
            },
        )

    def test_selector_uses_work_conserving_floor_only_for_placeable_advisory_work(self):
        env = "reserved|aws|r1|z1|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {
            env: {
                "p5": {
                    "gpus_per_instance": 1,
                    "free_instances": 1,
                }
            }
        }
        accounting = {
            "type": "place",
            "target_tps": 100.0,
            "admitted_tps": 25.0,
            "achieved_tps": 25.0,
            "served_fraction": 0.25,
            "admission_mode": "advisory",
            "sigma": -100.0,
        }
        candidates = [
            {
                **accounting,
                "job_id": "advisory",
                "ladder": [_rank(env, "p5")],
            },
            {
                **accounting,
                "job_id": "no-fit",
                "ladder": [],
            },
        ]
        with (
            patch.object(agent_tools._CTX, "resource_map", object()),
            patch.object(agent_tools, "get_resource_map", return_value=resources),
            patch.object(agent_tools, "instance_catalog", return_value=specs),
            patch.object(
                agent_tools,
                "get_pending_jobs",
                return_value=[{"job_id": "advisory"}, {"job_id": "no-fit"}],
            ),
            patch.object(agent_tools, "get_priority", return_value=[]),
        ):
            selected = agent_tools.jointly_select_placements(candidates)

        self.assertEqual([candidate["job_id"] for candidate in selected["chosen"]], ["advisory"])
        chosen = selected["chosen"][0]
        self.assertTrue(chosen["work_conserving_floor"])
        self.assertEqual(chosen["solver_gain"], agent_tools._WORK_CONSERVING_GAIN_FLOOR)
        self.assertEqual(selected["deferred"], ["no-fit"])

    def test_selector_uses_idle_capacity_for_exploration_without_displacing_service(self):
        env = "reserved|aws|r1|z1|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 1}}}
        assessment = {
            "basis": "aic_direct_point",
            "kind": "exploratory",
            "status": "unsupported_prediction",
            "queue_slo_verified": False,
        }
        exploratory = {
            "job_id": "explore",
            "type": "place",
            "ladder": [_rank(env, "p5")],
            "target_tps": 100.0,
            "prediction_assessment": assessment,
            "sigma": 0.0,
        }
        supported = {
            "job_id": "supported",
            "type": "place",
            "ladder": [_rank(env, "p5")],
            "target_tps": 100.0,
            "achieved_tps": 100.0,
            "served_fraction": 1.0,
            "prediction_assessment": {
                **assessment,
                "kind": "point",
                "status": "success",
            },
            "sigma": 1.0,
        }

        def select(candidates, pending):
            with (
                patch.object(agent_tools._CTX, "resource_map", object()),
                patch.object(agent_tools, "get_resource_map", return_value=resources),
                patch.object(agent_tools, "instance_catalog", return_value=specs),
                patch.object(agent_tools, "get_pending_jobs", return_value=pending),
                patch.object(agent_tools, "get_priority", return_value=[]),
            ):
                return agent_tools.jointly_select_placements(candidates)

        idle = select([exploratory], [{"job_id": "explore"}])
        contested = select(
            [exploratory, supported],
            [{"job_id": "explore"}, {"job_id": "supported"}],
        )
        negative_point = {
            **supported,
            "job_id": "point-floor",
            "sigma": -100.0,
            "prediction_assessment": dict(supported["prediction_assessment"]),
        }
        point_floor = select([negative_point], [{"job_id": "point-floor"}])

        self.assertEqual([candidate["job_id"] for candidate in idle["chosen"]], ["explore"])
        self.assertEqual(idle["chosen"][0]["served_credit"], 0.0)
        self.assertTrue(idle["chosen"][0]["work_conserving_floor"])
        self.assertEqual(idle["chosen"][0]["service_class"], "idle_capacity_fallback")
        self.assertEqual(
            idle["chosen"][0]["prediction_assessment"]["selection_mode"],
            "work_conserving",
        )
        self.assertNotIn("served_fraction", idle["chosen"][0])
        self.assertEqual(
            [candidate["job_id"] for candidate in contested["chosen"]],
            ["supported"],
        )
        self.assertEqual(
            point_floor["chosen"][0]["prediction_assessment"]["selection_mode"],
            "work_conserving",
        )
        self.assertEqual(
            point_floor["chosen"][0]["service_class"],
            "idle_capacity_fallback",
        )

    def test_solver_charges_whole_instance_gpu_footprint(self):
        env = "reserved|aws|r1|z1|H100"
        cost = agent_tools._ladder_capacity_cost(
            [_rank(env, "p5.48xlarge", n_replicas=2)],
            {
                env: {
                    "p5.48xlarge": {
                        "gpus_per_instance": 8,
                        "free_instances": 2,
                    }
                }
            },
        )

        self.assertEqual(cost[("gpu", env)], 16)
        self.assertEqual(cost[("pool", env, "p5.48xlarge")], 2)

    def test_queue_unstable_candidate_is_only_work_conserving(self):
        env = "reserved|aws|r1|z1|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 1}}}
        base = {
            "type": "place",
            "ladder": [_rank(env, "p5")],
            "target_tps": 100.0,
            "achieved_tps": 100.0,
            "served_fraction": 1.0,
            "prediction_assessment": {
                "basis": "aic_direct_point",
                "kind": "point",
                "status": "success",
                "queue_slo_verified": False,
            },
        }
        candidates = [
            {
                **base,
                "job_id": "unstable",
                "queue_state": "unstable",
                "sigma": 100.0,
                "prediction_assessment": dict(base["prediction_assessment"]),
            },
            {
                **base,
                "job_id": "stable",
                "queue_state": "stable",
                "sigma": -100.0,
                "prediction_assessment": dict(base["prediction_assessment"]),
            },
        ]
        with (
            patch.object(agent_tools._CTX, "resource_map", object()),
            patch.object(agent_tools, "get_resource_map", return_value=resources),
            patch.object(agent_tools, "instance_catalog", return_value=specs),
            patch.object(
                agent_tools,
                "get_pending_jobs",
                return_value=[{"job_id": "unstable"}, {"job_id": "stable"}],
            ),
            patch.object(
                agent_tools,
                "get_priority",
                return_value=[
                    {"job_id": "unstable", "priority_score": 100.0},
                    {"job_id": "stable", "priority_score": 1.0},
                ],
            ),
        ):
            result = agent_tools.jointly_select_placements(candidates)

        self.assertEqual([candidate["job_id"] for candidate in result["chosen"]], ["stable"])
        self.assertEqual(result["solver_mode"], "exact")

    def test_critical_zero_throughput_swap_uses_emergency_recovery_tier(self):
        env = "reserved|aws|r1|z1|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 1}}}
        candidate = {
            "job_id": "dead-active",
            "type": "swap",
            "ladder": [_rank(env, "p5")],
            "target_tps": 100.0,
            "achieved_tps": 50.0,
            "served_fraction": 0.5,
            "sigma": 10.0,
            "swap_gain_over_keep": 10.0,
            "queue_state": "unstable",
            "rehabilitation_status": "critical",
            "rehabilitation_reasons": ["zero_throughput", "queue_critical"],
            "prediction_assessment": {
                "basis": "aic_direct_point",
                "kind": "point",
                "status": "success",
                "queue_slo_verified": False,
            },
        }
        slow_loop = type("SlowLoop", (), {"get_sss_swap_budget_t": lambda self: 1})()

        with (
            patch.object(agent_tools._CTX, "resource_map", object()),
            patch.object(agent_tools._CTX, "slow_loop", slow_loop),
            patch.object(agent_tools, "get_resource_map", return_value=resources),
            patch.object(agent_tools, "instance_catalog", return_value=specs),
            patch.object(agent_tools, "get_pending_jobs", return_value=[]),
            patch.object(agent_tools, "get_priority", return_value=[]),
        ):
            result = agent_tools.jointly_select_placements([candidate])

        self.assertEqual([item["job_id"] for item in result["chosen"]], ["dead-active"])
        chosen = result["chosen"][0]
        self.assertEqual(chosen["prediction_assessment"]["selection_mode"], "emergency_recovery")
        self.assertEqual(chosen["service_class"], "partial")
        self.assertTrue(chosen["work_conserving_floor"])

    def test_candidate_pruning_reports_before_and_after_counts(self):
        env = "reserved|aws|r1|z1|H100"
        resources = {env: {"free": 8, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 8}}}
        candidates = [
            {
                "job_id": "job",
                "type": "place",
                "ladder": [_rank(env, "p5", n_replicas=replicas)],
                "target_tps": 100.0,
                "achieved_tps": 100.0,
                "served_fraction": 1.0,
                "meets_target": True,
                "sigma": 1.0,
                "prediction_assessment": {
                    "basis": "aic_direct_point",
                    "kind": "point",
                    "status": "success",
                    "queue_slo_verified": False,
                    "proposal_source": "generated",
                },
            }
            for replicas in range(1, 9)
        ]
        with (
            patch.object(agent_tools._CTX, "resource_map", object()),
            patch.object(agent_tools, "get_resource_map", return_value=resources),
            patch.object(agent_tools, "instance_catalog", return_value=specs),
            patch.object(agent_tools, "get_pending_jobs", return_value=[{"job_id": "job"}]),
            patch.object(agent_tools, "get_priority", return_value=[]),
        ):
            result = agent_tools.jointly_select_placements(candidates)

        self.assertEqual(result["candidate_count_before_pruning"], 8)
        self.assertLess(result["candidate_count_after_pruning"], 8)

    def test_joint_solver_enforces_aggregate_swap_budget(self):
        env = "reserved|aws|r1|z1|H100"
        resources = {env: {"free": 2, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 2}}}
        candidates = [
            {
                "job_id": f"active-{index}",
                "type": "swap",
                "ladder": [_rank(env, "p5")],
                "target_tps": 100.0,
                "achieved_tps": 100.0,
                "served_fraction": 1.0,
                "sigma": 10.0 - index,
                "keep_baseline_sigma": -10.0,
                "swap_gain_over_keep": 10.0 - index,
                "prediction_assessment": {
                    "basis": "aic_direct_point",
                    "kind": "point",
                    "status": "success",
                    "queue_slo_verified": False,
                },
            }
            for index in range(2)
        ]
        slow_loop = type("SlowLoop", (), {"get_sss_swap_budget_t": lambda self: 1})()
        with (
            patch.object(agent_tools._CTX, "resource_map", object()),
            patch.object(agent_tools._CTX, "slow_loop", slow_loop),
            patch.object(agent_tools, "get_resource_map", return_value=resources),
            patch.object(agent_tools, "instance_catalog", return_value=specs),
            patch.object(agent_tools, "get_pending_jobs", return_value=[]),
            patch.object(agent_tools, "get_priority", return_value=[]),
        ):
            result = agent_tools.jointly_select_placements(candidates)

        self.assertEqual(result["swap_budget"], 1)
        self.assertEqual(result["swap_count"], 1)
        self.assertEqual(len(result["chosen"]), 1)

    def test_joint_solver_orders_swaps_by_gain_over_keep(self):
        env = "reserved|aws|r1|z1|H100"
        resources = {env: {"free": 1, "gpu_type": "H100"}}
        specs = {env: {"p5": {"gpus_per_instance": 1, "free_instances": 1}}}
        base = {
            "job_id": "active",
            "type": "swap",
            "ladder": [_rank(env, "p5")],
            "target_tps": 100.0,
            "achieved_tps": 100.0,
            "served_fraction": 1.0,
            "prediction_assessment": {
                "basis": "aic_direct_point",
                "kind": "point",
                "status": "success",
                "queue_slo_verified": False,
            },
        }
        candidates = [
            {**base, "sigma": 100.0, "swap_gain_over_keep": 1.0, "marker": "absolute"},
            {**base, "sigma": 1.0, "swap_gain_over_keep": 3.0, "marker": "gain"},
        ]
        slow_loop = type("SlowLoop", (), {"get_sss_swap_budget_t": lambda self: 1})()

        with (
            patch.object(agent_tools._CTX, "resource_map", object()),
            patch.object(agent_tools._CTX, "slow_loop", slow_loop),
            patch.object(agent_tools, "get_resource_map", return_value=resources),
            patch.object(agent_tools, "instance_catalog", return_value=specs),
            patch.object(agent_tools, "get_pending_jobs", return_value=[]),
            patch.object(agent_tools, "get_priority", return_value=[]),
        ):
            result = agent_tools.jointly_select_placements(candidates)

        self.assertEqual(result["chosen"][0]["marker"], "gain")

    def test_finalization_preserves_soft_failure_for_exploratory_action(self):
        plan = self._sigma_plan("p5")
        plan.actions[0].prediction_assessment = {
            "basis": "aic_direct_point",
            "kind": "exploratory",
            "status": "unsupported_prediction",
            "queue_slo_verified": False,
        }
        snapshot = type(
            "Snapshot",
            (),
            {
                "pending_jobs_summary": lambda self: [
                    {"job_id": "job-final", "job_features": {"type": "batch"}}
                ]
            },
        )()

        with (
            patch.object(
                agent_tools,
                "_rank_prediction_payload",
                return_value={"job_config": {}, "job_features": {}},
            ),
            patch.object(
                agent_tools,
                "_predict_outcome_core",
                side_effect=agent_tools.SurrogateUnsupportedConfig("no profile"),
            ),
        ):
            stamped = agent_tools.stamp_plan_predictions(plan, snapshot)

        rank = stamped.actions[0].ladder[0]
        self.assertEqual(rank.predicted_y, {})
        self.assertEqual(rank.prediction_lineage["backends"]["primary"]["status"], "unsupported")
        self.assertEqual(
            rank.prediction_lineage["prediction_assessment"]["status"],
            "unsupported_prediction",
        )
        self.assertNotIn("decision_dro_band", rank.prediction_lineage)

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

    def test_direct_point_sizing_does_not_probe_synthetic_load(self):
        calls = []

        def predict(_config, features, **kwargs):
            calls.append((kwargs["method"], features.get("request_arrival_rate")))
            return {
                "y_hat": {
                    "p99_ttft_ms": 200.0,
                    "p99_tpot_ms": 20.0,
                    "throughput_token_per_sec": 60.0,
                }
            }

        _env, features, result = self._size_online(predict)

        self.assertEqual(calls, [(("AIC_Direct",), 10.0)])
        self.assertEqual(result["point_capacity_tps"], 60.0)
        self.assertEqual(result["achieved_tps"], 60.0)
        self.assertEqual(result["capacity_fraction"], 0.6)
        self.assertFalse(result["meets_target"])
        self.assertFalse(result["partial_online_admission"])
        self.assertEqual(result["per_rank"][0]["partial_search_probes"], 0)
        self.assertIn("queue unmodeled", result["per_rank"][0]["reason"])
        self.assertIsNone(agent_tools._online_sizing_rejection(result, features))

    def test_prediction_only_failures_remain_exploratory_candidates(self):
        outputs = {
            "incomplete": {"p99_ttft_ms": 20.0, "throughput_token_per_sec": 100.0},
            "zero": {
                "p99_ttft_ms": 20.0,
                "p99_tpot_ms": 2.0,
                "throughput_token_per_sec": 0.0,
            },
        }
        for kind, expected_status in (
            ("incomplete", "prediction_incomplete"),
            ("zero", "zero_predicted_capacity"),
        ):
            with self.subTest(kind=kind):
                _env, features, sized = self._size_online(
                    lambda *_args, kind=kind, **_kwargs: {"y_hat": outputs[kind]}
                )
                with (
                    patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
                    patch.object(agent_tools, "size_ladder", return_value=sized),
                    patch.object(
                        agent_tools,
                        "check_feasibility",
                        return_value={"feasible": True},
                    ),
                ):
                    scored = agent_tools._score_one_frame(
                        "job",
                        "user",
                        "slice",
                        _rank("reserved|aws|r1|z1|H100", "p5"),
                        features,
                    )

                self.assertEqual(len(sized["ranks"]), 1)
                self.assertEqual(sized["candidate_kind"], "exploratory")
                self.assertIsNotNone(scored["candidate"])
                self.assertEqual(
                    scored["candidate"]["prediction_assessment"]["status"], expected_status
                )
                self.assertNotIn("admitted_tps", scored["candidate"])
                self.assertNotIn("achieved_tps", scored["candidate"])

    def test_every_soft_prediction_status_reaches_candidate_selection(self):
        rank = _rank("reserved|aws|r1|z1|H100", "p5", n_replicas=3)
        features = {
            "type": "online",
            "target_p99_ttft_ms": 100.0,
            "target_p99_tpot_ms": 10.0,
        }
        for status in sorted(agent_tools._SOFT_PREDICTION_FAILURES):
            with self.subTest(status=status):
                sized = {
                    "ranks": [rank],
                    "candidate_kind": "exploratory",
                    "failure_kind": "soft",
                    "failure_status": status,
                    "failure_reason": "test uncertainty",
                    "target_tps": 100.0,
                    "point_capacity_tps": None,
                    "capacity_fraction": 0.0,
                    "achieved_tps": 0.0,
                    "unmet_tps": 100.0,
                    "meets_target": False,
                    "per_rank": [],
                }
                with (
                    patch.object(agent_tools, "_applicable_mechanism_id", return_value="M_test"),
                    patch.object(agent_tools, "size_ladder", return_value=sized),
                    patch.object(
                        agent_tools,
                        "check_feasibility",
                        return_value={"feasible": True},
                    ),
                    patch.object(agent_tools, "compute_sigma") as compute_sigma,
                ):
                    result = agent_tools._score_one_frame("job", "user", "slice", rank, features)

                candidate = result["candidate"]
                self.assertIsNotNone(candidate)
                self.assertEqual(candidate["ladder"][0]["n_replicas"], 3)
                self.assertEqual(candidate["prediction_assessment"]["status"], status)
                self.assertEqual(result["diag"]["status"], status)
                compute_sigma.assert_not_called()

    def test_point_capacity_does_not_use_uncertainty_lower_bound_as_capacity(self):
        for lower in (0.0, float("nan")):
            with self.subTest(lower=lower):
                _env, _features, result = self._size_online(
                    lambda *_args, lower=lower, **_kwargs: {
                        "y_hat": {
                            "p99_ttft_ms": 20.0,
                            "p99_tpot_ms": 2.0,
                            "throughput_token_per_sec": 100.0,
                        },
                        "throughput_token_per_sec_lower": lower,
                    }
                )

                self.assertEqual(result["point_capacity_tps"], 100.0)
                self.assertEqual(result["achieved_tps"], 100.0)
                self.assertEqual(len(result["ranks"]), 1)

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
                side_effect=lambda rank, features, **_kwargs: {
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
            ],
        )

    def test_size_ladder_preserves_specialist_replica_count(self):
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
                side_effect=lambda rank, features, **_kwargs: {
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

        self.assertFalse(result["meets_target"])
        self.assertEqual(result["ranks"][0]["n_replicas"], 3)
        self.assertEqual(
            calls,
            [
                (("AIC_Direct",), 3),
            ],
        )


if __name__ == "__main__":
    unittest.main()
