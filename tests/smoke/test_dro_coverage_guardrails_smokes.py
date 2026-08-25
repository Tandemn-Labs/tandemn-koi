import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from src.cost.dro import DRO
from src.orchestrator.fsm_states import FSMState, TickContext, TickRunner


def _coverage_runner(dro: DRO) -> tuple[TickRunner, Mock]:
    slow_loop = Mock()
    slow_loop.state = SimpleNamespace(
        tick=0,
        epsilon_dro=dro.epsilon,
        observed_coverage=dro.target,
    )
    slow_loop.anneal_targets.return_value = {}

    def slow_update_all(
        *, tick, observed_swap_rate, observed_coverage, r2_gradient, target_overrides
    ):
        slow_loop.state.tick = tick
        slow_loop.state.observed_coverage = observed_coverage
        slow_loop.state.epsilon_dro = dro.update_epsilon_dro(
            slow_loop.state.epsilon_dro,
            observed_coverage,
        )
        return slow_loop.state

    slow_loop.slow_update_all.side_effect = slow_update_all

    runner = TickRunner.__new__(TickRunner)
    runner.dro = dro
    runner.slow_loop = slow_loop
    runner.confidence_service = Mock()
    runner.mechanism_registry = Mock()
    runner.recalibrate_every = 0
    runner._last_swap_count = 0
    runner._last_active_count = 0
    return runner, slow_loop


def _row(*, band=None, observed=11.0, predicted=10.0, required=None, q_labels=None):
    lineage = {} if band is None else {"decision_dro_band": band}
    if required is not None:
        lineage["decision_required_objectives"] = required
    return SimpleNamespace(
        row_id="1_job_1_rank_1",
        job_id="job_1",
        rank_id="rank_1",
        env_label=("reserved", "aws", "region", "zone", "H100"),
        q_label_per_mechanism=q_labels or {},
        icp_result_per_edge={},
        y_predicted={"latency": predicted},
        y_observed_mean={"latency": observed},
        prediction_lineage=lineage,
    )


class DROCoverageGuardrailSmokes(unittest.TestCase):
    def test_missing_prediction_has_certain_violation_risk(self):
        dro = DRO()

        for prediction in ({}, {"latency": None}):
            with self.subTest(prediction=prediction):
                result = dro.dro_chance_constraint(
                    pred_y=prediction,
                    slo_thresholds={"latency": 100.0},
                )

                self.assertEqual(result["latency"], 1.0)
                self.assertEqual(result["_any_violated"], 1.0)
                self.assertEqual(set(result), {"latency", "_any_violated"})
                self.assertTrue(all(isinstance(value, float) for value in result.values()))

    def test_aggregate_uses_conservative_union_bound(self):
        dro = DRO()
        residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        with patch.object(dro, "get_residual_history", return_value=residuals):
            result = dro.dro_chance_constraint(
                pred_y={"latency": 0.0, "cost": 0.0},
                slo_thresholds={"latency": 0.0, "cost": 0.0},
                epsilon_dro=0.0,
            )

        self.assertAlmostEqual(result["latency"], 0.4)
        self.assertAlmostEqual(result["cost"], 0.4)
        self.assertAlmostEqual(result["_any_violated"], 0.8)

    def test_missing_empty_and_malformed_bands_are_not_covered(self):
        cases = {
            "missing objective": ({"latency": 5.0}, {}),
            "empty overlap": ({}, {"latency": {"lower": 0.0, "upper": 10.0}}),
            "missing bound": ({"latency": 5.0}, {"latency": {"lower": 0.0}}),
            "non-numeric bound": (
                {"latency": 5.0},
                {"latency": {"lower": "0", "upper": 10.0}},
            ),
            "non-finite bound": (
                {"latency": 5.0},
                {"latency": {"lower": 0.0, "upper": float("nan")}},
            ),
            "reversed bounds": (
                {"latency": 5.0},
                {"latency": {"lower": 10.0, "upper": 0.0}},
            ),
        }

        for name, (outcome, band) in cases.items():
            with self.subTest(name=name):
                self.assertFalse(DRO._all_objectives_inside(outcome, band))

        band = {"latency": {"lower": 0.0, "upper": 10.0}}
        self.assertTrue(DRO._all_objectives_inside({"latency": 5.0, "debug": 999.0}, band))
        self.assertIsNone(DRO._coverage_status({"debug": 999.0}, band, ["latency"]))
        self.assertFalse(
            DRO._all_objectives_inside(
                {"latency": 5.0},
                band,
                ["latency", "cost"],
            )
        )

    def test_no_persisted_band_is_neutral_for_epsilon(self):
        dro = DRO(epsilon_init=0.2, target_coverage=0.75)
        runner, slow_loop = _coverage_runner(dro)
        ctx = TickContext(tick=1, evidence_rows=[_row()])

        next_state = runner.S3(ctx)

        self.assertEqual(next_state, FSMState.S4_AGENTIC_PLAN)
        self.assertEqual(ctx.confidence_diagnostics, [])
        self.assertEqual(dro.epsilon, 0.2)
        self.assertEqual(slow_loop.state.epsilon_dro, 0.2)
        coverage = ctx.slow_update_diagnostics["dro"]["coverage"]
        self.assertFalse(coverage["has_signal"])
        self.assertEqual(coverage["reason"], "no_evaluable_decision_bands")
        self.assertEqual(coverage["value"], 0.75)

    def test_current_residual_is_appended_after_persisted_band_coverage(self):
        dro = DRO(epsilon_init=0.2)
        runner, _ = _coverage_runner(dro)
        band = {"latency": {"point": 5.0, "lower": 0.0, "upper": 10.0}}
        ctx = TickContext(
            tick=1,
            evidence_rows=[_row(band=band, observed=100.0, predicted=5.0, required=["latency"])],
        )
        events = []
        coverage_status = dro._coverage_status
        append_residual = dro.append_residual_history

        def record_coverage(outcome, decision_band, required_objectives=None):
            events.append("coverage")
            return coverage_status(outcome, decision_band, required_objectives)

        def record_append(*, pred_y, obs_y):
            events.append("append")
            append_residual(pred_y=pred_y, obs_y=obs_y)

        with (
            patch.object(dro, "_coverage_status", side_effect=record_coverage),
            patch.object(dro, "append_residual_history", side_effect=record_append),
            patch.object(
                dro,
                "compute_dro_band",
                side_effect=AssertionError("coverage must not reconstruct a DRO band"),
            ) as compute_band,
        ):
            runner.S3(ctx)

        coverage = ctx.slow_update_diagnostics["dro"]["coverage"]
        self.assertTrue(coverage["has_signal"])
        self.assertEqual(coverage["value"], 0.0)
        self.assertEqual(coverage["inside_rows"], 0)
        self.assertEqual(coverage["evaluable_row_count"], 1)
        self.assertLess(events.index("coverage"), events.index("append"))
        compute_band.assert_not_called()
        np.testing.assert_array_equal(dro.get_residual_history("latency"), np.array([95.0]))

    def test_malformed_persisted_band_is_measured_as_uncovered(self):
        dro = DRO(epsilon_init=0.2, target_coverage=0.75)
        runner, slow_loop = _coverage_runner(dro)
        malformed_band = {"latency": {"lower": 0.0}}
        ctx = TickContext(
            tick=1,
            evidence_rows=[_row(band=malformed_band, required=["latency"])],
        )

        runner.S3(ctx)

        coverage = ctx.slow_update_diagnostics["dro"]["coverage"]
        self.assertTrue(coverage["has_signal"])
        self.assertEqual(coverage["reason"], "measured")
        self.assertEqual(coverage["value"], 0.0)
        self.assertEqual(coverage["evaluable_row_count"], 1)
        self.assertGreater(slow_loop.state.epsilon_dro, 0.2)

    def test_residual_is_durable_before_confidence_failure(self):
        dro = DRO()
        runner, _ = _coverage_runner(dro)
        band = {"latency": {"lower": 0.0, "upper": 20.0}}
        row = _row(
            band=band,
            observed=11.0,
            predicted=10.0,
            required=["latency"],
            q_labels={"mechanism": "Q1"},
        )
        ctx = TickContext(tick=1, evidence_rows=[row])
        events = []
        coverage_status = dro._coverage_status
        append_residual = dro.append_residual_history

        def record_coverage(outcome, decision_band, required_objectives=None):
            events.append("coverage")
            return coverage_status(outcome, decision_band, required_objectives)

        def record_append(*, pred_y, obs_y):
            events.append("append")
            append_residual(pred_y=pred_y, obs_y=obs_y)

        def fail_confidence(*args, **kwargs):
            events.append("confidence")
            raise RuntimeError("confidence write failed")

        runner.confidence_service.get_mechanism_alpha_beta.return_value = (1.0, 1.0)
        runner.confidence_service.get_mechanism_confidence.return_value = 0.5
        runner.confidence_service.get_mechanism_visit_count.return_value = 0
        runner.confidence_service.get_delta_c_mechanism.return_value = (1.0, 0.0)
        runner.confidence_service.apply_delta_c_mechanism.side_effect = fail_confidence
        with (
            patch.object(dro, "_coverage_status", side_effect=record_coverage),
            patch.object(dro, "append_residual_history", side_effect=record_append),
            self.assertRaises(RuntimeError),
        ):
            runner.S3(ctx)

        self.assertEqual(events[:3], ["coverage", "append", "confidence"])
        np.testing.assert_array_equal(dro.get_residual_history("latency"), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
