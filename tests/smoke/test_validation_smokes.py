import unittest

import numpy as np
from src.core.candidate_graph import CandidateGraph
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, EvidenceRow, Mechanism, Node
from src.infra.resource_map import ResourceMapManager
from src.validation.cusum import Cusum, CusumDirection, CusumResult
from src.validation.icp import ICP, ICPResult
from src.validation.quadrants import Quadrant, QuadrantValidator
from src.validation.validator import Validator


def make_row(row_id, env_label, residuals_per_v=None, residuals_per_y=None, quadrant=None):
    return EvidenceRow(
        row_id=row_id,
        tick=1,
        deploy_timestamp_utc=0.0,
        job_id="job_1",
        rank_id=row_id,
        env_label=env_label,
        X={},
        V_observed_trajectory={},
        V_predicted_trajectory={},
        y_observed_trajectory={},
        y_predicted={},
        y_observed_mean={},
        residuals_per_v=residuals_per_v or {},
        residuals_per_y=residuals_per_y or {},
        mechanism_ids=["M_demo"],
        cusum_per_mechanism={"M_demo": (CusumResult.MATCHED, CusumResult.MATCHED)},
        q_label_per_mechanism={"M_demo": quadrant} if quadrant is not None else {},
        icp_result_per_edge={},
        w_t_snapshot={},
        z_star_snapshot={},
        J_realized=0.0,
        sigma_realized=0.0,
    )


class ValidationSmokeTests(unittest.TestCase):
    def test_mechanism_proposal_rejects_malformed_scope(self):
        xv = Edge("tp->kv_cache_util", "tp", "kv_cache_util", "X", "V")
        vy = Edge(
            "kv_cache_util->p99_tpot_ms",
            "kv_cache_util",
            "p99_tpot_ms",
            "V",
            "Y",
        )
        xv_2 = Edge("ep->expert_util", "ep", "expert_util", "X", "V")
        vy_2 = Edge(
            "expert_util->cost_per_token",
            "expert_util",
            "cost_per_token",
            "V",
            "Y",
        )
        graph = CandidateGraph(
            {
                "tp": Node("tp", "X"),
                "kv_cache_util": Node("kv_cache_util", "V"),
                "p99_tpot_ms": Node("p99_tpot_ms", "Y"),
                "ep": Node("ep", "X"),
                "expert_util": Node("expert_util", "V"),
                "cost_per_token": Node("cost_per_token", "Y"),
            },
            {
                xv.edge_id: xv,
                vy.edge_id: vy,
                xv_2.edge_id: xv_2,
                vy_2.edge_id: vy_2,
            },
            {edge.edge_id: EdgeMetadata(edge.edge_id) for edge in (xv, vy, xv_2, vy_2)},
        )
        validator = Validator(candidate_graph=graph, mechanism_registry=MechanismRegistry())
        valid = Mechanism(
            edge_ids=[xv.edge_id, vy.edge_id],
            scope={
                "x": ["tp"],
                "v": ["kv_cache_util"],
                "workload_type": "online",
                "model_type": "any",
                "conditions": [{"feature": "tp", "op": ">=", "value": 2}],
            },
            narrative="Tensor parallelism changes KV pressure and TPOT.",
        )
        self.assertTrue(validator.val_mechanism_proposal(valid)[0])

        cases = {
            "empty_edges": ([], valid.scope, "no edges"),
            "unknown_edge": (["missing->edge"], valid.scope, "not in CandidateGraph"),
            "xv_only": ([xv.edge_id], valid.scope, "no complete X->V->Y path"),
            "vy_only": ([vy.edge_id], valid.scope, "no complete X->V->Y path"),
            "disconnected": (
                [xv.edge_id, vy.edge_id, xv_2.edge_id, vy_2.edge_id],
                {"x": ["tp", "ep"], "v": ["kv_cache_util", "expert_util"]},
                "disconnected",
            ),
            "empty_scope": ([xv.edge_id], {}, "at least one X variable"),
            "v_only_scope": (
                [xv.edge_id, vy.edge_id],
                {"x": [], "v": ["kv_cache_util"]},
                "at least one X variable",
            ),
            "invalid_x": (
                [xv.edge_id],
                {"x": ["kv_cache_util"], "v": []},
                "not X",
            ),
            "unknown_workload": (
                [xv.edge_id],
                {"x": ["tp"], "v": [], "workload_type": "realtime"},
                "unknown workload_type",
            ),
            "unknown_model": (
                [xv.edge_id],
                {"x": ["tp"], "v": [], "model_type": "dense_medium"},
                "unknown model_type",
            ),
            "conditions_not_list": (
                [xv.edge_id],
                {"x": ["tp"], "v": [], "conditions": {}},
                "conditions must be a list",
            ),
            "conditions_none": (
                [xv.edge_id, vy.edge_id],
                {"x": ["tp"], "v": ["kv_cache_util"], "conditions": None},
                "conditions must be a list",
            ),
            "legacy_alias": (
                [xv.edge_id, vy.edge_id],
                {"subset_x": ["tp"], "v": ["kv_cache_util"]},
                "unknown scope keys",
            ),
            "set_scope": (
                [xv.edge_id, vy.edge_id],
                {"x": {"tp"}, "v": ["kv_cache_util"]},
                "must be a list",
            ),
            "unknown_scope_key": (
                [xv.edge_id, vy.edge_id],
                {"x": ["tp"], "v": ["kv_cache_util"], "extra": True},
                "unknown scope keys",
            ),
            "unknown_operator": (
                [xv.edge_id],
                {
                    "x": ["tp"],
                    "v": [],
                    "conditions": [{"feature": "tp", "op": "!=", "value": 1}],
                },
                "is unknown",
            ),
            "condition_not_x": (
                [xv.edge_id],
                {
                    "x": ["tp"],
                    "v": [],
                    "conditions": [{"feature": "kv_cache_util", "op": ">", "value": 0}],
                },
                "not X",
            ),
        }
        for name, (edges, scope, expected) in cases.items():
            with self.subTest(name=name):
                ok, violations = validator.val_mechanism_proposal(
                    Mechanism(edge_ids=edges, scope=scope, narrative=name)
                )
                self.assertFalse(ok)
                self.assertTrue(any(expected in violation for violation in violations))

    def test_quadrants_classify_and_aggregate(self):
        validator = QuadrantValidator()

        self.assertEqual(
            validator.classify_quadrant(CusumResult.MATCHED, CusumResult.MATCHED), Quadrant.Q1
        )
        self.assertEqual(
            validator.classify_quadrant(CusumResult.MATCHED, CusumResult.DIVERGED), Quadrant.Q2
        )
        self.assertEqual(
            validator.classify_quadrant(CusumResult.DIVERGED, CusumResult.MATCHED), Quadrant.Q3
        )
        self.assertEqual(
            validator.classify_quadrant(CusumResult.DIVERGED, CusumResult.DIVERGED), Quadrant.Q4
        )

        class Store:
            def iter_decided_per_mechanism(self, window, tick):
                rows = [
                    make_row("row_1", "env", quadrant=Quadrant.Q1),
                    make_row("row_2", "env", quadrant=Quadrant.Q4),
                    make_row("row_3", "env", quadrant=Quadrant.Q4),
                ][-window:]
                for row in rows:
                    yield row, "M_demo", row.q_label_per_mechanism["M_demo"]

        histogram = validator.aggregate_quadrant_histogram(Store(), window=3)
        self.assertEqual(histogram[Quadrant.Q1], 1)
        self.assertEqual(histogram[Quadrant.Q2], 0)
        self.assertEqual(histogram[Quadrant.Q3], 0)
        self.assertEqual(histogram[Quadrant.Q4], 2)

    def test_cusum_mechanism_and_single_variable(self):
        edge_xv = Edge("batch_size->kv_cache_pressure", "batch_size", "kv_cache_pressure", "X", "V")
        edge_vy = Edge("kv_cache_pressure->ttft_ms", "kv_cache_pressure", "ttft_ms", "V", "Y")
        mechanism = Mechanism(
            mechanism_id="M_demo",
            edge_ids=[edge_xv.edge_id, edge_vy.edge_id],
            scope={"x": ["batch_size"], "v": ["kv_cache_pressure"]},
            narrative="KV pressure mediates batch size and TTFT.",
        )
        graph = CandidateGraph(
            node_table={
                "batch_size": Node("batch_size", "X"),
                "kv_cache_pressure": Node("kv_cache_pressure", "V"),
                "ttft_ms": Node("ttft_ms", "Y"),
            },
            edge_table={edge_xv.edge_id: edge_xv, edge_vy.edge_id: edge_vy},
            edge_metadata_table={
                edge_xv.edge_id: EdgeMetadata(edge_xv.edge_id),
                edge_vy.edge_id: EdgeMetadata(edge_vy.edge_id),
            },
        )

        cusum = Cusum()
        v_verdict, y_verdict = cusum.cusum_per_mechanism(
            mechanism=mechanism,
            candidate_graph=graph,
            v_obs_traj={"kv_cache_pressure": np.array([0.21, 0.20, 0.22])},
            v_hat_traj={"kv_cache_pressure": 0.20},
            y_obs_traj={"ttft_ms": np.array([110.0, 111.0, 112.0])},
            y_hat_traj={"ttft_ms": 100.0},
            v_params={"kv_cache_pressure": (0.05, 0.20)},
            y_params={"ttft_ms": (1.0, 5.0)},
        )
        direction, fired, fire_tick = cusum.cusum_per_v(
            observed=np.array([110.0, 111.0, 112.0]),
            predicted=100.0,
            delta=1.0,
            h=5.0,
        )

        self.assertEqual(v_verdict, CusumResult.MATCHED)
        self.assertEqual(y_verdict, CusumResult.DIVERGED)
        self.assertEqual(direction, CusumDirection.UP)
        self.assertTrue(fired)
        self.assertEqual(fire_tick, 0)

    def test_icp_accepts_stable_edge_and_rejects_shifted_edge(self):
        class Store:
            def __init__(self, rows_by_edge):
                self.rows_by_edge = rows_by_edge

            def get_rows_for_edge(self, edge_id, limit=None):
                rows = list(self.rows_by_edge.get(edge_id, []))
                return rows if limit is None else rows[-limit:]

        def make_rows(edge, residuals_by_env):
            rows = []
            for idx, (env, residuals) in enumerate(residuals_by_env.items()):
                residual_dict = {edge.dst: np.asarray(residuals, dtype=float)}
                rows.append(
                    make_row(
                        row_id=f"row_{edge.edge_id}_{idx}",
                        env_label=env,
                        residuals_per_v=residual_dict if edge.dst_type == "V" else {},
                        residuals_per_y=residual_dict if edge.dst_type == "Y" else {},
                    )
                )
            return rows

        v_edge = Edge(
            "shared_prefix_length_avg->kvcache_hit_rate",
            "shared_prefix_length_avg",
            "kvcache_hit_rate",
            "X",
            "V",
        )
        y_edge = Edge("kvcache_hit_rate->p99_ttft_ms", "kvcache_hit_rate", "p99_ttft_ms", "V", "Y")
        envs = [
            ("reserved", "aws", "us-east-1", "use1-az1", "H100"),
            ("reserved", "aws", "us-west-2", "usw2-az1", "H100"),
            ("reserved", "gcp", "us-central1", "us-central1-a", "H100"),
        ]
        base = np.linspace(-0.2, 0.2, 15)
        store = Store(
            {
                v_edge.edge_id: make_rows(v_edge, {env: base.copy() for env in envs}),
                y_edge.edge_id: make_rows(
                    y_edge, {env: base + idx * 5.0 for idx, env in enumerate(envs)}
                ),
            }
        )

        icp = ICP()
        self.assertEqual(icp.compute_icp_per_edge(v_edge, store), ICPResult.ACCEPT)
        self.assertEqual(icp.compute_icp_per_edge(y_edge, store), ICPResult.REJECT)

    def test_validator_requires_launch_critical_rank_config(self):
        result = Validator().val_plan(
            _raw_place_plan({"instance_type": "p5.48xlarge", "gpu_count": 1, "tp": 1, "pp": 1})
        )
        self.assertTrue(result.feasible)

        cases = {
            "missing_instance": ({"gpu_count": 1, "tp": 1, "pp": 1}, "instance_type"),
            "missing_gpu_count": (
                {"instance_type": "p5.48xlarge", "tp": 1, "pp": 1},
                "gpu_count/count",
            ),
            "missing_tp": ({"instance_type": "p5.48xlarge", "gpu_count": 1, "pp": 1}, "tp"),
            "missing_pp": ({"instance_type": "p5.48xlarge", "gpu_count": 1, "tp": 1}, "pp"),
        }
        for name, (config, expected) in cases.items():
            with self.subTest(name=name):
                result = Validator().val_plan(_raw_place_plan(config))
                self.assertFalse(result.feasible)
                self.assertTrue(any(expected in violation for violation in result.violations))

    def test_validator_accepts_consistent_partial_admission_without_predictions(self):
        result = Validator().val_plan(_raw_partial_plan())

        self.assertTrue(result.feasible, result.violations)

    def test_validator_accepts_full_service_accounting_without_admission_fields(self):
        plan = _raw_partial_plan()
        action = plan["actions"][0]
        action.update(
            {
                "achieved_tps": 100.0,
                "unmet_tps": 0.0,
                "meets_target": True,
                "served_fraction": 1.0,
            }
        )
        action.pop("admitted_tps")
        action.pop("admission_mode")
        action["ladder"][0].pop("rank_traffic_share")

        result = Validator().val_plan(plan)

        self.assertTrue(result.feasible, result.violations)

        action["target_p99_ttft_ms"] = 100.0
        action["meets_target"] = False
        advisory_result = Validator(partial_online_admission_mode="advisory").val_plan(plan)

        self.assertTrue(advisory_result.feasible, advisory_result.violations)

    def test_validator_accepts_point_capacity_and_exploratory_accounting(self):
        point = _raw_partial_plan()
        point_action = point["actions"][0]
        point_action.pop("admitted_tps")
        point_action.pop("admission_mode")
        point_action["ladder"][0].pop("rank_traffic_share")
        point_action["target_p99_ttft_ms"] = 100.0
        point_action["prediction_assessment"] = {
            "basis": "aic_direct_point",
            "kind": "point",
            "status": "success",
            "queue_slo_verified": False,
        }

        point_result = Validator().val_plan(point)

        self.assertTrue(point_result.feasible, point_result.violations)

        point_action["ladder"].append(
            {
                **point_action["ladder"][0],
                "rank_traffic_share": 0.4,
            }
        )
        point_action["ladder"][0]["rank_traffic_share"] = 0.6
        multi_point_result = Validator().val_plan(point)

        self.assertTrue(multi_point_result.feasible, multi_point_result.violations)

        exploratory = _raw_place_plan(
            {"instance_type": "p5.48xlarge", "gpu_count": 1, "tp": 1, "pp": 1}
        )
        exploratory["actions"][0]["prediction_assessment"] = {
            "basis": "aic_direct_point",
            "kind": "exploratory",
            "status": "unsupported_prediction",
            "queue_slo_verified": False,
        }

        exploratory_result = Validator().val_plan(exploratory)

        self.assertTrue(exploratory_result.feasible, exploratory_result.violations)

        exploratory["actions"][0]["ladder"].append(
            {
                **exploratory["actions"][0]["ladder"][0],
                "rank_id": "second-exploratory-rank",
            }
        )
        multi_exploratory = Validator().val_plan(exploratory)

        self.assertFalse(multi_exploratory.feasible)
        self.assertTrue(
            any("rank_traffic_share" in violation for violation in multi_exploratory.violations)
        )

    def test_validator_accepts_legacy_batch_partial_without_online_admission_fields(self):
        plan = _raw_partial_plan()
        plan["actions"][0].pop("admitted_tps")
        plan["actions"][0].pop("admission_mode")

        result = Validator().val_plan(plan)

        self.assertTrue(result.feasible, result.violations)

    def test_validator_guards_online_partial_admission_by_mode(self):
        plan = _raw_partial_plan()
        action = plan["actions"][0]
        action["target_p99_ttft_ms"] = 100.0

        disabled = Validator().val_plan(plan)
        enabled = Validator(partial_online_admission_mode="advisory").val_plan(plan)

        self.assertFalse(disabled.feasible)
        self.assertTrue(any("disabled" in violation for violation in disabled.violations))
        self.assertTrue(enabled.feasible, enabled.violations)

        for field_name in ("admitted_tps", "admission_mode"):
            with self.subTest(field_name=field_name):
                missing = _raw_partial_plan()
                missing["actions"][0]["target_p99_tpot_ms"] = 10.0
                missing["actions"][0].pop(field_name)
                result = Validator(partial_online_admission_mode="advisory").val_plan(missing)
                self.assertFalse(result.feasible)
                self.assertTrue(
                    any(field_name in violation for violation in result.violations),
                    result.violations,
                )

        enforced = _raw_partial_plan()
        enforced["actions"][0]["target_p99_ttft_ms"] = 100.0
        enforced["actions"][0]["admission_mode"] = "enforced"
        result = Validator(partial_online_admission_mode="advisory").val_plan(enforced)
        self.assertFalse(result.feasible)
        self.assertTrue(any("unsupported" in violation for violation in result.violations))

        with self.assertRaisesRegex(ValueError, "off.*advisory"):
            Validator(partial_online_admission_mode="enforced")

    def test_validator_rejects_inconsistent_partial_admission_metadata(self):
        cases = (
            ("target_nonfinite", "target_tps", float("inf"), "target_tps must be"),
            ("target_zero", "target_tps", 0.0, "target_tps must be"),
            ("admitted_nonfinite", "admitted_tps", float("nan"), "admitted_tps must be"),
            ("admitted_zero", "admitted_tps", 0.0, "admitted_tps must be"),
            ("achieved_nonfinite", "achieved_tps", float("inf"), "achieved_tps must be"),
            ("achieved_zero", "achieved_tps", 0.0, "achieved_tps must be"),
            ("admitted_mismatch", "admitted_tps", 55.0, "approximately equal achieved_tps"),
            ("admitted_over_target", "admitted_tps", 110.0, "admitted_tps must be <="),
            ("achieved_over_target", "achieved_tps", 110.0, "achieved_tps must be <="),
            ("unmet_nonfinite", "unmet_tps", float("nan"), "unmet_tps must be"),
            ("unmet_negative", "unmet_tps", -1.0, "unmet_tps must be"),
            ("unmet_mismatch", "unmet_tps", 30.0, "target_tps - achieved_tps"),
            ("fraction_nonfinite", "served_fraction", float("inf"), "served_fraction must be"),
            ("fraction_zero", "served_fraction", 0.0, "served_fraction must be"),
            ("fraction_over_one", "served_fraction", 1.1, "served_fraction must be"),
            ("fraction_mismatch", "served_fraction", 0.7, "achieved_tps / target_tps"),
            ("meets_target_type", "meets_target", "false", "meets_target must be a boolean"),
            ("meets_target_partial", "meets_target", True, "must match whether achieved_tps"),
            ("admission_mode", "admission_mode", "automatic", "admission_mode must be"),
            ("admission_mode_type", "admission_mode", [], "admission_mode must be"),
            ("enforced_mode", "admission_mode", "enforced", "unsupported"),
        )
        for name, field_name, value, expected in cases:
            with self.subTest(name=name):
                plan = _raw_partial_plan()
                plan["actions"][0][field_name] = value

                result = Validator().val_plan(plan)

                self.assertFalse(result.feasible)
                self.assertTrue(
                    any(expected in violation for violation in result.violations),
                    result.violations,
                )

    def test_validator_requires_complete_partial_admission_metadata(self):
        for field_name in (
            "achieved_tps",
            "unmet_tps",
            "meets_target",
            "served_fraction",
        ):
            with self.subTest(field_name=field_name):
                plan = _raw_partial_plan()
                plan["actions"][0].pop(field_name)

                result = Validator().val_plan(plan)

                self.assertFalse(result.feasible)
                self.assertTrue(
                    any(field_name in violation for violation in result.violations),
                    result.violations,
                )

    def test_validator_rejects_partial_rank_share_errors(self):
        for share in (None, 0.0, 1.1, float("inf"), float("nan")):
            with self.subTest(share=share):
                plan = _raw_partial_plan()
                plan["actions"][0]["ladder"][0]["rank_traffic_share"] = share

                result = Validator().val_plan(plan)

                self.assertFalse(result.feasible)
                self.assertTrue(
                    any("rank_traffic_share must be finite" in v for v in result.violations),
                    result.violations,
                )

        plan = _raw_partial_plan()
        plan["actions"][0]["ladder"][0]["rank_traffic_share"] = 0.5
        result = Validator().val_plan(plan)
        self.assertFalse(result.feasible)
        self.assertTrue(
            any("rank_traffic_share sum" in violation for violation in result.violations),
            result.violations,
        )

    def test_validator_requires_complete_shares_for_multi_rank_and_explicit_share(self):
        plan = _raw_place_plan({"instance_type": "p5.48xlarge", "gpu_count": 1, "tp": 1, "pp": 1})
        plan["actions"][0]["ladder"].append(
            {
                "role": "aggregate",
                "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                "config": {
                    "instance_type": "p5.48xlarge",
                    "gpu_count": 1,
                    "tp": 1,
                    "pp": 1,
                },
                "n_replicas": 1,
            }
        )

        missing = Validator().val_plan(plan)
        self.assertFalse(missing.feasible)
        self.assertTrue(any("rank_traffic_share" in item for item in missing.violations))

        plan["actions"][0]["ladder"][0]["rank_traffic_share"] = 0.4
        plan["actions"][0]["ladder"][1]["rank_traffic_share"] = 0.6
        complete = Validator().val_plan(plan)
        self.assertTrue(complete.feasible, complete.violations)

        single = _raw_place_plan({"instance_type": "p5.48xlarge", "gpu_count": 1, "tp": 1, "pp": 1})
        single["actions"][0]["ladder"][0]["rank_traffic_share"] = 0.5
        explicit = Validator().val_plan(single)
        self.assertFalse(explicit.feasible)
        self.assertTrue(any("sum" in item for item in explicit.violations))

    def test_c5_rejects_cross_job_instance_pool_overallocation(self):
        env = "reserved|aws|us-east-1|us-east-1b|L40S"
        resources = {
            env: {
                "free": 16,
                "total": 16,
                "gpu_type": "L40S",
                "pools": [
                    {
                        "instance_type": "g6e.xlarge",
                        "gpus_per_instance": 1,
                        "free_instances": 4,
                        "free": 4,
                    },
                    {
                        "instance_type": "g6e.12xlarge",
                        "gpus_per_instance": 4,
                        "free_instances": 3,
                        "free": 12,
                    },
                ],
            }
        }

        class Snapshot:
            @staticmethod
            def resources_summary():
                return resources

        rank = {
            "role": "aggregate",
            "env": env.split("|"),
            "config": {
                "instance_type": "g6e.12xlarge",
                "gpu_count": 2,
                "tp": 2,
                "pp": 1,
            },
            "n_replicas": 2,
        }
        plan = {
            "actions": [
                {"job_id": "job_1", "type": "place", "ladder": [rank]},
                {"job_id": "job_2", "type": "place", "ladder": [rank]},
            ]
        }
        result = Validator(resource_map=ResourceMapManager(user_id="test")).val_plan(
            plan, Snapshot()
        )

        self.assertFalse(result.feasible)
        self.assertIn(
            f"C5 capacity: env {env} pool g6e.12xlarge requested 4 instances, only 3 free",
            result.violations,
        )


def _raw_place_plan(config):
    return {
        "actions": [
            {
                "job_id": "job_1",
                "type": "place",
                "ladder": [
                    {
                        "role": "aggregate",
                        "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                        "config": config,
                        "n_replicas": 1,
                    }
                ],
            }
        ]
    }


def _raw_partial_plan():
    plan = _raw_place_plan({"instance_type": "p5.48xlarge", "gpu_count": 1, "tp": 1, "pp": 1})
    action = plan["actions"][0]
    action.update(
        {
            "target_tps": 100.0,
            "admitted_tps": 60.0,
            "achieved_tps": 60.0,
            "unmet_tps": 40.0,
            "meets_target": False,
            "served_fraction": 0.6,
            "admission_mode": "advisory",
        }
    )
    action["ladder"][0]["rank_traffic_share"] = 0.6
    return plan


if __name__ == "__main__":
    unittest.main()
