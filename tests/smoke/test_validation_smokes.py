import unittest

import numpy as np
from src.core.candidate_graph import CandidateGraph
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, EvidenceRow, Mechanism, Node
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


if __name__ == "__main__":
    unittest.main()
