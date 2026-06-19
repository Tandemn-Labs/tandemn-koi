import unittest

import numpy as np
from src.core.candidate_graph import CandidateGraph
from src.core.models import Edge, EdgeMetadata, EvidenceRow, Mechanism, Node
from src.validation.cusum import Cusum, CusumDirection, CusumResult
from src.validation.icp import ICP, ICPResult
from src.validation.quadrants import Quadrant, QuadrantValidator


def make_row(row_id, env_label, residuals_per_v=None, residuals_per_y=None, quadrant=None):
    return EvidenceRow(
        row_id=row_id,
        tick=1,
        deploy_timestamp_utc=0.0,
        job_id="job_1",
        rank_id=row_id,
        env_label=env_label,
        X={},
        W_observed={},
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


if __name__ == "__main__":
    unittest.main()
