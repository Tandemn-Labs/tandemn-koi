import unittest
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text
from src.core.candidate_graph import CandidateGraph
from src.core.confidence_service import ConfidenceService
from src.core.evidence_service import EvidenceService
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, EvidenceRow, Mechanism, MechanismMetadata, Node
from src.validation.icp import ICPResult
from src.validation.quadrants import Quadrant
from tandemn_system_data.clients import PostgresClient
from tandemn_system_data.db import UserRow
from tandemn_system_data.ids import new_user_id


def make_row(
    row_id,
    tick,
    job_id="job_1",
    rank_id="rank_1",
    env_label=("reserved", "aws", "us-east-1", "use1-az1", "H100"),
    mechanism_ids=None,
    icp_result_per_edge=None,
    q_label_per_mechanism=None,
):
    return EvidenceRow(
        row_id=row_id,
        tick=tick,
        deploy_timestamp_utc=float(tick),
        job_id=job_id,
        rank_id=rank_id,
        env_label=env_label,
        X={"batch_size": 8},
        W_observed={"type": "online", "request_rate": 10.0},
        V_observed_trajectory={"kv_cache_pressure": np.array([0.2, 0.3])},
        V_predicted_trajectory={"kv_cache_pressure": np.array([0.2, 0.2])},
        y_observed_trajectory={"ttft_ms": np.array([100.0, 110.0])},
        y_predicted={"ttft_ms": 100.0},
        y_observed_mean={"ttft_ms": 105.0},
        residuals_per_v={"kv_cache_pressure": np.array([0.0, 0.1])},
        residuals_per_y={"ttft_ms": np.array([0.0, 10.0])},
        mechanism_ids=mechanism_ids or [],
        cusum_per_mechanism={},
        q_label_per_mechanism=q_label_per_mechanism or {},
        icp_result_per_edge=icp_result_per_edge or {},
        w_t_snapshot={"ttft_ms": 1.0},
        z_star_snapshot={"ttft_ms": 100.0},
        J_realized=-5.0,
        sigma_realized=1.0,
    )


class CoreSmokeTests(unittest.TestCase):
    def test_candidate_graph_indexes_and_topology(self):
        nodes = {
            "tp": Node("tp", "X"),
            "kv_cache_util": Node("kv_cache_util", "V"),
            "p99_ttft_ms": Node("p99_ttft_ms", "Y"),
        }
        edges = {
            "tp->kv_cache_util": Edge("tp->kv_cache_util", "tp", "kv_cache_util", "X", "V"),
            "kv_cache_util->p99_ttft_ms": Edge(
                "kv_cache_util->p99_ttft_ms", "kv_cache_util", "p99_ttft_ms", "V", "Y"
            ),
        }
        metadata = {edge_id: EdgeMetadata(edge_id=edge_id) for edge_id in edges}
        graph = CandidateGraph(nodes, edges, metadata)

        self.assertEqual(graph.x, ["tp"])
        self.assertEqual(graph.v, ["kv_cache_util"])
        self.assertEqual(graph.y, ["p99_ttft_ms"])
        self.assertTrue(graph.val_topology(graph.get_all_edges()))
        self.assertTrue(graph.check_connected(graph.get_all_edges()))
        self.assertEqual(graph.get_node_type("tp"), "X")
        self.assertEqual(len(graph.get_edges_from("tp")), 1)
        self.assertEqual(len(graph.get_edges_to("p99_ttft_ms")), 1)

    def test_mechanism_registry_indexes_duplicates_and_archive(self):
        registry = MechanismRegistry()
        prefix = Mechanism(
            edge_ids=["shared_prefix_length_avg->kvcache_hit_rate"],
            scope={"x": ["shared_prefix_length_avg", "gpu_type"], "v": ["kvcache_hit_rate"]},
            narrative="Shared prefixes should improve KV cache hits.",
        )
        pd = Mechanism(
            edge_ids=["pd_enabled->pd_inbalance"],
            scope={"x": ["pd_enabled"], "v": ["pd_inbalance"]},
            narrative="PD imbalance can affect online TPOT.",
        )

        prefix_id = registry.add_mechanism(prefix)
        pd_id = registry.add_mechanism(pd)
        duplicate_id = registry.add_mechanism(
            Mechanism(edge_ids=list(prefix.edge_ids), scope=dict(prefix.scope), narrative="dupe")
        )

        self.assertEqual(duplicate_id, prefix_id)
        self.assertEqual(registry.get_mechanism(prefix_id), prefix)
        self.assertIn(prefix_id, registry.mechanisms_by_edge[prefix.edge_ids[0]])
        self.assertTrue(registry.is_duplicate_mechanism(prefix)[0])
        self.assertGreater(
            registry.percentage_scope_match(["shared_prefix_length_avg"], [], prefix), 0
        )
        self.assertIn(prefix, registry.filter_by_scope(["shared_prefix_length_avg"], []))
        self.assertTrue(registry.archive_mechanism(pd_id, "demo archive"))
        self.assertIn(pd_id, registry.mechanisms_by_status["archived"])

    def test_evidence_service_indexes(self):
        client = PostgresClient()
        user_id = new_user_id()
        with client.begin() as session:
            session.add(
                UserRow(user_id=user_id, name="koi core smoke", created_at=datetime.now(UTC))
            )

        store = EvidenceService(user_id=user_id, postgres_client=client)
        env_a = ("reserved", "aws", "us-east-1", "use1-az1", "H100")
        env_b = ("reserved", "aws", "us-west-2", "usw2-az1", "H100")
        rows = [
            make_row(
                "row_1",
                1,
                env_label=env_a,
                mechanism_ids=["M1", "M2"],
                icp_result_per_edge={"e1": ICPResult.ACCEPT, "e2": ICPResult.UNDECIDED},
                q_label_per_mechanism={"M1": Quadrant.Q1, "M2": None},
            ),
            make_row(
                "row_2",
                2,
                env_label=env_b,
                mechanism_ids=["M1"],
                icp_result_per_edge={"e1": ICPResult.REJECT},
                q_label_per_mechanism={"M1": Quadrant.Q3},
            ),
            make_row(
                "row_3",
                3,
                job_id="job_2",
                rank_id="rank_2",
                env_label=env_a,
                mechanism_ids=["M2"],
                icp_result_per_edge={"e2": ICPResult.REJECT},
                q_label_per_mechanism={"M2": Quadrant.Q4},
            ),
        ]
        try:
            for row in rows:
                store.append_row(row)

            self.assertEqual(
                [r.row_id for r in store.get_row("job_1", "rank_1")], ["row_1", "row_2"]
            )
            self.assertEqual(
                [r.row_id for r in store.get_rows_in_window((1, 2))], ["row_1", "row_2"]
            )
            self.assertEqual([r.row_id for r in store.get_rows_for_edge("e1")], ["row_1", "row_2"])
            self.assertEqual([r.row_id for r in store.get_rows_for_edge("e1", limit=1)], ["row_2"])
            self.assertEqual(
                [r.row_id for r in store.get_rows_for_mechanism("M1")], ["row_1", "row_2"]
            )
            self.assertEqual(
                [r.row_id for r in store.get_rows_for_environment(env_a)], ["row_1", "row_3"]
            )
            self.assertEqual(store.count_visits_per_edge("e1"), 2)
            self.assertEqual(store.count_envs_per_edge("e1"), 2)
            self.assertEqual(store.last_touched_per_edge("e1"), 2)
            self.assertEqual(store.q3_rate_window("e1", (1, 3)), 0.5)
            self.assertEqual(
                [(row.row_id, mid, q) for row, mid, q in store.iter_decided_per_mechanism(3, 3)],
                [
                    ("row_1", "M1", Quadrant.Q1),
                    ("row_2", "M1", Quadrant.Q3),
                    ("row_3", "M2", Quadrant.Q4),
                ],
            )
        finally:
            with client.begin() as session:
                session.execute(
                    text("delete from users where user_id = :user_id"), {"user_id": user_id}
                )

    def test_confidence_service_updates(self):
        edge_id = "shared_prefix_length_avg->kvcache_hit_rate"
        mechanism_id = "M_demo"
        graph = CandidateGraph(
            node_table={
                "shared_prefix_length_avg": Node("shared_prefix_length_avg", "X"),
                "kvcache_hit_rate": Node("kvcache_hit_rate", "V"),
            },
            edge_table={
                edge_id: Edge(
                    edge_id,
                    "shared_prefix_length_avg",
                    "kvcache_hit_rate",
                    "X",
                    "V",
                )
            },
            edge_metadata_table={edge_id: EdgeMetadata(edge_id=edge_id, alpha=1.4, beta=0.6)},
        )
        registry = MechanismRegistry(
            mechanism_table={
                mechanism_id: Mechanism(
                    edge_ids=[edge_id],
                    scope={"x": ["shared_prefix_length_avg"], "v": ["kvcache_hit_rate"]},
                    narrative="Shared prefixes should improve KV cache hit rate.",
                    mechanism_id=mechanism_id,
                )
            },
            mechanism_metadata_table={mechanism_id: MechanismMetadata(mechanism_id)},
        )
        service = ConfidenceService(graph, registry)

        self.assertAlmostEqual(service.get_edge_confidence(edge_id), 0.7)
        self.assertEqual(service.get_mechanism_confidence(mechanism_id), 0.5)
        self.assertTrue(service.apply_delta_c_edge(edge_id, Quadrant.Q1, ICPResult.ACCEPT)[1])
        self.assertEqual(service.get_edge_visit_count(edge_id), 1)
        self.assertEqual(service.get_edge_q_histogram(edge_id)["Q1"], 1)
        self.assertTrue(service.apply_delta_c_mechanism(mechanism_id, Quadrant.Q4)[1])
        self.assertEqual(service.get_mechanism_visit_count(mechanism_id), 1)
        self.assertEqual(service.get_mechanism_q_histogram(mechanism_id)["Q4"], 1)


if __name__ == "__main__":
    unittest.main()
