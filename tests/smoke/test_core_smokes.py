import unittest
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text
from src.core.candidate_graph import CandidateGraph
from src.core.confidence_service import ConfidenceService
from src.core.evidence_service import EvidenceService
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import (
    Edge,
    EdgeMetadata,
    EvidenceRow,
    Mechanism,
    MechanismMetadata,
    Node,
    Plan,
    RankSpec,
)
from src.prediction.calibration import build_prediction_context, calibrate_prediction
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
        X={"batch_size": 8, "type": "online", "request_rate": 10.0},
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
        deployment_id=f"deploy-{row_id}",
        evidence_available_timestamp_utc=float(tick) + 0.5,
        prediction_lineage={"schema_version": 3, "composite_version": "test-v1"},
    )


class CoreSmokeTests(unittest.TestCase):
    def test_plan_action_preserves_online_targets(self):
        plan = Plan.from_raw(
            {
                "actions": [
                    {
                        "job_id": "job_online",
                        "type": "place",
                        "ladder": [],
                        "target_tps": 2200.0,
                        "target_p99_ttft_ms": 500.0,
                        "target_p99_tpot_ms": 50.0,
                    }
                ]
            },
            tick=1,
        )

        action = plan.actions[0]
        self.assertEqual(action.target_p99_ttft_ms, 500.0)
        self.assertEqual(action.target_p99_tpot_ms, 50.0)
        self.assertEqual(action.to_dict()["target_p99_ttft_ms"], 500.0)
        self.assertEqual(action.to_dict()["target_p99_tpot_ms"], 50.0)

    def test_plan_action_round_trips_partial_admission_metadata(self):
        raw_action = {
            "job_id": "job_partial",
            "type": "place",
            "ladder": [],
            "target_tps": 100.0,
            "admitted_tps": 60.0,
            "achieved_tps": 60.0,
            "unmet_tps": 40.0,
            "meets_target": False,
            "served_fraction": 0.6,
            "admission_mode": "advisory",
            "prediction_assessment": {
                "basis": "aic_direct_point",
                "kind": "point",
                "status": "success",
                "queue_slo_verified": False,
            },
            "point_capacity_covers_target": False,
            "base_latency_within_target": True,
            "queue_state": "unmodeled",
            "queue_slo_verified": False,
            "observed_slo_met": None,
            "sigma": -1.0,
            "solver_gain": 2.0,
        }

        action = Plan.from_raw({"actions": [raw_action]}, tick=1).actions[0]
        serialized = action.to_dict()

        for field_name in (
            "admitted_tps",
            "achieved_tps",
            "unmet_tps",
            "meets_target",
            "served_fraction",
            "admission_mode",
            "prediction_assessment",
            "point_capacity_covers_target",
            "base_latency_within_target",
            "queue_state",
            "queue_slo_verified",
        ):
            self.assertEqual(serialized[field_name], raw_action[field_name])
        self.assertNotIn("sigma", serialized)
        self.assertNotIn("solver_gain", serialized)

        reparsed = Plan.from_raw({"actions": [serialized]}, tick=1).actions[0]
        self.assertEqual(reparsed.to_dict(), serialized)

        legacy = Plan.from_raw(
            {"actions": [{"job_id": "job_legacy", "type": "defer"}]}, tick=1
        ).actions[0]
        self.assertNotIn("admitted_tps", legacy.to_dict())

    def test_rank_rejects_nonpositive_replica_count(self):
        for replicas in (0, -1, False, 1.5):
            with self.subTest(replicas=replicas), self.assertRaisesRegex(ValueError, "n_replicas"):
                RankSpec.from_dict(
                    {
                        "role": "aggregate",
                        "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                        "config": {"gpu_count": 1},
                        "n_replicas": replicas,
                    }
                )

    def test_plan_action_autofills_and_preserves_rank_ids(self):
        plan = Plan.from_raw(
            {
                "actions": [
                    {
                        "job_id": "job_online",
                        "type": "place",
                        "ladder": [
                            {
                                "role": "aggregate",
                                "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                                "config": {"gpu_count": 1},
                                "n_replicas": 1,
                                "predicted_y": {"p99_ttft_ms": 120.0},
                                "predicted_v": {"kv_cache_util": 0.4},
                            },
                            {
                                "role": "aggregate",
                                "rank_id": "latency_rank",
                                "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                                "config": {"gpu_count": 1},
                                "n_replicas": 1,
                            },
                        ],
                    }
                ]
            },
            tick=1,
        )

        action = plan.actions[0]
        self.assertTrue(action.ladder[0].rank_id.startswith("rank_"))
        self.assertNotEqual(action.ladder[0].rank_id, "latency_rank")
        self.assertEqual(action.ladder[1].rank_id, "latency_rank")
        self.assertEqual(
            [rank["rank_id"] for rank in action.to_dict()["ladder"]],
            [action.ladder[0].rank_id, "latency_rank"],
        )
        self.assertEqual(action.ladder[0].predicted_y, {"p99_ttft_ms": 120.0})
        self.assertEqual(action.ladder[0].predicted_v, {"kv_cache_util": 0.4})
        self.assertEqual(action.to_dict()["ladder"][0]["predicted_y"], {"p99_ttft_ms": 120.0})

    def test_plan_action_rejects_duplicate_rank_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate rank_id"):
            Plan.from_raw(
                {
                    "actions": [
                        {
                            "job_id": "job_online",
                            "type": "place",
                            "ladder": [
                                {
                                    "role": "aggregate",
                                    "rank_id": "rank_1",
                                    "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                                    "config": {"gpu_count": 1},
                                },
                                {
                                    "role": "aggregate",
                                    "rank_id": "rank_1",
                                    "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                                    "config": {"gpu_count": 1},
                                },
                            ],
                        }
                    ]
                },
                tick=1,
            )

    def test_plan_autofilled_rank_ids_are_global(self):
        plan = Plan.from_raw(
            {
                "actions": [
                    {
                        "job_id": "job_a",
                        "type": "place",
                        "ladder": [
                            {
                                "role": "aggregate",
                                "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                                "config": {"gpu_count": 1},
                            }
                        ],
                    },
                    {
                        "job_id": "job_b",
                        "type": "place",
                        "ladder": [
                            {
                                "role": "aggregate",
                                "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                                "config": {"gpu_count": 1},
                            }
                        ],
                    },
                ]
            },
            tick=1,
        )

        rank_ids = [action.ladder[0].rank_id for action in plan.actions]
        self.assertEqual(len(set(rank_ids)), 2)

    def test_plan_rejects_rank_ids_reused_across_jobs(self):
        with self.assertRaisesRegex(ValueError, "duplicate rank_id"):
            Plan.from_raw(
                {
                    "actions": [
                        {
                            "job_id": "job_a",
                            "type": "place",
                            "ladder": [{"role": "aggregate", "rank_id": "rank_shared"}],
                        },
                        {
                            "job_id": "job_b",
                            "type": "place",
                            "ladder": [{"role": "aggregate", "rank_id": "rank_shared"}],
                        },
                    ]
                },
                tick=1,
            )

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
        self.assertTrue(registry.archive_mechanism(pd_id, "demo archive"))
        self.assertIn(pd_id, registry.mechanisms_by_status["archived"])

    def test_mechanism_registry_matches_scope_values(self):
        registry = MechanismRegistry()
        prefix = Mechanism(
            edge_ids=["prefix_cache_enabled->kvcache_hit_rate"],
            scope={
                "x": [
                    "prefix_cache_enabled",
                    "shared_prefix_length_avg",
                    "workload_prefix_concentration",
                ],
                "v": ["kvcache_hit_rate"],
                "workload_type": "online",
                "model_type": "any",
                "conditions": [{"feature": "shared_prefix_length_avg", "op": ">", "value": 256}],
            },
            narrative="Shared prefixes can benefit from prefix caching.",
        )
        burst = Mechanism(
            edge_ids=["peak_to_mean_ratio->depth_req_q"],
            scope={
                "x": ["request_arrival_rate", "peak_to_mean_ratio", "max_num_seq"],
                "v": ["depth_req_q"],
                "workload_type": "online",
                "model_type": "any",
                "conditions": [{"feature": "peak_to_mean_ratio", "op": ">", "value": 2}],
            },
            narrative="Bursts build queues.",
        )
        dense = Mechanism(
            edge_ids=["tp->comm_overhead_pct"],
            scope={"x": ["tp"], "v": [], "model_type": "dense_large"},
            narrative="Dense model communication.",
        )
        moe = Mechanism(
            edge_ids=["ep->comm_overhead_pct"],
            scope={"x": ["ep"], "v": [], "model_type": "moe"},
            narrative="MoE communication.",
        )

        exact = registry.match_scope(
            prefix,
            {
                "type": "online",
                "prefix_cache_enabled": True,
                "shared_prefix_length_avg": 500,
                "workload_prefix_concentration": 0,
            },
        )
        self.assertEqual(exact["quality"], "exact")

        self.assertEqual(
            registry.match_scope(
                burst,
                {
                    "type": "online",
                    "request_arrival_rate": 1.0,
                    "peak_to_mean_ratio": 2,
                    "max_num_seq": 256,
                },
            )["quality"],
            "reject",
        )
        self.assertEqual(
            registry.match_scope(prefix, {"type": "online", "prefix_cache_enabled": True})[
                "quality"
            ],
            "partial",
        )
        self.assertEqual(
            registry.match_scope(dense, {"tp": 2})["quality"],
            "reject",
        )
        self.assertEqual(
            registry.match_scope(moe, {"ep": 2, "is_moe": True})["quality"],
            "exact",
        )
        self.assertEqual(
            registry.match_scope(moe, {"ep": 2, "is_moe": False})["quality"],
            "reject",
        )
        self.assertEqual(
            registry.match_scope(
                Mechanism(edge_ids=[], scope={"x": ["tp"], "v": []}, narrative="exact key"),
                {"throughput_token_per_sec": 100},
            )["quality"],
            "reject",
        )

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
            reloaded = EvidenceService(user_id=user_id, postgres_client=client)
            persisted = reloaded.get_row("job_1", "rank_1")[0]
            self.assertEqual(persisted.deployment_id, "deploy-row_1")
            self.assertEqual(persisted.evidence_available_timestamp_utc, 1.5)
            self.assertEqual(persisted.prediction_lineage["composite_version"], "test-v1")
        finally:
            with client.begin() as session:
                session.execute(
                    text("delete from users where user_id = :user_id"), {"user_id": user_id}
                )

    def test_persisted_lineage_drives_calibration_after_reload(self):
        client = PostgresClient()
        user_id = new_user_id()
        config = {
            "model_id": "model",
            "gpu_type": "H100",
            "weight_dtype": "bf16",
            "engine_name": "vllm",
            "engine_version": "0.22.0",
            "dp": 1,
        }
        features = {"type": "online"}
        context = build_prediction_context(config, features, scenario="peak")
        with client.begin() as session:
            session.add(
                UserRow(
                    user_id=user_id,
                    name="koi calibration smoke",
                    created_at=datetime.now(UTC),
                )
            )

        try:
            store = EvidenceService(user_id=user_id, postgres_client=client)
            for index in range(5):
                row = make_row(f"cal-{index}", index + 1)
                row.y_predicted = {"throughput_token_per_sec": 100.0}
                row.y_observed_mean = {"throughput_token_per_sec": 120.0}
                row.y_observed_trajectory = {"throughput_token_per_sec": np.array([120.0])}
                row.residuals_per_y = {"throughput_token_per_sec": np.array([20.0])}
                row.deployment_id = f"deploy-cal-{index}"
                row.evidence_available_timestamp_utc = float(index + 1)
                row.prediction_lineage = {
                    "schema_version": 3,
                    "composite_version": "test-v1",
                    "context": context,
                    "pre_calibration": {
                        "y_hat": {"throughput_token_per_sec": 100.0},
                        "v_hat": {},
                    },
                }
                store.append_row(row)

            reloaded = EvidenceService(user_id=user_id, postgres_client=client)
            result = calibrate_prediction(
                {"throughput_token_per_sec": 100.0},
                {},
                config,
                features,
                reloaded,
                "test-v1",
                scenario="peak",
                as_of_timestamp_utc=100.0,
            )

            self.assertEqual(result.status, "learned")
            self.assertGreater(result.y_hat["throughput_token_per_sec"], 100.0)
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
