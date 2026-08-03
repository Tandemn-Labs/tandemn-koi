import contextlib
import io
import unittest
from dataclasses import dataclass
from typing import Any

from src.core.candidate_graph import CandidateGraph
from src.core.confidence_service import ConfidenceService
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node, Plan, RankSpec
from src.cost.switch_cost import RankEntry, hourly_rate
from src.exploration.eig import aggregate_cluster_eig, compute_eig
from src.infra.resource_map import (
    ClusterResourceSnapshot,
    ResourceMapManager,
    _run_smoke,
    _run_used_capacity_check,
    _SmokeResourceMapManager,
)
from src.validation.validator import Validator


@dataclass
class Rank:
    mechanism_id: str
    mechanism: Mechanism
    config: dict[str, Any]
    n_replicas: int
    ladder: Any = None
    evidence_store: Any = None


@dataclass
class Ladder:
    ranks: list[Rank]
    duration_minutes: int
    applicable_mechanisms: list[Mechanism]

    def envs(self):
        return ["env_a", "env_b", "env_c"]


class FakeEvidenceStore:
    def __init__(self, edge_envs):
        self.edge_envs = edge_envs

    def envs_for_edge(self, edge_id):
        return self.edge_envs.get(edge_id, [])


def build_eig_case():
    e1 = Edge("edge_batch_to_kv", "batch_size", "kv_cache_pressure", "X", "V")
    e2 = Edge("edge_kv_to_latency", "kv_cache_pressure", "ttft_ms", "V", "Y")
    mechanism = Mechanism(
        mechanism_id="mech_kv_latency",
        edge_ids=[e1.edge_id, e2.edge_id],
        scope={"x": ["batch_size"], "v": ["kv_cache_pressure"]},
        narrative="KV pressure mediates batch size and TTFT.",
    )
    ranks = [
        Rank(
            mechanism.mechanism_id, mechanism, {"batch_size": 8, "kv_cache_pressure": "medium"}, 30
        ),
        Rank(
            mechanism.mechanism_id, mechanism, {"batch_size": 16, "kv_cache_pressure": "high"}, 30
        ),
    ]
    evidence_store = FakeEvidenceStore(
        {e1.edge_id: ["env_a", "env_b"], e2.edge_id: ["env_a", "env_b"]}
    )
    ladder = Ladder(ranks=ranks, duration_minutes=10, applicable_mechanisms=[mechanism])
    for rank in ranks:
        rank.ladder = ladder
        rank.evidence_store = evidence_store

    graph = CandidateGraph(
        node_table={
            "batch_size": Node("batch_size", "X"),
            "kv_cache_pressure": Node("kv_cache_pressure", "V"),
            "ttft_ms": Node("ttft_ms", "Y"),
        },
        edge_table={e1.edge_id: e1, e2.edge_id: e2},
        edge_metadata_table={
            e1.edge_id: EdgeMetadata(e1.edge_id, alpha=1.5, beta=1.5, visit_count=3),
            e2.edge_id: EdgeMetadata(e2.edge_id, alpha=5.6, beta=2.4, visit_count=8),
        },
    )
    registry = MechanismRegistry(
        mechanism_table={mechanism.mechanism_id: mechanism},
        mechanism_metadata_table={
            mechanism.mechanism_id: MechanismMetadata(
                mechanism_id=mechanism.mechanism_id,
                alpha=3.0,
                beta=2.0,
                visit_count=5,
            )
        },
    )
    return ladder, graph, registry, ConfidenceService(graph, registry), evidence_store, ranks


class EigResourceSmokeTests(unittest.TestCase):
    def test_eig_smoke(self):
        ladder, graph, registry, confidence_service, evidence_store, ranks = build_eig_case()

        alpha = compute_eig(ladder, graph, registry, confidence_service, evidence_store)
        alpha_cluster = aggregate_cluster_eig(
            cluster_plan={"job_1": "fake_action"},
            ranks=ranks,
            candidate_graph=graph,
            mechanism_registry=registry,
            confidence_service=confidence_service,
            evidence_store=evidence_store,
        )

        self.assertGreater(alpha, 0.0)
        self.assertGreater(alpha_cluster, 0.0)

    def test_resource_map_in_memory_smokes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_smoke(_SmokeResourceMapManager(), "in-memory")
            used = _run_used_capacity_check()

        self.assertEqual(result["label"], "in-memory")
        self.assertEqual(result["active_jobs"], 0)
        self.assertEqual(result["pending_jobs"], 0)
        self.assertEqual(used["free"], 64)
        self.assertEqual(used["total"], 80)

    def test_underfilled_cloud_instances_consume_full_instance_capacity(self):
        env = "reserved|aws|us-east-2|use2-az3|A100"

        class FakeResourceMap:
            market = ("reserved",)

            def scheduling_summary(self):
                return {
                    env: {
                        "market": "reserved",
                        "cloud": "aws",
                        "region": "us-east-2",
                        "zone": "use2-az3",
                        "gpu_type": "A100",
                        "total": 16,
                        "pools": [
                            {
                                "instance_type": "p4d.24xlarge",
                                "gpu_type": "A100",
                                "gpus_per_instance": 8,
                                "total_instances": 2,
                                "allocation_kind": "instance",
                            }
                        ],
                    }
                }

        class SmokeManager(ResourceMapManager):
            def __init__(self):
                super().__init__(user_id="underfilled_instance_smoke")

            def get_resource_map(self, user_id=None):
                return FakeResourceMap()

            def get_running_ranks(self, user_id=None):
                shape = {"count": 2, "gpu_count": 2, "instance_type": "p4d.24xlarge"}
                return [
                    {
                        "rank_id": "rank_a",
                        "n_replicas": 2,
                        "target_node": env,
                        "shape_json": dict(shape),
                    },
                ]

            def get_running_jobs(self, user_id=None):
                return []

            def get_waiting_jobs(self, user_id=None):
                return []

        manager = SmokeManager()
        resources = manager.resources_summary()
        self.assertEqual(resources[env]["free"], 0)
        self.assertEqual(resources[env]["pools"][0]["free_instances"], 0)
        self.assertEqual(resources[env]["pools"][0]["free"], 0)

        plan = {
            "actions": [
                {
                    "job_id": "job_new",
                    "type": "place",
                    "ladder": [
                        {
                            "role": "aggregate",
                            "env": env.split("|"),
                            "config": {"gpu_count": 2, "instance_type": "p4d.24xlarge"},
                            "n_replicas": 1,
                        }
                    ],
                }
            ]
        }
        ok, violations = manager.check_resource_feasibility(plan)
        self.assertFalse(ok)
        self.assertEqual(
            violations,
            [
                "env reserved|aws|us-east-2|use2-az3|A100 pool p4d.24xlarge: "
                "requested 1 instances, only 0 free"
            ],
        )

    def test_multi_instance_replica_reserves_and_prices_every_instance(self):
        env = "reserved|aws|us-east-2|use2-az3|A100"
        resources = {
            env: {
                "gpu_type": "A100",
                "total": 24,
                "free": 24,
                "pools": [
                    {
                        "instance_type": "p4d.24xlarge",
                        "gpus_per_instance": 8,
                        "allocation_kind": "instance",
                        "price_per_instance_hour": 10.0,
                        "free": 24,
                    }
                ],
            }
        }
        rank = RankSpec.from_dict(
            {
                "role": "aggregate",
                "env": env.split("|"),
                "config": {
                    "instance_type": "p4d.24xlarge",
                    "gpu_count": 16,
                    "tp": 16,
                    "pp": 1,
                },
            }
        )
        manager = ResourceMapManager(user_id="multi_instance_smoke")

        allocation = manager.rank_allocation_summary(rank, resources)
        by_env, by_pool = manager.requested_capacity(
            {
                "actions": [
                    {
                        "job_id": "job_1",
                        "type": "place",
                        "ladder": [rank.to_dict()],
                    }
                ]
            },
            resources,
        )
        pricing = manager.switch_pricing_map(resources)
        switch_rank = RankEntry(("aggregate", tuple(env.split("|")), ""), rank.config, env, 1)

        self.assertEqual(allocation["units_per_replica"], 2)
        self.assertEqual(allocation["capacity_per_replica"], 16)
        self.assertEqual(allocation["price_per_unit_hour"], 20.0)
        self.assertEqual(by_env[env], 16)
        self.assertEqual(by_pool[(env, "p4d.24xlarge")], {"units": 2, "gpus": 16})
        self.assertEqual(hourly_rate(switch_rank, pricing), 20.0)
        typed = Plan.from_raw(
            {
                "actions": [
                    {
                        "job_id": "job_1",
                        "type": "place",
                        "ladder": [rank.to_dict()],
                    }
                ]
            },
            tick=1,
        )
        snapshot = ClusterResourceSnapshot(1, resources, [], [])
        self.assertEqual(Validator(resource_map=manager)._check_chain_physics(typed, snapshot), [])

    def test_mixed_instance_pools_expose_free_instances(self):
        env = "reserved|aws|us-east-1|us-east-1b|L40S"

        class FakeResourceMap:
            market = ("reserved",)

            def scheduling_summary(self):
                return {
                    env: {
                        "gpu_type": "L40S",
                        "total": 16,
                        "total_instances": 7,
                        "pools": [
                            {
                                "instance_type": "g6e.xlarge",
                                "gpus_per_instance": 1,
                                "total_instances": 4,
                            },
                            {
                                "instance_type": "g6e.12xlarge",
                                "gpus_per_instance": 4,
                                "total_instances": 3,
                            },
                        ],
                    }
                }

        class SmokeManager(ResourceMapManager):
            def __init__(self):
                super().__init__(user_id="mixed_pool_smoke")

            def get_resource_map(self, user_id=None):
                return FakeResourceMap()

            def get_running_ranks(self, user_id=None):
                return []

        manager = SmokeManager()
        resources = manager.resources_summary()[env]
        pools = {pool["instance_type"]: pool for pool in resources["pools"]}

        self.assertEqual(resources["free"], 16)
        self.assertEqual(pools["g6e.xlarge"]["free_instances"], 4)
        self.assertEqual(pools["g6e.xlarge"]["free"], 4)
        self.assertEqual(pools["g6e.12xlarge"]["free_instances"], 3)
        self.assertEqual(pools["g6e.12xlarge"]["free"], 12)

        plan = {
            "actions": [
                {
                    "job_id": "job_new",
                    "type": "place",
                    "ladder": [
                        {
                            "role": "aggregate",
                            "env": env.split("|"),
                            "config": {
                                "instance_type": "g6e.12xlarge",
                                "gpu_count": 2,
                                "tp": 2,
                                "pp": 1,
                            },
                            "n_replicas": 4,
                        }
                    ],
                }
            ]
        }
        future = manager.simulate_future_resources(plan)[env]
        large_pool = next(
            pool for pool in future["pools"] if pool["instance_type"] == "g6e.12xlarge"
        )
        ok, violations = manager.check_resource_feasibility(plan)

        self.assertEqual(future["free_after"], 0)
        self.assertEqual(large_pool["free_units_after"], -1)
        self.assertFalse(ok)
        self.assertIn("requested 4 instances, only 3 free", violations[0])

        plan["actions"][0]["type"] = "keep"
        self.assertEqual(manager.requested_capacity(plan)[0], {})

    def test_gpu_pool_subtracts_running_gpu_usage(self):
        env = "reserved|onprem|local|rack-1|H100"

        class FakeResourceMap:
            market = ("reserved",)

            @staticmethod
            def scheduling_summary():
                return {
                    env: {
                        "gpu_type": "H100",
                        "total": 4,
                        "pools": [
                            {
                                "instance_type": "gpu-pool",
                                "allocation_unit": "gpu",
                                "gpus_per_unit": 1,
                                "total": 4,
                                "total_instances": 4,
                            }
                        ],
                    }
                }

        class SmokeManager(ResourceMapManager):
            def __init__(self):
                super().__init__(user_id="gpu_pool_smoke")

            def get_resource_map(self, user_id=None):
                return FakeResourceMap()

            def get_running_ranks(self, user_id=None):
                return [
                    {
                        "rank_id": "rank_1",
                        "n_replicas": 1,
                        "target_node": env,
                        "shape_json": {"instance_type": "gpu-pool", "count": 2},
                    }
                ]

        pool = SmokeManager().resources_summary()[env]["pools"][0]
        self.assertEqual(pool["free"], 2)


if __name__ == "__main__":
    unittest.main()
