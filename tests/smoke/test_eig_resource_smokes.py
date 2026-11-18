import contextlib
import io
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from src.agent.tools import agent_tools
from src.core.candidate_graph import CandidateGraph
from src.core.confidence_service import ConfidenceService
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node
from src.exploration.eig import aggregate_cluster_eig, compute_eig
from src.infra.resource_map import (
    ResourceMapManager,
    _run_smoke,
    _run_used_capacity_check,
    _SmokeResourceMapManager,
)


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
        self.assertEqual(used["free"], 72)
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

            def get_running_chains(self, user_id=None):
                shape = {"count": 2, "gpu_count": 2, "instance_type": "p4d.24xlarge"}
                return [
                    {"chain_id": "chain_a", "target_node": env, "shape_json": dict(shape)},
                    {"chain_id": "chain_b", "target_node": env, "shape_json": dict(shape)},
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

    def test_duplicate_p5_pools_aggregate_before_subtracting_running_instances(self):
        env = "reserved|aws|us-east-1|use1-az1|H100"
        first_pool = {
            "instance_type": "p5.48xlarge",
            "gpu_type": "H100",
            "gpus_per_instance": 8,
            "total_instances": 3,
            "total": 24,
            "allocation_kind": "instance",
            "price_per_instance_hour": 98.32,
            "fabric": "first-fabric",
        }

        class FakeResourceMap:
            market = ("reserved",)

            @staticmethod
            def scheduling_summary():
                return {
                    env: {
                        "gpu_type": "H100",
                        "total": 48,
                        "pools": [
                            first_pool,
                            {
                                **first_pool,
                                "price_per_instance_hour": 99.0,
                                "fabric": "second-fabric",
                            },
                        ],
                    }
                }

        class SmokeManager(ResourceMapManager):
            def __init__(self):
                super().__init__(user_id="duplicate_p5_pool_smoke")

            def get_resource_map(self, user_id=None):
                return FakeResourceMap()

            def get_running_chains(self, user_id=None):
                return [
                    {
                        "chain_id": f"chain_{index}",
                        "target_node": env,
                        "shape_json": {"count": 1, "instance_type": "p5.48xlarge"},
                    }
                    for index in range(3)
                ]

        manager = SmokeManager()
        resources = manager.resources_summary()
        pool = resources[env]["pools"][0]

        self.assertEqual(len(resources[env]["pools"]), 1)
        self.assertEqual(pool["total_instances"], 6)
        self.assertEqual(pool["total"], 48)
        self.assertEqual(pool["free_instances"], 3)
        self.assertEqual(pool["free"], 24)
        self.assertEqual(resources[env]["free"], 24)
        self.assertEqual(pool["price_per_instance_hour"], 98.32)
        self.assertEqual(pool["fabric"], "first-fabric")
        self.assertEqual(pool["merged_pool_count"], 2)
        self.assertEqual(
            [source["price_per_instance_hour"] for source in pool["merged_pool_sources"]],
            [98.32, 99.0],
        )

        capacity = manager.pool_capacity(resources)[(env, "p5.48xlarge")]
        self.assertEqual(capacity["available_units"], 3)
        self.assertEqual(capacity["free_gpus"], 24)

        with patch.object(agent_tools, "get_resource_map", return_value=resources):
            catalog = agent_tools.instance_catalog()
        self.assertEqual(catalog[env]["p5.48xlarge"]["free_instances"], 3)
        self.assertEqual(catalog[env]["p5.48xlarge"]["gpus_per_instance"], 8)

    def test_duplicate_a100_nc_and_nd_pools_aggregate_independently(self):
        env = "reserved|azure|eastus|zone-1|A100"
        nc_pool = {
            "instance_type": "Standard_NC24ads_A100_v4",
            "gpus_per_instance": 1,
            "total_instances": 2,
            "total": 2,
            "allocation_kind": "instance",
        }
        nd_pool = {
            "instance_type": "Standard_ND96amsr_A100_v4",
            "gpus_per_instance": 8,
            "total_instances": 3,
            "total": 24,
            "allocation_kind": "instance",
        }

        class FakeResourceMap:
            market = ("reserved",)

            @staticmethod
            def scheduling_summary():
                return {
                    env: {
                        "gpu_type": "A100",
                        "total": 52,
                        "pools": [nc_pool, nd_pool, dict(nc_pool), dict(nd_pool)],
                    }
                }

        resources = ResourceMapManager._normalized_scheduling_summary(FakeResourceMap())[env]
        pools = {pool["instance_type"]: pool for pool in resources["pools"]}

        self.assertEqual(len(pools), 2)
        self.assertEqual(pools["Standard_NC24ads_A100_v4"]["total_instances"], 4)
        self.assertEqual(pools["Standard_NC24ads_A100_v4"]["total"], 4)
        self.assertEqual(pools["Standard_NC24ads_A100_v4"]["free_instances"], 4)
        self.assertEqual(pools["Standard_NC24ads_A100_v4"]["free"], 4)
        self.assertEqual(pools["Standard_NC24ads_A100_v4"]["merged_pool_count"], 2)
        self.assertEqual(pools["Standard_ND96amsr_A100_v4"]["total_instances"], 6)
        self.assertEqual(pools["Standard_ND96amsr_A100_v4"]["total"], 48)
        self.assertEqual(pools["Standard_ND96amsr_A100_v4"]["free_instances"], 6)
        self.assertEqual(pools["Standard_ND96amsr_A100_v4"]["free"], 48)
        self.assertEqual(resources["free"], 52)

    def test_duplicate_pool_shape_conflicts_fail_clearly(self):
        env = "reserved|aws|us-east-1|use1-az1|H100"
        first_pool = {
            "instance_type": "p5.48xlarge",
            "gpus_per_instance": 8,
            "total_instances": 3,
            "total": 24,
            "allocation_kind": "instance",
        }

        class FakeResourceMap:
            market = ("reserved",)

            def __init__(self, second_pool):
                self.second_pool = second_pool

            def scheduling_summary(self):
                return {
                    env: {
                        "gpu_type": "H100",
                        "total": 48,
                        "pools": [first_pool, self.second_pool],
                    }
                }

        conflicts = [
            ({**first_pool, "allocation_kind": "gpu"}, "conflicting allocation kind"),
            ({**first_pool, "gpus_per_instance": 4}, "conflicting gpus_per_instance"),
        ]
        for second_pool, message in conflicts:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                ResourceMapManager._normalized_scheduling_summary(FakeResourceMap(second_pool))

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

            def get_running_chains(self, user_id=None):
                return []

        manager = SmokeManager()
        resources = manager.resources_summary()[env]
        pools = {pool["instance_type"]: pool for pool in resources["pools"]}

        self.assertEqual(resources["free"], 16)
        self.assertEqual(pools["g6e.xlarge"]["total_instances"], 4)
        self.assertEqual(pools["g6e.xlarge"]["total"], 4)
        self.assertEqual(pools["g6e.xlarge"]["free_instances"], 4)
        self.assertEqual(pools["g6e.xlarge"]["free"], 4)
        self.assertEqual(pools["g6e.12xlarge"]["total_instances"], 3)
        self.assertEqual(pools["g6e.12xlarge"]["total"], 12)
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
                                "total": 2,
                                "total_instances": 2,
                            },
                            {
                                "instance_type": "gpu-pool",
                                "allocation_unit": "gpu",
                                "gpus_per_unit": 1,
                                "total": 2,
                                "total_instances": 2,
                            },
                        ],
                    }
                }

        class SmokeManager(ResourceMapManager):
            def __init__(self):
                super().__init__(user_id="gpu_pool_smoke")

            def get_resource_map(self, user_id=None):
                return FakeResourceMap()

            def get_running_chains(self, user_id=None):
                return [
                    {
                        "chain_id": "chain_1",
                        "target_node": env,
                        "shape_json": {"instance_type": "gpu-pool", "count": 2},
                    }
                ]

        resources = SmokeManager().resources_summary()[env]
        pool = resources["pools"][0]
        self.assertEqual(len(resources["pools"]), 1)
        self.assertEqual(pool["total_instances"], 4)
        self.assertEqual(pool["total"], 4)
        self.assertEqual(pool["free"], 2)
        self.assertEqual(resources["free"], 2)


if __name__ == "__main__":
    unittest.main()
