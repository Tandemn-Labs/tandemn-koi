import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from src.agent.tools import agent_tools
from src.core.candidate_graph import CandidateGraph
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, Mechanism, Node, PlanAction, RankSpec
from src.infra.resource_map import ClusterResourceSnapshot, ResourceMapManager
from src.prediction.surrogate import SurrogateExecutionError, SurrogateMemoryNoFit


class _DRO:
    def compute_dro_band(self, y_hat):
        return {}


class _ResourceMap:
    def snapshot(self):
        return _Snapshot()

    def resources_summary(self):
        return {
            "reserved|aws|us-east-1|use1-az1|H100": {
                "free": 8,
                "gpu_type": "H100",
                "pools": [
                    {
                        "instance_type": "p5.48xlarge",
                        "gpu_type": "H100",
                        "gpus_per_instance": 8,
                        "fabric_type": "efa",
                    }
                ],
            }
        }

    def hardware_catalog(self):
        return {
            "regions": [
                {
                    "cloud": "aws",
                    "region": "us-east-1",
                    "instance_types": [
                        {
                            "instance_type": "p5.48xlarge",
                            "accelerators": [
                                {
                                    "kind": "gpu",
                                    "name": "H100",
                                    "canonical_gpu_name": "H100",
                                    "count": 8,
                                    "memory_mib_each": 81920,
                                    "gpu_bandwidth_gbps": 3350,
                                    "gpu_tflops_fp16": 989.5,
                                    "cuda_compute_capability": "9.0",
                                    "gpu_generation": "Hopper",
                                    "nvlink_bandwidth_gbps": 900,
                                    "pcie_bandwidth_gbps": 128,
                                    "gpu_watts": 700,
                                }
                            ],
                            "network": {"network_cards": [{"peak_bandwidth_gbps": 3200}]},
                        }
                    ],
                }
            ]
        }

    def rank_allocation_summary(self, rank, resources=None):
        gpus = rank.gpus_per_chain()
        return {
            "allocation_kind": "gpu",
            "instance_type": rank.config.get("instance_type"),
            "gpus_per_unit": gpus,
            "price_per_unit_hour": None,
            "capacity_per_replica": gpus,
        }

    def model_catalog(self, model_id):
        return {
            "model_id": model_id,
            "model_params_b": 70.0,
            "hidden_size": 8192,
            "engine_name": "vllm",
            "max_num_seq": [{"gpu_type": "H100", "value": 256}],
            "max_num_batched_tokens": [{"gpu_type": "H100", "value": 8192}],
            "block_size": [{"gpu_type": "H100", "value": 16}],
            "kvcache_dtype": [{"gpu_type": "H100", "value": "auto"}],
        }


class _EvidenceStore:
    def get_rows_for_job(self, job_id):
        return []

    def retrieve_similar_rows(self, job_features, top_k=10):
        return []


class _MechanismRegistry:
    def find_applicable(self, context, require_x_overlap=True):
        return []


class _ConfidenceService:
    pass


class _RecordingSurrogate:
    def __init__(self):
        self.calls = []

    def compose_prediction(
        self, job_config, job_features, candidate_graph, method=("AIC_DynoSim",), scenario="mean"
    ):
        self.calls.append((dict(job_config), dict(job_features)))
        return (
            {
                "p99_ttft_ms": 10.0,
                "p99_tpot_ms": 1.0,
                "throughput_token_per_sec": 1000.0,
            },
            {"kv_cache_util": 0.4},
        )


class AgentToolsSmokeTests(unittest.TestCase):
    def test_rank_prediction_payload_attaches_allocation_price(self):
        class PricedResourceMap(_ResourceMap):
            def resources_summary(self):
                resources = super().resources_summary()
                resources["reserved|aws|us-east-1|use1-az1|H100"]["pools"][0][
                    "price_per_instance_hour"
                ] = 55.04
                return resources

            def rank_allocation_summary(self, rank, resources=None):
                summary = super().rank_allocation_summary(rank, resources)
                summary["price_per_unit_hour"] = 55.04
                return summary

        saved = agent_tools._CTX.resource_map
        try:
            agent_tools._CTX.resource_map = PricedResourceMap()
            rank = RankSpec.from_dict(
                {
                    "role": "aggregate",
                    "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                    "config": {
                        "instance_type": "p5.48xlarge",
                        "gpu_count": 8,
                        "tp": 8,
                        "pp": 1,
                    },
                    "n_replicas": 3,
                }
            )

            payload = agent_tools._rank_prediction_payload(
                rank,
                {
                    "model_id": "Qwen/Qwen2.5-72B-Instruct",
                    "request_arrival_rate": 1.0,
                    "isl_token_avg": 100,
                    "osl_token_avg": 100,
                },
            )
        finally:
            agent_tools._CTX.resource_map = saved

        self.assertEqual(payload["job_config"]["price_per_hour"], 55.04)

    def test_set_new_mechanisms_uses_canonical_validation(self):
        xv = Edge("tp->kv_cache_util", "tp", "kv_cache_util", "X", "V")
        vy = Edge(
            "kv_cache_util->p99_tpot_ms",
            "kv_cache_util",
            "p99_tpot_ms",
            "V",
            "Y",
        )
        graph = CandidateGraph(
            {
                "tp": Node("tp", "X"),
                "kv_cache_util": Node("kv_cache_util", "V"),
                "p99_tpot_ms": Node("p99_tpot_ms", "Y"),
            },
            {xv.edge_id: xv, vy.edge_id: vy},
            {
                xv.edge_id: EdgeMetadata(xv.edge_id),
                vy.edge_id: EdgeMetadata(vy.edge_id),
            },
        )
        registry = MechanismRegistry()

        class Confidence:
            def __init__(self):
                self.seeded = []

            def seed_new_mechanism_confidence(self, mechanism_id):
                self.seeded.append(mechanism_id)
                return 0.5

        confidence = Confidence()
        saved = (
            agent_tools._CTX.candidate_graph,
            agent_tools._CTX.mechanism_registry,
            agent_tools._CTX.confidence_service,
        )
        try:
            agent_tools._CTX.candidate_graph = graph
            agent_tools._CTX.mechanism_registry = registry
            agent_tools._CTX.confidence_service = confidence
            empty = agent_tools.set_new_mechanisms(
                [],
                {"x": ["tp"], "v": ["kv_cache_util"]},
                "Empty bundle.",
            )
            malformed = agent_tools.set_new_mechanisms(
                [xv.edge_id],
                {
                    "x": ["tp"],
                    "v": ["kv_cache_util"],
                    "conditions": [{"feature": "tp", "op": "!=", "value": 1}],
                },
                "Malformed condition.",
            )
            set_scope = agent_tools.set_new_mechanisms(
                [xv.edge_id, vy.edge_id],
                {"x": {"tp"}, "v": ["kv_cache_util"]},
                "Non-serializable scope.",
            )
            valid = agent_tools.set_new_mechanisms(
                [xv.edge_id, vy.edge_id],
                {
                    "x": ["tp"],
                    "v": ["kv_cache_util"],
                    "workload_type": "online",
                    "model_type": "any",
                },
                "Tensor parallelism changes KV pressure and TPOT.",
            )
        finally:
            (
                agent_tools._CTX.candidate_graph,
                agent_tools._CTX.mechanism_registry,
                agent_tools._CTX.confidence_service,
            ) = saved

        self.assertFalse(empty["ok"])
        self.assertFalse(malformed["ok"])
        self.assertFalse(set_scope["ok"])
        self.assertEqual(len(registry.mechanism_table), 1)
        self.assertTrue(valid["ok"])
        self.assertEqual(confidence.seeded, [valid["mechanism_id"]])
        stored = registry.get_mechanism(valid["mechanism_id"])
        self.assertEqual(
            stored.scope,
            {
                "x": ["tp"],
                "v": ["kv_cache_util"],
                "workload_type": "online",
                "model_type": "any",
                "conditions": [],
            },
        )
        self.assertEqual(
            registry.match_scope(stored, {"type": "online", "tp": 2})["quality"],
            "exact",
        )

    def test_get_scope_uses_condition_values(self):
        registry = MechanismRegistry()
        prefix = Mechanism(
            edge_ids=["prefix_cache_enabled->kvcache_hit_rate"],
            scope={
                "x": ["prefix_cache_enabled", "shared_prefix_length_avg"],
                "v": ["kvcache_hit_rate"],
                "workload_type": "online",
                "conditions": [{"feature": "shared_prefix_length_avg", "op": ">", "value": 256}],
            },
            narrative="Shared prefixes can benefit from prefix caching.",
        )
        burst = Mechanism(
            edge_ids=["peak_to_mean_ratio->depth_req_q"],
            scope={
                "x": ["peak_to_mean_ratio"],
                "v": ["depth_req_q"],
                "workload_type": "online",
                "conditions": [{"feature": "peak_to_mean_ratio", "op": ">", "value": 2}],
            },
            narrative="Bursts build queues.",
        )
        prefix_id = registry.add_mechanism(prefix)
        registry.add_mechanism(burst)

        class Confidence:
            @staticmethod
            def get_mechanism_confidence(mechanism_id):
                return 0.5

            @staticmethod
            def get_mechanism_visit_count(mechanism_id):
                return 0

        saved = (agent_tools._CTX.mechanism_registry, agent_tools._CTX.confidence_service)
        try:
            agent_tools.bind_tools(mechanism_registry=registry, confidence_service=Confidence())
            matches = agent_tools.get_scope(
                {
                    "type": "online",
                    "shared_prefix_length_avg": 500,
                    "peak_to_mean_ratio": 2,
                }
            )
        finally:
            agent_tools._CTX.mechanism_registry, agent_tools._CTX.confidence_service = saved

        self.assertEqual([match["mechanism_id"] for match in matches], [prefix_id])
        self.assertEqual(matches[0]["match_quality"], "partial")

    def test_get_applicable_mechanisms_uses_rank_dp(self):
        registry = MechanismRegistry()
        mechanism_id = registry.add_mechanism(
            Mechanism(
                edge_ids=["dp->depth_req_q"],
                scope={
                    "x": ["dp", "request_arrival_rate", "priority_class"],
                    "v": ["depth_req_q"],
                    "workload_type": "online",
                },
                narrative="Replica count trades cost for queueing latency.",
            )
        )

        class Confidence:
            @staticmethod
            def get_mechanism_confidence(mechanism_id):
                return 0.5

            @staticmethod
            def get_mechanism_visit_count(mechanism_id):
                return 0

        saved = (
            agent_tools._CTX.mechanism_registry,
            agent_tools._CTX.confidence_service,
            agent_tools._CTX.resource_map,
        )
        try:
            agent_tools._CTX.mechanism_registry = registry
            agent_tools._CTX.confidence_service = Confidence()
            agent_tools._CTX.resource_map = None
            matches = agent_tools.get_applicable_mechanisms(
                {
                    "role": "aggregate",
                    "env": [
                        "reserved",
                        "aws",
                        "us-east-1",
                        "use1-az1",
                        "H100",
                    ],
                    "config": {"tp": 1, "pp": 1, "gpu_count": 1},
                    "n_replicas": 4,
                },
                {
                    "type": "online",
                    "request_arrival_rate": 1.0,
                    "priority_class": "STANDARD",
                },
            )
        finally:
            (
                agent_tools._CTX.mechanism_registry,
                agent_tools._CTX.confidence_service,
                agent_tools._CTX.resource_map,
            ) = saved

        self.assertEqual([match["mechanism_id"] for match in matches], [mechanism_id])
        self.assertEqual(matches[0]["match_quality"], "exact")

    def test_get_influencing_knobs_attaches_structured_scope_matches(self):
        graph = CandidateGraph(
            {
                "tp": Node("tp", "X"),
                "kv_cache_util": Node("kv_cache_util", "V"),
                "p99_tpot_ms": Node("p99_tpot_ms", "Y"),
            },
            {
                "tp->kv_cache_util": Edge("tp->kv_cache_util", "tp", "kv_cache_util", "X", "V"),
                "kv_cache_util->p99_tpot_ms": Edge(
                    "kv_cache_util->p99_tpot_ms",
                    "kv_cache_util",
                    "p99_tpot_ms",
                    "V",
                    "Y",
                ),
            },
            {
                edge_id: EdgeMetadata(edge_id=edge_id)
                for edge_id in ("tp->kv_cache_util", "kv_cache_util->p99_tpot_ms")
            },
        )
        registry = MechanismRegistry()
        partial_id = registry.add_mechanism(
            Mechanism(
                edge_ids=["tp->kv_cache_util"],
                scope={
                    "x": ["tp"],
                    "v": ["kv_cache_util"],
                    "workload_type": "online",
                    "conditions": [{"feature": "tp", "op": ">", "value": 1}],
                },
                narrative="Tensor parallelism changes KV pressure.",
            )
        )
        registry.add_mechanism(
            Mechanism(
                edge_ids=["tp->kv_cache_util"],
                scope={
                    "x": ["peak_to_mean_ratio"],
                    "v": ["kv_cache_util"],
                    "workload_type": "online",
                    "conditions": [{"feature": "peak_to_mean_ratio", "op": ">", "value": 2}],
                },
                narrative="Only bursty workloads use this mechanism.",
            )
        )

        class Confidence:
            @staticmethod
            def get_edge_confidence(edge_id):
                return 0.8

        saved = (
            agent_tools._CTX.candidate_graph,
            agent_tools._CTX.confidence_service,
            agent_tools._CTX.mechanism_registry,
        )
        try:
            agent_tools._CTX.candidate_graph = graph
            agent_tools._CTX.confidence_service = Confidence()
            agent_tools._CTX.mechanism_registry = registry
            knobs = agent_tools.get_influencing_knobs(
                {"type": "online", "peak_to_mean_ratio": 2},
                "p99_tpot_ms",
            )
        finally:
            (
                agent_tools._CTX.candidate_graph,
                agent_tools._CTX.confidence_service,
                agent_tools._CTX.mechanism_registry,
            ) = saved

        self.assertEqual(knobs[0]["knob"], "tp")
        self.assertEqual(knobs[0]["mechanisms"], [partial_id])

    def test_size_ladder_caps_each_instance_pool(self):
        env = "reserved|aws|us-east-1|us-east-1b|L40S"

        class MixedResourceMap:
            def resources_summary(self):
                return {
                    env: {
                        "free": 16,
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

            def rank_allocation_summary(self, rank, resources=None):
                info = (resources or self.resources_summary())[env]
                pool = next(
                    pool
                    for pool in info["pools"]
                    if pool["instance_type"] == rank.config["instance_type"]
                )
                return {
                    "allocation_kind": "instance",
                    "instance_type": pool["instance_type"],
                    "gpus_per_unit": pool["gpus_per_instance"],
                    "price_per_unit_hour": None,
                    "capacity_per_replica": pool["gpus_per_instance"],
                    "free_capacity_gpus": pool["free"],
                    "engine_gpus": rank.gpus_per_chain(),
                }

        saved_context = {
            name: getattr(agent_tools._CTX, name)
            for name in ("resource_map", "surrogate", "candidate_graph", "dro")
        }
        saved_payload = agent_tools._rank_prediction_payload
        saved_predict = agent_tools._predict_outcome_core
        try:
            agent_tools.bind_tools(
                resource_map=MixedResourceMap(),
                surrogate=object(),
                candidate_graph=object(),
                dro=_DRO(),
            )
            agent_tools._rank_prediction_payload = lambda rank, features: {
                "job_config": {},
                "job_features": {},
            }
            agent_tools._predict_outcome_core = lambda config, features: {
                "y_hat": {
                    "p99_ttft_ms": 10.0,
                    "p99_tpot_ms": 1.0,
                    "throughput_token_per_sec": 1000.0,
                }
            }
            features = {
                "type": "online",
                "target_p99_ttft_ms": 100.0,
                "target_p99_tpot_ms": 10.0,
            }

            result = agent_tools.size_ladder(
                [
                    {
                        "role": "aggregate",
                        "env": env.split("|"),
                        "config": {
                            "instance_type": "g6e.xlarge",
                            "gpu_count": 1,
                            "tp": 1,
                            "pp": 1,
                        },
                    },
                    {
                        "role": "aggregate",
                        "env": env.split("|"),
                        "config": {
                            "instance_type": "g6e.12xlarge",
                            "gpu_count": 2,
                            "tp": 2,
                            "pp": 1,
                        },
                    },
                ],
                features,
                target_tps=10_000,
            )

            self.assertEqual(
                [rank["max_replicas_by_capacity"] for rank in result["per_rank"]], [4, 3]
            )
            self.assertEqual([rank["n_replicas"] for rank in result["ranks"]], [4, 3])

            shared = agent_tools.size_ladder(
                [
                    {
                        "role": "aggregate",
                        "env": env.split("|"),
                        "config": {
                            "instance_type": "g6e.12xlarge",
                            "gpu_count": 2,
                            "tp": 2,
                            "pp": 1,
                        },
                    }
                    for _ in range(2)
                ],
                features,
                target_tps=10_000,
            )
            self.assertEqual(
                [rank["max_replicas_by_capacity"] for rank in shared["per_rank"]], [3, 0]
            )
        finally:
            agent_tools._rank_prediction_payload = saved_payload
            agent_tools._predict_outcome_core = saved_predict
            for name, value in saved_context.items():
                setattr(agent_tools._CTX, name, value)

    def test_size_ladder_marks_aic_memory_preflight_failure_no_fit(self):
        env = "reserved|aws|us-east-1|us-east-1b|H100"

        class ResourceMap:
            def resources_summary(self):
                return {
                    env: {
                        "free": 1,
                        "gpu_type": "H100",
                        "pools": [
                            {
                                "instance_type": "p5.4xlarge",
                                "gpus_per_instance": 1,
                                "free_instances": 1,
                                "free": 1,
                            }
                        ],
                    }
                }

            def rank_allocation_summary(self, rank, resources=None):
                return {
                    "allocation_kind": "instance",
                    "instance_type": "p5.4xlarge",
                    "gpus_per_unit": 1,
                    "price_per_unit_hour": 6.88,
                    "capacity_per_replica": 1,
                    "free_capacity_gpus": 1,
                    "engine_gpus": rank.gpus_per_chain(),
                }

        class FailingSurrogate:
            @staticmethod
            def compose_prediction(**_kwargs):
                raise SurrogateMemoryNoFit("AIC memory preflight no-fit: no KV budget")

        saved_context = {
            name: getattr(agent_tools._CTX, name)
            for name in ("resource_map", "surrogate", "candidate_graph", "dro")
        }
        saved_payload = agent_tools._rank_prediction_payload
        try:
            agent_tools.bind_tools(
                resource_map=ResourceMap(),
                surrogate=FailingSurrogate(),
                candidate_graph=object(),
                dro=_DRO(),
            )
            agent_tools._rank_prediction_payload = lambda rank, features: {
                "job_config": {},
                "job_features": {},
            }

            result = agent_tools.size_ladder(
                [
                    {
                        "role": "aggregate",
                        "env": env.split("|"),
                        "config": {
                            "instance_type": "p5.4xlarge",
                            "gpu_count": 1,
                            "tp": 1,
                            "pp": 1,
                        },
                    }
                ],
                {"type": "online", "target_p99_ttft_ms": 100.0, "target_p99_tpot_ms": 10.0},
                target_tps=100,
            )
        finally:
            agent_tools._rank_prediction_payload = saved_payload
            for name, value in saved_context.items():
                setattr(agent_tools._CTX, name, value)

        self.assertEqual(result["ranks"], [])
        self.assertEqual(result["per_rank"][0]["n_replicas"], 0)
        self.assertIn(
            "AIC memory preflight no-fit", result["per_rank"][0]["physical_violations"][0]
        )

    def test_size_ladder_propagates_surrogate_execution_error(self):
        class FailingSurrogate:
            @staticmethod
            def compose_prediction(**_kwargs):
                raise SurrogateExecutionError("AIC database unavailable")

        saved_context = {
            name: getattr(agent_tools._CTX, name)
            for name in ("resource_map", "surrogate", "candidate_graph", "dro")
        }
        saved_payload = agent_tools._rank_prediction_payload
        try:
            agent_tools.bind_tools(
                resource_map=_ResourceMap(),
                surrogate=FailingSurrogate(),
                candidate_graph=object(),
                dro=_DRO(),
            )
            agent_tools._rank_prediction_payload = lambda rank, features: {
                "job_config": {},
                "job_features": {},
            }
            with self.assertRaisesRegex(SurrogateExecutionError, "database unavailable"):
                agent_tools.size_ladder(
                    [
                        {
                            "role": "aggregate",
                            "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                            "config": {"instance_type": "p5.48xlarge", "gpu_count": 1},
                        }
                    ],
                    {"type": "online", "target_p99_ttft_ms": 100.0, "target_p99_tpot_ms": 10.0},
                    target_tps=100,
                )
        finally:
            agent_tools._rank_prediction_payload = saved_payload
            for name, value in saved_context.items():
                setattr(agent_tools._CTX, name, value)

    def test_budget_book_tracks_and_enforces_instance_pools(self):
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
        snapshot = ClusterResourceSnapshot(
            tick=1,
            resources=resources,
            active_jobs=[],
            pending_jobs=[{"job_id": "job_1", "user_id": "usr_test", "status": "waiting"}],
        )

        class BudgetResourceMap(ResourceMapManager):
            def __init__(self):
                super().__init__(user_id="usr_test")

            def snapshot(self):
                return snapshot

        class SlowLoop:
            state = type("State", (), {"tick": 1})()

            @staticmethod
            def get_sss_swap_budget_t():
                return 10

        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in (
                "resource_map",
                "slow_loop",
                "user_registry",
                "user_envelopes",
                "validated_budget_book",
                "cluster_snapshot",
            )
        }
        try:
            agent_tools._CTX.resource_map = BudgetResourceMap()
            agent_tools._CTX.slow_loop = SlowLoop()
            agent_tools._CTX.user_registry = None
            agent_tools._CTX.user_envelopes = None
            agent_tools._CTX.cluster_snapshot = snapshot
            book = agent_tools.allocate_budget_book()
            slice_ = book["job_budgets"]["job_1"]

            self.assertEqual(
                slice_["pool_budget"][env],
                {"g6e.xlarge": 4, "g6e.12xlarge": 3},
            )
            self.assertTrue(agent_tools.validate_budget_book(book)["ok"])

            legacy_book = {
                "job_budgets": {"job_1": {"user_id": "usr_test", "env_budget": {env: 16}}}
            }
            legacy_result = agent_tools.validate_budget_book(legacy_book)
            self.assertFalse(legacy_result["ok"])
            self.assertTrue(
                any("pool_budget is required" in v for v in legacy_result["violations"])
            )

            split_book = {
                "job_budgets": {
                    job_id: {
                        "user_id": "usr_test",
                        "env_budget": {env: 8},
                        "pool_budget": {env: {"g6e.12xlarge": 2}},
                    }
                    for job_id in ("job_1", "job_2")
                }
            }
            split_result = agent_tools.validate_budget_book(split_book)
            self.assertFalse(split_result["ok"])
            self.assertTrue(any("budgets sum to 4" in v for v in split_result["violations"]))

            action = PlanAction.from_dict(
                {
                    "job_id": "job_1",
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
            )
            self.assertIn(
                "pool g6e.12xlarge",
                agent_tools._budget_violations(action, slice_)[0],
            )
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

    def test_size_ladder_threads_rank_env_and_job_model_to_surrogate(self):
        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in ("resource_map", "surrogate", "candidate_graph", "dro")
        }
        surrogate = _RecordingSurrogate()
        try:
            agent_tools.bind_tools(
                resource_map=_ResourceMap(),
                surrogate=surrogate,
                candidate_graph=object(),
                dro=_DRO(),
            )
            result = agent_tools.size_ladder(
                ranks=[
                    {
                        "role": "aggregate",
                        "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                        "config": {
                            "instance_type": "p5.48xlarge",
                            "tp": 1,
                            "pp": 1,
                            "max_num_seq": 1,
                            "block_size": 1,
                        },
                    }
                ],
                job_features={
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "type": "online",
                    "request_arrival_rate": 1.0,
                    "output_len_tokens_avg": 100.0,
                    "headroom_factor": 1.0,
                    "target_p99_ttft_ms": 100.0,
                    "target_p99_tpot_ms": 10.0,
                    "max_num_batched_tokens": 1,
                },
            )

            self.assertTrue(result["meets_target"])
            job_config, job_features = surrogate.calls[0]
            self.assertEqual(job_config["model_id"], "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(job_config["model_params_b"], 70.0)
            self.assertEqual(job_config["hidden_size"], 8192)
            self.assertEqual(job_config["max_num_seq"], 256)
            self.assertEqual(job_config["max_num_batched_tokens"], 8192)
            self.assertEqual(job_config["block_size"], 16)
            self.assertEqual(job_config["gpu_mem_gb"], 80)
            self.assertEqual(job_config["gpu_bandwidth_gbps"], 3350)
            self.assertEqual(job_config["interconnect_type"], "efa")
            self.assertEqual(job_config["dp"], 1)
            self.assertEqual(job_features["gpu_type"], "H100")
            self.assertEqual(job_features["market"], "reserved")
            self.assertEqual(job_features["cloud"], "aws")
            self.assertEqual(job_features["region"], "us-east-1")
            self.assertEqual(job_features["zone"], "use1-az1")
            self.assertEqual(job_features["instance_type"], "p5.48xlarge")
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

    def test_predict_outcome_strips_engine_knobs_from_agent_inputs(self):
        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in ("surrogate", "candidate_graph", "dro")
        }
        surrogate = _RecordingSurrogate()
        try:
            agent_tools.bind_tools(surrogate=surrogate, candidate_graph=object(), dro=_DRO())
            agent_tools.predict_outcome(
                {
                    "job_config": {"model_id": "model", "max_num_seq": 1, "block_size": 1},
                    "job_features": {"max_num_batched_tokens": 1},
                }
            )

            job_config, job_features = surrogate.calls[0]
            self.assertEqual(job_config, {"model_id": "model"})
            self.assertEqual(job_features, {})
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

    def test_predictions_are_not_cached(self):
        class ScenarioSurrogate:
            def __init__(self):
                self.calls = []

            def compose_prediction(self, **kwargs):
                scenario = kwargs["scenario"]
                self.calls.append(scenario)
                return ({"throughput_token_per_sec": 10.0 if scenario == "mean" else 20.0}, {})

        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in ("surrogate", "candidate_graph", "dro")
        }
        surrogate = ScenarioSurrogate()
        saved_calls = agent_tools._surrogate_calls
        try:
            agent_tools.bind_tools(surrogate=surrogate, candidate_graph=object(), dro=_DRO())
            agent_tools._surrogate_calls = 0
            mean = agent_tools._predict_outcome_core({}, {}, scenario="mean")
            peak = agent_tools._predict_outcome_core({}, {}, scenario="peak")
            stress = agent_tools._predict_outcome_core({}, {}, scenario="peak_all_multiturn_stress")
            mean_cached = agent_tools._predict_outcome_core({}, {}, scenario="mean")
            calls_after = agent_tools._surrogate_calls
        finally:
            agent_tools._surrogate_calls = saved_calls
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

        self.assertEqual(mean["y_hat"]["throughput_token_per_sec"], 10.0)
        self.assertEqual(peak["y_hat"]["throughput_token_per_sec"], 20.0)
        self.assertEqual(stress["y_hat"]["throughput_token_per_sec"], 20.0)
        self.assertEqual(mean_cached["y_hat"]["throughput_token_per_sec"], 10.0)
        self.assertEqual(calls_after, 3)
        self.assertEqual(
            surrogate.calls,
            ["mean", "peak", "peak_all_multiturn_stress", "mean"],
        )

    def test_surrogate_calls_are_serialized(self):
        class SlowSurrogate:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.calls = 0

            def compose_prediction(self, **_kwargs):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls += 1
                time.sleep(0.01)
                self.active -= 1
                return ({"throughput_token_per_sec": 1.0}, {})

        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in ("surrogate", "candidate_graph", "dro")
        }
        saved_calls = agent_tools._surrogate_calls
        surrogate = SlowSurrogate()
        try:
            agent_tools.bind_tools(surrogate=surrogate, candidate_graph=object(), dro=_DRO())
            agent_tools._surrogate_calls = 0
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: agent_tools._predict_outcome_core({}, {}), range(4)))
        finally:
            agent_tools._surrogate_calls = saved_calls
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

        self.assertEqual(surrogate.calls, 4)
        self.assertEqual(surrogate.max_active, 1)

    def test_multiturn_stress_diagnostic_skip_and_failure_are_non_selecting(self):
        action = {"sigma": 7.0, "meets_target": True, "ladder": []}
        agent_tools._attach_peak_multiturn_stress(action, {"multi_turn_ratio": 0})
        self.assertNotIn("selection_diagnostics", action)

        def fail_stress(*_args, **_kwargs):
            raise SurrogateExecutionError("stress replay failed")

        saved_predict = agent_tools._predict_outcome_core
        saved_payload = agent_tools._rank_prediction_payload
        try:
            agent_tools._predict_outcome_core = fail_stress
            agent_tools._rank_prediction_payload = lambda rank, features: {
                "job_config": {},
                "job_features": features,
            }
            stressed = {
                "sigma": 7.0,
                "meets_target": True,
                "ladder": [
                    {
                        "role": "aggregate",
                        "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                        "config": {"instance_type": "p5.48xlarge", "gpu_count": 1},
                        "n_replicas": 1,
                    }
                ],
            }
            agent_tools._attach_peak_multiturn_stress(stressed, {"multi_turn_ratio": 0.5})
        finally:
            agent_tools._predict_outcome_core = saved_predict
            agent_tools._rank_prediction_payload = saved_payload

        diag = stressed["selection_diagnostics"]["peak_all_multiturn_stress"]
        self.assertEqual(stressed["sigma"], 7.0)
        self.assertTrue(stressed["meets_target"])
        self.assertEqual(len(stressed["ladder"]), 1)
        self.assertIn("stress replay failed", diag["error"])

        def predict_stress(*_args, **_kwargs):
            return {
                "y_hat_raw": {
                    "p99_ttft_ms": 11.0,
                    "p99_tpot_ms": 2.0,
                    "throughput_token_per_sec": 33.0,
                },
                "v_hat": {"completed_requests": 21},
            }

        ok = {"sigma": 7.0, "meets_target": True, "ladder": stressed["ladder"]}
        saved_predict = agent_tools._predict_outcome_core
        saved_payload = agent_tools._rank_prediction_payload
        try:
            agent_tools._predict_outcome_core = predict_stress
            agent_tools._rank_prediction_payload = lambda rank, features: {
                "job_config": {},
                "job_features": features,
            }
            agent_tools._attach_peak_multiturn_stress(ok, {"multi_turn_ratio": 0.5})
        finally:
            agent_tools._predict_outcome_core = saved_predict
            agent_tools._rank_prediction_payload = saved_payload

        ok_diag = ok["selection_diagnostics"]["peak_all_multiturn_stress"]
        self.assertEqual(ok_diag["p99_ttft_ms"], 11.0)
        self.assertEqual(ok_diag["p99_tpot_ms"], 2.0)
        self.assertEqual(ok_diag["throughput_token_per_sec"], 33.0)
        self.assertEqual(ok_diag["completed_requests"], 21)
        self.assertIsNone(ok_diag["error"])

    def test_predict_outcome_derives_gpu_type_from_env(self):
        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in ("surrogate", "candidate_graph", "dro")
        }
        surrogate = _RecordingSurrogate()
        try:
            agent_tools.bind_tools(surrogate=surrogate, candidate_graph=object(), dro=_DRO())
            agent_tools.predict_outcome(
                {"job_config": {"model_id": "model"}, "job_features": {}},
                env=["reserved", "aws", "us-east-1", "use1-az1", "H100"],
            )

            job_config, _job_features = surrogate.calls[0]
            self.assertEqual(job_config["gpu_type"], "H100")
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

    def test_eig_materialization_uses_committed_mechanisms_only(self):
        class Registry:
            def __init__(self):
                self.mechanisms = {
                    "M_test": type("Mechanism", (), {"mechanism_id": "M_test"})(),
                    "M_unrelated": type("Mechanism", (), {"mechanism_id": "M_unrelated"})(),
                }

            def get_mechanism(self, mechanism_id):
                return self.mechanisms[mechanism_id]

        saved = agent_tools._CTX.mechanism_registry
        registry = Registry()
        try:
            agent_tools.bind_tools(mechanism_registry=registry)
            ladder = agent_tools._materialize_ladder(
                [
                    {
                        "mechanism_id": "M_test",
                        "config": {"tp": 1, "max_num_seq": 1, "block_size": 1},
                    }
                ]
            )

            self.assertEqual(ladder.ranks[0].config, {"tp": 1})
            self.assertEqual(
                [mechanism.mechanism_id for mechanism in ladder.applicable_mechanisms],
                ["M_test"],
            )
            with self.assertRaisesRegex(ValueError, "unknown mechanism_id"):
                agent_tools._materialize_ladder([{"mechanism_id": "M_missing", "config": {}}])
        finally:
            agent_tools._CTX.mechanism_registry = saved

    def test_get_job_brief_includes_model_catalog(self):
        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in (
                "resource_map",
                "evidence_store",
                "mechanism_registry",
                "confidence_service",
            )
        }
        try:
            agent_tools.bind_tools(
                resource_map=_ResourceMap(),
                evidence_store=_EvidenceStore(),
                mechanism_registry=_MechanismRegistry(),
                confidence_service=_ConfidenceService(),
            )
            brief = agent_tools.get_job_brief("job_1")

            self.assertEqual(brief["job_features"]["model_id"], "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(brief["model_catalog"]["model_params_b"], 70.0)
            self.assertNotIn("model_params_b", brief["job_features"])
            self.assertEqual(brief["mechanism_candidates"], [])
            self.assertNotIn("applicable_mechanisms", brief)
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

    def test_stamp_plan_predictions_writes_raw_rank_predictions(self):
        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in ("surrogate", "candidate_graph", "dro")
        }
        try:
            agent_tools.bind_tools(
                surrogate=_RecordingSurrogate(),
                candidate_graph=object(),
                dro=_DRO(),
            )
            snapshot = _Snapshot()
            plan = agent_tools.stamp_plan_predictions(
                {
                    "actions": [
                        {
                            "job_id": "job_1",
                            "type": "place",
                            "ladder": [
                                {
                                    "role": "aggregate",
                                    "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                                    "config": {"instance_type": "p5.48xlarge", "gpu_count": 1},
                                    "n_replicas": 1,
                                }
                            ],
                        }
                    ]
                },
                snapshot,
            )

            rank = plan.actions[0].ladder[0]
            self.assertEqual(rank.predicted_y["p99_ttft_ms"], 10.0)
            self.assertEqual(rank.predicted_v, {"kv_cache_util": 0.4})
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)


class _Snapshot:
    def pending_jobs_summary(self):
        return [
            {
                "job_id": "job_1",
                "job_features": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
            }
        ]


if __name__ == "__main__":
    unittest.main()
