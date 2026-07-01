import unittest
from types import SimpleNamespace

import numpy as np
from src.core.candidate_graph import CandidateGraph
from src.core.models import Node
from src.infra.deployment_x import build_deployment_x_index
from src.infra.resource_map import ClusterResourceSnapshot
from src.orchestrator.fsm_states import TickContext, TickRunner

ENV = "reserved|aws|us-east-2|use2-az3|H100"
ENV_LABEL = tuple(ENV.split("|"))


def _x_fields():
    return [
        "model_params_b",
        "num_attn_heads",
        "num_kv_heads",
        "attn_heads_per_kv_head",
        "gpu_bandwidth_gbps",
        "gpu_tflops_fp16",
        "gpu_mem_gb",
        "gpu_per_node",
        "cuda_compute_capability",
        "gpu_generation",
        "nvlink_bandwidth_gbps",
        "internode_bandwidth_gbps",
        "pcie_bandwidth_gbps",
        "bandwidth_per_param",
        "flops_per_param",
        "gpu_watts",
        "request_arrival_rate",
        "total_token_budget",
        "deadline_hrs",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
        "cloud",
        "region",
        "market",
        "gpu_type",
        "instance_type",
        "num_nodes_per_chain",
        "tp",
        "pp",
        "engine_name",
        "prefix_cache_enabled",
    ]


def _snapshot():
    features = {
        "model_params_b": 70,
        "num_attn_heads": 64,
        "num_kv_heads": 8,
        "request_arrival_rate": 100,
        "total_token_budget": 1000,
        "deadline_hours": 2,
        "target_p99_ttft_ms": 200,
        "target_p99_tpot_ms": 40,
    }
    shape = {
        "rank_id": "rank_a",
        "env": list(ENV_LABEL),
        "count": 8,
        "gpu_count": 8,
        "instance_type": "p5.48xlarge",
        "tp": 8,
        "pp": 1,
        "engine_name": "vllm",
        "prefix_cache_enabled": True,
        "target_p99_ttft_ms": 200,
        "target_p99_tpot_ms": 40,
    }
    return ClusterResourceSnapshot(
        tick=1,
        resources={
            ENV: {
                "market": "reserved",
                "cloud": "aws",
                "region": "us-east-2",
                "zone": "use2-az3",
                "gpu_type": "H100",
                "total": 16,
                "free": 0,
                "pools": [
                    {
                        "instance_type": "p5.48xlarge",
                        "gpu_type": "H100",
                        "gpus_per_instance": 8,
                        "total_instances": 2,
                        "fabric_type": "efa",
                    }
                ],
            }
        },
        active_jobs=[
            {
                "job_id": "job_1",
                "user_id": "user_1",
                "job_features": features,
                "spec_json": {"job_features": features},
                "active_chains": [
                    {"chain_id": "chain_1", "target_node": ENV, "shape_json": dict(shape)},
                    {"chain_id": "chain_2", "target_node": ENV, "shape_json": dict(shape)},
                ],
            }
        ],
        pending_jobs=[],
    )


def _hardware_catalog():
    return {
        "cloud": "aws",
        "regions": [
            {
                "cloud": "aws",
                "region": "us-east-2",
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
                        "network": {
                            "efa_supported": True,
                            "network_cards": [{"peak_bandwidth_gbps": 3200}],
                        },
                    }
                ],
            }
        ],
    }


def _candidate_graph():
    nodes = {name: Node(name, "X") for name in _x_fields()}
    nodes["kv_cache_util"] = Node("kv_cache_util", "V")
    nodes["p99_ttft_ms"] = Node("p99_ttft_ms", "Y")
    return CandidateGraph(nodes, {}, {})


class DeploymentXSmokeTests(unittest.TestCase):
    def test_builds_rank_x_from_snapshot_and_catalog(self):
        index = build_deployment_x_index(
            _snapshot(),
            hardware_catalog=_hardware_catalog(),
            x_fields=_x_fields(),
        )

        deployment = index.resolve("job_1", "rank_a")
        self.assertIsNotNone(deployment)
        x = deployment.x

        self.assertEqual(x["gpu_mem_gb"], 80)
        self.assertEqual(x["gpu_tflops_fp16"], 989.5)
        self.assertEqual(x["internode_bandwidth_gbps"], 3200)
        self.assertEqual(x["request_arrival_rate"], 50)
        self.assertEqual(x["total_token_budget"], 500)
        self.assertEqual(x["deadline_hrs"], 2)
        self.assertEqual(x["num_nodes_per_chain"], 1)
        self.assertEqual(x["attn_heads_per_kv_head"], 8)
        self.assertAlmostEqual(x["bandwidth_per_param"], 3350 / 70)
        self.assertAlmostEqual(x["flops_per_param"], 989.5 / 70)
        with self.assertRaises(ValueError):
            index.resolve("job_1")
        with self.assertRaises(KeyError):
            index.resolve("job_1", "missing_rank")

    def test_missing_rank_id_is_contract_error(self):
        snapshot = _snapshot()
        shape = snapshot.active_jobs[0]["active_chains"][0]["shape_json"]
        shape.pop("rank_id")

        with self.assertRaises(ValueError):
            build_deployment_x_index(
                snapshot,
                hardware_catalog=_hardware_catalog(),
                x_fields=_x_fields(),
            )

    def test_missing_hardware_catalog_is_contract_error(self):
        with self.assertRaises(ValueError):
            build_deployment_x_index(_snapshot(), hardware_catalog={}, x_fields=_x_fields())

    def test_s2_writes_deployment_x_without_telemetry_x(self):
        evidence_store = _EvidenceStore()
        runner = TickRunner(
            evidence_store=evidence_store,
            telemetry=_Telemetry(),
            cusum=_Cusum(),
            icp=object(),
            quadrant_validator=object(),
            confidence_service=SimpleNamespace(candidate_graph=_candidate_graph()),
            slow_loop=_SlowLoop(),
            dro=_Dro(),
            mechanism_registry=_MechanismRegistry(),
            resource_map=_ResourceMap(),
            agent=object(),
            plan_validator=object(),
            executor=object(),
            candidate_graph=_candidate_graph(),
        )
        ctx = TickContext(tick=1, cluster_snapshot=_snapshot())

        runner.S1(ctx)
        runner.S2(ctx)

        self.assertEqual(len(evidence_store.rows), 1)
        row = evidence_store.rows[0]
        self.assertEqual(row.rank_id, "rank_a")
        self.assertEqual(row.env_label, ENV_LABEL)
        self.assertEqual(row.X["request_arrival_rate"], 50)
        self.assertEqual(row.X["gpu_generation"], "Hopper")


class _Telemetry:
    def collect_telemetry(self, tick_start, tick_end):
        return "bundle"

    def iter_per_rank(self, bundle):
        yield SimpleNamespace(
            job_id="job_1",
            rank_id="rank_a",
            W_observed={"type": "online"},
            v_observed={"kv_cache_util": np.array([0.2, 0.3])},
            v_predicted={"kv_cache_util": 0.1},
            y_observed={"p99_ttft_ms": np.array([100.0, 110.0])},
            y_predicted={"p99_ttft_ms": 90.0},
        )


class _EvidenceStore:
    def __init__(self):
        self.rows = []

    def append_row(self, row):
        self.rows.append(row)


class _SlowLoop:
    def __init__(self):
        self.typical_ranges = {}

    def get_sss_wt(self):
        return {}

    def get_sss_z_star_t(self):
        return {}

    def get_sss_cusum_params_v(self):
        return {}

    def get_sss_cusum_params_y(self):
        return {}


class _Cusum:
    def cusum_params_per_v(self, name, residuals):
        return 0.0, 1.0


class _Dro:
    def append_residual_history(self, pred_y, obs_y):
        pass


class _MechanismRegistry:
    def filter_by_scope(self, subset_x, subset_v):
        self.subset_x = subset_x
        self.subset_v = subset_v
        return []


class _ResourceMap:
    def hardware_catalog(self):
        return _hardware_catalog()


if __name__ == "__main__":
    unittest.main()
