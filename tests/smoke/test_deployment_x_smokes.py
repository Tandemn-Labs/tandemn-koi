import os
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import numpy as np
import psycopg
from db.postgres import postgres_pool
from psycopg import sql
from psycopg.types.json import Jsonb
from src.core.candidate_graph import CandidateGraph
from src.core.confidence_service import ConfidenceService
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node
from src.cost.dro import DRO
from src.infra.deployment_x import _gpu, build_deployment_x_index, materialize_launch_config
from src.infra.resource_map import ClusterResourceSnapshot, ResourceMapManager
from src.orchestrator.fsm_states import TickContext, TickRunner
from src.validation.cusum import Cusum, CusumResult
from src.validation.icp import ICP, ICPResult
from src.validation.quadrants import Quadrant, QuadrantValidator

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
        "multi_turn_avg_turns",
        "total_token_budget",
        "deadline_hrs",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
        "max_num_seq",
        "max_num_batched_tokens",
        "max_model_len",
        "block_size",
        "gpu_mem_util",
        "kv_cache_dtype",
        "chunked_prefill_enabled",
        "cloud",
        "region",
        "market",
        "gpu_type",
        "instance_type",
        "sp",
        "dp",
        "ep",
        "cp",
        "prefill_worker_count",
        "decode_worker_count",
        "num_nodes_per_chain",
        "tp",
        "pp",
        "engine_name",
        "prefix_cache_enabled",
    ]


def _snapshot():
    features = {
        "type": "online",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "request_arrival_rate": 100,
        "multi_turn_avg_turns": 2.0,
        "total_token_budget": 1000,
        "deadline_hours": 2,
        "target_p99_ttft_ms": 200,
        "target_p99_tpot_ms": 40,
        "gpu_mem_util": 0.99,
    }
    shape = {
        "rank_id": "rank_a",
        "env": list(ENV_LABEL),
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "count": 8,
        "gpu_count": 8,
        "instance_type": "p5.48xlarge",
        "tp": 8,
        "pp": 1,
        "target_p99_ttft_ms": 200,
        "target_p99_tpot_ms": 40,
        "predicted_y": {"p99_ttft_ms": 90.0},
        "predicted_v": {"kv_cache_util": 0.1},
        "prediction_lineage": {"schema_version": 3, "deployment_id": "deploy-a"},
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


def _azure_hardware_catalog():
    return {
        "cloud": "azure",
        "regions": ["eastus"],
        "instance_types": [
            {
                "instance_type": "Standard_ND96amsr_A100_v4",
                "accelerators": [],
                "network": {"network_cards": [{"peak_bandwidth_gbps": 1600}]},
                "offerings": [{"region": "eastus"}],
            }
        ],
    }


def _model_catalogs():
    return {
        "Qwen/Qwen2.5-72B-Instruct": {
            "model_id": "Qwen/Qwen2.5-72B-Instruct",
            "model_params_b": 70,
            "num_attn_heads": 64,
            "num_kv_heads": 8,
            "engine_name": "vllm",
            "gpu_mem_util": 0.85,
            "prefix_cache_enabled": True,
            "max_model_len": 8192,
            "chunked_prefill_enabled": True,
            "max_num_seq": [{"gpu_type": "H100", "value": 256}],
            "max_num_batched_tokens": [{"gpu_type": "H100", "value": 8192}],
            "block_size": [{"gpu_type": "H100", "value": 16}],
            "kv_cache_dtype": [{"gpu_type": "H100", "value": "auto"}],
        }
    }


def _candidate_graph():
    nodes = {name: Node(name, "X") for name in _x_fields()}
    nodes["kv_cache_util"] = Node("kv_cache_util", "V")
    nodes["p99_ttft_ms"] = Node("p99_ttft_ms", "Y")
    return CandidateGraph(nodes, {}, {})


def _diagnostic_graph():
    nodes = {name: Node(name, "X") for name in _x_fields()}
    nodes["kv_cache_util"] = Node("kv_cache_util", "V")
    nodes["p99_ttft_ms"] = Node("p99_ttft_ms", "Y")
    x_to_v = Edge("gpu_mem_util->kv_cache_util", "gpu_mem_util", "kv_cache_util", "X", "V")
    v_to_y = Edge("kv_cache_util->p99_ttft_ms", "kv_cache_util", "p99_ttft_ms", "V", "Y")
    graph = CandidateGraph(
        nodes,
        {x_to_v.edge_id: x_to_v, v_to_y.edge_id: v_to_y},
        {
            x_to_v.edge_id: EdgeMetadata(x_to_v.edge_id),
            v_to_y.edge_id: EdgeMetadata(v_to_y.edge_id),
        },
    )
    mechanism = Mechanism(
        mechanism_id="M_diagnostic",
        edge_ids=[x_to_v.edge_id, v_to_y.edge_id],
        scope={},
        narrative="Diagnostic mechanism.",
    )
    return graph, mechanism


def _catalog_x_assertions():
    return {
        "engine_name": "vllm",
        "prefix_cache_enabled": True,
        "max_num_seq": 256,
        "max_num_batched_tokens": 8192,
        "block_size": 16,
        "gpu_mem_util": 0.85,
        "kv_cache_dtype": "auto",
        "max_model_len": 8192,
        "chunked_prefill_enabled": True,
    }


class DeploymentXSmokeTests(unittest.TestCase):
    def test_gpu_catalog_match_normalizes_store_labels(self):
        """Match Store's underscore labels with catalog hyphen labels."""
        hardware = {
            "accelerators": [
                {
                    "kind": "gpu",
                    "name": "A100-80GB",
                    "canonical_gpu_name": "A100-40GB",
                }
            ]
        }

        gpu = _gpu(hardware, "A100_80GB")

        self.assertEqual(gpu["name"], "A100-80GB")

    def test_rtx_pro_6000_aliases_match_catalog_and_model_profiles(self):
        hardware = {
            "accelerators": [
                {
                    "kind": "gpu",
                    "name": "RTX-PRO-6000-BSE",
                    "canonical_gpu_name": "RTX-PRO-6000-BSE",
                }
            ]
        }
        catalog = {
            "model_id": "model",
            "max_num_seq": [{"gpu_type": "RTXPRO6000", "value": 128}],
        }

        self.assertEqual(_gpu(hardware, "RTX-PRO-6000")["name"], "RTX-PRO-6000-BSE")
        self.assertEqual(materialize_launch_config(catalog, "RTX-PRO-6000-BSE")["max_num_seq"], 128)

    def test_composite_ranks_split_workload_by_traffic_share(self):
        snapshot = _snapshot()
        first = snapshot.active_jobs[0]["active_chains"][0]
        second = snapshot.active_jobs[0]["active_chains"][1]
        first["shape_json"]["rank_traffic_share"] = 0.4
        second["shape_json"]["rank_id"] = "rank_b"
        second["shape_json"]["rank_traffic_share"] = 0.6
        for chain in (first, second):
            chain["shape_json"]["prediction_lineage"]["partial_admission"] = {
                "mode": "advisory",
                "enforced": False,
            }

        index = build_deployment_x_index(
            snapshot,
            hardware_catalog=_hardware_catalog(),
            model_catalogs=_model_catalogs(),
            x_fields=_x_fields(),
        )

        self.assertEqual(index.resolve("job_1", "rank_a").x["request_arrival_rate"], 40.0)
        self.assertEqual(index.resolve("job_1", "rank_b").x["request_arrival_rate"], 60.0)

    def test_single_rank_honors_explicit_traffic_share(self):
        snapshot = _snapshot()
        for chain in snapshot.active_jobs[0]["active_chains"]:
            chain["shape_json"]["rank_traffic_share"] = 0.4

        index = build_deployment_x_index(
            snapshot,
            hardware_catalog=_hardware_catalog(),
            model_catalogs=_model_catalogs(),
            x_fields=_x_fields(),
        )

        self.assertEqual(index.resolve("job_1", "rank_a").x["request_arrival_rate"], 40.0)

    def test_single_rank_advisory_unenforced_share_uses_full_observed_load(self):
        snapshot = _snapshot()
        evidence = {
            "mode": "advisory",
            "requested_tps": 100.0,
            "admitted_tps": 40.0,
            "served_fraction": 0.4,
            "enforced": False,
        }
        for chain in snapshot.active_jobs[0]["active_chains"]:
            chain["shape_json"]["rank_traffic_share"] = 0.4
            chain["shape_json"]["prediction_lineage"]["partial_admission"] = evidence

        index = build_deployment_x_index(
            snapshot,
            hardware_catalog=_hardware_catalog(),
            model_catalogs=_model_catalogs(),
            x_fields=_x_fields(),
        )

        self.assertEqual(index.resolve("job_1", "rank_a").x["request_arrival_rate"], 100.0)

    def test_explicit_traffic_share_must_be_finite_and_bounded(self):
        for share in (0.0, -0.1, 1.1, float("inf"), float("nan")):
            with self.subTest(share=share):
                snapshot = _snapshot()
                for chain in snapshot.active_jobs[0]["active_chains"]:
                    chain["shape_json"]["rank_traffic_share"] = share

                with self.assertRaisesRegex(ValueError, "finite and in"):
                    build_deployment_x_index(
                        snapshot,
                        hardware_catalog=_hardware_catalog(),
                        model_catalogs=_model_catalogs(),
                        x_fields=_x_fields(),
                    )

    def test_idle_index_needs_no_catalogs(self):
        snapshot = ClusterResourceSnapshot(
            tick=1,
            resources={},
            active_jobs=[],
            pending_jobs=[],
        )

        index = build_deployment_x_index(
            snapshot,
            hardware_catalog={},
            model_catalogs={},
            x_fields=["model_id"],
        )

        self.assertEqual(index.by_rank, {})

    def test_idle_s1_does_not_fetch_catalogs(self):
        snapshot = ClusterResourceSnapshot(
            tick=1,
            resources={},
            active_jobs=[],
            pending_jobs=[],
        )
        runner = TickRunner(
            evidence_store=object(),
            telemetry=_Telemetry(),
            cusum=object(),
            icp=object(),
            quadrant_validator=object(),
            confidence_service=SimpleNamespace(candidate_graph=_candidate_graph()),
            slow_loop=object(),
            dro=object(),
            mechanism_registry=object(),
            resource_map=_IdleResourceMap(),
            agent=object(),
            plan_validator=object(),
            executor=object(),
            candidate_graph=_candidate_graph(),
        )
        ctx = TickContext(tick=1, cluster_snapshot=snapshot)

        runner.S1(ctx)

        self.assertEqual(ctx.deployment_x.by_rank, {})

    def test_builds_rank_x_from_snapshot_and_catalog(self):
        index = build_deployment_x_index(
            _snapshot(),
            hardware_catalog=_hardware_catalog(),
            model_catalogs=_model_catalogs(),
            x_fields=_x_fields(),
        )

        deployment = index.resolve("job_1", "rank_a")
        self.assertIsNotNone(deployment)
        x = deployment.x

        self.assertEqual(x["gpu_mem_gb"], 80)
        self.assertEqual(x["gpu_tflops_fp16"], 989.5)
        self.assertEqual(x["internode_bandwidth_gbps"], 3200)
        self.assertEqual(x["request_arrival_rate"], 100)
        self.assertEqual(x["multi_turn_avg_turns"], 2.0)
        self.assertEqual(x["total_token_budget"], 1000)
        self.assertEqual(x["deadline_hrs"], 2)
        self.assertEqual(x["num_nodes_per_chain"], 1)
        self.assertEqual(x["dp"], 2)
        self.assertEqual(x["sp"], 1)
        self.assertEqual(x["ep"], 1)
        self.assertEqual(x["cp"], 1)
        self.assertEqual(x["prefill_worker_count"], 0)
        self.assertEqual(x["decode_worker_count"], 0)
        self.assertEqual(
            {key: x[key] for key in _catalog_x_assertions()},
            _catalog_x_assertions(),
        )
        self.assertEqual(x["attn_heads_per_kv_head"], 8)
        self.assertAlmostEqual(x["bandwidth_per_param"], 3350 / 70)
        self.assertAlmostEqual(x["flops_per_param"], 989.5 / 70)
        self.assertEqual(deployment.y_predicted, {"p99_ttft_ms": 90.0})
        self.assertEqual(deployment.v_predicted, {"kv_cache_util": 0.1})
        self.assertEqual(deployment.prediction_lineage["deployment_id"], "deploy-a")
        self.assertNotIn("predicted_y", x)
        with self.assertRaises(ValueError):
            index.resolve("job_1")
        with self.assertRaises(KeyError):
            index.resolve("job_1", "missing_rank")

    def test_materializes_catalog_launch_config_for_gpu(self):
        config = materialize_launch_config(_model_catalogs()["Qwen/Qwen2.5-72B-Instruct"], "H100")

        self.assertEqual(
            config,
            {
                "engine_name": "vllm",
                "gpu_mem_util": 0.85,
                "prefix_cache_enabled": True,
                "max_model_len": 8192,
                "chunked_prefill_enabled": True,
                "max_num_seq": 256,
                "max_num_batched_tokens": 8192,
                "block_size": 16,
                "kv_cache_dtype": "auto",
            },
        )

    def test_missing_rank_id_is_contract_error(self):
        snapshot = _snapshot()
        shape = snapshot.active_jobs[0]["active_chains"][0]["shape_json"]
        shape.pop("rank_id")

        with self.assertRaises(ValueError):
            build_deployment_x_index(
                snapshot,
                hardware_catalog=_hardware_catalog(),
                model_catalogs=_model_catalogs(),
                x_fields=_x_fields(),
            )

    def test_missing_hardware_catalog_is_contract_error(self):
        with self.assertRaises(ValueError):
            build_deployment_x_index(
                _snapshot(),
                hardware_catalog={},
                model_catalogs=_model_catalogs(),
                x_fields=_x_fields(),
            )

    def test_s2_writes_deployment_x_without_telemetry_x(self):
        evidence_store = _EvidenceStore()
        mechanism_registry = _MechanismRegistry()
        runner = TickRunner(
            evidence_store=evidence_store,
            telemetry=_Telemetry(),
            cusum=_Cusum(),
            icp=object(),
            quadrant_validator=object(),
            confidence_service=SimpleNamespace(candidate_graph=_candidate_graph()),
            slow_loop=_SlowLoop(),
            dro=_Dro(),
            mechanism_registry=mechanism_registry,
            resource_map=_ResourceMap(),
            agent=object(),
            plan_validator=object(),
            executor=object(),
            candidate_graph=_candidate_graph(),
        )
        ctx = TickContext(tick=1, cluster_snapshot=_snapshot())

        runner.S1(ctx)
        self.assertEqual(ctx.telemetry_diagnostics["expected_rank_count"], 1)
        self.assertEqual(ctx.telemetry_diagnostics["observed_rank_count"], 1)
        self.assertEqual(
            ctx.telemetry_diagnostics["observed_ranks"][0]["v_sample_counts"],
            {"kv_cache_util": 2},
        )
        runner.S2(ctx)

        self.assertEqual(len(evidence_store.rows), 1)
        row = evidence_store.rows[0]
        self.assertEqual(row.rank_id, "rank_a")
        self.assertEqual(row.env_label, ENV_LABEL)
        self.assertEqual(row.X["request_arrival_rate"], 100)
        self.assertEqual(row.X["workload_type"], "online")
        self.assertEqual(row.X["gpu_generation"], "Hopper")
        self.assertEqual(row.y_predicted, {"p99_ttft_ms": 90.0})
        self.assertEqual(row.V_predicted_trajectory, {"kv_cache_util": 0.1})
        self.assertEqual(row.deployment_id, "deploy-a")
        self.assertEqual(row.prediction_lineage["schema_version"], 3)
        self.assertEqual(mechanism_registry.context["type"], "online")
        self.assertEqual(mechanism_registry.context["request_arrival_rate"], 100)

    def test_prediction_ledger_restores_lineage_after_store_enrichment(self):
        snapshot = _snapshot()
        chains = snapshot.active_jobs[0]["active_chains"]
        for chain in chains:
            chain["shape_json"].pop("prediction_lineage")
            chain["shape_json"]["engine_version"] = "store-enriched"
        runner = TickRunner(
            evidence_store=_EvidenceStore(),
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
        signature = runner._prediction_shape_signature(
            {
                "env": list(ENV_LABEL),
                "config": {
                    "instance_type": "p5.48xlarge",
                    "gpu_count": 8,
                    "tp": 8,
                    "pp": 1,
                    "engine_version": "store-enriched",
                },
                "n_replicas": 2,
            }
        )
        runner._prediction_ledger[("job_1", "rank_a")] = {
            "predicted_y": {"p99_ttft_ms": 90.0},
            "predicted_v": {"kv_cache_util": 0.1},
            "prediction_lineage": {"schema_version": 3, "deployment_id": "deploy-a"},
            "mechanism_id": "M_committed",
            "shape_signature": signature,
        }
        ctx = TickContext(tick=1, cluster_snapshot=snapshot)

        index = runner._build_deployment_x_index(ctx)

        deployment = index.resolve("job_1", "rank_a")
        self.assertEqual(deployment.prediction_lineage["deployment_id"], "deploy-a")

    def test_s2_applicability_requires_x_values_and_preserves_committed(self):
        registry = MechanismRegistry()
        exact_id = registry.add_mechanism(
            Mechanism(
                edge_ids=["tp->comm_overhead_pct"],
                scope={
                    "x": ["tp"],
                    "v": ["comm_overhead_pct"],
                    "workload_type": "online",
                    "conditions": [{"feature": "tp", "op": ">", "value": 1}],
                },
                narrative="Tensor parallel communication.",
            )
        )
        partial_id = registry.add_mechanism(
            Mechanism(
                edge_ids=["tp->comm_overhead_pct"],
                scope={
                    "x": ["tp", "unknown_knob"],
                    "v": ["comm_overhead_pct"],
                    "workload_type": "online",
                    "conditions": [{"feature": "unknown_knob", "op": ">", "value": 0}],
                },
                narrative="Partially known communication mechanism.",
            )
        )
        false_id = registry.add_mechanism(
            Mechanism(
                edge_ids=["peak_to_mean_ratio->depth_req_q"],
                scope={
                    "x": ["peak_to_mean_ratio"],
                    "v": ["depth_req_q"],
                    "workload_type": "online",
                    "conditions": [{"feature": "peak_to_mean_ratio", "op": ">", "value": 2}],
                },
                narrative="Only bursty workloads use this mechanism.",
            )
        )
        runner = TickRunner.__new__(TickRunner)
        runner.mechanism_registry = registry
        runner.candidate_graph = CandidateGraph(
            node_table={
                "tp": Node("tp", "X"),
                "comm_overhead_pct": Node("comm_overhead_pct", "V"),
                "peak_to_mean_ratio": Node("peak_to_mean_ratio", "X"),
                "depth_req_q": Node("depth_req_q", "V"),
            },
            edge_table={
                "tp->comm_overhead_pct": Edge(
                    "tp->comm_overhead_pct", "tp", "comm_overhead_pct", "X", "V"
                ),
                "peak_to_mean_ratio->depth_req_q": Edge(
                    "peak_to_mean_ratio->depth_req_q",
                    "peak_to_mean_ratio",
                    "depth_req_q",
                    "X",
                    "V",
                ),
            },
            edge_metadata_table={
                "tp->comm_overhead_pct": EdgeMetadata("tp->comm_overhead_pct"),
                "peak_to_mean_ratio->depth_req_q": EdgeMetadata("peak_to_mean_ratio->depth_req_q"),
            },
        )
        context = {"type": "online", "tp": 2, "peak_to_mean_ratio": 2}

        matched = {m.mechanism_id for m in runner._applicable_mechanisms(context, None)}
        committed = {m.mechanism_id for m in runner._applicable_mechanisms(context, false_id)}

        self.assertEqual(matched, {exact_id})
        self.assertEqual(committed, {exact_id, false_id})
        self.assertNotIn(partial_id, matched)

    def test_s2_records_mechanism_cusum_and_icp_diagnostics(self):
        graph, mechanism = _diagnostic_graph()
        evidence_store = _EvidenceStore()
        runner = TickRunner(
            evidence_store=evidence_store,
            telemetry=_Telemetry(),
            cusum=Cusum(),
            icp=ICP(),
            quadrant_validator=QuadrantValidator(),
            confidence_service=SimpleNamespace(candidate_graph=graph),
            slow_loop=_SlowLoop(),
            dro=_Dro(),
            mechanism_registry=_MechanismRegistry([mechanism]),
            resource_map=_ResourceMap(),
            agent=object(),
            plan_validator=object(),
            executor=object(),
            candidate_graph=graph,
        )
        ctx = TickContext(tick=1, cluster_snapshot=_snapshot())

        runner.S1(ctx)
        runner.S2(ctx)

        diagnostic = ctx.mechanism_diagnostics[0]
        self.assertEqual(diagnostic["status"], "evaluated")
        self.assertEqual(diagnostic["q_label"].value, "Q4")
        self.assertEqual(diagnostic["cusum"]["V"][0]["name"], "kv_cache_util")
        self.assertEqual(diagnostic["cusum"]["V"][0]["observed"], [0.2, 0.3])
        self.assertEqual(diagnostic["cusum"]["V"][0]["predicted"], 0.1)
        self.assertTrue(diagnostic["cusum"]["V"][0]["fired"])
        self.assertEqual(diagnostic["icp"][0]["result"].value, "undecided")
        self.assertEqual(diagnostic["icp"][0]["reason"], "no_evidence")

    def test_s2_partially_evaluates_bundle_without_rewarding_missing_axis(self):
        graph, mechanism = _diagnostic_graph()
        evidence_store = _EvidenceStore()
        runner = TickRunner(
            evidence_store=evidence_store,
            telemetry=_Telemetry(),
            cusum=Cusum(),
            icp=ICP(),
            quadrant_validator=QuadrantValidator(),
            confidence_service=SimpleNamespace(candidate_graph=graph),
            slow_loop=_SlowLoop(),
            dro=_Dro(),
            mechanism_registry=_MechanismRegistry([mechanism]),
            resource_map=_ResourceMap(),
            agent=object(),
            plan_validator=object(),
            executor=object(),
            candidate_graph=graph,
        )
        ctx = TickContext(tick=1, cluster_snapshot=_snapshot())

        runner.S1(ctx)
        deployment = next(iter(ctx.deployment_x.by_rank.values()))
        deployment.v_predicted = {}
        runner.S2(ctx)

        diagnostic = ctx.mechanism_diagnostics[0]
        row = evidence_store.rows[0]
        self.assertEqual(diagnostic["status"], "partially_evaluated")
        self.assertIsNone(diagnostic["v_verdict"])
        self.assertEqual(diagnostic["y_verdict"], "diverged")
        self.assertIsNone(row.q_label_per_mechanism[mechanism.mechanism_id])
        self.assertEqual(
            row.cusum_per_mechanism[mechanism.mechanism_id][1].value,
            "diverged",
        )

    def test_s2_computes_icp_once_per_unique_edge_per_tick(self):
        graph, mechanism = _diagnostic_graph()
        snapshot = _snapshot()
        chains = snapshot.active_jobs[0]["active_chains"]
        for chain in chains:
            chain["shape_json"]["rank_traffic_share"] = 0.5
        second = {
            **chains[0],
            "chain_id": "chain_b",
            "shape_json": {
                **chains[0]["shape_json"],
                "rank_id": "rank_b",
                "rank_traffic_share": 0.5,
            },
        }
        chains.append(second)

        class TwoRankTelemetry(_Telemetry):
            def iter_per_rank(self, bundle):
                for rank_id in ("rank_a", "rank_b"):
                    yield SimpleNamespace(
                        job_id="job_1",
                        rank_id=rank_id,
                        v_observed={"kv_cache_util": np.array([0.2, 0.3])},
                        y_observed={"p99_ttft_ms": np.array([100.0, 110.0])},
                    )

        class CountingICP:
            def __init__(self):
                self.calls = []

            def compute_icp_details_per_edge(self, edge, evidence_store, before_tick=None):
                self.calls.append((edge.edge_id, before_tick))
                return {
                    "edge_id": edge.edge_id,
                    "result": ICPResult.UNDECIDED,
                    "reason": "no_evidence",
                }

        icp = CountingICP()
        runner = TickRunner(
            evidence_store=_EvidenceStore(),
            telemetry=TwoRankTelemetry(),
            cusum=Cusum(),
            icp=icp,
            quadrant_validator=QuadrantValidator(),
            confidence_service=SimpleNamespace(candidate_graph=graph),
            slow_loop=_SlowLoop(),
            dro=_Dro(),
            mechanism_registry=_MechanismRegistry([mechanism]),
            resource_map=_ResourceMap(),
            agent=object(),
            plan_validator=object(),
            executor=object(),
            candidate_graph=graph,
        )
        ctx = TickContext(tick=3, cluster_snapshot=snapshot)

        runner.S1(ctx)
        runner.S2(ctx)

        self.assertEqual(
            sorted(icp.calls),
            sorted((edge_id, 3) for edge_id in mechanism.edge_ids),
        )

    def test_s2_only_learns_semantically_comparable_online_outcomes(self):
        def evidence_for_depth(depth):
            snapshot = _snapshot()
            for chain in snapshot.active_jobs[0]["active_chains"]:
                chain["shape_json"]["predicted_y"].update(
                    throughput_token_per_sec=100.0,
                    p99_tpot_ms=5.0,
                )

            class SemanticTelemetry(_Telemetry):
                def iter_per_rank(self, bundle):
                    yield SimpleNamespace(
                        job_id="job_1",
                        rank_id="rank_a",
                        v_observed={"depth_req_q": np.array([depth])},
                        y_observed={
                            "throughput_token_per_sec": np.array([1.0]),
                            "p99_ttft_ms": np.array([50.0]),
                            "p99_tpot_ms": np.array([10.0]),
                        },
                    )

            graph = _candidate_graph()
            graph.node_table["throughput_token_per_sec"] = Node("throughput_token_per_sec", "Y")
            graph.node_table["p99_tpot_ms"] = Node("p99_tpot_ms", "Y")
            runner = TickRunner(
                evidence_store=_EvidenceStore(),
                telemetry=SemanticTelemetry(),
                cusum=Cusum(),
                icp=ICP(),
                quadrant_validator=QuadrantValidator(),
                confidence_service=SimpleNamespace(candidate_graph=graph),
                slow_loop=_SlowLoop(),
                dro=_Dro(),
                mechanism_registry=_MechanismRegistry(),
                resource_map=_ResourceMap(),
                agent=object(),
                plan_validator=object(),
                executor=object(),
                candidate_graph=graph,
            )
            ctx = TickContext(tick=1, cluster_snapshot=snapshot)
            runner.S1(ctx)
            runner.S2(ctx)
            return ctx.evidence_rows[0]

        idle = evidence_for_depth(0.0)
        queued = evidence_for_depth(100.0)

        self.assertNotIn("throughput_token_per_sec", idle.residuals_per_y)
        self.assertIn("p99_ttft_ms", idle.residuals_per_y)
        self.assertIn("throughput_token_per_sec", queued.residuals_per_y)
        self.assertNotIn("p99_ttft_ms", queued.residuals_per_y)
        self.assertNotIn("p99_tpot_ms", queued.residuals_per_y)

    def test_s3_records_confidence_and_slow_dro_diagnostics(self):
        graph, mechanism = _diagnostic_graph()
        registry = MechanismRegistry(
            mechanism_table={mechanism.mechanism_id: mechanism},
            mechanism_metadata_table={
                mechanism.mechanism_id: MechanismMetadata(mechanism.mechanism_id)
            },
        )
        confidence = ConfidenceService(graph, registry)
        dro = DRO()
        slow_loop = _S3SlowLoop(dro)
        runner = TickRunner(
            evidence_store=object(),
            telemetry=object(),
            cusum=object(),
            icp=object(),
            quadrant_validator=object(),
            confidence_service=confidence,
            slow_loop=slow_loop,
            dro=dro,
            mechanism_registry=registry,
            resource_map=object(),
            agent=object(),
            plan_validator=object(),
            executor=object(),
            candidate_graph=graph,
            recalibrate_every=0,
        )
        ctx = TickContext(
            tick=1,
            evidence_rows=[
                SimpleNamespace(
                    row_id="1_job_1_rank_a",
                    job_id="job_1",
                    rank_id="rank_a",
                    env_label=ENV_LABEL,
                    q_label_per_mechanism={mechanism.mechanism_id: Quadrant.Q4},
                    icp_result_per_edge=dict.fromkeys(mechanism.edge_ids, ICPResult.UNDECIDED),
                    residuals_per_y={"p99_ttft_ms": np.array([0.0])},
                    y_observed_mean={"p99_ttft_ms": 100.0},
                    y_predicted={"p99_ttft_ms": 100.0},
                    prediction_lineage={
                        "decision_dro_band": {
                            "p99_ttft_ms": {
                                "point": 100.0,
                                "lower": 99.85,
                                "upper": 100.15,
                            }
                        },
                        "decision_required_objectives": ["p99_ttft_ms"],
                    },
                )
            ],
        )

        runner.S3(ctx)

        confidence_change = ctx.confidence_diagnostics[0]
        self.assertEqual(confidence_change["mechanism"]["before"]["beta"], 1.0)
        self.assertEqual(confidence_change["mechanism"]["delta"], {"alpha": 0.0, "beta": 1.5})
        self.assertEqual(confidence_change["mechanism"]["after"]["beta"], 2.5)
        self.assertEqual(confidence_change["edges"][0]["icp_result"], "undecided")
        self.assertEqual(confidence_change["edges"][0]["delta"], {"alpha": 0.0, "beta": 0.5})

        slow_change = ctx.slow_update_diagnostics
        self.assertEqual(slow_change["before"]["beta_t"], 0.5)
        self.assertEqual(slow_change["after"]["beta_t"], 0.6)
        self.assertEqual(slow_change["dro"]["coverage"]["inside_rows"], 1)
        self.assertAlmostEqual(slow_change["dro"]["before"]["epsilon"], 0.15)
        self.assertAlmostEqual(slow_change["dro"]["after"]["epsilon"], 0.1425)

    def test_s3_penalizes_partial_divergence_without_claiming_q4(self):
        graph, mechanism = _diagnostic_graph()
        registry = MechanismRegistry(
            mechanism_table={mechanism.mechanism_id: mechanism},
            mechanism_metadata_table={
                mechanism.mechanism_id: MechanismMetadata(mechanism.mechanism_id)
            },
        )
        confidence = ConfidenceService(graph, registry)
        dro = DRO()
        runner = TickRunner(
            evidence_store=object(),
            telemetry=object(),
            cusum=object(),
            icp=object(),
            quadrant_validator=object(),
            confidence_service=confidence,
            slow_loop=_S3SlowLoop(dro),
            dro=dro,
            mechanism_registry=registry,
            resource_map=object(),
            agent=object(),
            plan_validator=object(),
            executor=object(),
            candidate_graph=graph,
            recalibrate_every=0,
        )
        ctx = TickContext(
            tick=1,
            evidence_rows=[
                SimpleNamespace(
                    row_id="1_job_1_rank_a",
                    job_id="job_1",
                    rank_id="rank_a",
                    env_label=ENV_LABEL,
                    q_label_per_mechanism={mechanism.mechanism_id: None},
                    cusum_per_mechanism={mechanism.mechanism_id: (None, CusumResult.DIVERGED)},
                    residuals_per_v={"kv_cache_util": np.array([0.1])},
                    residuals_per_y={"p99_ttft_ms": np.array([10.0])},
                    icp_result_per_edge={},
                    y_observed_mean={"p99_ttft_ms": 110.0},
                    y_predicted={"p99_ttft_ms": 100.0},
                    prediction_lineage=None,
                )
            ],
        )

        runner.S3(ctx)

        metadata = registry.mechanism_metadata_table[mechanism.mechanism_id]
        self.assertEqual(metadata.alpha, 1.0)
        self.assertEqual(metadata.beta, 1.5)
        self.assertEqual(metadata.q_histogram["Q4"], 0)
        self.assertTrue(ctx.confidence_diagnostics[0]["partial_divergence"])


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL") and shutil.which("dbmate"),
    "Requires dbmate and TEST_DATABASE_URL pointing to local PostgreSQL",
)
class ResourceMapCatalogPostgresTests(unittest.TestCase):
    def setUp(self):
        url = os.environ["TEST_DATABASE_URL"]
        parsed = urlsplit(url)
        self.assertIn(parsed.hostname, {"localhost", "127.0.0.1", "::1"})
        name = f"koi_catalog_test_{uuid4().hex}"
        admin = psycopg.connect(url, autocommit=True)
        self.addCleanup(admin.close)
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        self.addCleanup(
            admin.execute,
            sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)),
        )
        database_url = urlunsplit(parsed._replace(path=f"/{name}"))
        subprocess.run(
            [
                "dbmate",
                "--migration-lock",
                "--no-dump-schema",
                "--migrations-dir",
                str(Path(__file__).resolve().parents[2] / "db" / "migrations"),
                "migrate",
                "--strict",
            ],
            env={**os.environ, "DATABASE_URL": database_url},
            check=True,
            timeout=60,
        )
        self.pool = self.enterContext(postgres_pool(database_url))

    def test_waiting_jobs_are_read_from_jobs_and_intent(self):
        model_id = "test/queued-context"
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO model_architecture (model_id) VALUES (%s)", (model_id,))
            for job_id, kind, status in (
                ("queued-b", "batch", "queued"),
                ("queued-a", "online", "queued"),
                ("running-job", "online", "running"),
                ("finished-job", "batch", "finished"),
            ):
                conn.execute(
                    "INSERT INTO jobs (job_id, kind, status, model_id) VALUES (%s, %s, %s, %s)",
                    (job_id, kind, status, model_id),
                )
                conn.execute(
                    "INSERT INTO job_intent "
                    "(job_id, target_p99_ttft_ms, deadline_hrs, is_session_affinity) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        job_id,
                        600.0 if kind == "online" else None,
                        0.0 if kind == "online" else 2.0,
                        False,
                    ),
                )

        manager = ResourceMapManager(pool=self.pool, postgres_client=object())
        jobs = manager.get_waiting_jobs()
        self.assertEqual([job["job_id"] for job in jobs], ["queued-a", "queued-b"])
        for job, kind in zip(jobs, ("online", "batch"), strict=True):
            self.assertEqual(job["kind"], kind)
            self.assertEqual(job["status"], "queued")
            features = job["job_features"]
            self.assertEqual(features["type"], kind)
            self.assertEqual(features["model_id"], model_id)
            self.assertEqual(features["target_p99_ttft_ms"], 600.0 if kind == "online" else None)
            self.assertEqual(features["deadline_hrs"], 0.0 if kind == "online" else 2.0)
            self.assertFalse(features["is_session_affinity"])
            self.assertIsNone(features["priority_class"])
            self.assertNotIn("user_id", job)
            self.assertNotIn("spec_json", job)
            self.assertNotIn("recent_rank_failures", job)
        self.assertEqual(manager.get_submitted_jobs(), jobs)
        manager.resources_summary = lambda: {}
        manager.get_running_jobs = lambda **kwargs: []
        self.assertEqual(manager.snapshot_cluster_state(5).pending_jobs_summary(), jobs)
        with self.assertRaisesRegex(ValueError, "database pool is required"):
            ResourceMapManager().get_waiting_jobs()

    def test_catalogs_are_read_from_postgres(self):
        # Schema-supported values from the legacy model_catalog logged by
        # koi-3h-cohere-20260906T051213Z on 2026-09-06, using canonical names. updated_at and
        # runtime_image are not represented in the current schema.
        legacy_catalog = {
            "model_id": "meta-llama/Llama-3.3-70B-Instruct",
            "engine_name": "vllm",
            "engine_version": "0.16.0",
            "model_params_b": 70.553706496,
            "model_size_gb": 141.107412992,
            "num_hidden_layers": 80,
            "hidden_size": 8192,
            "num_attn_heads": 64,
            "num_kv_heads": 8,
            "attn_heads_per_kv_head": 8,
            "intermediate_size": 28672,
            "max_pos_embeddings": 131072,
            "vocab_size": 128256,
            "is_moe": False,
            "num_routed_experts": 0,
            "num_active_experts": 0,
            "flops_per_param": None,
            "weight_dtype": "bfloat16",
            "weight_quantization_method": "none",
            "weight_quantization_bits": 16,
            "sliding_window_size": None,
            "min_chain_warmup_time": 10.0,
            "max_model_len": 131072,
            "gpu_mem_util": 0.9,
            "max_num_seq": [
                {"gpu_type": "H100", "value": 1024},
                {"gpu_type": "A100-80GB", "value": 256},
                {"gpu_type": "RTXPRO6000", "value": 256},
            ],
            "max_num_batched_tokens": [
                {"gpu_type": "H100", "value": 8192},
                {"gpu_type": "A100-80GB", "value": 2048},
                {"gpu_type": "RTXPRO6000", "value": 2048},
            ],
            "block_size": [
                {"gpu_type": "H100", "value": 16},
                {"gpu_type": "A100-80GB", "value": 16},
                {"gpu_type": "RTXPRO6000", "value": 16},
            ],
            "kv_cache_dtype": [
                {"gpu_type": "H100", "value": "auto"},
                {"gpu_type": "A100-80GB", "value": "auto"},
                {"gpu_type": "RTXPRO6000", "value": "auto"},
            ],
            "attn_backend": "auto",
            "activation_quantization_method": "none",
            "activation_dtype": "bfloat16",
            "prefix_cache_enabled": True,
            "chunked_prefill_enabled": True,
            "lmcache_enabled": False,
            "spec_decoding_enabled": False,
            "draft_model_id": "",
            "spec_decoding_method": "none",
            "num_speculative_tokens": 0,
            "spec_acceptance_threshold": 0.0,
            "cuda_graph_enabled": True,
            "torch_compile_enabled": True,
            "preemption_policy": "recompute",
            "pd_enabled": False,
            "kv_transfer_method": "default",
            "chunk_size": 100,
            "max_chunked_steps_per_request": 2,
        }
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO hardware_catalog (cloud_name, catalog) VALUES (%s, %s)",
                ("aws", Jsonb(_hardware_catalog())),
            )
            conn.execute(
                "INSERT INTO hardware_catalog (cloud_name, catalog) VALUES (%s, %s)",
                ("azure", Jsonb(_azure_hardware_catalog())),
            )
            conn.execute(
                "INSERT INTO model_architecture "
                "(model_id, model_params_b, model_size_gb, num_hidden_layers, hidden_size, "
                "num_attn_heads, num_kv_heads, attn_heads_per_kv_head, intermediate_size, "
                "max_pos_embeddings, vocab_size, is_moe, num_routed_experts, "
                "num_active_experts, flops_per_param, weight_dtype, "
                "weight_quantization_method, weight_quantization_bits, sliding_window_size, "
                "min_chain_warmup_time) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s)",
                tuple(
                    legacy_catalog[key]
                    for key in (
                        "model_id",
                        "model_params_b",
                        "model_size_gb",
                        "num_hidden_layers",
                        "hidden_size",
                        "num_attn_heads",
                        "num_kv_heads",
                        "attn_heads_per_kv_head",
                        "intermediate_size",
                        "max_pos_embeddings",
                        "vocab_size",
                        "is_moe",
                        "num_routed_experts",
                        "num_active_experts",
                        "flops_per_param",
                        "weight_dtype",
                        "weight_quantization_method",
                        "weight_quantization_bits",
                        "sliding_window_size",
                        "min_chain_warmup_time",
                    )
                ),
            )
            conn.execute(
                "INSERT INTO job_engine_defaults "
                "(model_id, engine_name, engine_version, max_num_seq, max_num_batched_tokens, "
                "block_size, kv_cache_dtype, "
                "gpu_mem_util, max_model_len, attn_backend, activation_quantization_method, "
                "activation_dtype, prefix_cache_enabled, chunked_prefill_enabled, "
                "lmcache_enabled, spec_decoding_enabled, draft_model_id, spec_decoding_method, "
                "num_speculative_tokens, spec_acceptance_threshold, cuda_graph_enabled, "
                "torch_compile_enabled, preemption_policy, pd_enabled, kv_transfer_method, "
                "chunk_size, max_chunked_steps_per_request) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    legacy_catalog["model_id"],
                    legacy_catalog["engine_name"],
                    legacy_catalog["engine_version"],
                    Jsonb(legacy_catalog["max_num_seq"]),
                    Jsonb(legacy_catalog["max_num_batched_tokens"]),
                    Jsonb(legacy_catalog["block_size"]),
                    Jsonb(legacy_catalog["kv_cache_dtype"]),
                    legacy_catalog["gpu_mem_util"],
                    legacy_catalog["max_model_len"],
                    legacy_catalog["attn_backend"],
                    legacy_catalog["activation_quantization_method"],
                    legacy_catalog["activation_dtype"],
                    legacy_catalog["prefix_cache_enabled"],
                    legacy_catalog["chunked_prefill_enabled"],
                    legacy_catalog["lmcache_enabled"],
                    legacy_catalog["spec_decoding_enabled"],
                    legacy_catalog["draft_model_id"],
                    legacy_catalog["spec_decoding_method"],
                    legacy_catalog["num_speculative_tokens"],
                    legacy_catalog["spec_acceptance_threshold"],
                    legacy_catalog["cuda_graph_enabled"],
                    legacy_catalog["torch_compile_enabled"],
                    legacy_catalog["preemption_policy"],
                    legacy_catalog["pd_enabled"],
                    legacy_catalog["kv_transfer_method"],
                    legacy_catalog["chunk_size"],
                    legacy_catalog["max_chunked_steps_per_request"],
                ),
            )
            conn.execute(
                "INSERT INTO model_architecture (model_id) VALUES (%s)",
                ("architecture-without-defaults",),
            )

        manager = ResourceMapManager(pool=self.pool, postgres_client=object())
        hardware = manager.hardware_catalog()
        by_instance = {
            (region["cloud"], region["region"], instance["instance_type"])
            for region in hardware["regions"]
            for instance in region["instance_types"]
        }
        self.assertIn(("aws", "us-east-2", "p5.48xlarge"), by_instance)
        self.assertIn(("azure", "eastus", "Standard_ND96amsr_A100_v4"), by_instance)

        catalog = manager.model_catalog(legacy_catalog["model_id"])
        for field, expected_value in legacy_catalog.items():
            with self.subTest(field=field):
                self.assertEqual(catalog[field], expected_value)
        for gpu_type, expected in (
            ("H100", 1024),
            ("A100-80GB", 256),
            ("RTXPRO6000", 256),
        ):
            config = materialize_launch_config(catalog, gpu_type)
            legacy_config = materialize_launch_config(legacy_catalog, gpu_type)
            self.assertEqual(config, legacy_config)
            self.assertEqual(config["engine_name"], "vllm")
            self.assertEqual(config["engine_version"], "0.16.0")
            self.assertEqual(config["max_num_seq"], expected)
            self.assertEqual(config["kv_cache_dtype"], "auto")
            self.assertTrue(config["chunked_prefill_enabled"])

        for missing_model_id in (
            "missing-model",
            "architecture-without-defaults",
            "' OR 1=1 --",
        ):
            with (
                self.subTest(model_id=missing_model_id),
                self.assertRaisesRegex(ValueError, "model catalog missing"),
            ):
                manager.model_catalog(missing_model_id)


class _Telemetry:
    def collect_telemetry(self, tick_start, tick_end, snapshot):
        return "bundle"

    def iter_per_rank(self, bundle):
        yield SimpleNamespace(
            job_id="job_1",
            rank_id="rank_a",
            v_observed={"kv_cache_util": np.array([0.2, 0.3])},
            y_observed={"p99_ttft_ms": np.array([100.0, 110.0])},
        )


class _EvidenceStore:
    def __init__(self):
        self.rows = []

    def append_row(self, row):
        self.rows.append(row)

    def get_rows_for_edge(self, edge_id, limit=None):
        return []


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


class _S3SlowLoop:
    def __init__(self, dro):
        self.dro = dro
        self.state = SimpleNamespace(
            w_t={"p99_ttft_ms": 1.0},
            z_star_t={"p99_ttft_ms": 100.0},
            lambda_swit=0.05,
            beta_t=0.5,
            B_t=1,
            epsilon_dro=0.15,
            regret_slope=0.0,
            q1_rate=0.0,
            observed_swap_rate=0.0,
            observed_coverage=0.9,
            cusum_params_v={},
            cusum_params_y={},
            tick=0,
        )

    @staticmethod
    def anneal_targets(tick):
        return {"target_swap_rate": 0.1, "target_slope": 0.1}

    def slow_update_all(
        self,
        tick,
        observed_swap_rate,
        observed_coverage,
        r2_gradient,
        target_overrides,
    ):
        self.state.tick = tick
        self.state.observed_swap_rate = observed_swap_rate
        self.state.observed_coverage = observed_coverage
        self.state.beta_t = 0.6
        self.state.B_t = 10
        self.state.lambda_swit = 0.04
        self.state.epsilon_dro = self.dro.update_epsilon_dro(
            self.state.epsilon_dro, observed_coverage
        )
        return self.state


class _Cusum:
    def cusum_params_per_v(self, name, residuals):
        return 0.0, 1.0


class _Dro:
    def append_residual_history(self, pred_y, obs_y):
        pass


class _MechanismRegistry:
    def __init__(self, mechanisms=()):
        self.mechanisms = list(mechanisms)

    def find_applicable(self, context):
        self.context = context
        return [(mechanism, 1.0) for mechanism in self.mechanisms]


class _ResourceMap:
    def hardware_catalog(self):
        return _hardware_catalog()

    def model_catalog(self, model_id):
        return _model_catalogs()[model_id]


class _IdleResourceMap:
    def hardware_catalog(self):
        raise AssertionError("idle S1 must not fetch hardware catalog")

    def model_catalog(self, model_id):
        raise AssertionError("idle S1 must not fetch model catalog")


if __name__ == "__main__":
    unittest.main()
