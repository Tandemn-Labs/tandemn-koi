import contextlib
import io
import unittest
from dataclasses import dataclass
from typing import Any

from src.core.candidate_graph import CandidateGraph
from src.core.confidence_service import ConfidenceService
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node
from src.exploration.eig import aggregate_cluster_eig, compute_eig
from src.infra.resource_map import (
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


if __name__ == "__main__":
    unittest.main()
