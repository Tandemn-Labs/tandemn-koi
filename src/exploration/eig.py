"""EIG proxy for causal expected information gain.

Exact Bayesian EIG needs outcome distributions and mechanism posteriors,
neither of which we have. This deterministic proxy keeps the same purpose:
favor candidates that test uncertain and under-sampled edges and mechanisms.

    eig(L_prime) = sum_e a_e * u(c_e) * w_rare(n_e)
                 + mechanism_weight * sum_M a_M * u(c_M) * w_rare(n_M)

u(c) = 4*c*(1-c) peaks at uncertainty c=0.5. w_rare(n) = 1/sqrt(1+n) rewards
less-tested structures. Eligibility masks a_e and a_M decide what is tested.
The eig value is the exploration term in sigma, weighted by the annealed
exploration weight. Cluster EIG uses saturation aggregation to avoid
double-counting an edge tested by multiple ranks in one plan.

Resolution conventions:
    Mechanism.edge_ids is the canonical bundle representation (list[str]).
    Edges resolve via candidate_graph.edge_table[edge_id] -> Edge.
    Confidences and visit counts come from confidence_service, not from a
    separate edge_table. Eligibility gates read from evidence_store.

v0 scope: EIG iterates over each rank's committed mechanism_id. A rank may
also accrue Beta updates to other applicable mechanisms post-deploy via the
EvidenceStore fan-out in S3, but EIG conservatively counts only
committed-mechanism evidence at scoring time. v1 may extend to the
predicted-applicable-set.
"""

from collections.abc import Sequence

import numpy as np

DEFAULT_N_B = 15
DEFAULT_N_ENV_MIN = 3


def compute_eig(
    L_prime,
    candidate_graph,
    mechanism_registry,
    confidence_service,
    evidence_store,
    mechanism_weight: float = 1.0,
    visit_cap: int | None = None,
) -> float:
    """Compute the proxy Causal-EIG for one candidate ladder.

    Sums over edges and mechanisms touched by L_prime's committed
    mechanisms. This is the exploration term in
    sigma(L_prime) = J + beta * eig - gamma * Pr_DRO - lambda * SwitchCost.

    Args:
        L_prime: Ladder with .ranks; each rank carries .mechanism_id
            (committed), .config, .n_replicas.
        candidate_graph: CandidateGraph; resolves edge_id to Edge.
        mechanism_registry: MechanismRegistry exposing get_mechanism(id).
        confidence_service: ConfidenceService for confidence and visit
            count lookups.
        evidence_store: EvidenceStore for eligibility-gate inputs.
        mechanism_weight: Edge-vs-mechanism term weighting.
        visit_cap: Optional cap on visit count used in the rarity bonus.

    Returns:
        Non-negative EIG value.
    """
    if not L_prime.ranks:
        return 0.0

    deployed_mids = {r.mechanism_id for r in L_prime.ranks}
    if not deployed_mids:
        return 0.0

    touched_edge_ids: set[str] = set()
    for mid in deployed_mids:
        mech = mechanism_registry.get_mechanism(mid)
        touched_edge_ids |= set(mech.edge_ids)

    edge_sum = 0.0
    for edge_id in touched_edge_ids:
        edge = candidate_graph.edge_table[edge_id]
        if not _edge_eligible(edge, L_prime, evidence_store):
            continue
        c_e = confidence_service.get_edge_confidence(edge_id)
        n_e = confidence_service.get_edge_visit_count(edge_id)
        if visit_cap is not None:
            n_e = min(n_e, visit_cap)
        edge_sum += _bernoulli_variance(c_e) * _rarity(n_e)

    mech_sum = 0.0
    for mid in deployed_mids:
        mech = mechanism_registry.get_mechanism(mid)
        if not check_mechanism_eligibility(mech, L_prime, candidate_graph, evidence_store):
            continue
        c_m = confidence_service.get_mechanism_confidence(mid)
        n_m = confidence_service.get_mechanism_visit_count(mid)
        if visit_cap is not None:
            n_m = min(n_m, visit_cap)
        mech_sum += _bernoulli_variance(c_m) * _rarity(n_m)

    return edge_sum + mechanism_weight * mech_sum


def check_mechanism_eligibility(
    mechanism,
    L_prime,
    candidate_graph,
    evidence_store,
) -> bool:
    """Return True iff at least one X->V->Y path in the mechanism is eligible.

    a_M(L_prime) = 1 iff at least one X->V->Y path through M has both of
    its edges eligible. Ensures the mechanism is testable by L_prime:
    the ladder actually exercises some causal path through the bundle.

    Args:
        mechanism: Mechanism with .edge_ids.
        L_prime: Ladder.
        candidate_graph: CandidateGraph.
        evidence_store: EvidenceStore for the validator-support gate.

    Returns:
        True if any X->V->Y path is fully eligible.
    """
    for xv_edge, vy_edge in find_eligible_paths(mechanism, candidate_graph):
        if _edge_eligible(xv_edge, L_prime, evidence_store) and _edge_eligible(
            vy_edge, L_prime, evidence_store
        ):
            return True
    return False


def aggregate_cluster_eig(
    cluster_plan,
    ranks: Sequence,
    candidate_graph,
    mechanism_registry,
    confidence_service,
    evidence_store,
    mechanism_weight: float = 1.0,
) -> float:
    """Compute cluster-level EIG with saturation aggregation.

    A_e(P) = 1 - product_i(1 - a_e(L_i_prime)).
    eig_cluster(P) = sum_e u(c_e) * A_e + mechanism_weight * sum_M u(c_M) * A_M.

    Saturation prevents double-counting an edge tested by multiple ranks
    across the cluster's plan.

    Args:
        cluster_plan: Plan dict[job_id -> Action], retained for audit/logging.
        ranks: Flat list of ranks across all ladders in the plan.
        candidate_graph: CandidateGraph.
        mechanism_registry: MechanismRegistry.
        confidence_service: ConfidenceService.
        evidence_store: EvidenceStore.
        mechanism_weight: Edge-vs-mechanism term weighting.

    Returns:
        Non-negative cluster-level EIG.
    """
    if not ranks:
        return 0.0

    edges_to_ranks: dict[str, list] = {}
    mechs_to_ranks: dict[str, list] = {}
    for r in ranks:
        mech = mechanism_registry.get_mechanism(r.mechanism_id)
        for edge_id in mech.edge_ids:
            edges_to_ranks.setdefault(edge_id, []).append(r)
        mechs_to_ranks.setdefault(r.mechanism_id, []).append(r)

    edge_term = 0.0
    for edge_id, rank_list in edges_to_ranks.items():
        edge = candidate_graph.edge_table[edge_id]
        c_e = confidence_service.get_edge_confidence(edge_id)
        a_values = [
            1.0 if _edge_eligible(edge, r.ladder, evidence_store) else 0.0 for r in rank_list
        ]
        saturation = 1.0 - float(np.prod([1.0 - a for a in a_values]))
        edge_term += _bernoulli_variance(c_e) * saturation

    mech_term = 0.0
    for mid, rank_list in mechs_to_ranks.items():
        mech = mechanism_registry.get_mechanism(mid)
        c_m = confidence_service.get_mechanism_confidence(mid)
        a_values = [
            1.0
            if check_mechanism_eligibility(mech, r.ladder, candidate_graph, evidence_store)
            else 0.0
            for r in rank_list
        ]
        saturation = 1.0 - float(np.prod([1.0 - a for a in a_values]))
        mech_term += _bernoulli_variance(c_m) * saturation

    return edge_term + mechanism_weight * mech_term


def find_eligible_paths(mechanism, candidate_graph) -> list[tuple]:
    """Enumerate full X->V->Y paths through a mechanism's bundle.

    Args:
        mechanism: Mechanism with .edge_ids.
        candidate_graph: CandidateGraph.

    Returns:
        List of (xv_edge, vy_edge) pairs forming X->V->Y paths.
    """
    edges = [candidate_graph.edge_table[eid] for eid in mechanism.edge_ids]
    xv_edges = [e for e in edges if e.src_type == "X" and e.dst_type == "V"]
    vy_edges = [e for e in edges if e.src_type == "V" and e.dst_type == "Y"]
    return [(a, b) for a in xv_edges for b in vy_edges if a.dst == b.src]


def _bernoulli_variance(c: float) -> float:
    """Return 4*c*(1-c). Peaks at c=0.5, zero at c in {0, 1}."""
    return 4.0 * float(c) * (1.0 - float(c))


def _rarity(n: int) -> float:
    """Return 1 / sqrt(1 + n) as a UCB-style rarity bonus on visit count."""
    return 1.0 / np.sqrt(1.0 + float(n))


def _edge_eligible(edge, L_prime, evidence_store) -> bool:
    """Return True iff all six eligibility gates pass for this edge."""
    return (
        _gate_selected(edge, L_prime)
        and _gate_valid_contrast(edge, L_prime)
        and _gate_child_observed(edge)
        and _gate_enough_samples(edge, L_prime)
        and _gate_validator_support(edge, evidence_store)
        and _gate_relevance(edge, L_prime)
    )


def _gate_selected(edge, L_prime) -> bool:
    """Return True iff the X-side of edge is set or varied in the ladder."""
    return any(edge.src in r.config for r in L_prime.ranks)


def _gate_valid_contrast(edge, L_prime) -> bool:
    """Return True iff the ladder produces variation in edge.src.

    X->V edges need at least one distinct value of edge.src. V->Y edges
    have V variation mediated by upstream X-variation in the same ladder,
    which is validated structurally elsewhere.
    """
    values = {r.config.get(edge.src) for r in L_prime.ranks if edge.src in r.config}
    return len(values) >= 1


def _gate_child_observed(edge) -> bool:
    """Return True iff the V or Y on the dst side is in the telemetry catalog."""
    return getattr(edge, "dst_observable", True)


def _gate_enough_samples(edge, L_prime, n_b: int = DEFAULT_N_B) -> bool:
    """Return True iff the deployment provides at least n_b samples per env."""
    n_envs = max(1, len(L_prime.envs()))
    total = L_prime.duration_minutes * sum(r.n_replicas for r in L_prime.ranks)
    return total >= n_b * n_envs


def _gate_validator_support(
    edge,
    evidence_store,
    n_env_min: int = DEFAULT_N_ENV_MIN,
) -> bool:
    """Return True iff the edge will have >= n_env_min envs tested post-deploy."""
    return len(evidence_store.envs_for_edge(edge.edge_id)) + 1 >= n_env_min


def _gate_relevance(edge, L_prime) -> bool:
    """Return True iff the edge belongs to a mechanism applicable to L_prime."""
    return any(edge.edge_id in M.edge_ids for M in L_prime.applicable_mechanisms)


# if __name__ == "__main__":
#     from dataclasses import dataclass
#     from typing import Any

#     from src.core.candidate_graph import CandidateGraph
#     from src.core.confidence_service import ConfidenceService
#     from src.core.mechanism_registry import MechanismRegistry
#     from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node

#     @dataclass
#     class Rank:
#         mechanism_id: str
#         mechanism: Mechanism
#         config: dict[str, Any]
#         n_replicas: int
#         ladder: Any = None
#         evidence_store: Any = None

#     @dataclass
#     class Ladder:
#         ranks: list[Rank]
#         duration_minutes: int
#         applicable_mechanisms: list[Mechanism]

#         def envs(self):
#             return ["env_a", "env_b", "env_c"]

#     class FakeEvidenceStore:
#         def __init__(self, edge_envs):
#             self.edge_envs = edge_envs

#         def envs_for_edge(self, edge_id):
#             return self.edge_envs.get(edge_id, [])

#     def build_test_case():
#         e1 = Edge(
#             edge_id="edge_batch_to_kv",
#             src="batch_size",
#             dst="kv_cache_pressure",
#             src_type="X",
#             dst_type="V",
#         )
#         e2 = Edge(
#             edge_id="edge_kv_to_latency",
#             src="kv_cache_pressure",
#             dst="ttft_ms",
#             src_type="V",
#             dst_type="Y",
#         )

#         mechanism = Mechanism(
#             mechanism_id="mech_kv_latency",
#             edge_ids=[e1.edge_id, e2.edge_id],
#             scope={"x": ["batch_size"], "v": ["kv_cache_pressure"]},
#             narrative="KV pressure mediates batch size and TTFT.",
#         )

#         rank_1 = Rank(
#             mechanism_id=mechanism.mechanism_id,
#             mechanism=mechanism,
#             config={"batch_size": 8, "kv_cache_pressure": "medium"},
#             n_replicas=30,
#         )
#         rank_2 = Rank(
#             mechanism_id=mechanism.mechanism_id,
#             mechanism=mechanism,
#             config={"batch_size": 16, "kv_cache_pressure": "high"},
#             n_replicas=30,
#         )

#         evidence_store = FakeEvidenceStore(
#             edge_envs={
#                 e1.edge_id: ["env_a", "env_b"],
#                 e2.edge_id: ["env_a", "env_b"],
#             }
#         )
#         ladder = Ladder(
#             ranks=[rank_1, rank_2],
#             duration_minutes=10,
#             applicable_mechanisms=[mechanism],
#         )

#         rank_1.ladder = ladder
#         rank_2.ladder = ladder
#         rank_1.evidence_store = evidence_store
#         rank_2.evidence_store = evidence_store

#         node_table = {
#             "batch_size": Node(node_id="batch_size", node_type="X"),
#             "kv_cache_pressure": Node(node_id="kv_cache_pressure", node_type="V"),
#             "ttft_ms": Node(node_id="ttft_ms", node_type="Y"),
#         }
#         edge_table = {e1.edge_id: e1, e2.edge_id: e2}
#         edge_metadata_table = {
#             e1.edge_id: EdgeMetadata(edge_id=e1.edge_id, alpha=1.5, beta=1.5, visit_count=3),
#             e2.edge_id: EdgeMetadata(edge_id=e2.edge_id, alpha=5.6, beta=2.4, visit_count=8),
#         }
#         graph = CandidateGraph(
#             node_table=node_table,
#             edge_table=edge_table,
#             edge_metadata_table=edge_metadata_table,
#         )

#         registry = MechanismRegistry(
#             mechanism_table={mechanism.mechanism_id: mechanism},
#             mechanism_metadata_table={
#                 mechanism.mechanism_id: MechanismMetadata(
#                     mechanism_id=mechanism.mechanism_id,
#                     alpha=3.0,
#                     beta=2.0,
#                     visit_count=5,
#                 )
#             },
#         )
#         confidence_service = ConfidenceService(graph, registry)

#         return ladder, graph, registry, confidence_service, evidence_store, [rank_1, rank_2]

#     ladder, graph, registry, confidence_service, evidence_store, ranks = build_test_case()

#     alpha = compute_eig(
#         L_prime=ladder,
#         candidate_graph=graph,
#         mechanism_registry=registry,
#         confidence_service=confidence_service,
#         evidence_store=evidence_store,
#     )
#     print("compute_eig alpha:", alpha)

#     alpha_capped = compute_eig(
#         L_prime=ladder,
#         candidate_graph=graph,
#         mechanism_registry=registry,
#         confidence_service=confidence_service,
#         evidence_store=evidence_store,
#         visit_cap=2,
#     )
#     print("compute_eig alpha with evidence cap n=2:", alpha_capped)

#     alpha_cluster = aggregate_cluster_eig(
#         cluster_plan={"job_1": "fake_action"},
#         ranks=ranks,
#         candidate_graph=graph,
#         mechanism_registry=registry,
#         confidence_service=confidence_service,
#         evidence_store=evidence_store,
#     )
#     print("aggregate_cluster_eig alpha_cluster:", alpha_cluster)

#     assert alpha > 0
#     assert alpha_capped > alpha
#     assert alpha_cluster > 0
#     print("All EIG tests passed.")
