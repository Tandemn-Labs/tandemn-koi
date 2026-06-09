"""
EIG proxy for causal expected information gain.
Exact Bayesian EIG needs outcome distributions and mechanism posteriors, which
we do not have. This deterministic proxy keeps the same purpose: favor candidates
that test uncertain and under-sampled edges/mechanisms.
    alpha(L') = sum_e a_e*u(c_e)*w_rare(n_e) + kappa*sum_M a_M*u(c_M)*w_rare(n_M)
u(c)=4c(1-c) peaks at uncertainty c=0.5. w_rare(n)=1/sqrt(1+n) rewards
less-tested structures. Eligibility masks a_e and a_M decide what is tested.
alpha is the exploration term in sigma, weighted by annealed beta_t. Cluster EIG
uses saturation aggregation to avoid double-counting the same edge in one plan.
"""

from collections.abc import Sequence

import numpy as np
from src.config.hyperparameters import KAPPA

# Gate defaults
DEFAULT_N_B = 15  # min samples per env for ICP statistical power
DEFAULT_N_ENV_MIN = 3  # min envs required for ICP


def compute_eig(
    L_prime,
    edge_table,
    mechanism_registry,
    evidence_store,
    n: int | None = None,
) -> float:
    """
    Definition: Proxy Causal-EIG for one candidate ladder.
                    alpha(L') = sum_e a_e*u(c(e))*w_rare(n_e)
                          + kappa*sum_M a_M*u(c(M))*w_rare(n_M)
                Sums over edges/mechanisms touched by L'.
    Usage:      The alpha term in sigma(L') = J + beta*alpha - lambda*Pr_DRO - lambda*SwitchCost.
                Called by agent.tools.compute_eig per (config, mechanism)
    Inputs:
        L_prime            : Ladder with .ranks; each rank has .mechanism_id,
                             .config, .n_replicas
        edge_table         : EdgeTable with .confidence(edge), .visit_count(edge)
        mechanism_registry : MechanismRegistry with .get(id), .confidence(id),
                             .visit_count(id)
        evidence_store     : EvidenceStore (for eligibility-gate lookups)
        n                  : optional visit-count cap (n_e_used = min(n_e, n))
    Outputs:
        alpha : float >= 0
    """
    if not L_prime.ranks:
        return 0.0

    deployed_mids = {r.mechanism_id for r in L_prime.ranks}
    if not deployed_mids:
        return 0.0

    # Union of edges across all deployed mechanisms
    touched_edges = set()
    for mid in deployed_mids:
        touched_edges |= set(mechanism_registry.get(mid).edges)

    # Edge term
    edge_sum = 0.0
    for e in touched_edges:
        if not _edge_eligible(e, L_prime, evidence_store):
            continue
        c_e = edge_table.confidence(e)
        n_e = edge_table.visit_count(e)
        if n is not None:
            n_e = min(n_e, n)
        edge_sum += _u(c_e) * _rare(n_e)

    # Mechanism term
    mech_sum = 0.0
    for mid in deployed_mids:
        M = mechanism_registry.get(mid)
        if not check_mechanism_eligibility(M, L_prime, (edge_table, evidence_store)):
            continue
        c_m = mechanism_registry.confidence(mid)
        n_m = mechanism_registry.visit_count(mid)
        if n is not None:
            n_m = min(n_m, n)
        mech_sum += _u(c_m) * _rare(n_m)

    return edge_sum + KAPPA * mech_sum


def check_mechanism_eligibility(mechanism, L_prime, state) -> bool:
    """
    Definition: a_M(L') = 1 iff at least one X->V->Y path through M has
                BOTH edges eligible. Ensures mechanism is testable by L'.
    Usage:      Inner gate for compute_eig and aggregate_cluster_eig.
    Inputs:
        mechanism : Mechanism with .edges (each .kind, .src, .dst)
        L_prime   : Ladder
        state     : tuple (edge_table, evidence_store) - bundled context
    Outputs:
        bool
    """
    _edge_table, evidence_store = state
    for xv_edge, vy_edge in find_eligible_paths(mechanism):
        if _edge_eligible(xv_edge, L_prime, evidence_store) and _edge_eligible(
            vy_edge, L_prime, evidence_store
        ):
            return True
    return False


def aggregate_cluster_eig(
    cluster_plan,
    ranks: Sequence,
    edge_table,
    mechanism_registry,
) -> float:
    """
    Definition: Cluster-level EIG with saturation aggregation.
                    A_e(P) = 1 - prod_i(1 - a_e(L_i'))
                    alpha_cluster(P) = sum_e u(c(e))*A_e + kappa*sum_M u(c(M))*A_M
                Saturation prevents double-counting an edge tested by
                multiple ranks across the cluster's plan P.
    Usage:      agent.phase_4 cluster scoring; budget-reallocation delta-sigma check.
    Inputs:
        cluster_plan       : Plan (Dict[job_id -> Action]) - for logging/audit
        ranks              : flat List[Rank] across all ladders in the plan
        edge_table         : EdgeTable with .confidence_by_id(id)
        mechanism_registry : MechanismRegistry
    Outputs:
        alpha_cluster : float >= 0
    """
    if not ranks:
        return 0.0

    edges_to_ranks: dict[str, list] = {}
    mechs_to_ranks: dict[str, list] = {}
    for r in ranks:
        for e in r.mechanism.edges:
            edges_to_ranks.setdefault(e.id, []).append(r)
        mechs_to_ranks.setdefault(r.mechanism_id, []).append(r)

    # Edge saturation
    edge_term = 0.0
    for e_id, rank_list in edges_to_ranks.items():
        c_e = edge_table.confidence_by_id(e_id)
        a_values = [
            1.0 if _edge_eligible_by_id(e_id, r.ladder, r.evidence_store) else 0.0
            for r in rank_list
        ]
        A = 1.0 - float(np.prod([1.0 - a for a in a_values]))
        edge_term += _u(c_e) * A

    # Mechanism saturation
    mech_term = 0.0
    for m_id, rank_list in mechs_to_ranks.items():
        c_m = mechanism_registry.confidence(m_id)
        M = mechanism_registry.get(m_id)
        a_values = [
            1.0 if check_mechanism_eligibility(M, r.ladder, (edge_table, r.evidence_store)) else 0.0
            for r in rank_list
        ]
        A = 1.0 - float(np.prod([1.0 - a for a in a_values]))
        mech_term += _u(c_m) * A

    return edge_term + KAPPA * mech_term


def find_eligible_paths(mechanism) -> list[tuple]:
    """
    Definition: Enumerate (X->V edge, V->Y edge) path pairs through the
                mechanism's sub-DAG that share a common V node.
    Usage:      Inner helper for check_mechanism_eligibility.
    Inputs:
        mechanism : Mechanism with .edges (each .kind, .src, .dst)
    Outputs:
        List[(xv_edge, vy_edge)] - full X->V->Y paths through the bundle
    """
    xv = [e for e in mechanism.edges if e.kind == "X_to_V"]
    vy = [e for e in mechanism.edges if e.kind == "V_to_Y"]
    return [(a, b) for a in xv for b in vy if a.dst == b.src]


# math primitives


def _u(c: float) -> float:
    """Bernoulli variance: 4c(1-c). Peaks at c=0.5, zero at c in {0,1}."""
    return 4.0 * c * (1.0 - c)


def _rare(n: int) -> float:
    """1/sqrt(1+n)."""
    return 1.0 / np.sqrt(1.0 + float(n))


# eligibility gates


def _edge_eligible(edge, L_prime, evidence_store) -> bool:
    """All six gates must pass"""
    return (
        _gate_selected(edge, L_prime)
        and _gate_valid_contrast(edge, L_prime)
        and _gate_child_observed(edge)
        and _gate_enough_samples(edge, L_prime)
        and _gate_validator_support(edge, evidence_store)
        and _gate_relevance(edge, L_prime)
    )


def _edge_eligible_by_id(edge_id, L_prime, evidence_store) -> bool:
    """Resolve edge_id to Edge, then evaluate gates."""
    edge = next(
        (e for r in L_prime.ranks for e in r.mechanism.edges if e.id == edge_id),
        None,
    )
    if edge is None:
        return False
    return _edge_eligible(edge, L_prime, evidence_store)


def _gate_selected(edge, L_prime) -> bool:
    # X-side of edge is set / varied somewhere in the ladder.
    return any(edge.src in r.config for r in L_prime.ranks)


def _gate_valid_contrast(edge, L_prime) -> bool:
    """The ladder produces variation in edge.src across its ranks.
    X->V edges: need at least 1 distinct value of edge.src.
    V->Y edges: V variation is mediated by upstream X-variation
    in the same ladder (validated structurally elsewhere)."""
    values = {r.config.get(edge.src) for r in L_prime.ranks if edge.src in r.config}
    return len(values) >= 1


def _gate_child_observed(edge) -> bool:
    """V or Y on dst side is in our telemetry catalog."""
    return getattr(edge, "dst_observable", True)


def _gate_enough_samples(edge, L_prime, n_b: int = DEFAULT_N_B) -> bool:
    """Deployment provides at least n_b samples per env."""
    n_envs = max(1, len(L_prime.envs()))
    total = L_prime.duration_minutes * sum(r.n_replicas for r in L_prime.ranks)
    return total >= n_b * n_envs


def _gate_validator_support(
    edge,
    evidence_store,
    n_env_min: int = DEFAULT_N_ENV_MIN,
) -> bool:
    """After this deployment, edge will have at least n_env_min envs tested."""
    return len(evidence_store.envs_for_edge(edge.id)) + 1 >= n_env_min


def _gate_relevance(edge, L_prime) -> bool:
    """Edge belongs to at least one mechanism applicable to L_prime's job."""
    return any(edge in M.edges for M in L_prime.applicable_mechanisms)
