"""
Augmented Tchebycheff scalarization for per-job multi-objective J.
Weighted sums miss non-convex Pareto regions, which are common in inference trade-offs.
Augmented Tchebycheff uses normalized distance to the ideal point:
    max_j w_j * gap_norm_j + rho * sum_j w_j * gap_norm_j
The max term improves Pareto coverage, while the small rho term breaks weak Pareto ties. 
Typical rho: 1e-4 to 1e-2. Tandemn's sigma uses argmax, so we flip the sign: larger J means closer to ideal.
Gaps are normalized by typical_range so weights stay interpretable across scales.
"""

from typing import Dict, Iterable, Optional, Set


DEFAULT_MAXIMIZE: Set[str] = frozenset({
    "throughput_tokens_per_sec",
    "slo_margin",
})


def compute_tchebycheff(
    y_hat: Dict[str, float],
    w_t: Dict[str, float],
    z_star_t: Dict[str, float],
    normalization_range: Dict[str, float],
    rho: float = 1e-3,
    maximize_objs: Optional[Iterable[str]] = None,
) -> float:
    """
    Definition: Augmented Tchebycheff scalar
                    J = -[ max_j w_j · gap_norm_j  +  ρ · Σ_j w_j · gap_norm_j ]
                Higher J (closer to 0) = closer to ideal under w_t.
                Pareto-complete: sweeping w_t sweeps the entire, potentially non-convex, pareto front.
    Usage:      Exploit term in σ(L'). Called per (config, mechanism, ŷ) candidate via agent.tools.compute_tchebycheff.
    Inputs:
        y_hat               : objective → ŷ_j (from surrogate)
        w_t                 : objective → w_j on the Δ^{m-1} simplex
        z_star_t            : objective → z*_j  (reference point)
        normalization_range : objective → range_j (per-objective scale)
        rho                 : augmentation factor (default 1e-3)
        maximize_objs       : iterable of MAXIMIZED objectives; all others
                              are minimized. Defaults to
                              {throughput_*, slo_margin}.
    Outputs:
        J : float ∈ (-∞, 0]
    """
    maximize_set = set(maximize_objs) if maximize_objs is not None else DEFAULT_MAXIMIZE

    weighted_gaps = []
    for obj, y_j in y_hat.items():
        z = z_star_t[obj]
        r = normalization_range[obj]
        is_max = obj in maximize_set
        g_norm = compute_normalized_gap(y_j, z, r, is_max)
        weighted_gaps.append(compute_weighted_gap(g_norm, w_t[obj]))

    max_term = compute_max_norm(weighted_gaps)
    aug_term = compute_augmentation(weighted_gaps, rho)
    return -(max_term + aug_term)


def compute_normalized_gap(
    y_j: float,
    z_star_j: float,
    range_j: float,
    is_maximized: bool,
) -> float:
    """
    Definition: Sign-aware normalized gap from the reference point.
                    minimize:  gap = (y_j − z*_j) / range_j
                    maximize:  gap = (z*_j − y_j) / range_j
                Convention: positive gap = worse than ideal.
    Inputs:
        y_j          : objective value
        z_star_j     : reference for this objective
        range_j      : per-objective typical scale (must be > 0)
        is_maximized : bool — flips the sign
    Outputs:
        float (positive when worse than z*; slightly negative if
        the candidate exceeds z* (ok because z* has slack δ_j in by SlowLoop.compute_z_star_t).
    """
    diff = (z_star_j - y_j) if is_maximized else (y_j - z_star_j)
    return diff / max(range_j, 1e-9)


def compute_weighted_gap(gap_norm: float, w_j: float) -> float:
    """
    Definition: Weight a normalized gap by w_j.
    Inputs:
        gap_norm : float
        w_j      : objective weight (∈ [0, 1] for w_t on Δ^{m-1})
    Outputs:
        float
    """
    return w_j * gap_norm


def compute_max_norm(weighted_gaps: Iterable[float]) -> float:
    """
    Definition: max_j w_j · gap_norm_j — the worst weighted gap dominates that recovers non-convex pareto regions.
    Inputs:
        weighted_gaps : iterable of floats
    Outputs:
        float
    """
    gaps = list(weighted_gaps)
    return max(gaps) if gaps else 0.0


def compute_augmentation(weighted_gaps: Iterable[float], rho: float) -> float:
    """
    Definition: Steuer-Choo augmentation:
                    aug = ρ · Σ_j weighted_gap_j
                Small linear term that breaks max-norm ties and ensures Pareto-completeness.
    Inputs:
        weighted_gaps : iterable of floats
        rho           : augmentation factor (typically 1e-4..1e-2)
    Outputs:
        float
    """
    return rho * float(sum(weighted_gaps))


def compute_tchebycheff_dro(
    y_hat: Dict[str, float],
    dro_band: Dict[str, Dict[str, float]],
    w_t: Dict[str, float],
    z_star_t: Dict[str, float],
    normalization_range: Dict[str, float],
    rho: float = 1e-3,
    maximize_objs: Optional[Iterable[str]] = None,
) -> float:
    """
    Definition: Robustified Tchebycheff. For each objective, pick the
                worst edge of the Wasserstein-DRO band:
                    minimize obj → band["upper"]   (worst-case high)
                    maximize obj → band["lower"]   (worst-case low)
                Then run compute_tchebycheff on this worst-case y.
    Usage:      Robust scoring for high-stakes candidates
                (e.g., near SLO boundary). Conservative J that is robust
                to drift inside the Wasserstein-ε DRO ball.
    Inputs:
        y_hat               : objective → ŷ_j (point prediction)
        dro_band            : objective → {"upper": float, "lower": float}
                              (also "point" and "median" present but unused here)
        w_t, z_star_t,
        normalization_range,
        rho, maximize_objs  : as in compute_tchebycheff
    Outputs:
        J_DRO : float ∈ (-∞, 0]
    """
    maximize_set = set(maximize_objs) if maximize_objs is not None else DEFAULT_MAXIMIZE

    y_worst: Dict[str, float] = {}
    for obj in y_hat:
        band = dro_band[obj]
        if obj in maximize_set:
            y_worst[obj] = band["lower"]
        else:
            y_worst[obj] = band["upper"]

    return compute_tchebycheff(
        y_worst, w_t, z_star_t, normalization_range, rho, maximize_objs
    )