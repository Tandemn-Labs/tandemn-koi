"""Pure helpers for ranking runtime trigger suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Optional

from koi.costing import evaluate_cost_roofline


@dataclass(frozen=True)
class RuntimeJobState:
    trigger_type: str
    elapsed_hours: float
    time_left_hours: float
    tokens_remaining: int
    aggregate_tps: float
    cost_roofline_usd: Optional[float] = None


@dataclass(frozen=True)
class RuntimeChainState:
    replica_id: str
    gpu_type: str
    tp: int
    pp: int
    smoothed_tps: float
    predicted_tps: float
    cost_per_hour: float
    status: str


@dataclass(frozen=True)
class ScaleUpCandidate:
    gpu_type: str
    tp: int
    pp: int
    predicted_tps: float
    cost_per_hour: float
    source: str


@dataclass(frozen=True)
class RankedSuggestion:
    kind: str
    label: str
    source: str
    gpu_type: Optional[str]
    tp: Optional[int]
    pp: Optional[int]
    replica_id: Optional[str]
    projected_post_action_tps: float
    projected_total_cost_usd: Optional[float]
    meets_slo: bool
    cost_overage_usd: Optional[float]
    priority_tps: float = 0.0


def compute_required_tps(tokens_remaining: int, time_left_hours: float) -> float:
    if tokens_remaining <= 0:
        return 0.0
    if time_left_hours <= 0:
        return inf
    return tokens_remaining / (time_left_hours * 3600.0)


def filter_live_chains(chains: list[RuntimeChainState]) -> list[RuntimeChainState]:
    return [
        chain
        for chain in chains
        if chain.status not in {"failed", "completed", "dead", "killed"}
    ]


def _project_post_action_total_cost(
    current_cost_per_hour: float,
    post_action_cost_per_hour: float,
    elapsed_hours: float,
    tokens_remaining: int,
    post_action_tps: float,
) -> Optional[float]:
    if post_action_cost_per_hour <= 0:
        return None
    if post_action_tps <= 0:
        return inf
    remaining_hours = tokens_remaining / post_action_tps / 3600.0
    spent_so_far = current_cost_per_hour * elapsed_hours
    return spent_so_far + (post_action_cost_per_hour * remaining_hours)


def filter_dominated_actions(
    candidates: list[RankedSuggestion],
) -> list[RankedSuggestion]:
    kept: list[RankedSuggestion] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            if other.meets_slo != candidate.meets_slo:
                continue
            same_or_better_tps = (
                other.projected_post_action_tps >= candidate.projected_post_action_tps
            )
            same_or_lower_cost = (
                other.projected_total_cost_usd is not None
                and candidate.projected_total_cost_usd is not None
                and other.projected_total_cost_usd <= candidate.projected_total_cost_usd
            )
            strictly_better = (
                other.projected_post_action_tps > candidate.projected_post_action_tps
                or (
                    other.projected_total_cost_usd is not None
                    and candidate.projected_total_cost_usd is not None
                    and other.projected_total_cost_usd < candidate.projected_total_cost_usd
                )
            )
            if same_or_better_tps and same_or_lower_cost and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return kept


def rank_falling_behind_suggestions(
    job: RuntimeJobState,
    chains: list[RuntimeChainState],
    candidates: list[ScaleUpCandidate],
) -> list[RankedSuggestion]:
    live_chains = filter_live_chains(chains)
    current_hourly_cost = sum(max(0.0, chain.cost_per_hour) for chain in live_chains)
    required_tps = compute_required_tps(job.tokens_remaining, job.time_left_hours)
    suggestions: list[RankedSuggestion] = []

    for candidate in candidates:
        post_action_tps = job.aggregate_tps + max(0.0, candidate.predicted_tps)
        projected_total_cost = _project_post_action_total_cost(
            current_hourly_cost,
            current_hourly_cost + max(0.0, candidate.cost_per_hour),
            job.elapsed_hours,
            job.tokens_remaining,
            post_action_tps,
        )
        meets_slo = post_action_tps >= required_tps
        _, overage = evaluate_cost_roofline(projected_total_cost, job.cost_roofline_usd)
        suggestions.append(
            RankedSuggestion(
                kind="scale_up",
                label=(
                    f"scale_up {candidate.gpu_type} TP={candidate.tp} PP={candidate.pp} count=1"
                ),
                source=candidate.source,
                gpu_type=candidate.gpu_type,
                tp=candidate.tp,
                pp=candidate.pp,
                replica_id=None,
                projected_post_action_tps=post_action_tps,
                projected_total_cost_usd=projected_total_cost,
                meets_slo=meets_slo,
                cost_overage_usd=overage,
                priority_tps=candidate.predicted_tps,
            )
        )

    suggestions.sort(
        key=lambda suggestion: (
            not suggestion.meets_slo,
            suggestion.projected_total_cost_usd
            if suggestion.projected_total_cost_usd is not None
            else inf,
            suggestion.label,
        )
    )
    return suggestions


def rank_overprovisioned_suggestions(
    job: RuntimeJobState,
    chains: list[RuntimeChainState],
    safety_margin: float = 1.10,
) -> list[RankedSuggestion]:
    live_chains = filter_live_chains(chains)
    current_hourly_cost = sum(max(0.0, chain.cost_per_hour) for chain in live_chains)
    required_tps = compute_required_tps(job.tokens_remaining, job.time_left_hours)
    safe_floor = safety_margin * required_tps
    suggestions: list[RankedSuggestion] = []

    for chain in live_chains:
        post_action_tps = max(0.0, job.aggregate_tps - max(0.0, chain.smoothed_tps))
        projected_total_cost = _project_post_action_total_cost(
            current_hourly_cost,
            max(0.0, current_hourly_cost - max(0.0, chain.cost_per_hour)),
            job.elapsed_hours,
            job.tokens_remaining,
            post_action_tps,
        )
        meets_slo = post_action_tps >= safe_floor
        _, overage = evaluate_cost_roofline(projected_total_cost, job.cost_roofline_usd)
        suggestions.append(
            RankedSuggestion(
                kind="kill_replica",
                label=f"kill_replica {chain.replica_id}",
                source="running_chain",
                gpu_type=chain.gpu_type,
                tp=chain.tp,
                pp=chain.pp,
                replica_id=chain.replica_id,
                projected_post_action_tps=post_action_tps,
                projected_total_cost_usd=projected_total_cost,
                meets_slo=meets_slo,
                cost_overage_usd=overage,
                priority_tps=chain.smoothed_tps,
            )
        )

    suggestions = [suggestion for suggestion in suggestions if suggestion.meets_slo]
    suggestions.sort(
        key=lambda suggestion: (
            suggestion.priority_tps,
            suggestion.projected_total_cost_usd
            if suggestion.projected_total_cost_usd is not None
            else inf,
            suggestion.replica_id or "",
        )
    )
    return suggestions
