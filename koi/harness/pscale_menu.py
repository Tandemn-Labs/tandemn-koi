"""Pscale menu builder.

Phase 2.5 split out of `koi/agent.py:_rank_runtime_policy_suggestions`. The
goals from the architecture doc:

* deterministic code generates a diverse, feasible candidate pool;
* the LLM still reasons over the menu and may explore via packet tools or
  request_custom_scale_option, but execution stays bounded;
* exclusions are surfaced so the model can challenge them, not silently dropped.

Generators:
  current_config        — clone the failing tracker config
  running_chain         — clone any live chain config
  redecide_exact        — fresh cost table for the exact model
  redecide_proxy        — physics-vector similar models from PerfDB
  gpu_family_alternate  — best feasible config in each GPU family in the cluster
  tp_pp_alternate       — feasible (tp, pp) tuples on the current GPU
  market_alternate      — on_demand mirror when spot is risky
  kill_replica          — OVER_PROV only

Caps:
  Option C — reserve 1 slot per source that produced any valid candidate, then
  fill remaining slots by global cost_per_mtoken_usd ranking.

This module is read-only against agent state. It does not mutate the cluster.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from koi.costing import evaluate_cost_roofline
from koi.harness.recent_failures import recent_failure_for_scope, recent_failure_penalty
from koi.logging_config import get_logger
from koi.runtime_policy import (
    RuntimeChainState,
    RuntimeJobState,
    ScaleUpCandidate,
    compute_required_tps,
    filter_live_chains,
    rank_falling_behind_suggestions,
    rank_overprovisioned_suggestions,
)
from koi.schemas import MonitoringStatus, MonitoringTrigger

logger = get_logger("koi.harness.pscale_menu")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MenuCandidate:
    """Rich scale candidate carrying feasibility + provenance."""

    kind: str  # "scale_up" | "kill_replica" | "noop"
    source: str  # see generator names above
    gpu_type: str
    tp: int
    pp: int
    dp: int
    market: str
    instance_type: Optional[str]
    predicted_tps: float
    cost_per_hour: float
    prediction_source: str
    prediction_confidence: float
    feasibility: dict[str, Any] = field(default_factory=dict)
    physics: dict[str, Any] = field(default_factory=dict)
    replica_id: Optional[str] = None
    proxy_model: Optional[str] = None
    proxy_distance: Optional[float] = None
    recent_failure: Optional[dict[str, Any]] = None

    def key(self) -> tuple[str, int, int]:
        return (self.gpu_type, self.tp, self.pp)


@dataclass
class ExclusionRecord:
    summary: str
    source: str
    reason: str


@dataclass
class MenuBuildResult:
    suggestions: list  # list[RankedSuggestion]
    candidates_by_action: dict[str, MenuCandidate]
    excluded: list[ExclusionRecord]
    counts_by_source: dict[str, int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _live_chain_states(agent: Any, trigger: MonitoringTrigger) -> list[RuntimeChainState]:
    tracker = trigger.job_tracker
    config = tracker.get("config", {})
    chains: list[RuntimeChainState] = []
    if agent.monitor and tracker.get("group_id"):
        group_chains = agent.monitor.get_group_chains(tracker["group_id"]) or {}
        for rid, chain in group_chains.items():
            chains.append(
                RuntimeChainState(
                    replica_id=rid,
                    gpu_type=getattr(chain.config, "gpu_type", "?"),
                    tp=int(getattr(chain.config, "tp", 1) or 1),
                    pp=int(getattr(chain.config, "pp", 1) or 1),
                    smoothed_tps=float(getattr(chain, "smoothed_tps", 0.0) or 0.0),
                    predicted_tps=float(getattr(chain, "predicted_tps", 0.0) or 0.0),
                    cost_per_hour=float(
                        getattr(chain, "predicted_cost_per_hour", 0.0) or 0.0
                    ),
                    status=getattr(getattr(chain, "status", None), "value", None)
                    or str(getattr(chain, "status", "unknown")),
                )
            )
    if not chains:
        chains.append(
            RuntimeChainState(
                replica_id=trigger.job_id,
                gpu_type=str(config.get("gpu_type", "?")),
                tp=int(config.get("tp", 1) or 1),
                pp=int(config.get("pp", 1) or 1),
                smoothed_tps=float(tracker.get("smoothed_tps", 0.0) or 0.0),
                predicted_tps=float(tracker.get("predicted_tps", 0.0) or 0.0),
                cost_per_hour=float(tracker.get("predicted_cost_per_hour", 0.0) or 0.0),
                status=str(trigger.trigger_type.value),
            )
        )
    return chains


def _job_state(
    trigger: MonitoringTrigger, chain_states: list[RuntimeChainState]
) -> RuntimeJobState:
    tracker = trigger.job_tracker
    slo_hours = float(tracker.get("slo_deadline_hours") or 0.0)
    elapsed = float(tracker.get("elapsed_hours") or 0.0)
    time_left_hours = max(0.0, slo_hours - elapsed)
    tokens_remaining = int(tracker.get("tokens_remaining") or 0)
    cost_roofline = tracker.get("cost_roofline_usd")
    aggregate_tps = sum(chain.smoothed_tps for chain in filter_live_chains(chain_states))
    return RuntimeJobState(
        trigger_type=trigger.trigger_type.value,
        elapsed_hours=elapsed,
        time_left_hours=time_left_hours,
        tokens_remaining=tokens_remaining,
        aggregate_tps=aggregate_tps,
        cost_roofline_usd=(float(cost_roofline) if cost_roofline is not None else None),
    )


def _summarize_candidate(candidate: MenuCandidate) -> str:
    return (
        f"{candidate.kind} {candidate.gpu_type} TP={candidate.tp} "
        f"PP={candidate.pp} count=1 [{candidate.source}]"
    )


def _resource_map_for(agent: Any):
    """Best-effort live ResourceMap with ledger applied. Returns None on failure."""
    if not getattr(agent, "orca", None):
        return None
    try:
        from koi.tools.resources import parse_orca_resources

        coro = agent.orca.get_resources()
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside async context; caller should pre-fetch.
            return None
        raw = loop.run_until_complete(coro)
        rm = parse_orca_resources(raw)
        if getattr(agent, "ledger", None) is not None:
            rm = agent.ledger.apply_to_resource_map(rm)
        return rm
    except Exception as exc:
        logger.debug("resource_map_fetch_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def gen_current_config(trigger: MonitoringTrigger) -> list[MenuCandidate]:
    tracker = trigger.job_tracker
    config = tracker.get("config", {}) or {}
    predicted_tps = float(tracker.get("predicted_tps") or 0.0)
    cost_per_hour = float(tracker.get("predicted_cost_per_hour") or 0.0)
    if predicted_tps <= 0:
        return []
    return [
        MenuCandidate(
            kind="scale_up",
            source="current_config",
            gpu_type=str(config.get("gpu_type", "?")),
            tp=int(config.get("tp", 1) or 1),
            pp=int(config.get("pp", 1) or 1),
            dp=int(config.get("dp", 1) or 1),
            market=str(config.get("market", "on_demand")),
            instance_type=config.get("instance_type"),
            predicted_tps=predicted_tps,
            cost_per_hour=cost_per_hour,
            prediction_source="memory_verified",
            prediction_confidence=0.9,
        )
    ]


def gen_running_chains(
    chain_states: list[RuntimeChainState], current_key: tuple[str, int, int]
) -> list[MenuCandidate]:
    out: list[MenuCandidate] = []
    seen: set[tuple[str, int, int]] = {current_key}
    for chain in filter_live_chains(chain_states):
        observed_tps = chain.smoothed_tps if chain.smoothed_tps > 0 else chain.predicted_tps
        if observed_tps <= 0:
            continue
        key = (chain.gpu_type, chain.tp, chain.pp)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            MenuCandidate(
                kind="scale_up",
                source=f"running:{chain.replica_id}",
                gpu_type=chain.gpu_type,
                tp=chain.tp,
                pp=chain.pp,
                dp=1,
                market="on_demand",
                instance_type=None,
                predicted_tps=float(observed_tps),
                cost_per_hour=float(chain.cost_per_hour),
                prediction_source="runtime_observed",
                prediction_confidence=0.95,
            )
        )
    return out


def candidates_from_redecide(
    redecide_rows: list[Any],
) -> list[MenuCandidate]:
    """Convert ScaleUpCandidate rows from agent._build_redecide_candidates."""
    out: list[MenuCandidate] = []
    for row in redecide_rows or []:
        out.append(
            MenuCandidate(
                kind="scale_up",
                source="redecide_exact",
                gpu_type=row.gpu_type,
                tp=row.tp,
                pp=row.pp,
                dp=1,
                market="on_demand",
                instance_type=None,
                predicted_tps=float(row.predicted_tps),
                cost_per_hour=float(row.cost_per_hour),
                prediction_source="perfdb_exact",
                prediction_confidence=0.85,
            )
        )
    return out


def gen_tp_pp_alternate(
    agent: Any,
    parent_decision: Optional[dict],
    rm,
    current_key: tuple[str, int, int],
) -> list[MenuCandidate]:
    """Alternate (tp, pp) on the current GPU type."""
    if rm is None or parent_decision is None:
        return []
    current_gpu = current_key[0]
    resource = rm.get_resource(current_gpu) if hasattr(rm, "get_resource") else None
    if resource is None:
        return []

    perfdb = getattr(agent, "perfdb", None)
    model_name = parent_decision.get("model_name")
    if perfdb is None or not model_name:
        return []

    try:
        records = perfdb.query(model_name=str(model_name), gpu_type=current_gpu, limit=20) or []
    except Exception as exc:
        logger.debug("tp_pp_alternate_perfdb_failed", error=str(exc))
        records = []

    out: list[MenuCandidate] = []
    seen: set[tuple[str, int, int]] = {current_key}
    for r in records:
        try:
            tp = int(r.get("tp", 1) or 1)
            pp = int(r.get("pp", 1) or 1)
            tps = float(r.get("throughput_tps", 0) or 0.0)
        except Exception:
            continue
        if tps <= 0:
            continue
        key = (current_gpu, tp, pp)
        if key in seen:
            continue
        seen.add(key)
        num_gpus = tp * pp
        if num_gpus > resource.available_gpus:
            continue
        num_instances = max(1, -(-num_gpus // resource.gpus_per_instance))
        cost_hr = num_instances * resource.cost_per_instance_hour_usd
        out.append(
            MenuCandidate(
                kind="scale_up",
                source="tp_pp_alternate",
                gpu_type=current_gpu,
                tp=tp,
                pp=pp,
                dp=1,
                market="on_demand",
                instance_type=resource.instance_type,
                predicted_tps=tps,
                cost_per_hour=cost_hr,
                prediction_source="perfdb_exact",
                prediction_confidence=0.82,
            )
        )
    return out


def gen_gpu_family_alternate(
    agent: Any,
    parent_decision: Optional[dict],
    rm,
    current_key: tuple[str, int, int],
) -> list[MenuCandidate]:
    """For each GPU family in the cluster, surface the best feasible config."""
    if rm is None or parent_decision is None:
        return []

    perfdb = getattr(agent, "perfdb", None)
    model_name = parent_decision.get("model_name")
    if perfdb is None or not model_name:
        return []

    out: list[MenuCandidate] = []
    current_gpu = current_key[0]
    for resource in getattr(rm, "resources", []) or []:
        gpu_type = resource.gpu_type
        if gpu_type == current_gpu:
            continue
        if resource.available_gpus <= 0:
            continue
        try:
            records = (
                perfdb.query(
                    model_name=str(model_name),
                    gpu_type=gpu_type,
                    sort_by="throughput_tps",
                    limit=10,
                )
                or []
            )
        except Exception as exc:
            logger.debug("gpu_family_perfdb_failed", error=str(exc), gpu=gpu_type)
            continue
        chosen = None
        for r in records:
            try:
                tp = int(r.get("tp", 1) or 1)
                pp = int(r.get("pp", 1) or 1)
                tps = float(r.get("throughput_tps", 0) or 0.0)
            except Exception:
                continue
            if tps <= 0:
                continue
            num_gpus = tp * pp
            if num_gpus > resource.available_gpus:
                continue
            chosen = (tp, pp, tps)
            break
        if chosen is None:
            continue
        tp, pp, tps = chosen
        num_gpus = tp * pp
        num_instances = max(1, -(-num_gpus // resource.gpus_per_instance))
        cost_hr = num_instances * resource.cost_per_instance_hour_usd
        out.append(
            MenuCandidate(
                kind="scale_up",
                source="gpu_family_alternate",
                gpu_type=gpu_type,
                tp=tp,
                pp=pp,
                dp=1,
                market="on_demand",
                instance_type=resource.instance_type,
                predicted_tps=tps,
                cost_per_hour=cost_hr,
                prediction_source="perfdb_exact",
                prediction_confidence=0.85,
            )
        )
    return out


def gen_redecide_proxy(
    agent: Any,
    parent_decision: Optional[dict],
    rm,
    current_key: tuple[str, int, int],
    seen_keys: set[tuple[str, int, int]],
) -> list[MenuCandidate]:
    """Physics-vector proxies: try similar models in PerfDB on cluster GPUs."""
    if rm is None or parent_decision is None:
        return []
    perfdb = getattr(agent, "perfdb", None)
    model_name = parent_decision.get("model_name")
    if perfdb is None or not model_name:
        return []

    try:
        from koi.tools.physics import find_similar_models, get_model_features

        target = get_model_features(
            str(model_name), dtype=parent_decision.get("quantization") or "fp16"
        )
        distinct = (
            perfdb.get_distinct_models() if hasattr(perfdb, "get_distinct_models") else []
        )
        proxies = find_similar_models(target, distinct or [])
    except Exception as exc:
        logger.debug("proxy_lookup_failed", error=str(exc))
        return []

    if not proxies:
        return []

    out: list[MenuCandidate] = []
    for proxy in proxies[:3]:
        proxy_name = proxy.get("model_name")
        confidence = float(proxy.get("confidence", 0.0))
        distance = float(proxy.get("distance", 0.0))
        if distance > 0.30:
            continue
        try:
            records = (
                perfdb.query(
                    model_name=str(proxy_name),
                    sort_by="throughput_tps",
                    limit=10,
                )
                or []
            )
        except Exception:
            continue
        for r in records:
            try:
                gpu_type = str(r.get("gpu_type", ""))
                tp = int(r.get("tp", 1) or 1)
                pp = int(r.get("pp", 1) or 1)
                tps = float(r.get("throughput_tps", 0) or 0.0)
            except Exception:
                continue
            if tps <= 0:
                continue
            resource = rm.get_resource(gpu_type) if hasattr(rm, "get_resource") else None
            if resource is None or resource.available_gpus < tp * pp:
                continue
            key = (gpu_type, tp, pp)
            if key in seen_keys or key == current_key:
                continue
            seen_keys.add(key)
            num_instances = max(1, -(-(tp * pp) // resource.gpus_per_instance))
            cost_hr = num_instances * resource.cost_per_instance_hour_usd
            out.append(
                MenuCandidate(
                    kind="scale_up",
                    source="redecide_proxy",
                    gpu_type=gpu_type,
                    tp=tp,
                    pp=pp,
                    dp=1,
                    market="on_demand",
                    instance_type=resource.instance_type,
                    predicted_tps=tps,
                    cost_per_hour=cost_hr,
                    prediction_source="physics_proxy",
                    prediction_confidence=max(0.4, min(0.85, confidence)),
                    proxy_model=str(proxy_name),
                    proxy_distance=distance,
                )
            )
            break  # one row per proxy is enough
    return out


def gen_market_alternate(
    candidates: list[MenuCandidate],
    agent: Any,
    parent_decision: Optional[dict],
) -> list[MenuCandidate]:
    """Mirror spot candidates as on_demand if there's a recent spot risk signal."""
    memory = getattr(agent, "memory", None)
    if memory is None or parent_decision is None:
        return []

    out: list[MenuCandidate] = []
    seen_pairs: set[tuple[str, int, int, str]] = {
        (c.gpu_type, c.tp, c.pp, c.market) for c in candidates
    }
    for cand in candidates:
        if cand.market != "spot":
            continue
        try:
            summary = memory.get_failure_summary(cand.gpu_type, market="spot")
            preempts = int(summary.get("spot_preemptions_6h", 0) or 0)
        except Exception:
            preempts = 0
        if preempts <= 0:
            continue
        mirror_key = (cand.gpu_type, cand.tp, cand.pp, "on_demand")
        if mirror_key in seen_pairs:
            continue
        seen_pairs.add(mirror_key)
        out.append(
            MenuCandidate(
                kind=cand.kind,
                source="market_alternate",
                gpu_type=cand.gpu_type,
                tp=cand.tp,
                pp=cand.pp,
                dp=cand.dp,
                market="on_demand",
                instance_type=cand.instance_type,
                predicted_tps=cand.predicted_tps,
                cost_per_hour=cand.cost_per_hour,
                prediction_source=cand.prediction_source,
                prediction_confidence=cand.prediction_confidence,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Feasibility prune + dominance
# ---------------------------------------------------------------------------


def _annotate_feasibility(
    candidate: MenuCandidate,
    parent_decision: Optional[dict],
    rm,
    memory: Any = None,
) -> tuple[MenuCandidate, Optional[str]]:
    """Compute hard feasibility flags. Returns (candidate, reject_reason or None)."""
    feasibility: dict[str, Any] = {}
    physics: dict[str, Any] = {}
    reject_reason: Optional[str] = None

    resource = (
        rm.get_resource(candidate.gpu_type)
        if rm is not None and hasattr(rm, "get_resource")
        else None
    )

    if parent_decision and parent_decision.get("model_name"):
        try:
            from koi.model_features import compute_config_features, get_model_features

            mf = get_model_features(
                str(parent_decision["model_name"]),
                dtype=parent_decision.get("quantization") or "fp16",
            )
            gpus_per_node = resource.gpus_per_instance if resource else 8
            price_per_gpu_hour = (
                resource.cost_per_gpu_hour_usd if resource else 0.0
            )
            feats = compute_config_features(
                mf,
                gpu_type=candidate.gpu_type,
                tp=candidate.tp,
                pp=candidate.pp,
                dp=candidate.dp,
                input_len=int(parent_decision.get("avg_input_tokens", 512) or 512),
                output_len=int(parent_decision.get("avg_output_tokens", 256) or 256),
                gpus_per_node=gpus_per_node,
                price_per_gpu_hour=price_per_gpu_hour,
                gpu_memory_gb_override=resource.gpu_memory_gb if resource else None,
            )
            vram_headroom_gb = float(feats.get("vram_headroom_gb", 0.0) or 0.0)
            feasibility["vram_fit"] = vram_headroom_gb >= 8.0
            feasibility["vram_headroom_gb"] = round(vram_headroom_gb, 2)
            feasibility["tp_heads_valid"] = mf.num_attention_heads % candidate.tp == 0
            feasibility["pp_layers_valid"] = mf.num_layers % candidate.pp == 0
            feasibility["kv_heads_per_tp_shard"] = round(
                float(feats.get("kv_heads_per_tp_shard", 0.0) or 0.0), 3
            )
            feasibility["crosses_node_boundary"] = bool(
                feats.get("crosses_node_boundary", 0)
            )
            physics = {
                "bandwidth_per_param": round(
                    float(feats.get("bandwidth_per_param", 0.0) or 0.0), 3
                ),
                "flops_per_param": round(
                    float(feats.get("flops_per_param", 0.0) or 0.0), 3
                ),
                "roofline_decode_tps": round(
                    float(feats.get("roofline_decode_tps", 0.0) or 0.0), 1
                ),
            }
        except Exception as exc:
            logger.debug("feasibility_compute_failed", error=str(exc))
    else:
        feasibility.setdefault("vram_fit", True)
        feasibility.setdefault("tp_heads_valid", True)
        feasibility.setdefault("pp_layers_valid", True)

    if resource is not None:
        num_gpus = candidate.tp * candidate.pp * candidate.dp
        feasibility["capacity_ok"] = num_gpus <= resource.available_gpus
    else:
        feasibility.setdefault("capacity_ok", True)

    feasibility.setdefault("runtime_supported", True)

    # Determine the first hard-fail reason, if any.
    if feasibility.get("vram_fit") is False:
        reject_reason = "vram_oom"
    elif feasibility.get("tp_heads_valid") is False:
        reject_reason = "tp_heads_invalid"
    elif feasibility.get("pp_layers_valid") is False:
        reject_reason = "pp_layers_invalid"
    elif feasibility.get("capacity_ok") is False:
        reject_reason = "no_live_quota"
    elif feasibility.get("runtime_supported") is False:
        reject_reason = "runtime_unsupported"

    candidate.feasibility = feasibility
    candidate.physics = physics
    region = None
    if resource is not None:
        region = resource.region
    signal = recent_failure_for_scope(
        memory,
        gpu_type=candidate.gpu_type,
        instance_type=candidate.instance_type or (resource.instance_type if resource else None),
        region=region,
        market=candidate.market,
        tp=candidate.tp,
        pp=candidate.pp,
        dp=candidate.dp,
    )
    candidate.recent_failure = signal
    return candidate, reject_reason


def _dominance_filter(
    candidates: list[MenuCandidate], required_tps: float, current_aggregate: float
) -> tuple[list[MenuCandidate], list[ExclusionRecord]]:
    """Drop candidates strictly dominated by another with same/better TPS at same/lower cost.

    Domination only applies among options with the same SLO outcome. We never
    drop an option that uniquely meets SLO when no peer does.
    """

    excluded: list[ExclusionRecord] = []

    def _post_tps(c: MenuCandidate) -> float:
        return current_aggregate + max(0.0, c.predicted_tps)

    def _meets_slo(c: MenuCandidate) -> bool:
        return _post_tps(c) >= required_tps

    kept: list[MenuCandidate] = []
    for c in candidates:
        dominated_by: Optional[MenuCandidate] = None
        for other in candidates:
            if other is c:
                continue
            if _meets_slo(other) != _meets_slo(c):
                continue
            if recent_failure_penalty(other) > recent_failure_penalty(c):
                # A recently failed option should not dominate a safer option;
                # keep the safer option visible even if raw cost/TPS is worse.
                continue
            same_or_better_tps = _post_tps(other) >= _post_tps(c)
            same_or_lower_cost = other.cost_per_hour <= c.cost_per_hour
            strictly_better = (
                _post_tps(other) > _post_tps(c) or other.cost_per_hour < c.cost_per_hour
            )
            if same_or_better_tps and same_or_lower_cost and strictly_better:
                dominated_by = other
                break
        if dominated_by is not None:
            excluded.append(
                ExclusionRecord(
                    summary=_summarize_candidate(c),
                    source=c.source,
                    reason=(
                        f"dominated_by={dominated_by.gpu_type} "
                        f"TP={dominated_by.tp} PP={dominated_by.pp} "
                        f"[{dominated_by.source}]"
                    ),
                )
            )
            continue
        kept.append(c)
    return kept, excluded


# ---------------------------------------------------------------------------
# Source caps
# ---------------------------------------------------------------------------


def _cost_per_mtoken(c: MenuCandidate, current_aggregate: float) -> float:
    post_tps = current_aggregate + max(0.0, c.predicted_tps)
    if post_tps <= 0 or c.cost_per_hour <= 0:
        return float("inf")
    return (c.cost_per_hour / post_tps) * (1_000_000.0 / 3600.0)


def _apply_source_caps(
    candidates: list[MenuCandidate],
    current_aggregate: float,
    max_total: int,
) -> list[MenuCandidate]:
    """Option C: reserve 1 slot per source with valid candidates, fill rest globally."""
    if not candidates or max_total <= 0:
        return []

    by_source: dict[str, list[MenuCandidate]] = {}
    for c in candidates:
        by_source.setdefault(c.source, []).append(c)
    for src in list(by_source.keys()):
        by_source[src].sort(
            key=lambda c: (recent_failure_penalty(c), _cost_per_mtoken(c, current_aggregate))
        )

    reserved: list[MenuCandidate] = []
    for src, items in by_source.items():
        if items:
            reserved.append(items[0])

    reserved_keys = {(c.source, c.gpu_type, c.tp, c.pp) for c in reserved}
    remaining_pool = [
        c for c in candidates if (c.source, c.gpu_type, c.tp, c.pp) not in reserved_keys
    ]
    remaining_pool.sort(
        key=lambda c: (recent_failure_penalty(c), _cost_per_mtoken(c, current_aggregate))
    )

    slots_left = max_total - len(reserved)
    if slots_left <= 0:
        return reserved[:max_total]
    return reserved + remaining_pool[:slots_left]


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


async def build_menu(
    agent: Any,
    trigger: MonitoringTrigger,
    *,
    max_options: int,
    precomputed_redecide: Optional[list[Any]] = None,
) -> MenuBuildResult:
    """Build the deterministic candidate pool for a Pscale trigger.

    Returns ranked `RankedSuggestion`s (compatible with the existing executor),
    a map from action_id-style keys to `MenuCandidate`s, and exclusion records.
    """

    chain_states = _live_chain_states(agent, trigger)
    job = _job_state(trigger, chain_states)
    required_tps = compute_required_tps(job.tokens_remaining, job.time_left_hours)
    aggregate_tps = job.aggregate_tps

    # Pull live ResourceMap for feasibility (already async; handle Orca outage).
    rm = None
    parent_decision = None
    if getattr(agent, "memory", None) and trigger.job_tracker.get("decision_id"):
        try:
            parent_decision = agent.memory.get_decision(
                trigger.job_tracker["decision_id"]
            )
        except Exception:
            parent_decision = None
    if getattr(agent, "orca", None):
        try:
            from koi.tools.resources import parse_orca_resources

            raw = await agent.orca.get_resources()
            rm = parse_orca_resources(raw)
            if getattr(agent, "ledger", None) is not None:
                rm = agent.ledger.apply_to_resource_map(rm)
        except Exception as exc:
            logger.debug(
                "menu_resource_fetch_failed",
                job_id=trigger.job_id,
                error=str(exc),
            )
            rm = None

    if trigger.trigger_type == MonitoringStatus.OVER_PROVISIONED:
        return _build_overprov(
            agent=agent,
            trigger=trigger,
            chain_states=chain_states,
            job=job,
            max_options=max_options,
        )

    # FALLING_BEHIND path
    candidates: list[MenuCandidate] = []
    config = trigger.job_tracker.get("config", {}) or {}
    current_key = (
        str(config.get("gpu_type", "?")),
        int(config.get("tp", 1) or 1),
        int(config.get("pp", 1) or 1),
    )

    candidates.extend(gen_current_config(trigger))
    candidates.extend(gen_running_chains(chain_states, current_key))
    candidates.extend(candidates_from_redecide(precomputed_redecide or []))

    seen_keys = {c.key() for c in candidates}
    seen_keys.add(current_key)

    candidates.extend(
        _filter_new(
            gen_tp_pp_alternate(agent, parent_decision, rm, current_key),
            seen_keys,
        )
    )
    candidates.extend(
        _filter_new(
            gen_gpu_family_alternate(agent, parent_decision, rm, current_key),
            seen_keys,
        )
    )
    candidates.extend(
        gen_redecide_proxy(agent, parent_decision, rm, current_key, seen_keys)
    )
    candidates.extend(gen_market_alternate(candidates, agent, parent_decision))

    # Annotate feasibility, drop hard-fails into excluded.
    excluded: list[ExclusionRecord] = []
    valid: list[MenuCandidate] = []
    for c in candidates:
        annotated, reject = _annotate_feasibility(
            c,
            parent_decision,
            rm,
            getattr(agent, "memory", None),
        )
        if reject is not None:
            excluded.append(
                ExclusionRecord(
                    summary=_summarize_candidate(annotated),
                    source=annotated.source,
                    reason=reject,
                )
            )
            continue
        valid.append(annotated)

    valid, dom_excluded = _dominance_filter(valid, required_tps, aggregate_tps)
    excluded.extend(dom_excluded)

    if not valid:
        return MenuBuildResult(
            suggestions=[],
            candidates_by_action={},
            excluded=excluded,
            counts_by_source={},
        )

    valid = _apply_source_caps(valid, aggregate_tps, max_options)

    counts_by_source: dict[str, int] = {}
    for c in valid:
        counts_by_source[c.source] = counts_by_source.get(c.source, 0) + 1

    scale_up = [
        ScaleUpCandidate(
            gpu_type=c.gpu_type,
            tp=c.tp,
            pp=c.pp,
            predicted_tps=c.predicted_tps,
            cost_per_hour=c.cost_per_hour,
            source=c.source,
        )
        for c in valid
    ]
    suggestions = rank_falling_behind_suggestions(job, chain_states, scale_up)

    candidates_by_label = {
        f"{c.kind} {c.gpu_type} TP={c.tp} PP={c.pp} count=1": c for c in valid
    }
    suggestions.sort(
        key=lambda suggestion: (
            not suggestion.meets_slo,
            recent_failure_penalty(candidates_by_label.get(suggestion.label)),
            suggestion.cost_per_mtoken_usd
            if suggestion.cost_per_mtoken_usd is not None
            else float("inf"),
            suggestion.label,
        )
    )
    candidates_by_action: dict[str, MenuCandidate] = {}
    for sugg in suggestions:
        cand = candidates_by_label.get(sugg.label)
        if cand is None:
            continue
        candidates_by_action[sugg.label] = cand

    return MenuBuildResult(
        suggestions=suggestions,
        candidates_by_action=candidates_by_action,
        excluded=excluded,
        counts_by_source=counts_by_source,
    )


def _filter_new(
    new: list[MenuCandidate], seen_keys: set[tuple[str, int, int]]
) -> list[MenuCandidate]:
    out: list[MenuCandidate] = []
    for c in new:
        if c.key() in seen_keys:
            continue
        seen_keys.add(c.key())
        out.append(c)
    return out


def _build_overprov(
    *,
    agent: Any,
    trigger: MonitoringTrigger,
    chain_states: list[RuntimeChainState],
    job: RuntimeJobState,
    max_options: int,
) -> MenuBuildResult:
    suggestions = rank_overprovisioned_suggestions(job, chain_states)[:max_options]
    candidates_by_action: dict[str, MenuCandidate] = {}
    counts_by_source: dict[str, int] = {}
    for sugg in suggestions:
        cand = MenuCandidate(
            kind="kill_replica",
            source="running_chain",
            gpu_type=str(sugg.gpu_type or "?"),
            tp=int(sugg.tp or 1),
            pp=int(sugg.pp or 1),
            dp=1,
            market="on_demand",
            instance_type=None,
            predicted_tps=0.0,
            cost_per_hour=0.0,
            prediction_source="runtime_observed",
            prediction_confidence=1.0,
            replica_id=sugg.replica_id,
        )
        candidates_by_action[sugg.label] = cand
        counts_by_source["running_chain"] = counts_by_source.get("running_chain", 0) + 1
    return MenuBuildResult(
        suggestions=suggestions,
        candidates_by_action=candidates_by_action,
        excluded=[],
        counts_by_source=counts_by_source,
    )
