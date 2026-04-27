"""P1 launch recovery harness.

Phase 3 keeps the Orca-owned launch flow intact: Koi returns a recovery plan
from /job/launch-failed, and Orca decides whether to retry that plan.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from koi.event_tap import emit_event
from koi.harness.decision_utils import (
    alternative_payloads,
    placement_config_from_payload,
    source_to_prediction_source,
)
from koi.harness.failures import (
    config_key,
    decision_chain as load_decision_chain,
    failed_entries as build_failed_entries,
    matches_failed_same_scope,
    retry_budget_limit,
    retry_budget_used,
    same_topology_different_market,
)
from koi.harness.feasibility import physics_for_row
from koi.harness.ids import action_id as make_action_id
from koi.harness.packet_tools import build_packet_read_tools
from koi.harness.reasoner import HarnessReasoner
from koi.harness.resources import resource_map_for
from koi.harness.schemas import (
    ActionOption,
    ChosenAction,
    HarnessState,
    TransitionPacket,
    TransitionType,
    ValidatedAction,
)
from koi.harness.validator import NoValidActionError, validate_choice
from koi.logging_config import get_logger
from koi.schemas import JobRequest, ResourceMap
from koi.tools.memory import AgenticMemory

logger = get_logger("koi.harness.p1")

P1_TIMEOUT = 120.0
P1_MAX_ITERATIONS = 3
MAX_MENU_OPTIONS = 8

_KNOWN_SECTIONS = (
    "physics",
    "perfdb_exact",
    "memory_success",
    "memory_failure",
    "quota",
    "recent_failures",
    "failure",
    "executor_payload",
    "row",
)


def _reconstruct_job_request(
    *,
    decision: dict[str, Any],
    job_id: str,
    force_on_demand: bool,
) -> JobRequest:
    market = decision.get("market")
    if force_on_demand:
        market = "on_demand"
    if market not in {"spot", "on_demand"}:
        market = None
    return JobRequest(
        job_id=job_id,
        model_name=str(decision.get("model_name") or "unknown"),
        avg_input_tokens=max(1, int(decision.get("avg_input_tokens") or 1)),
        avg_output_tokens=max(1, int(decision.get("avg_output_tokens") or 1)),
        num_requests=decision.get("num_requests"),
        slo_deadline_hours=decision.get("slo_deadline_hours") or None,
        objective=decision.get("objective") or "cheapest",
        cost_roofline_usd=decision.get("cost_roofline_usd"),
        preferred_market=market,
        quantization=decision.get("quantization"),
    )


def _source_for_row(row: dict[str, Any], failed: list[dict[str, Any]]) -> str:
    if same_topology_different_market(row, failed):
        return "market_alternate"
    failed_gpus = {item.get("gpu_type") for item in failed if item.get("gpu_type")}
    if row.get("gpu_type") and row.get("gpu_type") not in failed_gpus:
        return "gpu_family_alternate"
    return "topology_or_instance_alternate"


def _section_keys_for(action_id: str) -> list[str]:
    return [
        f"physics:{action_id}",
        f"perfdb_exact:{action_id}",
        f"memory_success:{action_id}",
        f"memory_failure:{action_id}",
        f"quota:{action_id}",
        f"recent_failures:{action_id}",
        f"failure:{action_id}",
        f"executor_payload:{action_id}",
        f"row:{action_id}",
    ]


def _detail_sections_for(
    *,
    agent: Any,
    memory: AgenticMemory,
    req: JobRequest,
    rm: Optional[ResourceMap],
    row: dict[str, Any],
    failed_entries: list[dict[str, Any]],
    action_id: str,
) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    gpu_type = str(row.get("gpu_type") or "unknown")
    market = row.get("planned_market") or row.get("market") or req.preferred_market or "on_demand"
    resource = rm.get_resource(gpu_type) if rm is not None else None
    region = row.get("region") or (resource.region if resource else None)
    sections[f"physics:{action_id}"] = {
        "gpu_type": gpu_type,
        "tp": int(row.get("tp") or 1),
        "pp": int(row.get("pp") or 1),
        "dp": int(row.get("dp") or 1),
        "physics": row.get("physics", {}),
        "hard_feasibility": row.get("hard_feasibility", {}),
    }

    perfdb_exact: list[dict[str, Any]] = []
    perfdb = getattr(agent, "perfdb", None)
    if perfdb is not None:
        try:
            perfdb_exact = perfdb.query(
                model_name=req.model_name,
                gpu_type=gpu_type,
                tp=int(row.get("tp") or 1),
                pp=int(row.get("pp") or 1),
                limit=10,
            ) or []
        except Exception as exc:
            perfdb_exact = [{"error": str(exc)}]
    sections[f"perfdb_exact:{action_id}"] = perfdb_exact

    try:
        memory_success = memory.query_outcomes(model_name=req.model_name, status="succeeded", limit=10) or []
    except Exception as exc:
        memory_success = [{"error": str(exc)}]
    try:
        memory_failure = memory.query_outcomes(model_name=req.model_name, status="failed", limit=10) or []
    except Exception as exc:
        memory_failure = [{"error": str(exc)}]
    sections[f"memory_success:{action_id}"] = memory_success
    sections[f"memory_failure:{action_id}"] = memory_failure

    quota_section: dict[str, Any] = {
        "gpu_type": gpu_type,
        "region": region,
        "market": market,
    }
    if resource is not None:
        quota_section.update(
            {
                "available_gpus": resource.available_gpus,
                "total_gpus": resource.total_gpus,
                "allocated_gpus": resource.allocated_gpus,
                "instance_type": resource.instance_type,
                "interconnect": resource.interconnect,
            }
        )
    try:
        quota_section["failure_summary"] = memory.get_failure_summary(
            gpu_type,
            region=region,
            market=market,
        )
    except Exception as exc:
        quota_section["failure_summary_error"] = str(exc)
    sections[f"quota:{action_id}"] = quota_section
    sections[f"recent_failures:{action_id}"] = quota_section.get("failure_summary", {})

    related_failures = [
        item for item in failed_entries if item.get("gpu_type") == gpu_type
    ]
    sections[f"failure:{action_id}"] = {
        "related_failed_attempts": related_failures,
        "all_failed_attempts": failed_entries,
    }
    sections[f"executor_payload:{action_id}"] = {
        "tool": "return_launch_recovery_plan",
        "gpu_type": gpu_type,
        "instance_type": row.get("instance_type"),
        "tp": int(row.get("tp") or 1),
        "pp": int(row.get("pp") or 1),
        "dp": int(row.get("dp") or 1),
        "market": market,
        "region": region or (rm.region if rm is not None else "unknown"),
        "predicted_tps": row.get("predicted_tps"),
        "predicted_cost_per_hour": row.get("cost_per_hour"),
        "predicted_total_cost": row.get("total_cost"),
        "predicted_runtime_hours": row.get("eta_h"),
        "source": row.get("source"),
        "prediction_source": row.get("prediction_source"),
    }
    sections[f"row:{action_id}"] = {"row": row}
    return sections


def _candidate_rows_from_cost_table(
    *,
    agent: Any,
    req: JobRequest,
    rm: ResourceMap,
    failed_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not hasattr(agent, "_build_cost_table"):
        return []
    try:
        _, rows = agent._build_cost_table(req, rm)
    except Exception as exc:
        logger.warning("p1_cost_table_failed", job_id=req.job_id, error=str(exc))
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in rows:
        row = dict(raw)
        row["planned_market"] = row.get("planned_market") or req.preferred_market or "on_demand"
        if matches_failed_same_scope(row, failed_entries):
            continue
        source = _source_for_row(row, failed_entries)
        row["prediction_source"] = row.get("source", "unknown")
        row["source"] = source
        key = config_key({**row, "market": row["planned_market"]}, include_market=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(row)
        if len(candidates) >= MAX_MENU_OPTIONS - 1:
            break
    return candidates


def _abort_option(rank: int, reason: str) -> ActionOption:
    action_id = make_action_id(rank - 1)
    return ActionOption(
        action_id=action_id,
        action_type="abort_launch",
        summary=f"Abort launch recovery: {reason}",
        rank=rank,
        valid=True,
        evidence={"source": "policy_guard", "reason": reason},
        executor_payload_ref=f"executor_payload:{action_id}",
        detail_refs=[f"failure:{action_id}", f"executor_payload:{action_id}"],
    )


async def build_p1_packet(
    agent: Any,
    req: Any,
    memory: AgenticMemory,
    *,
    ledger: Any = None,
    resource_map: Optional[ResourceMap] = None,
    retry_budget: Optional[int] = None,
) -> TransitionPacket:
    decision_id = getattr(req, "decision_id", None)
    decision_chain = load_decision_chain(memory, decision_id)
    decision = decision_chain[0] if decision_chain else None
    budget_limit = retry_budget_limit() if retry_budget is None else retry_budget
    budget_used = retry_budget_used(decision_chain)
    budget_remaining = max(0, budget_limit - budget_used)
    failed_entries = build_failed_entries(req, decision)
    failure_categories = sorted({item["failure_category"] for item in failed_entries})
    force_on_demand = any(
        item.get("market") == "spot"
        and item.get("failure_category") in {"spot_preemption", "no_capacity", "quota"}
        for item in failed_entries
    )

    detail_sections: dict[str, Any] = {}
    options: list[ActionOption] = []
    job_context: dict[str, Any] = {"job_id": getattr(req, "job_id", "unknown")}
    rm = await resource_map_for(agent, ledger=ledger, resource_map=resource_map)
    reconstructed: Optional[JobRequest] = None

    if decision is not None:
        reconstructed = _reconstruct_job_request(
            decision=decision,
            job_id=getattr(req, "job_id", decision.get("job_id", "unknown")),
            force_on_demand=force_on_demand,
        )
        job_context = {
            "job_id": reconstructed.job_id,
            "model_name": reconstructed.model_name,
            "objective": reconstructed.objective.value,
            "avg_input_tokens": reconstructed.avg_input_tokens,
            "avg_output_tokens": reconstructed.avg_output_tokens,
            "num_requests": reconstructed.num_requests,
            "total_tokens": reconstructed.total_tokens,
            "slo_deadline_hours": reconstructed.slo_deadline_hours,
            "required_tps": reconstructed.required_tps,
            "preferred_market": reconstructed.preferred_market,
            "cost_roofline_usd": reconstructed.cost_roofline_usd,
            "quantization": reconstructed.quantization,
            "parent_decision_id": decision_id,
        }

    if decision is not None and reconstructed is not None and rm is not None and budget_remaining > 0:
        rows = _candidate_rows_from_cost_table(
            agent=agent,
            req=reconstructed,
            rm=rm,
            failed_entries=failed_entries,
        )
        for idx, row in enumerate(rows[: MAX_MENU_OPTIONS - 1]):
            action_id = make_action_id(idx)
            physics_payload = physics_for_row(reconstructed, rm, row)
            hard = physics_payload["hard_feasibility"]
            physics = physics_payload["physics"]
            valid = bool(row.get("meets_slo", True)) and all(
                hard.get(key, True)
                for key in ("capacity_ok", "runtime_supported", "vram_fit", "tp_heads_valid", "pp_layers_valid")
            )
            row["hard_feasibility"] = hard
            row["physics"] = physics
            sections = _detail_sections_for(
                agent=agent,
                memory=memory,
                req=reconstructed,
                rm=rm,
                row=row,
                failed_entries=failed_entries,
                action_id=action_id,
            )
            detail_sections.update(sections)
            source = row.get("source", "cost_table")
            summary = (
                f"Recover launch with {row.get('gpu_type')} TP={row.get('tp')} PP={row.get('pp')} "
                f"DP={row.get('dp', 1)} {row.get('planned_market', 'on_demand')} | "
                f"TPS={float(row.get('predicted_tps') or 0.0):.0f} | "
                f"total=${float(row.get('total_cost') or 0.0):.2f} | source={source}"
            )
            options.append(
                ActionOption(
                    action_id=action_id,
                    action_type="retry_launch",
                    summary=summary,
                    rank=idx + 1,
                    valid=valid,
                    hard_feasibility=hard,
                    performance={
                        "predicted_tps": float(row.get("predicted_tps") or 0.0),
                        "required_tps": reconstructed.required_tps,
                        "meets_slo": bool(row.get("meets_slo", True)),
                        "prediction_source": source,
                    },
                    physics=physics,
                    evidence={
                        "source": source,
                        "failure_categories": failure_categories,
                    },
                    availability=sections[f"quota:{action_id}"].get("failure_summary", {}),
                    cost={
                        "cost_per_hour": row.get("cost_per_hour"),
                        "projected_total_cost_usd": row.get("total_cost"),
                        "under_roofline": row.get("under_cost_roofline"),
                        "cost_overage_usd": row.get("cost_overage_usd"),
                    },
                    risk={
                        "fresh_failures_same_gpu": len(sections[f"failure:{action_id}"]["related_failed_attempts"]),
                    },
                    executor_payload_ref=f"executor_payload:{action_id}",
                    detail_refs=_section_keys_for(action_id),
                )
            )

    if decision is None:
        abort_reason = "missing original decision; cannot safely reconstruct launch request"
    elif budget_remaining <= 0:
        abort_reason = "retry budget exhausted"
    elif not any(o.valid and o.action_type == "retry_launch" for o in options):
        abort_reason = "no safe recovery candidate"
    else:
        abort_reason = "operator chooses not to retry"

    abort_rank = len(options) + 1
    abort = _abort_option(abort_rank, abort_reason)
    detail_sections[f"failure:{abort.action_id}"] = {
        "failed_attempts": failed_entries,
        "failure_categories": failure_categories,
        "retry_budget_remaining": budget_remaining,
    }
    detail_sections[f"executor_payload:{abort.action_id}"] = {
        "tool": "return_abort_launch_recovery",
        "reason": abort_reason,
    }
    options.append(abort)

    return TransitionPacket(
        packet_id=f"p1-{getattr(req, 'job_id', 'unknown')}",
        job_id=getattr(req, "job_id", "unknown"),
        state=HarnessState.LAUNCH_FAILED,
        transition_type=TransitionType.LAUNCH_RECOVERY,
        job_context=job_context,
        failure_context={
            "configs_tried": failed_entries,
            "failure_categories": failure_categories,
            "total_time_seconds": getattr(req, "total_time_seconds", 0.0),
        },
        policy_context={
            "retry_budget_limit": budget_limit,
            "retry_budget_used": budget_used,
            "retry_budget_remaining_before_choice": budget_remaining,
        },
        evidence_summary={
            "candidate_count": len(options),
            "valid_recovery_count": sum(1 for option in options if option.valid and option.action_type == "retry_launch"),
            "abort_available": True,
            "resource_map_available": rm is not None,
        },
        action_options=options,
        detail_sections=detail_sections,
        guards={
            "max_menu_options": MAX_MENU_OPTIONS,
            "retry_budget_enforced": True,
            "no_direct_launch_tool": True,
        },
    )


def render_p1_prompt(packet: TransitionPacket) -> str:
    lines = [
        "P1 LAUNCH RECOVERY",
        "Choose one valid action_id from the recovery menu.",
        "Prefer a safe retry/switch when retry budget remains and the candidate avoids the failed scope.",
        "Choose abort only when no safe candidate exists or retry budget is exhausted.",
        "Do not invent executable actions; Koi will only return the validated recovery plan to Orca.",
        "",
        "JOB CONTEXT:",
        json.dumps(packet.job_context, indent=2, sort_keys=True),
        "",
        "FAILURE CONTEXT:",
        json.dumps(packet.failure_context, indent=2, sort_keys=True),
        "",
        "POLICY CONTEXT:",
        json.dumps(packet.policy_context, indent=2, sort_keys=True),
        "",
        "ACTION MENU:",
    ]
    for option in packet.action_options:
        lines.append(f"{option.rank}. action_id={option.action_id} type={option.action_type} valid={option.valid}")
        lines.append(f"   {option.summary}")
        if option.hard_feasibility:
            lines.append(f"   feasibility={json.dumps(option.hard_feasibility, sort_keys=True)}")
        if option.cost:
            lines.append(f"   cost={json.dumps(option.cost, sort_keys=True)}")
        if option.availability:
            lines.append(f"   availability={json.dumps(option.availability, sort_keys=True)}")
        if option.risk:
            lines.append(f"   risk={json.dumps(option.risk, sort_keys=True)}")
    lines.extend(["", "Return your final answer as the typed ChosenAction schema."])
    return "\n".join(lines)


def _packet_tools(memory: AgenticMemory, packet: TransitionPacket) -> dict[str, Any]:
    tools = build_packet_read_tools(packet, known_sections=_KNOWN_SECTIONS)

    async def get_failure_summary(gpu_type: str, region: Optional[str] = None, market: Optional[str] = None) -> str:
        try:
            return json.dumps(
                memory.get_failure_summary(gpu_type, region=region, market=market),
                indent=2,
                default=str,
            )
        except Exception as exc:
            return f"get_failure_summary failed: {exc}"

    tools["get_failure_summary"] = get_failure_summary
    return tools


def _abort_plan(packet: TransitionPacket, validated: Optional[ValidatedAction] = None, reason: Optional[str] = None) -> dict[str, Any]:
    option = validated.option if validated is not None else next(
        (candidate for candidate in packet.valid_actions() if candidate.action_type == "abort_launch"),
        None,
    )
    rationale = reason or (validated.choice.rationale if validated is not None else None) or (option.summary if option else "launch recovery aborted")
    return {
        "action": "abort",
        "decision_id": None,
        "parent_decision_id": packet.job_context.get("parent_decision_id"),
        "reasoning": rationale,
        "confidence": validated.choice.confidence if validated is not None else 1.0,
        "retry_budget_remaining": packet.policy_context.get("retry_budget_remaining_before_choice", 0),
    }


def _recovery_plan_from_action(
    *,
    packet: TransitionPacket,
    memory: AgenticMemory,
    ledger: Any,
    validated: ValidatedAction,
) -> dict[str, Any]:
    option = validated.option
    if option.action_type == "abort_launch":
        return _abort_plan(packet, validated)

    payload = packet.detail_sections.get(option.executor_payload_ref or "", {})
    config = placement_config_from_payload(
        payload,
        fallback_region=packet.job_context.get("region", "unknown"),
    )
    parent_decision_id = packet.job_context.get("parent_decision_id")
    predicted_tps = float(payload.get("predicted_tps") or 0.0)
    predicted_cost = float(payload.get("predicted_cost_per_hour") or 0.0)
    decision_id = memory.record_decision(
        job_id=packet.job_id,
        model_name=str(packet.job_context.get("model_name") or "unknown"),
        instance_type=config.instance_type,
        gpu_type=config.gpu_type,
        tp=config.tp,
        pp=config.pp,
        dp=config.dp,
        num_gpus=config.num_gpus,
        predicted_tps=predicted_tps,
        predicted_cost_per_hour=predicted_cost,
        predicted_total_cost=payload.get("predicted_total_cost"),
        predicted_runtime_hours=payload.get("predicted_runtime_hours"),
        prediction_confidence=validated.choice.confidence,
        prediction_source=source_to_prediction_source(str(payload.get("prediction_source") or payload.get("source") or "")),
        slo_deadline_hours=float(packet.job_context.get("slo_deadline_hours") or 0.0),
        objective=str(packet.job_context.get("objective") or "cheapest"),
        avg_input_tokens=int(packet.job_context.get("avg_input_tokens") or 0),
        avg_output_tokens=int(packet.job_context.get("avg_output_tokens") or 0),
        num_requests=packet.job_context.get("num_requests"),
        quantization=packet.job_context.get("quantization"),
        triggered_by="launch_recovery",
        parent_decision_id=parent_decision_id,
        cost_roofline_usd=packet.job_context.get("cost_roofline_usd"),
        market=config.market,
    )
    if ledger is not None:
        ledger.reserve(
            decision_id=decision_id,
            gpu_type=config.gpu_type,
            num_gpus=config.num_gpus,
            region=config.region,
            instance_type=config.instance_type,
        )
    remaining_before = int(packet.policy_context.get("retry_budget_remaining_before_choice") or 0)
    rationale = validated.choice.rationale or option.summary
    if validated.fallback_used:
        rationale = f"[HARNESS FALLBACK] {rationale}"
    return {
        "action": "retry_launch",
        "decision_id": decision_id,
        "parent_decision_id": parent_decision_id,
        "action_id": option.action_id,
        "config": config.model_dump(mode="json"),
        "alternatives": alternative_payloads(
            packet,
            option.action_id,
            action_type="retry_launch",
        ),
        "reasoning": rationale,
        "confidence": validated.choice.confidence,
        "retry_budget_remaining": max(0, remaining_before - 1),
    }


async def run_launch_recovery(
    agent: Any,
    req: Any,
    memory: AgenticMemory,
    *,
    ledger: Any = None,
    resource_map: Optional[ResourceMap] = None,
    retry_budget: Optional[int] = None,
) -> dict[str, Any]:
    t0 = time.time()
    packet = await build_p1_packet(
        agent,
        req,
        memory,
        ledger=ledger,
        resource_map=resource_map,
        retry_budget=retry_budget,
    )
    retry_actions = [
        option for option in packet.valid_actions() if option.action_type == "retry_launch"
    ]
    if not retry_actions:
        return _abort_plan(packet, reason="no valid launch recovery action")

    prompt = render_p1_prompt(packet)
    reasoner = HarnessReasoner(
        model=agent._model,
        tools=_packet_tools(memory, packet),
    )
    try:
        tool_calls, choice = await reasoner.choose(
            prompt,
            job_id=packet.job_id,
            label="p1",
            max_iterations=P1_MAX_ITERATIONS,
            timeout=P1_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("p1_timeout", job_id=packet.job_id, timeout=P1_TIMEOUT)
        choice = ChosenAction(
            action_id=retry_actions[0].action_id,
            confidence=0.3,
            rationale="P1 timed out; deterministic fallback selected top recovery candidate.",
        )
        tool_calls = 0

    try:
        validated = validate_choice(packet, choice)
    except NoValidActionError:
        return _abort_plan(packet, reason="no valid launch recovery action")

    plan = _recovery_plan_from_action(
        packet=packet,
        memory=memory,
        ledger=ledger,
        validated=validated,
    )
    emit_event(
        "harness.p1.decided",
        job_id=packet.job_id,
        action=plan.get("action"),
        action_id=validated.option.action_id,
        fallback_used=validated.fallback_used,
        tool_calls=tool_calls,
        elapsed_s=round(time.time() - t0, 2),
    )
    return plan
