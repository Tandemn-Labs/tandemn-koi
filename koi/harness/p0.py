"""P0 initial placement harness.

Phase 1 keeps the public /decide contract stable: the harness chooses from a
precomputed menu, then returns a normal AgentDecision.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from koi.costing import evaluate_cost_roofline
from koi.event_tap import emit_event
from koi.harness.decision_utils import source_to_data_source
from koi.harness.feasibility import estimate_num_instances, physics_for_row
from koi.harness.ids import action_id as make_action_id
from koi.harness.packet_tools import build_packet_read_tools
from koi.harness.recent_failures import annotate_and_rank_rows
from koi.harness.reasoner import HarnessReasoner
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
from koi.model_features import get_model_features
from koi.schemas import (
    AgentDecision,
    EngineConfig,
    JobRequest,
    PlacementConfig,
    ResourceMap,
)

logger = get_logger("koi.harness.p0")

P0_TIMEOUT = 120.0
P0_MAX_ITERATIONS = 3
MAX_MENU_OPTIONS = 8


def _section_keys_for(action_id: str) -> list[str]:
    return [
        f"physics:{action_id}",
        f"perfdb_exact:{action_id}",
        f"perfdb_proxy:{action_id}",
        f"memory_success:{action_id}",
        f"memory_failure:{action_id}",
        f"quota:{action_id}",
        f"recent_failures:{action_id}",
        f"executor_payload:{action_id}",
        f"row:{action_id}",
    ]


def _build_p0_detail_sections(
    agent: Any,
    req: JobRequest,
    rm: ResourceMap,
    row: dict[str, Any],
    physics_payload: dict[str, Any],
    action_id: str,
) -> dict[str, Any]:
    """Build granular per-action detail sections per ARCHITECTURE-HARNESS section 8."""

    sections: dict[str, Any] = {}
    gpu_type = row.get("gpu_type")
    tp = int(row.get("tp", 1) or 1)
    pp = int(row.get("pp", 1) or 1)
    dp = int(row.get("dp", 1) or 1)

    # physics: full per-config feature dict + arch summary.
    physics_detail = dict(physics_payload.get("detail", {}))
    try:
        mf = get_model_features(req.model_name, dtype=req.quantization or "fp16")
        physics_detail.setdefault(
            "model_arch",
            {
                "params_billions": getattr(mf, "num_params_billions", None),
                "num_layers": getattr(mf, "num_layers", None),
                "num_attention_heads": getattr(mf, "num_attention_heads", None),
                "num_kv_heads": getattr(mf, "num_kv_heads", None),
                "gqa_ratio": getattr(mf, "gqa_ratio", None),
                "hidden_dim": getattr(mf, "hidden_dim", None),
                "is_moe": getattr(mf, "is_moe", False),
            },
        )
    except Exception as exc:
        physics_detail.setdefault("model_arch_error", str(exc))
    sections[f"physics:{action_id}"] = physics_detail

    # perfdb_exact: best-effort exact rows from PerfDB for this gpu/tp/pp.
    perfdb = getattr(agent, "perfdb", None)
    perfdb_exact: list[dict[str, Any]] = []
    perfdb_proxy: list[dict[str, Any]] = []
    if perfdb is not None and gpu_type:
        try:
            perfdb_exact = perfdb.query(
                model_name=req.model_name,
                gpu_type=str(gpu_type),
                tp=tp,
                pp=pp,
                limit=10,
            ) or []
        except Exception as exc:
            perfdb_exact = [{"error": str(exc)}]

        if not perfdb_exact:
            try:
                from koi.tools.physics import find_similar_models, get_model_features as _gmf

                target = _gmf(req.model_name, dtype=req.quantization or "fp16")
                distinct = perfdb.get_distinct_models() if hasattr(perfdb, "get_distinct_models") else []
                proxies = find_similar_models(target, distinct or [])
                for proxy in proxies[:3]:
                    proxy_name = proxy.get("model_name")
                    proxy_records = perfdb.query(
                        model_name=str(proxy_name),
                        gpu_type=str(gpu_type),
                        tp=tp,
                        pp=pp,
                        limit=5,
                    ) or []
                    perfdb_proxy.append(
                        {
                            "proxy_model": proxy_name,
                            "distance": proxy.get("distance"),
                            "confidence": proxy.get("confidence"),
                            "records": proxy_records,
                        }
                    )
            except Exception as exc:
                perfdb_proxy = [{"error": str(exc)}]
    sections[f"perfdb_exact:{action_id}"] = perfdb_exact
    sections[f"perfdb_proxy:{action_id}"] = perfdb_proxy

    # memory_success / memory_failure: per-model history from AgenticMemory.
    memory = getattr(agent, "memory", None)
    memory_success: list[dict[str, Any]] = []
    memory_failure: list[dict[str, Any]] = []
    if memory is not None:
        try:
            memory_success = memory.query_outcomes(
                model_name=req.model_name, status="succeeded", limit=10
            ) or []
        except Exception as exc:
            memory_success = [{"error": str(exc)}]
        try:
            memory_failure = memory.query_outcomes(
                model_name=req.model_name, status="failed", limit=10
            ) or []
        except Exception as exc:
            memory_failure = [{"error": str(exc)}]
    sections[f"memory_success:{action_id}"] = memory_success
    sections[f"memory_failure:{action_id}"] = memory_failure

    # quota: live snapshot per (gpu_type, market) using availability priors.
    quota_section: dict[str, Any] = {
        "gpu_type": gpu_type,
        "preferred_market": req.preferred_market,
    }
    if memory is not None and gpu_type:
        try:
            quota_section["failure_summary"] = memory.get_failure_summary(
                str(gpu_type), market=req.preferred_market
            )
        except Exception as exc:
            quota_section["failure_summary_error"] = str(exc)
    resource = rm.get_resource(str(gpu_type)) if gpu_type else None
    if resource is not None:
        quota_section.update(
            {
                "available_gpus": resource.available_gpus,
                "total_gpus": resource.total_gpus,
                "allocated_gpus": resource.allocated_gpus,
                "instance_type": resource.instance_type,
                "region": resource.region,
                "interconnect": resource.interconnect,
            }
        )
    sections[f"quota:{action_id}"] = quota_section

    sections[f"recent_failures:{action_id}"] = {
        "failure_summary": quota_section.get("failure_summary", {}),
        "recent_failure": row.get("recent_failure"),
    }

    # executor_payload: how this action will execute deterministically.
    sections[f"executor_payload:{action_id}"] = {
        "gpu_type": gpu_type,
        "instance_type": row.get("instance_type"),
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "planned_market": row.get("planned_market") or req.preferred_market,
        "cost_per_hour": row.get("cost_per_hour"),
        "predicted_tps": row.get("predicted_tps"),
        "total_cost": row.get("total_cost"),
        "eta_h": row.get("eta_h"),
    }

    # row: keep the raw cost-table row for full transparency.
    sections[f"row:{action_id}"] = {"row": row}
    return sections


def build_p0_packet(agent: Any, req: JobRequest, rm: ResourceMap) -> TransitionPacket:
    _, rows = agent._build_cost_table(req, rm)
    rows = annotate_and_rank_rows(
        getattr(agent, "memory", None),
        rows,
        rm,
        default_market=req.preferred_market or "on_demand",
    )
    options: list[ActionOption] = []
    detail_sections: dict[str, Any] = {}

    for idx, row in enumerate(rows[:MAX_MENU_OPTIONS]):
        action_id = make_action_id(idx)
        physics_payload = physics_for_row(req, rm, row)
        hard_feasibility = physics_payload["hard_feasibility"]
        row_meets_slo = bool(row.get("meets_slo"))
        valid = row_meets_slo and all(
            hard_feasibility.get(key, True)
            for key in ("vram_fit", "tp_heads_valid", "pp_layers_valid", "capacity_ok", "runtime_supported")
        )
        performance = {
            "predicted_tps": float(row.get("predicted_tps") or 0.0),
            "required_tps": req.required_tps,
            "meets_slo": row_meets_slo,
            "prediction_source": row.get("source", "unknown"),
            "prediction_confidence": 0.9 if row.get("source") == "VERIFIED" else 0.75,
        }
        cost = {
            "cost_per_hour": float(row.get("cost_per_hour") or 0.0),
            "projected_total_cost_usd": row.get("total_cost"),
            "under_roofline": row.get("under_cost_roofline"),
            "cost_overage_usd": row.get("cost_overage_usd"),
        }
        availability = {
            "live_quota": hard_feasibility.get("capacity_ok"),
            "beta_launch_success_pct": row.get("avail_pct"),
            "availability_uncertainty_pct": row.get("avail_unc"),
            "recent_no_capacity_failures": None,
            "recent_failure": row.get("recent_failure"),
        }
        summary = (
            f"Launch {row.get('gpu_type')} TP={row.get('tp')} PP={row.get('pp')} "
            f"DP={row.get('dp')} {row.get('planned_market', 'on_demand')} | "
            f"TPS={performance['predicted_tps']:.0f} | total=${float(row.get('total_cost') or 0.0):.2f} | "
            f"SLO={'yes' if row_meets_slo else 'no'}"
        )
        per_action_sections = _build_p0_detail_sections(
            agent, req, rm, row, physics_payload, action_id
        )
        detail_sections.update(per_action_sections)
        evidence = {
            "source": row.get("source", "unknown"),
            "memory_successes": len(per_action_sections[f"memory_success:{action_id}"]) if isinstance(per_action_sections[f"memory_success:{action_id}"], list) else 0,
            "memory_failures": len(per_action_sections[f"memory_failure:{action_id}"]) if isinstance(per_action_sections[f"memory_failure:{action_id}"], list) else 0,
            "perfdb_exact_rows": len(per_action_sections[f"perfdb_exact:{action_id}"]) if isinstance(per_action_sections[f"perfdb_exact:{action_id}"], list) else 0,
            "perfdb_proxy_rows": len(per_action_sections[f"perfdb_proxy:{action_id}"]) if isinstance(per_action_sections[f"perfdb_proxy:{action_id}"], list) else 0,
            "proxy_model": (
                per_action_sections[f"perfdb_proxy:{action_id}"][0].get("proxy_model")
                if per_action_sections[f"perfdb_proxy:{action_id}"]
                and isinstance(per_action_sections[f"perfdb_proxy:{action_id}"][0], dict)
                and "proxy_model" in per_action_sections[f"perfdb_proxy:{action_id}"][0]
                else None
            ),
            "proxy_distance": (
                per_action_sections[f"perfdb_proxy:{action_id}"][0].get("distance")
                if per_action_sections[f"perfdb_proxy:{action_id}"]
                and isinstance(per_action_sections[f"perfdb_proxy:{action_id}"][0], dict)
                and "distance" in per_action_sections[f"perfdb_proxy:{action_id}"][0]
                else None
            ),
            "recent_failure": row.get("recent_failure"),
        }
        risk = {}
        if row.get("recent_failure"):
            risk["recent_failure"] = row["recent_failure"]
        options.append(
            ActionOption(
                action_id=action_id,
                action_type="launch",
                summary=summary,
                rank=idx + 1,
                valid=valid,
                hard_feasibility=hard_feasibility,
                performance=performance,
                physics=physics_payload["physics"],
                evidence=evidence,
                availability=availability,
                cost=cost,
                risk=risk,
                executor_payload_ref=f"executor_payload:{action_id}",
                detail_refs=_section_keys_for(action_id),
            )
        )

    return TransitionPacket(
        packet_id=f"p0-{req.job_id}",
        job_id=req.job_id,
        state=HarnessState.REQUESTED,
        transition_type=TransitionType.INITIAL_PLACEMENT,
        job_context={
            "model_name": req.model_name,
            "task_type": req.task_type.value,
            "objective": req.objective.value,
            "avg_input_tokens": req.avg_input_tokens,
            "avg_output_tokens": req.avg_output_tokens,
            "num_requests": req.num_requests,
            "total_tokens": req.total_tokens,
            "slo_deadline_hours": req.slo_deadline_hours,
            "required_tps": req.required_tps,
            "preferred_market": req.preferred_market,
            "cost_roofline_usd": req.cost_roofline_usd,
        },
        evidence_summary={
            "candidate_count": len(options),
            "valid_candidate_count": sum(1 for option in options if option.valid),
            "source": "cost_table",
        },
        action_options=options,
        detail_sections=detail_sections,
        guards={
            "slo_is_hard": True,
            "cost_roofline_is_soft": True,
            "max_menu_options": MAX_MENU_OPTIONS,
        },
    )


def render_p0_prompt(packet: TransitionPacket) -> str:
    lines = [
        "P0 INITIAL PLACEMENT",
        "Choose one valid action_id from the launch menu.",
        "SLO is hard. Cost roofline is a soft preference unless no SLO-valid option exists.",
        "The ranking is guidance, not a command; explain if you choose a lower-ranked option.",
        "",
        "JOB CONTEXT:",
        json.dumps(packet.job_context, indent=2, sort_keys=True),
        "",
        "ACTION MENU:",
    ]
    for option in packet.action_options:
        lines.append(f"{option.rank}. action_id={option.action_id} valid={option.valid}")
        lines.append(f"   {option.summary}")
        lines.append(
            "   "
            f"feasibility={json.dumps(option.hard_feasibility, sort_keys=True)}"
        )
        lines.append(
            "   "
            f"cost={json.dumps(option.cost, sort_keys=True)} availability={json.dumps(option.availability, sort_keys=True)}"
        )
        if option.physics:
            lines.append(f"   physics={json.dumps(option.physics, sort_keys=True)}")
        if option.risk:
            lines.append(f"   risk={json.dumps(option.risk, sort_keys=True)}")
    lines.extend([
        "",
        "Return your final answer as the typed ChosenAction schema.",
    ])
    return "\n".join(lines)


_KNOWN_SECTIONS = (
    "physics",
    "perfdb_exact",
    "perfdb_proxy",
    "memory_success",
    "memory_failure",
    "quota",
    "recent_failures",
    "executor_payload",
    "row",
)


def _packet_tools(packet: TransitionPacket) -> dict[str, Any]:
    return build_packet_read_tools(packet, known_sections=_KNOWN_SECTIONS)


def _decision_from_action(
    packet: TransitionPacket,
    req: JobRequest,
    rm: ResourceMap,
    validated: ValidatedAction,
    *,
    tool_calls: int,
    elapsed: float,
    agent_model: str,
) -> AgentDecision:
    option = validated.option
    row_ref = f"row:{option.action_id}"
    row = packet.detail_sections.get(row_ref, {}).get("row", {})
    if not row:
        # Backward-compat for older bundled rows still keyed by executor_payload_ref.
        legacy = packet.detail_sections.get(option.executor_payload_ref or "", {})
        row = legacy.get("row", legacy if isinstance(legacy, dict) else {})
    gpu_type = row.get("gpu_type") or option.hard_feasibility.get("gpu_type") or "L40S"
    tp = int(row.get("tp", 1) or 1)
    pp = int(row.get("pp", 1) or 1)
    dp = int(row.get("dp", 1) or 1)
    num_gpus = tp * pp * dp
    resource = rm.get_resource(str(gpu_type))
    instance_type = row.get("instance_type") or (resource.instance_type if resource else "unknown")
    num_instances = estimate_num_instances(row, rm)
    planned_market = row.get("planned_market") or req.preferred_market or "on_demand"
    cost_per_hour = float(row.get("cost_per_hour") or option.cost.get("cost_per_hour") or 0.0)
    predicted_tps = float(row.get("predicted_tps") or option.performance.get("predicted_tps") or 0.0)
    total_cost = row.get("total_cost")
    runtime_hours = row.get("eta_h")
    meets_cost_roofline, overage = evaluate_cost_roofline(total_cost, req.cost_roofline_usd)
    cost_warning = None
    if meets_cost_roofline is False:
        cost_warning = (
            "Projected cost exceeds roofline, but this is the selected "
            "SLO-meeting harness option."
        )
    reasoning = validated.choice.rationale or option.summary
    if validated.fallback_used:
        reasoning = f"[HARNESS FALLBACK] {reasoning}"

    return AgentDecision(
        job_id=req.job_id,
        model_name=req.model_name,
        config=PlacementConfig(
            gpu_type=str(gpu_type),
            instance_type=str(instance_type),
            num_gpus=num_gpus,
            num_instances=num_instances,
            tp=tp,
            pp=pp,
            dp=dp,
            region=rm.region,
            engine_config=EngineConfig(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                quantization=req.quantization,
            ),
            market=planned_market,
        ),
        planned_market=planned_market,
        predicted_tps=predicted_tps,
        predicted_cost_per_hour=cost_per_hour,
        predicted_total_cost=total_cost,
        predicted_runtime_hours=runtime_hours,
        meets_cost_roofline=meets_cost_roofline,
        cost_roofline_usd=req.cost_roofline_usd,
        projected_cost_overage_usd=overage,
        cost_warning=cost_warning,
        reasoning=reasoning,
        confidence=validated.choice.confidence,
        data_source=source_to_data_source(str(row.get("source", ""))),
        agent_model=agent_model,
        tool_calls_made=tool_calls,
        latency_seconds=elapsed,
    )


def _populate_alternatives(decision: AgentDecision, rows: list[dict[str, Any]]) -> None:
    primary = (
        decision.config.gpu_type,
        decision.config.tp,
        decision.config.pp,
        decision.config.dp,
    )
    alternatives = []
    for row in rows:
        if not row.get("meets_slo"):
            continue
        candidate = (row["gpu_type"], row["tp"], row["pp"], row.get("dp", 1))
        if candidate == primary:
            continue
        if row.get("dp") != decision.config.dp:
            continue
        alt = dict(row)
        alt["planned_market"] = decision.planned_market
        alternatives.append(alt)
        if len(alternatives) >= 3:
            break
    decision.alternatives = alternatives


async def run_initial_placement(agent: Any, req: JobRequest, rm: ResourceMap) -> AgentDecision:
    t0 = time.time()
    packet = build_p0_packet(agent, req, rm)
    if not packet.valid_actions():
        return agent._fallback_decision(req, rm, time.time() - t0)

    prompt = render_p0_prompt(packet)
    reasoner = HarnessReasoner(
        model=agent._model,
        tools=_packet_tools(packet),
    )
    try:
        tool_calls, choice = await reasoner.choose(
            prompt,
            job_id=req.job_id,
            label="p0",
            max_iterations=P0_MAX_ITERATIONS,
            timeout=P0_TIMEOUT,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        logger.error("p0_timeout", job_id=req.job_id, timeout=P0_TIMEOUT)
        return agent._fallback_decision(req, rm, elapsed)
    except Exception as exc:
        logger.warning("p0_reasoner_failed", job_id=req.job_id, error=str(exc))
        raise

    try:
        validated = validate_choice(packet, choice)
    except NoValidActionError:
        return agent._fallback_decision(req, rm, time.time() - t0)

    elapsed = time.time() - t0
    decision = _decision_from_action(
        packet,
        req,
        rm,
        validated,
        tool_calls=tool_calls,
        elapsed=elapsed,
        agent_model=agent.model,
    )
    _populate_alternatives(decision, getattr(agent, "_last_cost_rows", []))
    emit_event(
        "harness.p0.decided",
        job_id=req.job_id,
        action_id=validated.option.action_id,
        fallback_used=validated.fallback_used,
    )
    return decision
