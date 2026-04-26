"""P0 initial placement harness.

Phase 1 keeps the public /decide contract stable: the harness chooses from a
precomputed menu, then returns a normal AgentDecision.
"""

from __future__ import annotations

import asyncio
import json
import string
import time
from typing import Any, Optional

from koi.costing import evaluate_cost_roofline
from koi.event_tap import emit_event
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
from koi.model_features import compute_config_features, get_model_features
from koi.schemas import (
    AgentDecision,
    DataSource,
    EngineConfig,
    JobRequest,
    PlacementConfig,
    ResourceMap,
)

logger = get_logger("koi.harness.p0")

P0_TIMEOUT = 120.0
P0_MAX_ITERATIONS = 3
MAX_MENU_OPTIONS = 8


def _action_id(index: int) -> str:
    if index < len(string.ascii_lowercase):
        return string.ascii_lowercase[index]
    return f"a{index + 1}"


def _source_to_data_source(source: str) -> DataSource:
    if source == "VERIFIED":
        return DataSource.MEMORY
    if source == "PerfDB":
        return DataSource.EXACT_MATCH
    return DataSource.ANALYTICAL


def _estimate_num_instances(row: dict[str, Any], rm: ResourceMap) -> int:
    resource = rm.get_resource(str(row.get("gpu_type", "")))
    num_gpus = int(row.get("tp", 1)) * int(row.get("pp", 1)) * int(row.get("dp", 1))
    if not resource:
        return max(1, -(-num_gpus // 8))
    return max(1, -(-num_gpus // resource.gpus_per_instance))


def _physics_for_row(req: JobRequest, rm: ResourceMap, row: dict[str, Any]) -> dict[str, Any]:
    gpu_type = str(row.get("gpu_type", "unknown"))
    resource = rm.get_resource(gpu_type)
    if resource is None:
        return {
            "hard_feasibility": {
                "capacity_ok": False,
                "runtime_supported": False,
            },
            "physics": {},
        }

    tp = int(row.get("tp", 1) or 1)
    pp = int(row.get("pp", 1) or 1)
    dp = int(row.get("dp", 1) or 1)
    try:
        mf = get_model_features(req.model_name, dtype=req.quantization or "fp16")
        feats = compute_config_features(
            mf,
            gpu_type=gpu_type,
            tp=tp,
            pp=pp,
            dp=dp,
            input_len=req.avg_input_tokens,
            output_len=req.avg_output_tokens,
            gpus_per_node=resource.gpus_per_instance,
            price_per_gpu_hour=resource.cost_per_gpu_hour_usd,
            gpu_memory_gb_override=resource.gpu_memory_gb,
        )
        vram_headroom_gb = float(feats.get("vram_headroom_gb", 0.0) or 0.0)
        hard_feasibility = {
            "vram_fit": vram_headroom_gb >= 8.0,
            "vram_headroom_gb": round(vram_headroom_gb, 2),
            "tp_heads_valid": mf.num_attention_heads % tp == 0,
            "pp_layers_valid": mf.num_layers % pp == 0,
            "kv_heads_per_tp_shard": round(float(feats.get("kv_heads_per_tp_shard", 0.0) or 0.0), 3),
            "crosses_node_boundary": bool(feats.get("crosses_node_boundary", 0)),
            "capacity_ok": int(row.get("tp", 1)) * int(row.get("pp", 1)) * int(row.get("dp", 1)) <= resource.available_gpus,
            "runtime_supported": True,
        }
        physics = {
            "bandwidth_per_param": round(float(feats.get("bandwidth_per_param", 0.0) or 0.0), 3),
            "flops_per_param": round(float(feats.get("flops_per_param", 0.0) or 0.0), 3),
            "roofline_decode_tps": round(float(feats.get("roofline_decode_tps", 0.0) or 0.0), 1),
            "io_ratio": round(float(feats.get("io_ratio", req.prefill_decode_ratio) or 0.0), 3),
            "gqa_ratio": round(float(getattr(mf, "gqa_ratio", 0.0) or 0.0), 3),
        }
        return {
            "hard_feasibility": hard_feasibility,
            "physics": physics,
            "detail": feats,
        }
    except Exception as exc:
        logger.warning(
            "p0_physics_failed",
            job_id=req.job_id,
            gpu_type=gpu_type,
            error=str(exc),
        )
        return {
            "hard_feasibility": {
                "capacity_ok": True,
                "runtime_supported": True,
            },
            "physics": {},
        }


def build_p0_packet(agent: Any, req: JobRequest, rm: ResourceMap) -> TransitionPacket:
    _, rows = agent._build_cost_table(req, rm)
    options: list[ActionOption] = []
    detail_sections: dict[str, Any] = {}

    for idx, row in enumerate(rows[:MAX_MENU_OPTIONS]):
        action_id = _action_id(idx)
        physics_payload = _physics_for_row(req, rm, row)
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
            "fresh_preempt_same_scope": False,
            "cooloff_remaining_min": 0,
        }
        summary = (
            f"Launch {row.get('gpu_type')} TP={row.get('tp')} PP={row.get('pp')} "
            f"DP={row.get('dp')} {row.get('planned_market', 'on_demand')} | "
            f"TPS={performance['predicted_tps']:.0f} | total=${float(row.get('total_cost') or 0.0):.2f} | "
            f"SLO={'yes' if row_meets_slo else 'no'}"
        )
        detail_key = f"row:{action_id}"
        detail_sections[detail_key] = {
            "row": row,
            "physics": physics_payload.get("detail", {}),
        }
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
                evidence={
                    "source": row.get("source", "unknown"),
                    "memory_successes": 1 if row.get("source") == "VERIFIED" else 0,
                    "memory_failures": 0,
                    "proxy_model": None,
                    "proxy_distance": None,
                },
                availability=availability,
                cost=cost,
                risk={},
                executor_payload_ref=detail_key,
                detail_refs=[detail_key],
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
    lines.extend([
        "",
        "Return your final answer as the typed ChosenAction schema.",
    ])
    return "\n".join(lines)


def _packet_tools(packet: TransitionPacket) -> dict[str, Any]:
    async def read_option_detail(action_id: str, section: str = "row") -> str:
        """Read a precomputed detail section for one action_id."""
        option = packet.get_action(action_id)
        if option is None:
            return f"unknown action_id={action_id!r}"
        details = {
            ref: packet.detail_sections.get(ref)
            for ref in option.detail_refs
            if section == "all" or ref.startswith(f"{section}:") or ref.startswith("row:")
        }
        return json.dumps(details, indent=2, default=str)

    async def compare_options(action_ids: list[str], lens: str = "summary") -> str:
        """Compare precomputed option summaries for the requested action IDs."""
        selected = []
        for action_id in action_ids:
            option = packet.get_action(action_id)
            if option is None:
                continue
            selected.append(
                {
                    "action_id": option.action_id,
                    "rank": option.rank,
                    "valid": option.valid,
                    "summary": option.summary,
                    "performance": option.performance,
                    "cost": option.cost,
                    "availability": option.availability,
                    "physics": option.physics if lens == "physics" else {},
                }
            )
        return json.dumps(selected, indent=2, default=str)

    return {
        "read_option_detail": read_option_detail,
        "compare_options": compare_options,
    }


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
    detail = packet.detail_sections.get(option.executor_payload_ref or "", {})
    row = detail.get("row", {})
    gpu_type = row.get("gpu_type") or option.hard_feasibility.get("gpu_type") or "L40S"
    tp = int(row.get("tp", 1) or 1)
    pp = int(row.get("pp", 1) or 1)
    dp = int(row.get("dp", 1) or 1)
    num_gpus = tp * pp * dp
    resource = rm.get_resource(str(gpu_type))
    instance_type = row.get("instance_type") or (resource.instance_type if resource else "unknown")
    num_instances = _estimate_num_instances(row, rm)
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
        data_source=_source_to_data_source(str(row.get("source", ""))),
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
