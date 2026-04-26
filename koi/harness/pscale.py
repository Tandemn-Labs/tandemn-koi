"""Pscale runtime scaling harness for DEGRADED and OVERPROV states."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from koi.event_tap import emit_event
from koi.harness.reasoner import HarnessReasoner
from koi.harness.schemas import (
    ActionOption,
    HarnessState,
    TransitionPacket,
    TransitionType,
)
from koi.harness.validator import NoValidActionError, validate_choice
from koi.logging_config import get_logger
from koi.schemas import MonitoringStatus, MonitoringTrigger

logger = get_logger("koi.harness.pscale")

# Runtime control should fail over quickly. If the model cannot pick from a
# precomputed scale menu within this window, execute the top valid option.
PSCALE_TIMEOUT = 45.0
PSCALE_MAX_ITERATIONS = 3
MAX_MENU_OPTIONS = 8


def _action_id(index: int) -> str:
    return chr(ord("a") + index) if index < 26 else f"a{index + 1}"


def _group_id(trigger: MonitoringTrigger) -> str:
    tracker = trigger.job_tracker
    return tracker.get("group_id") or trigger.job_id


def _state_for_trigger(trigger: MonitoringTrigger) -> HarnessState:
    if trigger.trigger_type == MonitoringStatus.FALLING_BEHIND:
        return HarnessState.DEGRADED
    if trigger.trigger_type == MonitoringStatus.OVER_PROVISIONED:
        return HarnessState.OVERPROV
    return HarnessState.AT_RISK


def _runtime_context(agent: Any, trigger: MonitoringTrigger) -> dict[str, Any]:
    tracker = trigger.job_tracker
    group_id = _group_id(trigger)
    group_chains = {}
    if agent.monitor and tracker.get("group_id"):
        group_chains = agent.monitor.get_group_chains(tracker["group_id"])
    live_chains = []
    for chain_id, chain in group_chains.items():
        status = getattr(getattr(chain, "status", None), "value", None) or str(
            getattr(chain, "status", "unknown")
        )
        if status in {"failed", "completed", "dead", "killed"}:
            continue
        live_chains.append(
            {
                "replica_id": chain_id,
                "gpu_type": getattr(chain.config, "gpu_type", "unknown"),
                "tp": getattr(chain.config, "tp", 1),
                "pp": getattr(chain.config, "pp", 1),
                "smoothed_tps": getattr(chain, "smoothed_tps", 0.0),
                "predicted_tps": getattr(chain, "predicted_tps", 0.0),
                "cost_per_hour": getattr(chain, "predicted_cost_per_hour", 0.0),
                "status": status,
            }
        )

    return {
        "job_id": trigger.job_id,
        "group_id": group_id,
        "trigger_type": trigger.trigger_type.value,
        "diagnosis_hint": trigger.diagnosis_hint,
        "smoothed_tps": tracker.get("smoothed_tps", 0.0),
        "slo_headroom_pct": tracker.get("slo_headroom_pct", 0.0),
        "elapsed_hours": tracker.get("elapsed_hours", 0.0),
        "tokens_remaining": tracker.get("tokens_remaining", 0),
        "slo_deadline_hours": tracker.get("slo_deadline_hours"),
        "projected_total_cost_usd": tracker.get("projected_total_cost_usd"),
        "cost_roofline_usd": tracker.get("cost_roofline_usd"),
        "cost_overage_usd": tracker.get("cost_overage_usd"),
        "live_chains": live_chains,
        "action_in_progress": tracker.get("action_in_progress", False),
    }


async def build_pscale_packet(
    agent: Any,
    trigger: MonitoringTrigger,
    precomputed_candidates: Optional[list[Any]] = None,
) -> TransitionPacket:
    suggestions = agent._rank_runtime_policy_suggestions(
        trigger,
        precomputed_candidates=precomputed_candidates,
    )
    options: list[ActionOption] = []
    detail_sections: dict[str, Any] = {}

    for idx, suggestion in enumerate(suggestions[: MAX_MENU_OPTIONS - 1]):
        action_id = _action_id(idx)
        if suggestion.kind == "scale_up":
            action_type = "scale_up"
            executor_payload = {
                "tool": "scale_chain_tool",
                "job_id": _group_id(trigger),
                "gpu_type": suggestion.gpu_type,
                "tp": suggestion.tp,
                "pp": suggestion.pp,
                "count": 1,
            }
        elif suggestion.kind == "kill_replica":
            action_type = "kill_replica"
            executor_payload = {
                "tool": "kill_replica_tool",
                "job_id": _group_id(trigger),
                "replica_ids": [suggestion.replica_id],
            }
        else:
            action_type = suggestion.kind
            executor_payload = {"tool": "noop"}

        detail_key = f"suggestion:{action_id}"
        detail_sections[detail_key] = {
            "suggestion": suggestion.__dict__,
            "executor_payload": executor_payload,
        }
        options.append(
            ActionOption(
                action_id=action_id,
                action_type=action_type,
                summary=suggestion.label,
                rank=idx + 1,
                valid=True,
                performance={
                    "gpu_type": suggestion.gpu_type,
                    "tp": suggestion.tp,
                    "pp": suggestion.pp,
                    "replica_id": suggestion.replica_id,
                    "projected_post_action_tps": suggestion.projected_post_action_tps,
                    "meets_slo": suggestion.meets_slo,
                    "cost_per_mtoken_usd": suggestion.cost_per_mtoken_usd,
                },
                evidence={
                    "source": suggestion.source,
                    "gpu_type": suggestion.gpu_type,
                },
                cost={
                    "projected_total_cost_usd": suggestion.projected_total_cost_usd,
                    "cost_overage_usd": suggestion.cost_overage_usd,
                },
                risk={},
                executor_payload_ref=detail_key,
                detail_refs=[detail_key],
            )
        )

    noop_id = _action_id(len(options))
    noop_detail_key = f"suggestion:{noop_id}"
    detail_sections[noop_detail_key] = {"executor_payload": {"tool": "noop"}}
    options.append(
        ActionOption(
            action_id=noop_id,
            action_type="noop",
            summary="No action; wait for the next monitor tick.",
            rank=len(options) + 1,
            valid=len(suggestions) == 0,
            risk={
                "reason": "Use only when no listed scale action is safe or necessary."
            },
            executor_payload_ref=noop_detail_key,
            detail_refs=[noop_detail_key],
        )
    )

    return TransitionPacket(
        packet_id=f"pscale-{trigger.job_id}",
        job_id=trigger.job_id,
        state=_state_for_trigger(trigger),
        transition_type=TransitionType.SCALE,
        runtime_context=_runtime_context(agent, trigger),
        failure_context={"diagnosis_hint": trigger.diagnosis_hint},
        evidence_summary={
            "suggestion_count": len(suggestions),
            "valid_action_count": len(options),
            "source": "runtime_policy",
        },
        action_options=options,
        detail_sections=detail_sections,
        guards={
            "slo_is_hard": True,
            "kill_at_most_one_replica": True,
            "scale_actions_are_frozen_after_execute": True,
        },
    )


def render_pscale_prompt(packet: TransitionPacket) -> str:
    lines = [
        "PSCALE RUNTIME DECISION",
        "Choose one valid action_id from the runtime menu.",
        "SLO is hard. Cost is a soft ranking signal unless all SLO-saving actions are costly.",
        "The ranking is guidance, not a command; explain non-top choices.",
        "",
        "RUNTIME CONTEXT:",
        json.dumps(packet.runtime_context, indent=2, sort_keys=True, default=str),
        "",
        "ACTION MENU:",
    ]
    for option in packet.action_options:
        lines.append(f"{option.rank}. action_id={option.action_id} type={option.action_type} valid={option.valid}")
        lines.append(f"   {option.summary}")
        if option.performance:
            lines.append(f"   performance={json.dumps(option.performance, sort_keys=True, default=str)}")
        if option.cost:
            lines.append(f"   cost={json.dumps(option.cost, sort_keys=True, default=str)}")
        if option.risk:
            lines.append(f"   risk={json.dumps(option.risk, sort_keys=True, default=str)}")
    lines.extend([
        "",
        "Return your final answer as the typed ChosenAction schema.",
    ])
    return "\n".join(lines)


def _packet_tools(packet: TransitionPacket) -> dict[str, Any]:
    async def read_option_detail(action_id: str, section: str = "all") -> str:
        """Read a precomputed detail section for one runtime action option."""
        option = packet.get_action(action_id)
        if option is None:
            return f"unknown action_id={action_id!r}"
        details = {ref: packet.detail_sections.get(ref) for ref in option.detail_refs}
        return json.dumps(details, indent=2, default=str)

    async def compare_options(action_ids: list[str], lens: str = "summary") -> str:
        """Compare precomputed runtime options."""
        selected = []
        for action_id in action_ids:
            option = packet.get_action(action_id)
            if option is None:
                continue
            selected.append(
                {
                    "action_id": option.action_id,
                    "rank": option.rank,
                    "type": option.action_type,
                    "summary": option.summary,
                    "performance": option.performance,
                    "cost": option.cost,
                    "risk": option.risk,
                }
            )
        return json.dumps(selected, indent=2, default=str)

    return {
        "read_option_detail": read_option_detail,
        "compare_options": compare_options,
    }


async def _execute_validated_action(agent: Any, packet: TransitionPacket, action_id: str) -> str:
    option = packet.get_action(action_id)
    if option is None:
        return f"No action executed: unknown action_id={action_id!r}"
    detail = packet.detail_sections.get(option.executor_payload_ref or "", {})
    payload = detail.get("executor_payload", {})
    tool_name = payload.get("tool")
    if tool_name == "noop" or option.action_type == "noop":
        return "No action executed: noop selected."

    tools = agent._build_tools(monitor=agent.monitor)
    if tool_name == "scale_chain_tool":
        tool = tools.get("scale_chain_tool")
        if tool is None:
            return "Scale action unavailable: no Orca action tool configured."
        return await tool(
            job_id=payload["job_id"],
            gpu_type=payload["gpu_type"],
            tp=int(payload["tp"]),
            pp=int(payload["pp"]),
            count=int(payload["count"]),
        )
    if tool_name == "kill_replica_tool":
        tool = tools.get("kill_replica_tool")
        if tool is None:
            return "Kill action unavailable: no Orca action tool configured."
        result = await tool(
            job_id=payload["job_id"],
            replica_ids=list(payload["replica_ids"]),
        )
        _record_scale_down_decision(agent, packet, option)
        return result
    return f"No action executed: unsupported executor tool={tool_name!r}."


def _record_scale_down_decision(
    agent: Any,
    packet: TransitionPacket,
    option: ActionOption,
) -> None:
    if not getattr(agent, "memory", None):
        return
    group_id = packet.runtime_context.get("group_id") or packet.job_id
    parent_decision_id = None
    parent = None
    if getattr(agent, "monitor", None):
        for tracker in agent.monitor.tracked_jobs.values():
            tracker_group_id = getattr(tracker, "group_id", None)
            tracker_decision_id = getattr(tracker, "decision_id", None)
            if tracker_group_id == group_id and tracker_decision_id:
                parent_decision_id = tracker_decision_id
                parent = agent.memory.get_decision(parent_decision_id)
                break
    gpu_type = (
        option.evidence.get("gpu_type")
        or option.performance.get("gpu_type")
        or option.cost.get("gpu_type")
        or "unknown"
    )
    try:
        agent.memory.record_decision(
            job_id=group_id,
            model_name=parent.get("model_name", "unknown") if parent else "unknown",
            instance_type=parent.get("instance_type", "unknown") if parent else "unknown",
            gpu_type=str(gpu_type),
            tp=int(option.performance.get("tp") or 1),
            pp=int(option.performance.get("pp") or 1),
            dp=1,
            num_gpus=int(option.performance.get("tp") or 1)
            * int(option.performance.get("pp") or 1),
            predicted_tps=0,
            predicted_cost_per_hour=0,
            slo_deadline_hours=parent.get("slo_deadline_hours", 0) if parent else 0,
            objective=parent.get("objective", "cheapest") if parent else "cheapest",
            avg_input_tokens=parent.get("avg_input_tokens", 0) if parent else 0,
            avg_output_tokens=parent.get("avg_output_tokens", 0) if parent else 0,
            triggered_by="scale_down",
            parent_decision_id=parent_decision_id,
            cost_roofline_usd=parent.get("cost_roofline_usd") if parent else None,
            market=parent.get("market", "unknown") if parent else "unknown",
        )
    except Exception as exc:
        logger.warning(
            "pscale_scale_down_record_failed",
            job_id=group_id,
            error=str(exc),
        )


async def run_runtime_scale(
    agent: Any,
    trigger: MonitoringTrigger,
    precomputed_candidates: Optional[list[Any]] = None,
) -> str:
    packet = await build_pscale_packet(
        agent,
        trigger,
        precomputed_candidates=precomputed_candidates,
    )
    prompt = render_pscale_prompt(packet)
    reasoner = HarnessReasoner(
        model=agent._model,
        tools=_packet_tools(packet),
    )
    try:
        _, choice = await reasoner.choose(
            prompt,
            job_id=trigger.job_id,
            label="pscale",
            max_iterations=PSCALE_MAX_ITERATIONS,
            timeout=PSCALE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("pscale_timeout", job_id=trigger.job_id, timeout=PSCALE_TIMEOUT)
        choice = None
    except Exception as exc:
        logger.warning("pscale_reasoner_failed", job_id=trigger.job_id, error=str(exc))
        raise

    if choice is None:
        try:
            option = packet.valid_actions()[0]
        except IndexError:
            return "[HARNESS FALLBACK] No valid runtime action available."
        action_id = option.action_id
        fallback_used = True
    else:
        try:
            validated = validate_choice(packet, choice)
        except NoValidActionError:
            return "[HARNESS FALLBACK] No valid runtime action available."
        action_id = validated.choice.action_id
        fallback_used = validated.fallback_used

    result = await _execute_validated_action(agent, packet, action_id)
    emit_event(
        "harness.pscale.executed",
        job_id=trigger.job_id,
        action_id=action_id,
        fallback_used=fallback_used,
    )
    return result
