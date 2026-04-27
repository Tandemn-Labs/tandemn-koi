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


_PSCALE_SECTION_KEYS = (
    "physics",
    "perfdb_exact",
    "perfdb_proxy",
    "memory_success",
    "memory_failure",
    "quota",
    "recent_failures",
    "runtime_metrics",
    "executor_payload",
    "suggestion",
)


def _pscale_section_keys_for(action_id: str) -> list[str]:
    return [f"{section}:{action_id}" for section in _PSCALE_SECTION_KEYS]


def _build_pscale_detail_sections(
    agent: Any,
    trigger: MonitoringTrigger,
    suggestion: Any,
    action_id: str,
    executor_payload: dict[str, Any],
) -> dict[str, Any]:
    """Granular per-action detail sections for Pscale (architecture spec §8)."""

    sections: dict[str, Any] = {}
    tracker = trigger.job_tracker
    config = tracker.get("config", {}) if isinstance(tracker, dict) else {}
    gpu_type = getattr(suggestion, "gpu_type", None) or config.get("gpu_type")
    tp = getattr(suggestion, "tp", None) or config.get("tp", 1) or 1
    pp = getattr(suggestion, "pp", None) or config.get("pp", 1) or 1
    market = config.get("market", "on_demand")
    region = config.get("region")

    # physics: derive per-config physics for the suggestion's target topology.
    physics_section: dict[str, Any] = {
        "gpu_type": gpu_type,
        "tp": tp,
        "pp": pp,
    }
    parent_model_name: Optional[str] = None
    parent_decision = None
    parent_decision_id = tracker.get("decision_id") if isinstance(tracker, dict) else None
    memory = getattr(agent, "memory", None)
    if memory is not None and parent_decision_id:
        try:
            parent_decision = memory.get_decision(parent_decision_id)
        except Exception:
            parent_decision = None
    if parent_decision and isinstance(parent_decision, dict):
        parent_model_name = parent_decision.get("model_name")
    if parent_model_name:
        try:
            from koi.model_features import compute_config_features, get_model_features
            from koi.tools.physics import lookup_gpu_spec

            gpu_spec = lookup_gpu_spec(str(gpu_type)) if gpu_type else {}
            mf = get_model_features(parent_model_name, dtype=parent_decision.get("quantization") or "fp16")
            feats = compute_config_features(
                mf,
                gpu_type=str(gpu_type or "unknown"),
                tp=int(tp),
                pp=int(pp),
                dp=1,
                input_len=int(parent_decision.get("avg_input_tokens", 512) or 512),
                output_len=int(parent_decision.get("avg_output_tokens", 256) or 256),
                gpus_per_node=int(gpu_spec.get("gpus_per_node", 8) or 8) if gpu_spec else 8,
                price_per_gpu_hour=0.0,
            )
            physics_section.update(
                {
                    "model_name": parent_model_name,
                    "model_arch": {
                        "params_billions": getattr(mf, "num_params_billions", None),
                        "num_layers": getattr(mf, "num_layers", None),
                        "num_attention_heads": getattr(mf, "num_attention_heads", None),
                        "num_kv_heads": getattr(mf, "num_kv_heads", None),
                        "is_moe": getattr(mf, "is_moe", False),
                    },
                    "config_features": feats,
                }
            )
        except Exception as exc:
            physics_section["error"] = str(exc)
    sections[f"physics:{action_id}"] = physics_section

    # perfdb_exact / perfdb_proxy: rows for this gpu/tp/pp from PerfDB.
    perfdb = getattr(agent, "perfdb", None)
    perfdb_exact: list[dict[str, Any]] = []
    perfdb_proxy: list[dict[str, Any]] = []
    if perfdb is not None and parent_model_name and gpu_type:
        try:
            perfdb_exact = perfdb.query(
                model_name=str(parent_model_name),
                gpu_type=str(gpu_type),
                tp=int(tp),
                pp=int(pp),
                limit=10,
            ) or []
        except Exception as exc:
            perfdb_exact = [{"error": str(exc)}]
        if not perfdb_exact:
            try:
                from koi.tools.physics import find_similar_models, get_model_features as _gmf

                target = _gmf(str(parent_model_name), dtype=parent_decision.get("quantization") or "fp16")
                distinct = perfdb.get_distinct_models() if hasattr(perfdb, "get_distinct_models") else []
                for proxy in (find_similar_models(target, distinct or []) or [])[:3]:
                    proxy_records = perfdb.query(
                        model_name=str(proxy.get("model_name")),
                        gpu_type=str(gpu_type),
                        tp=int(tp),
                        pp=int(pp),
                        limit=5,
                    ) or []
                    perfdb_proxy.append(
                        {
                            "proxy_model": proxy.get("model_name"),
                            "distance": proxy.get("distance"),
                            "confidence": proxy.get("confidence"),
                            "records": proxy_records,
                        }
                    )
            except Exception as exc:
                perfdb_proxy = [{"error": str(exc)}]
    sections[f"perfdb_exact:{action_id}"] = perfdb_exact
    sections[f"perfdb_proxy:{action_id}"] = perfdb_proxy

    # memory_success / memory_failure
    memory_success: list[dict[str, Any]] = []
    memory_failure: list[dict[str, Any]] = []
    if memory is not None and parent_model_name:
        try:
            memory_success = memory.query_outcomes(
                model_name=str(parent_model_name), status="succeeded", limit=10
            ) or []
        except Exception as exc:
            memory_success = [{"error": str(exc)}]
        try:
            memory_failure = memory.query_outcomes(
                model_name=str(parent_model_name), status="failed", limit=10
            ) or []
        except Exception as exc:
            memory_failure = [{"error": str(exc)}]
    sections[f"memory_success:{action_id}"] = memory_success
    sections[f"memory_failure:{action_id}"] = memory_failure

    # quota / recent_failures from the long-term beta priors slice.
    quota_section: dict[str, Any] = {
        "gpu_type": gpu_type,
        "region": region,
        "market": market,
    }
    if memory is not None and gpu_type:
        try:
            quota_section["failure_summary"] = memory.get_failure_summary(
                str(gpu_type), region=region, market=market
            )
        except Exception as exc:
            quota_section["failure_summary_error"] = str(exc)
    sections[f"quota:{action_id}"] = quota_section
    sections[f"recent_failures:{action_id}"] = quota_section.get("failure_summary", {})

    # runtime_metrics: per-replica live metrics for kill suggestions, fleet
    # snapshot for scale-up suggestions.
    runtime_metrics: dict[str, Any] = {}
    if agent.monitor and tracker.get("group_id"):
        group_chains = agent.monitor.get_group_chains(tracker["group_id"]) or {}
        for chain_id, chain in group_chains.items():
            runtime_metrics[chain_id] = {
                "smoothed_tps": getattr(chain, "smoothed_tps", 0.0),
                "predicted_tps": getattr(chain, "predicted_tps", 0.0),
                "cost_per_hour": getattr(chain, "predicted_cost_per_hour", 0.0),
                "status": getattr(getattr(chain, "status", None), "value", None)
                or str(getattr(chain, "status", "unknown")),
                "gpu_type": getattr(chain.config, "gpu_type", None),
                "tp": getattr(chain.config, "tp", None),
                "pp": getattr(chain.config, "pp", None),
            }
    sections[f"runtime_metrics:{action_id}"] = runtime_metrics

    # executor_payload + suggestion blob (deterministic execution + raw)
    sections[f"executor_payload:{action_id}"] = executor_payload
    sections[f"suggestion:{action_id}"] = (
        suggestion.__dict__ if hasattr(suggestion, "__dict__") else {}
    )
    return sections


async def build_pscale_packet(
    agent: Any,
    trigger: MonitoringTrigger,
    precomputed_candidates: Optional[list[Any]] = None,
) -> TransitionPacket:
    suggestions = agent._rank_runtime_policy_suggestions(
        trigger,
        precomputed_candidates=precomputed_candidates,
        limit=MAX_MENU_OPTIONS - 1,
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

        per_action_sections = _build_pscale_detail_sections(
            agent, trigger, suggestion, action_id, executor_payload
        )
        detail_sections.update(per_action_sections)
        evidence = {
            "source": suggestion.source,
            "gpu_type": suggestion.gpu_type,
            "memory_successes": (
                len(per_action_sections[f"memory_success:{action_id}"])
                if isinstance(per_action_sections[f"memory_success:{action_id}"], list)
                else 0
            ),
            "memory_failures": (
                len(per_action_sections[f"memory_failure:{action_id}"])
                if isinstance(per_action_sections[f"memory_failure:{action_id}"], list)
                else 0
            ),
            "perfdb_exact_rows": (
                len(per_action_sections[f"perfdb_exact:{action_id}"])
                if isinstance(per_action_sections[f"perfdb_exact:{action_id}"], list)
                else 0
            ),
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
                evidence=evidence,
                cost={
                    "projected_total_cost_usd": suggestion.projected_total_cost_usd,
                    "cost_overage_usd": suggestion.cost_overage_usd,
                },
                risk={},
                executor_payload_ref=f"executor_payload:{action_id}",
                detail_refs=_pscale_section_keys_for(action_id),
            )
        )

    noop_id = _action_id(len(options))
    detail_sections[f"executor_payload:{noop_id}"] = {"tool": "noop"}
    detail_sections[f"suggestion:{noop_id}"] = {
        "kind": "noop",
        "reason": "no scale or kill suggested",
    }
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
            executor_payload_ref=f"executor_payload:{noop_id}",
            detail_refs=[
                f"executor_payload:{noop_id}",
                f"suggestion:{noop_id}",
            ],
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
        "You may use packet/read-only tools to inspect details or explore alternatives before choosing.",
        "If the recommended menu is missing a necessary DEGRADED scale-up, call request_custom_scale_option first, then choose the returned action_id.",
        "Do not execute raw cluster mutations directly; final execution still happens through the chosen action_id.",
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


def _next_custom_action_id(packet: TransitionPacket) -> str:
    used = {option.action_id for option in packet.action_options}
    idx = 1
    while f"x{idx}" in used:
        idx += 1
    return f"x{idx}"


def _packet_tools(agent: Any, packet: TransitionPacket) -> dict[str, Any]:
    async def list_detail_sections(action_id: str) -> str:
        """List the named detail sections available for one action_id."""
        option = packet.get_action(action_id)
        if option is None:
            return f"unknown action_id={action_id!r}"
        return json.dumps(option.detail_refs, indent=2)

    async def read_option_detail(action_id: str, section: str = "all") -> str:
        """Read a specific detail section for one runtime action option.

        Sections include: physics, perfdb_exact, perfdb_proxy, memory_success,
        memory_failure, quota, recent_failures, runtime_metrics,
        executor_payload, suggestion, all.
        """
        option = packet.get_action(action_id)
        if option is None:
            return f"unknown action_id={action_id!r}"
        if section == "all":
            details = {ref: packet.detail_sections.get(ref) for ref in option.detail_refs}
            return json.dumps(details, indent=2, default=str)
        ref = f"{section}:{action_id}"
        if ref not in option.detail_refs:
            return (
                f"unknown section={section!r} for action_id={action_id!r}; "
                f"available={option.detail_refs}"
            )
        return json.dumps(
            {"section": ref, "data": packet.detail_sections.get(ref)},
            indent=2,
            default=str,
        )

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

    async def read_packet_section(section_id: str) -> str:
        """Read a named section from the transition packet."""
        if section_id == "runtime_context":
            return json.dumps(packet.runtime_context, indent=2, default=str)
        if section_id == "evidence_summary":
            return json.dumps(packet.evidence_summary, indent=2, default=str)
        if section_id == "guards":
            return json.dumps(packet.guards, indent=2, default=str)
        section = packet.detail_sections.get(section_id)
        if section is None:
            return f"unknown section_id={section_id!r}"
        return json.dumps(section, indent=2, default=str)

    async def request_custom_scale_option(
        gpu_type: str,
        tp: int,
        pp: int,
        count: int = 1,
        on_demand: Optional[bool] = None,
        reason: str = "",
    ) -> str:
        """Add a custom scale-up option after bounded exploration.

        This preserves the harness boundary: the model may propose an explored
        config, but it must still choose the returned action_id as its final
        typed output before any cluster mutation happens.
        """
        if packet.state != HarnessState.DEGRADED:
            return "custom scale options are only allowed for DEGRADED scale-up decisions"
        if count <= 0:
            return "custom scale option rejected: count must be positive"
        if tp <= 0 or pp <= 0:
            return "custom scale option rejected: tp and pp must be positive"
        if len([o for o in packet.action_options if o.action_id.startswith("x")]) >= 2:
            return "custom scale option rejected: custom option limit reached"

        action_id = _next_custom_action_id(packet)
        executor_payload = {
            "tool": "scale_chain_tool",
            "job_id": packet.runtime_context.get("group_id") or packet.job_id,
            "gpu_type": gpu_type,
            "tp": tp,
            "pp": pp,
            "count": count,
            "on_demand": on_demand,
        }
        packet.detail_sections[f"executor_payload:{action_id}"] = executor_payload
        packet.detail_sections[f"suggestion:{action_id}"] = {
            "kind": "custom_scale_up",
            "source": "llm_exploration",
            "reason": reason,
        }
        # Best-effort: leave other granular section slots empty so list_detail_sections
        # reports them as available (returns None when read).
        for placeholder in (
            "physics",
            "perfdb_exact",
            "perfdb_proxy",
            "memory_success",
            "memory_failure",
            "quota",
            "recent_failures",
            "runtime_metrics",
        ):
            packet.detail_sections.setdefault(f"{placeholder}:{action_id}", None)
        packet.action_options.append(
            ActionOption(
                action_id=action_id,
                action_type="scale_up",
                summary=(
                    f"Custom scale_up {gpu_type} TP={tp} PP={pp} "
                    f"count={count}"
                ),
                rank=len(packet.action_options) + 1,
                valid=True,
                performance={
                    "gpu_type": gpu_type,
                    "tp": tp,
                    "pp": pp,
                    "replica_id": None,
                    "meets_slo": None,
                    "source": "llm_exploration",
                },
                evidence={"source": "llm_exploration", "gpu_type": gpu_type},
                risk={
                    "reason": reason,
                    "note": "Custom option was proposed by the LLM after reading packet/tool context.",
                },
                executor_payload_ref=f"executor_payload:{action_id}",
                detail_refs=_pscale_section_keys_for(action_id),
            )
        )
        return json.dumps(
            {
                "added_action_id": action_id,
                "instruction": "Use this action_id in your final ChosenAction if you want to execute it.",
                "executor_payload": executor_payload,
            },
            indent=2,
            default=str,
        )

    tools: dict[str, Any] = {
        "list_detail_sections": list_detail_sections,
        "read_option_detail": read_option_detail,
        "compare_options": compare_options,
        "read_packet_section": read_packet_section,
        "request_custom_scale_option": request_custom_scale_option,
    }

    # Existing production read tools remain available as an exploration escape
    # hatch. Keep mutating tools out of this surface; execution still goes
    # through validated action_id payloads. Each wrapper catches backend errors
    # and returns them as tool text so an exploratory miss does not abort the
    # whole runtime decision.
    try:
        production_tools = agent._build_tools(monitor=agent.monitor)
    except Exception as exc:
        logger.warning("pscale_read_tool_build_failed", error=str(exc))
        production_tools = {}

    async def query_perfdb_tool(
        model_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
        tp: Optional[int] = None,
        pp: Optional[int] = None,
        limit: int = 10,
    ) -> str:
        """Explore PerfDB rows without mutating state."""
        tool = production_tools.get("query_perfdb")
        if tool is None:
            return "query_perfdb unavailable"
        try:
            return await tool(
                model_name=model_name,
                gpu_type=gpu_type,
                tp=tp,
                pp=pp,
                limit=limit,
            )
        except Exception as exc:
            return f"query_perfdb failed: {exc}"

    async def query_memory_tool(
        model_name: Optional[str] = None,
        job_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Explore prior decisions/outcomes without mutating state."""
        tool = production_tools.get("query_memory_tool")
        if tool is None:
            return "query_memory_tool unavailable"
        try:
            return await tool(
                model_name=model_name,
                job_id=job_id,
                status=status,
                limit=limit,
            )
        except Exception as exc:
            return f"query_memory_tool failed: {exc}"

    async def get_gpu_physics_tool(
        gpu_type: str = "L40S",
        model_name: Optional[str] = None,
    ) -> str:
        """Explore GPU/model physics without mutating state."""
        tool = production_tools.get("get_gpu_physics_tool")
        if tool is None:
            return "get_gpu_physics_tool unavailable"
        try:
            return await tool(gpu_type=gpu_type, model_name=model_name)
        except Exception as exc:
            return f"get_gpu_physics_tool failed: {exc}"

    async def get_model_arch_tool(model_name: str = "unknown") -> str:
        """Explore model architecture without mutating state."""
        tool = production_tools.get("get_model_arch_tool")
        if tool is None:
            return "get_model_arch_tool unavailable"
        try:
            return await tool(model_name=model_name)
        except Exception as exc:
            return f"get_model_arch_tool failed: {exc}"

    async def find_similar_models_tool(model_name: str = "unknown") -> str:
        """Explore physics-similar models without mutating state."""
        tool = production_tools.get("find_similar_models_tool")
        if tool is None:
            return "find_similar_models_tool unavailable"
        try:
            return await tool(model_name=model_name)
        except Exception as exc:
            return f"find_similar_models_tool failed: {exc}"

    async def get_resources_tool() -> str:
        """Explore current resources without mutating state."""
        tool = production_tools.get("get_resources_tool")
        if tool is None:
            return "get_resources_tool unavailable"
        try:
            return await tool()
        except Exception as exc:
            return f"get_resources_tool failed: {exc}"

    async def get_quota_status_tool(
        gpu_type: Optional[str] = None,
        region: Optional[str] = None,
        market: Optional[str] = None,
    ) -> str:
        """Explore quota state without mutating state."""
        tool = production_tools.get("get_quota_status_tool")
        if tool is None:
            return "get_quota_status_tool unavailable"
        try:
            return await tool(gpu_type=gpu_type, region=region, market=market)
        except Exception as exc:
            return f"get_quota_status_tool failed: {exc}"

    async def get_failure_summary_tool(
        gpu_type: str = "L40S",
        region: Optional[str] = None,
        market: Optional[str] = None,
    ) -> str:
        """Explore failure priors without mutating state."""
        tool = production_tools.get("get_failure_summary_tool")
        if tool is None:
            return "get_failure_summary_tool unavailable"
        try:
            return await tool(gpu_type=gpu_type, region=region, market=market)
        except Exception as exc:
            return f"get_failure_summary_tool failed: {exc}"

    async def get_job_metrics_tool(job_id: str = "") -> str:
        """Explore live job metrics without mutating state."""
        tool = production_tools.get("get_job_metrics_tool")
        if tool is None:
            return "get_job_metrics_tool unavailable"
        try:
            return await tool(job_id=job_id or packet.job_id)
        except Exception as exc:
            return f"get_job_metrics_tool failed: {exc}"

    tools.update(
        {
            "query_perfdb_tool": query_perfdb_tool,
            "query_memory_tool": query_memory_tool,
            "get_gpu_physics_tool": get_gpu_physics_tool,
            "get_model_arch_tool": get_model_arch_tool,
            "find_similar_models_tool": find_similar_models_tool,
            "get_resources_tool": get_resources_tool,
            "get_quota_status_tool": get_quota_status_tool,
            "get_failure_summary_tool": get_failure_summary_tool,
            "get_job_metrics_tool": get_job_metrics_tool,
        }
    )
    return tools


async def _execute_validated_action(agent: Any, packet: TransitionPacket, action_id: str) -> str:
    option = packet.get_action(action_id)
    if option is None:
        return f"No action executed: unknown action_id={action_id!r}"
    detail = packet.detail_sections.get(option.executor_payload_ref or "", {})
    if isinstance(detail, dict) and "tool" in detail:
        payload = detail
    else:
        payload = detail.get("executor_payload", {}) if isinstance(detail, dict) else {}
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
            on_demand=payload.get("on_demand"),
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
        tools=_packet_tools(agent, packet),
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
