"""P5j job post-mortem harness.

P5j synthesizes a terminal, job-level diagnosis from chain outcomes,
launch-failure evidence, and existing P5c diagnoses. It is read-only with
respect to cluster state: no scale, kill, or launch tools are exposed.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from typing import Any, Optional

from pydantic import BaseModel, Field

from koi.harness.failures import classify_failure
from koi.harness.p5c import P5cDiagnosis, deterministic_diagnosis
from koi.harness.packet_tools import build_packet_read_tools
from koi.harness.prompts import HARNESS_SYSTEM_PROMPT
from koi.harness.schemas import HarnessState, TransitionPacket, TransitionType
from koi.llm import KoiToolRunner
from koi.logging_config import get_logger
from koi.tools.memory import AgenticMemory

logger = get_logger("koi.harness.p5j")

P5J_TIMEOUT = 60.0
P5J_MAX_ITERATIONS = 2

_KNOWN_SECTIONS = (
    "terminal",
    "chains",
    "chain_diagnoses",
    "chain_outcomes",
    "launch_failures",
    "failure_summary",
    "cooloffs",
)


class P5jDiagnosis(BaseModel):
    diagnosis_code: str = Field(min_length=1)
    bottleneck: str = "unknown"
    next_fix: str = "operator_review"
    failure_scope: str = "job"
    terminal_status: str = "failed"
    failed_chains: int = 0
    diagnosed_chains: int = 0
    chain_diagnoses: list[dict[str, Any]] = Field(default_factory=list)
    event_at: float = Field(default_factory=time.time)
    rationale: str = ""


def _reason_text(req: Any) -> str:
    reason_code = getattr(req, "reason_code", None)
    reason_detail = getattr(req, "reason_detail", None)
    parts = []
    if reason_code:
        parts.append(str(getattr(reason_code, "value", reason_code)))
    if reason_detail:
        parts.append(str(reason_detail))
    status = getattr(req, "status", None)
    if status:
        parts.append(f"terminal status={status}")
    return "; ".join(parts) or "terminal job failure"


def _chain_status(chain: Any) -> str:
    status = getattr(chain, "status", "unknown")
    return str(getattr(status, "value", status))


def _chain_summary(chain_id: str, chain: Any) -> dict[str, Any]:
    config = getattr(chain, "config", None)
    return {
        "chain_id": chain_id,
        "status": _chain_status(chain),
        "decision_id": getattr(chain, "decision_id", None),
        "gpu_type": getattr(config, "gpu_type", None),
        "instance_type": getattr(config, "instance_type", None),
        "region": getattr(config, "region", None),
        "market": getattr(config, "market", None),
        "tp": getattr(config, "tp", None),
        "pp": getattr(config, "pp", None),
        "dp": getattr(config, "dp", None),
        "smoothed_tps": getattr(chain, "smoothed_tps", None),
        "predicted_tps": getattr(chain, "predicted_tps", None),
        "tokens_completed": getattr(chain, "tokens_completed", None),
        "tokens_remaining": getattr(chain, "tokens_remaining", None),
        "slo_headroom_pct": getattr(chain, "slo_headroom_pct", None),
        "elapsed_hours": getattr(chain, "elapsed_hours", None),
    }


def _parse_outcome_diagnosis(outcome: dict[str, Any]) -> Optional[dict[str, Any]]:
    raw = outcome.get("diff_from_parent")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("diagnosis_code"):
                return {
                    "chain_id": outcome.get("job_id"),
                    "decision_id": outcome.get("decision_id"),
                    "diagnosis_code": parsed.get("diagnosis_code"),
                    "bottleneck": parsed.get("bottleneck"),
                    "next_fix": parsed.get("next_fix"),
                    "failure_scope": parsed.get("failure_scope"),
                    "source": "p5c_outcome",
                    "rationale": parsed.get("rationale", ""),
                }
        except (TypeError, json.JSONDecodeError):
            pass

    diagnosis = outcome.get("diagnosis") or ""
    category = outcome.get("failure_category") or classify_failure(diagnosis)
    if category and category != "unknown":
        return {
            "chain_id": outcome.get("job_id"),
            "decision_id": outcome.get("decision_id"),
            "diagnosis_code": category,
            "bottleneck": outcome.get("bottleneck") or _bottleneck_for_code(category),
            "next_fix": _next_fix_for_code(category),
            "failure_scope": "chain",
            "source": "outcome",
            "rationale": diagnosis,
        }
    return None


def _diagnosis_from_chain(
    *,
    req: Any,
    chain_id: str,
    chain: Any,
    terminal_reason: str,
) -> dict[str, Any]:
    config = getattr(chain, "config", None)
    region = str(getattr(config, "region", "unknown") or "unknown")
    market = str(getattr(config, "market", "unknown") or "unknown")
    category = classify_failure(terminal_reason)
    diag = deterministic_diagnosis(
        req=req,
        tracker=chain,
        failure_category=category,
        region=region,
        market=market,
        actual_tps_before_death=getattr(chain, "smoothed_tps", None),
    )
    return {
        "chain_id": chain_id,
        "decision_id": getattr(chain, "decision_id", None),
        "diagnosis_code": diag.diagnosis_code,
        "bottleneck": diag.bottleneck,
        "next_fix": diag.next_fix,
        "failure_scope": diag.failure_scope,
        "source": "p5j_fanout_deterministic",
        "rationale": diag.rationale,
    }


def _bottleneck_for_code(code: str) -> str:
    if code in {"spot_preemption", "no_capacity", "quota", "quota_exhausted"}:
        return "market_capacity"
    if code in {"oom", "memory_bound", "job_memory_bound"}:
        return "memory_bound"
    if code in {"heartbeat_timeout", "runtime_unhealthy"}:
        return "runtime_unhealthy"
    return "unknown"


def _next_fix_for_code(code: str) -> str:
    if code == "spot_preemption":
        return "retry_same_topology_on_demand"
    if code in {"no_capacity", "quota", "quota_exhausted"}:
        return "switch_market_region_or_gpu_family"
    if code in {"oom", "memory_bound", "job_memory_bound"}:
        return "increase_vram_or_reduce_memory_pressure"
    if code in {"heartbeat_timeout", "runtime_unhealthy"}:
        return "replace_unhealthy_chains_and_review_runtime_logs"
    return "operator_review"


def _job_code_for_chain_code(code: str) -> str:
    if code in {"oom", "memory_bound"}:
        return "job_memory_bound"
    if code in {"spot_preemption", "no_capacity", "quota", "quota_exhausted"}:
        return "job_capacity_exhausted"
    if code in {"heartbeat_timeout", "runtime_unhealthy"}:
        return "job_runtime_unhealthy"
    return "job_failed_unknown"


def _dominant_chain_code(chain_diagnoses: list[dict[str, Any]]) -> str:
    counts = Counter(
        str(diag.get("diagnosis_code") or "unknown") for diag in chain_diagnoses
    )
    if not counts:
        return "unknown"
    priority = [
        "oom",
        "memory_bound",
        "quota_exhausted",
        "quota",
        "no_capacity",
        "spot_preemption",
        "heartbeat_timeout",
        "runtime_unhealthy",
    ]
    max_count = max(counts.values())
    leaders = {code for code, count in counts.items() if count == max_count}
    for code in priority:
        if code in leaders:
            return code
    return sorted(leaders)[0]


def deterministic_job_diagnosis(
    *,
    req: Any,
    chain_diagnoses: list[dict[str, Any]],
    launch_failures: list[dict[str, Any]],
    chain_count: int,
    now: Optional[float] = None,
) -> P5jDiagnosis:
    event_at = time.time() if now is None else now
    terminal_status = str(getattr(req, "status", "failed") or "failed")
    if chain_diagnoses:
        dominant = _dominant_chain_code(chain_diagnoses)
        code = _job_code_for_chain_code(dominant)
        bottleneck = _bottleneck_for_code(dominant)
        next_fix = _next_fix_for_code(dominant)
        rationale = (
            f"Terminal job failure synthesized from {len(chain_diagnoses)} "
            f"chain diagnoses; dominant chain code is {dominant}."
        )
    elif launch_failures:
        categories = Counter(
            str(row.get("failure_category") or "unknown") for row in launch_failures
        )
        dominant = categories.most_common(1)[0][0]
        code = _job_code_for_chain_code(dominant)
        if code == "job_failed_unknown":
            code = "job_launch_failed"
        bottleneck = _bottleneck_for_code(dominant)
        next_fix = _next_fix_for_code(dominant)
        rationale = (
            f"Terminal job failure synthesized from {len(launch_failures)} "
            f"failed launch configs; dominant launch category is {dominant}."
        )
    else:
        dominant = classify_failure(_reason_text(req))
        code = _job_code_for_chain_code(dominant)
        bottleneck = _bottleneck_for_code(dominant)
        next_fix = _next_fix_for_code(dominant)
        rationale = f"Terminal job failure had limited evidence: {_reason_text(req)}."

    return P5jDiagnosis(
        diagnosis_code=code,
        bottleneck=bottleneck,
        next_fix=next_fix,
        failure_scope=str(getattr(req, "job_id", "job")),
        terminal_status=terminal_status,
        failed_chains=chain_count,
        diagnosed_chains=len(chain_diagnoses),
        chain_diagnoses=chain_diagnoses,
        event_at=event_at,
        rationale=rationale,
    )


def collect_chain_diagnoses(
    *,
    req: Any,
    memory: AgenticMemory,
    group_chains: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    job_id = str(getattr(req, "job_id", ""))
    outcomes = memory.query_outcomes(job_id=job_id, limit=100)
    by_decision: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        parsed = _parse_outcome_diagnosis(outcome)
        decision_id = outcome.get("decision_id")
        if parsed is not None and decision_id:
            by_decision[str(decision_id)] = parsed

    terminal_reason = _reason_text(req)
    diagnoses = list(by_decision.values())
    seen_decisions = set(by_decision)
    for chain_id, chain in (group_chains or {}).items():
        decision_id = getattr(chain, "decision_id", None)
        if decision_id and str(decision_id) in seen_decisions:
            continue
        if getattr(chain, "config", None) is None:
            continue
        diag = _diagnosis_from_chain(
            req=req,
            chain_id=chain_id,
            chain=chain,
            terminal_reason=terminal_reason,
        )
        diagnoses.append(diag)
        if decision_id:
            seen_decisions.add(str(decision_id))
    return diagnoses


def _failure_summaries(
    memory: AgenticMemory,
    *,
    group_chains: Optional[dict[str, Any]],
    launch_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    scopes: set[tuple[str, Optional[str], Optional[str]]] = set()
    for chain in (group_chains or {}).values():
        config = getattr(chain, "config", None)
        gpu_type = getattr(config, "gpu_type", None)
        if gpu_type:
            scopes.add(
                (
                    str(gpu_type),
                    getattr(config, "region", None),
                    getattr(config, "market", None),
                )
            )
    for row in launch_failures:
        gpu_type = row.get("gpu_type")
        if gpu_type:
            scopes.add((str(gpu_type), row.get("region"), row.get("market")))

    summaries: dict[str, Any] = {}
    for gpu_type, region, market in sorted(
        scopes,
        key=lambda item: tuple(str(part or "") for part in item),
    ):
        try:
            key = "|".join(
                str(part or "unknown") for part in (gpu_type, region, market)
            )
            summaries[key] = memory.get_failure_summary(
                gpu_type,
                region=region if region != "unknown" else None,
                market=market if market != "unknown" else None,
            )
        except Exception as exc:
            summaries[str(gpu_type)] = {"error": str(exc)}
    return summaries


def build_p5j_packet(
    *,
    req: Any,
    memory: AgenticMemory,
    group_chains: Optional[dict[str, Any]] = None,
    chain_diagnoses: Optional[list[dict[str, Any]]] = None,
) -> TransitionPacket:
    job_id = str(getattr(req, "job_id", "unknown"))
    chains = {
        chain_id: _chain_summary(chain_id, chain)
        for chain_id, chain in (group_chains or {}).items()
    }
    launch_failures = memory.get_failed_configs(job_id)
    diagnoses = chain_diagnoses if chain_diagnoses is not None else collect_chain_diagnoses(
        req=req,
        memory=memory,
        group_chains=group_chains,
    )
    chain_outcomes = memory.query_outcomes(job_id=job_id, limit=100)
    try:
        cooloffs = memory.get_active_cooloffs(limit=50)
    except Exception as exc:
        cooloffs = [{"error": str(exc)}]
    failure_summaries = _failure_summaries(
        memory,
        group_chains=group_chains,
        launch_failures=launch_failures,
    )
    dominant = _dominant_chain_code(diagnoses) if diagnoses else "unknown"
    terminal_context = {
        "job_id": job_id,
        "group_id": getattr(req, "group_id", None),
        "decision_id": getattr(req, "decision_id", None),
        "terminal_status": getattr(req, "status", "failed"),
        "reason_code": str(getattr(getattr(req, "reason_code", None), "value", getattr(req, "reason_code", None)) or ""),
        "reason_detail": getattr(req, "reason_detail", None),
        "metrics": getattr(req, "metrics", {}) or {},
    }
    detail_sections = {
        "terminal:job": terminal_context,
        "chains:summary": chains,
        "chain_diagnoses:all": diagnoses,
        "chain_outcomes:job": chain_outcomes,
        "launch_failures:job": launch_failures,
        "failure_summary:scopes": failure_summaries,
        "cooloffs:active": cooloffs,
    }
    return TransitionPacket(
        packet_id=f"p5j-{job_id}",
        job_id=job_id,
        state=HarnessState.TERMINAL_FAILED,
        transition_type=TransitionType.JOB_POSTMORTEM,
        job_context=terminal_context,
        runtime_context={
            "chains": chains,
            "aggregate_tps": sum(
                float(row.get("smoothed_tps") or 0.0) for row in chains.values()
            ),
        },
        failure_context={
            "terminal": terminal_context,
            "chain_diagnoses": diagnoses,
            "launch_failures": launch_failures,
            "dominant_chain_code": dominant,
        },
        evidence_summary={
            "chain_count": len(chains),
            "diagnosed_chain_count": len(diagnoses),
            "launch_failure_count": len(launch_failures),
            "chain_outcome_count": len(chain_outcomes),
            "dominant_chain_code": dominant,
        },
        detail_sections=detail_sections,
        guards={
            "read_only": True,
            "write_terminal_diagnosis_only": True,
            "no_cluster_mutation_tools": True,
        },
    )


def render_p5j_prompt(packet: TransitionPacket) -> str:
    lines = [
        "P5J JOB POST-MORTEM",
        "Synthesize why the overall job reached terminal failure.",
        "Do not propose or execute cluster actions. Return a typed P5jDiagnosis.",
        "Prefer existing P5c chain diagnoses when present; otherwise use launch and terminal evidence.",
        "",
        "JOB CONTEXT:",
        json.dumps(packet.job_context, indent=2, sort_keys=True),
        "",
        "FAILURE CONTEXT:",
        json.dumps(packet.failure_context, indent=2, sort_keys=True, default=str),
        "",
        "EVIDENCE SUMMARY:",
        json.dumps(packet.evidence_summary, indent=2, sort_keys=True),
        "",
        "DETAIL REFS:",
        json.dumps(sorted(packet.detail_sections), indent=2),
        "",
        "Diagnosis fields must be concise and machine-readable.",
    ]
    return "\n".join(lines)


def _packet_tools(memory: AgenticMemory, packet: TransitionPacket) -> dict[str, Any]:
    tools = build_packet_read_tools(packet, known_sections=_KNOWN_SECTIONS, include_packet_sections=True)

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


def _normalize_diagnosis(
    diagnosis: P5jDiagnosis,
    *,
    fallback: P5jDiagnosis,
) -> P5jDiagnosis:
    data = diagnosis.model_dump()
    if not data.get("chain_diagnoses"):
        data["chain_diagnoses"] = fallback.chain_diagnoses
    if not data.get("failed_chains"):
        data["failed_chains"] = fallback.failed_chains
    if not data.get("diagnosed_chains"):
        data["diagnosed_chains"] = len(data.get("chain_diagnoses") or [])
    if not data.get("failure_scope") or data["failure_scope"] == "job":
        data["failure_scope"] = fallback.failure_scope
    return P5jDiagnosis(**data)


async def run_job_postmortem(
    *,
    agent: Any,
    req: Any,
    memory: AgenticMemory,
    group_chains: Optional[dict[str, Any]] = None,
) -> P5jDiagnosis:
    chain_diagnoses = collect_chain_diagnoses(
        req=req,
        memory=memory,
        group_chains=group_chains,
    )
    launch_failures = memory.get_failed_configs(str(getattr(req, "job_id", "")))
    fallback = deterministic_job_diagnosis(
        req=req,
        chain_diagnoses=chain_diagnoses,
        launch_failures=launch_failures,
        chain_count=len(group_chains or {}),
    )
    model = getattr(agent, "_model", None)
    if model is None:
        return fallback

    packet = build_p5j_packet(
        req=req,
        memory=memory,
        group_chains=group_chains,
        chain_diagnoses=chain_diagnoses,
    )
    runner = KoiToolRunner(
        model=model,
        system_prompt=HARNESS_SYSTEM_PROMPT,
        tools=_packet_tools(memory, packet),
    )
    try:
        _, diagnosis = await runner.run_typed(
            render_p5j_prompt(packet),
            label="p5j",
            job_id=packet.job_id,
            max_iterations=P5J_MAX_ITERATIONS,
            timeout=P5J_TIMEOUT,
            output_type=P5jDiagnosis,
        )
        return _normalize_diagnosis(diagnosis, fallback=fallback)
    except asyncio.TimeoutError:
        logger.error("p5j_timeout", job_id=packet.job_id, timeout=P5J_TIMEOUT)
        return fallback
    except Exception as exc:
        logger.warning("p5j_reasoner_failed", job_id=packet.job_id, error=str(exc))
        return fallback
