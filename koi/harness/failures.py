"""Shared failure and retry-budget helpers for harness recovery prompts."""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from koi.tools.memory import AgenticMemory

DEFAULT_RETRY_BUDGET = 2

_FAILURE_PATTERNS = [
    (re.compile(r"spot|preempt", re.I), "spot_preemption"),
    (re.compile(r"insufficient.?capacity|no.?capacity", re.I), "no_capacity"),
    (re.compile(r"oom|out.?of.?memory|cuda.?oom", re.I), "oom"),
    (re.compile(r"quota", re.I), "quota"),
]


def retry_budget_limit(env_var: str = "KOI_HARNESS_P1_RETRY_BUDGET") -> int:
    raw = os.environ.get(env_var, str(DEFAULT_RETRY_BUDGET))
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_RETRY_BUDGET


def classify_failure(reason: str) -> str:
    for pattern, category in _FAILURE_PATTERNS:
        if pattern.search(reason or ""):
            return category
    return "unknown"


def decision_chain(memory: AgenticMemory, decision_id: Optional[str]) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = decision_id
    while current and current not in seen:
        seen.add(current)
        row = memory.get_decision(current)
        if not row:
            break
        chain.append(row)
        current = row.get("parent_decision_id")
    return chain


def retry_budget_used(chain: list[dict[str, Any]], triggered_by: str = "launch_recovery") -> int:
    return sum(1 for row in chain if row.get("triggered_by") == triggered_by)


def failed_entries(req: Any, decision: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    configs = list(getattr(req, "configs_tried", []) or [])
    reasons = list(getattr(req, "failure_reasons", []) or [])
    entries: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs):
        item = dict(cfg or {})
        if decision and item.get("gpu_type") == decision.get("gpu_type"):
            item.setdefault("tp", decision.get("tp"))
            item.setdefault("pp", decision.get("pp"))
            item.setdefault("dp", decision.get("dp"))
        reason = reasons[idx] if idx < len(reasons) else "unknown"
        item["failure_reason"] = reason
        item["failure_category"] = classify_failure(reason)
        entries.append(item)
    return entries


def config_key(config: dict[str, Any], *, include_market: bool = True) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        config.get("gpu_type"),
        config.get("instance_type"),
        int(config.get("tp") or 0),
        int(config.get("pp") or 0),
        int(config.get("dp") or 1),
    )
    if include_market:
        key = key + (config.get("market") or config.get("planned_market") or "unknown",)
    return key


def matches_failed_same_scope(row: dict[str, Any], failed: list[dict[str, Any]]) -> bool:
    row_key = config_key(row, include_market=True)
    for item in failed:
        if config_key(item, include_market=True) == row_key:
            return True
        if not item.get("tp") and not item.get("pp"):
            if (
                item.get("gpu_type") == row.get("gpu_type")
                and item.get("instance_type") == row.get("instance_type")
                and (item.get("market") or "unknown")
                == (row.get("planned_market") or row.get("market") or "unknown")
            ):
                return True
    return False


def same_topology_different_market(row: dict[str, Any], failed: list[dict[str, Any]]) -> bool:
    row_no_market = config_key(row, include_market=False)
    row_market = row.get("planned_market") or row.get("market") or "unknown"
    for item in failed:
        if config_key(item, include_market=False) == row_no_market:
            return (item.get("market") or "unknown") != row_market
    return False
