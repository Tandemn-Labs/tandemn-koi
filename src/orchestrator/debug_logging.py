"""Durable debug logging for the Koi runner."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

STATE_FIELDS = {
    "S0_ENTER_TICK": ("cluster_snapshot",),
    "S1_OBSERVE": ("telemetry", "deployment_x"),
    "S2_VALIDATE": ("evidence_rows",),
    "S3_SLOW_UPDATE": ("new_slow_state",),
    "S4_AGENTIC_PLAN": ("candidate_plan",),
    "S5_VALIDATE_PLAN": ("validated_plan", "s5_repair_count"),
    "S6_DEPLOY": ("validated_plan", "deploy_acks"),
}
_COMPACT_LIMITS = {
    "summary": {"depth": 4, "items": 8, "string": 500},
    "full": {"depth": 8, "items": 80, "string": 5000},
}


class DebugLogger:
    """Write runner debug events to a per-run JSONL file."""

    def __init__(self, log_dir: str | Path, trace: str = "summary", run_id: str | None = None):
        """Create the run log directory and event paths."""
        if trace not in {"summary", "full"}:
            raise ValueError("trace must be 'summary' or 'full'")
        self.trace = trace
        self.run_id = run_id or self._default_run_id()
        self.run_dir = Path(log_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.runner_log_path = self.run_dir / "runner.log"
        self.events_path = self.run_dir / "events.jsonl"

    def persist_runner_tick(self, ctx: Any, agent: Any, llm: Any) -> None:
        """Persist one tick's runner, agent, and LLM debug records."""
        self.write_event("tick_summary", self._tick_summary(ctx), tick=getattr(ctx, "tick", None))
        self.write_event("llm_summary", self._llm_summary(llm), tick=getattr(ctx, "tick", None))
        self.write_event(
            "agent_summary", self._agent_summary(agent), tick=getattr(ctx, "tick", None)
        )
        if self.trace == "full":
            self.write_event(
                "llm_calls",
                {"calls": list(getattr(llm, "calls", []) or [])},
                tick=getattr(ctx, "tick", None),
            )
            trace = getattr(agent, "trace", None)
            self.write_event(
                "agent_trace",
                {"events": list(getattr(trace, "events", []) or [])},
                tick=getattr(ctx, "tick", None),
            )

    def persist_state(self, state: Any, ctx: Any) -> None:
        """Persist one compact FSM state snapshot."""
        state_name = getattr(state, "value", str(state))
        duration_ms = (getattr(ctx, "state_durations_ms", {}) or {}).get(state_name)
        payload = {"state": state_name, "duration_ms": duration_ms}
        payload.update(_state_summary(state_name, ctx, self.trace))
        self.write_event("state", payload, tick=getattr(ctx, "tick", None))

    def write_event(self, kind: str, payload: dict[str, Any], tick: int | None = None) -> None:
        """Append one structured event to ``events.jsonl``."""
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "tick": tick,
            "kind": kind,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")

    @staticmethod
    def _default_run_id() -> str:
        """Return a filesystem-safe run id."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-pid{os.getpid()}"

    @staticmethod
    def _tick_summary(ctx: Any) -> dict[str, Any]:
        """Return the compact tick outcome payload."""
        return {
            "states": [state.value for state in getattr(ctx, "state_history", [])],
            "evidence_rows": len(getattr(ctx, "evidence_rows", []) or []),
            "candidate_plan": _plan_summary(getattr(ctx, "candidate_plan", None)),
            "validated_plan": _plan_summary(getattr(ctx, "validated_plan", None)),
            "deploy_acks": getattr(ctx, "deploy_acks", []) or [],
            "error": repr(getattr(ctx, "error", None)) if getattr(ctx, "error", None) else None,
            "state_durations_ms": getattr(ctx, "state_durations_ms", {}) or {},
        }

    @staticmethod
    def _llm_summary(llm: Any) -> dict[str, Any]:
        """Return call counts and response previews without full prompts."""
        calls = list(getattr(llm, "calls", []) or [])
        return {
            "call_count": len(calls),
            "elapsed_sec": round(sum(float(call.get("elapsed_sec", 0.0)) for call in calls), 3),
            "responses": [_preview(call.get("response", "")) for call in calls],
        }

    @staticmethod
    def _agent_summary(agent: Any) -> dict[str, Any]:
        """Return agent trace event counts by kind."""
        trace = getattr(agent, "trace", None)
        events = list(getattr(trace, "events", []) or [])
        counts: dict[str, int] = {}
        for event in events:
            kind = str(event.get("kind", "unknown")) if isinstance(event, dict) else "unknown"
            counts[kind] = counts.get(kind, 0) + 1
        return {"event_count": len(events), "by_kind": counts}


def _plan_summary(plan: Any) -> dict[str, Any] | None:
    """Return compact action details for a Koi plan-like object."""
    if plan is None:
        return None
    actions = []
    for action in getattr(plan, "actions", []) or []:
        ladder = getattr(action, "ladder", None) or []
        action_type = getattr(action, "type", None)
        actions.append(
            {
                "job_id": getattr(action, "job_id", None),
                "type": getattr(action_type, "value", action_type),
                "rank_count": len(ladder),
                "rationale": getattr(action, "rationale", None),
            }
        )
    return {
        "action_count": len(actions),
        "actions": actions,
        "tick_rationale": getattr(plan, "tick_rationale", None),
    }


# TODO: REVIEW - add more fields, text, and compaction rules if traces are too large or insufficient.
def _state_summary(state_name: str, ctx: Any, trace: str) -> dict[str, Any]:
    """Return a compact debug payload for one FSM state."""
    out = {}
    for field in STATE_FIELDS.get(state_name, ()):
        value = getattr(ctx, field, None)
        out[field] = _plan_summary(value) if field.endswith("plan") else _compact(value, trace)
    return out


def _compact(value: Any, trace: str, depth: int = 0) -> Any:
    """Convert runtime objects into bounded JSON-friendly debug values."""
    limits = _COMPACT_LIMITS["full" if trace == "full" else "summary"]
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _preview(value, limits["string"])
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if depth >= limits["depth"]:
        return _preview(value, limits["string"])

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    else:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            value = vars(value)

    if isinstance(value, dict):
        items = list(value.items())
        out = {str(key): _compact(item, trace, depth + 1) for key, item in items[: limits["items"]]}
        if len(items) > limits["items"]:
            out["_truncated_items"] = len(items) - limits["items"]
        return out
    if isinstance(value, list | tuple | set):
        values = sorted(value, key=str) if isinstance(value, set) else list(value)
        compacted = [_compact(item, trace, depth + 1) for item in values[: limits["items"]]]
        if len(values) > limits["items"]:
            compacted.append(f"... {len(values) - limits['items']} more")
        return compacted
    return _preview(value, limits["string"])


def _preview(value: Any, limit: int = 500) -> str:
    """Return a bounded string preview for summary events."""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated at {limit} chars]"


def _json_default(value: Any) -> Any:
    """Best-effort JSON conversion for debug-only payloads."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(value)
