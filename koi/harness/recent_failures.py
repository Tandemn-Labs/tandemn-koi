"""Recent-failure evidence for harness menu ranking.

The backing table is still named ``cooloffs`` for implementation continuity,
but Phase 4.5 treats those rows as ranking evidence, not hard cooldowns.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

from koi.schemas import ResourceMap


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def recommendation_for_diagnosis(diagnosis_code: Optional[str]) -> str:
    code = (diagnosis_code or "").lower()
    if code == "spot_preemption":
        return "Prefer on_demand or another region/GPU if available"
    if code in {"no_capacity", "quota_exhausted", "quota"}:
        return "Prefer alternate market, region, or GPU family if available"
    if code == "oom":
        return "Prefer higher VRAM or a different topology if available"
    return "Treat this scope as recently risky; prefer safer alternatives if available"


def recent_failure_for_scope(
    memory: Any,
    *,
    gpu_type: str,
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    market: Optional[str] = None,
    tp: Optional[int] = None,
    pp: Optional[int] = None,
    dp: Optional[int] = None,
    now: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Return a recent-failure annotation for an exact-ish candidate scope.

    A signal is same-scope when all specific fields present on the signal match
    the candidate. Topology is only considered when the signal stored it (OOM
    style failures). Unknown candidate fields are allowed to match, so old rows
    with partial scope still provide evidence.
    """
    if memory is None or not hasattr(memory, "get_active_cooloffs") or not gpu_type:
        return None
    current = time.time() if now is None else now
    try:
        rows = memory.get_active_cooloffs(
            gpu_type=gpu_type,
            region=region if region and region != "unknown" else None,
            market=market if market and market != "unknown" else None,
            now=current,
            limit=20,
        )
    except Exception:
        return None

    def _matches(row: dict[str, Any]) -> bool:
        for field, value in (
            ("instance_type", instance_type),
            ("region", region),
            ("market", market),
        ):
            row_value = row.get(field)
            if row_value and value and row_value != value:
                return False
        for field, value in (("tp", tp), ("pp", pp), ("dp", dp)):
            row_value = row.get(field)
            if row_value is not None and value is not None and int(row_value) != int(value):
                return False
        return True

    matches = [row for row in rows if _matches(row)]
    if not matches:
        return None
    row = max(matches, key=lambda item: float(item.get("created_at") or 0.0))
    created_at = float(row.get("created_at") or current)
    age_minutes = max(0.0, (current - created_at) / 60.0)
    diagnosis_code = row.get("diagnosis_code")
    return {
        "same_scope": True,
        "key": row.get("key"),
        "last_failed_at": _iso(created_at),
        "age_minutes": round(age_minutes, 1),
        "diagnosis_code": diagnosis_code,
        "reason": row.get("reason"),
        "recommendation": recommendation_for_diagnosis(diagnosis_code),
        "source_event_id": row.get("source_event_id"),
    }


def annotate_row_recent_failure(
    memory: Any,
    row: dict[str, Any],
    rm: Optional[ResourceMap],
    *,
    default_market: Optional[str] = None,
    now: Optional[float] = None,
) -> dict[str, Any]:
    gpu_type = str(row.get("gpu_type") or "")
    resource = rm.get_resource(gpu_type) if rm is not None and gpu_type else None
    market = row.get("planned_market") or row.get("market") or default_market or "on_demand"
    region = row.get("region") or (resource.region if resource else None) or (rm.region if rm else None)
    signal = recent_failure_for_scope(
        memory,
        gpu_type=gpu_type,
        instance_type=row.get("instance_type") or (resource.instance_type if resource else None),
        region=region,
        market=market,
        tp=int(row.get("tp") or 1),
        pp=int(row.get("pp") or 1),
        dp=int(row.get("dp") or 1),
        now=now,
    )
    if signal:
        row["recent_failure"] = signal
    else:
        row.pop("recent_failure", None)
    return row


def recent_failure_penalty(item: Any) -> int:
    if isinstance(item, dict):
        signal = item.get("recent_failure")
    else:
        signal = getattr(item, "recent_failure", None)
    return 1 if signal and signal.get("same_scope") else 0


def annotate_and_rank_rows(
    memory: Any,
    rows: list[dict[str, Any]],
    rm: Optional[ResourceMap],
    *,
    default_market: Optional[str] = None,
    now: Optional[float] = None,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for idx, raw in enumerate(rows):
        row = annotate_row_recent_failure(
            memory,
            dict(raw),
            rm,
            default_market=default_market,
            now=now,
        )
        row["_original_rank"] = idx
        annotated.append(row)
    if not any(row.get("recent_failure") for row in annotated):
        for row in annotated:
            row.pop("_original_rank", None)
        return annotated
    annotated.sort(
        key=lambda row: (
            not bool(row.get("meets_slo", True)),
            recent_failure_penalty(row),
            row.get("under_cost_roofline") is False,
            float(row.get("total_cost") or 0.0),
            int(row.get("_original_rank") or 0),
        )
    )
    for row in annotated:
        row.pop("_original_rank", None)
    return annotated
