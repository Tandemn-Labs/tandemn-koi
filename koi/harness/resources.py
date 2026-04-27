"""Shared resource-map helpers for harness packet builders."""

from __future__ import annotations

from typing import Any, Optional

from koi.schemas import ResourceMap
from koi.tools.resources import parse_orca_resources


async def resource_map_for(
    agent: Any,
    *,
    ledger: Any = None,
    resource_map: Optional[ResourceMap] = None,
) -> Optional[ResourceMap]:
    if resource_map is not None:
        return ledger.apply_to_resource_map(resource_map) if ledger is not None else resource_map
    orca = getattr(agent, "orca", None)
    if orca is None or not hasattr(orca, "get_resources"):
        return None
    raw = await orca.get_resources()
    rm = parse_orca_resources(raw)
    if ledger is not None:
        rm = ledger.apply_to_resource_map(rm)
    return rm
