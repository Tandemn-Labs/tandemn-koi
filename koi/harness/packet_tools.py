"""Shared packet-scoped read tools for harness prompts."""

from __future__ import annotations

import json
from typing import Any, Iterable

from koi.harness.schemas import TransitionPacket


def build_packet_read_tools(
    packet: TransitionPacket,
    *,
    known_sections: Iterable[str] = (),
    include_packet_sections: bool = False,
) -> dict[str, Any]:
    known = set(known_sections)

    async def list_detail_sections(action_id: str) -> str:
        """List the named detail sections available for one action_id."""
        option = packet.get_action(action_id)
        if option is None:
            return f"unknown action_id={action_id!r}"
        return json.dumps(option.detail_refs, indent=2)

    async def read_option_detail(action_id: str, section: str = "all") -> str:
        """Read one action's detail section or all referenced sections."""
        option = packet.get_action(action_id)
        if option is None:
            return f"unknown action_id={action_id!r}"
        if section == "all":
            return json.dumps(
                {ref: packet.detail_sections.get(ref) for ref in option.detail_refs},
                indent=2,
                default=str,
            )
        ref = f"{section}:{action_id}"
        if ref not in option.detail_refs and section not in known:
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
        """Compare precomputed option summaries for requested action IDs."""
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
                    "valid": option.valid,
                    "summary": option.summary,
                    "performance": option.performance,
                    "cost": option.cost,
                    "availability": option.availability,
                    "evidence": option.evidence,
                    "risk": option.risk,
                    "physics": option.physics if lens == "physics" else {},
                }
            )
        return json.dumps(selected, indent=2, default=str)

    tools: dict[str, Any] = {
        "list_detail_sections": list_detail_sections,
        "read_option_detail": read_option_detail,
        "compare_options": compare_options,
    }

    if include_packet_sections:
        async def read_packet_section(section_id: str) -> str:
            """Read a named section from the transition packet."""
            if section_id == "job_context":
                return json.dumps(packet.job_context, indent=2, default=str)
            if section_id == "runtime_context":
                return json.dumps(packet.runtime_context, indent=2, default=str)
            if section_id == "failure_context":
                return json.dumps(packet.failure_context, indent=2, default=str)
            if section_id == "policy_context":
                return json.dumps(packet.policy_context, indent=2, default=str)
            if section_id == "evidence_summary":
                return json.dumps(packet.evidence_summary, indent=2, default=str)
            if section_id == "guards":
                return json.dumps(packet.guards, indent=2, default=str)
            section = packet.detail_sections.get(section_id)
            if section is None:
                return f"unknown section_id={section_id!r}"
            return json.dumps(section, indent=2, default=str)

        tools["read_packet_section"] = read_packet_section

    return tools
