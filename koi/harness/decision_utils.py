"""Shared placement and decision conversion helpers for harness executors."""

from __future__ import annotations

from typing import Any

from koi.schemas import DataSource, EngineConfig, PlacementConfig


def source_to_data_source(source: str) -> DataSource:
    if source == "VERIFIED" or source == "memory_verified":
        return DataSource.MEMORY
    if source == "PerfDB" or source == "perfdb_exact":
        return DataSource.EXACT_MATCH
    return DataSource.ANALYTICAL


def source_to_prediction_source(source: str) -> str:
    if source == "VERIFIED":
        return "memory_verified"
    if source == "PerfDB":
        return "perfdb_exact"
    return source or "analytical"


def placement_config_from_payload(
    payload: dict[str, Any],
    *,
    fallback_region: str = "unknown",
) -> PlacementConfig:
    tp = int(payload.get("tp") or 1)
    pp = int(payload.get("pp") or 1)
    dp = int(payload.get("dp") or 1)
    num_gpus = tp * pp * dp
    return PlacementConfig(
        gpu_type=str(payload.get("gpu_type") or "unknown"),
        instance_type=str(payload.get("instance_type") or "unknown"),
        num_gpus=num_gpus,
        num_instances=max(1, int(payload.get("num_instances") or -(-num_gpus // 8))),
        tp=tp,
        pp=pp,
        dp=dp,
        region=str(payload.get("region") or fallback_region or "unknown"),
        engine_config=EngineConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        market=str(payload.get("market") or payload.get("planned_market") or "on_demand"),
    )


def alternative_payloads(
    packet,
    selected_action_id: str,
    *,
    action_type: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    for option in packet.valid_actions():
        if option.action_id == selected_action_id or option.action_type != action_type:
            continue
        payload = packet.detail_sections.get(option.executor_payload_ref or "", {})
        alternatives.append(
            {
                "gpu_type": payload.get("gpu_type"),
                "instance_type": payload.get("instance_type"),
                "tp": payload.get("tp"),
                "pp": payload.get("pp"),
                "dp": payload.get("dp", 1),
                "region": payload.get("region"),
                "market": payload.get("market") or payload.get("planned_market"),
                "predicted_tps": payload.get("predicted_tps"),
                "source": payload.get("source"),
            }
        )
        if len(alternatives) >= limit:
            break
    return alternatives
