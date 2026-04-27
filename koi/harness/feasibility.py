"""Shared config feasibility and physics summaries for harness cards."""

from __future__ import annotations

from typing import Any, Optional

from koi.logging_config import get_logger
from koi.model_features import compute_config_features, get_model_features
from koi.schemas import JobRequest, ResourceMap

logger = get_logger("koi.harness.feasibility")


def estimate_num_instances(row: dict[str, Any], rm: Optional[ResourceMap]) -> int:
    tp = int(row.get("tp", 1) or 1)
    pp = int(row.get("pp", 1) or 1)
    dp = int(row.get("dp", 1) or 1)
    num_gpus = tp * pp * dp
    resource = rm.get_resource(str(row.get("gpu_type", ""))) if rm is not None else None
    if resource is None:
        return max(1, -(-num_gpus // 8))
    return max(1, -(-num_gpus // max(1, resource.gpus_per_instance)))


def physics_for_row(
    req: JobRequest,
    rm: Optional[ResourceMap],
    row: dict[str, Any],
    *,
    require_vram_headroom_gb: float = 8.0,
) -> dict[str, Any]:
    gpu_type = str(row.get("gpu_type", "unknown"))
    resource = rm.get_resource(gpu_type) if rm is not None else None
    if resource is None:
        return {
            "hard_feasibility": {
                "gpu_type": gpu_type,
                "instance_type": row.get("instance_type"),
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
            "gpu_type": gpu_type,
            "instance_type": row.get("instance_type"),
            "vram_fit": vram_headroom_gb >= require_vram_headroom_gb,
            "vram_headroom_gb": round(vram_headroom_gb, 2),
            "tp_heads_valid": mf.num_attention_heads % tp == 0,
            "pp_layers_valid": mf.num_layers % pp == 0,
            "kv_heads_per_tp_shard": round(float(feats.get("kv_heads_per_tp_shard", 0.0) or 0.0), 3),
            "crosses_node_boundary": bool(feats.get("crosses_node_boundary", 0)),
            "capacity_ok": tp * pp * dp <= resource.available_gpus,
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
            "harness_physics_failed",
            job_id=req.job_id,
            gpu_type=gpu_type,
            error=str(exc),
        )
        return {
            "hard_feasibility": {
                "gpu_type": gpu_type,
                "instance_type": row.get("instance_type"),
                "capacity_ok": True,
                "runtime_supported": True,
                "physics_error": str(exc),
            },
            "physics": {},
        }
