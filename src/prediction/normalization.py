"""Canonical, non-mutating normalization for surrogate candidates."""

import json
import math
from typing import Any

_ALIASES = {
    "gpu_mem_gb": ("gpu_memory_gb", "gpu_vram_gb", "vram_gb_per_gpu"),
    "isl_token_avg": ("input_len_tokens_avg", "input_length_avg"),
    "model_params_b": ("params_billion", "num_params_billions"),
    "num_attn_heads": ("num_attention_heads",),
    "num_hidden_layers": ("num_layers",),
    "num_kv_heads": ("num_key_value_heads",),
    "osl_token_avg": ("output_len_tokens_avg", "output_length_avg"),
}


def normalize_candidate_inputs(
    job_config: dict[str, Any] | None,
    job_features: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return canonical copies without mutating caller-owned dictionaries."""
    config = dict(job_config or {})
    features = dict(job_features or {})

    for key in ("model_config", "model_config_json", "model_metadata"):
        for name, value in _as_dict(features.get(key)).items():
            features.setdefault(name, value)

    for canonical, aliases in _ALIASES.items():
        if config.get(canonical) is None:
            value = _first(config, aliases)
            if value is not None:
                config[canonical] = value
        if features.get(canonical) is None:
            value = _first(features, aliases)
            if value is not None:
                features[canonical] = value

    return config, features


def merged_candidate(candidate: Any) -> dict[str, Any]:
    """Return normalized features overlaid by normalized config."""
    config, features = normalize_candidate_inputs(candidate.job_config, candidate.job_features)
    return {**features, **config}


def normalize_precision(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return {"bfloat16": "bf16", "float16": "fp16"}.get(normalized, normalized)


def normalize_workload_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"batch", "batched", "offline", "offline_batch", "capacity"}:
        return "batch"
    if normalized in {"online", "interactive", "streaming"}:
        return "online"
    return normalized


def architecture_signature(values: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return a strict model architecture identity when all required fields exist."""
    model_class = values.get("model_architecture") or values.get("model_class")
    architectures = values.get("architectures")
    if model_class is None and isinstance(architectures, list):
        model_class = next(iter(architectures), None)
    hidden_size = values.get("hidden_size")
    hidden_layers = values.get("num_hidden_layers")
    attention_heads = values.get("num_attn_heads")
    kv_heads = values.get("num_kv_heads")
    if (
        model_class is None
        or hidden_size is None
        or hidden_layers is None
        or attention_heads is None
        or kv_heads is None
    ):
        return None
    is_moe = _as_bool(values.get("is_moe", False))
    experts = values.get("num_routed_experts") or values.get("num_experts")
    active = values.get("num_active_experts") or values.get("num_experts_active")
    if is_moe and (experts is None or active is None):
        return None
    try:
        expert_count = int(float(experts)) if experts is not None and is_moe else None
        active_count = int(float(active)) if active is not None and is_moe else None
        return (
            str(model_class),
            int(hidden_size),
            int(hidden_layers),
            int(attention_heads),
            int(kv_heads),
            is_moe,
            expert_count,
            active_count,
        )
    except (TypeError, ValueError, OverflowError):
        return None


def distance_features(values: dict[str, Any]) -> dict[str, float] | None:
    """Return canonical positive numeric features used for measured neighbors."""
    max_seq = values.get("max_num_seq") or values.get("max_num_seqs") or values.get("batch_size")
    raw = {
        "tp": values.get("tp", 1),
        "pp": values.get("pp", 1),
        "isl": values.get("isl_token_avg"),
        "osl": values.get("osl_token_avg"),
        "max_num_seq": max_seq,
        "effective_batch_size": effective_batch_size(values),
    }
    try:
        result = {name: float(value) for name, value in raw.items()}
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        return None
    return result


def effective_batch_size(values: dict[str, Any]) -> float | None:
    explicit = _positive_float(values.get("effective_batch_size"))
    if explicit is not None:
        return explicit
    mode = normalize_workload_type(values.get("workload_type") or values.get("type"))
    if mode == "online":
        return _positive_float(
            values.get("estimated_concurrency")
            or values.get("max_concurrent_streaming")
            or values.get("max_concurrent_requests")
        )
    max_seq = _positive_float(
        values.get("max_num_seq") or values.get("max_num_seqs") or values.get("batch_size")
    )
    max_tokens = _positive_float(values.get("max_num_batched_tokens"))
    isl = _positive_float(values.get("isl_token_avg"))
    osl = _positive_float(values.get("osl_token_avg"))
    if None in (max_seq, max_tokens, isl, osl):
        return None
    assert max_seq is not None and max_tokens is not None and isl is not None and osl is not None
    return max(1.0, min(max_seq, math.floor(max_tokens / (isl + osl))))


def candidate_gpu_count(values: dict[str, Any]) -> int:
    explicit = values.get("gpu_count", values.get("count"))
    if explicit is not None:
        return max(1, int(explicit))
    return max(1, int(values.get("tp", 1)) * int(values.get("pp", 1)))


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _first(values: dict[str, Any], names: tuple[str, ...]) -> Any:
    return next((values[name] for name in names if values.get(name) is not None), None)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None
