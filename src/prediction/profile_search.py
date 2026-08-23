"""Joint GPU/model/workload matching for surrogate support profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.prediction.compatibility import GPUProfile, canonicalize_dtype, gpu_profile_distance

PROFILE_SEARCH_VERSION = "joint-knn-v1"


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    architecture: str
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    parameter_count: float
    is_moe: bool
    routed_experts: int
    active_experts: int
    weight_dtype: str
    max_context: int
    modality: str = "text"
    moe_intermediate_size: int = 0
    fmha_dtype: str = "bf16"
    kv_cache_dtype: str = "bf16"


@dataclass(frozen=True)
class WorkloadProfile:
    workload_type: str
    input_tokens: int
    output_tokens: int
    batch_size: int
    request_rate: float | None = None


@dataclass(frozen=True)
class TopologyProfile:
    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1


@dataclass(frozen=True)
class PredictionProfile:
    gpu: GPUProfile
    model: ModelProfile
    workload: WorkloadProfile
    topology: TopologyProfile
    engine_name: str
    engine_version: str | None


@dataclass(frozen=True)
class SupportedProfile:
    profile_id: str
    gpu: GPUProfile
    model: ModelProfile
    engine_name: str
    engine_version: str
    aic_system: str


@dataclass(frozen=True)
class OperationSignature:
    weight_bytes_per_gpu: float
    kv_bytes_per_token_per_gpu: float
    prefill_flops: float
    decode_flops_per_token: float
    prefill_memory_bytes: float
    decode_memory_bytes_per_token: float
    communication_bytes_per_token: float


@dataclass(frozen=True)
class ProfileMatch:
    supported: SupportedProfile
    distance: float
    confidence: float
    prefill_speed_ratio: float
    decode_speed_ratio: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_version": PROFILE_SEARCH_VERSION,
            "profile_id": self.supported.profile_id,
            "gpu": self.supported.gpu.canonical_name,
            "model_id": self.supported.model.model_id,
            "aic_system": self.supported.aic_system,
            "engine_name": self.supported.engine_name,
            "engine_version": self.supported.engine_version,
            "distance": self.distance,
            "confidence": self.confidence,
            "prefill_speed_ratio": self.prefill_speed_ratio,
            "decode_speed_ratio": self.decode_speed_ratio,
            "reasons": list(self.reasons),
        }


def model_profile_from_values(model_id: str, values: dict[str, Any]) -> ModelProfile | None:
    """Build a model profile from Store/AIC catalog fields."""
    raw_value = values.get("raw_config")
    raw: dict[str, Any] = raw_value if isinstance(raw_value, dict) else {}
    architecture = _text(
        values.get("architecture")
        or values.get("model_architecture")
        or values.get("model_class")
        or _first(values.get("architectures"))
        or _first(raw.get("architectures"))
    )
    layers = _integer(
        values.get("layers")
        or values.get("num_hidden_layers")
        or values.get("num_layers")
        or raw.get("num_hidden_layers")
    )
    hidden = _integer(values.get("hidden_size") or raw.get("hidden_size"))
    intermediate = _integer(
        values.get("inter_size") or values.get("intermediate_size") or raw.get("intermediate_size")
    )
    heads = _integer(
        values.get("n")
        or values.get("num_attn_heads")
        or values.get("num_attention_heads")
        or raw.get("num_attention_heads")
    )
    kv_heads = _integer(
        values.get("n_kv")
        or values.get("num_kv_heads")
        or values.get("num_key_value_heads")
        or raw.get("num_key_value_heads")
        or heads
    )
    if None in (architecture, layers, hidden, intermediate, heads, kv_heads):
        return None
    assert layers is not None and hidden is not None and intermediate is not None
    assert heads is not None and kv_heads is not None and architecture is not None
    head_dim = _integer(values.get("d") or values.get("head_dim")) or hidden // heads
    vocab = _integer(values.get("vocab") or values.get("vocab_size") or raw.get("vocab_size")) or 0
    experts = (
        _integer(
            values.get("num_experts") or values.get("num_routed_experts") or raw.get("num_experts")
        )
        or 0
    )
    active = (
        _integer(
            values.get("topk") or values.get("num_active_experts") or raw.get("num_experts_per_tok")
        )
        or 0
    )
    is_moe = bool(values.get("is_moe") or experts > 0)
    moe_intermediate = (
        _integer(
            values.get("moe_inter_size")
            or values.get("moe_intermediate_size")
            or raw.get("moe_intermediate_size")
        )
        or 0
    )
    params = _number(values.get("model_params_b") or values.get("params_billion"))
    parameter_count = (
        params * 1e9
        if params is not None
        else _estimate_parameter_count(
            layers,
            hidden,
            intermediate,
            heads,
            kv_heads,
            head_dim,
            vocab,
            experts,
            moe_intermediate,
        )
    )
    dtype = (
        canonicalize_dtype(
            values.get("weight_dtype")
            or _dtype_from_model_id(model_id)
            or values.get("torch_dtype")
            or raw.get("torch_dtype")
            or "bf16"
        )
        or "bf16"
    )
    fmha_dtype = (
        canonicalize_dtype(
            values.get("fmha_quant_mode") or values.get("activation_dtype") or "bf16"
        )
        or "bf16"
    )
    kv_cache_dtype = (
        canonicalize_dtype(
            values.get("kvcache_quant_mode") or values.get("kvcache_dtype") or "bf16"
        )
        or "bf16"
    )
    context = (
        _integer(
            values.get("context")
            or values.get("max_pos_embeddings")
            or raw.get("max_position_embeddings")
        )
        or 8192
    )
    return ModelProfile(
        model_id=str(model_id),
        architecture=architecture,
        layers=layers,
        hidden_size=hidden,
        intermediate_size=intermediate,
        attention_heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        parameter_count=parameter_count,
        is_moe=is_moe,
        routed_experts=experts,
        active_experts=active,
        weight_dtype=dtype,
        max_context=context,
        modality=(
            "multimodal"
            if "vl" in model_id.lower() or "vision" in model_id.lower() or raw.get("vision_config")
            else "text"
        ),
        moe_intermediate_size=moe_intermediate,
        fmha_dtype=fmha_dtype,
        kv_cache_dtype=kv_cache_dtype,
    )


def prediction_profile_from_values(
    values: dict[str, Any], gpu: GPUProfile
) -> PredictionProfile | None:
    model_id = _text(values.get("model_id"))
    if model_id is None:
        return None
    model = model_profile_from_values(model_id, values)
    input_tokens = _integer(values.get("isl_token_avg"))
    output_tokens = _integer(values.get("osl_token_avg"))
    batch = _integer(
        values.get("effective_batch_size")
        or values.get("max_concurrent_streaming")
        or values.get("max_num_seq")
    )
    if model is None or input_tokens is None or output_tokens is None or batch is None:
        return None
    request_rate = _number(values.get("request_arrival_rate"))
    return PredictionProfile(
        gpu=gpu,
        model=model,
        workload=WorkloadProfile(
            workload_type=str(values.get("type") or values.get("workload_type") or "batch").lower(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            batch_size=batch,
            request_rate=request_rate,
        ),
        topology=TopologyProfile(
            tp=max(1, _integer(values.get("tp")) or 1),
            pp=max(1, _integer(values.get("pp")) or 1),
            ep=max(1, _integer(values.get("ep")) or 1),
            dp=max(1, _integer(values.get("dp")) or 1),
        ),
        engine_name=str(values.get("engine_name") or "vllm"),
        engine_version=_text(values.get("engine_version")),
    )


def build_operation_signature(
    model: ModelProfile,
    workload: WorkloadProfile,
    topology: TopologyProfile,
) -> OperationSignature:
    """Return target-comparable model work and memory quantities."""
    dtype_bytes = _dtype_bytes(model.weight_dtype)
    nonexpert_parameters, expert_parameters = _parameter_components(model)
    dense_shards = max(1, topology.tp * topology.pp)
    expert_shards = max(1, topology.ep * topology.pp)
    weight_parameters_per_gpu = nonexpert_parameters / dense_shards
    weight_parameters_per_gpu += expert_parameters / expert_shards
    active_expert_parameters = expert_parameters
    if model.is_moe and model.routed_experts > 0 and model.active_experts > 0:
        active_expert_parameters *= model.active_experts / model.routed_experts
    active_parameters_per_gpu = nonexpert_parameters / dense_shards
    active_parameters_per_gpu += active_expert_parameters / expert_shards
    weight_bytes_per_gpu = weight_parameters_per_gpu * dtype_bytes
    kv_bytes = (
        2.0
        * model.layers
        * model.kv_heads
        * model.head_dim
        * _dtype_bytes(model.kv_cache_dtype)
        / max(1, topology.tp * topology.pp)
    )
    prefill_flops = 2.0 * active_parameters_per_gpu * workload.input_tokens * workload.batch_size
    prefill_flops += (
        4.0
        * model.layers
        * workload.input_tokens**2
        * model.hidden_size
        * workload.batch_size
        / dense_shards
    )
    context = workload.input_tokens + max(1, workload.output_tokens // 2)
    decode_flops = 2.0 * active_parameters_per_gpu * workload.batch_size
    decode_flops += (
        4.0 * model.layers * context * model.hidden_size * workload.batch_size / dense_shards
    )
    communication = (
        4.0
        * model.layers
        * model.hidden_size
        * dtype_bytes
        * max(0, topology.tp - 1)
        / max(1, topology.tp)
    )
    communication += max(0, topology.pp - 1) * model.hidden_size * dtype_bytes
    if model.is_moe and topology.ep > 1:
        communication += (
            2.0 * model.layers * model.hidden_size * dtype_bytes * (topology.ep - 1) / topology.ep
        )
    return OperationSignature(
        weight_bytes_per_gpu=weight_bytes_per_gpu,
        kv_bytes_per_token_per_gpu=kv_bytes,
        prefill_flops=prefill_flops,
        decode_flops_per_token=decode_flops,
        prefill_memory_bytes=weight_bytes_per_gpu
        + kv_bytes * workload.input_tokens * workload.batch_size,
        decode_memory_bytes_per_token=weight_bytes_per_gpu
        + kv_bytes * context * workload.batch_size,
        communication_bytes_per_token=communication,
    )


def rank_profiles(
    requested: PredictionProfile,
    available: tuple[SupportedProfile, ...] | list[SupportedProfile],
    *,
    limit: int = 5,
) -> tuple[ProfileMatch, ...]:
    """Rank backend support points by joint GPU/model/workload similarity."""
    requested_signature = build_operation_signature(
        requested.model, requested.workload, requested.topology
    )
    matches = []
    for supported in available:
        if supported.engine_name != requested.engine_name:
            continue
        if supported.model.attention_heads % requested.topology.tp != 0:
            continue
        if supported.model.layers < requested.topology.pp:
            continue
        if requested.gpu.memory_gb is not None and requested_signature.weight_bytes_per_gpu > (
            requested.gpu.memory_gb * (1 << 30)
        ):
            continue
        proxy_signature = build_operation_signature(
            supported.model, requested.workload, requested.topology
        )
        gpu_distance, gpu_reasons = gpu_profile_distance(
            requested.gpu, supported.gpu, allow_cross_vendor=True
        )
        model_distance, model_reasons = _model_distance(
            requested.model,
            supported.model,
            requested_signature,
            proxy_signature,
        )
        workload_distance, workload_reasons = _workload_distance(
            requested.workload, supported.model
        )
        weight_distance, weight_reasons = _dtype_distance(
            requested.model.weight_dtype, supported.model.weight_dtype
        )
        fmha_distance, fmha_reasons = _dtype_distance(
            requested.model.fmha_dtype, supported.model.fmha_dtype
        )
        kv_distance, kv_reasons = _dtype_distance(
            requested.model.kv_cache_dtype, supported.model.kv_cache_dtype
        )
        dtype_distance = math.sqrt(weight_distance**2 + fmha_distance**2 + kv_distance**2)
        dtype_reasons = (*weight_reasons, *fmha_reasons, *kv_reasons)
        distance = math.sqrt(
            gpu_distance**2 + model_distance**2 + workload_distance**2 + dtype_distance**2
        )
        target_prefill = _phase_time(
            requested.gpu,
            requested_signature.prefill_flops,
            requested_signature.prefill_memory_bytes,
            requested_signature.communication_bytes_per_token
            * requested.workload.input_tokens
            * requested.workload.batch_size,
        )
        proxy_prefill = _phase_time(
            supported.gpu,
            proxy_signature.prefill_flops,
            proxy_signature.prefill_memory_bytes,
            proxy_signature.communication_bytes_per_token
            * requested.workload.input_tokens
            * requested.workload.batch_size,
        )
        target_decode = _phase_time(
            requested.gpu,
            requested_signature.decode_flops_per_token,
            requested_signature.decode_memory_bytes_per_token,
            requested_signature.communication_bytes_per_token * requested.workload.batch_size,
        )
        proxy_decode = _phase_time(
            supported.gpu,
            proxy_signature.decode_flops_per_token,
            proxy_signature.decode_memory_bytes_per_token,
            proxy_signature.communication_bytes_per_token * requested.workload.batch_size,
        )
        prefill_ratio = _bounded_ratio(proxy_prefill, target_prefill)
        decode_ratio = _bounded_ratio(proxy_decode, target_decode)
        matches.append(
            ProfileMatch(
                supported=supported,
                distance=distance,
                confidence=max(0.05, min(1.0, math.exp(-0.5 * distance))),
                prefill_speed_ratio=prefill_ratio,
                decode_speed_ratio=decode_ratio,
                reasons=(*gpu_reasons, *model_reasons, *workload_reasons, *dtype_reasons),
            )
        )
    matches.sort(key=lambda match: (match.distance, match.supported.profile_id))
    return tuple(matches[: max(1, int(limit))])


def resolve_model_reference(
    model_id: str,
    values: dict[str, Any],
    aliases: dict[str, str],
    reference_sizes_b: dict[str, float],
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve a simple analytical model reference when no full profile index is used."""
    exact = aliases.get(model_id.lower())
    if exact is not None:
        return exact, None
    if values.get("solver_config_dir"):
        return model_id or "custom", None
    params_b = _number(values.get("model_params_b"))
    if params_b is None or values.get("is_moe"):
        return None, None
    family = "qwen" if "qwen" in model_id.lower() else "llama"
    candidates = [name for name in reference_sizes_b if family in name]
    if not candidates:
        return None, None
    resolved = min(candidates, key=lambda name: abs(params_b - reference_sizes_b[name]))
    throughput_scale = min(1.0, reference_sizes_b[resolved] / params_b)
    return resolved, {
        "requested_model_id": model_id,
        "resolved_peer_model": resolved,
        "reason": "nearest_parameter_count_roofline_model",
        "throughput_scale": throughput_scale,
        "latency_scale": 1.0 / max(throughput_scale, 1e-12),
    }


def _model_distance(
    requested: ModelProfile,
    supported: ModelProfile,
    requested_signature: OperationSignature,
    supported_signature: OperationSignature,
):
    terms = [
        2.0 * _log_ratio(requested_signature.prefill_flops, supported_signature.prefill_flops) ** 2,
        2.0
        * _log_ratio(
            requested_signature.decode_flops_per_token,
            supported_signature.decode_flops_per_token,
        )
        ** 2,
        1.5
        * _log_ratio(
            requested_signature.kv_bytes_per_token_per_gpu,
            supported_signature.kv_bytes_per_token_per_gpu,
        )
        ** 2,
        _log_ratio(
            requested_signature.weight_bytes_per_gpu,
            supported_signature.weight_bytes_per_gpu,
        )
        ** 2,
    ]
    reasons = ["model_operation_signature"]
    if requested.is_moe != supported.is_moe:
        terms.append(4.0)
        reasons.append("dense_moe_penalty")
    if requested.modality != supported.modality:
        terms.append(4.0)
        reasons.append("modality_penalty")
    if _model_family(requested) != _model_family(supported):
        terms.append(1.0)
        reasons.append("model_family_penalty")
    if requested.architecture != supported.architecture:
        terms.append(0.25)
        reasons.append("architecture_penalty")
    return math.sqrt(sum(terms)), tuple(reasons)


def _workload_distance(workload: WorkloadProfile, model: ModelProfile):
    total_tokens = workload.input_tokens + workload.output_tokens
    context_penalty = max(0.0, math.log(total_tokens / max(1, model.max_context)))
    batch_penalty = max(0.0, math.log(workload.batch_size / 256.0))
    reasons = []
    if context_penalty:
        reasons.append("context_extrapolation")
    if batch_penalty:
        reasons.append("batch_extrapolation")
    return math.sqrt(context_penalty**2 + batch_penalty**2), tuple(reasons)


def _dtype_distance(requested: str, supported: str):
    requested_dtype = canonicalize_dtype(requested) or requested
    supported_dtype = canonicalize_dtype(supported) or supported
    if requested_dtype == supported_dtype:
        return 0.0, ()
    if {requested_dtype, supported_dtype} == {"bf16", "fp16"}:
        return 0.25, ("same_width_dtype_proxy",)
    if requested_dtype.startswith("fp8") and supported_dtype == "bf16":
        return 0.75, ("higher_precision_dtype_proxy",)
    return 1.5, ("dtype_family_penalty",)


def _phase_time(gpu: GPUProfile, flops: float, memory_bytes: float, communication_bytes: float):
    compute = flops / max((gpu.fp16_tflops or 1.0) * 1e12, 1.0)
    memory = memory_bytes / max((gpu.memory_bandwidth_gbps or 1.0) * 1e9, 1.0)
    communication = communication_bytes / max(
        (gpu.nvlink_bandwidth_gbps or gpu.pcie_bandwidth_gbps or 1.0) * 1e9,
        1.0,
    )
    return max(compute, memory) + communication


def _estimate_parameter_count(
    layers,
    hidden,
    intermediate,
    heads,
    kv_heads,
    head_dim,
    vocab,
    experts,
    moe_intermediate,
):
    model = ModelProfile(
        model_id="estimated",
        architecture="estimated",
        layers=layers,
        hidden_size=hidden,
        intermediate_size=intermediate,
        attention_heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        parameter_count=1.0,
        is_moe=experts > 0,
        routed_experts=experts,
        active_experts=experts,
        weight_dtype="bf16",
        max_context=1,
        moe_intermediate_size=moe_intermediate,
    )
    nonexpert, expert = _raw_parameter_components(model)
    return nonexpert + expert


def _parameter_components(model: ModelProfile) -> tuple[float, float]:
    nonexpert, expert = _raw_parameter_components(model)
    estimated_total = nonexpert + expert
    scale = model.parameter_count / max(estimated_total, 1.0)
    return nonexpert * scale, expert * scale


def _raw_parameter_components(model: ModelProfile) -> tuple[float, float]:
    layers = model.layers
    hidden = model.hidden_size
    intermediate = model.intermediate_size
    kv_heads = model.kv_heads
    head_dim = model.head_dim
    vocab = model.vocab_size
    experts = model.routed_experts
    moe_intermediate = model.moe_intermediate_size
    kv_width = kv_heads * head_dim
    attention = 2.0 * hidden**2 + 2.0 * hidden * kv_width
    mlp = 3.0 * hidden * intermediate
    expert_mlp = 3.0 * hidden * (moe_intermediate or intermediate)
    nonexpert = layers * (attention + mlp) + vocab * hidden
    expert = layers * expert_mlp * experts if experts else 0.0
    return nonexpert, expert


def _dtype_from_model_id(model_id: str) -> str | None:
    lowered = model_id.lower()
    if "fp8" in lowered:
        return "fp8"
    if "int8" in lowered:
        return "int8"
    if "int4" in lowered or "awq" in lowered or "gptq" in lowered:
        return "int4"
    return None


def _model_family(model: ModelProfile) -> str:
    text = f"{model.model_id} {model.architecture}".lower()
    for family in ("qwen", "llama", "deepseek", "mistral", "gemma", "minimax"):
        if family in text:
            return family
    return model.architecture.lower()


def _dtype_bytes(dtype: str) -> float:
    return {
        "fp32": 4.0,
        "bf16": 2.0,
        "fp16": 2.0,
        "fp8": 1.0,
        "fp8_e4m3": 1.0,
        "fp8_e5m2": 1.0,
        "int8": 1.0,
        "int4": 0.5,
        "fp4": 0.5,
        "nf4": 0.5,
    }.get(canonicalize_dtype(dtype) or dtype, 2.0)


def _bounded_ratio(proxy_time: float, target_time: float) -> float:
    return max(0.05, min(20.0, proxy_time / max(target_time, 1e-12)))


def _log_ratio(left: float, right: float) -> float:
    return abs(math.log(max(left, 1e-12) / max(right, 1e-12)))


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _first(value: Any):
    return value[0] if isinstance(value, list) and value else None
