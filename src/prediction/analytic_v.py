"""Analytic per-GPU memory and KV mediator estimates."""

GIB = 1024**3
ACTIVATION_RESERVE_GB = 2.0

_DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "float8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
    "nf4": 0.5,
}


def model_weight_gb(values: dict) -> float | None:
    """Return total model weight GiB, preferring candidate quantization."""
    params_b = _get(values, "model_params_b")
    bits = _get(values, "weight_quantization_bits")
    if params_b is not None and bits is not None:
        bytes_per_param = _weight_bytes_per_param(values)
        return None if bytes_per_param is None else float(params_b) * 1e9 * bytes_per_param / GIB
    explicit = _get(values, "model_size_gb")
    if explicit is not None:
        return float(explicit)
    if params_b is None:
        return None
    bytes_per_param = _weight_bytes_per_param(values)
    return None if bytes_per_param is None else float(params_b) * 1e9 * bytes_per_param / GIB


def kv_bytes_per_token(values: dict) -> float | None:
    """Return KV bytes per token for the whole model."""
    layers = _get(values, "num_hidden_layers")
    hidden = _get(values, "hidden_size")
    heads = _get(values, "num_attn_heads")
    kv_heads = _get(values, "num_kv_heads", "num_attn_heads")
    head_dim = _get(values, "head_dim")
    if head_dim is None:
        if hidden is None or not heads:
            return None
        head_dim = float(hidden) / float(heads)
    bytes_per_elem = _kv_bytes_per_elem(values)
    if layers is None or kv_heads is None or bytes_per_elem is None:
        return None
    return 2.0 * float(layers) * float(kv_heads) * float(head_dim) * bytes_per_elem


def compute_memory_v(
    job_config: dict,
    job_features: dict | None = None,
) -> dict[str, float]:
    """Return computable per-GPU V only; missing inputs cause omission."""
    values = {**(job_features or {}), **(job_config or {})}
    output: dict[str, float] = {}
    weight_gb = model_weight_gb(values)
    per_gpu_weight_gb = weight_gb / _weight_shards(values) if weight_gb is not None else None
    gpu_mem_gb = _get(values, "gpu_mem_gb", "gpu_memory_gb", "vram_gb_per_gpu")
    if gpu_mem_gb is None or per_gpu_weight_gb is None:
        return output

    memory_gb = float(gpu_mem_gb)
    if memory_gb <= 0:
        return output
    kv_per_token = kv_bytes_per_token(values)
    kv_shards = _kv_shards(values)
    per_gpu_kv_per_token = (
        kv_per_token / kv_shards if kv_per_token is not None and kv_shards is not None else None
    )
    demand = _kv_token_demand(values)
    if demand is not None and per_gpu_kv_per_token is None:
        return output

    demand_kv_gb = 0.0
    if demand is not None and per_gpu_kv_per_token is not None:
        demand_kv_gb = demand * per_gpu_kv_per_token / GIB
    used_gb = per_gpu_weight_gb + ACTIVATION_RESERVE_GB + demand_kv_gb
    output["vram_headroom_gb"] = memory_gb - used_gb
    output["gpu_mem_used_fraction"] = used_gb / memory_gb

    if per_gpu_kv_per_token is not None and per_gpu_kv_per_token > 0:
        gpu_mem_util = float(_get(values, "gpu_mem_util") or 0.9)
        budget_bytes = (
            max(0.0, memory_gb * gpu_mem_util - per_gpu_weight_gb - ACTIVATION_RESERVE_GB) * GIB
        )
        token_budget = budget_bytes / per_gpu_kv_per_token
        if demand is not None:
            pressure = demand / max(token_budget, 1.0)
            output["kv_cache_util"] = min(1.0, pressure)
            output["kv_pressure_score"] = pressure
    return output


def _get(values: dict, *names):
    return next((values[name] for name in names if values.get(name) is not None), None)


def _weight_bytes_per_param(values: dict) -> float | None:
    bits = _get(values, "weight_quantization_bits")
    if bits is not None:
        try:
            return float(bits) / 8.0
        except (TypeError, ValueError):
            return None
    dtype = _get(values, "weight_dtype", "activation_dtype")
    return _DTYPE_BYTES.get(str(dtype).lower()) if dtype is not None else None


def _kv_bytes_per_elem(values: dict) -> float | None:
    dtype = _get(values, "kvcache_dtype", "kvcache_quantization")
    return _DTYPE_BYTES.get(str(dtype).lower()) if dtype is not None else None


def _weight_shards(values: dict) -> int:
    return max(1, int(_get(values, "tp") or 1) * int(_get(values, "pp") or 1))


def _kv_shards(values: dict) -> int | None:
    tp = int(_get(values, "tp") or 1)
    pp = int(_get(values, "pp") or 1)
    kv_heads = _get(values, "num_kv_heads", "num_attn_heads")
    if kv_heads is None:
        return None
    count = int(kv_heads)
    if count <= 0 or (count % tp != 0 and tp % count != 0):
        return None
    return max(1, min(tp, count) * pp)


def _kv_token_demand(values: dict) -> float | None:
    explicit = _get(values, "kv_tokens_in_use", "active_kv_tokens")
    if explicit is not None:
        return float(explicit)
    workload_type = str(_get(values, "type", "workload_type") or "").lower()
    if workload_type == "online":
        concurrency = _get(
            values,
            "max_concurrent_streaming",
            "max_concurrent_requests",
            "max_num_seq",
        )
        isl = _get(values, "isl_token_avg", "input_len_tokens_avg")
        osl = _get(values, "osl_token_avg", "output_len_tokens_avg")
        if concurrency is not None and isl is not None and osl is not None:
            return float(concurrency) * (float(isl) + float(osl))
    max_seq = _get(values, "max_num_seq", "max_num_seqs")
    max_tokens = _get(values, "max_num_batched_tokens")
    isl = _get(values, "isl_token_avg", "input_len_tokens_avg")
    osl = _get(values, "osl_token_avg", "output_len_tokens_avg")
    if None in (max_seq, max_tokens, isl, osl):
        return None
    tokens_per_request = float(isl) + float(osl)
    if tokens_per_request <= 0:
        return None
    concurrency = min(float(max_seq), int(float(max_tokens) / tokens_per_request))
    return max(1.0, concurrency) * tokens_per_request
