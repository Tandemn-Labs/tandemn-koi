"""
koi/tools/physics.py — GPU specs, roofline analysis, physics-vector similarity.

Ported from v1 model_features.py with additions:
- Physics-vector similarity search for unknown models
- get_gpu_physics() / get_model_arch() / find_similar_models() agent tools
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPE_BYTES: Dict[str, float] = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0,
    "fp8": 1.0, "int8": 1.0, "int4": 0.5,
}

# GPU specs — FP16 TFLOPS are dense tensor core WITHOUT sparsity.
# Memory is total VRAM (not "usable after reserved").
# Sources: NVIDIA datasheets (nvidia.com/data-center/*)
GPU_SPECS: Dict[str, Dict[str, float]] = {
    "H100_SXM":  {"bandwidth_gbps": 3350, "fp16_tflops": 989,  "mem_gb": 80.0, "interconnect": "NVLink", "generation": "Hopper", "fp8_native": True},
    "H100":      {"bandwidth_gbps": 2000, "fp16_tflops": 756,  "mem_gb": 80.0, "interconnect": "PCIe",   "generation": "Hopper", "fp8_native": True},   # PCIe, NOT SXM
    "H200":      {"bandwidth_gbps": 4800, "fp16_tflops": 1979, "mem_gb": 141.0, "interconnect": "NVLink", "generation": "Hopper", "fp8_native": True},
    "A100-80GB": {"bandwidth_gbps": 2039, "fp16_tflops": 312,  "mem_gb": 80.0, "interconnect": "NVLink", "generation": "Ampere", "fp8_native": False},
    "A100-40GB": {"bandwidth_gbps": 1555, "fp16_tflops": 312,  "mem_gb": 40.0, "interconnect": "NVLink", "generation": "Ampere", "fp8_native": False},
    "A100":      {"bandwidth_gbps": 2039, "fp16_tflops": 312,  "mem_gb": 80.0, "interconnect": "NVLink", "generation": "Ampere", "fp8_native": False},  # defaults to 80GB SXM
    "L40S":      {"bandwidth_gbps":  864, "fp16_tflops": 362,  "mem_gb": 48.0, "interconnect": "PCIe",   "generation": "Ada",    "fp8_native": True},   # 733 with sparsity
    "A10G":      {"bandwidth_gbps":  600, "fp16_tflops": 125,  "mem_gb": 24.0, "interconnect": "PCIe",   "generation": "Ampere", "fp8_native": False},
    "L4":        {"bandwidth_gbps":  300, "fp16_tflops": 121,  "mem_gb": 24.0, "interconnect": "PCIe",   "generation": "Ada",    "fp8_native": True},
    "B200":      {"bandwidth_gbps": 8000, "fp16_tflops": 2250, "mem_gb": 192.0, "interconnect": "NVLink", "generation": "Blackwell", "fp8_native": True},
    "GB200":     {"bandwidth_gbps": 8000, "fp16_tflops": 2250, "mem_gb": 192.0, "interconnect": "NVLink", "generation": "Blackwell", "fp8_native": True},
}

ARCH_FAMILIES: Dict[str, list] = {
    "llama": ["llama", "alpaca", "vicuna"],
    "qwen": ["qwen"],
    "deepseek": ["deepseek"],
    "mistral": ["mistral", "mixtral"],
    "phi": ["phi"],
    "gemma": ["gemma"],
}


# ---------------------------------------------------------------------------
# ModelFeatures (ported from v1)
# ---------------------------------------------------------------------------

@dataclass
class ModelFeatures:
    """Model architecture features for placement reasoning."""
    model_name: str
    num_params_billions: float
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    is_moe: bool = False
    num_experts: int = 0
    active_experts: int = 0
    architecture_family: str = "unknown"
    dtype: str = "fp16"

    # Derived
    gqa_ratio: float = field(init=False)
    model_size_gb: float = field(init=False)
    dtype_bytes: float = field(init=False)

    def __post_init__(self):
        self.dtype_bytes = DTYPE_BYTES.get(self.dtype, 2.0)
        self.model_size_gb = self.num_params_billions * 1e9 * self.dtype_bytes / 1e9
        self.gqa_ratio = self.num_attention_heads / max(self.num_kv_heads, 1)


# Known models — imported from model_features.py (single source of truth)
from koi.model_features import _KNOWN_MODELS


# ---------------------------------------------------------------------------
# GPU spec lookup
# ---------------------------------------------------------------------------

def lookup_gpu_spec(gpu_type: str) -> Dict[str, Any]:
    """Case-insensitive GPU spec lookup. Specific keys before generic."""
    gpu_upper = gpu_type.upper()
    for key, spec in GPU_SPECS.items():
        if key.upper() == gpu_upper:
            return spec
    # Fuzzy fallback
    for key, spec in GPU_SPECS.items():
        if key.upper() in gpu_upper or gpu_upper in key.upper():
            return spec
    return {"bandwidth_gbps": 400.0, "fp16_tflops": 300.0, "mem_gb": 40.0,
            "interconnect": "PCIe", "generation": "unknown", "fp8_native": False}


# ---------------------------------------------------------------------------
# Model architecture lookup
# ---------------------------------------------------------------------------

def get_model_features(model_name: str, dtype: str = "fp16") -> ModelFeatures:
    """
    Get model architecture. Priority:
      1. Known models registry (instant, no network)
      2. HuggingFace Hub config.json fetch (requires network)
      3. Fallback: Llama-70B template (last resort, low confidence)
    """
    # 1. Known registry (exact + fuzzy)
    known = _KNOWN_MODELS.get(model_name)
    if not known:
        mn = model_name.lower()
        for key, spec in _KNOWN_MODELS.items():
            if mn in key.lower() or key.lower() in mn:
                known = spec
                break
    if known:
        return ModelFeatures(model_name=model_name, dtype=dtype, **known)

    # 2. HuggingFace Hub fetch
    hf_result = _fetch_hf_config(model_name, dtype)
    if hf_result:
        return hf_result

    # 3. Last resort: infer from name (UNRELIABLE — logs warning)
    import logging
    logging.getLogger("koi.physics").warning(
        f"Model '{model_name}' not in registry and HF fetch failed. "
        f"Using heuristic fallback — architecture features may be inaccurate."
    )
    return _infer_from_name(model_name, dtype)


def _fetch_hf_config(model_name: str, dtype: str = "fp16") -> Optional[ModelFeatures]:
    """Fetch config.json from HuggingFace Hub and parse architecture."""
    import os
    try:
        import urllib.request
        import json
        hf_token = os.environ.get("HF_TOKEN", "")
        url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            cfg = json.loads(resp.read())

        num_layers = cfg.get("num_hidden_layers", 32)
        hidden_dim = cfg.get("hidden_size", 4096)
        num_heads = cfg.get("num_attention_heads", 32)
        num_kv = cfg.get("num_key_value_heads", num_heads)
        vocab_size = cfg.get("vocab_size", 32000)
        num_experts = cfg.get("num_experts", cfg.get("num_local_experts", 0))
        active_experts = cfg.get("num_experts_per_tok", cfg.get("num_selected_experts", 0))
        is_moe = num_experts > 1

        # Compute params from architecture
        intermediate = cfg.get("intermediate_size", hidden_dim * 4)
        d_head = hidden_dim / max(num_heads, 1)
        kv_dim = num_kv * d_head
        qo = 2 * hidden_dim * hidden_dim
        kv = 2 * hidden_dim * kv_dim
        if is_moe and num_experts > 0:
            moe_intermediate = cfg.get("moe_intermediate_size", intermediate)
            ffn = 3 * hidden_dim * moe_intermediate * num_experts
        else:
            ffn = 3 * hidden_dim * intermediate
        layer_p = qo + kv + ffn
        embed_p = vocab_size * hidden_dim * 2
        params_b = (embed_p + num_layers * layer_p) / 1e9

        family = "unknown"
        archs = cfg.get("architectures", [])
        if archs:
            arch_lower = archs[0].lower()
            for fam, patterns in ARCH_FAMILIES.items():
                if any(p in arch_lower for p in patterns):
                    family = fam
                    break

        return ModelFeatures(
            model_name=model_name,
            num_params_billions=round(params_b, 2),
            num_layers=num_layers, hidden_dim=hidden_dim,
            num_attention_heads=num_heads, num_kv_heads=num_kv,
            vocab_size=vocab_size, is_moe=is_moe,
            num_experts=num_experts, active_experts=active_experts,
            architecture_family=family, dtype=dtype,
        )
    except Exception:
        return None


def _infer_from_name(model_name: str, dtype: str) -> ModelFeatures:
    """
    LAST RESORT fallback. Extracts param count from name, uses Llama-class
    template for architecture. Results are approximate — the agent should
    treat these with low confidence.
    """
    mn = model_name.lower()
    m = re.search(r"(\d+(?:\.\d+)?)b", mn, re.IGNORECASE)
    params = float(m.group(1)) if m else 7.0

    is_moe = any(x in mn for x in ["moe", "mixtral", "deepseek-v", "a22b", "a14b", "8x"])

    family = "unknown"
    for fam, patterns in ARCH_FAMILIES.items():
        if any(p in mn for p in patterns):
            family = fam
            break

    # Use closest known model as template instead of hardcoded tables
    best_match = None
    best_diff = float("inf")
    for key, spec in _KNOWN_MODELS.items():
        diff = abs(spec["num_params_billions"] - params)
        if spec.get("is_moe", False) == is_moe and diff < best_diff:
            best_diff = diff
            best_match = spec

    if best_match:
        return ModelFeatures(
            model_name=model_name, dtype=dtype,
            num_params_billions=params,
            num_layers=best_match["num_layers"],
            hidden_dim=best_match["hidden_dim"],
            num_attention_heads=best_match["num_attention_heads"],
            num_kv_heads=best_match["num_kv_heads"],
            vocab_size=best_match["vocab_size"],
            is_moe=is_moe,
            num_experts=best_match.get("num_experts", 0),
            active_experts=best_match.get("active_experts", 0),
            architecture_family=family,
        )

    # Absolute fallback: Llama-70B template
    return ModelFeatures(
        model_name=model_name, num_params_billions=params,
        num_layers=80, hidden_dim=8192,
        num_attention_heads=64, num_kv_heads=8,
        vocab_size=128256, is_moe=is_moe,
        num_experts=8 if is_moe else 0,
        active_experts=2 if is_moe else 0,
        architecture_family=family, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Physics-vector similarity search
# ---------------------------------------------------------------------------

_SIMILARITY_WEIGHTS = {
    "model_size_gb": 0.35,
    "kv_bytes_per_token": 0.25,
    "flops_per_fwd": 0.15,
    "gqa_ratio": 0.10,
    "num_layers": 0.05,
    "num_attention_heads": 0.05,
    "is_moe": 0.05,
}


def compute_physics_vector(model: ModelFeatures) -> Dict[str, float]:
    """Compute the 7-feature physics vector for similarity search."""
    head_dim = model.hidden_dim / max(model.num_attention_heads, 1)
    kv_dim = model.num_kv_heads * head_dim

    # KV cache bytes per token per layer: 2 (K+V) * kv_dim * dtype_bytes
    kv_bytes_per_token = 2 * model.num_layers * kv_dim * model.dtype_bytes

    # Rough FLOPs per forward pass (single token decode)
    # Attention projections: 4 * hidden^2 per layer (Q, K, V, O)
    # FFN: 3 * hidden * intermediate (≈ 4*hidden) per layer
    flops_per_layer = 4 * model.hidden_dim ** 2 + 3 * model.hidden_dim * (model.hidden_dim * 4)
    flops_per_fwd = flops_per_layer * model.num_layers

    return {
        "model_size_gb": model.model_size_gb,
        "kv_bytes_per_token": kv_bytes_per_token,
        "flops_per_fwd": flops_per_fwd,
        "gqa_ratio": model.gqa_ratio,
        "num_layers": float(model.num_layers),
        "num_attention_heads": float(model.num_attention_heads),
        "is_moe": float(model.is_moe),
    }


def physics_distance(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Weighted L1 distance between two physics vectors. 0 = identical."""
    dist = 0.0
    for key, weight in _SIMILARITY_WEIGHTS.items():
        a = v1.get(key, 0.0)
        b = v2.get(key, 0.0)
        denom = max(abs(a), abs(b), 1e-9)
        dist += weight * abs(a - b) / denom
    return dist


def find_similar_models(
    target_model: ModelFeatures,
    perfdb_models: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Find PerfDB models most similar to the target by physics-vector distance.

    Args:
        target_model: ModelFeatures for the unknown model
        perfdb_models: list of dicts with keys: model_name, num_params_billions,
                       num_layers, hidden_dim, num_attention_heads, num_kv_heads,
                       vocab_size, is_moe, records_count

    Returns:
        Sorted list of {model_name, distance, confidence, records_count}
    """
    target_vec = compute_physics_vector(target_model)
    results = []

    for pm in perfdb_models:
        proxy = ModelFeatures(
            model_name=pm["model_name"],
            num_params_billions=pm.get("num_params_billions", 7.0),
            num_layers=pm.get("num_layers", 32),
            hidden_dim=pm.get("hidden_dim", 4096),
            num_attention_heads=pm.get("num_attention_heads", 32),
            num_kv_heads=pm.get("num_kv_heads", 8),
            vocab_size=pm.get("vocab_size", 128256),
            is_moe=pm.get("is_moe", False),
        )
        proxy_vec = compute_physics_vector(proxy)
        dist = physics_distance(target_vec, proxy_vec)
        confidence = max(0.0, min(1.0, 1.0 - dist))

        results.append({
            "model_name": pm["model_name"],
            "distance": round(dist, 4),
            "confidence": round(confidence, 4),
            "records_count": pm.get("records_count", 0),
        })

    results.sort(key=lambda x: x["distance"])
    return results


# ---------------------------------------------------------------------------
# Agent tool functions
# ---------------------------------------------------------------------------

def get_gpu_physics(gpu_type: str, model_name: Optional[str] = None) -> str:
    """Get GPU hardware specs. If model_name provided, includes per-model analysis."""
    spec = lookup_gpu_spec(gpu_type)
    lines = [
        f"GPU: {gpu_type}",
        f"  VRAM: {spec['mem_gb']} GB",
        f"  Bandwidth: {spec['bandwidth_gbps']} GB/s",
        f"  FP16 TFLOPS: {spec['fp16_tflops']}",
        f"  Interconnect: {spec.get('interconnect', 'unknown')}",
        f"  Generation: {spec.get('generation', 'unknown')}",
        f"  FP8 native: {spec.get('fp8_native', False)}",
    ]

    if model_name:
        mf = get_model_features(model_name)
        for tp in [1, 2, 4, 8]:
            weight_per_gpu = mf.model_size_gb / max(tp, 1)
            headroom = spec["mem_gb"] - weight_per_gpu
            bw_per_param = (spec["bandwidth_gbps"] * tp) / max(mf.num_params_billions, 0.1)
            fits = "YES" if headroom >= 8.0 else "NO (OOM)"
            roofline = (spec["bandwidth_gbps"] * tp / max(mf.model_size_gb, 0.1)) * 0.65
            lines.append(
                f"  TP={tp}: weight/GPU={weight_per_gpu:.0f}GB, "
                f"headroom={headroom:.0f}GB, fits={fits}, "
                f"bw/param={bw_per_param:.1f}, roofline~{roofline:.0f} tok/s"
            )

    return "\n".join(lines)


def get_model_arch(model_name: str) -> str:
    """Get model architecture features."""
    mf = get_model_features(model_name)
    moe_str = f"MoE ({mf.num_experts} experts, {mf.active_experts} active)" if mf.is_moe else "Dense"
    return "\n".join([
        f"Model: {mf.model_name}",
        f"  Params: {mf.num_params_billions:.1f}B ({moe_str})",
        f"  Layers: {mf.num_layers} (valid PP: {[p for p in [1,2,4,8] if mf.num_layers % p == 0]})",
        f"  Heads: {mf.num_attention_heads} attn, {mf.num_kv_heads} KV (GQA={mf.gqa_ratio:.0f}x)",
        f"  Hidden: {mf.hidden_dim} (valid TP: {[t for t in [1,2,4,8,16] if mf.num_attention_heads % t == 0]})",
        f"  Vocab: {mf.vocab_size:,}",
        f"  Size: {mf.model_size_gb:.1f} GB at {mf.dtype}",
        f"  Family: {mf.architecture_family}",
    ])
