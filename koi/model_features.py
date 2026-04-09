"""
koi/model_features.py — Model architecture features + derived hardware-aware metrics.

ModelFeatures captures everything about a model that affects placement decisions:
  Structural  : params, layers, heads, MoE config, vocab size, architecture family
  Dtype       : quantization / precision
  Derived     : model_size_gb, gqa_ratio, active_expert_ratio (computed post-init)

compute_config_features() adds hardware + parallelism derived features for a
specific (model, GPU, TP, PP, DP) combination:
  vram_headroom        : fraction of VRAM free for KV cache after weights
  bandwidth_per_param  : aggregate bandwidth ÷ params → decode speed proxy
  flops_per_param      : aggregate TFLOPS ÷ params → prefill speed proxy
  crosses_node_boundary: TP > gpus_per_node (adds inter-node latency)
  kv_heads_per_tp_shard: < 1 means KV heads are replicated across TP shards

get_model_features() is a stub for the HF model description API endpoint.
It falls back to name-based heuristics when the endpoint is unavailable.

TODO: Replace get_model_features() with the actual HF API endpoint call.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPE_BYTES: Dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8":  1.0,
    "int8": 1.0,
    "int4": 0.5,
}

ARCH_FAMILIES: Dict[str, list] = {
    "llama":     ["llama", "alpaca", "vicuna"],
    "qwen":      ["qwen"],
    "deepseek":  ["deepseek"],
    "mistral":   ["mistral", "mixtral"],
    "phi":       ["phi"],
    "gemma":     ["gemma"],
    "falcon":    ["falcon"],
    "mpt":       ["mpt"],
    "bloom":     ["bloom"],
    "gpt":       ["gpt", "opt"],
    "starcoder": ["starcoder", "codegen"],
}

# GPU specs — imported from physics.py (single source of truth)
from koi.tools.physics import GPU_SPECS as _GPU_SPECS


# ---------------------------------------------------------------------------
# ModelFeatures dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelFeatures:
    """
    Complete model architecture feature set for RAG embedding and LLM reasoning.

    All 30-70 placement-relevant variables are captured here or computed in
    compute_config_features(). LLMs receive this full set so they can reason
    about memory, bandwidth, and compute constraints from first principles.
    """
    model_name: str

    # --- Structural (from HF model config) ---
    num_params_billions: float          # total parameters (all experts for MoE)
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int                   # GQA: < num_attention_heads; MHA: equal
    vocab_size: int
    is_moe: bool = False
    num_experts: int = 0                # 0 if dense; total expert count for MoE
    active_experts: int = 0             # top-k routing count (0 if dense)
    architecture_family: str = "unknown"

    # --- Dtype ---
    dtype: str = "fp16"                 # "fp32", "fp16", "bf16", "fp8", "int8", "int4"

    # --- Derived (set in __post_init__) ---
    gqa_ratio: float = field(init=False)          # num_attention_heads / num_kv_heads
    active_expert_ratio: float = field(init=False) # active_experts / num_experts (1.0 if dense)
    model_size_gb: float = field(init=False)       # total weight memory footprint
    dtype_bytes: float = field(init=False)         # bytes per parameter

    def __post_init__(self):
        self.dtype_bytes = DTYPE_BYTES.get(self.dtype, 2.0)
        self.model_size_gb = self.num_params_billions * 1e9 * self.dtype_bytes / 1e9
        self.gqa_ratio = self.num_attention_heads / max(self.num_kv_heads, 1)
        if self.is_moe and self.num_experts > 0:
            self.active_expert_ratio = self.active_experts / self.num_experts
        else:
            self.active_expert_ratio = 1.0

    def to_embedding_text(self) -> str:
        """Rich physics-aware text for FAISS embedding queries."""
        arch = (
            f"MoE num_experts={self.num_experts} active={self.active_experts} "
            f"active_ratio={self.active_expert_ratio:.2f}"
            if self.is_moe else "dense"
        )
        return (
            f"model={self.model_name} family={self.architecture_family} "
            f"params={self.num_params_billions:.1f}B size={self.model_size_gb:.1f}GB "
            f"dtype={self.dtype} dtype_bytes={self.dtype_bytes} "
            f"layers={self.num_layers} hidden={self.hidden_dim} "
            f"attn_heads={self.num_attention_heads} kv_heads={self.num_kv_heads} "
            f"gqa_ratio={self.gqa_ratio:.1f} vocab={self.vocab_size} "
            f"architecture={arch}"
        )

    def to_llm_context(self) -> str:
        """Formatted string for injection into LLM prompts."""
        moe_str = (
            f"MoE: {self.num_experts} experts, {self.active_experts} active "
            f"(active_ratio={self.active_expert_ratio:.2f}, "
            f"effective_params={self.num_params_billions * self.active_expert_ratio:.1f}B active per token)"
            if self.is_moe else "Dense (all params active per token)"
        )
        return "\n".join([
            f"MODEL ARCHITECTURE — {self.model_name}:",
            f"  num_params_billions  : {self.num_params_billions:.1f}B",
            f"  architecture         : {moe_str}",
            f"  architecture_family  : {self.architecture_family}",
            f"  num_layers           : {self.num_layers}  ← PP must evenly divide this",
            f"  hidden_dim           : {self.hidden_dim}",
            f"  num_attention_heads  : {self.num_attention_heads}  ← TP must evenly divide this",
            f"  num_kv_heads         : {self.num_kv_heads}  ← TP must divide or TP ≤ num_kv_heads",
            f"  gqa_ratio            : {self.gqa_ratio:.1f}x  (1.0=MHA, >1=GQA, high GQA=less KV memory)",
            f"  vocab_size           : {self.vocab_size:,}",
            f"  dtype                : {self.dtype} ({self.dtype_bytes} bytes/param)",
            f"  model_size_gb        : {self.model_size_gb:.1f} GB total weight footprint",
        ])


# ---------------------------------------------------------------------------
# Config-level derived features
# ---------------------------------------------------------------------------

def compute_config_features(
    model: ModelFeatures,
    gpu_type: str,
    tp: int,
    pp: int,
    dp: int,
    input_len: int = 512,
    output_len: int = 128,
    gpus_per_node: int = 8,
    price_per_gpu_hour: float = 0.0,
    gpu_memory_gb_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute the full set of derived hardware+config features for a specific
    (model, GPU, TP, PP, DP) combination. These are the 'physics fingerprint'
    of a config — used for embedding, LLM context, and RAG similarity.

    Variables computed:
      params_per_gpu          : how much model each GPU holds
      weight_gb_per_gpu       : weight footprint per GPU
      vram_headroom           : fraction of VRAM free for KV cache + activations
      vram_headroom_gb        : absolute VRAM headroom in GB
      model_fits_single_gpu   : 1 if model fits on one GPU without TP
      bandwidth_per_param     : aggregate bandwidth / params → decode speed proxy
      flops_per_param         : aggregate TFLOPS / params → prefill speed proxy
      crosses_node_boundary   : 1 if TP spans multiple nodes (inter-node latency)
      kv_heads_per_tp_shard   : < 1 means KV heads are replicated
      total_cost_per_hour     : num_gpus_total × price_per_gpu_hour
      io_ratio                : input_len / output_len (prefill vs decode balance)
      total_context           : input_len + output_len
      roofline_decode_bound   : estimated decode throughput from bandwidth model
    """
    gpu_spec = _get_gpu_spec(gpu_type)
    bw_gbps = gpu_spec["bandwidth_gbps"]
    fp16_tflops = gpu_spec["fp16_tflops"]
    vram_gb = gpu_memory_gb_override or gpu_spec["mem_gb"]

    num_gpus_total = tp * pp * dp
    weight_gb_per_gpu = model.model_size_gb / max(tp * pp, 1)
    vram_headroom_gb = max(0.0, vram_gb - weight_gb_per_gpu)
    vram_headroom = vram_headroom_gb / max(vram_gb, 1)

    aggregate_bw = bw_gbps * tp
    aggregate_tflops = fp16_tflops * tp
    bandwidth_per_param = aggregate_bw / max(model.num_params_billions, 0.1)
    flops_per_param = aggregate_tflops / max(model.num_params_billions, 0.1)

    # Roofline decode estimate: tokens/sec ≈ bandwidth / bytes_to_load_weights
    # For MoE, only active experts are loaded per token
    effective_size_gb = model.model_size_gb * getattr(model, 'active_expert_ratio', 1.0)
    roofline_decode = (aggregate_bw / max(effective_size_gb, 0.1)) * 0.65  # 65% efficiency

    io_ratio = input_len / max(output_len, 1)
    total_context = input_len + output_len

    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "num_gpus_total": num_gpus_total,
        "gpu_type": gpu_type,
        "params_per_gpu_b": model.num_params_billions / max(tp * pp, 1),
        "weight_gb_per_gpu": weight_gb_per_gpu,
        "vram_headroom": vram_headroom,
        "vram_headroom_gb": vram_headroom_gb,
        "model_fits_single_gpu": int(model.model_size_gb < vram_gb),
        "bandwidth_per_param": bandwidth_per_param,
        "flops_per_param": flops_per_param,
        "roofline_decode_tps": roofline_decode,
        "crosses_node_boundary": int(tp > gpus_per_node),
        "kv_heads_per_tp_shard": model.num_kv_heads / max(tp, 1),
        "total_cost_per_hour": num_gpus_total * price_per_gpu_hour,
        "input_len": input_len,
        "output_len": output_len,
        "io_ratio": io_ratio,
        "total_context": total_context,
        "is_prefill_heavy": int(io_ratio > 2.0),
        "is_decode_heavy": int(io_ratio < 0.5),
    }


def config_features_to_llm_context(feats: Dict[str, Any]) -> str:
    """Format compute_config_features() output for LLM prompt injection."""
    vram_pct = (1.0 - feats["vram_headroom"]) * 100
    node_warn = " ⚠ CROSSES NODE BOUNDARY (inter-node latency applies)" if feats["crosses_node_boundary"] else ""
    kv_warn = " ⚠ KV HEADS REPLICATED (< 1 shard/head)" if feats["kv_heads_per_tp_shard"] < 1 else ""
    io_label = "prefill-heavy" if feats["is_prefill_heavy"] else ("decode-heavy" if feats["is_decode_heavy"] else "balanced")

    return "\n".join([
        f"CONFIG PHYSICS — TP={feats['tp']} PP={feats['pp']} DP={feats['dp']} on {feats['gpu_type']}:",
        f"  num_gpus_total       : {feats['num_gpus_total']}",
        f"  params_per_gpu       : {feats['params_per_gpu_b']:.1f}B",
        f"  weight_per_gpu       : {feats['weight_gb_per_gpu']:.1f} GB",
        f"  vram_usage           : {vram_pct:.0f}% of VRAM used by weights",
        f"  vram_headroom        : {feats['vram_headroom_gb']:.1f} GB free for KV cache + activations",
        f"  model_fits_single_gpu: {'yes' if feats['model_fits_single_gpu'] else 'NO — TP required'}",
        f"  bandwidth_per_param  : {feats['bandwidth_per_param']:.2f} GB/s per B params (decode proxy){node_warn}",
        f"  flops_per_param      : {feats['flops_per_param']:.2f} TFLOPS per B params (prefill proxy)",
        f"  roofline_decode_tps  : ~{feats['roofline_decode_tps']:.0f} tokens/s (bandwidth-bound estimate)",
        f"  crosses_node_boundary: {'YES' + node_warn if feats['crosses_node_boundary'] else 'no'}",
        f"  kv_heads_per_tp_shard: {feats['kv_heads_per_tp_shard']:.2f}{kv_warn}",
        f"  total_cost_per_hour  : ${feats['total_cost_per_hour']:.2f}/hr",
        f"  io_ratio             : {feats['io_ratio']:.2f}x ({io_label})",
        f"  total_context        : {feats['total_context']} tokens",
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_features(
    model_name: str,
    hf_description: Optional[str] = None,
    dtype: str = "fp16",
) -> ModelFeatures:
    """
    Get model architecture features.

    Priority:
      1. Known model registry (hardcoded for common models)
      2. Parse hf_description string — placeholder for HF API endpoint
      3. Heuristic inference from model name

    TODO: Replace step 2 with actual HF API endpoint call.
    The endpoint will return a JSON with num_hidden_layers, hidden_size,
    num_attention_heads, num_key_value_heads, etc. from the model card.
    """
    known = _KNOWN_MODELS.get(model_name) or _fuzzy_match(model_name)
    if known:
        return ModelFeatures(model_name=model_name, dtype=dtype, **known)

    if hf_description:
        parsed = _parse_hf_description(hf_description)
        if parsed:
            return ModelFeatures(model_name=model_name, dtype=dtype, **parsed)

    return _infer_from_name(model_name, dtype)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_gpu_spec(gpu_type: str) -> Dict[str, float]:
    """Case-insensitive GPU spec lookup with fallback."""
    for key, spec in _GPU_SPECS.items():
        if key.upper() in gpu_type.upper() or gpu_type.upper() in key.upper():
            return spec
    return {"bandwidth_gbps": 400.0, "fp16_tflops": 300.0, "mem_gb": 40.0}


_KNOWN_MODELS: Dict[str, Dict] = {
    "Qwen/Qwen2.5-72B-Instruct": dict(
        num_params_billions=72, num_layers=80, hidden_dim=8192,
        num_attention_heads=64, num_kv_heads=8, vocab_size=152064,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="qwen",
    ),
    "Qwen/Qwen2.5-72B": dict(
        num_params_billions=72, num_layers=80, hidden_dim=8192,
        num_attention_heads=64, num_kv_heads=8, vocab_size=152064,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="qwen",
    ),
    "Qwen/Qwen2.5-7B-Instruct": dict(
        num_params_billions=7.6, num_layers=28, hidden_dim=3584,
        num_attention_heads=28, num_kv_heads=4, vocab_size=152064,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="qwen",
    ),
    "Qwen/Qwen3-32B": dict(
        num_params_billions=32, num_layers=64, hidden_dim=5120,
        num_attention_heads=64, num_kv_heads=8, vocab_size=151936,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="qwen",
    ),
    "Qwen/Qwen3-235B-A22B": dict(
        num_params_billions=235, num_layers=94, hidden_dim=4096,
        num_attention_heads=64, num_kv_heads=4, vocab_size=151936,
        is_moe=True, num_experts=128, active_experts=8, architecture_family="qwen",
    ),
    "deepseek-ai/DeepSeek-R1": dict(
        num_params_billions=671, num_layers=61, hidden_dim=7168,
        num_attention_heads=128, num_kv_heads=128, vocab_size=129280,
        is_moe=True, num_experts=256, active_experts=8, architecture_family="deepseek",
    ),
    "deepseek-ai/DeepSeek-V3": dict(
        num_params_billions=671, num_layers=61, hidden_dim=7168,
        num_attention_heads=128, num_kv_heads=128, vocab_size=129280,
        is_moe=True, num_experts=256, active_experts=8, architecture_family="deepseek",
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": dict(
        num_params_billions=70, num_layers=80, hidden_dim=8192,
        num_attention_heads=64, num_kv_heads=8, vocab_size=128256,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="llama",
    ),
    "meta-llama/Llama-3-70B": dict(
        num_params_billions=70, num_layers=80, hidden_dim=8192,
        num_attention_heads=64, num_kv_heads=8, vocab_size=128256,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="llama",
    ),
    "meta-llama/Llama-3-8B": dict(
        num_params_billions=8, num_layers=32, hidden_dim=4096,
        num_attention_heads=32, num_kv_heads=8, vocab_size=128256,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="llama",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": dict(
        num_params_billions=8, num_layers=32, hidden_dim=4096,
        num_attention_heads=32, num_kv_heads=8, vocab_size=128256,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="llama",
    ),
    "meta-llama/Llama-3.1-70B-Instruct": dict(
        num_params_billions=70, num_layers=80, hidden_dim=8192,
        num_attention_heads=64, num_kv_heads=8, vocab_size=128256,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="llama",
    ),
    "meta-llama/Llama-3.1-405B-Instruct": dict(
        num_params_billions=405, num_layers=126, hidden_dim=16384,
        num_attention_heads=128, num_kv_heads=8, vocab_size=128256,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="llama",
    ),
    "meta-llama/Llama-3.3-70B-Instruct": dict(
        num_params_billions=70, num_layers=80, hidden_dim=8192,
        num_attention_heads=64, num_kv_heads=8, vocab_size=128256,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="llama",
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": dict(
        num_params_billions=46.7, num_layers=32, hidden_dim=4096,
        num_attention_heads=32, num_kv_heads=8, vocab_size=32000,
        is_moe=True, num_experts=8, active_experts=2, architecture_family="mistral",
    ),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": dict(
        num_params_billions=141, num_layers=56, hidden_dim=6144,
        num_attention_heads=48, num_kv_heads=8, vocab_size=32000,
        is_moe=True, num_experts=8, active_experts=2, architecture_family="mistral",
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": dict(
        num_params_billions=7.2, num_layers=32, hidden_dim=4096,
        num_attention_heads=32, num_kv_heads=8, vocab_size=32768,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="mistral",
    ),
    "microsoft/Phi-4": dict(
        num_params_billions=14, num_layers=40, hidden_dim=5120,
        num_attention_heads=40, num_kv_heads=10, vocab_size=100352,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="phi",
    ),
    "google/gemma-2-27b-it": dict(
        num_params_billions=27, num_layers=46, hidden_dim=4608,
        num_attention_heads=32, num_kv_heads=16, vocab_size=256000,
        is_moe=False, num_experts=0, active_experts=0, architecture_family="gemma",
    ),
}


def _fuzzy_match(model_name: str) -> Optional[Dict]:
    mn = model_name.lower()
    for key, spec in _KNOWN_MODELS.items():
        if mn in key.lower() or key.lower() in mn:
            return spec
    return None


def _parse_hf_description(description: str) -> Optional[Dict]:
    """
    Parse HF model description to extract architecture features.
    TODO: Replace with actual HF API JSON parsing.
    Currently a no-op placeholder.
    """
    return None


def _infer_from_name(model_name: str, dtype: str) -> ModelFeatures:
    """Infer architecture features from model name heuristics."""
    mn = model_name.lower()

    m = re.search(r"(\d+(?:\.\d+)?)b", mn, re.IGNORECASE)
    params = float(m.group(1)) if m else 7.0

    is_moe = any(x in mn for x in ["moe", "mixtral", "deepseek-v", "a22b", "a14b", "8x"])
    num_experts = 8 if is_moe else 0
    active_experts = 2 if is_moe else 0

    family = "unknown"
    for fam, patterns in ARCH_FAMILIES.items():
        if any(p in mn for p in patterns):
            family = fam
            break

    if params >= 400:
        num_layers, hidden_dim, num_heads, num_kv = 126, 16384, 128, 8
    elif params >= 200:
        num_layers, hidden_dim, num_heads, num_kv = 94, 7168, 64, 4
    elif params >= 70:
        num_layers, hidden_dim, num_heads, num_kv = 80, 8192, 64, 8
    elif params >= 30:
        num_layers, hidden_dim, num_heads, num_kv = 64, 5120, 40, 8
    elif params >= 13:
        num_layers, hidden_dim, num_heads, num_kv = 40, 5120, 40, 8
    elif params >= 7:
        num_layers, hidden_dim, num_heads, num_kv = 32, 4096, 32, 8
    else:
        num_layers, hidden_dim, num_heads, num_kv = 28, 3072, 24, 8

    return ModelFeatures(
        model_name=model_name,
        num_params_billions=params,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv,
        vocab_size=128256,
        is_moe=is_moe,
        num_experts=num_experts,
        active_experts=active_experts,
        architecture_family=family,
        dtype=dtype,
    )
