"""
koi/oracle.py — The Oracle: performance DB + feasibility pruning + interpolation.

This is the pure-numerical layer. Zero LLM calls here.
The Oracle runs BEFORE the ensemble so LLMs receive a pre-pruned, pre-scored
candidate list rather than reasoning from scratch about hardware constraints.

Interpolation layers (applied in order, first hit wins):
  1. Exact match       — same (model, gpu, tp, pp) + closest I/O length in DB
  2. Interpolated      — same (gpu, tp, pp), scale throughput by I/O length ratio
  3. Cross-GPU         — known DB GPU → target GPU, scale by bandwidth/compute ratios
  4. Analytical        — pure roofline model, no nearby data needed
  5. VPC-corrected     — apply per-VPC learned delta on top of any of the above
"""

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from koi.schemas import (
    DataSource,
    EngineConfig,
    GPUResource,
    JobRequest,
    OracleCandidate,
    PlacementConfig,
    PredictedMetrics,
    ResourceMap,
    TaskType,
)

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

GPU_SPECS: Dict[str, Dict[str, float]] = {
    # bandwidth_gbps: memory bandwidth
    # fp16_tflops:    tensor core fp16 TFLOPS
    # mem_gb:         usable VRAM (slightly under physical due to OS reservation)
    "H100_SXM":  {"bandwidth_gbps": 3350, "fp16_tflops": 989, "mem_gb": 79.0,  "generation": "Hopper"},
    "H100":      {"bandwidth_gbps": 3350, "fp16_tflops": 989, "mem_gb": 79.0,  "generation": "Hopper"},
    "H200":      {"bandwidth_gbps": 4800, "fp16_tflops": 989, "mem_gb": 140.0, "generation": "Hopper"},
    "A100":      {"bandwidth_gbps": 2000, "fp16_tflops": 312, "mem_gb": 79.0,  "generation": "Ampere"},
    "L40S":      {"bandwidth_gbps":  864, "fp16_tflops": 733, "mem_gb": 45.5,  "generation": "Ada"},
    "A10G":      {"bandwidth_gbps":  600, "fp16_tflops": 125, "mem_gb": 23.0,  "generation": "Ampere"},
    "L4":        {"bandwidth_gbps":  300, "fp16_tflops": 121, "mem_gb": 23.0,  "generation": "Ada"},
}

# AWS instance → GPU type mapping
INSTANCE_TO_GPU: Dict[str, str] = {
    "g6e.xlarge":    "L40S",
    "g6e.2xlarge":   "L40S",
    "g6e.4xlarge":   "L40S",
    "g6e.8xlarge":   "L40S",
    "g6e.12xlarge":  "L40S",
    "g6e.16xlarge":  "L40S",
    "g6e.24xlarge":  "L40S",
    "g6e.48xlarge":  "L40S",
    "p4d.24xlarge":  "A100",
    "p4de.24xlarge": "A100",
    "p5.48xlarge":   "H100",
    "p5e.48xlarge":  "H100",
    "g5.xlarge":     "A10G",
    "g5.2xlarge":    "A10G",
    "g5.4xlarge":    "A10G",
    "g5.8xlarge":    "A10G",
    "g5.12xlarge":   "A10G",
    "g5.16xlarge":   "A10G",
    "g5.48xlarge":   "A10G",
}

# Known model architectural specs needed for feasibility checks
# (TP must evenly divide num_attention_heads AND num_kv_heads)
# (PP must evenly divide num_layers)
MODEL_ARCH: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen2.5-72B-Instruct": {
        "params_billion": 72, "num_layers": 80,
        "num_attention_heads": 64, "num_kv_heads": 8,
        "is_moe": False,
    },
    "Qwen/Qwen2.5-72B": {
        "params_billion": 72, "num_layers": 80,
        "num_attention_heads": 64, "num_kv_heads": 8,
        "is_moe": False,
    },
    "Qwen/Qwen3-32B": {
        "params_billion": 32, "num_layers": 64,
        "num_attention_heads": 64, "num_kv_heads": 8,
        "is_moe": False,
    },
    "Qwen/Qwen3-235B-A22B": {
        "params_billion": 235, "active_billion": 22,
        "num_layers": 94, "num_attention_heads": 64, "num_kv_heads": 4,
        "is_moe": True, "num_experts": 128, "num_active_experts": 8,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
        "params_billion": 70, "num_layers": 80,
        "num_attention_heads": 64, "num_kv_heads": 8,
        "is_moe": False,
    },
    "meta-llama/Llama-3-70B": {
        "params_billion": 70, "num_layers": 80,
        "num_attention_heads": 64, "num_kv_heads": 8,
        "is_moe": False,
    },
    "meta-llama/Llama-3-8B": {
        "params_billion": 8, "num_layers": 32,
        "num_attention_heads": 32, "num_kv_heads": 8,
        "is_moe": False,
    },
}

BYTES_PER_PARAM = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}

# ---------------------------------------------------------------------------
# Internal perf DB record
# ---------------------------------------------------------------------------

@dataclass
class PerfEntry:
    """Normalized performance DB record."""
    model_name: str
    gpu_type: str
    tp: int
    pp: int
    dp: int
    input_len: int
    output_len: int
    concurrency: int             # number of concurrent requests at benchmark time
    tokens_per_sec_total: float  # total throughput (across all DP replicas)
    num_gpus: int
    cost_per_hour_usd: float     # total cost per hour for this config
    data_source_name: str        # "our_experiment", "dynamo_swept", etc.
    tpot_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    instance_type: str = ""

    @property
    def tokens_per_sec_per_gpu(self) -> float:
        return self.tokens_per_sec_total / max(self.num_gpus, 1)

    @property
    def cost_per_gpu_hour(self) -> float:
        return self.cost_per_hour_usd / max(self.num_gpus, 1)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class Oracle:
    """
    Numerical performance prediction engine.

    Usage:
        oracle = Oracle(perfdb_path="./perfdb")
        candidates = oracle.get_candidates(request, resource_map)
        # returns List[OracleCandidate] sorted by cost
    """

    def __init__(self, perfdb_path: str = "./perfdb"):
        self.perfdb_path = Path(perfdb_path)
        self.entries: List[PerfEntry] = []
        self._load_perf_db()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_perf_db(self) -> None:
        """Load all perf data from perfdb/ directory."""
        loaded = 0

        # 1. Try canonical data.csv (schema.md format)
        csv_path = self.perfdb_path / "data.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
                for _, row in df.iterrows():
                    entry = self._parse_schema_csv_row(row)
                    if entry:
                        self.entries.append(entry)
                        loaded += 1
                print(f"[Oracle] Loaded {loaded} entries from data.csv")
            except Exception as e:
                print(f"[Oracle] Warning: could not load data.csv: {e}")

        # 2. Load results.json files (our experiment format)
        for json_file in self.perfdb_path.glob("**/*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        entry = self._parse_results_json_item(item)
                        if entry:
                            self.entries.append(entry)
                            loaded += 1
                elif isinstance(data, dict):
                    entry = self._parse_results_json_item(data)
                    if entry:
                        self.entries.append(entry)
                        loaded += 1
            except Exception as e:
                print(f"[Oracle] Warning: could not load {json_file}: {e}")

        # 3. Load root-level results.json if perfdb is empty
        root_results = Path("results.json")
        if not self.entries and root_results.exists():
            try:
                with open(root_results) as f:
                    data = json.load(f)
                for item in data:
                    entry = self._parse_results_json_item(item)
                    if entry:
                        self.entries.append(entry)
                print(f"[Oracle] Loaded {len(self.entries)} entries from root results.json")
            except Exception as e:
                print(f"[Oracle] Warning: could not load results.json: {e}")

        print(f"[Oracle] Total perf DB entries: {len(self.entries)}")
        if self.entries:
            gpu_types = set(e.gpu_type for e in self.entries)
            models = set(e.model_name for e in self.entries)
            print(f"[Oracle] GPU types: {gpu_types}")
            print(f"[Oracle] Models: {models}")

    def _parse_results_json_item(self, item: Dict) -> Optional[PerfEntry]:
        """Parse a results.json-style benchmark record."""
        try:
            # Extract GPU type from instance_type string
            instance_type_raw = item.get("instance_type", "")
            # "4x g6e.12xlarge" → "g6e.12xlarge"
            instance_type = re.sub(r"^\d+x\s*", "", instance_type_raw).strip()
            gpu_type = INSTANCE_TO_GPU.get(instance_type, "")
            if not gpu_type:
                # try matching any key that appears in instance_type_raw
                for k, v in INSTANCE_TO_GPU.items():
                    if k in instance_type_raw:
                        gpu_type = v
                        instance_type = k
                        break
            if not gpu_type:
                return None

            total_gpus = int(item.get("total_gpus", item.get("gpus_per_node", 1)))
            price_per_hour = float(item.get("price_per_hour", 0.0))

            return PerfEntry(
                model_name=item.get("model", item.get("model_name", "")),
                gpu_type=gpu_type,
                tp=int(item.get("tp", 1)),
                pp=int(item.get("pp", 1)),
                dp=int(item.get("dp", 1)),
                input_len=int(item.get("max_input_length", item.get("input_len_tokens_fixed", 128))),
                output_len=int(item.get("max_output_length", item.get("output_len_tokens_fixed", 128))),
                concurrency=int(item.get("benchmark_target_concurrency", 1)),
                tokens_per_sec_total=float(item.get("total_tokens_per_sec", item.get("tokens_per_sec_total", 0))),
                num_gpus=total_gpus,
                cost_per_hour_usd=price_per_hour,
                data_source_name="our_experiment",
                tpot_ms=item.get("tpot_ms_p50"),
                ttft_ms=item.get("ttft_ms_p50"),
                instance_type=instance_type,
            )
        except (KeyError, ValueError, TypeError):
            return None

    def _parse_schema_csv_row(self, row) -> Optional[PerfEntry]:
        """Parse a row from the canonical data.csv."""
        try:
            tps = float(row.get("tokens_per_sec_total", 0) or 0)
            if tps <= 0:
                return None
            return PerfEntry(
                model_name=str(row.get("model_name", "")),
                gpu_type=str(row.get("gpu_model", "")),
                tp=int(row.get("tp", 1) or 1),
                pp=int(row.get("pp", 1) or 1),
                dp=int(row.get("dp", 1) or 1),
                input_len=int(float(row.get("input_len_tokens_fixed", 128) or 128)),
                output_len=int(float(row.get("output_len_tokens_fixed", 128) or 128)),
                concurrency=int(float(row.get("max_num_seqs", 1) or 1)),
                tokens_per_sec_total=tps,
                num_gpus=int(float(row.get("gpu_count_total", 1) or 1)),
                cost_per_hour_usd=float(row.get("price_per_instance_hour_usd", 0) or 0),
                data_source_name=str(row.get("data_source", "unknown")),
                tpot_ms=float(row["tpot_ms_p50"]) if row.get("tpot_ms_p50") else None,
                ttft_ms=float(row["ttft_ms_p50"]) if row.get("ttft_ms_p50") else None,
                instance_type=str(row.get("instance_type", "")),
            )
        except (KeyError, ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Model architecture helpers
    # ------------------------------------------------------------------

    def get_model_spec(self, model_name: str) -> Dict[str, Any]:
        """Return architectural spec for a model, with fallbacks for unknown models."""
        if model_name in MODEL_ARCH:
            return MODEL_ARCH[model_name]

        # Fuzzy match (e.g. "Qwen2.5-72B-Instruct" matches "Qwen/Qwen2.5-72B-Instruct")
        for key, spec in MODEL_ARCH.items():
            if model_name.lower() in key.lower() or key.lower() in model_name.lower():
                return spec

        # Extract param count from name heuristic
        match = re.search(r"(\d+(?:\.\d+)?)B", model_name, re.IGNORECASE)
        params = float(match.group(1)) if match else 7.0
        is_moe = any(x in model_name.lower() for x in ["moe", "mixtral", "deepseek-v", "a22b"])

        # Reasonable defaults based on param count
        if params >= 70:
            num_layers, num_heads, num_kv = 80, 64, 8
        elif params >= 30:
            num_layers, num_heads, num_kv = 64, 40, 8
        elif params >= 13:
            num_layers, num_heads, num_kv = 40, 40, 8
        else:
            num_layers, num_heads, num_kv = 32, 32, 8

        return {
            "params_billion": params,
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_kv_heads": num_kv,
            "is_moe": is_moe,
        }

    # ------------------------------------------------------------------
    # Feasibility checks
    # ------------------------------------------------------------------

    def _model_weight_gb(self, params_billion: float, precision: str = "fp16") -> float:
        bpp = BYTES_PER_PARAM.get(precision, 2)
        return params_billion * 1e9 * bpp / 1e9  # in GB

    def _check_memory(
        self, model_name: str, gpu: GPUResource, tp: int, precision: str = "fp16"
    ) -> Tuple[bool, float, str]:
        """
        Returns (feasible, vram_headroom_gb, note).
        Headroom = per-GPU VRAM minus per-GPU weight footprint.
        We need at least 10GB headroom for KV cache + activations.
        """
        spec = self.get_model_spec(model_name)
        params = spec["params_billion"]

        # For MoE: ALL expert weights must be loaded even though only k are active
        weight_gb = self._model_weight_gb(params, precision)
        weight_per_gpu = weight_gb / tp
        headroom = gpu.gpu_memory_gb - weight_per_gpu

        if headroom < 8:
            return False, headroom, f"Only {headroom:.1f}GB headroom after weights ({weight_per_gpu:.1f}GB/GPU)"
        return True, headroom, f"{headroom:.1f}GB headroom"

    def _check_parallelism(
        self, model_name: str, tp: int, pp: int
    ) -> Tuple[bool, str]:
        """TP must divide num_attention_heads and num_kv_heads. PP must divide num_layers."""
        spec = self.get_model_spec(model_name)
        num_heads = spec.get("num_attention_heads", 32)
        num_kv = spec.get("num_kv_heads", num_heads)
        num_layers = spec.get("num_layers", 32)

        if num_heads % tp != 0:
            return False, f"TP={tp} does not divide num_attention_heads={num_heads}"
        if num_kv % tp != 0 and tp > num_kv:
            # vLLM can replicate KV heads when tp > num_kv_heads, but warn
            pass
        if num_layers % pp != 0:
            return False, f"PP={pp} does not divide num_layers={num_layers}"
        return True, "OK"

    # ------------------------------------------------------------------
    # Throughput prediction / interpolation
    # ------------------------------------------------------------------

    def _io_work_units(self, input_len: int, output_len: int) -> float:
        """
        Compute a 'work units' scalar for I/O length scaling.

        For a typical mixed decode-bound workload:
        - Prefill is fast (parallelized across all tokens at once) → weight 0.3
        - Decode is slow (sequential, memory bandwidth bound) → weight 1.0
        """
        return 0.3 * input_len + 1.0 * output_len

    def _scale_throughput_for_io(
        self,
        base_tps: float,
        base_input: int,
        base_output: int,
        target_input: int,
        target_output: int,
    ) -> float:
        """
        Scale a known throughput (tokens/sec) to a different I/O length.
        Longer sequences → more work per request → lower requests/sec but
        similar total tokens/sec (memory bandwidth bound in decode).

        In practice, total token throughput drops ~20-40% when sequences are much longer
        due to KV cache memory pressure and prefill overhead.
        We apply a mild correction bounded to [0.5, 1.3].
        """
        base_work = self._io_work_units(base_input, base_output)
        target_work = self._io_work_units(target_input, target_output)
        if base_work <= 0:
            return base_tps

        # Scale: longer work → lower requests/sec → lower total tokens/sec
        # But total token throughput doesn't drop linearly since tokens-per-request also increases
        # Net effect: tokens/sec ≈ constant for bandwidth-bound, decreases for compute-bound
        # Using a dampened scale factor with square root to model the partial cancellation
        ratio = base_work / target_work
        # Apply dampened correction: sqrt of the raw ratio
        scale = math.pow(ratio, 0.5)
        scale = max(0.5, min(1.4, scale))
        return base_tps * scale

    def _analytical_throughput(
        self,
        model_name: str,
        gpu: GPUResource,
        tp: int,
        pp: int,
        input_len: int,
        output_len: int,
    ) -> Tuple[float, float]:
        """
        Pure roofline-based throughput estimate. Returns (tokens_per_sec, confidence).

        For decode-bound workloads (typical for generation):
        tokens_per_sec ≈ bandwidth_gbps × tp / (params_billion × bytes_per_param × pp_efficiency)

        This is the arithmetic intensity crossover: if the model is bandwidth-bound (most are
        at batch_size=1), throughput = memory_bandwidth / bytes_to_load_all_weights.
        """
        spec = self.get_model_spec(model_name)
        params_b = spec["params_billion"]
        gpu_spec = GPU_SPECS.get(gpu.gpu_type, {})
        bw_gbps = gpu_spec.get("bandwidth_gbps", 400) * tp  # aggregate TP bandwidth

        # Decode bandwidth-bound estimate
        model_gb = self._model_weight_gb(params_b, "fp16")
        # tokens/sec per DP replica ≈ bandwidth / bytes_to_load_weights_per_token
        tps_decode = (bw_gbps / model_gb) * 0.65  # 0.65 efficiency factor

        # Pipeline overhead: PP adds bubble fraction ~ 1/(1 + 1/pp) roughly
        pp_efficiency = 1.0 - (pp - 1) / (pp * 8)  # rough bubble estimate
        tps_decode *= pp_efficiency

        # I/O length correction
        tps = self._scale_throughput_for_io(tps_decode, 128, 128, input_len, output_len)

        # Confidence is low for pure analytical
        confidence = 0.35
        return tps, confidence

    def _find_best_db_match(
        self,
        model_name: str,
        gpu_type: str,
        tp: int,
        pp: int,
        input_len: int,
        output_len: int,
    ) -> Optional[Tuple[PerfEntry, DataSource, float]]:
        """
        Find the best matching perf DB entry for the query.
        Returns (entry, data_source, confidence) or None.

        Search priority:
        1. Same model + gpu_type + tp + pp → interpolate I/O
        2. Same model + tp + pp, different gpu_type → cross-GPU scale
        3. Different model (same param count) + gpu_type + tp + pp → model-agnostic scale
        """
        model_spec = self.get_model_spec(model_name)
        target_params = model_spec.get("params_billion", 7)

        # Tier 1: exact model + GPU + TP + PP
        tier1 = [
            e for e in self.entries
            if e.model_name == model_name and e.gpu_type == gpu_type
            and e.tp == tp and e.pp == pp
        ]
        if tier1:
            best = min(tier1, key=lambda e: abs(
                self._io_work_units(e.input_len, e.output_len) -
                self._io_work_units(input_len, output_len)
            ))
            return best, DataSource.INTERPOLATED, 0.80

        # Tier 2: same GPU + TP + PP, any model
        tier2 = [
            e for e in self.entries
            if e.gpu_type == gpu_type and e.tp == tp and e.pp == pp
        ]
        if tier2:
            # prefer entry with closest param count
            best = min(tier2, key=lambda e: abs(
                self.get_model_spec(e.model_name).get("params_billion", 7) - target_params
            ))
            # scale by param count ratio (larger model → lower throughput)
            return best, DataSource.INTERPOLATED, 0.55

        # Tier 3: same model + TP + PP, different GPU type
        tier3 = [
            e for e in self.entries
            if e.model_name == model_name and e.tp == tp and e.pp == pp
        ]
        if tier3:
            best = min(tier3, key=lambda e: abs(
                self._io_work_units(e.input_len, e.output_len) -
                self._io_work_units(input_len, output_len)
            ))
            return best, DataSource.CROSS_GPU, 0.45

        # Tier 4: any entry with close params and same TP/PP
        tier4 = [
            e for e in self.entries
            if e.tp == tp and e.pp == pp
        ]
        if tier4:
            best = min(tier4, key=lambda e: abs(
                self.get_model_spec(e.model_name).get("params_billion", 7) - target_params
            ))
            return best, DataSource.CROSS_GPU, 0.30

        return None

    def _predict_metrics(
        self,
        request: JobRequest,
        gpu: GPUResource,
        tp: int,
        pp: int,
        dp: int,
    ) -> PredictedMetrics:
        """
        Predict throughput and cost for a specific (gpu, tp, pp, dp) config.
        Returns PredictedMetrics.
        """
        num_gpus = tp * pp * dp
        cost_per_hour = gpu.cost_per_gpu_hour_usd * num_gpus
        gpu_spec = GPU_SPECS.get(gpu.gpu_type, {})

        # Try DB match
        match = self._find_best_db_match(
            request.model_name,
            gpu.gpu_type,
            tp, pp,
            request.avg_input_tokens,
            request.avg_output_tokens,
        )

        if match:
            entry, data_source, base_confidence = match
            base_tps = entry.tokens_per_sec_total  # already for that DP count

            # If the entry's DP != our DP, scale linearly (more replicas = more throughput)
            if entry.dp != dp:
                base_tps = base_tps * (dp / max(entry.dp, 1))

            # Scale for I/O length difference
            tps = self._scale_throughput_for_io(
                base_tps,
                entry.input_len, entry.output_len,
                request.avg_input_tokens, request.avg_output_tokens,
            )

            # Cross-GPU scaling: scale by bandwidth ratio (decode bound)
            if data_source == DataSource.CROSS_GPU:
                src_gpu_spec = GPU_SPECS.get(entry.gpu_type, {})
                src_bw = src_gpu_spec.get("bandwidth_gbps", 400)
                tgt_bw = gpu_spec.get("bandwidth_gbps", 400)
                # Decode is bandwidth bound, prefill is compute bound
                # Weight scale by prefill/decode ratio
                pdr = request.prefill_decode_ratio
                src_flops = src_gpu_spec.get("fp16_tflops", 100)
                tgt_flops = gpu_spec.get("fp16_tflops", 100)
                # If heavy prefill (pdr > 2): lean more on FLOPS scaling
                bw_scale = tgt_bw / max(src_bw, 1)
                flops_scale = tgt_flops / max(src_flops, 1)
                prefill_weight = min(1.0, pdr / 4.0)
                decode_weight = 1.0 - prefill_weight
                gpu_scale = decode_weight * bw_scale + prefill_weight * flops_scale
                tps *= gpu_scale

            # Scale for different model params if entry model != request model
            spec = self.get_model_spec(request.model_name)
            entry_spec = self.get_model_spec(entry.model_name)
            target_params = spec.get("params_billion", 7)
            entry_params = entry_spec.get("params_billion", 7)
            if abs(target_params - entry_params) / max(entry_params, 1) > 0.1:
                # Larger model → lower throughput (bandwidth bound: tps ∝ 1/params)
                param_scale = entry_params / max(target_params, 1)
                param_scale = max(0.3, min(3.0, param_scale))
                tps *= param_scale
                base_confidence *= 0.8

            tpot_ms = entry.tpot_ms
            ttft_ms = entry.ttft_ms
            nearest_desc = f"{entry.model_name} {entry.gpu_type} TP{entry.tp} PP{entry.pp} ({entry.input_len}in/{entry.output_len}out)"
        else:
            # Pure analytical fallback
            tps, base_confidence = self._analytical_throughput(
                request.model_name, gpu, tp, pp,
                request.avg_input_tokens, request.avg_output_tokens,
            )
            tps *= dp  # multiple replicas
            data_source = DataSource.ANALYTICAL
            tpot_ms = None
            ttft_ms = None
            nearest_desc = "analytical roofline"

        # Estimate latency from throughput if not directly available
        if tpot_ms is None and tps > 0:
            # TPOT: at the expected concurrency, how many ms per output token?
            # Single replica serves: expected_concurrency / dp users
            concurrency_per_replica = max(1, (request.expected_concurrency or 10) / dp)
            # tpot = (concurrency_per_replica × output_len / replica_tps_decode) × 1000
            replica_tps = tps / dp
            if replica_tps > 0:
                # Rough estimate: each token takes 1/(tps/output_len_per_active_request) ms
                tpot_ms = (concurrency_per_replica / replica_tps) * request.avg_output_tokens * 1000
                tpot_ms = min(tpot_ms, 5000)  # sanity cap

        # Estimate runtime and cost for batch jobs
        estimated_runtime_hours = None
        total_cost_usd = None
        if request.task_type == TaskType.BATCH and request.total_tokens:
            if tps > 0:
                estimated_runtime_hours = (request.total_tokens / tps) / 3600.0
                total_cost_usd = estimated_runtime_hours * cost_per_hour

        # Hardware utilization estimate
        model_spec = self.get_model_spec(request.model_name)
        params_b = model_spec.get("params_billion", 7)
        weight_gb = self._model_weight_gb(params_b, "fp16")
        weight_per_gpu = weight_gb / tp
        mem_used_estimate = weight_per_gpu + 4.0  # rough KV cache + activations

        cost_per_1m = None
        if tps > 0:
            cost_per_1m = (cost_per_hour / tps / 3600) * 1e6

        return PredictedMetrics(
            throughput_tokens_per_sec=tps,
            throughput_per_gpu_tokens_per_sec=tps / max(num_gpus, 1),
            estimated_runtime_hours=estimated_runtime_hours,
            total_cost_usd=total_cost_usd,
            tpot_ms=tpot_ms,
            ttft_ms=ttft_ms,
            cost_per_hour_usd=cost_per_hour,
            cost_per_1m_tokens_usd=cost_per_1m,
            estimated_gpu_mem_used_gb=weight_per_gpu,
            estimated_gpu_mem_pct=(weight_per_gpu / gpu.gpu_memory_gb) * 100,
            confidence=base_confidence,
            data_source=data_source,
            nearest_db_entry=nearest_desc,
        )

    def estimate_for_config(
        self,
        request: JobRequest,
        resource: Optional["GPUResource"],
        tp: int,
        pp: int,
        dp: int,
    ) -> Optional["PredictedMetrics"]:
        """
        Public method: get a PredictedMetrics estimate for a specific (gpu, tp, pp, dp).
        Called by the ensemble AFTER the LLM proposes a config to get a performance prior.
        Returns None if resource is None (GPU not found in VPC).
        """
        if resource is None:
            return None
        try:
            return self._predict_metrics(request, resource, tp, pp, dp)
        except Exception as e:
            print(f"[Oracle] estimate_for_config failed for {resource.gpu_type} TP={tp} PP={pp}: {e}")
            return None

    # ------------------------------------------------------------------
    # SLO checking
    # ------------------------------------------------------------------

    def _check_slo(
        self, request: JobRequest, metrics: PredictedMetrics
    ) -> Tuple[bool, Optional[float]]:
        """Returns (meets_slo, margin_pct). margin_pct > 0 means headroom."""
        if request.task_type == TaskType.BATCH and request.slo_deadline_hours:
            if metrics.estimated_runtime_hours is None:
                return True, None  # can't check, assume ok
            margin = (request.slo_deadline_hours - metrics.estimated_runtime_hours) / request.slo_deadline_hours * 100
            return margin >= 0, margin

        if request.task_type == TaskType.ONLINE:
            violations = []
            if request.slo_tpot_ms and metrics.tpot_ms:
                margin = (request.slo_tpot_ms - metrics.tpot_ms) / request.slo_tpot_ms * 100
                violations.append(margin)
            if request.slo_ttft_ms and metrics.ttft_ms:
                margin = (request.slo_ttft_ms - metrics.ttft_ms) / request.slo_ttft_ms * 100
                violations.append(margin)
            if violations:
                worst = min(violations)
                return worst >= 0, worst

        return True, None  # no SLO to check

    # ------------------------------------------------------------------
    # Engine config recommendation
    # ------------------------------------------------------------------

    def _recommend_engine_config(
        self,
        request: JobRequest,
        gpu: GPUResource,
        tp: int,
        pp: int,
        dp: int,
    ) -> EngineConfig:
        """Recommend vLLM engine config for a given placement."""
        model_spec = self.get_model_spec(request.model_name)
        params_b = model_spec.get("params_billion", 7)
        weight_gb = self._model_weight_gb(params_b, "fp16")
        headroom_gb = (gpu.gpu_memory_gb * tp) - (weight_gb / dp)

        # max_model_len: use input + output SLO, cap by context window
        max_model_len = request.avg_input_tokens + request.avg_output_tokens
        max_model_len = min(max_model_len * 4, 32768)  # generous buffer

        # max_num_seqs: batch size heuristic based on available KV cache headroom
        # KV cache per seq per layer ≈ 2 × num_kv_heads × head_dim × 2bytes × 2(k+v) × seq_len
        # Simplified: more headroom → more sequences
        kv_per_seq_gb = (max_model_len * 0.001)  # rough: ~1MB per 1000 tokens
        max_num_seqs = max(8, min(512, int(headroom_gb / max(kv_per_seq_gb, 0.01))))

        # Chunked prefill: helps for long-context online serving
        chunked_prefill = request.task_type == TaskType.ONLINE and request.avg_input_tokens > 1024

        return EngineConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.90,
            dtype="auto",
            enable_chunked_prefill=chunked_prefill,
        )

    # ------------------------------------------------------------------
    # Main: enumerate all feasible candidates
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
    ) -> List[OracleCandidate]:
        """
        Enumerate all feasible (gpu_type, tp, pp, dp) combinations,
        predict metrics for each, and return sorted by cost (cheapest first).

        This is the main entry point — called before the LLM ensemble.
        """
        candidates: List[OracleCandidate] = []
        model_spec = self.get_model_spec(request.model_name)

        tp_values = [1, 2, 4, 8]
        pp_values = [1, 2, 4]

        for gpu_res in resource_map.resources:
            if gpu_res.available_gpus == 0:
                continue
            if request.preferred_gpu_types and gpu_res.gpu_type not in request.preferred_gpu_types:
                continue

            for tp in tp_values:
                # Check TP parallelism validity
                tp_ok, tp_note = self._check_parallelism(request.model_name, tp, 1)
                if not tp_ok:
                    continue

                # Check memory with this TP
                mem_ok, headroom, mem_note = self._check_memory(
                    request.model_name, gpu_res, tp
                )
                if not mem_ok:
                    continue

                for pp in pp_values:
                    # Check PP validity
                    pp_ok, pp_note = self._check_parallelism(request.model_name, tp, pp)
                    if not pp_ok:
                        continue

                    gpus_per_replica = tp * pp

                    # DP = floor(available_gpus / gpus_per_replica)
                    max_dp = gpu_res.available_gpus // gpus_per_replica
                    if request.max_total_gpus:
                        max_dp = min(max_dp, request.max_total_gpus // gpus_per_replica)
                    if max_dp < 1:
                        continue

                    # Try DP values: 1 up to max_dp (skip large dp values for online)
                    dp_values_to_try = list(range(1, min(max_dp + 1, 9)))
                    if request.task_type == TaskType.ONLINE:
                        dp_values_to_try = [d for d in dp_values_to_try if d <= 4]

                    for dp in dp_values_to_try:
                        total_gpus = gpus_per_replica * dp
                        num_instances = math.ceil(total_gpus / gpu_res.gpus_per_instance)

                        metrics = self._predict_metrics(request, gpu_res, tp, pp, dp)
                        meets_slo, margin = self._check_slo(request, metrics)
                        engine_cfg = self._recommend_engine_config(
                            request, gpu_res, tp, pp, dp
                        )

                        config = PlacementConfig(
                            gpu_type=gpu_res.gpu_type,
                            instance_type=gpu_res.instance_type,
                            num_gpus=total_gpus,
                            num_instances=num_instances,
                            tp=tp,
                            pp=pp,
                            dp=dp,
                            region=resource_map.region,
                            engine_config=engine_cfg,
                        )

                        notes = [f"Memory: {mem_note}"]
                        if not meets_slo:
                            notes.append(f"SLO miss: {margin:.1f}% short")

                        candidates.append(OracleCandidate(
                            config=config,
                            metrics=metrics,
                            meets_slo=meets_slo,
                            slo_margin_pct=margin,
                            feasibility_notes=notes,
                        ))

        # Sort: SLO-meeting first, then by total cost
        candidates.sort(
            key=lambda c: (
                0 if c.meets_slo else 1,
                c.metrics.total_cost_usd or c.metrics.cost_per_hour_usd * 24,
            )
        )

        slo_count = sum(1 for c in candidates if c.meets_slo)
        print(f"[Oracle] {len(candidates)} candidates generated, {slo_count} meet SLO")
        return candidates
