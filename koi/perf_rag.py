"""
koi/perf_rag.py — FAISS-backed RAG over heterogeneous performance CSVs.

Design philosophy:
  The performance DB is sparse and heterogeneous: different CSVs have different
  column names, cover different GPU types, and were collected under different
  workload conditions. Classical interpolation breaks down when data is missing.

  Instead, we embed every CSV row as a physics-aware text blob and use FAISS
  cosine search to retrieve the most similar observed configs for any query.

  Key innovations:
  1. Physics-anchored embedding: derived roofline features (bandwidth_per_param,
     vram_headroom, flops_per_param, io_ratio) are included alongside raw config
     values. This lets FAISS find configs that are PHYSICALLY similar across
     GPU types — an H100 TP=4 and an A100 TP=8 might be similarly bandwidth-
     constrained for a given model, and will cluster together.

  2. Multi-query retrieval: instead of one query, we sweep (gpu_type × TP × PP)
     for the available hardware and merge results. This ensures diverse coverage
     across the feasible config space.

  3. Column normalization: ~50 known column name aliases are mapped to a
     canonical schema. Unknown columns are preserved as-is but not embedded.

  4. Confidence scoring: cosine similarity distance → retrieval confidence.
     Records very far from the query get lower weight in LLM context.

Usage:
    rag = PerfRAG(csv_dir="./perfdb")
    records = rag.retrieve_multi_query(
        model_name="meta-llama/Llama-3-70B",
        num_params_billions=70,
        is_moe=False,
        available_gpu_types=["H100", "A100"],
        tp_options=[1, 2, 4, 8],
        pp_options=[1, 2],
        input_len=512,
        output_len=128,
        k=10,
    )
    context_str = rag.format_records_for_llm(records)
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column alias map: canonical_name → [known column name variants]
# ---------------------------------------------------------------------------

COLUMN_ALIASES: Dict[str, List[str]] = {
    "model_name": [
        "model", "model_id", "hf_model", "model_name", "llm_model", "model_path",
    ],
    "gpu_type": [
        "gpu_model", "gpu", "gpu_type", "hardware", "accelerator", "gpu_name",
        "device", "gpu_class",
    ],
    "tp": [
        "tensor_parallel", "tensor_parallel_size", "tp_size", "tp",
        "tensor_parallelism", "num_tp",
    ],
    "pp": [
        "pipeline_parallel", "pipeline_parallel_size", "pp_size", "pp",
        "pipeline_parallelism", "num_pp",
    ],
    "dp": [
        "data_parallel", "data_parallel_size", "dp_size", "dp",
        "replicas", "num_replicas", "num_dp",
    ],
    "input_len": [
        "input_tokens", "isl", "input_seq_len", "avg_input_tokens",
        "max_input_length", "input_length", "prompt_len", "prompt_length",
        "input_len", "seq_len", "context_len", "prefill_len",
    ],
    "output_len": [
        "output_tokens", "osl", "output_seq_len", "avg_output_tokens",
        "max_output_length", "output_length", "gen_len", "generation_length",
        "output_len", "decode_len", "generation_len",
    ],
    "throughput_tps": [
        "tokens_per_sec_total", "tps", "throughput", "tokens_per_second",
        "total_tps", "total_tokens_per_sec", "output_tokens_per_second",
        "generation_throughput", "throughput_tps", "tok_per_sec",
        "token_throughput", "tokens_per_sec",
    ],
    "tpot_ms": [
        "tpot_ms", "tpot_ms_p50", "tpot_p50_ms", "time_per_output_token_ms",
        "tpot", "decode_latency_ms", "per_token_latency_ms", "token_latency_ms",
        "inter_token_latency_ms",
    ],
    "ttft_ms": [
        "ttft_ms", "ttft_ms_p50", "ttft_p50_ms", "time_to_first_token_ms",
        "ttft", "prefill_latency_ms", "first_token_latency_ms",
        "time_to_first_token", "e2e_latency_ms",
    ],
    "num_gpus": [
        "gpu_count_total", "total_gpus", "num_gpus", "gpus", "gpu_count",
        "n_gpus", "total_gpu_count", "gpus_total",
    ],
    "concurrency": [
        "max_num_seqs", "concurrency", "batch_size", "concurrent_requests",
        "num_concurrent", "benchmark_target_concurrency", "num_requests",
        "request_rate",
    ],
    "cost_per_hour": [
        "price_per_hour", "cost_per_hour", "price_per_instance_hour_usd",
        "cost_per_hour_usd", "hourly_cost", "cost_usd_per_hour",
    ],
    "gpu_memory_gb": [
        "gpu_memory_gb", "vram_gb", "gpu_mem_gb", "memory_gb",
        "gpu_vram", "hbm_gb", "gpu_memory",
    ],
    "interconnect": [
        "interconnect", "network", "gpu_interconnect", "nvlink", "topology",
    ],
    "quantization": [
        "quantization", "dtype", "precision", "quant", "data_type",
        "weight_dtype", "compute_dtype",
    ],
    "num_params_billions": [
        "params_billion", "num_params_billions", "params_b", "model_size_b",
        "parameters_billion", "num_params_b",
    ],
    "num_layers": [
        "num_layers", "num_hidden_layers", "n_layers", "depth", "num_decoder_layers",
    ],
    "hidden_dim": [
        "hidden_dim", "hidden_size", "d_model", "embed_dim", "model_dim",
    ],
    "num_attention_heads": [
        "num_attention_heads", "num_heads", "n_heads", "attn_heads",
    ],
    "num_kv_heads": [
        "num_kv_heads", "num_key_value_heads", "kv_heads", "n_kv_heads",
    ],
    "framework": [
        "framework", "backend", "engine", "serving_framework", "inference_engine",
    ],
    "data_source": [
        "data_source", "source", "benchmark_source", "experiment_name", "origin",
    ],
    "is_moe": [
        "is_moe", "moe", "mixture_of_experts",
    ],
    "num_experts": [
        "num_experts", "n_experts", "total_experts",
    ],
    "active_experts": [
        "active_experts", "top_k_experts", "experts_per_token",
    ],
    "ep": [
        "expert_parallel", "expert_parallel_size", "ep_size", "ep",
    ],
}

# GPU specs for physics feature computation (shared with model_features.py)
_GPU_SPECS: Dict[str, Dict[str, float]] = {
    "H100_SXM": {"bandwidth_gbps": 3350, "fp16_tflops": 989,  "mem_gb": 79.0},
    "H100":     {"bandwidth_gbps": 3350, "fp16_tflops": 989,  "mem_gb": 79.0},
    "H200":     {"bandwidth_gbps": 4800, "fp16_tflops": 989,  "mem_gb": 140.0},
    "A100":     {"bandwidth_gbps": 2000, "fp16_tflops": 312,  "mem_gb": 79.0},
    "L40S":     {"bandwidth_gbps":  864, "fp16_tflops": 733,  "mem_gb": 45.5},
    "A10G":     {"bandwidth_gbps":  600, "fp16_tflops": 125,  "mem_gb": 23.0},
    "L4":       {"bandwidth_gbps":  300, "fp16_tflops": 121,  "mem_gb": 23.0},
    "B200":     {"bandwidth_gbps": 8000, "fp16_tflops": 2250, "mem_gb": 192.0},
    "GB200":    {"bandwidth_gbps": 8000, "fp16_tflops": 2250, "mem_gb": 192.0},
}

_DTYPE_BYTES: Dict[str, float] = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0,
    "fp8": 1.0, "int8": 1.0, "int4": 0.5,
}

_NUMERIC_FIELDS = [
    "tp", "pp", "dp", "ep", "input_len", "output_len", "concurrency", "num_gpus",
    "throughput_tps", "tpot_ms", "ttft_ms", "cost_per_hour",
    "num_params_billions", "num_layers", "hidden_dim", "num_attention_heads",
    "num_kv_heads", "gpu_memory_gb", "num_experts", "active_experts", "is_moe",
]


# ---------------------------------------------------------------------------
# PerfRAG
# ---------------------------------------------------------------------------

class PerfRAG:
    """
    FAISS-backed retrieval engine over heterogeneous performance CSVs.

    Loads all CSVs from a directory, normalizes column names, computes
    physics-derived features, builds a FAISS cosine-similarity index,
    and provides retrieve() and retrieve_multi_query() for similarity search.
    """

    def __init__(
        self,
        csv_dir: str = "./perfdb",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.csv_dir = Path(csv_dir)
        self.embedding_model_name = embedding_model
        self.records: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self._encoder = None

        self._load_csvs()
        if self.records:
            self._build_index()

    # ------------------------------------------------------------------
    # Loading + normalization
    # ------------------------------------------------------------------

    def _load_csvs(self) -> None:
        csv_files = list(self.csv_dir.glob("**/*.csv"))
        if not csv_files:
            print(f"[PerfRAG] No CSV files found in {self.csv_dir}")
            return

        for csv_path in csv_files:
            try:
                df = pd.read_csv(
                    csv_path, keep_default_na=False,
                    na_values=["", "N/A", "n/a", "null", "NULL", "None"],
                )
                # Normalize column names: lowercase, strip, replace spaces/dashes
                df.columns = [
                    c.strip().lower().replace(" ", "_").replace("-", "_")
                    for c in df.columns
                ]
                source = csv_path.stem
                count = 0
                for _, row in df.iterrows():
                    rec = self._normalize_row(dict(row), source)
                    if rec is not None:
                        self.records.append(rec)
                        count += 1
                print(f"[PerfRAG] {csv_path.name}: {count}/{len(df)} rows loaded")
            except Exception as e:
                print(f"[PerfRAG] Warning — could not load {csv_path.name}: {e}")

        print(f"[PerfRAG] Total records: {len(self.records)}")
        if self.records:
            gpu_types = set(r.get("gpu_type", "?") for r in self.records)
            models = set(r.get("model_name", "?") for r in self.records)
            print(f"[PerfRAG] GPU types: {gpu_types}")
            print(f"[PerfRAG] Models: {models}")

    def _normalize_row(self, row: Dict, source: str) -> Optional[Dict]:
        """
        Map heterogeneous column names to canonical schema.
        Returns None if the row lacks a positive throughput value.
        """
        normalized: Dict[str, Any] = {"_source": source}
        row_lower = {k.lower(): v for k, v in row.items()}

        # Map aliases
        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                val = row_lower.get(alias)
                if val is not None and val != "" and not _is_nan(val):
                    normalized[canonical] = val
                    break

        # Require positive throughput
        tps = _safe_float(normalized.get("throughput_tps"), 0.0)
        if tps <= 0:
            return None

        # Defaults
        normalized.setdefault("tp", 1)
        normalized.setdefault("pp", 1)
        normalized.setdefault("dp", 1)
        normalized.setdefault("ep", 1)
        normalized.setdefault("input_len", 512)
        normalized.setdefault("output_len", 128)
        normalized.setdefault("concurrency", 1)

        # Coerce numeric fields
        for key in _NUMERIC_FIELDS:
            if key in normalized:
                normalized[key] = _safe_float(normalized[key], None)

        # Derive num_gpus
        if not normalized.get("num_gpus"):
            try:
                normalized["num_gpus"] = float(normalized["tp"]) * float(normalized["pp"]) * float(normalized["dp"])
            except (KeyError, TypeError):
                normalized["num_gpus"] = 1.0

        # Compute physics-derived features
        self._add_physics_features(normalized)
        return normalized

    def _add_physics_features(self, rec: Dict) -> None:
        """
        Add derived physics features. These anchor the embedding in physical
        reality so FAISS clusters configs by hardware utilization, not just
        by config name similarity.
        """
        gpu_type = str(rec.get("gpu_type", "")).upper()
        gpu_spec = _lookup_gpu_spec(gpu_type)

        tp = float(rec.get("tp", 1) or 1)
        params_b = float(rec.get("num_params_billions", 70) or 70)
        vram_gb = float(rec.get("gpu_memory_gb") or gpu_spec["mem_gb"])

        quant = str(rec.get("quantization", "fp16") or "fp16").lower()
        dtype_bytes = _DTYPE_BYTES.get(quant, 2.0)
        model_size_gb = params_b * 1e9 * dtype_bytes / 1e9
        weight_per_gpu_gb = model_size_gb / max(tp, 1)

        rec["_dtype_bytes"] = dtype_bytes
        rec["_model_size_gb"] = model_size_gb
        rec["_weight_per_gpu_gb"] = weight_per_gpu_gb
        rec["_vram_headroom"] = max(0.0, (vram_gb - weight_per_gpu_gb) / max(vram_gb, 1))
        rec["_bandwidth_per_param"] = (gpu_spec["bandwidth_gbps"] * tp) / max(params_b, 0.1)
        rec["_flops_per_param"] = (gpu_spec["fp16_tflops"] * tp) / max(params_b, 0.1)
        input_len = float(rec.get("input_len", 512) or 512)
        output_len = float(rec.get("output_len", 128) or 128)
        rec["_io_ratio"] = input_len / max(output_len, 1)
        rec["_total_context"] = input_len + output_len
        rec["_is_decode_heavy"] = int(rec["_io_ratio"] < 0.5)
        rec["_is_prefill_heavy"] = int(rec["_io_ratio"] > 2.0)
        rec["_roofline_tps"] = (gpu_spec["bandwidth_gbps"] * tp / max(model_size_gb, 0.1)) * 0.65

    def _row_to_embedding_text(self, rec: Dict) -> str:
        """
        Convert a normalized record to embedding text.
        Includes raw config values AND derived physics features so FAISS
        can find configs that are physically similar across GPU types.
        """
        model = rec.get("model_name", "unknown")
        gpu = rec.get("gpu_type", "unknown")
        tp = int(rec.get("tp", 1) or 1)
        pp = int(rec.get("pp", 1) or 1)
        dp = int(rec.get("dp", 1) or 1)
        input_len = float(rec.get("input_len", 512) or 512)
        output_len = float(rec.get("output_len", 128) or 128)
        tps = float(rec.get("throughput_tps", 0) or 0)
        tpot = rec.get("tpot_ms", "?")
        params = rec.get("num_params_billions", "?")
        quant = rec.get("quantization", "fp16") or "fp16"
        is_moe = int(rec.get("is_moe", 0) or 0)

        bw_per_param = rec.get("_bandwidth_per_param", 0)
        flops_per_param = rec.get("_flops_per_param", 0)
        vram_headroom = rec.get("_vram_headroom", 0)
        io_ratio = rec.get("_io_ratio", 1.0)
        model_size_gb = rec.get("_model_size_gb", 0)
        total_context = rec.get("_total_context", 0)
        workload = (
            "decode-heavy" if rec.get("_is_decode_heavy")
            else "prefill-heavy" if rec.get("_is_prefill_heavy")
            else "balanced"
        )
        roofline = rec.get("_roofline_tps", 0)

        return (
            f"model={model} params={params}B size={model_size_gb:.0f}GB dtype={quant} "
            f"is_moe={is_moe} "
            f"gpu={gpu} tp={tp} pp={pp} dp={dp} "
            f"input={input_len:.0f} output={output_len:.0f} context={total_context:.0f} "
            f"io_ratio={io_ratio:.2f} workload={workload} "
            f"tps={tps:.0f} tpot={tpot} "
            f"bw_per_param={bw_per_param:.2f} flops_per_param={flops_per_param:.2f} "
            f"vram_headroom={vram_headroom:.2f} roofline_tps={roofline:.0f}"
        )

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model_name)
                print(f"[PerfRAG] Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._encoder

    def _build_index(self) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

        encoder = self._get_encoder()
        texts = [self._row_to_embedding_text(r) for r in self.records]

        print(f"[PerfRAG] Encoding {len(texts)} records...")
        emb = encoder.encode(
            texts, batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.embeddings = np.array(emb, dtype=np.float32)

        dim = self.embeddings.shape[1]
        import faiss as _faiss
        self.index = _faiss.IndexFlatIP(dim)  # cosine via normalized embeddings
        self.index.add(self.embeddings)
        print(f"[PerfRAG] FAISS index ready: {self.index.ntotal} vectors (dim={dim})")

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def build_query_text(
        self,
        model_name: str,
        num_params_billions: float,
        is_moe: bool,
        gpu_type: str,
        tp: int,
        pp: int,
        dp: int,
        input_len: int,
        output_len: int,
        dtype: str = "fp16",
        num_layers: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> str:
        """Build a query text that mirrors _row_to_embedding_text() format."""
        gpu_spec = _lookup_gpu_spec(gpu_type)
        dtype_bytes = _DTYPE_BYTES.get(dtype, 2.0)
        model_size_gb = num_params_billions * 1e9 * dtype_bytes / 1e9
        vram_gb = gpu_spec["mem_gb"]
        weight_per_gpu = model_size_gb / max(tp, 1)
        vram_headroom = max(0.0, (vram_gb - weight_per_gpu) / max(vram_gb, 1))
        bw_per_param = (gpu_spec["bandwidth_gbps"] * tp) / max(num_params_billions, 0.1)
        flops_per_param = (gpu_spec["fp16_tflops"] * tp) / max(num_params_billions, 0.1)
        io_ratio = input_len / max(output_len, 1)
        total_context = input_len + output_len
        workload = (
            "decode-heavy" if io_ratio < 0.5
            else "prefill-heavy" if io_ratio > 2.0
            else "balanced"
        )
        roofline = (gpu_spec["bandwidth_gbps"] * tp / max(model_size_gb, 0.1)) * 0.65

        return (
            f"model={model_name} params={num_params_billions:.1f}B size={model_size_gb:.0f}GB "
            f"dtype={dtype} is_moe={int(is_moe)} "
            f"gpu={gpu_type} tp={tp} pp={pp} dp={dp} "
            f"input={input_len} output={output_len} context={total_context} "
            f"io_ratio={io_ratio:.2f} workload={workload} "
            f"bw_per_param={bw_per_param:.2f} flops_per_param={flops_per_param:.2f} "
            f"vram_headroom={vram_headroom:.2f} roofline_tps={roofline:.0f}"
        )

    def retrieve(self, query_text: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k records by cosine similarity to query_text.
        Returns records with added _similarity and _rank fields.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        encoder = self._get_encoder()
        q_emb = encoder.encode(
            [query_text], normalize_embeddings=True
        )
        q_emb = np.array(q_emb, dtype=np.float32)

        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, actual_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            rec = dict(self.records[int(idx)])
            rec["_similarity"] = float(score)
            rec["_rank"] = rank
            results.append(rec)

        return results

    def retrieve_multi_query(
        self,
        model_name: str,
        num_params_billions: float,
        is_moe: bool,
        available_gpu_types: List[str],
        tp_options: List[int],
        pp_options: List[int],
        input_len: int,
        output_len: int,
        dtype: str = "fp16",
        k: int = 10,
        **model_kwargs,
    ) -> List[Dict]:
        """
        Run FAISS queries for each (gpu_type × tp × pp) combination, deduplicate,
        and return top-k by similarity.

        This multi-query approach ensures diverse coverage: we don't just search
        for one config point but sweep the feasible space. Records that appear
        across multiple queries (high similarity to many query points) bubble up.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        seen_keys: set = set()
        all_results: List[Dict] = []

        # Limit combination explosion: max 3 GPUs × 4 TPs × 3 PPs = 36 queries
        for gpu_type in available_gpu_types[:3]:
            for tp in tp_options[:4]:
                for pp in pp_options[:3]:
                    query = self.build_query_text(
                        model_name=model_name,
                        num_params_billions=num_params_billions,
                        is_moe=is_moe,
                        gpu_type=gpu_type,
                        tp=tp, pp=pp, dp=1,
                        input_len=input_len,
                        output_len=output_len,
                        dtype=dtype,
                    )
                    # Retrieve k//2 per sub-query, deduplicate globally
                    results = self.retrieve(query, k=max(k // 2, 5))
                    for rec in results:
                        key = (
                            str(rec.get("model_name", "")),
                            str(rec.get("gpu_type", "")),
                            str(rec.get("tp", 1)),
                            str(rec.get("pp", 1)),
                            str(rec.get("input_len", 0)),
                            str(rec.get("output_len", 0)),
                            str(rec.get("_source", "")),
                        )
                        if key not in seen_keys:
                            seen_keys.add(key)
                            all_results.append(rec)

        all_results.sort(key=lambda r: -r.get("_similarity", 0.0))
        return all_results[:k]

    def format_records_for_llm(self, records: List[Dict], max_show: int = 10) -> str:
        """
        Format retrieved records as a structured, LLM-readable block.
        Each record shows: similarity, config, observed metrics, physics features.
        """
        if not records:
            return "PERFORMANCE DATABASE: No matching records retrieved."

        lines = [
            f"RETRIEVED PERFORMANCE DATABASE RECORDS ({len(records)} records):",
            "These are real observed benchmarks — use them as evidence for your proposals.",
            "sim=similarity to your query (1.0=perfect). Higher sim = more relevant.",
            "",
        ]
        for i, rec in enumerate(records[:max_show]):
            tpot_str = f"TPOT={rec['tpot_ms']:.0f}ms " if _safe_float(rec.get("tpot_ms")) else ""
            ttft_str = f"TTFT={rec['ttft_ms']:.0f}ms " if _safe_float(rec.get("ttft_ms")) else ""
            cost_str = f"cost=${rec['cost_per_hour']:.2f}/hr " if _safe_float(rec.get("cost_per_hour")) else ""
            conc_str = f"conc={int(rec.get('concurrency', 0))} " if rec.get("concurrency") else ""
            sim = rec.get("_similarity", 0)
            source = rec.get("_source", "?")
            fw = rec.get("framework", "?")

            lines.append(
                f"  [{i+1:2d}] sim={sim:.3f} | "
                f"{rec.get('model_name', '?')} | "
                f"{rec.get('gpu_type', '?')} "
                f"TP={int(rec.get('tp', 1))} PP={int(rec.get('pp', 1))} DP={int(rec.get('dp', 1))} | "
                f"{int(rec.get('num_gpus', 0))} GPUs | "
                f"in={int(rec.get('input_len', 0))} out={int(rec.get('output_len', 0))} | "
                f"TPS={rec.get('throughput_tps', 0):.0f} {tpot_str}{ttft_str}"
                f"{cost_str}{conc_str}| "
                f"bw_pp={rec.get('_bandwidth_per_param', 0):.1f} "
                f"vram_hd={rec.get('_vram_headroom', 0):.2f} "
                f"io={rec.get('_io_ratio', 0):.2f} | "
                f"src={source} fw={fw}"
            )

        if len(records) > max_show:
            lines.append(f"  ... ({len(records) - max_show} more records not shown)")

        lines += [
            "",
            "Key physics columns:",
            "  bw_pp     = bandwidth_per_param: aggregate GPU bandwidth ÷ params (decode speed proxy)",
            "  vram_hd   = vram_headroom: fraction of VRAM free after weights (KV cache space)",
            "  io        = io_ratio: input_len/output_len (>2=prefill-heavy, <0.5=decode-heavy)",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookup_gpu_spec(gpu_type: str) -> Dict[str, float]:
    """Case-insensitive GPU spec lookup with fallback."""
    gpu_upper = gpu_type.upper()
    for key, spec in _GPU_SPECS.items():
        if key.upper() in gpu_upper or gpu_upper in key.upper():
            return spec
    return {"bandwidth_gbps": 400.0, "fp16_tflops": 300.0, "mem_gb": 40.0}


def _safe_float(val: Any, default: Any = None) -> Any:
    if val is None or val == "":
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _is_nan(val: Any) -> bool:
    try:
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False
