"""
koi/tools/perfdb.py — PerfDB query tool via pandas.

Replaces v1's FAISS RAG with structured SQL-like queries.
The agent calls query_perfdb() to find benchmark records.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column aliases (ported from v1 perf_rag.py)
# ---------------------------------------------------------------------------

COLUMN_ALIASES: Dict[str, List[str]] = {
    "model_name": ["model", "model_id", "hf_model", "model_name", "llm_model"],
    "gpu_type": ["gpu_model", "gpu", "gpu_type", "hardware", "accelerator"],
    "tp": ["tensor_parallel", "tensor_parallel_size", "tp_size", "tp"],
    "pp": ["pipeline_parallel", "pipeline_parallel_size", "pp_size", "pp"],
    "dp": ["data_parallel", "data_parallel_size", "dp_size", "dp", "replicas", "num_replicas"],
    "input_len": [
        "input_tokens", "isl", "input_seq_len", "avg_input_tokens",
        "input_length", "input_len", "input_len_tokens_avg",
    ],
    "output_len": [
        "output_tokens", "osl", "output_seq_len", "avg_output_tokens",
        "output_length", "output_len", "output_len_tokens_avg",
    ],
    "throughput_tps": [
        "tokens_per_sec_total", "tps", "throughput", "tokens_per_second",
        "throughput_tps", "token_throughput",
    ],
    "tpot_ms": ["tpot_ms", "tpot_ms_p50", "tpot_p50_ms", "time_per_output_token_ms"],
    "ttft_ms": ["ttft_ms", "ttft_ms_p50", "ttft_p50_ms", "time_to_first_token_ms"],
    "num_gpus": ["gpu_count_total", "total_gpus", "num_gpus", "gpus", "gpu_count"],
    "cost_per_hour": ["price_per_hour", "cost_per_hour", "price_per_instance_hour_usd", "cost_per_hour_usd"],
    "cost_per_m_tokens": ["cost_per_1m_tokens_total_usd", "cost_per_m_tokens"],
    "gpu_memory_gb": ["gpu_memory_gb", "vram_gb", "gpu_mem_gb"],
    "instance_type": ["instance_type", "instance", "machine_type"],
    "interconnect": ["interconnect", "network", "gpu_interconnect"],
    "quantization": ["quantization", "dtype", "precision", "quant"],
    "num_params_billions": ["params_billion", "num_params_billions", "params_b", "model_size_b"],
    "is_moe": ["is_moe", "moe"],
    "concurrency": ["max_num_seqs", "concurrency", "batch_size", "benchmark_target_concurrency"],
    "gpu_bandwidth_gbps": ["gpu_bandwidth_gbps"],
    "gpu_tflops_fp16": ["gpu_tflops_fp16"],
    "vram_headroom_gb": ["vram_headroom_gb"],
    "bandwidth_per_param": ["bandwidth_per_param"],
    "avg_sm_util_pct": ["avg_sm_util_pct", "gpu_sm_util_pct"],
    "avg_mem_bw_util_pct": ["avg_mem_bw_util_pct", "gpu_mem_bw_util_pct"],
    "kv_cache_util_pct": ["kv_cache_util_pct_avg", "kv_cache_util_pct"],
    "model_size_gb": ["model_size_gb"],
    "status": ["status"],
}

# Key columns to return to the agent (not all 687)
KEY_COLUMNS = [
    "model_name", "gpu_type", "instance_type", "tp", "pp", "dp",
    "input_len", "output_len", "throughput_tps", "tpot_ms",
    "cost_per_hour", "cost_per_m_tokens", "num_gpus", "gpu_memory_gb",
    "vram_headroom_gb", "avg_sm_util_pct", "avg_mem_bw_util_pct",
    "kv_cache_util_pct", "concurrency", "interconnect", "quantization",
    "num_params_billions", "is_moe", "model_size_gb", "bandwidth_per_param",
]


# ---------------------------------------------------------------------------
# PerfDB class
# ---------------------------------------------------------------------------

class PerfDB:
    """Pandas-based structured query over performance benchmark CSVs."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = self._load_and_normalize()
        self._add_derived_columns()

    def _load_and_normalize(self) -> pd.DataFrame:
        """Load CSV and normalize column names using aliases."""
        df = pd.read_csv(self.csv_path, keep_default_na=False, na_values=["", "N/A", "null", "NULL", "None"])

        # Lowercase + strip column names
        df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

        # Apply alias mapping
        rename_map = {}
        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical in df.columns:
                continue  # already has canonical name
            for alias in aliases:
                if alias in df.columns:
                    rename_map[alias] = canonical
                    break
        df = df.rename(columns=rename_map)

        # Filter to rows with positive throughput
        if "throughput_tps" in df.columns:
            df["throughput_tps"] = pd.to_numeric(df["throughput_tps"], errors="coerce")
            df = df[df["throughput_tps"] > 0].copy()

        # Filter out failed runs if status column exists
        if "status" in df.columns:
            df = df[df["status"].isin(["success", ""])].copy()

        # Coerce numeric columns
        for col in ["tp", "pp", "dp", "input_len", "output_len", "num_gpus",
                     "tpot_ms", "ttft_ms", "cost_per_hour", "cost_per_m_tokens",
                     "gpu_memory_gb", "vram_headroom_gb", "num_params_billions",
                     "concurrency", "avg_sm_util_pct", "avg_mem_bw_util_pct",
                     "kv_cache_util_pct", "bandwidth_per_param", "model_size_gb"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Defaults
        if "tp" in df.columns:
            df["tp"] = df["tp"].fillna(1).astype(int)
        if "pp" in df.columns:
            df["pp"] = df["pp"].fillna(1).astype(int)
        if "dp" in df.columns:
            df["dp"] = df["dp"].fillna(1).astype(int)

        return df

    def _add_derived_columns(self):
        """Add computed columns."""
        if "input_len" in self.df.columns and "output_len" in self.df.columns:
            self.df["io_ratio"] = self.df["input_len"] / self.df["output_len"].clip(lower=1)
        if "throughput_tps" in self.df.columns and "cost_per_hour" in self.df.columns:
            tps = self.df["throughput_tps"].clip(lower=1)
            cph = self.df["cost_per_hour"]
            self.df["cost_per_m_tokens_computed"] = (cph / tps) * (1e6 / 3600)

    @property
    def record_count(self) -> int:
        return len(self.df)

    @property
    def models(self) -> List[str]:
        if "model_name" in self.df.columns:
            return self.df["model_name"].dropna().unique().tolist()
        return []

    @property
    def gpu_types(self) -> List[str]:
        if "gpu_type" in self.df.columns:
            return self.df["gpu_type"].dropna().unique().tolist()
        return []

    def query(
        self,
        model_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
        tp: Optional[int] = None,
        pp: Optional[int] = None,
        io_ratio_min: Optional[float] = None,
        io_ratio_max: Optional[float] = None,
        sort_by: str = "throughput_tps",
        sort_ascending: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Query PerfDB with filters. Returns list of row dicts with key columns."""
        df = self.df.copy()

        if model_name and "model_name" in df.columns:
            df = df[df["model_name"].str.contains(model_name, case=False, na=False)]
        if gpu_type and "gpu_type" in df.columns:
            df = df[df["gpu_type"].str.contains(gpu_type, case=False, na=False)]
        if tp is not None and "tp" in df.columns:
            df = df[df["tp"] == tp]
        if pp is not None and "pp" in df.columns:
            df = df[df["pp"] == pp]
        if io_ratio_min is not None and "io_ratio" in df.columns:
            df = df[df["io_ratio"] >= io_ratio_min]
        if io_ratio_max is not None and "io_ratio" in df.columns:
            df = df[df["io_ratio"] <= io_ratio_max]

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=sort_ascending, na_position="last")

        df = df.head(limit)

        # Return only key columns that exist
        cols = [c for c in KEY_COLUMNS if c in df.columns]
        return df[cols].to_dict("records")

    def get_distinct_models(self) -> List[Dict[str, Any]]:
        """One entry per distinct model with architecture features and record count."""
        if "model_name" not in self.df.columns:
            return []

        result = []
        for model, group in self.df.groupby("model_name"):
            if pd.isna(model) or model == "":
                continue
            entry: Dict[str, Any] = {"model_name": model, "records_count": len(group)}

            # Pull arch features from first row
            first = group.iloc[0]
            for col in ["num_params_billions", "model_size_gb", "is_moe",
                        "gpu_memory_gb"]:
                if col in group.columns and pd.notna(first.get(col)):
                    entry[col] = first[col]

            # Compute from available data
            if "num_params_billions" not in entry and "model_size_gb" in entry:
                entry["num_params_billions"] = entry["model_size_gb"] / 2.0  # assume fp16

            result.append(entry)

        return result


# ---------------------------------------------------------------------------
# Agent tool function
# ---------------------------------------------------------------------------

def query_perfdb(
    perfdb: PerfDB,
    model_name: Optional[str] = None,
    gpu_type: Optional[str] = None,
    tp: Optional[int] = None,
    pp: Optional[int] = None,
    io_ratio_min: Optional[float] = None,
    io_ratio_max: Optional[float] = None,
    sort_by: str = "throughput_tps",
    limit: int = 20,
) -> str:
    """Query PerfDB. Returns formatted table of matching benchmark records."""
    records = perfdb.query(
        model_name=model_name, gpu_type=gpu_type, tp=tp, pp=pp,
        io_ratio_min=io_ratio_min, io_ratio_max=io_ratio_max,
        sort_by=sort_by, limit=limit,
    )

    if not records:
        filters = []
        if model_name:
            filters.append(f"model={model_name}")
        if gpu_type:
            filters.append(f"gpu={gpu_type}")
        return f"No PerfDB records found for: {', '.join(filters) or 'no filters'}"

    lines = [f"PerfDB: {len(records)} records (sorted by {sort_by}):\n"]
    for i, r in enumerate(records):
        gpu = r.get("gpu_type", "?")
        tp_v = int(r.get("tp", 1))
        pp_v = int(r.get("pp", 1))
        dp_v = int(r.get("dp", 1))
        tps = r.get("throughput_tps", 0)
        tpot = r.get("tpot_ms")
        cost = r.get("cost_per_hour")
        inp = int(r.get("input_len", 0))
        out = int(r.get("output_len", 0))
        ngpu = int(r.get("num_gpus", 0))
        vram_hd = r.get("vram_headroom_gb")

        tpot_s = f"TPOT={tpot:.0f}ms" if tpot and not pd.isna(tpot) else ""
        cost_s = f"${cost:.2f}/hr" if cost and not pd.isna(cost) else ""
        vram_s = f"vram_hd={vram_hd:.1f}GB" if vram_hd and not pd.isna(vram_hd) else ""

        lines.append(
            f"  [{i+1:2d}] {r.get('model_name', '?')[:30]} | {gpu} TP={tp_v} PP={pp_v} DP={dp_v} | "
            f"{ngpu}GPUs | in={inp} out={out} | TPS={tps:.0f} {tpot_s} {cost_s} {vram_s}"
        )

    return "\n".join(lines)
