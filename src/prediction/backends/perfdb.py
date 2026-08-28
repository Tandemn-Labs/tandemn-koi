"""Configured CSV PerfDB backend with strict compatible-scope matching."""

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.prediction.backends.base import Candidate, SurrogateEstimate
from src.prediction.compatibility import canonicalize_gpu, resolve_dtype, resolve_gpu
from src.prediction.normalization import (
    architecture_signature,
    candidate_gpu_count,
    distance_features,
    merged_candidate,
    normalize_candidate_inputs,
    normalize_precision,
    normalize_workload_type,
)

PERFDB_K = 8
PERFDB_K_MIN = 5
PERFDB_TAU = 1.0
PERFDB_TAU_GPU = 1.0
PERFDB_MAX_DP_EXTRAPOLATION = 4
PERFDB_FEATURE_WEIGHTS = {
    "tp": 1.0,
    "pp": 1.0,
    "isl": 1.0,
    "osl": 1.0,
    "max_num_seq": 1.0,
    "effective_batch_size": 1.0,
}
PERFDB_FALLBACK_SCALES = {
    "tp": 1.0,
    "pp": 1.0,
    "isl": 1024.0,
    "osl": 1024.0,
    "max_num_seq": 128.0,
    "effective_batch_size": 32.0,
}

_Y_NODES = ("throughput_token_per_sec", "p99_ttft_ms", "p99_tpot_ms")
_V_NODES = (
    "sm_utilization",
    "mem_bandwidth_utilization",
    "gpu_mem_used_fraction",
    "kv_cache_util",
)
_REQUIRED_COLUMNS = {
    "avg_mem_bw_util_pct",
    "avg_mem_util_pct",
    "avg_sm_util_pct",
    "benchmark_target_concurrency",
    "gpu_count_total",
    "gpu_model",
    "input_len_tokens_avg",
    "kv_cache_util_pct_avg",
    "max_num_seqs",
    "model_architecture",
    "model_config_json",
    "output_len_tokens_avg",
    "output_tokens_per_sec",
    "pp",
    "precision",
    "status",
    "task_type",
    "tpot_ms_p99",
    "tp",
    "ttft_ms_p99",
}


@dataclass(frozen=True)
class _PerfDBRow:
    row_id: str
    architecture: tuple
    gpu_type: str
    precision: str
    workload_type: str
    features: dict[str, float]
    gpu_count: int
    y: dict[str, float]
    v: dict[str, float]


class PerfDBBackend:
    """Distance-weighted measurements within architecture/GPU/precision/workload."""

    name = "perfdb"

    def __init__(self, csv_path: str | Path, *, enforce_readiness: bool = True):
        path = Path(csv_path).resolve()
        data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.csv_path = str(path)
        self.version = f"perfdb:v2:{data_hash[:16]}"
        self.enforce_readiness = enforce_readiness
        self.rows = _load_rows(self.csv_path, data_hash)

    def provides(self) -> set[str]:
        return {*_Y_NODES, *_V_NODES}

    def estimate(
        self,
        candidate: Candidate,
        *,
        candidate_graph=None,
        method=("AIC_Direct",),
        scenario: str = "mean",
    ) -> SurrogateEstimate:
        del candidate_graph, method, scenario
        values = merged_candidate(candidate)
        try:
            dp = int(values.get("dp") or 1)
        except (TypeError, ValueError, OverflowError):
            return self._empty("unsupported", "invalid_dp")
        if dp < 1:
            return self._empty("unsupported", "invalid_dp")
        query = _candidate_query(values)
        if query is None:
            return self._empty("unsupported", "missing_hard_scope_or_distance_features")
        architecture, gpu_type, precision, workload_type, features, gpu_count = query
        effective_dp = min(dp, PERFDB_MAX_DP_EXTRAPOLATION)
        compatible_scope = [
            row
            for row in self.rows
            if row.architecture == architecture and row.workload_type == workload_type
        ]
        gpu_resolution = resolve_gpu(
            gpu_type,
            backend="perfdb",
            available={row.gpu_type for row in compatible_scope},
            requested_profile=values,
        )
        if not gpu_resolution.supported:
            return self._empty(
                "no_coverage",
                "no_compatible_gpu_scope",
                {"compatibility": {"gpu": gpu_resolution.to_dict()}},
            )
        gpu_scope = [row for row in compatible_scope if row.gpu_type == gpu_resolution.resolved]
        dtype_resolution = resolve_dtype(
            precision,
            backend="perfdb",
            component="performance",
            available={row.precision for row in gpu_scope},
        )
        if not dtype_resolution.supported:
            return self._empty(
                "no_coverage",
                "no_compatible_dtype_scope",
                {
                    "compatibility": {
                        "gpu": gpu_resolution.to_dict(),
                        "performance_dtype": dtype_resolution.to_dict(),
                    }
                },
            )
        scope = [row for row in gpu_scope if row.precision == dtype_resolution.resolved]
        distinct = {tuple(row.features.items()) for row in scope}
        if not scope:
            return self._empty("no_coverage", "hard_scope_empty")
        if self.enforce_readiness and len(distinct) < PERFDB_K_MIN:
            return self._empty(
                "insufficient_coverage",
                "fewer_than_minimum_distinct_points",
                {"distinct_points": len(distinct), "required_points": PERFDB_K_MIN},
            )

        scales = _feature_scales(scope)
        neighbors = sorted(
            ((_distance(features, row.features, scales), row) for row in scope),
            key=lambda item: (item[0], item[1].row_id),
        )[:PERFDB_K]
        y_hat: dict[str, float] = {}
        v_hat: dict[str, float] = {}
        coverage: dict[str, float] = {}
        spread: dict[str, float] = {}
        for node in self.provides():
            if node in _V_NODES and (gpu_resolution.approximate or dtype_resolution.approximate):
                continue
            values_for_node = []
            for distance, row in neighbors:
                source = row.y if node in _Y_NODES else row.v
                value = source.get(node)
                if value is None:
                    continue
                gpu_ratio = gpu_count * effective_dp / row.gpu_count
                if node == "throughput_token_per_sec":
                    value *= (
                        gpu_ratio
                        * gpu_resolution.throughput_scale
                        * dtype_resolution.throughput_scale
                    )
                elif node in {"p99_ttft_ms", "p99_tpot_ms"}:
                    value *= gpu_resolution.latency_scale * dtype_resolution.latency_scale
                values_for_node.append((distance, row, float(value), gpu_ratio))
            if not values_for_node:
                continue

            exact = [item for item in values_for_node if item[0] == 0 and item[3] == 1]
            selected = exact or values_for_node
            weights = [1.0 if exact else 1.0 / max(item[0], 1e-12) for item in selected]
            measured = _weighted_mean([item[2] for item in selected], weights)
            node_spread = _weighted_std([item[2] for item in selected], weights, measured)
            nearest = min(
                values_for_node,
                key=lambda item: (item[0], abs(math.log(item[3])), item[1].row_id),
            )
            if exact:
                node_coverage = 1.0
            else:
                node_coverage = min(1.0, len(values_for_node) / PERFDB_K_MIN)
                node_coverage *= math.exp(-((nearest[0] / PERFDB_TAU) ** 2))
                node_coverage *= math.exp(-abs(math.log(nearest[3])) / PERFDB_TAU_GPU)
            node_coverage *= gpu_resolution.confidence * dtype_resolution.confidence
            target = y_hat if node in _Y_NODES else v_hat
            target[node] = measured
            coverage[node] = node_coverage
            spread[node] = node_spread

        compatibility = {
            "gpu": gpu_resolution.to_dict(),
            "performance_dtype": dtype_resolution.to_dict(),
        }
        approximate = gpu_resolution.approximate or dtype_resolution.approximate
        version = self.version
        if approximate:
            identity = f"{gpu_resolution.fingerprint()}:{dtype_resolution.fingerprint()}"
            version = f"{version}:compat-{hashlib.sha256(identity.encode()).hexdigest()[:8]}"
        return SurrogateEstimate(
            y_hat=y_hat,
            v_hat=v_hat,
            status="success" if coverage else "no_coverage",
            version=version,
            coverage=coverage,
            spread=spread,
            source=self.name,
            metadata={
                "neighbor_count": len(neighbors),
                "scope_size": len(scope),
                "source_dp": 1,
                "requested_dp": dp,
                "effective_dp": effective_dp,
                "dp_approximation": dp != 1,
                "dp_extrapolation_capped": dp != effective_dp,
                "compatibility": compatibility,
            },
        )

    def _empty(
        self,
        status: str,
        reason: str,
        metadata: dict | None = None,
    ) -> SurrogateEstimate:
        return SurrogateEstimate(
            status=status,
            version=self.version,
            source=self.name,
            metadata={"reason": reason, **(metadata or {})},
        )


@lru_cache(maxsize=8)
def _load_rows(csv_path: str, data_hash: str) -> tuple[_PerfDBRow, ...]:
    del data_hash
    rows = []
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"PerfDB missing required columns: {sorted(missing)}")
        for raw in reader:
            try:
                row = _parse_row(raw)
            except (TypeError, ValueError, OverflowError):
                row = None
            if row is not None:
                rows.append(row)
    return tuple(rows)


def _parse_row(raw: dict[str, str]) -> _PerfDBRow | None:
    if str(raw.get("status", "")).strip().lower() != "success":
        return None
    if _number(raw.get("num_preemptions"), default=0) != 0:
        return None
    try:
        model_config = json.loads(raw.get("model_config_json") or "{}")
    except (TypeError, ValueError):
        return None
    values = {
        **model_config,
        "model_architecture": raw.get("model_architecture"),
        "is_moe": raw.get("is_moe"),
        "num_active_experts": raw.get("num_experts_active"),
        "num_routed_experts": model_config.get("num_experts"),
    }
    _, values = normalize_candidate_inputs({}, values)
    architecture = architecture_signature(values)
    gpu_type = canonicalize_gpu(raw.get("gpu_model"))
    precision = normalize_precision(raw.get("precision"))
    workload_type = normalize_workload_type(raw.get("task_type"))
    raw_dp = raw.get("dp")
    dp_value = 1.0 if raw_dp is None or not str(raw_dp).strip() else _number(raw_dp)
    if dp_value is None or int(dp_value) != 1:
        return None
    if architecture is None or not gpu_type or not precision or not workload_type:
        return None
    feature_values = {
        "tp": _number(raw.get("tp")),
        "pp": _number(raw.get("pp")),
        "isl": _number(raw.get("input_len_tokens_avg")),
        "osl": _number(raw.get("output_len_tokens_avg")),
        "max_num_seq": _number(raw.get("max_num_seqs") or raw.get("batch_size")),
        "effective_batch_size": _number(raw.get("benchmark_target_concurrency")),
    }
    if any(value is None or value <= 0 for value in feature_values.values()):
        return None
    gpu_count = int(_number(raw.get("gpu_count_total")) or 0)
    if gpu_count <= 0:
        return None
    y = _mapped_values(
        raw,
        {
            "throughput_token_per_sec": ("output_tokens_per_sec", 1.0),
            "p99_ttft_ms": ("ttft_ms_p99", 1.0),
            "p99_tpot_ms": ("tpot_ms_p99", 1.0),
        },
    )
    v = _mapped_values(
        raw,
        {
            "sm_utilization": ("avg_sm_util_pct", 0.01),
            "mem_bandwidth_utilization": ("avg_mem_bw_util_pct", 0.01),
            "gpu_mem_used_fraction": ("avg_mem_util_pct", 0.01),
            "kv_cache_util": ("kv_cache_util_pct_avg", 0.01),
        },
    )
    if any(value < 0 for value in y.values()) or any(
        value < 0 or value > 1 for value in v.values()
    ):
        return None
    return _PerfDBRow(
        row_id=raw.get("exp_id") or raw.get("timeseries_file") or repr(raw),
        architecture=architecture,
        gpu_type=gpu_type,
        precision=precision,
        workload_type=workload_type,
        features={
            name: float(value) for name, value in feature_values.items() if value is not None
        },
        gpu_count=gpu_count,
        y=y,
        v=v,
    )


def _candidate_query(values: dict) -> tuple | None:
    architecture = architecture_signature(values)
    gpu_type = canonicalize_gpu(values.get("gpu_type"))
    precision = normalize_precision(values.get("precision") or values.get("weight_dtype"))
    workload_type = normalize_workload_type(
        values.get("workload_type") or values.get("type") or values.get("task_type")
    )
    features = distance_features(values)
    try:
        gpu_count = candidate_gpu_count(values)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        architecture is None
        or not gpu_type
        or not precision
        or not workload_type
        or features is None
    ):
        return None
    return architecture, gpu_type, precision, workload_type, features, gpu_count


def _feature_scales(rows: list[_PerfDBRow]) -> dict[str, float]:
    scales = {}
    for name in PERFDB_FEATURE_WEIGHTS:
        values = np.asarray([row.features[name] for row in rows], dtype=float)
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        scales[name] = iqr if iqr > 0 else PERFDB_FALLBACK_SCALES[name]
    return scales


def _distance(query: dict[str, float], row: dict[str, float], scales: dict[str, float]) -> float:
    return math.sqrt(
        sum(
            PERFDB_FEATURE_WEIGHTS[name] * ((query[name] - row[name]) / scales[name]) ** 2
            for name in PERFDB_FEATURE_WEIGHTS
        )
    )


def _mapped_values(raw: dict[str, str], mapping: dict[str, tuple[str, float]]):
    output = {}
    for node, (column, scale) in mapping.items():
        value = _number(raw.get(column))
        if value is not None:
            output[node] = value * scale
    return output


def _number(value, default=None) -> float | None:
    if value is None or not str(value).strip():
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _text(value) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)


def _weighted_std(values: list[float], weights: list[float], mean: float) -> float:
    variance = sum(
        weight * (value - mean) ** 2 for value, weight in zip(values, weights, strict=True)
    ) / sum(weights)
    return math.sqrt(variance)
