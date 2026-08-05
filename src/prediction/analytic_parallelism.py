"""Analytic TP/PP communication mediators with no outcome correction."""

import math

from src.prediction.normalization import effective_batch_size, merged_candidate

_DTYPE_BYTES = {
    "bf16": 2.0,
    "bfloat16": 2.0,
    "float16": 2.0,
    "float32": 4.0,
    "float8": 1.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "fp8": 1.0,
}


def compute_parallelism_v(candidate, y_hat: dict[str, float]) -> dict[str, float]:
    """Compute per-worker V while treating primary throughput as DP aggregate."""
    values = merged_candidate(candidate)
    if int(values.get("sp") or 1) != 1 or int(values.get("cp") or 1) != 1:
        return {}
    tp = max(1, int(values.get("tp") or 1))
    pp = max(1, int(values.get("pp") or 1))
    dp = max(1, int(values.get("dp") or 1))
    aggregate_throughput = _positive_float(y_hat.get("throughput_token_per_sec"))
    worker_throughput = aggregate_throughput / dp if aggregate_throughput is not None else None
    ideal_time = 1.0 / worker_throughput if worker_throughput is not None else None

    output: dict[str, float] = {}
    bubble = _pipeline_bubble(values, pp)
    if bubble is not None:
        output["pipeline_bubble_fraction"] = bubble
    hidden = _positive_float(values.get("hidden_size"))
    layers = _positive_float(values.get("num_hidden_layers"))
    activation_bytes = _activation_bytes(values)
    if hidden is None or layers is None or activation_bytes is None or layers % pp != 0:
        return output

    tp_bytes = 4.0 * (layers / pp) * hidden * activation_bytes * (tp - 1) / tp
    pp_bytes = (pp - 1) * hidden * activation_bytes
    output["per_tok_comm_bytes"] = tp_bytes + pp_bytes
    tp_time = _communication_time(values, tp_bytes, _tp_crosses_nodes(values, tp))
    pp_time = _communication_time(values, pp_bytes, _pp_crosses_nodes(values, tp, pp))
    if tp_time is not None and pp_time is not None and ideal_time is not None:
        output["comm_overhead_pct"] = (tp_time + pp_time) / (ideal_time + tp_time + pp_time)
    return output


def _pipeline_bubble(values: dict, pp: int) -> float | None:
    if pp == 1:
        return 0.0
    batch = effective_batch_size(values)
    micro_batch = _positive_float(
        values.get("pipeline_micro_batch_size") or values.get("micro_batch_size") or 1
    )
    if batch is None or micro_batch is None:
        return None
    microbatches = max(1, math.ceil(batch / micro_batch))
    return (pp - 1) / (microbatches + pp - 1)


def _activation_bytes(values: dict) -> float | None:
    dtype = values.get("activation_dtype") or values.get("weight_dtype")
    return _DTYPE_BYTES.get(str(dtype).lower()) if dtype is not None else None


def _tp_crosses_nodes(values: dict, tp: int) -> bool | None:
    if tp == 1:
        return False
    per_node = _positive_float(values.get("gpu_per_node") or values.get("gpus_per_node"))
    return tp > per_node if per_node is not None else None


def _pp_crosses_nodes(values: dict, tp: int, pp: int) -> bool | None:
    if pp == 1:
        return False
    per_node = _positive_float(values.get("gpu_per_node") or values.get("gpus_per_node"))
    return tp * pp > per_node if per_node is not None else None


def _communication_time(values: dict, byte_count: float, crosses_nodes: bool | None):
    if byte_count == 0:
        return 0.0
    if crosses_nodes is None:
        return None
    bandwidth = (
        _positive_float(values.get("internode_bandwidth_gbps"))
        if crosses_nodes
        else _positive_float(
            values.get("nvlink_bandwidth_gbps") or values.get("pcie_bandwidth_gbps")
        )
    )
    return None if bandwidth is None else byte_count / (bandwidth * 1e9 / 8.0)


def _positive_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None
