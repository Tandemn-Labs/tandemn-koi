"""Named rank-level observation fields in Koi's graph vocabulary."""

from typing import TypedDict

import numpy as np


class ObservedV(TypedDict, total=False):
    """Runtime condition series; omit unavailable measurements, not real zeros."""

    gpu_mem_used_fraction: np.ndarray
    kv_cache_util: np.ndarray
    activation_mem_pressure: np.ndarray
    vram_headroom_gb: np.ndarray
    live_batch_size: np.ndarray
    depth_req_q: np.ndarray
    input_length_observed: np.ndarray
    output_length_observed: np.ndarray
    sm_utilization: np.ndarray
    mem_bandwidth_utilization: np.ndarray
    nvlink_tput_observed: np.ndarray
    pcie_tput_observed: np.ndarray
    kvcache_hit_rate: np.ndarray
    prefill_iteration_counts_per_second: np.ndarray
    decode_itr_counts_per_second: np.ndarray
    pd_inbalance: np.ndarray
    expert_inbalance: np.ndarray
    comm_overhead_pct: np.ndarray
    pipeline_bubble_fraction: np.ndarray
    per_tok_comm_bytes: np.ndarray
    kv_pressure_score: np.ndarray
    dispatch_overhead_ms: np.ndarray


class ObservedY(TypedDict, total=False):
    """Result series; omit unavailable measurements, not real zeros."""

    cost_per_token: np.ndarray
    p99_ttft_ms: np.ndarray
    p99_tpot_ms: np.ndarray
    throughput_token_per_sec: np.ndarray
    slo_margin: np.ndarray
