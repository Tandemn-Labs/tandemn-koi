import math

from src.prediction.analytic_parallelism import compute_parallelism_v
from src.prediction.analytic_v import compute_memory_v, kv_bytes_per_token, model_weight_gb
from src.prediction.backends.base import Candidate
from src.prediction.normalization import (
    architecture_signature,
    distance_features,
    normalize_candidate_inputs,
)

MODEL = {
    "model_params_b": 70.6,
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "num_attn_heads": 64,
    "num_kv_heads": 8,
    "weight_dtype": "bf16",
    "kvcache_dtype": "fp16",
}
WORKLOAD = {
    "type": "batch",
    "max_num_seq": 8,
    "max_num_batched_tokens": 8192,
    "isl_token_avg": 512,
    "osl_token_avg": 512,
    "gpu_mem_gb": 80,
}


def test_normalization_aliases_nested_architecture_and_distance_do_not_mutate():
    config = {"input_len_tokens_avg": 128, "tp": 2, "pp": 1}
    features = {
        "model_config_json": (
            '{"hidden_size":4096,"num_layers":32,"num_attention_heads":32,"num_key_value_heads":8}'
        ),
        "model_architecture": "LlamaForCausalLM",
        "output_len_tokens_avg": 64,
        "max_num_seq": 8,
        "max_num_batched_tokens": 2048,
        "type": "batch",
    }

    normalized_config, normalized_features = normalize_candidate_inputs(config, features)

    assert normalized_config["isl_token_avg"] == 128
    assert normalized_features["osl_token_avg"] == 64
    assert normalized_features["num_hidden_layers"] == 32
    assert architecture_signature({**normalized_features, **normalized_config})[:5] == (
        "LlamaForCausalLM",
        4096,
        32,
        32,
        8,
    )
    assert distance_features({**normalized_features, **normalized_config}) == {
        "tp": 2.0,
        "pp": 1.0,
        "isl": 128.0,
        "osl": 64.0,
        "max_num_seq": 8.0,
        "effective_batch_size": 8.0,
    }
    assert "isl_token_avg" not in config
    assert "num_hidden_layers" not in features


def test_memory_v_quantization_tp_pp_missing_and_dp_invariance():
    assert math.isclose(model_weight_gb(MODEL), 70.6e9 * 2 / (1024**3))
    assert kv_bytes_per_token(MODEL) == 2 * 80 * 8 * 128 * 2
    fp16 = compute_memory_v({**MODEL, **WORKLOAD, "tp": 4, "pp": 2, "dp": 1})
    int4 = compute_memory_v(
        {**MODEL, **WORKLOAD, "tp": 4, "pp": 2, "dp": 1, "weight_quantization_bits": 4}
    )
    dp8 = compute_memory_v({**MODEL, **WORKLOAD, "tp": 4, "pp": 2, "dp": 8})
    less_sharded = compute_memory_v({**MODEL, **WORKLOAD, "tp": 2, "pp": 1})

    assert int4["vram_headroom_gb"] > fp16["vram_headroom_gb"]
    assert fp16 == dp8
    assert fp16["vram_headroom_gb"] > less_sharded["vram_headroom_gb"]
    assert compute_memory_v({"tp": 2}, {"gpu_mem_gb": 80}) == {}


def test_parallelism_v_uses_singular_aggregate_throughput_and_never_changes_y():
    base = {
        "activation_dtype": "bf16",
        "effective_batch_size": 8,
        "gpu_per_node": 8,
        "hidden_size": 4096,
        "internode_bandwidth_gbps": 400,
        "num_hidden_layers": 32,
        "nvlink_bandwidth_gbps": 8,
        "pp": 1,
        "tp": 2,
    }
    y_hat = {"throughput_token_per_sec": 2000.0, "p99_tpot_ms": 10.0}
    dp1 = compute_parallelism_v(Candidate({**base, "dp": 1}, {}), y_hat)
    dp2 = compute_parallelism_v(Candidate({**base, "dp": 2}, {}), y_hat)
    dp2_scaled = compute_parallelism_v(
        Candidate({**base, "dp": 2}, {}),
        {"throughput_token_per_sec": 4000.0},
    )

    expected_bytes = 4 * 32 * 4096 * 2 * 0.5
    expected_time = expected_bytes / 1e9
    assert dp1["per_tok_comm_bytes"] == expected_bytes
    assert math.isclose(dp1["comm_overhead_pct"], expected_time / (1 / 2000 + expected_time))
    assert dp2["comm_overhead_pct"] < dp1["comm_overhead_pct"]
    assert dp2_scaled["comm_overhead_pct"] == dp1["comm_overhead_pct"]
    assert y_hat == {"throughput_token_per_sec": 2000.0, "p99_tpot_ms": 10.0}
    assert "throughput_tokens_per_sec" not in y_hat


def test_pipeline_bubble_and_missing_topology_omit_only_unavailable_v():
    candidate = Candidate(
        {
            "activation_dtype": "bf16",
            "effective_batch_size": 4,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "pp": 4,
            "tp": 1,
        },
        {},
    )
    result = compute_parallelism_v(candidate, {"throughput_token_per_sec": 1000.0})
    assert result["pipeline_bubble_fraction"] == 3 / 7
    assert "per_tok_comm_bytes" in result
    assert "comm_overhead_pct" not in result
