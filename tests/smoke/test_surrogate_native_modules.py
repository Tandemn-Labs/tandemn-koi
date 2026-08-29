import math

from src.prediction.analytic_parallelism import compute_parallelism_v
from src.prediction.analytic_v import (
    compute_memory_v,
    kv_bytes_per_token,
    model_weight_gb,
    target_memory_fit,
)
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


def test_memory_v_moe_expert_weights_shard_by_pp_not_tp_at_ep_one():
    # EP is fixed at 1, so every TP rank holds the full expert set: an MoE model
    # needs MORE per-GPU memory than a dense one of the same size at the same
    # tp/pp, and widening TP alone does not reduce it. PP does.
    dense = compute_memory_v({**MODEL, **WORKLOAD, "tp": 8, "pp": 2, "ep": 1})
    moe = compute_memory_v({**MODEL, **WORKLOAD, "tp": 8, "pp": 2, "ep": 1, "is_moe": True})
    moe_tp1 = compute_memory_v({**MODEL, **WORKLOAD, "tp": 1, "pp": 2, "ep": 1, "is_moe": True})
    moe_pp1 = compute_memory_v({**MODEL, **WORKLOAD, "tp": 8, "pp": 1, "ep": 1, "is_moe": True})

    assert moe["vram_headroom_gb"] < dense["vram_headroom_gb"]
    # Only the KV share differs between tp=1 and tp=8 for MoE; weights are identical.
    weight_gb = model_weight_gb(MODEL)
    assert moe_tp1["gpu_mem_used_fraction"] * 80 >= weight_gb / 2
    assert moe["gpu_mem_used_fraction"] * 80 >= weight_gb / 2
    assert moe["vram_headroom_gb"] > moe_pp1["vram_headroom_gb"]


def test_mixtral_fixed_ep_one_fits_only_through_pp():
    # 46.7B fp16 = ~87 GiB of weights. On an 80 GB GPU (72 GiB usable) no TP degree
    # fits at pp=1 - this is the placement the serving engine OOMs - while pp=2
    # halves the per-GPU weight and fits.
    base = {
        "model_params_b": 46.7,
        "weight_quantization_bits": 16,
        "gpu_mem_gb": 80,
        "gpu_mem_util": 0.9,
        "ep": 1,
        "is_moe": True,
    }

    for tp in (1, 2, 4, 8):
        assert target_memory_fit({**base, "tp": tp, "pp": 1})["status"] == "physical_no_fit"
    pp2 = target_memory_fit({**base, "tp": 1, "pp": 2})
    assert pp2["status"] in {"fit", "unknown"}
    assert float(pp2["required_gb"]) < float(
        target_memory_fit({**base, "tp": 8, "pp": 1})["required_gb"]
    )
    # A dense model of the same size does shard by TP.
    dense = {**base, "is_moe": False}
    assert target_memory_fit({**dense, "tp": 2, "pp": 1})["status"] in {"fit", "unknown"}


def test_online_kv_memory_is_distributed_across_replicas():
    online = {
        **MODEL,
        "type": "online",
        "gpu_mem_gb": 80,
        "gpu_mem_util": 0.9,
        "max_concurrent_streaming": 64,
        "isl_token_avg": 1024,
        "osl_token_avg": 512,
        "tp": 4,
        "pp": 1,
    }

    dp1 = compute_memory_v({**online, "dp": 1})
    dp8 = compute_memory_v({**online, "dp": 8})

    assert dp8["vram_headroom_gb"] > dp1["vram_headroom_gb"]
    assert dp8["kv_pressure_score"] < dp1["kv_pressure_score"]


def test_kvcache_auto_prefers_activation_then_weight_dtype():
    auto_activation = {**MODEL, "kvcache_dtype": "auto", "activation_dtype": "fp8"}
    auto_weight = {**MODEL, "kvcache_dtype": "auto"}

    assert kv_bytes_per_token(auto_activation) == 2 * 80 * 8 * 128
    assert kv_bytes_per_token(auto_weight) == 2 * 80 * 8 * 128 * 2


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
