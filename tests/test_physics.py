"""Tests for koi/tools/physics.py"""

import pytest
from koi.tools.physics import (
    GPU_SPECS, lookup_gpu_spec, get_model_features, ModelFeatures,
    compute_physics_vector, physics_distance, find_similar_models,
    get_gpu_physics, get_model_arch,
)


class TestGPUSpecs:
    def test_h100_specs(self):
        s = GPU_SPECS["H100"]
        assert s["bandwidth_gbps"] == 3350
        assert s["fp16_tflops"] == 989
        assert s["mem_gb"] == 79.0

    def test_l40s_corrected(self):
        """L40S FP16 should be 362 (no sparsity), not 733."""
        s = GPU_SPECS["L40S"]
        assert s["fp16_tflops"] == 362
        assert s["bandwidth_gbps"] == 864

    def test_a10g_corrected(self):
        """A10G FP16 tensor core is 70 TFLOPS."""
        s = GPU_SPECS["A10G"]
        assert s["fp16_tflops"] == 70

    def test_l4_corrected(self):
        """L4 FP16 tensor core is 242 TFLOPS."""
        s = GPU_SPECS["L4"]
        assert s["fp16_tflops"] == 242

    def test_a100_40gb_vs_80gb(self):
        assert GPU_SPECS["A100-40GB"]["mem_gb"] == 39.0
        assert GPU_SPECS["A100-80GB"]["mem_gb"] == 79.0
        assert GPU_SPECS["A100-40GB"]["bandwidth_gbps"] == GPU_SPECS["A100-80GB"]["bandwidth_gbps"]


class TestLookup:
    def test_exact_match(self):
        s = lookup_gpu_spec("L40S")
        assert s["mem_gb"] == 45.5

    def test_case_insensitive(self):
        s = lookup_gpu_spec("h100")
        assert s["fp16_tflops"] == 989

    def test_a100_80gb_specific(self):
        s = lookup_gpu_spec("A100-80GB")
        assert s["mem_gb"] == 79.0

    def test_unknown_fallback(self):
        s = lookup_gpu_spec("UNKNOWN_GPU_XYZ")
        assert s["bandwidth_gbps"] == 400.0


class TestModelFeatures:
    def test_known_model(self):
        mf = get_model_features("Qwen/Qwen2.5-72B-Instruct")
        assert mf.num_params_billions == 72
        assert mf.num_layers == 80
        assert mf.num_attention_heads == 64
        assert mf.num_kv_heads == 8
        assert mf.gqa_ratio == 8.0
        assert mf.model_size_gb == pytest.approx(144.0, abs=1.0)

    def test_fuzzy_match(self):
        mf = get_model_features("Qwen/Qwen2.5-72B")
        assert mf.num_layers == 80

    def test_unknown_model_fallback(self):
        """Unknown model should not crash — returns a ModelFeatures."""
        mf = get_model_features("some/unknown-model-13B")
        assert mf.num_params_billions == 13.0
        assert mf.num_layers > 0

    def test_moe_model(self):
        mf = get_model_features("Qwen/Qwen3-235B-A22B")
        assert mf.is_moe is True
        assert mf.num_experts == 128


class TestPhysicsVector:
    def test_compute(self):
        mf = get_model_features("Qwen/Qwen2.5-72B-Instruct")
        vec = compute_physics_vector(mf)
        assert vec["model_size_gb"] == pytest.approx(144.0, abs=1.0)
        assert vec["num_layers"] == 80.0
        assert vec["is_moe"] == 0.0
        assert vec["gqa_ratio"] == 8.0

    def test_identical_distance_zero(self):
        mf = get_model_features("Qwen/Qwen2.5-72B-Instruct")
        v = compute_physics_vector(mf)
        assert physics_distance(v, v) == 0.0

    def test_similar_models_small_distance(self):
        """Qwen-72B and Llama-70B should be very close."""
        qwen = get_model_features("Qwen/Qwen2.5-72B-Instruct")
        llama = get_model_features("meta-llama/Llama-3-70B")
        v1 = compute_physics_vector(qwen)
        v2 = compute_physics_vector(llama)
        dist = physics_distance(v1, v2)
        assert dist < 0.10  # very similar architectures

    def test_different_models_large_distance(self):
        """72B vs 8B should have large distance."""
        big = get_model_features("Qwen/Qwen2.5-72B-Instruct")
        small = get_model_features("meta-llama/Llama-3-8B")
        v1 = compute_physics_vector(big)
        v2 = compute_physics_vector(small)
        dist = physics_distance(v1, v2)
        assert dist > 0.25

    def test_find_similar_models(self):
        target = get_model_features("Qwen/Qwen2.5-72B-Instruct")
        perfdb_models = [
            {"model_name": "meta-llama/Llama-3-70B", "num_params_billions": 70,
             "num_layers": 80, "hidden_dim": 8192, "num_attention_heads": 64,
             "num_kv_heads": 8, "vocab_size": 128256, "records_count": 50},
            {"model_name": "meta-llama/Llama-3-8B", "num_params_billions": 8,
             "num_layers": 32, "hidden_dim": 4096, "num_attention_heads": 32,
             "num_kv_heads": 8, "vocab_size": 128256, "records_count": 30},
        ]
        results = find_similar_models(target, perfdb_models)
        assert len(results) == 2
        assert results[0]["model_name"] == "meta-llama/Llama-3-70B"
        assert results[0]["distance"] < results[1]["distance"]


class TestToolFunctions:
    def test_get_gpu_physics(self):
        result = get_gpu_physics("A100-80GB")
        assert "80" in result
        assert "2000" in result

    def test_get_gpu_physics_with_model(self):
        result = get_gpu_physics("L40S", model_name="Qwen/Qwen2.5-72B-Instruct")
        assert "TP=4" in result
        assert "headroom" in result

    def test_get_model_arch(self):
        result = get_model_arch("Qwen/Qwen2.5-72B-Instruct")
        assert "72" in result
        assert "80" in result  # layers
        assert "qwen" in result.lower()
