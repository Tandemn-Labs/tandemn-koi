import contextlib
import io
import math
import unittest

from src.prediction.surrogate import SurrogatePrediction
from src.prediction.tchebycheff import (
    DEFAULT_MAXIMIZE,
    compute_tchebycheff,
    compute_tchebycheff_dro,
)


class MockCandidateGraph:
    x = (
        "model_params_b",
        "model_size_gb",
        "num_hidden_layers",
        "hidden_size",
        "num_attn_heads",
        "num_kv_heads",
        "attn_heads_per_kv_head",
        "intermediate_size",
        "max_pos_embeddings",
        "vocab_size",
        "is_moe",
        "num_routed_experts",
        "num_active_experts",
        "gpu_bandwidth_gbps",
        "gpu_tflops_fp16",
        "gpu_mem_gb",
        "cuda_compute_capability",
        "gpu_generation",
        "gpu_per_node",
        "nvlink_bandwidth_gbps",
        "internode_bandwidth_gbps",
        "pcie_bandwidth_gbps",
        "bandwidth_per_param",
        "flops_per_param",
        "gpu_watts",
        "isl_token_avg",
        "isl_token_min",
        "isl_token_max",
        "isl_distribution_type",
        "osl_token_avg",
        "osl_token_min",
        "osl_token_max",
        "osl_distribution_type",
        "pd_ratio",
        "request_arrival_rate",
        "request_arrival_pattern",
        "peak_to_mean_ratio",
        "workload_prefix_concentration",
        "multi_turn_ratio",
        "shared_prefix_length_avg",
        "is_session_affinity",
        "total_token_budget",
        "deadline_hrs",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
        "priority_class",
        "cloud",
        "region",
        "market",
        "gpu_type",
        "instance_type",
        "num_nodes_per_chain",
        "interconnect_type",
        "tp",
        "pp",
        "sp",
        "dp",
        "ep",
        "cp",
        "engine_name",
        "engine_version",
        "attn_backend",
        "runtime_image",
        "max_num_seq",
        "max_num_batched_tokens",
        "gpu_mem_util",
        "max_model_len",
        "swap_space_gb",
        "block_size",
        "kvcache_dtype",
        "kvcache_quantization",
        "weight_dtype",
        "weight_quantization_method",
        "weight_quantization_bits",
        "activation_quantization_method",
        "activation_dtype",
        "prefix_cache_enabled",
        "chunked_prefill_enable",
        "chunk_size",
        "sliding_window_size",
        "lmcache_enabled",
        "sparse_attn_pattern",
        "spec_decoding_enabled",
        "draft_model_id",
        "spec_decoding_method",
        "num_speculative_tokens",
        "spec_acceptance_threshold",
        "pd_enabled",
        "prefill_worker_count",
        "decode_worker_count",
        "kv_transfer_method",
        "cuda_graph_enabled",
        "torch_compile_enabled",
        "compile_mode",
        "num_jit_warmup_steps",
        "scheduling_policy",
        "preemption_policy",
        "max_chunked_steps_per_request",
        "router_policy",
        "expert_offload_enabled",
        "gpu_shared_fraction",
        "max_concurrent_streaming",
        "min_chain_warmup_time",
    )
    v = (
        "gpu_mem_used_fraction",
        "kv_cache_util",
        "activation_mem_pressure",
        "vram_headroom_gb",
        "live_batch_size",
        "depth_req_q",
        "input_length_observed",
        "output_length_observed",
        "sm_utilization",
        "mem_bandwidth_utilization",
        "nvlink_tput_observed",
        "pcie_tput_observed",
        "kvcache_hit_rate",
        "prefill_iteration_counts_per_second",
        "decode_itr_counts_per_second",
        "pd_inbalance",
        "expert_inbalance",
        "comm_overhead_pct",
        "pipeline_bubble_fraction",
        "per_tok_comm_bytes",
        "kv_pressure_score",
        "dispatch_overhead_ms",
    )
    y = (
        "cost_per_token",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "throughput_tokens_per_sec",
        "slo_margin",
    )


class PredictionSmokeTests(unittest.TestCase):
    def test_tchebycheff_and_dro_scores_are_finite(self):
        y_hat = {"throughput_tokens_per_sec": 1000, "slo_margin": 100}
        w_t = {"throughput_tokens_per_sec": 0.5, "slo_margin": 0.5}
        z_star_t = {"throughput_tokens_per_sec": 1000, "slo_margin": 100}
        normalization_range = {"throughput_tokens_per_sec": 1000, "slo_margin": 100}
        dro_band = {
            "throughput_tokens_per_sec": {"upper": 1100, "lower": 900},
            "slo_margin": {"upper": 110, "lower": 90},
        }

        j = compute_tchebycheff(y_hat, w_t, z_star_t, normalization_range, 1e-3, DEFAULT_MAXIMIZE)
        j_dro = compute_tchebycheff_dro(
            y_hat, dro_band, w_t, z_star_t, normalization_range, 1e-3, DEFAULT_MAXIMIZE
        )

        self.assertTrue(math.isfinite(j))
        self.assertTrue(math.isfinite(j_dro))
        self.assertEqual(j, -0.0)
        self.assertLess(j_dro, j)

    def test_surrogate_full_dynosim_smoke(self):
        predictor = SurrogatePrediction(objective="batched")
        direct_x, derive_x, direct_v, derive_v, direct_y, derive_y = (
            predictor.resolve_prediction_scope(MockCandidateGraph(), "AIC_DynoSim")
        )
        job_config = {
            "model_id": "nvidia/Llama-3.1-8B-Instruct-FP8",
            "engine_name": "vllm",
            "engine_version": "0.19.0",
            "tp": 1,
            "ep": 1,
            "block_size": 64,
            "max_num_seq": 256,
            "max_num_batched_tokens": 8192,
            "prefix_cache_enabled": True,
            "chunked_prefill_enable": True,
            "pd_enabled": False,
            "prefill_worker_count": 1,
            "decode_worker_count": 1,
            "preemption_policy": "lifo",
            "router_policy": "round_robin",
        }
        job_features = {
            "cloud": "aws",
            "region": "us-east-1",
            "market": "reserved",
            "zone": "use1-az1",
            "gpu_type": "H200",
            "instance_type": "p5e.48xlarge",
            "num_nodes_per_chain": 1,
            "interconnect_type": "nvlink",
            "isl_token_avg": 4000,
            "osl_token_avg": 500,
            "request_arrival_rate": 100,
            "workload_prefix_concentration": 0.20,
            "shared_prefix_length_avg": 1024,
            "is_session_affinity": False,
            "target_p99_ttft_ms": 200,
            "target_p99_tpot_ms": 10,
        }

        self.assertIn("gpu_type", direct_x)
        self.assertIn("target_p99_ttft_ms", derive_x)
        self.assertIn("input_length_observed", direct_v)
        self.assertIn("kv_pressure_score", derive_v)
        self.assertIn("p99_ttft_ms", direct_y)
        self.assertIn("cost_per_token", derive_y)

        env_vector = predictor.get_env_row(job_features)
        direct_x_values = predictor.extract_x_values(
            direct_x=direct_x,
            job_config=job_config,
            job_features=job_features,
            env_vector=env_vector,
        )
        direct_x_values["model_id"] = job_config["model_id"]
        self.assertEqual(env_vector["gpu_type"], "H200")
        self.assertEqual(direct_x_values["gpu_type"], "H200")
        self.assertEqual(predictor.map_gpu_to_aic_system(direct_x_values["gpu_type"]), "h200_sxm")

        simulator_controls = predictor._build_simulator_controls(
            objective=predictor.objective,
            job_config=job_config,
            job_features=job_features,
            direct_x_values=direct_x_values,
        )
        surrogate_input = predictor.build_surrogate_inputs(
            direct_x_values=direct_x_values,
            simulator_controls=simulator_controls,
            method=("AIC_DynoSim",),
        )
        self.assertEqual(simulator_controls["replay_mode"], "offline")
        self.assertGreater(simulator_controls["request_count"], 0)
        self.assertEqual(surrogate_input["method"], "AIC_DynoSim")
        self.assertEqual(surrogate_input["engine_args"]["aic_system"], "h200_sxm")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            y_hat_direct, v_hat_direct = predictor.run_aic_dynosim(surrogate_input)
            y_hat_cp, v_hat_cp = predictor.compose_prediction(
                job_config=job_config,
                job_features=job_features,
                candidate_graph=MockCandidateGraph(),
                method=("AIC_DynoSim",),
            )

        price_vector = {"price_per_hour": 98.32}
        y_hat_derived, v_hat_derived = predictor.derive_outputs(
            derive_v=derive_v,
            derive_y=derive_y,
            y_hat_direct=y_hat_direct,
            v_hat_direct=v_hat_direct,
            job_config=job_config,
            job_features=job_features,
            price_vector=price_vector,
        )
        y_hat = predictor.merge_outputs(y_hat_direct, y_hat_derived)
        v_hat = predictor.merge_outputs(v_hat_direct, v_hat_derived)

        self.assertIn("p99_ttft_ms", y_hat_direct)
        self.assertIn("throughput_tokens_per_sec", y_hat_direct)
        self.assertIn("input_length_observed", v_hat_direct)
        self.assertIn("p99_ttft_ms", y_hat)
        self.assertIn("p99_tpot_ms", y_hat)
        self.assertIn("throughput_tokens_per_sec", y_hat)
        self.assertIn("cost_per_token", y_hat)
        self.assertIn("slo_margin", y_hat)
        self.assertIn("input_length_observed", v_hat)
        self.assertIn("output_length_observed", v_hat)
        self.assertIn("kv_pressure_score", v_hat)
        self.assertIn("cost_per_token", y_hat_cp)
        self.assertIn("slo_margin", y_hat_cp)
        self.assertIn("kv_pressure_score", v_hat_cp)
        self.assertGreater(len(direct_x), 0)


if __name__ == "__main__":
    unittest.main()
