import builtins
import contextlib
import io
import math
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from src.prediction import surrogate as surrogate_module
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
        "block_size",
        "kvcache_dtype",
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
        "scheduling_policy",
        "preemption_policy",
        "max_chunked_steps_per_request",
        "router_policy",
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
        "throughput_token_per_sec",
        "slo_margin",
    )


class PredictionSmokeTests(unittest.TestCase):
    def test_tchebycheff_and_dro_scores_are_finite(self):
        y_hat = {"throughput_token_per_sec": 1000, "slo_margin": 100}
        w_t = {"throughput_token_per_sec": 0.5, "slo_margin": 0.5}
        z_star_t = {"throughput_token_per_sec": 1000, "slo_margin": 100}
        normalization_range = {"throughput_token_per_sec": 1000, "slo_margin": 100}
        dro_band = {
            "throughput_token_per_sec": {"upper": 1100, "lower": 900},
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

    def test_online_request_rate_mode_uses_arrival_timing(self):
        controls = SurrogatePrediction(objective="online")._build_simulator_controls(
            objective="online",
            job_config={},
            job_features={"_traffic_mode": "request_rate", "request_arrival_rate": 0.1},
            direct_x_values={},
        )

        self.assertEqual(controls["replay_mode"], "offline")
        self.assertEqual(controls["request_count"], 100)
        self.assertEqual(controls["arrival_interval_ms"], 10000.0)
        self.assertNotIn("replay_concurrency", controls)

        high_rate_controls = SurrogatePrediction(objective="online")._build_simulator_controls(
            objective="online",
            job_config={},
            job_features={"_traffic_mode": "request_rate", "request_arrival_rate": 100.0},
            direct_x_values={},
        )
        self.assertEqual(high_rate_controls["request_count"], 500)
        self.assertEqual(high_rate_controls["arrival_interval_ms"], 10.0)

    def test_online_concurrency_mode_uses_replay_concurrency(self):
        controls = SurrogatePrediction(objective="online")._build_simulator_controls(
            objective="online",
            job_config={},
            job_features={"_traffic_mode": "concurrency", "max_concurrent_streaming": 7.2},
            direct_x_values={},
        )

        self.assertEqual(controls["replay_mode"], "offline")
        self.assertEqual(controls["replay_concurrency"], 8)
        self.assertEqual(controls["request_count"], 160)
        self.assertNotIn("arrival_interval_ms", controls)

    def test_offline_single_worker_kv_router_downgrades_to_round_robin(self):
        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = lambda **_: 1234
        surrogate_input = predictor.build_surrogate_inputs(
            direct_x_values={
                "model_id": "m",
                "gpu_type": "H100",
                "router_policy": "kv_router",
                "isl_token_avg": 1,
                "osl_token_avg": 1,
            },
            simulator_controls={"request_count": 1, "replay_mode": "offline"},
            method=("AIC_DynoSim",),
        )

        self.assertEqual(surrogate_input["replay_args"]["num_workers"], 1)
        self.assertEqual(surrogate_input["replay_args"]["router_mode"], "round_robin")

    def test_aic_memory_preflight_sets_num_gpu_blocks(self):
        captured = {}

        def estimate(**kwargs):
            captured.update(kwargs)
            return 1234

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = estimate
        surrogate_input = predictor.build_surrogate_inputs(
            direct_x_values={
                "model_id": "m",
                "gpu_type": "H100",
                "gpu_mem_gb": 80,
                "tp": 2,
                "pp": 1,
                "dp": 4,
                "isl_token_avg": 1,
                "osl_token_avg": 1,
            },
            simulator_controls={"request_count": 1, "replay_mode": "offline"},
            method=("AIC_DynoSim",),
        )

        self.assertEqual(surrogate_input["engine_args"]["num_gpu_blocks"], 1234)
        self.assertEqual(captured["model_path"], "m")
        self.assertEqual(captured["system"], "h100_sxm")
        self.assertEqual(captured["backend"], "vllm")
        self.assertEqual(captured["tp_size"], 2)
        self.assertEqual(captured["attention_dp_size"], 1)
        self.assertEqual(captured["memory_fraction_kind"], "of_total")
        self.assertEqual(captured["gpu_memory_capacity_bytes_override"], 80 * (1 << 30))
        self.assertEqual(surrogate_input["replay_args"]["num_workers"], 4)

    def test_aic_attention_dp_requires_explicit_engine_knob(self):
        captured = {}

        def estimate(**kwargs):
            captured.update(kwargs)
            return 1234

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = estimate
        surrogate_input = predictor.build_surrogate_inputs(
            direct_x_values={
                "model_id": "m",
                "gpu_type": "H100",
                "dp": 4,
                "aic_attention_dp_size": 2,
            },
            simulator_controls={"request_count": 1, "replay_mode": "offline"},
            method=("AIC_DynoSim",),
        )

        self.assertEqual(captured["attention_dp_size"], 2)
        self.assertEqual(surrogate_input["replay_args"]["num_workers"], 4)

    def test_aic_memory_preflight_failure_raises_before_replay(self):
        def estimate(**_kwargs):
            raise ValueError("no KV budget")

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = estimate
        with self.assertRaisesRegex(ValueError, "AIC memory preflight failed: no KV budget"):
            predictor.build_surrogate_inputs(
                direct_x_values={
                    "model_id": "m",
                    "gpu_type": "H100",
                    "isl_token_avg": 1,
                    "osl_token_avg": 1,
                },
                simulator_controls={"request_count": 1, "replay_mode": "offline"},
                method=("AIC_DynoSim",),
            )

    def test_compose_prediction_keeps_consumed_non_direct_x_values(self):
        captured_memory = {}
        captured_surrogate = {}

        def estimate(**kwargs):
            captured_memory.update(kwargs)
            return 1234

        def run_surrogate(surrogate_input, _method):
            captured_surrogate.update(surrogate_input)
            return (
                {"p99_ttft_ms": 10.0, "p99_tpot_ms": 1.0, "throughput_token_per_sec": 100.0},
                {"input_length_observed": 1.0, "output_length_observed": 1.0},
            )

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = estimate
        predictor.run_surrogate = run_surrogate
        predictor.compose_prediction(
            job_config={
                "model_id": "m",
                "max_num_seq": 8,
                "max_num_batched_tokens": 128,
                "scheduling_policy": "wspt",
                "pp": 1,
                "gemm_quant_mode": "fp8",
                "kvcache_quant_mode": "fp8",
            },
            job_features={
                "gpu_type": "H100",
                "gpu_mem_gb": 80,
                "isl_token_avg": 1,
                "osl_token_avg": 1,
            },
            candidate_graph=MockCandidateGraph(),
            method=("AIC_DynoSim",),
        )

        self.assertEqual(captured_memory["gpu_memory_capacity_bytes_override"], 80 * (1 << 30))
        self.assertEqual(captured_memory["pp_size"], 1)
        self.assertEqual(captured_memory["gemm_quant_mode"], "fp8")
        self.assertEqual(captured_memory["kvcache_quant_mode"], "fp8")
        self.assertEqual(captured_surrogate["engine_args"]["router_queue_policy"], "wspt")

    def test_compose_prediction_rejects_pp_until_dynosim_supports_it(self):
        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = lambda **_: self.fail(
            "memory preflight should not run"
        )
        predictor.run_surrogate = lambda *_: self.fail("surrogate should not run")

        with self.assertRaisesRegex(ValueError, "pp != 1"):
            predictor.compose_prediction(
                job_config={
                    "model_id": "m",
                    "max_num_seq": 8,
                    "max_num_batched_tokens": 128,
                    "pp": 2,
                },
                job_features={
                    "gpu_type": "H100",
                    "gpu_mem_gb": 80,
                    "isl_token_avg": 1,
                    "osl_token_avg": 1,
                },
                candidate_graph=MockCandidateGraph(),
                method=("AIC_DynoSim",),
            )

    def test_aic_memory_estimator_does_not_lazy_import_legacy_module_under_threads(self):
        predictor = SurrogatePrediction()
        real_import = builtins.__import__

        def estimate(**kwargs):
            return kwargs["worker_id"]

        def block_legacy_import(name, *args, **kwargs):
            if name == "aiconfigurator.sdk.memory":
                raise AssertionError("legacy AIC import used")
            return real_import(name, *args, **kwargs)

        def estimate_in_worker(worker_id):
            return predictor._estimate_num_gpu_blocks(worker_id=worker_id)

        with (
            patch.object(surrogate_module, "estimate_num_gpu_blocks", estimate),
            patch("builtins.__import__", block_legacy_import),
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            results = list(pool.map(estimate_in_worker, range(32)))

        self.assertEqual(results, list(range(32)))

    def test_cost_per_token_uses_explicit_price_only(self):
        predictor = SurrogatePrediction()
        y_hat, _ = predictor.derive_outputs(
            derive_v=[],
            derive_y=["cost_per_token"],
            y_hat_direct={"throughput_token_per_sec": 100.0},
            v_hat_direct={},
            job_config={},
            job_features={},
            price_vector={"price_per_instance_hour": 10.0},
        )
        self.assertEqual(y_hat["cost_per_token"], 10.0 / (100.0 * 3600.0))

        no_price, _ = predictor.derive_outputs(
            derive_v=[],
            derive_y=["cost_per_token"],
            y_hat_direct={"throughput_token_per_sec": 100.0},
            v_hat_direct={},
            job_config={},
            job_features={},
            price_vector=None,
        )
        self.assertNotIn("cost_per_token", no_price)

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
        self.assertEqual(surrogate_input["engine_args"]["aic_backend_version"], "0.14.0")

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
        self.assertIn("throughput_token_per_sec", y_hat_direct)
        self.assertIn("input_length_observed", v_hat_direct)
        self.assertIn("p99_ttft_ms", y_hat)
        self.assertIn("p99_tpot_ms", y_hat)
        self.assertIn("throughput_token_per_sec", y_hat)
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
