import builtins
import contextlib
import io
import math
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from src.prediction import surrogate as surrogate_module
from src.prediction.compatibility import GPUProfile
from src.prediction.profile_search import ModelProfile, SupportedProfile, model_profile_from_values
from src.prediction.surrogate import (
    ONLINE_MIN_REQUESTS,
    SurrogateExecutionError,
    SurrogateMemoryNoFit,
    SurrogatePrediction,
    SurrogateUnsupportedConfig,
)
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
        "multi_turn_avg_turns",
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
    def test_a100_labels_use_a100_sxm(self):
        predictor = SurrogatePrediction()

        for gpu_type in ("A100", "A100_80GB", "A100-40GB"):
            self.assertEqual(predictor.map_gpu_to_aic_system(gpu_type), "a100_sxm")
        self.assertEqual(predictor.map_gpu_to_aic_system("A100_PCIE"), "a100_pcie")
        predictor.map_gpu_to_aic_system("A100-40GB")
        self.assertLess(
            predictor.last_metadata["compatibility"]["gpu"]["throughput_scale"],
            1.0,
        )

    def test_a100_40gb_proxy_preserves_requested_memory_capacity(self):
        captured = {}
        predictor = SurrogatePrediction()

        def estimate(**kwargs):
            captured.update(kwargs)
            return 100

        predictor._estimate_num_gpu_blocks = estimate
        predictor.build_surrogate_inputs(
            {
                "model_id": "model",
                "gpu_type": "A100-40GB",
                "max_num_seq": 4,
                "max_num_batched_tokens": 2048,
                "weight_dtype": "fp16",
                "kvcache_dtype": "fp8",
            },
            {"request_count": 4, "replay_mode": "offline"},
            ("AIC_DynoSim",),
        )

        self.assertEqual(captured["gpu_memory_capacity_bytes_override"], 40 * (1 << 30))
        self.assertEqual(captured["gemm_quant_mode"], "bfloat16")
        self.assertEqual(captured["kvcache_quant_mode"], "fp8")

    def test_unexpected_gpu_labels_resolve_to_best_effort_aic_systems(self):
        predictor = SurrogatePrediction()

        self.assertEqual(predictor.map_gpu_to_aic_system("nvidia-a10g"), "a30")
        self.assertEqual(
            predictor.last_metadata["compatibility"]["gpu"]["kind"],
            "nearest",
        )
        self.assertEqual(predictor.map_gpu_to_aic_system("nvidia-L4"), "l4")
        self.assertEqual(
            predictor.map_gpu_to_aic_system("nvidia-RTXPRO6000"),
            "rtx_pro_6000_server",
        )
        self.assertEqual(predictor.map_gpu_to_aic_system("GB200"), "gb200")

    def test_legacy_gpu_labels_use_explicit_aic_proxies_or_fail_safely(self):
        predictor = SurrogatePrediction()

        self.assertEqual(predictor.map_gpu_to_aic_system("T4"), "l4")
        self.assertEqual(predictor.map_gpu_to_aic_system("V100"), "a30")
        self.assertEqual(predictor.map_gpu_to_aic_system("V100_PCIE"), "a30")
        self.assertEqual(predictor.map_gpu_to_aic_system("L40"), "l40s")
        for gpu_type in ("GB10", "MI300"):
            with self.assertRaisesRegex(SurrogateUnsupportedConfig, "No compatible AIC system"):
                predictor.map_gpu_to_aic_system(gpu_type)

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
        self.assertEqual(controls["request_count"], ONLINE_MIN_REQUESTS)
        self.assertEqual(controls["arrival_interval_ms"], 10000.0)
        self.assertEqual(controls["turns_per_session"], 1)
        self.assertNotIn("replay_concurrency", controls)

        high_rate_controls = SurrogatePrediction(objective="online")._build_simulator_controls(
            objective="online",
            job_config={},
            job_features={"_traffic_mode": "request_rate", "request_arrival_rate": 100.0},
            direct_x_values={},
        )
        self.assertEqual(high_rate_controls["request_count"], 100)
        self.assertEqual(high_rate_controls["arrival_interval_ms"], 10.0)

    def test_online_peak_and_multiturn_stress_scenarios_preserve_turn_rate(self):
        predictor = SurrogatePrediction(objective="online")
        features = {
            "_traffic_mode": "request_rate",
            "request_arrival_rate": 0.1,
            "peak_to_mean_ratio": 2,
            "multi_turn_ratio": 0.5,
            "multi_turn_avg_turns": 3,
        }

        mean = predictor._build_simulator_controls("online", {}, features, {}, scenario="mean")
        peak = predictor._build_simulator_controls("online", {}, features, {}, scenario="peak")
        stress = predictor._build_simulator_controls(
            "online", {}, features, {}, scenario="peak_all_multiturn_stress"
        )

        self.assertEqual(mean["turns_per_session"], 1)
        self.assertEqual(mean["arrival_interval_ms"], 10000.0)
        self.assertEqual(peak["turns_per_session"], 1)
        self.assertEqual(peak["arrival_interval_ms"], 5000.0)
        self.assertEqual(stress["turns_per_session"], 3)
        self.assertEqual(stress["arrival_interval_ms"], 15000.0)
        self.assertGreaterEqual(stress["expected_completed_requests"], ONLINE_MIN_REQUESTS)
        self.assertEqual(
            stress["expected_completed_requests"],
            stress["request_count"] * stress["turns_per_session"],
        )
        self.assertEqual(stress["inter_turn_delay_ms"], 0.0)
        self.assertEqual(stress["shared_prefix_ratio"], 0.0)
        self.assertEqual(stress["num_prefix_groups"], 0)

    def test_multiturn_stress_defaults_and_ceilings(self):
        predictor = SurrogatePrediction(objective="online")
        base = {
            "_traffic_mode": "request_rate",
            "request_arrival_rate": 1.0,
            "multi_turn_ratio": 0.5,
        }

        default = predictor._build_simulator_controls(
            "online", {}, base, {}, scenario="peak_all_multiturn_stress"
        )
        noninteger = predictor._build_simulator_controls(
            "online",
            {},
            {**base, "multi_turn_avg_turns": 2.5},
            {},
            scenario="peak_all_multiturn_stress",
        )

        self.assertEqual(default["turns_per_session"], 2)
        self.assertEqual(noninteger["turns_per_session"], 3)

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

    def test_session_affinity_does_not_double_replay_turns(self):
        predictor_online = SurrogatePrediction(objective="online")
        features = {
            "_traffic_mode": "request_rate",
            "request_arrival_rate": 1.0,
            "peak_to_mean_ratio": 2.0,
            "is_session_affinity": True,
        }
        mean = predictor_online._build_simulator_controls(
            "online", {}, features, {}, scenario="mean"
        )
        peak = predictor_online._build_simulator_controls(
            "online", {}, features, {}, scenario="peak"
        )

        self.assertEqual(mean["turns_per_session"], 1)
        self.assertEqual(peak["turns_per_session"], 1)

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = lambda **_: 1234
        surrogate_input = predictor.build_surrogate_inputs(
            direct_x_values={
                "model_id": "m",
                "gpu_type": "H100",
                "is_session_affinity": True,
                "isl_token_avg": 1,
                "osl_token_avg": 1,
            },
            simulator_controls={"request_count": 20, "replay_mode": "offline"},
            method=("AIC_DynoSim",),
        )

        self.assertEqual(surrogate_input["replay_args"]["request_count"], 20)
        self.assertEqual(surrogate_input["replay_args"]["turns_per_session"], 1)

    def test_prefix_concentration_does_not_enable_replay_prefix_benefits(self):
        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = lambda **_: 1234

        for scenario in ("mean", "peak"):
            with self.subTest(scenario=scenario):
                controls = SurrogatePrediction(objective="online")._build_simulator_controls(
                    "online",
                    {},
                    {
                        "type": "online",
                        "_traffic_mode": "request_rate",
                        "request_arrival_rate": 1.0,
                        "peak_to_mean_ratio": 2.0,
                    },
                    {},
                    scenario=scenario,
                )
                surrogate_input = predictor.build_surrogate_inputs(
                    direct_x_values={
                        "model_id": "m",
                        "gpu_type": "H100",
                        "workload_prefix_concentration": 0.9,
                        "shared_prefix_length_avg": 1024,
                        "isl_token_avg": 1,
                        "osl_token_avg": 1,
                    },
                    simulator_controls=controls,
                    method=("AIC_DynoSim",),
                )

                self.assertEqual(surrogate_input["replay_args"]["shared_prefix_ratio"], 0.0)
                self.assertEqual(surrogate_input["replay_args"]["num_prefix_groups"], 0)

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
        self.assertNotIn("attention_dp_size", captured)
        self.assertEqual(captured["memory_fraction_kind"], "of_total")
        self.assertEqual(captured["gpu_memory_capacity_bytes_override"], 80 * (1 << 30))
        self.assertEqual(surrogate_input["replay_args"]["num_workers"], 4)

    def test_aic_attention_dp_is_not_koi_x(self):
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

        self.assertNotIn("attention_dp_size", captured)
        self.assertEqual(surrogate_input["replay_args"]["num_workers"], 4)

    def test_aic_memory_preflight_no_fit_raises_before_replay(self):
        def estimate(**_kwargs):
            raise ValueError("no KV budget")

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = estimate
        with self.assertRaisesRegex(SurrogateMemoryNoFit, "no KV budget"):
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

    def test_aic_memory_preflight_unsupported_config_is_structured(self):
        def estimate(**_kwargs):
            raise ValueError("unsupported model/backend/GPU for KV-cache estimation")

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = estimate
        with self.assertRaisesRegex(SurrogateUnsupportedConfig, "unsupported"):
            predictor.build_surrogate_inputs(
                direct_x_values={"model_id": "m", "gpu_type": "H100"},
                simulator_controls={"request_count": 1, "replay_mode": "offline"},
                method=("AIC_DynoSim",),
            )

    def test_aic_memory_preflight_execution_error_is_structured(self):
        def estimate(**_kwargs):
            raise ImportError("AIC import failed")

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = estimate
        with self.assertRaisesRegex(SurrogateExecutionError, "AIC import failed"):
            predictor.build_surrogate_inputs(
                direct_x_values={"model_id": "m", "gpu_type": "H100"},
                simulator_controls={"request_count": 1, "replay_mode": "offline"},
                method=("AIC_DynoSim",),
            )

    def test_aic_memory_preflight_native_panic_is_structured(self):
        class PanicException(BaseException):
            pass

        def estimate(**_kwargs):
            raise PanicException("native memory estimator panic")

        predictor = SurrogatePrediction()
        predictor._estimate_num_gpu_blocks = estimate
        with self.assertRaisesRegex(SurrogateExecutionError, "native memory estimator panic"):
            predictor.build_surrogate_inputs(
                direct_x_values={"model_id": "m", "gpu_type": "H100"},
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
            },
            job_features={
                "type": "batch",
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
        self.assertIsNone(captured_memory["gemm_quant_mode"])
        self.assertIsNone(captured_memory["kvcache_quant_mode"])
        self.assertEqual(captured_surrogate["engine_args"]["router_queue_policy"], "wspt")

    def test_compose_prediction_routes_pp_to_aic_only(self):
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
                "pp": 2,
            },
            job_features={
                "type": "batch",
                "gpu_type": "H100",
                "gpu_mem_gb": 80,
                "isl_token_avg": 1,
                "osl_token_avg": 1,
            },
            candidate_graph=MockCandidateGraph(),
            method=("AIC_DynoSim",),
        )

        self.assertEqual(captured_memory["pp_size"], 2)
        self.assertTrue(captured_surrogate["aic_only"])
        self.assertEqual(captured_surrogate["aic_args"]["pp_size"], 2)

    def test_compose_prediction_resolves_regime_per_job_not_instance(self):
        replay_args = []

        def run_surrogate(surrogate_input, _method):
            replay_args.append(dict(surrogate_input["replay_args"]))
            return (
                {"p99_ttft_ms": 10.0, "p99_tpot_ms": 1.0, "throughput_token_per_sec": 100.0},
                {"input_length_observed": 1.0, "output_length_observed": 1.0},
            )

        predictor = SurrogatePrediction(objective="online")
        predictor._estimate_num_gpu_blocks = lambda **_: 1234
        predictor.run_surrogate = run_surrogate
        common_config = {"model_id": "m", "max_num_seq": 8, "max_num_batched_tokens": 128}
        common_features = {"gpu_type": "H100", "isl_token_avg": 1, "osl_token_avg": 1}

        predictor.compose_prediction(
            job_config=common_config,
            job_features={
                **common_features,
                "type": "online",
                "_traffic_mode": "request_rate",
                "request_arrival_rate": 2.0,
            },
            candidate_graph=MockCandidateGraph(),
        )
        predictor.compose_prediction(
            job_config=common_config,
            job_features={**common_features, "type": "batch"},
            candidate_graph=MockCandidateGraph(),
        )

        self.assertEqual(replay_args[0]["arrival_interval_ms"], 500.0)
        self.assertNotIn("replay_concurrency", replay_args[0])
        self.assertEqual(replay_args[1]["arrival_interval_ms"], 0.0)
        self.assertIn("replay_concurrency", replay_args[1])

    def test_compose_prediction_requires_job_type(self):
        predictor = SurrogatePrediction()
        with self.assertRaisesRegex(ValueError, r"job_features\['type'\]"):
            predictor.compose_prediction(
                job_config={"model_id": "m"},
                job_features={"gpu_type": "H100"},
                candidate_graph=MockCandidateGraph(),
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

        aggregate_rank, _ = predictor.derive_outputs(
            derive_v=[],
            derive_y=["cost_per_token"],
            y_hat_direct={"throughput_token_per_sec": 300.0},
            v_hat_direct={},
            job_config={},
            job_features={},
            price_vector={"price_per_instance_hour": 10.0},
            replay_args={"num_workers": 3},
        )
        self.assertEqual(aggregate_rank["cost_per_token"], 30.0 / (300.0 * 3600.0))

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

    def test_total_token_budget_stays_x_not_derived_v(self):
        predictor = SurrogatePrediction()
        direct_x, _derive_x, _direct_v, derive_v, _direct_y, _derive_y = (
            predictor.resolve_prediction_scope(MockCandidateGraph(), "AIC_DynoSim")
        )
        self.assertIn("total_token_budget", MockCandidateGraph.x)
        self.assertNotIn("total_token_budget", direct_x)
        self.assertNotIn("total_token_budget", derive_v)

        _y_hat, v_hat = predictor.derive_outputs(
            derive_v=["total_token_budget"],
            derive_y=[],
            y_hat_direct={},
            v_hat_direct={"input_length_observed": 10.0, "output_length_observed": 5.0},
            job_config={"max_num_batched_tokens": 100},
            job_features={},
            price_vector=None,
        )
        self.assertNotIn("total_token_budget", v_hat)

    def test_kv_cache_v_outputs_are_explicit_placeholders(self):
        predictor = SurrogatePrediction()
        _y_hat, v_hat = predictor.derive_outputs(
            derive_v=["kv_pressure_score", "kv_cache_util"],
            derive_y=[],
            y_hat_direct={},
            v_hat_direct={"input_length_observed": 10.0, "output_length_observed": 5.0},
            job_config={"max_num_batched_tokens": 100},
            job_features={},
            price_vector=None,
        )

        self.assertEqual(v_hat["kv_pressure_score"], 0.0)
        self.assertEqual(v_hat["kv_cache_util"], 0.0)

    def test_aic_replay_completed_requests_must_match_expected(self):
        predictor = SurrogatePrediction()
        raw_report = {
            "completed_requests": 2,
            "total_input_tokens": 10,
            "total_output_tokens": 6,
            "p99_ttft_ms": 1.0,
            "p99_tpot_ms": 2.0,
            "output_throughput_tok_s": 100.0,
        }

        y_hat, v_hat = predictor.canonicalize_aic_dynosim_output(raw_report, expected_requests=2)
        self.assertEqual(y_hat["p99_tpot_ms"], 2.0)
        self.assertEqual(v_hat["input_length_observed"], 5.0)
        self.assertEqual(v_hat["output_length_observed"], 3.0)
        self.assertEqual(v_hat["completed_requests"], 2)

        with self.assertRaisesRegex(SurrogateExecutionError, "completed 0/1 requests"):
            predictor.canonicalize_aic_dynosim_output(
                {**raw_report, "completed_requests": 0}, expected_requests=1
            )
        with self.assertRaisesRegex(SurrogateExecutionError, "completed 1/2 requests"):
            predictor.canonicalize_aic_dynosim_output(
                {**raw_report, "completed_requests": 1}, expected_requests=2
            )
        with self.assertRaisesRegex(SurrogateExecutionError, "missing completed_requests"):
            predictor.canonicalize_aic_dynosim_output(
                {key: value for key, value in raw_report.items() if key != "completed_requests"},
                expected_requests=2,
            )

    def test_dynosim_native_missing_perf_data_falls_back_to_aic_sol(self):
        class PanicException(BaseException):
            pass

        dynamo = ModuleType("dynamo")
        llm = ModuleType("dynamo.llm")
        replay = ModuleType("dynamo.replay")
        replay_api = ModuleType("dynamo.replay.api")
        llm.MockEngineArgs = lambda **kwargs: kwargs

        def panic(*_args, **_kwargs):
            raise PanicException(
                "PerfDataNotAvailableError: FMHA bf16 batch=3 sequence=682 "
                "heads=8 kv_heads=2 head_size=64"
            )

        replay_api.run_synthetic_trace_replay = panic
        modules = {
            "dynamo": dynamo,
            "dynamo.llm": llm,
            "dynamo.replay": replay,
            "dynamo.replay.api": replay_api,
        }
        surrogate_input = {
            "engine_args": {
                "aic_model_path": "model",
                "aic_system": "h100_sxm",
                "aic_backend": "vllm",
            },
            "replay_args": {
                "input_tokens": 682,
                "output_tokens": 1,
                "request_count": 3,
                "expected_completed_requests": 3,
                "replay_mode": "offline",
                "router_mode": "round_robin",
            },
        }

        attempted_modes = []

        def estimate(**kwargs):
            attempted_modes.append(kwargs["database_mode"])
            if kwargs["database_mode"] != "SOL":
                raise ValueError("PerfDataNotAvailableError: no empirical slice")
            return SimpleNamespace(
                raw={"tokens/s": 42.0},
                ttft=12.0,
                tpot=3.0,
            )

        predictor = SurrogatePrediction()
        predictor._aic_estimate = estimate
        with patch.dict(sys.modules, modules):
            y_hat, _ = predictor.run_aic_dynosim(surrogate_input)

        self.assertEqual(attempted_modes, ["HYBRID", "EMPIRICAL", "SOL"])
        self.assertEqual(y_hat["throughput_token_per_sec"], 42.0)
        self.assertEqual(predictor.last_metadata["aic_database_mode"], "SOL")
        self.assertIn("batch=3 sequence=682", predictor.last_metadata["aic_fallback"]["reason"])

    def test_online_aic_capacity_fallback_omits_queueing_latency(self):
        predictor = SurrogatePrediction()
        predictor._aic_estimate = lambda **_kwargs: SimpleNamespace(
            raw={"tokens/s": 42.0},
            ttft=12.0,
            tpot=3.0,
        )
        surrogate_input = {
            "objective": "online",
            "engine_args": {
                "aic_model_path": "model",
                "aic_system": "h100_sxm",
                "aic_backend": "vllm",
            },
            "replay_args": {
                "input_tokens": 512,
                "output_tokens": 170,
                "request_count": 3,
                "expected_completed_requests": 3,
            },
        }

        y_hat, _ = predictor._run_aic_mode(surrogate_input, "SOL")

        self.assertEqual(y_hat, {"throughput_token_per_sec": 42.0})
        self.assertEqual(
            predictor.last_metadata["aic_fallback_omitted_nodes"],
            ["p99_tpot_ms", "p99_ttft_ms"],
        )

    def test_aic_proxy_gpu_scales_direct_estimate_conservatively(self):
        predictor = SurrogatePrediction()
        predictor.map_gpu_to_aic_system("nvidia-a10g")
        predictor._aic_estimate = lambda **_kwargs: SimpleNamespace(
            raw={"tokens/s": 100.0},
            ttft=10.0,
            tpot=2.0,
        )
        surrogate_input = {
            "engine_args": {
                "aic_model_path": "model",
                "aic_system": "a30",
                "aic_backend": "vllm",
            },
            "replay_args": {
                "input_tokens": 512,
                "output_tokens": 170,
                "request_count": 3,
                "expected_completed_requests": 3,
            },
        }

        y_hat, _ = predictor._run_aic_mode(surrogate_input, "SOL")

        scale = predictor.last_metadata["compatibility"]["gpu"]["throughput_scale"]
        self.assertEqual(y_hat["throughput_token_per_sec"], 100.0 * scale)
        self.assertAlmostEqual(y_hat["p99_ttft_ms"], 10.0 / scale)
        self.assertAlmostEqual(y_hat["p99_tpot_ms"], 2.0 / scale)

    def test_direct_aic_throughput_combines_prefill_and_decode_scaling(self):
        predictor = SurrogatePrediction()
        predictor.last_metadata = {
            "aic_profile_match": {
                "prefill_speed_ratio": 0.25,
                "decode_speed_ratio": 1.0,
            }
        }
        predictor._aic_estimate = lambda **_kwargs: SimpleNamespace(
            raw={"tokens/s": 100.0},
            ttft=10.0,
            tpot=2.0,
        )
        surrogate_input = {
            "engine_args": {
                "aic_model_path": "proxy",
                "aic_system": "h100_sxm",
                "aic_backend": "vllm",
            },
            "replay_args": {
                "input_tokens": 512,
                "output_tokens": 128,
                "request_count": 3,
                "expected_completed_requests": 3,
            },
        }

        y_hat, _ = predictor._run_aic_mode(surrogate_input, "SOL")

        expected_scale = (10 + 2 * 128) / (10 / 0.25 + 2 * 128)
        self.assertEqual(y_hat["throughput_token_per_sec"], 100.0 * expected_scale)

    def test_joint_aic_profile_search_returns_ranked_bounded_matches(self):
        predictor = SurrogatePrediction()
        values = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_architecture": "Qwen2ForCausalLM",
            "model_params_b": 7.6,
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_attn_heads": 28,
            "num_kv_heads": 4,
            "head_dim": 128,
            "vocab_size": 152064,
            "max_pos_embeddings": 32768,
            "gpu_type": "H100",
            "gpu_vendor": "nvidia",
            "gpu_generation": "hopper",
            "gpu_mem_gb": 80,
            "gpu_bandwidth_gbps": 3350,
            "gpu_tflops_fp16": 989,
            "nvlink_bandwidth_gbps": 900,
            "weight_dtype": "fp16",
            "type": "online",
            "isl_token_avg": 512,
            "osl_token_avg": 170,
            "request_arrival_rate": 2.0,
            "max_num_seq": 8,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "engine_name": "vllm",
            "engine_version": "0.22.0",
        }

        matches = predictor._rank_aic_profiles(values)

        self.assertEqual(len(matches), 5)
        self.assertTrue(all(match.supported.aic_system == "h100_sxm" for match in matches))
        self.assertTrue(all(match.prefill_speed_ratio > 0 for match in matches))
        self.assertTrue(all(match.decode_speed_ratio > 0 for match in matches))

    def test_joint_profile_search_uses_partial_model_metadata(self):
        predictor = SurrogatePrediction()
        predictor._enrich_requested_model_values = lambda values: values
        values = {
            "model_id": "acme/private-8b",
            "model_params_b": 8,
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attn_heads": 32,
            "gpu_type": "H100",
            "gpu_mem_gb": 80,
            "gpu_bandwidth_gbps": 3350,
            "gpu_tflops_fp16": 989,
            "weight_dtype": "bf16",
            "type": "online",
            "isl_token_avg": 512,
            "osl_token_avg": 128,
            "max_num_seq": 8,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "engine_name": "vllm",
        }

        matches = predictor._rank_aic_profiles(values)

        self.assertEqual(len(matches), 5)
        self.assertTrue(all(match.confidence < 1.0 for match in matches))

    def test_direct_retries_next_ranked_aic_profile(self):
        attempts = []
        matches = (
            {
                "profile_id": "first",
                "aic_system": "first_system",
                "model_id": "first-model",
                "prefill_speed_ratio": 0.5,
                "decode_speed_ratio": 0.25,
            },
            {
                "profile_id": "second",
                "aic_system": "second_system",
                "model_id": "second-model",
                "prefill_speed_ratio": 0.8,
                "decode_speed_ratio": 0.4,
            },
        )
        predictor = SurrogatePrediction()

        def run_modes(surrogate_input, _modes):
            engine_args = surrogate_input["engine_args"]
            attempts.append(engine_args)
            if engine_args["aic_system"] == "first_system":
                raise SurrogateUnsupportedConfig("PerfDataNotAvailableError: missing slice")
            return {"throughput_token_per_sec": 50.0}, {}

        predictor._run_aic_modes = run_modes
        predictor.run_aic_dynosim = lambda *_args, **_kwargs: self.fail("Direct called DynoSim")
        y_hat, _ = predictor.run_aic_only(
            {
                "engine_args": {"aic_system": "target", "aic_model_path": "target-model"},
                "aic_profile_matches": matches,
            }
        )

        self.assertEqual([attempt["aic_system"] for attempt in attempts], ["first_system", "second_system"])
        self.assertEqual(y_hat["throughput_token_per_sec"], 50.0)
        self.assertEqual(predictor.last_metadata["aic_profile_match"]["profile_id"], "second")
        self.assertEqual(predictor.last_metadata["aic_profile_attempts"][0]["profile_id"], "first")

    def test_huggingface_enrichment_falls_back_from_aic_lookup(self):
        predictor = SurrogatePrediction()
        config = {
            "model_params_b": 14,
            "num_hidden_layers": 40,
            "hidden_size": 5120,
            "num_attention_heads": 40,
        }

        with (
            patch(
                "aiconfigurator.sdk.utils.get_model_config_from_model_path",
                side_effect=ValueError("unknown AIC model"),
            ),
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(to_dict=lambda: config),
            ) as load_config,
        ):
            enriched = predictor._enrich_requested_model_values({"model_id": "microsoft/phi-4"})

        self.assertEqual(enriched["model_params_b"], 14)
        self.assertEqual(predictor.last_metadata["model_profile_enrichment"], {
            "status": "success",
            "source": "huggingface",
        })
        self.assertEqual(load_config.call_args.kwargs["token"], None)

    def test_direct_memory_fit_uses_enriched_requested_model(self):
        predictor = SurrogatePrediction()
        predictor._enrich_requested_model_values = lambda values: {
            **values,
            "model_params_b": 70,
            "weight_dtype": "bf16",
            "gpu_mem_gb": 24,
        }

        with self.assertRaisesRegex(SurrogateMemoryNoFit, "requested model memory no-fit"):
            predictor.build_surrogate_inputs(
                {
                    "model_id": "acme/large-model",
                    "gpu_type": "A10G",
                    "tp": 1,
                    "pp": 1,
                },
                {"request_count": 1, "replay_mode": "offline"},
                ("AIC_Direct",),
            )

        self.assertEqual(predictor.last_metadata["target_memory_fit"]["status"], "no_fit")

    def test_profile_uses_fp8_quantization_method(self):
        profile = model_profile_from_values(
            "acme/fp8-model",
            {
                "model_params_b": 8,
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attn_heads": 32,
                "weight_dtype": "bf16",
                "weight_quantization_method": "fp8",
                "weight_quantization_bits": 8,
            },
        )

        self.assertIsNotNone(profile)
        self.assertEqual(profile.weight_dtype, "fp8")

    def test_aggregate_profile_search_pins_system_and_ranks_parameters(self):
        gpu = GPUProfile(
            "A100",
            vendor="nvidia",
            architecture="ampere",
            memory_gb=80,
            memory_bandwidth_gbps=2039,
            fp16_tflops=312,
        )
        target = ModelProfile(
            model_id="acme/dense-30b",
            architecture="DenseForCausalLM",
            layers=32,
            hidden_size=4096,
            intermediate_size=16384,
            attention_heads=32,
            kv_heads=8,
            head_dim=128,
            vocab_size=32000,
            parameter_count=30e9,
            is_moe=False,
            routed_experts=0,
            active_experts=0,
            weight_dtype="bf16",
            max_context=8192,
        )
        profiles = (
            SupportedProfile(
                "a100-near",
                gpu,
                ModelProfile(**{**target.__dict__, "model_id": "aic/dense-32b", "parameter_count": 32e9}),
                "vllm",
                "0.22.0",
                "a100_sxm",
            ),
            SupportedProfile(
                "a100-far",
                gpu,
                ModelProfile(**{**target.__dict__, "model_id": "aic/dense-70b", "parameter_count": 70e9}),
                "vllm",
                "0.22.0",
                "a100_sxm",
            ),
            SupportedProfile(
                "a100-moe",
                gpu,
                ModelProfile(
                    **{
                        **target.__dict__,
                        "model_id": "aic/moe-30b",
                        "is_moe": True,
                        "routed_experts": 8,
                        "active_experts": 2,
                    }
                ),
                "vllm",
                "0.22.0",
                "a100_sxm",
            ),
            SupportedProfile("b200-near", gpu, target, "vllm", "0.22.0", "b200_sxm"),
        )
        predictor = SurrogatePrediction()
        predictor._enrich_requested_model_values = lambda values: values
        values = {
            "model_id": target.model_id,
            "model_params_b": 30,
            "model_architecture": target.architecture,
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_attn_heads": 32,
            "num_kv_heads": 8,
            "weight_dtype": "bf16",
            "gpu_type": "A100-PCIe-40GB",
            "gpu_mem_gb": 80,
            "type": "online",
            "isl_token_avg": 512,
            "osl_token_avg": 128,
            "max_num_seq": 8,
            "tp": 8,
            "pp": 1,
            "engine_name": "vllm",
        }

        with patch("src.prediction.aic_support.load_aic_support_profiles", return_value=profiles):
            matches = predictor._rank_aic_profiles(values, aic_system="a100_sxm")

        self.assertEqual(
            [match.supported.model.model_id for match in matches],
            ["aic/dense-32b", "aic/dense-70b"],
        )

    def test_direct_finds_proxy_alternates_for_phi_and_mixtral(self):
        models = (
            {
                "model_id": "microsoft/phi-4",
                "architecture": "Phi3ForCausalLM",
                "model_params_b": 14,
                "num_hidden_layers": 40,
                "hidden_size": 5120,
                "intermediate_size": 17920,
                "num_attn_heads": 40,
                "num_kv_heads": 10,
                "head_dim": 128,
                "vocab_size": 100352,
                "weight_dtype": "bf16",
                "kvcache_dtype": "bf16",
                "tp": 2,
            },
            {
                "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "architecture": "MixtralForCausalLM",
                "model_params_b": 46.7,
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_attn_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "vocab_size": 32000,
                "is_moe": True,
                "num_experts": 8,
                "num_active_experts": 2,
                "moe_intermediate_size": 14336,
                "weight_dtype": "bf16",
                "kvcache_dtype": "bf16",
                "tp": 8,
            },
        )

        for model in models:
            with self.subTest(model=model["model_id"]):
                values = {
                    **model,
                    "gpu_type": "H100",
                    "gpu_mem_gb": 80,
                    "gpu_bandwidth_gbps": 3350,
                    "gpu_tflops_fp16": 989,
                    "type": "online",
                    "isl_token_avg": 512,
                    "osl_token_avg": 128,
                    "max_num_seq": 8,
                    "max_num_batched_tokens": 8192,
                    "pp": 1,
                    "dp": 1,
                    "engine_name": "vllm",
                }
                predictor = SurrogatePrediction()
                predictor._enrich_requested_model_values = lambda candidate: candidate
                matches = predictor._rank_aic_profiles(values)

                self.assertGreaterEqual(len(matches), 2)
                first_proxy = matches[0].supported.model.model_id
                second_proxy = matches[1].supported.model.model_id
                self.assertNotEqual(first_proxy, model["model_id"])
                self.assertNotEqual(second_proxy, model["model_id"])
                attempts = []

                def run_modes(surrogate_input, _modes):
                    model_path = surrogate_input["engine_args"]["aic_model_path"]
                    attempts.append(model_path)
                    if model_path == first_proxy:
                        raise SurrogateUnsupportedConfig("unsupported model")
                    return {"throughput_token_per_sec": 50.0}, {}

                predictor._run_aic_modes = run_modes
                predictor.run_aic_dynosim = lambda *_args, **_kwargs: self.fail(
                    "Direct called DynoSim"
                )
                y_hat, _ = predictor.run_aic_only(
                    {
                        "engine_args": {
                            "aic_system": "target",
                            "aic_model_path": model["model_id"],
                        },
                        "aic_profile_matches": tuple(match.to_dict() for match in matches),
                    }
                )

                self.assertEqual(attempts, [first_proxy, second_proxy])
                self.assertEqual(y_hat["throughput_token_per_sec"], 50.0)
                self.assertEqual(
                    predictor.last_metadata["aic_profile_match"]["model_id"],
                    second_proxy,
                )

    def test_dynosim_retries_next_ranked_aic_profile(self):
        class PanicException(BaseException):
            pass

        attempts = []
        dynamo = ModuleType("dynamo")
        llm = ModuleType("dynamo.llm")
        replay = ModuleType("dynamo.replay")
        replay_api = ModuleType("dynamo.replay.api")
        llm.MockEngineArgs = lambda **kwargs: kwargs

        def replay_call(*_args, **kwargs):
            engine = kwargs["extra_engine_args"]
            attempts.append(engine)
            if engine["aic_system"] == "first_system":
                raise PanicException("PerfDataNotAvailableError: missing slice")
            return {
                "completed_requests": 3,
                "total_input_tokens": 1536,
                "total_output_tokens": 510,
                "p99_ttft_ms": 20.0,
                "p99_tpot_ms": 4.0,
                "output_throughput_tok_s": 50.0,
            }

        replay_api.run_synthetic_trace_replay = replay_call
        modules = {
            "dynamo": dynamo,
            "dynamo.llm": llm,
            "dynamo.replay": replay,
            "dynamo.replay.api": replay_api,
        }
        matches = (
            {
                "profile_id": "first",
                "aic_system": "first_system",
                "model_id": "first-model",
                "prefill_speed_ratio": 0.5,
                "decode_speed_ratio": 0.25,
            },
            {
                "profile_id": "second",
                "aic_system": "second_system",
                "model_id": "second-model",
                "prefill_speed_ratio": 0.8,
                "decode_speed_ratio": 0.4,
            },
        )
        surrogate_input = {
            "engine_args": {
                "aic_model_path": "target-model",
                "aic_system": "target-system",
                "aic_backend": "vllm",
            },
            "replay_args": {
                "input_tokens": 512,
                "output_tokens": 170,
                "request_count": 3,
                "expected_completed_requests": 3,
                "replay_mode": "offline",
                "router_mode": "round_robin",
            },
            "aic_profile_matches": matches,
        }

        predictor = SurrogatePrediction()
        with patch.dict(sys.modules, modules):
            y_hat, _ = predictor.run_aic_dynosim(surrogate_input)

        self.assertEqual(
            [attempt["aic_system"] for attempt in attempts], ["first_system", "second_system"]
        )
        self.assertEqual(attempts[1]["aic_model_path"], "second-model")
        self.assertEqual(attempts[1]["speedup_ratio"], 0.8)
        self.assertEqual(attempts[1]["decode_speedup_ratio"], 0.5)
        self.assertEqual(y_hat["throughput_token_per_sec"], 50.0)
        self.assertEqual(predictor.last_metadata["aic_profile_match"]["profile_id"], "second")

        with patch.dict(sys.modules, modules):
            predictor.run_aic_dynosim(surrogate_input)
        self.assertEqual(
            [attempt["aic_system"] for attempt in attempts],
            ["first_system", "second_system", "second_system"],
        )

        changed_workload = {
            **surrogate_input,
            "replay_args": {**surrogate_input["replay_args"], "arrival_interval_ms": 10.0},
        }
        with patch.dict(sys.modules, modules):
            predictor.run_aic_dynosim(changed_workload)
        self.assertEqual(
            [attempt["aic_system"] for attempt in attempts[-2:]],
            ["first_system", "second_system"],
        )

    def test_joint_profile_search_runs_real_online_dynosim(self):
        predictor = SurrogatePrediction()
        config = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_architecture": "Qwen2ForCausalLM",
            "model_params_b": 7.6,
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_attn_heads": 28,
            "num_kv_heads": 4,
            "head_dim": 128,
            "vocab_size": 152064,
            "max_pos_embeddings": 32768,
            "gpu_type": "H100",
            "gpu_vendor": "nvidia",
            "gpu_generation": "hopper",
            "gpu_mem_gb": 80,
            "gpu_bandwidth_gbps": 3350,
            "gpu_tflops_fp16": 989,
            "nvlink_bandwidth_gbps": 900,
            "weight_dtype": "fp16",
            "activation_dtype": "fp16",
            "kvcache_dtype": "fp16",
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "max_num_seq": 8,
            "max_num_batched_tokens": 2048,
            "engine_name": "vllm",
            "engine_version": "0.22.0",
        }
        features = {
            "type": "online",
            "_traffic_mode": "request_rate",
            "isl_token_avg": 512,
            "osl_token_avg": 170,
            "request_arrival_rate": 2.0,
            "target_p99_ttft_ms": 1000,
            "target_p99_tpot_ms": 100,
        }

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            y_hat, _ = predictor.compose_prediction(config, features, MockCandidateGraph())

        assert y_hat["throughput_token_per_sec"] > 0
        assert y_hat["p99_ttft_ms"] >= 0
        assert y_hat["p99_tpot_ms"] >= 0
        assert predictor.last_metadata["aic_profile_match"]["profile_id"]

    def test_aic_fallback_preserves_infrastructure_failure_classification(self):
        def unavailable(**_kwargs):
            raise ImportError("AIC runtime unavailable")

        predictor = SurrogatePrediction()
        predictor._aic_estimate = unavailable
        surrogate_input = {
            "engine_args": {
                "aic_model_path": "model",
                "aic_system": "h100_sxm",
                "aic_backend": "vllm",
            },
            "replay_args": {
                "input_tokens": 512,
                "output_tokens": 170,
                "request_count": 3,
            },
        }

        with self.assertRaisesRegex(SurrogateExecutionError, "AIC runtime unavailable"):
            predictor._run_aic_modes(surrogate_input, ("HYBRID", "EMPIRICAL", "SOL"))

    def test_aic_unsupported_dtype_does_not_run_default_precision(self):
        predictor = SurrogatePrediction()
        predictor.last_metadata = {
            "compatibility": {
                "weights_dtype": {
                    "kind": "unsupported",
                    "resolved": None,
                }
            }
        }
        predictor.run_aic_dynosim = lambda *_args, **_kwargs: self.fail(
            "unsupported dtype reached DynoSim"
        )

        with self.assertRaisesRegex(SurrogateUnsupportedConfig, "compatible dtype"):
            predictor.run_surrogate({}, ("AIC_DynoSim",))

    def test_aic_nondefault_dtype_uses_direct_mode(self):
        predictor = SurrogatePrediction()
        predictor.last_metadata = {
            "compatibility": {
                "weights_dtype": {
                    "kind": "exact",
                    "resolved": "fp8",
                }
            }
        }
        predictor.run_aic_only = lambda *_args, **_kwargs: ({"source": "direct"}, {})
        predictor.run_aic_dynosim = lambda *_args, **_kwargs: self.fail(
            "nondefault dtype reached DynoSim"
        )

        y_hat, _ = predictor.run_surrogate({}, ("AIC_DynoSim",))

        self.assertEqual(y_hat["source"], "direct")

    def test_profile_match_routes_dtype_proxy_through_dynosim(self):
        predictor = SurrogatePrediction()
        predictor.last_metadata = {
            "compatibility": {
                "kv_cache_dtype": {
                    "kind": "nearest",
                    "canonical": "fp8",
                    "resolved": "bf16",
                }
            }
        }
        predictor.run_aic_dynosim = lambda *_args, **_kwargs: ({"source": "dynosim"}, {})
        predictor.run_aic_only = lambda *_args, **_kwargs: self.fail(
            "profile-backed dtype proxy bypassed DynoSim"
        )

        y_hat, _ = predictor.run_surrogate(
            {"aic_profile_matches": ({"profile_id": "proxy"},)},
            ("AIC_DynoSim",),
        )

        self.assertEqual(y_hat["source"], "dynosim")

    def test_aic_replay_required_outputs_must_be_valid(self):
        predictor = SurrogatePrediction()
        raw_report = {
            "completed_requests": 1,
            "total_input_tokens": 10,
            "total_output_tokens": 6,
            "p99_ttft_ms": 1.0,
            "p99_tpot_ms": 2.0,
            "output_throughput_tok_s": 100.0,
        }

        invalid_cases = (
            ("p99_ttft_ms", None),
            ("p99_tpot_ms", -1.0),
            ("p99_tpot_ms", float("nan")),
            ("output_throughput_tok_s", 0.0),
            ("output_throughput_tok_s", float("inf")),
        )
        for key, value in invalid_cases:
            with self.subTest(key=key, value=value):
                report = dict(raw_report)
                report[key] = value
                with self.assertRaisesRegex(SurrogateExecutionError, f"invalid {key}"):
                    predictor.canonicalize_aic_dynosim_output(report, expected_requests=1)

    def test_surrogate_full_dynosim_smoke(self):
        predictor = SurrogatePrediction(objective="batch")
        direct_x, derive_x, direct_v, derive_v, direct_y, derive_y = (
            predictor.resolve_prediction_scope(MockCandidateGraph(), "AIC_DynoSim")
        )
        job_config = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
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
            "price_per_hour": 98.32,
        }
        job_features = {
            "type": "batch",
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
            objective="batch",
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
        self.assertEqual(surrogate_input["engine_args"]["aic_backend_version"], "0.22.0")

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

    def test_surrogate_full_aic_only_pp_smoke(self):
        predictor = SurrogatePrediction(objective="batch")
        job_config = {
            "model_id": "nvidia/Llama-3.1-8B-Instruct-FP8",
            "engine_name": "vllm",
            "tp": 1,
            "pp": 2,
            "ep": 1,
            "block_size": 64,
            "max_num_seq": 8,
            "max_num_batched_tokens": 512,
            "gpu_mem_util": 0.9,
            "price_per_hour": 98.32,
        }
        job_features = {
            "type": "batch",
            "cloud": "aws",
            "region": "us-east-1",
            "market": "reserved",
            "zone": "use1-az1",
            "gpu_type": "H200",
            "instance_type": "p5e.48xlarge",
            "isl_token_avg": 128,
            "osl_token_avg": 32,
            "target_p99_ttft_ms": 200,
            "target_p99_tpot_ms": 10,
        }
        real_aic_estimate = predictor._aic_estimate

        def sol_aic_estimate(**kwargs):
            return real_aic_estimate(**{**kwargs, "database_mode": "SOL"})

        predictor._aic_estimate = sol_aic_estimate

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            y_hat, v_hat = predictor.compose_prediction(
                job_config=job_config,
                job_features=job_features,
                candidate_graph=MockCandidateGraph(),
                method=("AIC_DynoSim",),
            )

        self.assertGreater(y_hat["p99_ttft_ms"], 0.0)
        self.assertGreater(y_hat["p99_tpot_ms"], 0.0)
        self.assertGreater(y_hat["throughput_token_per_sec"], 0.0)
        self.assertTrue(math.isfinite(y_hat["p99_ttft_ms"]))
        self.assertTrue(math.isfinite(y_hat["p99_tpot_ms"]))
        self.assertTrue(math.isfinite(y_hat["throughput_token_per_sec"]))
        self.assertIn("cost_per_token", y_hat)
        self.assertEqual(v_hat["input_length_observed"], 128.0)
        self.assertEqual(v_hat["output_length_observed"], 32.0)
        self.assertGreater(v_hat["completed_requests"], 0)
        self.assertEqual(v_hat["kv_pressure_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
