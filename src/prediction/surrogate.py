import logging
import math

from aiconfigurator.sdk.memory import (  # type: ignore[import-untyped]
    estimate_num_gpu_blocks,
)

from src.prediction.compatibility import (
    backend_dtype_name,
    canonicalize_dtype,
    gpu_profile_from_values,
    resolve_dtype,
    resolve_gpu,
)
from src.prediction.normalization import normalize_candidate_inputs
from src.prediction.profile_search import prediction_profile_from_values, rank_profiles

for _logger_name in ("aiconfigurator", "aiconfigurator_core"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

ONLINE_REPLAY_WINDOW_S = 1
ONLINE_MIN_REQUESTS = 20
_AIC_FALLBACK_MODES = ("HYBRID", "EMPIRICAL", "SOL")
_AIC_GPU_CHOICES = frozenset(
    {
        "A30",
        "A100",
        "A100 80GB",
        "A100 PCIE",
        "B200",
        "B300",
        "B60",
        "GB200",
        "GB300",
        "H100",
        "H100 PCIE",
        "H200",
        "L4",
        "L40S",
        "RTX PRO 6000",
    }
)
_AIC_DTYPES = frozenset({"bf16", "fp8", "fp8_e4m3", "fp8_e5m2", "int8"})

AIC_MEMORY_X_FIELDS = frozenset(
    {
        "gpu_mem_gb",
        "gpu_mem_util",
        "tp",
        "pp",
        "aic_attention_dp_size",
        "ep",
        "gemm_quant_mode",
        "moe_quant_mode",
        "kvcache_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
        "kvcache_dtype",
        "weight_dtype",
        "activation_dtype",
    }
)
AIC_PERFORMANCE_X_FIELDS = frozenset(
    {
        "model_id",
        "gpu_type",
        "engine_name",
        "block_size",
        "max_num_seq",
        "max_num_batched_tokens",
        "tp",
        "ep",
        "prefix_cache_enabled",
        "chunked_prefill_enable",
        "preemption_policy",
        "scheduling_policy",
    }
)
DYNOSIM_WORKLOAD_X_FIELDS = frozenset(
    {
        "isl_token_avg",
        "osl_token_avg",
        "request_arrival_rate",
        "max_concurrent_streaming",
        "max_concurrent_requests",
        "concurrency",
        "workload_prefix_concentration",
        "is_session_affinity",
        "peak_to_mean_ratio",
        "multi_turn_ratio",
        "multi_turn_avg_turns",
        "pd_enabled",
        "prefill_worker_count",
        "decode_worker_count",
        "router_policy",
        "num_workers",
        "dp",
    }
)
POST_PROCESSING_X_FIELDS = frozenset(
    {
        "max_num_batched_tokens",
        "pd_enabled",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
        "tp",
        "pp",
        "ep",
    }
)
PROFILE_X_FIELDS = frozenset(
    {
        "activation_dtype",
        "architectures",
        "effective_batch_size",
        "gpu_bandwidth_gbps",
        "gpu_generation",
        "gpu_vendor",
        "gpu_tflops_fp16",
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "is_moe",
        "model_architecture",
        "model_config",
        "model_config_json",
        "model_metadata",
        "model_params_b",
        "max_pos_embeddings",
        "num_active_experts",
        "num_attention_heads",
        "num_attn_heads",
        "num_experts",
        "num_hidden_layers",
        "num_key_value_heads",
        "num_kv_heads",
        "num_layers",
        "num_routed_experts",
        "nvlink_bandwidth_gbps",
        "pcie_bandwidth_gbps",
        "context",
        "type",
        "vocab_size",
        "params_billion",
    }
)
SURROGATE_CONSUMED_X_FIELDS = (
    AIC_MEMORY_X_FIELDS
    | AIC_PERFORMANCE_X_FIELDS
    | DYNOSIM_WORKLOAD_X_FIELDS
    | POST_PROCESSING_X_FIELDS
    | PROFILE_X_FIELDS
)


class SurrogateMemoryNoFit(Exception):
    pass


class SurrogateUnsupportedConfig(Exception):
    pass


class SurrogateExecutionError(Exception):
    pass


class SurrogatePrediction:
    def __init__(self, objective="batch"):
        self.objective = objective
        self.last_metadata = {}
        self._failed_profile_slices = set()

    def compose_prediction(
        self, job_config, job_features, candidate_graph, method=("AIC_DynoSim",), scenario="mean"
    ):
        self.last_metadata = {}
        env_vector = self.get_env_row(job_features)
        price_vector = self.fetch_cloud_prices(job_config, job_features, env_vector)
        # 1. Resolve what this surrogate is allowed to use/produce in the prediction
        direct_x, _derive_x, _direct_v, derive_v, _direct_y, derive_y = (
            self.resolve_prediction_scope(candidate_graph, method)
        )
        # field names only

        # 2. Pull only X values consumed by this surrogate stack.
        direct_x_values = self.extract_x_values(
            set(direct_x) | SURROGATE_CONSUMED_X_FIELDS,
            job_config,
            job_features,
            env_vector,
        )
        model_id = job_config.get("model_id") or job_features.get("model_id")
        if model_id is None:
            raise SurrogateUnsupportedConfig("AIC_DynoSim needs model_id")
        direct_x_values["model_id"] = model_id
        # field names -> actual values from job_config/env_vector

        # 3. Add simulator-only controls that are NOT in the DAG
        # num_requests, replay_concurrency, arrival_interval_ms, replay_mode="offline/online"
        # this is a very AIC/DynoSim specific control
        # so TODO - maybe we can move into a different function?
        # to maintain a very modular architecture
        objective = self._prediction_objective(job_features)
        simulator_controls = self._build_simulator_controls(
            objective,
            job_config,
            job_features,
            direct_x_values,
            scenario=scenario,
        )
        # objective-specific DynoSim controls, not DAG nodes

        # 4. Translate everything into DynoSim/AIC argument names
        surrogate_input = self.build_surrogate_inputs(
            direct_x_values,
            simulator_controls,
            method,
        )
        surrogate_input["objective"] = objective
        surrogate_input["scenario"] = scenario
        # direct_x_values + simulator_controls -> AIC_dynosim args

        # 5. Run DynoSim/AIC and get direct outputs
        y_hat_direct, v_hat_direct = self.run_surrogate(
            surrogate_input,
            method,
        )
        y_hat_direct = y_hat_direct or {}
        v_hat_direct = v_hat_direct or {}
        # execute simulator

        # 6. Later: use derive_x + direct outputs to compute derived outputs
        y_hat_derived, v_hat_derived = self.derive_outputs(
            derive_v,
            derive_y,
            y_hat_direct,
            v_hat_direct,
            job_config,
            job_features,
            price_vector,
            surrogate_input["replay_args"],
        )
        # post process direct outputs to get derived outputs

        # 7. Later: merge direct + derived
        y_hat = self.merge_outputs(y_hat_direct, y_hat_derived)
        v_hat = self.merge_outputs(v_hat_direct, v_hat_derived)
        # final output
        return y_hat, v_hat

    def get_env_row(self, job_features):
        # Fetch the Env and the cloud we want for the prediction
        # Inputs: JobFeatures[Environment, Hardware]
        # Outputs: EnvVector
        env_vector = {
            "cloud": job_features.get("cloud"),
            "region": job_features.get("region"),
            "zone": job_features.get("zone"),
            "market": job_features.get("market"),
            "gpu_type": job_features.get("gpu_type"),
            "instance_type": job_features.get("instance_type"),
            "num_nodes_per_chain": job_features.get("num_nodes_per_chain"),
            "interconnect_type": job_features.get("interconnect_type"),
        }
        return env_vector

    def extract_x_values(self, direct_x, job_config, job_features, env_vector):
        # Convert direct X field names into actual values.
        # Priority: JobConfig > JobFeatures > EnvVector.
        direct_x_values = {}

        for x_name in direct_x:
            if x_name in job_config and job_config[x_name] is not None:
                direct_x_values[x_name] = job_config[x_name]
            elif x_name in job_features and job_features[x_name] is not None:
                direct_x_values[x_name] = job_features[x_name]
            elif x_name in env_vector and env_vector[x_name] is not None:
                direct_x_values[x_name] = env_vector[x_name]

        return direct_x_values

    def map_gpu_to_aic_system(self, gpu_type, values=None):
        resolution = resolve_gpu(
            gpu_type,
            backend="aic",
            available=_AIC_GPU_CHOICES,
            requested_profile=values,
            allow_larger_memory_proxy=True,
        )
        self.last_metadata.setdefault("compatibility", {})["gpu"] = resolution.to_dict()
        if not resolution.supported or resolution.backend_value is None:
            raise SurrogateUnsupportedConfig(f"No compatible AIC system for gpu_type={gpu_type}")
        return resolution.backend_value

    def resolve_prediction_scope(self, candidate_graph, method):
        # Resolve the prediction scope for the surrogate stack
        # The Idea is to include only the features that have SOME chance of being used
        # in the prediction model.
        # Inputs: CandidateGraph, Method
        # Outputs: (Direct_X, Derive_X, Direct_V, Derive_V, Direct_Y, Derive_Y)
        method_name = method[0] if isinstance(method, (list, tuple)) else method
        candidate_x = set(candidate_graph.x)
        candidate_v = set(candidate_graph.v)
        candidate_y = set(candidate_graph.y)
        method_scope = {
            "AIC_DynoSim": {
                "direct_x": {
                    "gpu_type",
                    "engine_name",
                    "engine_version",
                    "tp",
                    "ep",
                    "block_size",
                    "max_num_seq",
                    "max_num_batched_tokens",
                    "gpu_mem_util",
                    "prefix_cache_enabled",
                    "chunked_prefill_enable",
                    "pd_enabled",
                    "prefill_worker_count",
                    "decode_worker_count",
                    "kv_transfer_method",
                    "preemption_policy",
                    "router_policy",
                    "isl_token_avg",
                    "osl_token_avg",
                    "request_arrival_rate",
                    "workload_prefix_concentration",
                    "shared_prefix_length_avg",
                    "is_session_affinity",
                },
                "derive_x": {
                    "cloud",
                    "region",
                    "zone",
                    "market",
                    "instance_type",
                    "interconnect_type",
                    "num_nodes_per_chain",
                    "target_p99_ttft_ms",
                    "target_p99_tpot_ms",
                    "gpu_mem_gb",
                    "gpu_bandwidth_gbps",
                    "gpu_tflops_fp16",
                    "nvlink_bandwidth_gbps",
                    "pcie_bandwidth_gbps",
                    "internode_bandwidth_gbps",
                    "gpu_watts",
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
                    "weight_dtype",
                    "kvcache_dtype",
                    "weight_quantization_bits",
                },
                "direct_v": {"input_length_observed", "output_length_observed", "kvcache_hit_rate"},
                "derive_v": {
                    "gpu_mem_used_fraction",
                    "kv_cache_util",
                    "vram_headroom_gb",
                    "kv_pressure_score",
                    "per_tok_comm_bytes",
                    "comm_overhead_pct",
                    "pd_inbalance",
                },
                "direct_y": {"p99_ttft_ms", "p99_tpot_ms", "throughput_token_per_sec"},
                "derive_y": {"cost_per_token", "slo_margin"},
            }
        }
        if method_name not in method_scope:
            raise SurrogateUnsupportedConfig(f"Unsupported surrogate method: {method_name}")

        scope = method_scope[method_name]
        direct_x = sorted(candidate_x & scope["direct_x"])
        derive_x = sorted(candidate_x & scope["derive_x"])

        direct_v = sorted(candidate_v & scope["direct_v"])
        derive_v = sorted(candidate_v & scope["derive_v"])

        direct_y = sorted(candidate_y & scope["direct_y"])
        derive_y = sorted(candidate_y & scope["derive_y"])

        return (direct_x, derive_x, direct_v, derive_v, direct_y, derive_y)

    def fetch_cloud_prices(self, *sources):
        """Return explicit hourly allocation price supplied by Koi, or None.

        The resource map owns pricing. Do not invent a cloud fallback here: a fake
        shared price makes faster GPUs look artificially cheap per token.
        """
        for source in sources:
            price = self.get_price_per_hour(source)
            if price is not None:
                return {"price_per_hour": price}
        return None

    @staticmethod
    def _prediction_objective(job_features):
        objective = str((job_features or {}).get("type") or "").lower()
        if objective not in {"online", "batch"}:
            raise ValueError("prediction job_features['type'] must be 'online' or 'batch'")
        return objective

    def _build_simulator_controls(
        self, objective, job_config, job_features, direct_x_values, scenario="mean"
    ):
        # Build DynoSim run controls. These are not DAG X values.
        # Inputs: objective, JobConfig, JobFeatures, direct_x_values
        # Outputs: simulator_controls
        if objective == "batch":
            sources = (direct_x_values, job_features, job_config)
            isl = self._first_positive(sources, "isl_token_avg", "input_len_tokens_avg")
            osl = self._first_positive(sources, "osl_token_avg", "output_len_tokens_avg")
            max_num_seq = direct_x_values.get("max_num_seq")
            max_num_batched_tokens = direct_x_values.get("max_num_batched_tokens")

            if max_num_seq is None or max_num_batched_tokens is None:
                raise ValueError("Batch simulation needs max_num_seq and max_num_batched_tokens")
            if isl is None or osl is None:
                raise ValueError("Batch simulation needs positive input/output token lengths")

            tokens_per_request = isl + osl
            target_concurrency = int(
                min(
                    max_num_seq,
                    max_num_batched_tokens / tokens_per_request,
                )
            )
            target_concurrency = max(1, target_concurrency)
            sim_num_waves = 20  # TODO - hardcoded for now, need discussion

            return {
                "request_count": target_concurrency * sim_num_waves,
                "replay_concurrency": target_concurrency,
                "arrival_interval_ms": 0.0,
                "replay_mode": "offline",
                "expected_completed_requests": target_concurrency * sim_num_waves,
            }

        if objective == "online":
            # Online workload semantics, offline replay execution: DynoSim uses a
            # logical clock instead of live async workers. This is not batch/offline serving.
            scenario = str(scenario or "mean")
            if scenario not in {"mean", "peak", "peak_all_multiturn_stress"}:
                raise ValueError(f"Unknown online replay scenario: {scenario!r}")
            raw_traffic_mode = job_features.get("_traffic_mode") or job_config.get("_traffic_mode")
            request_arrival_rate = self._first_positive(
                (direct_x_values, job_features, job_config), "request_arrival_rate"
            )
            has_concurrency = (
                self._first_positive(
                    (direct_x_values, job_features, job_config),
                    "max_concurrent_streaming",
                    "max_concurrent_requests",
                    "concurrency",
                )
                is not None
            )
            if raw_traffic_mode is None and request_arrival_rate is not None and has_concurrency:
                raise ValueError("online job requires explicit _traffic_mode")
            traffic_mode = str(raw_traffic_mode or "request_rate")
            if traffic_mode not in {"request_rate", "concurrency"}:
                raise ValueError("online job requires explicit _traffic_mode")

            if traffic_mode == "concurrency":
                replay_concurrency = self._first_positive(
                    (direct_x_values, job_features, job_config),
                    "max_concurrent_streaming",
                    "max_concurrent_requests",
                    "concurrency",
                )
                if replay_concurrency is None:
                    raise ValueError("Online concurrency replay needs positive max concurrency")
                replay_concurrency = max(1, math.ceil(replay_concurrency))
                return {
                    "request_count": replay_concurrency * 20,
                    "replay_concurrency": replay_concurrency,
                    "replay_mode": "offline",
                    "turns_per_session": 1,
                    "expected_completed_requests": replay_concurrency * 20,
                }
            if request_arrival_rate is None:
                raise ValueError("Online simulation needs positive request_arrival_rate")

            peak_to_mean_ratio = (
                self._first_positive(
                    (direct_x_values, job_features, job_config), "peak_to_mean_ratio"
                )
                or 1.0
            )
            turn_rate = request_arrival_rate * (peak_to_mean_ratio if scenario != "mean" else 1.0)
            if turn_rate <= 0:
                raise ValueError("Online simulation needs positive turn rate")
            turns_per_session = 1
            if scenario == "peak_all_multiturn_stress":
                multi_turn_ratio = (
                    self._first_positive(
                        (direct_x_values, job_features, job_config), "multi_turn_ratio"
                    )
                    or 0.0
                )
                if multi_turn_ratio <= 0:
                    raise ValueError("peak_all_multiturn_stress requires multi_turn_ratio > 0")
                multi_turn_avg_turns = (
                    self._first_positive(
                        (direct_x_values, job_features, job_config), "multi_turn_avg_turns"
                    )
                    or 2.0
                )
                turns_per_session = max(2, math.ceil(multi_turn_avg_turns))

                # request_arrival_rate counts emitted turns/sec, but DynoSim's
                # arrival_interval_ms spaces new sessions when turns_per_session > 1.
                # Divide turn rate by turns/session so the stress scenario does not
                # multiply the offered traffic.
                # TODO: DynoSim needs one integer turn count per synthetic session;
                # v0 uses all-multiturn peak stress instead of mixing single- and
                # multi-turn sessions in one replay.
                arrival_interval_ms = 1000.0 * turns_per_session / turn_rate
            else:
                arrival_interval_ms = 1000.0 / turn_rate

            target_emitted_turns = max(
                ONLINE_MIN_REQUESTS,
                math.ceil(turn_rate * ONLINE_REPLAY_WINDOW_S),
            )
            request_count = (
                math.ceil(target_emitted_turns / turns_per_session)
                if scenario == "peak_all_multiturn_stress"
                else target_emitted_turns
            )
            expected_completed_requests = request_count * turns_per_session

            controls = {
                "request_count": request_count,
                "arrival_interval_ms": arrival_interval_ms,
                "replay_mode": "offline",
                "turns_per_session": turns_per_session,
                "expected_completed_requests": expected_completed_requests,
            }
            if scenario == "peak_all_multiturn_stress":
                controls.update(
                    {
                        "inter_turn_delay_ms": 0.0,
                        "shared_prefix_ratio": 0.0,
                        "num_prefix_groups": 0,
                    }
                )
            return controls

        raise ValueError("prediction objective must be 'online' or 'batch'")

    @staticmethod
    def _first_positive(sources, *names):
        for source in sources:
            for name in names:
                value = source.get(name)
                if value is not None and float(value) > 0:
                    return float(value)
        return None

    def build_surrogate_inputs(self, direct_x_values, simulator_controls, method):
        # Translate direct X values + simulator controls into AIC/DynoSim args.
        # Inputs: direct_x_values, simulator_controls, method
        # Outputs: SurrogateInput
        method_name = (
            method[0] if isinstance(method, (list, tuple)) and len(method) == 1 else method
        )

        if method_name != "AIC_DynoSim":
            raise SurrogateUnsupportedConfig(
                f"Unsupported method or multi method is not supported yet: {method}"
            )

        model_id = direct_x_values.get("model_id")
        if model_id is None:
            raise SurrogateUnsupportedConfig("AIC_DynoSim needs model_id")

        gpu_type = direct_x_values.get("gpu_type")
        if gpu_type is None:
            raise SurrogateUnsupportedConfig("AIC_DynoSim needs gpu_type")

        pp = int(direct_x_values.get("pp") or 1)
        aic_only = pp != 1
        if aic_only and direct_x_values.get("pd_enabled", False):
            raise SurrogateUnsupportedConfig(
                "AIC-only PP prediction supports aggregate serving only"
            )

        profile_matches = self._rank_aic_profiles(direct_x_values)
        selected_match = profile_matches[0] if profile_matches else None
        resolved_system = (
            selected_match.supported.aic_system
            if selected_match is not None
            else self.map_gpu_to_aic_system(gpu_type, direct_x_values)
        )
        resolved_model = (
            selected_match.supported.model.model_id if selected_match is not None else model_id
        )
        engine_args = {
            "engine_type": direct_x_values.get("engine_name", "vllm"),
            "block_size": direct_x_values.get("block_size", 64),
            "max_num_seqs": direct_x_values.get("max_num_seq"),
            "max_num_batched_tokens": direct_x_values.get("max_num_batched_tokens"),
            "aic_backend": direct_x_values.get("engine_name", "vllm"),
            # AIC's bundled performance database currently supports this version.
            "aic_backend_version": "0.22.0",
            "aic_system": resolved_system,
            "aic_model_path": resolved_model,
            "aic_tp_size": direct_x_values.get("tp", 1),
            "aic_moe_ep_size": direct_x_values.get("ep", 1),
            "enable_prefix_caching": direct_x_values.get("prefix_cache_enabled", False),
            "enable_chunked_prefill": direct_x_values.get("chunked_prefill_enable", False),
            "preemption_mode": direct_x_values.get("preemption_policy"),
        }
        if selected_match is not None:
            engine_args["speedup_ratio"] = selected_match.prefill_speed_ratio
            engine_args["decode_speedup_ratio"] = (
                selected_match.decode_speed_ratio / selected_match.prefill_speed_ratio
            )
            self.last_metadata["aic_profile_match"] = selected_match.to_dict()
            self.last_metadata["aic_profile_candidates"] = [
                match.to_dict() for match in profile_matches
            ]
            self.last_metadata["engine_resolution"] = {
                "requested_name": direct_x_values.get("engine_name"),
                "requested_version": direct_x_values.get("engine_version"),
                "resolved_name": selected_match.supported.engine_name,
                "resolved_version": selected_match.supported.engine_version,
            }
        self._resolve_aic_dtypes(direct_x_values)
        if any(
            resolution.get("kind") == "unsupported"
            for name, resolution in (self.last_metadata.get("compatibility") or {}).items()
            if name.endswith("_dtype")
        ):
            raise SurrogateUnsupportedConfig("AIC has no compatible dtype for this candidate")
        memory_x_values = {
            key: direct_x_values[key]
            for key in AIC_MEMORY_X_FIELDS
            if direct_x_values.get(key) is not None
        }
        if memory_x_values.get("gpu_mem_gb") is None:
            requested_profile = gpu_profile_from_values(str(gpu_type), direct_x_values)
            if requested_profile is not None and requested_profile.memory_gb is not None:
                memory_x_values["gpu_mem_gb"] = requested_profile.memory_gb
        memory_engine_args = {**engine_args, "aic_model_path": model_id}
        self._resolve_aic_num_gpu_blocks(memory_engine_args, memory_x_values)
        engine_args["num_gpu_blocks"] = memory_engine_args["num_gpu_blocks"]

        queue_policy = direct_x_values.get("scheduling_policy")
        if queue_policy in {"fcfs", "lcfs", "wspt"}:
            engine_args["router_queue_policy"] = queue_policy

        num_workers = int(direct_x_values.get("num_workers") or direct_x_values.get("dp") or 1)
        router_mode = self._router_mode_for_replay(
            direct_x_values.get(
                "router_policy",
                "kv_router" if direct_x_values.get("pd_enabled", False) else "round_robin",
            ),
            simulator_controls.get("replay_mode", "offline"),
            num_workers,
        )

        replay_args = {
            "input_tokens": direct_x_values.get("isl_token_avg"),
            "output_tokens": direct_x_values.get("osl_token_avg"),
            # TODO: map workload_prefix_concentration/shared_prefix_length_avg to DynoSim
            # prefix groups once replay prefix semantics are calibrated.
            # "shared_prefix_ratio": direct_x_values.get("workload_prefix_concentration", 0.0),
            "shared_prefix_ratio": 0.0,
            "num_prefix_groups": 0,
            "turns_per_session": 1,
            "pd_enabled": direct_x_values.get("pd_enabled", False),
            "prefill_worker_count": direct_x_values.get("prefill_worker_count", 1),
            "decode_worker_count": direct_x_values.get("decode_worker_count", 1),
            "num_workers": num_workers,
            "router_mode": router_mode,
            **simulator_controls,
        }
        if (
            replay_args.get("expected_completed_requests") is None
            and replay_args.get("request_count") is not None
        ):
            turns_per_session = int(replay_args.get("turns_per_session") or 1)
            replay_args["expected_completed_requests"] = (
                int(replay_args["request_count"]) * turns_per_session
            )

        engine_args = {key: value for key, value in engine_args.items() if value is not None}
        replay_args = {key: value for key, value in replay_args.items() if value is not None}
        aic_args = {
            "pp_size": pp,
            "gemm_quant_mode": self._resolved_aic_dtype("weights_dtype"),
            "fmha_quant_mode": self._resolved_aic_dtype("fmha_dtype"),
            "kvcache_quant_mode": self._resolved_aic_dtype("kv_cache_dtype"),
        }
        aic_args = {key: value for key, value in aic_args.items() if value is not None}

        return {
            "method": method_name,
            "aic_only": aic_only,
            "aic_args": aic_args,
            "engine_args": engine_args,
            "replay_args": replay_args,
            "aic_profile_matches": tuple(match.to_dict() for match in profile_matches),
        }

    def _rank_aic_profiles(self, values):
        try:
            from src.prediction.aic_support import load_aic_support_profiles

            _, normalized_values = normalize_candidate_inputs({}, values)
            normalized_values = {**values, **normalized_values}
            requested_gpu = gpu_profile_from_values(
                str(normalized_values.get("gpu_type") or ""), normalized_values
            )
            if requested_gpu is None:
                return ()
            requested = prediction_profile_from_values(normalized_values, requested_gpu)
            if requested is None:
                return ()
            available = load_aic_support_profiles(
                str(normalized_values.get("engine_name") or "vllm"),
                "0.22.0",
            )
            return rank_profiles(requested, available, limit=5)
        except Exception as exc:
            self.last_metadata["aic_profile_search"] = {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return ()

    def _resolve_aic_dtypes(self, values):
        requested = {
            "weights": values.get("weight_dtype"),
            "fmha": values.get("fmha_quant_mode")
            or values.get("activation_dtype")
            or values.get("weight_dtype"),
            "kv_cache": values.get("kvcache_quant_mode") or values.get("kvcache_dtype"),
        }
        compatibility = self.last_metadata.setdefault("compatibility", {})
        for component, dtype in requested.items():
            if dtype is None or canonicalize_dtype(dtype) == "auto":
                continue
            compatibility[f"{component}_dtype"] = resolve_dtype(
                dtype,
                backend="aic",
                component=component,
                available=_AIC_DTYPES if component == "weights" else frozenset({"bf16"}),
            ).to_dict()

    def _resolved_aic_dtype(self, name):
        resolution = (self.last_metadata.get("compatibility") or {}).get(name) or {}
        return resolution.get("backend_value") if resolution.get("supported", True) else None

    def _requested_aic_dtype(self, name):
        resolution = (self.last_metadata.get("compatibility") or {}).get(name) or {}
        canonical = resolution.get("canonical")
        if canonical == "fp16":
            canonical = "bf16"
        return backend_dtype_name(canonical, "aic") if canonical else None

    def _resolve_aic_num_gpu_blocks(self, engine_args, memory_x_values):
        """Run AIC's memory fit/KV-capacity estimator before DynoSim replay."""
        if engine_args.get("num_gpu_blocks") is not None or not engine_args.get("aic_backend"):
            return
        if not engine_args.get("aic_model_path"):
            raise SurrogateUnsupportedConfig(
                "AIC memory preflight unsupported config: missing aic_model_path"
            )

        try:
            blocks = self._estimate_num_gpu_blocks(
                model_path=engine_args["aic_model_path"],
                system=engine_args.get("aic_system"),
                backend=engine_args["aic_backend"],
                backend_version=engine_args.get("aic_backend_version"),
                scheduler_block_size=int(engine_args.get("block_size") or 64),
                max_num_tokens=int(engine_args.get("max_num_batched_tokens") or 8192),
                max_batch_size=int(engine_args.get("max_num_seqs") or 256),
                memory_fraction_kind=self._memory_fraction_kind(engine_args["aic_backend"]),
                memory_fraction_value=float(memory_x_values.get("gpu_mem_util") or 0.9),
                tp_size=int(engine_args.get("aic_tp_size") or 1),
                pp_size=int(memory_x_values.get("pp") or 1),
                # Not Koi X today; AIC uses attention_dp=1 unless explicitly injected.
                # attention_dp_size=int(memory_x_values.get("aic_attention_dp_size") or 1),
                moe_ep_size=engine_args.get("aic_moe_ep_size"),
                # Not Koi X today; AIC infers these from the model config or defaults.
                gemm_quant_mode=self._requested_aic_dtype("weights_dtype"),
                # moe_quant_mode=memory_x_values.get("moe_quant_mode"),
                kvcache_quant_mode=self._requested_aic_dtype("kv_cache_dtype"),
                fmha_quant_mode=self._requested_aic_dtype("fmha_dtype"),
                # comm_quant_mode=memory_x_values.get("comm_quant_mode"),
                gpu_memory_capacity_bytes_override=self._gpu_memory_capacity_bytes(memory_x_values),
                allow_naive_fallback=False,
                allow_hf_config_download=False,
            )
        except (SurrogateMemoryNoFit, SurrogateUnsupportedConfig, SurrogateExecutionError):
            raise
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            raise self._memory_preflight_error(exc) from exc
        if int(blocks) <= 0:
            raise SurrogateMemoryNoFit(f"AIC memory preflight no-fit: num_gpu_blocks={blocks}")
        engine_args["num_gpu_blocks"] = int(blocks)

    @staticmethod
    def _memory_preflight_error(exc):
        text = str(exc).lower()
        type_name = type(exc).__name__.lower()
        if (
            "no kv budget" in text
            or "insufficientmemory" in type_name
            or "kvcachecapacity" in type_name
        ):
            return SurrogateMemoryNoFit(f"AIC memory preflight no-fit: {exc}")
        if "unsupported" in text or "not supported" in text or "unknown backend" in text:
            return SurrogateUnsupportedConfig(f"AIC memory preflight unsupported config: {exc}")
        return SurrogateExecutionError(f"AIC memory preflight execution failed: {exc}")

    @staticmethod
    def _estimate_num_gpu_blocks(**kwargs):
        return estimate_num_gpu_blocks(**kwargs)

    @staticmethod
    def _memory_fraction_kind(backend):
        if backend == "trtllm":
            return "of_free"
        if backend in {"vllm", "sglang"}:
            return "of_total"
        raise SurrogateUnsupportedConfig(f"unknown backend {backend!r} for AIC memory preflight")

    @staticmethod
    def _gpu_memory_capacity_bytes(direct_x_values):
        gpu_mem_gb = direct_x_values.get("gpu_mem_gb")
        if gpu_mem_gb is None:
            return None
        return int(float(gpu_mem_gb) * (1 << 30))

    @staticmethod
    def _router_mode_for_replay(router_mode, replay_mode, num_workers):
        """Return a DynoSim-valid router mode for the replay call.

        TODO: Find a better / more realistic solution for this. Koi currently
        predicts one worker per rank and scales replicas outside DynoSim; offline
        ``kv_router`` requires multiple workers, and with one worker it is
        equivalent to ``round_robin`` anyway.
        """
        if replay_mode == "offline" and router_mode == "kv_router" and int(num_workers) <= 1:
            return "round_robin"
        return router_mode

    def run_surrogate(self, surrogate_input, method, accumulate_logic="average"):
        # Run the surrogate model.
        # Inputs: SurrogateInput, Method=List[DynoSim, LLMSimulator, etc], accumulate_logic: average,llm decides
        # Outputs: y_hat, v_hat
        if len(method) == 1 and method[0] == "AIC_DynoSim":
            dtype_resolutions = [
                resolution
                for name, resolution in (self.last_metadata.get("compatibility") or {}).items()
                if name.endswith("_dtype")
            ]
            if any(resolution.get("kind") == "unsupported" for resolution in dtype_resolutions):
                raise SurrogateUnsupportedConfig("AIC has no compatible dtype for this candidate")
            requires_direct_dtype = any(
                resolution.get("kind") == "nearest"
                or resolution.get("resolved") not in {None, "bf16"}
                for resolution in dtype_resolutions
            )
            # dont accumulate, just run the surrogate model
            if surrogate_input.get("aic_only"):
                return self.run_aic_only(surrogate_input)
            if surrogate_input.get("aic_profile_matches"):
                return self.run_aic_dynosim(surrogate_input)
            if requires_direct_dtype:
                return self.run_aic_only(surrogate_input)
            return self.run_aic_dynosim(surrogate_input)

    def run_aic_only(self, surrogate_input):
        return self._run_aic_modes(surrogate_input, ("SILICON", *_AIC_FALLBACK_MODES))

    def _run_aic_modes(self, surrogate_input, modes, fallback_reason=None):
        failures = []
        mapped_failures = []
        for database_mode in modes:
            try:
                result = self._run_aic_mode(surrogate_input, database_mode)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                mapped = self._aic_only_error(exc)
                if isinstance(mapped, SurrogateMemoryNoFit):
                    raise mapped from exc
                mapped_failures.append(mapped)
                failures.append(f"{database_mode}: {type(exc).__name__}: {exc}")
                continue
            self.last_metadata["aic_database_mode"] = database_mode
            if database_mode != "SILICON":
                self.last_metadata["aic_fallback"] = {
                    "reason": fallback_reason or "silicon_estimate_unavailable",
                    "attempts": failures,
                    "selected_database_mode": database_mode,
                }
            return result
        message = "AIC has no usable silicon, empirical, or SOL estimate: " + "; ".join(failures)
        if any(isinstance(failure, SurrogateExecutionError) for failure in mapped_failures):
            raise SurrogateExecutionError(message)
        raise SurrogateUnsupportedConfig(message)

    def _run_aic_mode(self, surrogate_input, database_mode):
        engine_args = surrogate_input["engine_args"]
        replay_args = surrogate_input["replay_args"]
        aic_args = surrogate_input.get("aic_args") or {}

        input_tokens = int(replay_args["input_tokens"])
        output_tokens = int(replay_args["output_tokens"])
        request_count = int(replay_args["request_count"])
        expected_requests = int(
            replay_args.get("expected_completed_requests")
            or request_count * int(replay_args.get("turns_per_session") or 1)
        )
        num_workers = int(replay_args.get("num_workers") or 1)

        estimate = self._aic_estimate(
            model_path=engine_args["aic_model_path"],
            system_name=engine_args["aic_system"],
            mode="agg",
            backend_name=engine_args["aic_backend"],
            backend_version=engine_args.get("aic_backend_version"),
            database_mode=database_mode,
            isl=input_tokens,
            osl=output_tokens,
            batch_size=self._aic_batch_size(engine_args, replay_args),
            ctx_tokens=int(engine_args.get("max_num_batched_tokens") or input_tokens),
            tp_size=int(engine_args.get("aic_tp_size") or 1),
            pp_size=int(aic_args.get("pp_size") or 1),
            moe_ep_size=engine_args.get("aic_moe_ep_size"),
            gemm_quant_mode=aic_args.get("gemm_quant_mode"),
            fmha_quant_mode=aic_args.get("fmha_quant_mode"),
            kvcache_quant_mode=aic_args.get("kvcache_quant_mode"),
        )
        raw = estimate.raw or {}
        throughput = float(raw.get("tokens/s") or getattr(estimate, "tokens_per_second", 0.0))
        throughput *= num_workers
        prefill_speed, decode_speed = self._phase_compatibility_scales()
        baseline_duration = float(estimate.ttft) + float(estimate.tpot) * output_tokens
        scaled_duration = float(estimate.ttft) / prefill_speed + (
            float(estimate.tpot) * output_tokens / decode_speed
        )
        throughput_speed = baseline_duration / max(
            scaled_duration,
            1e-12,
        )
        raw_report = {
            "completed_requests": expected_requests,
            "total_input_tokens": input_tokens * expected_requests,
            "total_output_tokens": output_tokens * expected_requests,
            "p99_ttft_ms": float(estimate.ttft) / prefill_speed,
            "p99_tpot_ms": float(estimate.tpot) / decode_speed,
            "output_throughput_tok_s": throughput * throughput_speed,
            "prefix_cache_reused_ratio": None,
        }
        y_hat, v_hat = self.canonicalize_aic_dynosim_output(raw_report, expected_requests)
        if surrogate_input.get("objective") == "online":
            y_hat.pop("p99_ttft_ms", None)
            y_hat.pop("p99_tpot_ms", None)
            self.last_metadata["aic_fallback_omitted_nodes"] = ["p99_tpot_ms", "p99_ttft_ms"]
        return y_hat, v_hat

    def _phase_compatibility_scales(self):
        match = self.last_metadata.get("aic_profile_match") or {}
        if match:
            return (
                max(float(match.get("prefill_speed_ratio") or 1.0), 1e-12),
                max(float(match.get("decode_speed_ratio") or 1.0), 1e-12),
            )
        compatibility = self.last_metadata.get("compatibility") or {}
        gpu_scale = float((compatibility.get("gpu") or {}).get("throughput_scale") or 1.0)
        dtype_scales = [
            float(resolution.get("throughput_scale") or 1.0)
            for name, resolution in compatibility.items()
            if name.endswith("_dtype")
        ]
        throughput_scale = gpu_scale * min(dtype_scales, default=1.0)
        return throughput_scale, throughput_scale

    @staticmethod
    def _aic_batch_size(engine_args, replay_args):
        if replay_args.get("replay_concurrency") is not None:
            return max(1, int(replay_args["replay_concurrency"]))
        max_num_seqs = int(engine_args.get("max_num_seqs") or 1)
        max_num_tokens = int(engine_args.get("max_num_batched_tokens") or 0)
        input_tokens = int(replay_args.get("input_tokens") or 0)
        output_tokens = int(replay_args.get("output_tokens") or 0)
        if max_num_tokens > 0 and input_tokens + output_tokens > 0:
            return max(1, min(max_num_seqs, max_num_tokens // (input_tokens + output_tokens)))
        return max(1, max_num_seqs)

    @staticmethod
    def _aic_estimate(**kwargs):
        from aiconfigurator.cli.api import cli_estimate  # type: ignore[import-untyped]

        return cli_estimate(**kwargs)

    @staticmethod
    def _aic_only_error(exc):
        text = str(exc).lower()
        if "oom" in text or "kv cache" in text or "kv-cache" in text or "does not fit" in text:
            return SurrogateMemoryNoFit(f"AIC-only PP no-fit: {exc}")
        if (
            "unsupported" in text
            or "not supported" in text
            or "no database" in text
            or "perfdatanotavailableerror" in text
            or "empiricalnotimplementederror" in text
            or ("performance data" in text and "not available" in text)
        ):
            return SurrogateUnsupportedConfig(f"AIC-only PP unsupported config: {exc}")
        return SurrogateExecutionError(f"AIC-only PP execution failed: {exc}")

    def run_aic_dynosim(self, surrogate_input):
        # Run the AIC DynoSim model.
        # Inputs: SurrogateInput
        # Outputs: y_hat, v_hat
        try:
            from dynamo.llm import MockEngineArgs
            from dynamo.replay.api import run_synthetic_trace_replay
        except Exception as exc:
            return self._run_aic_modes(
                surrogate_input,
                _AIC_FALLBACK_MODES,
                fallback_reason=f"DynoSim unavailable: {type(exc).__name__}: {exc}",
            )

        base_engine_args = surrogate_input["engine_args"]
        replay_args = surrogate_input["replay_args"]

        input_tokens = int(replay_args["input_tokens"])
        output_tokens = int(replay_args["output_tokens"])
        request_count = int(replay_args["request_count"])
        expected_requests = int(
            replay_args.get("expected_completed_requests")
            or request_count * int(replay_args.get("turns_per_session") or 1)
        )
        replay_mode = replay_args.get("replay_mode", "offline")
        router_mode = replay_args.get("router_mode", "round_robin")
        pd_enabled = replay_args.get("pd_enabled", False)

        if pd_enabled and replay_mode == "online":
            raise NotImplementedError(
                "Online PD is not supported by the current DynoSim replay path. "
                "Use offline PD replay for now; add AIC_Direct later for online PD."
            )

        common_replay_args = {
            "replay_mode": replay_mode,
            "router_mode": router_mode,
            "turns_per_session": replay_args.get("turns_per_session", 1),
            "shared_prefix_ratio": replay_args.get("shared_prefix_ratio", 0.0),
        }
        if replay_args.get("num_prefix_groups") is not None:
            common_replay_args["num_prefix_groups"] = replay_args["num_prefix_groups"]
        if replay_args.get("inter_turn_delay_ms") is not None:
            common_replay_args["inter_turn_delay_ms"] = replay_args["inter_turn_delay_ms"]
        if replay_args.get("replay_concurrency") is not None:
            common_replay_args["replay_concurrency"] = replay_args["replay_concurrency"]
        elif replay_args.get("arrival_interval_ms") is not None:
            common_replay_args["arrival_interval_ms"] = replay_args["arrival_interval_ms"]

        configured_matches = tuple(surrogate_input.get("aic_profile_matches") or ())
        matches = tuple(
            match
            for match in configured_matches
            if self._profile_failure_key(match, replay_args, base_engine_args)
            not in self._failed_profile_slices
        )
        if configured_matches and not matches:
            return self._run_aic_modes(
                surrogate_input,
                _AIC_FALLBACK_MODES,
                fallback_reason="all ranked AIC profile slices previously failed",
            )
        attempts = matches or (None,)
        failures = []
        for match in attempts:
            engine_args = self._engine_args_for_profile(base_engine_args, match)
            try:
                raw_report = self._run_dynosim_replay(
                    run_synthetic_trace_replay,
                    MockEngineArgs,
                    engine_args,
                    replay_args,
                    input_tokens,
                    output_tokens,
                    request_count,
                    common_replay_args,
                    pd_enabled,
                )
            # PyO3 PanicException inherits BaseException directly, not Exception.
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                details = f"{type(exc).__name__}: {exc}"
                failures.append(
                    {
                        "profile_id": (match or {}).get("profile_id"),
                        "error": details,
                    }
                )
                if match is not None and self._is_retryable_profile_failure(exc):
                    if self._is_aic_coverage_failure(exc):
                        self._failed_profile_slices.add(
                            self._profile_failure_key(match, replay_args, base_engine_args)
                        )
                    continue
                return self._run_aic_modes(
                    surrogate_input,
                    _AIC_FALLBACK_MODES,
                    fallback_reason=details,
                )
            if match is not None:
                self.last_metadata["aic_profile_match"] = dict(match)
            if failures:
                self.last_metadata["aic_profile_attempts"] = failures
            if match is not None:
                return self.canonicalize_aic_dynosim_output(raw_report, expected_requests)

            prefill_speed, decode_speed = self._phase_compatibility_scales()
            scaled_report = dict(raw_report)
            if scaled_report.get("p99_ttft_ms") is not None:
                scaled_report["p99_ttft_ms"] = float(scaled_report["p99_ttft_ms"]) / prefill_speed
            if scaled_report.get("p99_tpot_ms") is not None:
                scaled_report["p99_tpot_ms"] = float(scaled_report["p99_tpot_ms"]) / decode_speed
            if scaled_report.get("output_throughput_tok_s") is not None:
                scaled_report["output_throughput_tok_s"] = (
                    float(scaled_report["output_throughput_tok_s"]) * decode_speed
                )
            return self.canonicalize_aic_dynosim_output(scaled_report, expected_requests)

        self.last_metadata["aic_profile_attempts"] = failures
        details = "; ".join(attempt["error"] for attempt in failures)
        return self._run_aic_modes(
            surrogate_input,
            _AIC_FALLBACK_MODES,
            fallback_reason=details,
        )

    @staticmethod
    def _engine_args_for_profile(base_engine_args, match):
        engine_args = dict(base_engine_args)
        if match is None:
            return engine_args
        prefill_speed = max(float(match.get("prefill_speed_ratio") or 1.0), 1e-12)
        decode_speed = max(float(match.get("decode_speed_ratio") or 1.0), 1e-12)
        engine_args.update(
            {
                "aic_system": match["aic_system"],
                "aic_model_path": match["model_id"],
                "speedup_ratio": prefill_speed,
                "decode_speedup_ratio": decode_speed / prefill_speed,
            }
        )
        return engine_args

    @staticmethod
    def _run_dynosim_replay(
        run_replay,
        mock_engine_args,
        engine_args,
        replay_args,
        input_tokens,
        output_tokens,
        request_count,
        common_replay_args,
        pd_enabled,
    ):
        if pd_enabled:
            prefill_engine_args = dict(engine_args)
            decode_engine_args = dict(engine_args)
            prefill_engine_args.update({"worker_type": "prefill", "decode_speedup_ratio": 1.0})
            decode_speed = float(engine_args.get("speedup_ratio") or 1.0) * float(
                engine_args.get("decode_speedup_ratio") or 1.0
            )
            decode_engine_args.update(
                {
                    "worker_type": "decode",
                    "speedup_ratio": decode_speed,
                    "decode_speedup_ratio": 1.0,
                }
            )
            return run_replay(
                input_tokens,
                output_tokens,
                request_count,
                prefill_engine_args=mock_engine_args(**prefill_engine_args),
                decode_engine_args=mock_engine_args(**decode_engine_args),
                num_prefill_workers=int(replay_args.get("prefill_worker_count", 1)),
                num_decode_workers=int(replay_args.get("decode_worker_count", 1)),
                **common_replay_args,
            )
        return run_replay(
            input_tokens,
            output_tokens,
            request_count,
            extra_engine_args=mock_engine_args(**engine_args),
            num_workers=int(replay_args.get("num_workers", 1)),
            **common_replay_args,
        )

    @staticmethod
    def _is_aic_coverage_failure(exc):
        text = f"{type(exc).__name__}: {exc}".lower()
        return (
            "perfdatanotavailableerror" in text
            or "no database" in text
            or ("performance data" in text and "not available" in text)
        )

    @classmethod
    def _is_retryable_profile_failure(cls, exc):
        text = f"{type(exc).__name__}: {exc}".lower()
        return cls._is_aic_coverage_failure(exc) or "unsupported model" in text

    @staticmethod
    def _profile_failure_key(match, replay_args, engine_args):
        return (
            match.get("profile_id"),
            int(replay_args.get("input_tokens") or 0),
            int(replay_args.get("output_tokens") or 0),
            int(replay_args.get("replay_concurrency") or engine_args.get("max_num_seqs") or 1),
            int(engine_args.get("max_num_batched_tokens") or 0),
            int(replay_args.get("num_workers") or 1),
            int(engine_args.get("aic_tp_size") or 1),
            int(engine_args.get("aic_moe_ep_size") or 1),
            str(replay_args.get("replay_mode") or "offline"),
            str(replay_args.get("router_mode") or "round_robin"),
            float(replay_args.get("arrival_interval_ms") or 0.0),
            int(replay_args.get("turns_per_session") or 1),
            bool(replay_args.get("pd_enabled", False)),
            int(replay_args.get("prefill_worker_count") or 1),
            int(replay_args.get("decode_worker_count") or 1),
            int(engine_args.get("block_size") or 0),
            int(engine_args.get("num_gpu_blocks") or 0),
            bool(engine_args.get("enable_prefix_caching", False)),
            bool(engine_args.get("enable_chunked_prefill", False)),
            str(engine_args.get("preemption_mode") or ""),
            str(engine_args.get("aic_backend_version") or ""),
        )

    def canonicalize_aic_dynosim_output(self, raw_report, expected_requests):
        # TODO - general helper, can be moved out of this file/class
        # Convert raw DynoSim report keys into DAG V/Y names.
        if "completed_requests" not in raw_report:
            raise SurrogateExecutionError("DynoSim report missing completed_requests")
        completed_requests = int(raw_report["completed_requests"])
        if completed_requests != int(expected_requests):
            raise SurrogateExecutionError(
                f"completed {completed_requests}/{int(expected_requests)} requests"
            )
        p99_ttft_ms = self._required_report_metric(raw_report, "p99_ttft_ms")
        p99_tpot_ms = self._required_report_metric(raw_report, "p99_tpot_ms")
        throughput = self._required_report_metric(
            raw_report, "output_throughput_tok_s", positive=True
        )

        v_hat_direct = {
            "input_length_observed": raw_report.get("total_input_tokens", 0) / completed_requests,
            "output_length_observed": raw_report.get("total_output_tokens", 0) / completed_requests,
            "kvcache_hit_rate": raw_report.get("prefix_cache_reused_ratio"),
            "completed_requests": completed_requests,
        }

        y_hat_direct = {
            "p99_ttft_ms": p99_ttft_ms,
            "p99_tpot_ms": p99_tpot_ms,
            "throughput_token_per_sec": throughput,
        }

        return y_hat_direct, v_hat_direct

    @staticmethod
    def _required_report_metric(raw_report, key, positive=False):
        value = raw_report.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SurrogateExecutionError(f"invalid {key}: {value!r}") from exc
        if not math.isfinite(number) or number < 0 or (positive and number <= 0):
            raise SurrogateExecutionError(f"invalid {key}: {value!r}")
        return number

    def derive_outputs(
        self,
        derive_v,
        derive_y,
        y_hat_direct,
        v_hat_direct,
        job_config,
        job_features,
        price_vector,
        replay_args=None,
    ):
        # Use direct DynoSim outputs + known config/features to compute extra DAG V/Y.
        v_hat_derived = {}
        y_hat_derived = {}
        requested_v = set(derive_v)
        requested_y = set(derive_y)

        input_tokens = v_hat_direct.get("input_length_observed")
        output_tokens = v_hat_direct.get("output_length_observed")
        throughput = y_hat_direct.get("throughput_token_per_sec")

        if input_tokens is not None and output_tokens is not None:
            # TODO: derive kv_cache_util and kv_pressure_score from Profiling DB / Dynamo
            # KV occupancy metrics. Do not estimate them from token length heuristics.
            if "kv_pressure_score" in requested_v:
                v_hat_derived["kv_pressure_score"] = 0.0
            if "kv_cache_util" in requested_v:
                v_hat_derived["kv_cache_util"] = 0.0

        is_single_worker = (
            job_config.get("tp", 1) == 1
            and job_config.get("pp", 1) == 1
            and job_config.get("ep", 1) == 1
            and not job_config.get("pd_enabled", False)
        )

        if is_single_worker:
            if "comm_overhead_pct" in requested_v:
                v_hat_derived["comm_overhead_pct"] = 0.0
            if "per_tok_comm_bytes" in requested_v:
                v_hat_derived["per_tok_comm_bytes"] = 0.0
            if "pd_inbalance" in requested_v:
                v_hat_derived["pd_inbalance"] = 0.0

        if "cost_per_token" in requested_y:
            price_per_hour = self.get_price_per_hour(price_vector)
            if price_per_hour is not None and throughput:
                n_replicas = int((replay_args or {}).get("num_workers") or 1)
                rank_price_per_hour = price_per_hour * n_replicas
                y_hat_derived["cost_per_token"] = rank_price_per_hour / (throughput * 3600.0)

        if "slo_margin" in requested_y:
            ttft_target = job_features.get("target_p99_ttft_ms") or job_config.get(
                "target_p99_ttft_ms"
            )
            tpot_target = job_features.get("target_p99_tpot_ms") or job_config.get(
                "target_p99_tpot_ms"
            )

            ttft_margin = None
            tpot_margin = None
            if ttft_target is not None and y_hat_direct.get("p99_ttft_ms") is not None:
                ttft_margin = ttft_target - y_hat_direct["p99_ttft_ms"]
            if tpot_target is not None and y_hat_direct.get("p99_tpot_ms") is not None:
                tpot_margin = tpot_target - y_hat_direct["p99_tpot_ms"]

            margins = [m for m in (ttft_margin, tpot_margin) if m is not None]
            if margins:
                y_hat_derived["slo_margin"] = min(margins)

        return y_hat_derived, v_hat_derived

    def get_price_per_hour(self, price_vector):
        # Accept a few common pricing shapes until the real pricing helper exists.
        if price_vector is None:
            return None
        if isinstance(price_vector, (int, float)):
            return float(price_vector)
        if not isinstance(price_vector, dict):
            return None

        for key in (
            "price_per_hour",
            "price_per_unit_hour",
            "price_per_instance_hour",
            "hourly_price",
            "usd_per_hour",
            "cost_per_hour",
        ):
            if price_vector.get(key) is not None:
                return float(price_vector[key])

        return None

    def merge_outputs(self, direct_outputs, derived_outputs):
        merged = {}
        merged.update(direct_outputs or {})
        merged.update(derived_outputs or {})
        return merged
