import math

from aiconfigurator_core.sdk.memory import (  # type: ignore[import-untyped]
    estimate_num_gpu_blocks,
)

ONLINE_REPLAY_WINDOW_S = 1
ONLINE_MIN_REQUESTS = 20


class SurrogatePrediction:
    def __init__(self, objective="batched"):
        self.objective = objective  # can be online or batched

    def compose_prediction(
        self, job_config, job_features, candidate_graph, method=("AIC_DynoSim",)
    ):
        env_vector = self.get_env_row(job_features)
        price_vector = self.fetch_cloud_prices(job_config, job_features, env_vector)
        # 1. Resolve what this surrogate is allowed to use/produce in the prediction
        direct_x, _derive_x, _direct_v, derive_v, _direct_y, derive_y = (
            self.resolve_prediction_scope(candidate_graph, method)
        )
        # field names only

        # 2. Pull actual values for the DAG X fields
        direct_x_values = self.extract_x_values(
            direct_x,
            job_config,
            job_features,
            env_vector,
        )
        model_id = job_config.get("model_id") or job_features.get("model_id")
        if model_id is None:
            raise ValueError("AIC_DynoSim needs model_id")
        direct_x_values["model_id"] = model_id
        # field names -> actual values from job_config/env_vector

        # 3. Add simulator-only controls that are NOT in the DAG
        # num_requests, replay_concurrency, arrival_interval_ms, replay_mode="offline/online"
        # this is a very AIC/DynoSim specific control
        # so TODO - maybe we can move into a different function?
        # to maintain a very modular architecture
        simulator_controls = self._build_simulator_controls(
            self.objective,
            job_config,
            job_features,
            direct_x_values,
        )
        # objective-specific DynoSim controls, not DAG nodes

        # 4. Translate everything into DynoSim/AIC argument names
        surrogate_input = self.build_surrogate_inputs(
            direct_x_values,
            simulator_controls,
            method,
        )
        # direct_x_values + simulator_controls -> AIC_dynosim args

        # 5. Run DynoSim/AIC and get direct outputs
        y_hat_direct, v_hat_direct = self.run_surrogate(
            surrogate_input,
            method,
        )
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
        )
        # post process direct outputs to get derived outputs

        # 7. Later: merge direct + derived
        y_hat = self.merge_outputs(y_hat_direct, y_hat_derived)
        v_hat = self.merge_outputs(v_hat_direct, v_hat_derived)
        # final output
        return y_hat, v_hat

    # def get_model_config(self, model_id):
    #     # Fetch model architecture from Huggingface or a similar place
    #     # Inputs: model_id
    #     # Outputs: config.json
    #     load_dotenv()
    #     hf_token = os.getenv("HF_TOKEN")
    #     if not hf_token:
    #         raise ValueError("HF_TOKEN is not set")

    #     config_path = hf_hub_download(
    #         repo_id=model_id,
    #         filename="config.json",
    #         token=hf_token,
    #     )

    #     with open(config_path) as f:
    #         return json.load(f)

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

    def map_gpu_to_aic_system(self, gpu_type):
        # TODO - general helper, can be moved out of this file/class
        # Convert common GPU names into Dynamo/AIC system names.
        gpu_to_aic_system = {
            "GB200": "gb200_sxm",
            "GB200_SXM": "gb200_sxm",
            "GB10": "gb10",
            "B200": "b200_sxm",
            "B200_SXM": "b200_sxm",
            "H200": "h200_sxm",
            "H200_SXM": "h200_sxm",
            "H100": "h100_sxm",
            "H100_SXM": "h100_sxm",
            "H100_PCIE": "h100_pcie",
            "A100": "a100_sxm",
            "A100_SXM": "a100_sxm",
            "A100_PCIE": "a100_pcie",
            "A30": "a30",
            "L40S": "l40s",
            "L40": "l40",
            "L4": "l4",
            "V100": "v100_sxm",
            "V100_SXM": "v100_sxm",
            "V100_PCIE": "v100_pcie",
            "T4": "t4",
            "MI200": "mi200",
            "MI300": "mi300",
        }

        supported_aic_systems = set(gpu_to_aic_system.values())
        normalized_gpu_type = str(gpu_type).strip()
        normalized_key = normalized_gpu_type.upper().replace("-", "_").replace(" ", "_")
        normalized_value = normalized_gpu_type.lower()

        if normalized_value in supported_aic_systems:
            return normalized_value

        if normalized_key in gpu_to_aic_system:
            return gpu_to_aic_system[normalized_key]

        raise ValueError(f"No AIC system mapping for gpu_type={gpu_type}")

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
                    "total_token_budget",
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
            raise ValueError(f"Unsupported surrogate method: {method_name}")

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

    def _build_simulator_controls(self, objective, job_config, job_features, direct_x_values):
        # Build DynoSim run controls. These are not DAG X values.
        # Inputs: objective, JobConfig, JobFeatures, direct_x_values
        # Outputs: simulator_controls
        if objective == "batched":
            sources = (direct_x_values, job_features, job_config)
            isl = self._first_positive(sources, "isl_token_avg", "input_len_tokens_avg")
            osl = self._first_positive(sources, "osl_token_avg", "output_len_tokens_avg")
            max_num_seq = direct_x_values.get("max_num_seq")
            max_num_batched_tokens = direct_x_values.get("max_num_batched_tokens")

            if max_num_seq is None or max_num_batched_tokens is None:
                raise ValueError("Batched simulation needs max_num_seq and max_num_batched_tokens")
            if isl is None or osl is None:
                raise ValueError("Batched simulation needs positive input/output token lengths")

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
            }

        if objective == "online":
            # Online workload semantics, offline replay execution: DynoSim uses a
            # logical clock instead of live async workers. This is not batch/offline serving.
            traffic_mode = job_features.get("_traffic_mode") or job_config.get("_traffic_mode")
            traffic_mode = str(traffic_mode or "request_rate")
            request_arrival_rate = self._first_positive(
                (direct_x_values, job_features, job_config), "request_arrival_rate"
            )
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
                }
            if traffic_mode != "request_rate":
                raise ValueError(f"Unknown online _traffic_mode: {traffic_mode!r}")
            if request_arrival_rate is None:
                raise ValueError("Online simulation needs positive request_arrival_rate")

            return {
                "request_count": max(
                    ONLINE_MIN_REQUESTS,
                    math.ceil(request_arrival_rate * ONLINE_REPLAY_WINDOW_S),
                ),
                "arrival_interval_ms": 1000.0 / request_arrival_rate,
                "replay_mode": "offline",
            }

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
            raise ValueError(f"Unsupported method or multi method is not supported yet: {method}")

        model_id = direct_x_values.get("model_id")
        if model_id is None:
            raise ValueError("AIC_DynoSim needs model_id")

        gpu_type = direct_x_values.get("gpu_type")
        if gpu_type is None:
            raise ValueError("AIC_DynoSim needs gpu_type")

        engine_args = {
            "engine_type": direct_x_values.get("engine_name", "vllm"),
            "block_size": direct_x_values.get("block_size", 64),
            "max_num_seqs": direct_x_values.get("max_num_seq"),
            "max_num_batched_tokens": direct_x_values.get("max_num_batched_tokens"),
            "aic_backend": direct_x_values.get("engine_name", "vllm"),
            # AIC's bundled performance database currently supports this version.
            "aic_backend_version": "0.14.0",
            "aic_system": self.map_gpu_to_aic_system(gpu_type),
            "aic_model_path": model_id,
            "aic_tp_size": direct_x_values.get("tp", 1),
            "aic_moe_ep_size": direct_x_values.get("ep", 1),
            "enable_prefix_caching": direct_x_values.get("prefix_cache_enabled", False),
            "enable_chunked_prefill": direct_x_values.get("chunked_prefill_enable", False),
            "preemption_mode": direct_x_values.get("preemption_policy"),
        }
        self._resolve_aic_num_gpu_blocks(engine_args, direct_x_values)

        queue_policy = direct_x_values.get("scheduling_policy")
        if queue_policy in {"fcfs", "lcfs", "wspt"}:
            engine_args["router_queue_policy"] = queue_policy

        num_workers = int(direct_x_values.get("num_workers") or 1)
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
            "shared_prefix_ratio": direct_x_values.get("workload_prefix_concentration", 0.0),
            "turns_per_session": 2 if direct_x_values.get("is_session_affinity") else 1,
            "pd_enabled": direct_x_values.get("pd_enabled", False),
            "prefill_worker_count": direct_x_values.get("prefill_worker_count", 1),
            "decode_worker_count": direct_x_values.get("decode_worker_count", 1),
            "num_workers": num_workers,
            "router_mode": router_mode,
            **simulator_controls,
        }

        engine_args = {key: value for key, value in engine_args.items() if value is not None}
        replay_args = {key: value for key, value in replay_args.items() if value is not None}

        return {
            "method": method_name,
            "engine_args": engine_args,
            "replay_args": replay_args,
        }

    def _resolve_aic_num_gpu_blocks(self, engine_args, direct_x_values):
        """Run AIC's memory fit/KV-capacity estimator before DynoSim replay."""
        if engine_args.get("num_gpu_blocks") is not None or not engine_args.get("aic_backend"):
            return
        if not engine_args.get("aic_model_path"):
            raise ValueError("AIC memory preflight failed: missing aic_model_path")

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
                memory_fraction_value=float(direct_x_values.get("gpu_mem_util") or 0.9),
                tp_size=int(engine_args.get("aic_tp_size") or 1),
                pp_size=int(direct_x_values.get("pp") or 1),
                attention_dp_size=int(direct_x_values.get("dp") or 1),
                moe_ep_size=engine_args.get("aic_moe_ep_size"),
                gpu_memory_capacity_bytes_override=self._gpu_memory_capacity_bytes(direct_x_values),
                allow_naive_fallback=False,
                allow_hf_config_download=False,
            )
        except Exception as exc:
            raise ValueError(f"AIC memory preflight failed: {exc}") from exc
        if int(blocks) <= 0:
            raise ValueError(f"AIC memory preflight failed: num_gpu_blocks={blocks}")
        engine_args["num_gpu_blocks"] = int(blocks)

    @staticmethod
    def _estimate_num_gpu_blocks(**kwargs):
        return estimate_num_gpu_blocks(**kwargs)

    @staticmethod
    def _memory_fraction_kind(backend):
        if backend == "trtllm":
            return "of_free"
        if backend in {"vllm", "sglang"}:
            return "of_total"
        raise ValueError(f"unknown backend {backend!r} for AIC memory preflight")

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
            # dont accumulate, just run the surrogate model
            return self.run_aic_dynosim(surrogate_input)

    def run_aic_dynosim(self, surrogate_input):
        # Run the AIC DynoSim model.
        # Inputs: SurrogateInput
        # Outputs: y_hat, v_hat
        from dynamo.llm import MockEngineArgs
        from dynamo.replay.api import run_synthetic_trace_replay

        engine_args = surrogate_input["engine_args"]
        replay_args = surrogate_input["replay_args"]

        input_tokens = int(replay_args["input_tokens"])
        output_tokens = int(replay_args["output_tokens"])
        request_count = int(replay_args["request_count"])
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
        if replay_args.get("replay_concurrency") is not None:
            common_replay_args["replay_concurrency"] = replay_args["replay_concurrency"]
        elif replay_args.get("arrival_interval_ms") is not None:
            common_replay_args["arrival_interval_ms"] = replay_args["arrival_interval_ms"]

        if pd_enabled:
            prefill_engine_args = dict(engine_args)
            decode_engine_args = dict(engine_args)
            prefill_engine_args["worker_type"] = "prefill"
            decode_engine_args["worker_type"] = "decode"

            raw_report = run_synthetic_trace_replay(
                input_tokens,
                output_tokens,
                request_count,
                prefill_engine_args=MockEngineArgs(**prefill_engine_args),
                decode_engine_args=MockEngineArgs(**decode_engine_args),
                num_prefill_workers=int(replay_args.get("prefill_worker_count", 1)),
                num_decode_workers=int(replay_args.get("decode_worker_count", 1)),
                **common_replay_args,
            )
        else:
            raw_report = run_synthetic_trace_replay(
                input_tokens,
                output_tokens,
                request_count,
                extra_engine_args=MockEngineArgs(**engine_args),
                num_workers=int(replay_args.get("num_workers", 1)),
                **common_replay_args,
            )

        return self.canonicalize_aic_dynosim_output(raw_report)

    def canonicalize_aic_dynosim_output(self, raw_report):
        # TODO - general helper, can be moved out of this file/class
        # Convert raw DynoSim report keys into DAG V/Y names.
        completed_requests = (
            raw_report.get("completed_requests") or raw_report.get("num_requests") or 1
        )

        v_hat_direct = {
            "input_length_observed": raw_report.get("total_input_tokens", 0) / completed_requests,
            "output_length_observed": raw_report.get("total_output_tokens", 0) / completed_requests,
            "kvcache_hit_rate": raw_report.get("prefix_cache_reused_ratio"),
        }

        y_hat_direct = {
            "p99_ttft_ms": raw_report.get("p99_ttft_ms"),
            "p99_tpot_ms": raw_report.get("p99_tpot_ms", raw_report.get("p99_itl_ms")),
            "throughput_token_per_sec": raw_report.get("output_throughput_tok_s"),
        }

        return y_hat_direct, v_hat_direct

    def derive_outputs(
        self,
        derive_v,
        derive_y,
        y_hat_direct,
        v_hat_direct,
        job_config,
        job_features,
        price_vector,
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
            total_tokens = input_tokens + output_tokens

            if "total_token_budget" in requested_v:
                v_hat_derived["total_token_budget"] = total_tokens

            if "kv_pressure_score" in requested_v:
                max_tokens = job_config.get("max_num_batched_tokens")
                if max_tokens:
                    v_hat_derived["kv_pressure_score"] = min(1.0, total_tokens / max_tokens)

            if "kv_cache_util" in requested_v:
                # Placeholder until we derive real KV blocks from memory/block size.
                v_hat_derived["kv_cache_util"] = v_hat_derived.get("kv_pressure_score")

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
                y_hat_derived["cost_per_token"] = price_per_hour / (throughput * 3600.0)

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
