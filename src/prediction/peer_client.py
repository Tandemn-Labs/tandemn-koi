"""Map authoritative Koi rank X into the shared predictor facade."""

from __future__ import annotations

import importlib
import math
from copy import copy
from typing import Any

_MODEL_KEYS = {
    "meta-llama/meta-llama-3-8b": "llama3-8b",
    "meta-llama/meta-llama-3.1-8b": "llama3-8b",
    "meta-llama/meta-llama-3.1-8b-instruct": "llama3-8b",
    "meta-llama/meta-llama-3-70b": "llama3-70b",
    "meta-llama/meta-llama-3.1-70b": "llama3-70b",
    "meta-llama/meta-llama-3.1-70b-instruct": "llama3-70b",
    "qwen/qwen2.5-72b-instruct": "qwen-72b",
    "qwen/qwen3-8b": "qwen3-8b",
    "qwen/qwen3-32b": "qwen3-32b",
}
_DEVICE_KEYS = {
    "A100": "A100",
    "A100-80GB": "A100",
    "A10": "A10G",
    "A10G": "A10G",
    "H100": "H100",
    "H100-80GB": "H100",
    "L4": "L4",
    "L40S": "L40S",
    "RTXPRO6000": "L40S",
    "RTX-PRO-6000": "L40S",
}
_SOLVER_FAMILIES = {
    "A10G": "g5.48xlarge",
    "L4": "g6.48xlarge",
    "L40S": "g6e.48xlarge",
}


class PeerPredictorClient:
    """Call the optional external predictor package for one Koi rank."""

    def __init__(self, names=("solver", "blis"), predict_fn=None):
        self.names = tuple(names)
        self._predict_fn = predict_fn

    def predict(
        self,
        job_config: dict[str, Any],
        job_features: dict[str, Any],
        *,
        scenario: str,
    ) -> dict[str, dict[str, Any]]:
        query = self._query(job_config, job_features, scenario=scenario)
        if query is None:
            return {}
        predict_fn = self._predict_fn
        if predict_fn is None:
            # Maintained separately in LLM_placement_solver; install it into Koi's environment.
            try:
                default_predict = importlib.import_module("predictor_compare").predict
            except ImportError as exc:
                raise RuntimeError(
                    "peer prediction requires the optional tandemn-predictors package "
                    "from Tandemn-Labs/LLM_placement_solver"
                ) from exc

            predict_fn = default_predict

        results = {
            name: result.to_dict()
            for name, result in predict_fn(query, predictors=self.names).items()
        }
        solver = results.get("solver")
        if solver is not None and solver.get("status") in {"unsupported", "failed"}:
            fallback = self._retry_roofline(predict_fn, query)
            if fallback is not None:
                results["solver"] = fallback
        input_approximations = {
            name: query.context[name]
            for name in ("model_approximation", "hardware_approximation")
            if query.context.get(name)
        }
        if input_approximations:
            for result in results.values():
                hardware = input_approximations.get("hardware_approximation") or {}
                scale = float(hardware.get("throughput_scale") or 1.0)
                if result.get("status") == "success" and scale != 1.0:
                    for node in ("total_tps", "input_tps", "output_tps"):
                        if result.get(node) is not None:
                            result[node] = float(result[node]) * scale
                    for node in (
                        "ttft_ms_p50",
                        "ttft_ms_p99",
                        "tpot_ms_p50",
                        "tpot_ms_p99",
                        "e2e_ms_p50",
                        "e2e_ms_p99",
                    ):
                        if result.get(node) is not None:
                            result[node] = float(result[node]) / scale
                    if result.get("dollar_per_mtok") is not None:
                        result["dollar_per_mtok"] = float(result["dollar_per_mtok"]) / scale
                approximation = dict(result.get("approximation") or {})
                approximation["inputs"] = input_approximations
                result["approximation"] = approximation
        return results

    @staticmethod
    def _retry_roofline(predict_fn, query) -> dict[str, Any] | None:
        requested = {
            "task": query.task,
            "precision": query.precision,
            "num_replicas": query.num_replicas,
            "tp": query.tp,
            "pp": query.pp,
        }
        variants = (
            (query.tp, query.pp),
            (query.tp, 1),
            (1, 1),
        )
        seen = set()
        for tp, pp in variants:
            resolved = ("capacity", "bf16", 1, tp, pp)
            if resolved in seen or resolved == tuple(requested.values()):
                continue
            seen.add(resolved)
            candidate = copy(query)
            candidate.task = "capacity"
            candidate.precision = "bf16"
            candidate.num_replicas = 1
            candidate.tp = tp
            candidate.pp = pp
            try:
                retry = predict_fn(candidate, predictors=("solver",)).get("solver")
            except Exception:
                continue
            if retry is None:
                continue
            result = retry.to_dict()
            if result.get("status") != "success":
                continue
            replicas = max(1, int(requested["num_replicas"]))
            for node in ("total_tps", "input_tps", "output_tps", "cost_per_hour"):
                if result.get(node) is not None:
                    result[node] = float(result[node]) * replicas
            result["approximation"] = {
                "reason": "nearest_supported_roofline_config",
                "requested": requested,
                "resolved": {
                    "task": candidate.task,
                    "precision": candidate.precision,
                    "num_replicas": candidate.num_replicas,
                    "tp": candidate.tp,
                    "pp": candidate.pp,
                },
            }
            return result
        return None

    @staticmethod
    def _query(job_config: dict[str, Any], job_features: dict[str, Any], *, scenario: str):
        values = {**job_features, **job_config}
        model_id = str(values.get("model_id") or "").strip()
        model, model_approximation = _resolve_model(values, model_id)
        gpu = _normalize_gpu(values.get("gpu_type"))
        device = _DEVICE_KEYS.get(gpu)
        if model is None or not gpu:
            return None
        hardware_approximation = None
        if device is None:
            device = "L40S"
            hardware_approximation = {
                "requested_gpu_type": values.get("gpu_type"),
                "resolved_peer_device": device,
                "reason": "generic_supported_roofline_device",
                "throughput_scale": _generic_roofline_scale(values),
            }

        input_length = _positive_int(values.get("isl_token_avg"))
        output_length = _positive_int(values.get("osl_token_avg"))
        if input_length is None or output_length is None:
            return None
        try:
            Query = importlib.import_module("predictor_compare").Query
        except ImportError as exc:
            raise RuntimeError(
                "peer prediction requires the optional tandemn-predictors package "
                "from Tandemn-Labs/LLM_placement_solver"
            ) from exc
        dp = max(1, int(values.get("dp") or values.get("n_replicas") or 1))
        batch = max(
            1,
            int(
                values.get("max_num_seq")
                or values.get("max_concurrent_streaming")
                or values.get("effective_batch_size")
                or 1
            ),
        )
        mode = str(values.get("type") or "").lower()
        task = "online" if mode == "online" else "capacity"
        qps = float(values.get("request_arrival_rate") or 10_000.0)
        if not math.isfinite(qps) or qps <= 0:
            return None
        request_count = int(
            values.get("num_requests") or max(32, qps * 60 if task == "online" else batch * 20)
        )

        represented = {
            "model_id",
            "gpu_type",
            "tp",
            "pp",
            "dp",
            "n_replicas",
            "isl_token_avg",
            "osl_token_avg",
            "request_arrival_rate",
            "max_num_seq",
            "max_num_batched_tokens",
            "engine_name",
            "engine_version",
            "block_size",
            "gpu_mem_util",
            "prefix_cache_enabled",
            "chunked_prefill_enable",
            "router_policy",
            "preemption_policy",
        }
        context = {key: value for key, value in values.items() if key not in represented}
        context["scenario"] = scenario
        if model_approximation:
            context["model_approximation"] = model_approximation
        if hardware_approximation is not None:
            context["hardware_approximation"] = hardware_approximation
        elif gpu in {"RTXPRO6000", "RTX-PRO-6000"}:
            context["hardware_approximation"] = {
                "requested_gpu_type": values.get("gpu_type"),
                "resolved_peer_device": device,
                "reason": "nearest_supported_roofline_device",
                "throughput_scale": _generic_roofline_scale(values),
            }
        solver_family = values.get("instance_type") or _SOLVER_FAMILIES.get(device)
        if gpu in {"RTXPRO6000", "RTX-PRO-6000"}:
            solver_family = _SOLVER_FAMILIES[device]
        return Query(
            model=model,
            device=device,
            tp=max(1, int(values.get("tp") or 1)),
            pp=max(1, int(values.get("pp") or 1)),
            num_replicas=dp,
            input_len=input_length,
            output_len=output_length,
            num_requests=max(1, request_count),
            qps=qps,
            batch_size=batch,
            max_num_batched_tokens=max(
                1,
                int(values.get("max_num_batched_tokens") or batch * (input_length + output_length)),
            ),
            precision=str(values.get("weight_dtype") or "bf16"),
            request_pattern=str(values.get("request_arrival_pattern") or mode or "batch"),
            task=task,
            engine_name=str(values.get("engine_name") or "vllm"),
            engine_version=(
                str(values["engine_version"]) if values.get("engine_version") is not None else None
            ),
            block_size=max(1, int(values.get("block_size") or 64)),
            gpu_memory_utilization=float(values.get("gpu_mem_util") or 0.9),
            prefix_cache_enabled=bool(values.get("prefix_cache_enabled", False)),
            chunked_prefill_enabled=bool(values.get("chunked_prefill_enable", False)),
            router_policy=str(values.get("router_policy") or "round-robin"),
            preemption_policy=values.get("preemption_policy"),
            context=context,
            dynosim_model=model_id,
            blis_model=values.get("blis_model"),
            solver_config_dir=values.get("solver_config_dir"),
            solver_instance_family=solver_family,
        )


def _positive_int(value) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _normalize_gpu(value) -> str:
    normalized = str(value or "").strip().upper().replace("_", "-").replace(" ", "-")
    if normalized.startswith("NVIDIA-"):
        normalized = normalized.removeprefix("NVIDIA-")
    return normalized


def _resolve_model(values: dict[str, Any], model_id: str) -> tuple[str | None, dict | None]:
    exact = _MODEL_KEYS.get(model_id.lower())
    if exact is not None:
        return exact, None
    if values.get("solver_config_dir"):
        return model_id or "custom", None
    raw_params = values.get("model_params_b")
    if raw_params is None:
        return None, None
    try:
        params_b = float(raw_params)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(params_b) or params_b <= 0 or values.get("is_moe"):
        return None, None
    is_qwen = "qwen" in model_id.lower()
    candidates = ("qwen3-8b", "qwen3-32b", "qwen-72b") if is_qwen else ("llama3-8b", "llama3-70b")
    sizes = {
        "qwen3-8b": 8.0,
        "qwen3-32b": 32.0,
        "qwen-72b": 72.0,
        "llama3-8b": 8.0,
        "llama3-70b": 70.0,
    }
    resolved = min(candidates, key=lambda name: abs(params_b - sizes[name]))
    return resolved, {
        "requested_model_id": model_id,
        "resolved_peer_model": resolved,
        "reason": "nearest_parameter_count_roofline_model",
    }


def _generic_roofline_scale(values: dict[str, Any]) -> float:
    ratios = []
    for name, reference in (("gpu_tflops_fp16", 362.0), ("gpu_bandwidth_gbps", 864.0)):
        raw = values.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            ratios.append(value / reference)
    return max(0.1, min(1.0, min(ratios))) if ratios else 0.5
