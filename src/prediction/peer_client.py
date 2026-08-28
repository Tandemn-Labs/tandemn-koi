"""Map authoritative Koi rank X into the shared predictor facade."""

from __future__ import annotations

import importlib
import math
from copy import copy
from typing import Any

from src.prediction.analytic_v import model_weight_gb
from src.prediction.compatibility import gpu_profile_from_values, resolve_dtype, resolve_gpu
from src.prediction.profile_search import resolve_model_reference

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
_MODEL_REFERENCE_SIZES_B = {
    "llama3-8b": 8.0,
    "llama3-70b": 70.0,
    "qwen3-8b": 8.0,
    "qwen3-32b": 32.0,
    "qwen-72b": 72.0,
}
_SOLVER_GPU_CHOICES = frozenset({"A10G", "A100", "H100", "L4", "L40S"})
_SOLVER_FAMILIES = {
    "A10G": "g5.48xlarge",
    "L4": "g6.48xlarge",
    "L40S": "g6e.48xlarge",
}


class PeerPredictorClient:
    """Call the optional external predictor package for one Koi rank."""

    def __init__(self, names=("solver", "blis"), predict_fn=None, query_cls=None):
        self.names = tuple(names)
        self._predict_fn = predict_fn
        self._query_cls = query_cls

    def predict(
        self,
        job_config: dict[str, Any],
        job_features: dict[str, Any],
        *,
        scenario: str,
    ) -> dict[str, dict[str, Any]]:
        query = self._query(
            job_config,
            job_features,
            scenario=scenario,
            query_cls=self._query_cls,
        )
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
        compatibility = query.context.get("compatibility") or {}
        approximate_compatibility = {
            name: resolution
            for name, resolution in compatibility.items()
            if resolution.get("kind") == "nearest"
        }
        input_approximations = {}
        if approximate_compatibility:
            input_approximations["compatibility"] = approximate_compatibility
        if query.context.get("model_approximation"):
            input_approximations["model"] = query.context["model_approximation"]
        for result in results.values():
            result["compatibility"] = compatibility
            if input_approximations:
                resolutions = approximate_compatibility.values()
                scale = math.prod(
                    float(item.get("throughput_scale") or 1.0) for item in resolutions
                )
                scale *= float(
                    (input_approximations.get("model") or {}).get("throughput_scale") or 1.0
                )
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
            if requested["task"] == "online":
                for node in (
                    "ttft_ms_p50",
                    "ttft_ms_p99",
                    "tpot_ms_p50",
                    "tpot_ms_p99",
                    "e2e_ms_p50",
                    "e2e_ms_p99",
                ):
                    result.pop(node, None)
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
    def _query(
        job_config: dict[str, Any],
        job_features: dict[str, Any],
        *,
        scenario: str,
        query_cls=None,
    ):
        values = {**job_features, **job_config}
        model_id = str(values.get("model_id") or "").strip()
        model, model_approximation = resolve_model_reference(
            model_id,
            values,
            _MODEL_KEYS,
            _MODEL_REFERENCE_SIZES_B,
        )
        if not _requested_model_fits(values, require_proof=model_approximation is not None):
            return None
        gpu_resolution = resolve_gpu(
            values.get("gpu_type"),
            backend="solver",
            available=_SOLVER_GPU_CHOICES,
            requested_profile=values,
        )
        dtype_resolution = resolve_dtype(
            values.get("weight_dtype") or "bf16",
            backend="solver",
            component="weights",
            available=frozenset({"bf16"}),
        )
        if (
            model is None
            or not gpu_resolution.supported
            or gpu_resolution.resolved is None
            or gpu_resolution.backend_value is None
            or not dtype_resolution.supported
        ):
            return None
        device = gpu_resolution.backend_value
        resolved_gpu = gpu_resolution.resolved

        input_length = _positive_int(values.get("isl_token_avg"))
        output_length = _positive_int(values.get("osl_token_avg"))
        if input_length is None or output_length is None:
            return None
        Query = query_cls
        if Query is None:
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
        context["compatibility"] = {
            "gpu": gpu_resolution.to_dict(),
            "weight_dtype": dtype_resolution.to_dict(),
        }
        if model_approximation:
            context["model_approximation"] = model_approximation
        solver_family = values.get("instance_type") or _SOLVER_FAMILIES.get(resolved_gpu)
        if gpu_resolution.approximate:
            solver_family = _SOLVER_FAMILIES.get(resolved_gpu)
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
            precision=str(dtype_resolution.backend_value),
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


def _requested_model_fits(values: dict[str, Any], *, require_proof: bool) -> bool:
    weight_gb = model_weight_gb(values)
    raw_memory = values.get("gpu_mem_gb")
    if raw_memory is None:
        profile = gpu_profile_from_values(str(values.get("gpu_type") or ""), values)
        raw_memory = profile.memory_gb if profile is not None else None
    if weight_gb is None or raw_memory is None:
        return not require_proof
    try:
        shards = max(1, int(values.get("tp") or 1) * int(values.get("pp") or 1))
        usable_memory = float(raw_memory) * float(values.get("gpu_mem_util") or 0.9)
    except (TypeError, ValueError, OverflowError):
        return False
    return weight_gb / shards + 2.0 <= usable_memory
