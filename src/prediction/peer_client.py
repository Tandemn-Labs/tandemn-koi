"""Map authoritative Koi rank X into the shared predictor facade."""

from __future__ import annotations

import math
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
    "H100": "H100",
    "H100-80GB": "H100",
    "L40S": "L40S",
}


class PeerPredictorClient:
    """Call external peers for one DP-aggregate Koi rank prediction."""

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
            from predictor_compare import predict as default_predict

            predict_fn = default_predict

        return {
            name: result.to_dict()
            for name, result in predict_fn(query, predictors=self.names).items()
        }

    @staticmethod
    def _query(job_config: dict[str, Any], job_features: dict[str, Any], *, scenario: str):
        from predictor_compare import Query

        values = {**job_features, **job_config}
        model_id = str(values.get("model_id") or "").strip()
        model = _MODEL_KEYS.get(model_id.lower())
        gpu = str(values.get("gpu_type") or "").replace("NVIDIA ", "").strip()
        device = _DEVICE_KEYS.get(gpu.upper(), _DEVICE_KEYS.get(gpu))
        if model is None or device is None:
            return None

        input_length = _positive_int(values.get("isl_token_avg"))
        output_length = _positive_int(values.get("osl_token_avg"))
        if input_length is None or output_length is None:
            return None
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
            solver_instance_family=values.get("instance_type"),
        )


def _positive_int(value) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
