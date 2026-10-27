"""Translate SGLang v0.5.21 Prometheus metrics into Koi's engine fields."""

import json
from collections.abc import Iterable
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import ClassVar, cast

from src.core.telemetry_models import ObservedV, ObservedY
from src.infra.metrics.backends.prometheus import collect_measurements
from src.infra.metrics.base import MetricReader, RankContext, RankObservations


# TODO - Always deploy sglang using --collect-tokens-histogram argument
# otherwise we will end up not getting a LOT of these metrics
class SGLangAdapter:
    """Query and translate rank-level SGLang measurements."""

    kind = "engine"
    name = "sglang"

    # TODO: Leave pd_inbalance absent; SGLang v0.5.21 has no equivalent to vLLM's
    # per-request prefill/decode timing pair. Add it when equivalent metrics are available.
    V_FIELDS: ClassVar[dict[str, str]] = {
        "running_requests": "live_batch_size",
        "waiting_requests": "depth_req_q",
        "full_token_usage": "kv_cache_util",
        "prefix_cache_hit_fraction": "kvcache_hit_rate",
        "prompt_tokens_per_request": "input_length_observed",
        "generated_tokens_per_request": "output_length_observed",
        # shortcut: these count request observations, not scheduler iterations; replace when iteration counters exist.
        "prefill_observations_per_second": "prefill_iteration_counts_per_second",
        "decode_observations_per_second": "decode_itr_counts_per_second",
    }
    Y_FIELDS: ClassVar[dict[str, tuple[str, float]]] = {
        "ttft_p99_seconds": ("p99_ttft_ms", 1000.0),
        "tpot_p99_seconds": ("p99_tpot_ms", 1000.0),
        "generation_throughput": ("throughput_token_per_sec", 1.0),
    }

    def __init__(self, reader: MetricReader) -> None:
        self.reader = reader
        profile = json.loads(
            (Path(__file__).resolve().parents[1] / "profiles/sglang_promql.json").read_text()
        )
        self.queries = {
            name: profile["queries"][name]["expression"]
            for name in (*self.V_FIELDS, *self.Y_FIELDS)
        }

    def collect(
        self,
        contexts: Iterable[RankContext],
        *,
        start: datetime,
        end: datetime,
        step_seconds: int = 10,
    ) -> dict[tuple[str, str], RankObservations]:
        """Fill supported V/Y fields and leave unreported measurements absent."""
        contexts = list(contexts)
        measurements, warnings, infos = collect_measurements(
            self.reader, contexts, self.queries, start=start, end=end, step_seconds=step_seconds
        )
        outputs = {}
        for context in contexts:
            key = (context.job_id, context.rank_id)
            samples = measurements[key]
            output = RankObservations(warnings=warnings, infos=infos)
            output.v = cast(
                ObservedV,
                {
                    field: samples[name].values
                    for name, field in self.V_FIELDS.items()
                    if name in samples
                },
            )
            output.y = cast(
                ObservedY,
                {
                    field: samples[name].values * scale
                    for name, (field, scale) in self.Y_FIELDS.items()
                    if name in samples
                },
            )
            for name, field in self.V_FIELDS.items():
                if name in samples:
                    output.timestamps[field] = samples[name].timestamps
            for name, (field, _) in self.Y_FIELDS.items():
                if name in samples:
                    output.timestamps[field] = samples[name].timestamps

            target = context.job_features.get("target_p99_ttft_ms")
            if (
                not isinstance(target, bool)
                and isinstance(target, int | float)
                and isfinite(target)
                and target > 0
                and "p99_ttft_ms" in output.y
            ):
                output.y["slo_margin"] = (target - output.y["p99_ttft_ms"]) / target
                output.timestamps["slo_margin"] = output.timestamps["p99_ttft_ms"]
            outputs[key] = output
        return outputs
