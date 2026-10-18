"""Translate vLLM Prometheus measurements into Koi's observed engine fields."""

import json
from collections.abc import Iterable
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
from src.core.telemetry_models import ObservedV, ObservedY
from src.infra.metrics.backends.prometheus import collect_measurements
from src.infra.metrics.base import (
    MetricQueryError,
    MetricReader,
    QueryResult,
    RankContext,
    RankObservations,
    TimeSeries,
)


class VLLMAdapter:
    """Query and translate compatible vLLM ranks in one collection pass."""

    kind = "engine"
    name = "vllm"

    V_FIELDS: ClassVar[dict[str, str]] = {
        "running_requests": "live_batch_size",
        "waiting_requests": "depth_req_q",
        "kv_cache_usage_fraction": "kv_cache_util",
        "prefix_cache_hit_fraction": "kvcache_hit_rate",
        "prompt_tokens_per_request": "input_length_observed",
        "generated_tokens_per_request": "output_length_observed",
        "prefill_observations_per_second": "prefill_iteration_counts_per_second",
        "decode_observations_per_second": "decode_itr_counts_per_second",
        "prefill_decode_time_ratio": "pd_inbalance",
    }
    Y_FIELDS: ClassVar[dict[str, tuple[str, float]]] = {
        "ttft_p99_seconds": ("p99_ttft_ms", 1000.0),
        "tpot_p99_seconds": ("p99_tpot_ms", 1000.0),
        "generated_output_tokens_per_second": ("throughput_token_per_sec", 1.0),
    }

    def __init__(self, reader: MetricReader) -> None:
        self.reader = reader
        profile = json.loads(
            (Path(__file__).resolve().parents[1] / "profiles/vllm_promql.json").read_text()
        )
        self.queries = {
            name: profile["queries"][name]["expression"]
            for name in (*self.V_FIELDS, *self.Y_FIELDS, "num_kv_blocks_mean")
        }

    def collect(
        self,
        contexts: Iterable[RankContext],
        *,
        start: datetime,
        end: datetime,
        step_seconds: int = 10,
    ) -> dict[tuple[str, str], RankObservations]:
        """Fill observed V/Y, preserving sample times, missing fields, and zeros."""
        contexts = list(contexts)
        measurements, warnings, infos = collect_measurements(
            self.reader,
            contexts,
            self.queries,
            start=start,
            end=end,
            step_seconds=step_seconds,
            parse_result=self._parse_cache_blocks,
        )
        outputs = {}
        for context in contexts:
            key = (context.job_id, context.rank_id)
            samples = measurements[key]
            output = RankObservations(warnings=warnings, infos=infos)
            if "num_kv_blocks_mean" in samples:
                output.features["num_kv_blocks_mean"] = samples["num_kv_blocks_mean"].values
                output.timestamps["num_kv_blocks_mean"] = samples["num_kv_blocks_mean"].timestamps
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
                # TODO - Change SLO Margin for Online/Offline Inference  - have a discussion
                output.y["slo_margin"] = (target - output.y["p99_ttft_ms"]) / target
                output.timestamps["slo_margin"] = output.timestamps["p99_ttft_ms"]
            outputs[key] = output
        return outputs

    @staticmethod
    def _parse_cache_blocks(name: str, result: QueryResult) -> QueryResult:
        """An info gauge is 1; capacity is its label, common to a homogeneous rank."""
        if name != "num_kv_blocks_mean":
            return result
        samples: dict[tuple[str, str], dict[float, float]] = {}
        warnings = list(result.warnings)
        for series in result.series:
            if set(series.labels) - {"num_gpu_blocks"} != {"job_id", "rank_id"}:
                raise MetricQueryError("cache capacity requires job_id, rank_id and num_gpu_blocks")
            text = series.labels.get("num_gpu_blocks", "")
            if text in ("None", ""):
                warnings.append("num_gpu_blocks is unavailable")
                continue
            if not text.isdecimal() or not isfinite(float(text)):
                raise MetricQueryError(f"invalid num_gpu_blocks: {text!r}")
            blocks = float(text)
            key = (series.labels["job_id"], series.labels["rank_id"])
            values = samples.setdefault(key, {})
            for timestamp in series.timestamps:
                if timestamp in values and values[timestamp] != blocks:
                    raise MetricQueryError(f"KV capacity disagreement for job/rank {key!r}")
                values[timestamp] = blocks
        translated = []
        for (job_id, rank_id), values in samples.items():
            timestamps = np.asarray(sorted(values), dtype=float)
            translated.append(
                TimeSeries(
                    {"job_id": job_id, "rank_id": rank_id},
                    timestamps,
                    np.asarray([values[timestamp] for timestamp in timestamps], dtype=float),
                )
            )
        return QueryResult(tuple(translated), tuple(warnings), result.infos)
