"""Translate DCGM Prometheus measurements into Koi's observed device fields."""

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import ClassVar, cast

from src.core.telemetry_models import ObservedV
from src.infra.metrics.backends.prometheus import collect_measurements
from src.infra.metrics.base import MetricReader, RankContext, RankObservations


class DCGMAdapter:
    """Query rank-aggregated device measurements; DCGM activity is already a rate."""

    kind = "device"
    name = "dcgm"

    V_FIELDS: ClassVar[dict[str, tuple[str, float]]] = {
        "gpu_memory_used_fraction_mean": ("gpu_mem_used_fraction", 1.0),
        "gpu_memory_free_mib_mean": ("vram_headroom_gb", 1.0 / 1024),
        "sm_active_fraction_mean": ("sm_utilization", 1.0),
        "dram_active_fraction_mean": ("mem_bandwidth_utilization", 1.0),
        "pcie_bytes_per_second_mean": ("pcie_tput_observed", 1.0),
        "nvlink_bytes_per_second_mean": ("nvlink_tput_observed", 1.0),
    }

    def __init__(self, reader: MetricReader) -> None:
        self.reader = reader
        profile = json.loads(
            (Path(__file__).resolve().parents[1] / "profiles/dcgm_promql.json").read_text()
        )
        self.queries = {name: profile["queries"][name]["expression"] for name in self.V_FIELDS}

    def collect(
        self,
        contexts: Iterable[RankContext],
        *,
        start: datetime,
        end: datetime,
        step_seconds: int = 10,
    ) -> dict[tuple[str, str], RankObservations]:
        """Fill device V fields; retain bytes/s and the existing MiB/1024 convention."""
        measurements, warnings, infos = collect_measurements(
            self.reader, contexts, self.queries, start=start, end=end, step_seconds=step_seconds
        )
        outputs = {}
        for key, samples in measurements.items():
            output = RankObservations(warnings=warnings, infos=infos)
            output.v = cast(
                ObservedV,
                {
                    field: samples[name].values * scale
                    for name, (field, scale) in self.V_FIELDS.items()
                    if name in samples
                },
            )
            output.timestamps = {
                field: samples[name].timestamps
                for name, (field, _) in self.V_FIELDS.items()
                if name in samples
            }
            outputs[key] = output
        return outputs
