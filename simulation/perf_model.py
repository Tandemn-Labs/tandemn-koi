"""Demo-only performance modeling for the realistic simulator profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from koi.tools.perfdb import PerfDB
from koi.tools.physics import lookup_gpu_spec

from simulation.model_registry import ModelSpec, resolve_model_spec


@dataclass(frozen=True)
class LaunchTiming:
    searching_capacity_s: float
    provisioning_s: float
    bootstrapping_s: float
    waiting_model_ready_s: float

    @property
    def total_seconds(self) -> float:
        return (
            self.searching_capacity_s
            + self.provisioning_s
            + self.bootstrapping_s
            + self.waiting_model_ready_s
        )


@dataclass(frozen=True)
class ReplicaConfigAssessment:
    feasible: bool
    reason: Optional[str]
    weight_gb_per_gpu: float
    vram_headroom_gb: float


class LegacyPerfModel:
    """Current mock_orca-style fixed baseline behavior."""

    def estimate_replica_tps(
        self,
        *,
        base_tps: float = 1200.0,
        **_: object,
    ) -> float:
        return float(base_tps)

    def estimate_launch_timing(self, **_: object) -> LaunchTiming:
        return LaunchTiming(
            searching_capacity_s=1.0,
            provisioning_s=3.0,
            bootstrapping_s=1.0,
            waiting_model_ready_s=1.0,
        )


class DemoPerfModel:
    """PerfDB-seeded, architecture-aware demo throughput model."""

    MIN_VRAM_HEADROOM_GB = 8.0

    def __init__(self, perfdb_path: Optional[str] = None, prefer_perfdb: bool = True):
        self.prefer_perfdb = prefer_perfdb
        self.perfdb = None
        path = perfdb_path or str(
            Path(__file__).resolve().parents[1] / "perfdb" / "perfdb_all.csv"
        )
        perf_path = Path(path)
        if prefer_perfdb and perf_path.exists():
            self.perfdb = PerfDB(str(perf_path))

    def resolve_model(
        self,
        model_name: str,
        *,
        dtype: str = "fp16",
        overrides: Optional[dict] = None,
    ) -> ModelSpec:
        return resolve_model_spec(model_name, dtype=dtype, overrides=overrides)

    def estimate_replica_tps(
        self,
        *,
        model_name: str,
        gpu_type: str,
        tp: int,
        pp: int,
        input_tokens: int,
        output_tokens: int,
        dtype: str = "fp16",
        overrides: Optional[dict] = None,
    ) -> float:
        spec = self.resolve_model(model_name, dtype=dtype, overrides=overrides)
        assessment = self._assess_spec(
            spec=spec,
            gpu_type=gpu_type,
            tp=tp,
            pp=pp,
        )
        if not assessment.feasible:
            return 0.0

        if self.perfdb is not None:
            perfdb_tps = self._estimate_from_perfdb(
                spec=spec,
                gpu_type=gpu_type,
                tp=tp,
                pp=pp,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if perfdb_tps is not None:
                return perfdb_tps

        return self._estimate_from_physics(
            spec=spec,
            gpu_type=gpu_type,
            tp=tp,
            pp=pp,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def estimate_launch_timing(
        self,
        *,
        gpu_type: str,
        market: str = "on_demand",
        capacity_pressure: float = 0.0,
    ) -> LaunchTiming:
        market_penalty = 0.9 if market == "spot" else 0.0
        gpu_penalty = {
            "H100": 0.8,
            "A100": 0.6,
            "L40S": 0.4,
            "L4": 0.3,
            "A10G": 0.3,
        }.get(gpu_type, 0.5)
        pressure = max(0.0, min(capacity_pressure, 1.0))
        return LaunchTiming(
            searching_capacity_s=1.2 + pressure * 2.0 + market_penalty,
            provisioning_s=2.0 + gpu_penalty,
            bootstrapping_s=1.4 + gpu_penalty * 0.6,
            waiting_model_ready_s=1.3 + gpu_penalty * 0.4,
        )

    def assess_replica_config(
        self,
        *,
        model_name: str,
        gpu_type: str,
        tp: int,
        pp: int,
        dtype: str = "fp16",
        overrides: Optional[dict] = None,
    ) -> ReplicaConfigAssessment:
        spec = self.resolve_model(model_name, dtype=dtype, overrides=overrides)
        return self._assess_spec(spec=spec, gpu_type=gpu_type, tp=tp, pp=pp)

    def _estimate_from_perfdb(
        self,
        *,
        spec: ModelSpec,
        gpu_type: str,
        tp: int,
        pp: int,
        input_tokens: int,
        output_tokens: int,
    ) -> Optional[float]:
        rows = self.perfdb.query(
            model_name=spec.model_name,
            gpu_type=gpu_type,
            tp=tp,
            pp=pp,
            limit=20,
        )
        if not rows:
            return None

        target_ratio = input_tokens / max(output_tokens, 1)

        def score(row: dict) -> tuple[float, float]:
            row_input = row.get("input_len") or input_tokens
            row_output = row.get("output_len") or output_tokens
            row_ratio = row_input / max(row_output, 1)
            ratio_gap = abs(row_ratio - target_ratio)
            total_gap = abs(row_input - input_tokens) + abs(row_output - output_tokens)
            return ratio_gap, total_gap

        best = min(rows, key=score)
        base_tps = float(best.get("throughput_tps") or 0.0)
        if base_tps <= 0:
            return None

        row_input = best.get("input_len") or input_tokens
        row_output = best.get("output_len") or output_tokens
        io_pressure_row = self._io_pressure(row_input, row_output)
        io_pressure_target = self._io_pressure(input_tokens, output_tokens)
        adjusted = base_tps * (io_pressure_row / max(io_pressure_target, 0.1))
        return max(25.0, adjusted)

    def _estimate_from_physics(
        self,
        *,
        spec: ModelSpec,
        gpu_type: str,
        tp: int,
        pp: int,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calibrated roofline-style estimator for aggregate decode tok/s
        at continuous-batching scale (vLLM-like serving).

        Design goals:
          - Dense decode on large models is memory-bandwidth bound.
          - FLOPS contribute meaningfully (prefill/MoE experts, tensor-core speedup).
          - TP reduces weight GB per GPU → sublinear scaling; NVLink scales better than PCIe at TP≥8.
          - PP adds bubble/pipeline latency overhead but is otherwise ~free on memory.
          - Aggregate numbers should land in realistic bands, e.g.:
              Llama-3.1-70B A100-80GB TP8 PP1 1024/256 ~ 3.0–4.2k tok/s
              Llama-3.1-70B H100-PCIe TP8 PP1 1024/256 ~ 3.8–5.5k tok/s
              Llama-3.1-70B H100-SXM  TP8 PP1 1024/256 ~ 5.5–8.0k tok/s
              Llama-3.1-8B  L40S     TP1 PP1 1024/256 ~ 2.5–4.5k tok/s
        """
        gpu = lookup_gpu_spec(gpu_type)
        active_params = max(spec.active_params_billions, 0.1)
        model_size_gb = max(spec.model_size_gb, 1.0)
        gpu_mem_gb = float(gpu.get("mem_gb", 40.0) or 40.0)

        interconnect = str(gpu.get("interconnect", "PCIe"))
        tp_scale = self._tp_scale(tp, interconnect=interconnect)

        # Per-GPU weight slice and KV-cache headroom (memory-fit only).
        weights_gb_per_gpu = model_size_gb / max(tp * pp, 1)
        free_vram_per_gpu = max(gpu_mem_gb - weights_gb_per_gpu, 2.0)

        # --- Bandwidth-bound decode roofline ---
        # For a single replica, TP parallelizes bandwidth within a pipeline stage
        # (each GPU reads 1/tp of weights in parallel), while PP serializes stages:
        # every generated token must traverse the *whole* model once. So effective
        # bandwidth = bw × tp_scale, and the weight mass to traverse per decoded
        # token is model_size_gb / tp. PP doesn't help per-replica throughput here —
        # its benefit shows up only through batching amp (more VRAM headroom).
        weights_per_tp_group = max(model_size_gb / max(tp, 1), 1.0)
        bw_roof = gpu["bandwidth_gbps"] * tp_scale / weights_per_tp_group

        # --- FLOPS-bound roofline (smaller models and prefill-heavy traffic) ---
        flops_roof = gpu["fp16_tflops"] * tp_scale / max(active_params, 0.1)

        # Blend: take the minimum (the stricter bottleneck), but add a small share from the
        # other roof to avoid an abrupt transition. This is closer to reality than harmonic mean.
        dominant = min(bw_roof, flops_roof)
        slack = 0.15 * max(bw_roof, flops_roof)
        roof = dominant + slack

        # --- Continuous-batching amplification ---
        # More free VRAM after weights → more KV-cache → more concurrent tokens in flight.
        # Empirically this scales sublinearly with free VRAM up to a saturation cap.
        batching_amp = min(8.0, 1.5 + 0.9 * math.sqrt(free_vram_per_gpu))

        # --- Generation-level tensor-core efficiency ---
        gen_boost = {
            "Blackwell": 1.12,
            "Hopper": 1.08,
            "Ada": 1.00,
            "Ampere": 1.00,
        }.get(str(gpu.get("generation", "unknown")), 1.00)

        # Pipeline parallelism adds bubble overhead (~10% per extra stage).
        pp_efficiency = 1.0 / (1.0 + 0.10 * max(pp - 1, 0))

        # I/O pressure reduces aggregate decode throughput.
        io_pressure = self._io_pressure(input_tokens, output_tokens)

        # Calibration constant hand-fit against public vLLM benchmarks:
        #   Llama-3.1-70B A100-80GB TP8 PP1 1024/256 ≈ 3.5k tok/s
        #   Llama-3.1-70B H100-PCIe TP8 PP1 1024/256 ≈ 4.5k tok/s
        #   Llama-3.1-70B H100-SXM  TP8 PP1 1024/256 ≈ 6.5k tok/s
        #   Llama-3.1-8B  A100-80GB TP1 PP1 1024/256 ≈ 3.0k tok/s
        #   Llama-3.1-8B  L40S      TP1 PP1 1024/256 ≈ 2.3k tok/s
        CALIBRATION = 3.1

        tps = (
            CALIBRATION * roof * batching_amp * gen_boost * pp_efficiency / io_pressure
        )
        return max(25.0, tps)

    def _assess_spec(
        self,
        *,
        spec: ModelSpec,
        gpu_type: str,
        tp: int,
        pp: int,
    ) -> ReplicaConfigAssessment:
        if tp <= 0 or pp <= 0:
            return ReplicaConfigAssessment(
                feasible=False,
                reason=f"Invalid parallelism for {gpu_type}: TP={tp}, PP={pp}.",
                weight_gb_per_gpu=0.0,
                vram_headroom_gb=0.0,
            )

        if spec.num_attention_heads % tp != 0:
            return ReplicaConfigAssessment(
                feasible=False,
                reason=(
                    f"{gpu_type} TP={tp} is invalid for {spec.model_name}: "
                    f"TP must divide {spec.num_attention_heads} attention heads."
                ),
                weight_gb_per_gpu=0.0,
                vram_headroom_gb=0.0,
            )

        if spec.num_layers % pp != 0:
            return ReplicaConfigAssessment(
                feasible=False,
                reason=(
                    f"{gpu_type} PP={pp} is invalid for {spec.model_name}: "
                    f"PP must divide {spec.num_layers} layers."
                ),
                weight_gb_per_gpu=0.0,
                vram_headroom_gb=0.0,
            )

        gpu = lookup_gpu_spec(gpu_type)
        gpu_mem_gb = float(gpu.get("mem_gb", 0.0) or 0.0)
        weight_gb_per_gpu = spec.model_size_gb / max(tp * pp, 1)
        vram_headroom_gb = gpu_mem_gb - weight_gb_per_gpu
        if vram_headroom_gb < self.MIN_VRAM_HEADROOM_GB:
            return ReplicaConfigAssessment(
                feasible=False,
                reason=(
                    f"{gpu_type} TP={tp} PP={pp} is not feasible for {spec.model_name}: "
                    f"{weight_gb_per_gpu:.1f}GB weights per GPU leaves "
                    f"{vram_headroom_gb:.1f}GB headroom on {gpu_mem_gb:.1f}GB VRAM "
                    f"(need at least {self.MIN_VRAM_HEADROOM_GB:.1f}GB)."
                ),
                weight_gb_per_gpu=weight_gb_per_gpu,
                vram_headroom_gb=vram_headroom_gb,
            )

        return ReplicaConfigAssessment(
            feasible=True,
            reason=None,
            weight_gb_per_gpu=weight_gb_per_gpu,
            vram_headroom_gb=vram_headroom_gb,
        )

    @staticmethod
    def _tp_scale(tp: int, *, interconnect: str) -> float:
        """Sublinear TP scaling. NVLink keeps good efficiency at TP>=8; PCIe taper is real but
        not catastrophic for well-tuned serving runtimes."""
        tp = max(tp, 1)
        if interconnect == "NVLink":
            # ~0.85 exponent: TP1=1, TP2=1.80, TP4=3.25, TP8=5.86
            return tp**0.85
        # PCIe: good up to TP4, softer beyond.
        if tp <= 4:
            return tp**0.80
        # TP8+ on PCIe suffers but not collapse.
        return (4**0.80) * ((tp / 4) ** 0.55)

    @staticmethod
    def _io_pressure(input_tokens: int, output_tokens: int) -> float:
        return 1.0 + (input_tokens / 2048.0) * 0.25 + (output_tokens / 1024.0) * 0.55
