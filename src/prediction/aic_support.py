"""Cached AIC support-profile inventory built from public AIC SDK APIs."""

from __future__ import annotations

from functools import lru_cache

from src.prediction.compatibility import GPUProfile, canonicalize_gpu
from src.prediction.profile_search import SupportedProfile, model_profile_from_values


@lru_cache(maxsize=8)
def load_aic_support_profiles(backend: str, version: str) -> tuple[SupportedProfile, ...]:
    """Return measured aggregate AIC model/system support points."""
    from aiconfigurator.sdk import common  # type: ignore[import-untyped]
    from aiconfigurator.sdk.perf_database import (  # type: ignore[import-untyped]
        load_system_spec,
    )
    from aiconfigurator.sdk.utils import (  # type: ignore[import-untyped]
        get_model_config_from_model_path,
    )

    rows = [
        row
        for row in common.get_support_matrix()
        if row.get("Mode") == "agg"
        and row.get("Status") == "PASS"
        and row.get("Backend") == backend
        and row.get("Version") == version
        and ("Source" not in row or row.get("Source") == "silicon")
    ]
    systems = {}
    models = {}
    output = []
    seen = set()
    for row in rows:
        system = str(row["System"])
        model_id = str(row["HuggingFaceID"])
        key = (system, model_id)
        if key in seen:
            continue
        seen.add(key)
        if system not in systems:
            systems[system] = _gpu_profile(system, load_system_spec(system))
        if model_id not in models:
            try:
                config = get_model_config_from_model_path(model_id)
            except Exception:
                config = None
            models[model_id] = (
                model_profile_from_values(model_id, config) if isinstance(config, dict) else None
            )
        gpu = systems[system]
        model = models[model_id]
        if gpu is None or model is None:
            continue
        output.append(
            SupportedProfile(
                profile_id=f"{backend}:{version}:{system}:{model_id}",
                gpu=gpu,
                model=model,
                engine_name=backend,
                engine_version=version,
                aic_system=system,
            )
        )
    return tuple(sorted(output, key=lambda profile: profile.profile_id))


def _gpu_profile(system: str, spec: dict) -> GPUProfile | None:
    gpu = spec.get("gpu") or {}
    node = spec.get("node") or {}
    canonical = canonicalize_gpu(system)
    if canonical is None or not gpu:
        return None
    return GPUProfile(
        canonical_name=canonical,
        vendor="nvidia",
        architecture=_architecture(gpu.get("sm_version")),
        memory_gb=_scale(gpu.get("mem_capacity"), 1 << 30),
        memory_bandwidth_gbps=_scale(gpu.get("mem_bw"), 1e9),
        fp16_tflops=_scale(
            gpu.get("bfloat16_tc_flops") or gpu.get("float16_tc_flops"),
            1e12,
        ),
        nvlink_bandwidth_gbps=_scale(node.get("intra_node_bw"), 1e9),
        pcie_bandwidth_gbps=_scale(node.get("pcie_bw"), 1e9),
        supported_dtypes=frozenset(
            dtype
            for dtype, field in (
                ("bf16", "bfloat16_tc_flops"),
                ("fp16", "float16_tc_flops"),
                ("fp8", "fp8_tc_flops"),
                ("int8", "int8_tc_flops"),
            )
            if gpu.get(field)
        ),
    )


def _architecture(sm_version) -> str | None:
    try:
        sm = int(sm_version)
    except (TypeError, ValueError):
        return None
    if sm >= 100:
        return "blackwell"
    if sm >= 90:
        return "hopper"
    if sm >= 89:
        return "ada"
    if sm >= 80:
        return "ampere"
    if sm >= 75:
        return "turing"
    return "volta" if sm >= 70 else None


def _scale(value, divisor: float) -> float | None:
    try:
        return float(value) / divisor
    except (TypeError, ValueError):
        return None
