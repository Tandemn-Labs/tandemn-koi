"""Backend-neutral GPU and data-type compatibility resolution.

Callers supply backend availability and optional catalog facts. Resolutions never
mutate candidates or call a predictor; adapters apply the returned backend value,
scales, confidence, and provenance to their own estimates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

ResolutionKind = Literal["exact", "alias", "nearest", "unsupported"]
MAX_GPU_DISTANCE = 3.0


@dataclass(frozen=True)
class GPUProfile:
    canonical_name: str
    vendor: str = "nvidia"
    architecture: str | None = None
    memory_gb: float | None = None
    memory_bandwidth_gbps: float | None = None
    fp16_tflops: float | None = None
    nvlink_bandwidth_gbps: float | None = None
    pcie_bandwidth_gbps: float | None = None
    supported_dtypes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CompatibilityResolution:
    dimension: Literal["gpu", "dtype"]
    backend: str
    requested: str
    canonical: str
    resolved: str | None
    backend_value: str | None
    kind: ResolutionKind
    distance: float
    confidence: float
    throughput_scale: float
    latency_scale: float
    reasons: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return self.resolved is not None

    @property
    def approximate(self) -> bool:
        return self.kind == "nearest"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


_GPU_ALIASES = {
    "A10": "A10G",
    "A10G": "A10G",
    "A30": "A30",
    "A100": "A100",
    "A10040GB": "A100 40GB",
    "A10080GB": "A100 80GB",
    "A100PCIE": "A100 PCIE",
    "A100SXM": "A100",
    "B200": "B200",
    "B200SXM": "B200",
    "B300": "B300",
    "B300SXM": "B300",
    "B60": "B60",
    "GB10": "GB10",
    "GB200": "GB200",
    "GB200SXM": "GB200",
    "GB300": "GB300",
    "H100": "H100",
    "H10080GB": "H100",
    "H100PCIE": "H100 PCIE",
    "H100SXM": "H100",
    "H200": "H200",
    "H200SXM": "H200",
    "L4": "L4",
    "L40": "L40",
    "L40S": "L40S",
    "MI200": "MI200",
    "MI300": "MI300",
    "RTXPRO6000": "RTX PRO 6000",
    "RTXPRO6000SERVER": "RTX PRO 6000",
    "T4": "T4",
    "V100": "V100",
    "V100PCIE": "V100 PCIE",
    "V100SXM": "V100",
}

_BACKEND_GPU_NAMES = {
    "aic": {
        "A30": "a30",
        "A100": "a100_sxm",
        "A100 40GB": "a100_sxm",
        "A100 80GB": "a100_sxm",
        "A100 PCIE": "a100_pcie",
        "B200": "b200_sxm",
        "B300": "b300_sxm",
        "B60": "b60",
        "GB200": "gb200",
        "GB300": "gb300",
        "H100": "h100_sxm",
        "H100 PCIE": "h100_pcie",
        "H200": "h200_sxm",
        "L4": "l4",
        "L40S": "l40s",
        "RTX PRO 6000": "rtx_pro_6000_server",
    },
    "solver": {
        "A10G": "A10G",
        "A100": "A100",
        "A100 40GB": "A100",
        "A100 80GB": "A100",
        "H100": "H100",
        "L4": "L4",
        "L40S": "L40S",
    },
}

_GPU_PROFILES = {
    "A10G": GPUProfile(
        "A10G", architecture="ampere", memory_gb=24, memory_bandwidth_gbps=600, fp16_tflops=62.5
    ),
    "A30": GPUProfile(
        "A30", architecture="ampere", memory_gb=24, memory_bandwidth_gbps=933, fp16_tflops=165
    ),
    "A100": GPUProfile(
        "A100",
        architecture="ampere",
        memory_gb=80,
        memory_bandwidth_gbps=2039,
        fp16_tflops=312,
        nvlink_bandwidth_gbps=600,
    ),
    "A100 40GB": GPUProfile(
        "A100 40GB",
        architecture="ampere",
        memory_gb=40,
        memory_bandwidth_gbps=1555,
        fp16_tflops=312,
        nvlink_bandwidth_gbps=600,
    ),
    "A100 80GB": GPUProfile(
        "A100 80GB",
        architecture="ampere",
        memory_gb=80,
        memory_bandwidth_gbps=2039,
        fp16_tflops=312,
        nvlink_bandwidth_gbps=600,
    ),
    "H100": GPUProfile(
        "H100",
        architecture="hopper",
        memory_gb=80,
        memory_bandwidth_gbps=3350,
        fp16_tflops=989,
        nvlink_bandwidth_gbps=900,
    ),
    "H200": GPUProfile(
        "H200",
        architecture="hopper",
        memory_gb=141,
        memory_bandwidth_gbps=4800,
        fp16_tflops=989,
        nvlink_bandwidth_gbps=900,
    ),
    "L4": GPUProfile(
        "L4", architecture="ada", memory_gb=24, memory_bandwidth_gbps=300, fp16_tflops=121
    ),
    "L40": GPUProfile(
        "L40", architecture="ada", memory_gb=48, memory_bandwidth_gbps=864, fp16_tflops=181
    ),
    "L40S": GPUProfile(
        "L40S", architecture="ada", memory_gb=48, memory_bandwidth_gbps=864, fp16_tflops=362
    ),
    "RTX PRO 6000": GPUProfile(
        "RTX PRO 6000",
        architecture="blackwell",
        memory_gb=96,
        memory_bandwidth_gbps=1792,
        fp16_tflops=467.8,
        pcie_bandwidth_gbps=64,
    ),
    "T4": GPUProfile(
        "T4", architecture="turing", memory_gb=16, memory_bandwidth_gbps=320, fp16_tflops=65
    ),
    "V100": GPUProfile(
        "V100", architecture="volta", memory_gb=32, memory_bandwidth_gbps=900, fp16_tflops=125
    ),
    "V100 PCIE": GPUProfile(
        "V100 PCIE",
        architecture="volta",
        memory_gb=16,
        memory_bandwidth_gbps=900,
        fp16_tflops=125,
    ),
}

_PREFERRED_GPU_PROXIES = {
    "aic": {
        "A10G": "A30",
        "A100 40GB": "A100",
        "GB10": "L40S",
        "L40": "L40S",
        "T4": "L4",
        "V100": "A30",
        "V100 PCIE": "A30",
    },
    "solver": {"RTX PRO 6000": "L40S"},
}

_BACKEND_DTYPE_NAMES = {
    "aic": {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp8": "fp8",
        "fp8_e4m3": "fp8",
        "fp8_e5m2": "fp8",
        "int8": "int8",
    }
}

_DTYPE_ALIASES = {
    "auto": "auto",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float8": "fp8",
    "float8e4m3fn": "fp8_e4m3",
    "float8e5m2": "fp8_e5m2",
    "fp16": "fp16",
    "fp32": "fp32",
    "fp4": "fp4",
    "fp8": "fp8",
    "fp8e4m3": "fp8_e4m3",
    "fp8e5m2": "fp8_e5m2",
    "half": "fp16",
    "int4": "int4",
    "int8": "int8",
    "nf4": "nf4",
}

_DTYPE_FALLBACKS = {
    "auto": ("bf16", "fp16"),
    "bf16": ("fp16",),
    "fp16": ("bf16",),
    "fp8": ("fp8_e4m3", "fp8_e5m2", "bf16", "fp16"),
    "fp8_e4m3": ("fp8", "fp8_e5m2", "bf16", "fp16"),
    "fp8_e5m2": ("fp8", "fp8_e4m3", "bf16", "fp16"),
    "int8": ("bf16", "fp16"),
    "int4": ("int8", "bf16", "fp16"),
    "nf4": ("int4", "int8", "bf16", "fp16"),
    "fp4": ("fp8", "bf16", "fp16"),
}


def canonicalize_gpu(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    key = re.sub(r"[^A-Z0-9]", "", text.upper())
    if key.startswith("NVIDIA"):
        key = key.removeprefix("NVIDIA")
    return _GPU_ALIASES.get(key, text.upper().replace("_", " ").replace("-", " "))


def canonicalize_dtype(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    key = re.sub(r"[^a-z0-9]", "", text)
    return _DTYPE_ALIASES.get(key, text)


def gpu_profile_from_values(name: str, values: dict[str, Any] | None) -> GPUProfile | None:
    canonical = canonicalize_gpu(name)
    if canonical is None:
        return None
    values = values or {}
    builtin = _GPU_PROFILES.get(canonical)
    vendor = str(values.get("gpu_vendor") or (builtin.vendor if builtin else _vendor(canonical)))
    return GPUProfile(
        canonical_name=canonical,
        vendor=vendor.lower(),
        architecture=_text(values.get("gpu_generation"))
        or (builtin.architecture if builtin else None),
        memory_gb=_number(values.get("gpu_mem_gb")) or (builtin.memory_gb if builtin else None),
        memory_bandwidth_gbps=_number(values.get("gpu_bandwidth_gbps"))
        or (builtin.memory_bandwidth_gbps if builtin else None),
        fp16_tflops=_number(values.get("gpu_tflops_fp16"))
        or (builtin.fp16_tflops if builtin else None),
        nvlink_bandwidth_gbps=_number(values.get("nvlink_bandwidth_gbps"))
        or (builtin.nvlink_bandwidth_gbps if builtin else None),
        pcie_bandwidth_gbps=_number(values.get("pcie_bandwidth_gbps"))
        or (builtin.pcie_bandwidth_gbps if builtin else None),
        supported_dtypes=frozenset(
            canonicalize_dtype(dtype) or str(dtype) for dtype in values.get("supported_dtypes", ())
        )
        or (builtin.supported_dtypes if builtin else frozenset()),
    )


def resolve_gpu(
    requested: Any,
    *,
    backend: str,
    available: set[str] | frozenset[str],
    requested_profile: dict[str, Any] | GPUProfile | None = None,
    candidate_profiles: dict[str, GPUProfile | dict[str, Any]] | None = None,
    allow_approximation: bool = True,
    allow_larger_memory_proxy: bool = False,
    default_proxy: str | None = None,
) -> CompatibilityResolution:
    """Resolve a requested GPU to an exact, aliased, or nearest backend choice."""
    requested_text = str(requested or "")
    canonical = canonicalize_gpu(requested)
    choices = {choice for raw in available if (choice := canonicalize_gpu(raw)) is not None}
    if canonical is None:
        return _unsupported("gpu", backend, requested_text, "", "missing_gpu")
    if canonical in choices:
        kind: ResolutionKind = "exact" if requested_text.strip().upper() == canonical else "alias"
        return _resolution(
            "gpu",
            backend,
            requested_text,
            canonical,
            canonical,
            kind,
            0.0,
            1.0,
            1.0,
            ("exact_backend_support",),
        )
    if not allow_approximation or not choices:
        return _unsupported("gpu", backend, requested_text, canonical, "no_compatible_gpu")

    profile = (
        requested_profile
        if isinstance(requested_profile, GPUProfile)
        else gpu_profile_from_values(canonical, requested_profile)
    )
    profiles: dict[str, GPUProfile] = dict(_GPU_PROFILES)
    for name, raw_profile in (candidate_profiles or {}).items():
        candidate_name = canonicalize_gpu(name)
        if candidate_name is None:
            continue
        profiles[candidate_name] = (
            raw_profile
            if isinstance(raw_profile, GPUProfile)
            else gpu_profile_from_values(candidate_name, raw_profile) or GPUProfile(candidate_name)
        )

    preferred = canonicalize_gpu((_PREFERRED_GPU_PROXIES.get(backend) or {}).get(canonical))
    if preferred in choices:
        preferred_profile = profiles.get(preferred)
        if (
            profile is not None
            and preferred_profile is not None
            and (
                profile.vendor != preferred_profile.vendor
                or (
                    not allow_larger_memory_proxy
                    and profile.memory_gb
                    and preferred_profile.memory_gb
                    and preferred_profile.memory_gb > profile.memory_gb * 1.05
                )
                or (allow_larger_memory_proxy and profile.memory_gb is None)
            )
        ):
            preferred = None
    if preferred in choices:
        preferred_profile = profiles.get(preferred)
        distance, distance_reasons = (
            _gpu_distance(profile, preferred_profile)
            if profile is not None and preferred_profile is not None
            else (1.0, ())
        )
        if distance <= MAX_GPU_DISTANCE:
            return _nearest_gpu(
                requested_text,
                canonical,
                backend,
                preferred,
                profile,
                preferred_profile,
                distance,
                ("preferred_backend_proxy", *distance_reasons),
            )
    fallback = canonicalize_gpu(default_proxy)
    fallback_profile = profiles.get(fallback) if fallback is not None else None
    if (
        profile is not None
        and fallback_profile is not None
        and profile.vendor != fallback_profile.vendor
    ):
        fallback = None
    if (
        fallback in choices
        and profile is not None
        and profile.memory_gb is not None
        and fallback_profile is not None
        and (
            fallback_profile.memory_gb is None
            or fallback_profile.memory_gb <= profile.memory_gb * 1.05
            or allow_larger_memory_proxy
        )
        and not _profile_has_capacity(profile)
    ):
        return _nearest_gpu(
            requested_text,
            canonical,
            backend,
            fallback,
            profile,
            fallback_profile,
            1.0,
            ("default_backend_proxy",),
        )
    if profile is None or not _profile_has_capacity(profile):
        return _unsupported(
            "gpu", backend, requested_text, canonical, "missing_gpu_capability_profile"
        )

    ranked = []
    for choice in choices:
        candidate_profile = profiles.get(choice)
        if profile is None or candidate_profile is None:
            continue
        if (
            profile.vendor
            and candidate_profile.vendor
            and profile.vendor != candidate_profile.vendor
        ):
            continue
        if (
            profile.memory_gb
            and candidate_profile.memory_gb
            and candidate_profile.memory_gb > profile.memory_gb * 1.05
        ):
            continue
        distance, reasons = _gpu_distance(profile, candidate_profile)
        if distance > MAX_GPU_DISTANCE:
            continue
        ranked.append((distance, choice, candidate_profile, reasons))
    if ranked:
        distance, choice, candidate, reasons = min(ranked, key=lambda item: (item[0], item[1]))
        return _nearest_gpu(
            requested_text, canonical, backend, choice, profile, candidate, distance, reasons
        )
    return _unsupported("gpu", backend, requested_text, canonical, "no_compatible_gpu_profile")


def resolve_dtype(
    requested: Any,
    *,
    backend: str,
    component: str,
    available: set[str] | frozenset[str],
    allow_approximation: bool = True,
) -> CompatibilityResolution:
    """Resolve one component dtype using directional, conservative fallback rules."""
    requested_text = str(requested or "")
    canonical = canonicalize_dtype(requested)
    choices = {choice for raw in available if (choice := canonicalize_dtype(raw)) is not None}
    if canonical is None:
        return _unsupported("dtype", backend, requested_text, "", "missing_dtype")
    if canonical in choices:
        kind: ResolutionKind = "exact" if requested_text.strip().lower() == canonical else "alias"
        return _resolution(
            "dtype",
            backend,
            requested_text,
            canonical,
            canonical,
            kind,
            0.0,
            1.0,
            1.0,
            (f"exact_{component}_dtype",),
        )
    if not allow_approximation:
        return _unsupported(
            "dtype", backend, requested_text, canonical, f"unsupported_{component}_dtype"
        )
    for index, choice in enumerate(_DTYPE_FALLBACKS.get(canonical, ()), start=1):
        if choice not in choices:
            continue
        confidence = max(0.35, 0.8 - 0.1 * (index - 1))
        if {canonical, choice} == {"bf16", "fp16"}:
            reason = "same_width_dtype_proxy"
            throughput_scale = 1.0
        else:
            reason = "higher_precision_conservative_proxy"
            throughput_scale = 1.0
        return _resolution(
            "dtype",
            backend,
            requested_text,
            canonical,
            choice,
            "nearest",
            float(index),
            confidence,
            throughput_scale,
            (reason, f"component:{component}"),
        )
    return _unsupported(
        "dtype", backend, requested_text, canonical, f"unsupported_{component}_dtype"
    )


def backend_gpu_name(canonical: str, backend: str) -> str | None:
    return (_BACKEND_GPU_NAMES.get(backend) or {}).get(canonical)


def backend_dtype_name(canonical: str, backend: str) -> str:
    return (_BACKEND_DTYPE_NAMES.get(backend) or {}).get(canonical, canonical)


def gpu_profile_distance(
    requested: GPUProfile,
    candidate: GPUProfile,
    *,
    allow_cross_vendor: bool = False,
) -> tuple[float, tuple[str, ...]]:
    """Return normalized capability distance and its contributing reasons."""
    if requested.vendor != candidate.vendor and not allow_cross_vendor:
        return math.inf, ("cross_vendor_rejected",)
    distance, reasons = _gpu_distance(requested, candidate)
    if requested.vendor != candidate.vendor:
        distance = math.sqrt(distance**2 + 12.25)
        reasons = (*reasons, "cross_vendor_penalty")
    return distance, reasons


def _resolution(
    dimension,
    backend,
    requested,
    canonical,
    resolved,
    kind,
    distance,
    confidence,
    throughput_scale,
    reasons,
):
    backend_value = (
        backend_gpu_name(resolved, backend)
        if dimension == "gpu"
        else backend_dtype_name(resolved, backend)
    )
    if dimension == "gpu" and backend_value is None and backend == "perfdb":
        backend_value = resolved
    return CompatibilityResolution(
        dimension=dimension,
        backend=backend,
        requested=requested,
        canonical=canonical,
        resolved=resolved,
        backend_value=backend_value,
        kind=kind,
        distance=float(distance),
        confidence=max(0.0, min(1.0, float(confidence))),
        throughput_scale=float(throughput_scale),
        latency_scale=1.0 / max(float(throughput_scale), 1e-12),
        reasons=tuple(reasons),
    )


def _unsupported(dimension, backend, requested, canonical, reason):
    return CompatibilityResolution(
        dimension=dimension,
        backend=backend,
        requested=requested,
        canonical=canonical,
        resolved=None,
        backend_value=None,
        kind="unsupported",
        distance=math.inf,
        confidence=0.0,
        throughput_scale=0.0,
        latency_scale=math.inf,
        reasons=(reason,),
    )


def _nearest_gpu(
    requested, canonical, backend, resolved, requested_profile, candidate_profile, distance, reasons
):
    scale = _throughput_scale(requested_profile, candidate_profile)
    confidence = max(0.2, min(0.75, math.exp(-0.5 * distance)))
    return _resolution(
        "gpu",
        backend,
        requested,
        canonical,
        resolved,
        "nearest",
        distance,
        confidence,
        scale,
        (*reasons, "nearest_compatible_gpu"),
    )


def _gpu_distance(requested: GPUProfile, candidate: GPUProfile) -> tuple[float, tuple[str, ...]]:
    terms = []
    reasons = []
    for name, weight in (
        ("fp16_tflops", 2.0),
        ("memory_bandwidth_gbps", 2.0),
        ("memory_gb", 1.0),
        ("nvlink_bandwidth_gbps", 0.5),
        ("pcie_bandwidth_gbps", 0.5),
    ):
        left = getattr(requested, name)
        right = getattr(candidate, name)
        if left and right:
            terms.append(weight * math.log(left / right) ** 2)
            reasons.append(f"compared:{name}")
    if requested.architecture and candidate.architecture:
        if requested.architecture.lower() == candidate.architecture.lower():
            reasons.append("same_architecture")
        else:
            terms.append(2.25)
            reasons.append("architecture_penalty")
    return math.sqrt(sum(terms)) if terms else 1.0, tuple(reasons)


def _throughput_scale(requested: GPUProfile | None, candidate: GPUProfile | None) -> float:
    if requested is None or candidate is None:
        return 0.25
    ratios = []
    for name in ("fp16_tflops", "memory_bandwidth_gbps"):
        left = getattr(requested, name)
        right = getattr(candidate, name)
        if left and right:
            ratios.append(left / right)
    return max(0.01, min(1.0, min(ratios))) if ratios else 0.25


def _profile_has_capacity(profile: GPUProfile) -> bool:
    return bool(profile.fp16_tflops or profile.memory_bandwidth_gbps)


def _vendor(canonical: str) -> str:
    return "amd" if canonical.startswith("MI") else "nvidia"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
