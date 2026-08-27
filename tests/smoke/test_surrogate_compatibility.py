import math

from src.prediction.compatibility import (
    CompatibilityResolution,
    canonicalize_dtype,
    canonicalize_gpu,
    resolve_dtype,
    resolve_gpu,
)


def test_gpu_aliases_are_canonical_and_exact_backend_values_are_preserved():
    assert canonicalize_gpu("nvidia-a10g") == "A10G"
    assert canonicalize_gpu("A100_80GB") == "A100 80GB"
    assert canonicalize_gpu("A100_40GB") == "A100 40GB"
    assert canonicalize_gpu("nvidia-RTXPRO6000") == "RTX PRO 6000"

    resolution = resolve_gpu(
        "NVIDIA_L4",
        backend="aic",
        available=frozenset({"L4", "L40S"}),
    )

    assert resolution.kind == "alias"
    assert resolution.resolved == "L4"
    assert resolution.backend_value == "l4"
    assert resolution.confidence == 1.0


def test_a10g_resolves_to_l4_aic_gpu_with_catalog_profiles():
    resolution = resolve_gpu(
        "nvidia-a10g",
        backend="aic",
        available=frozenset({"A30", "L4", "L40S", "H100"}),
    )

    assert resolution.kind == "nearest"
    assert resolution.resolved == "L4"
    assert resolution.backend_value == "l4"
    assert 0 < resolution.throughput_scale < 1
    assert resolution.latency_scale > 1
    assert "preferred_backend_proxy" in resolution.reasons


def test_unknown_gpu_uses_catalog_facts_and_conservative_scaling():
    resolution = resolve_gpu(
        "future-gpu",
        backend="solver",
        available=frozenset({"H100", "L40S"}),
        requested_profile={
            "gpu_vendor": "nvidia",
            "gpu_generation": "ada",
            "gpu_mem_gb": 48,
            "gpu_bandwidth_gbps": 432,
            "gpu_tflops_fp16": 181,
        },
        default_proxy="L40S",
    )

    assert resolution.resolved == "L40S"
    assert resolution.backend_value == "L40S"
    assert resolution.throughput_scale == 0.5
    assert resolution.latency_scale == 2.0


def test_unknown_gpu_without_facts_is_not_arbitrarily_substituted():
    resolution = resolve_gpu(
        "future-gpu",
        backend="solver",
        available=frozenset({"A100", "L40S"}),
        default_proxy="L40S",
    )

    assert resolution.kind == "unsupported"
    assert resolution.resolved is None


def test_cross_vendor_gpu_is_not_silently_substituted():
    resolution = resolve_gpu(
        "MI300",
        backend="aic",
        available=frozenset({"A100", "H100", "L40S"}),
        requested_profile={
            "gpu_vendor": "amd",
            "gpu_bandwidth_gbps": 5000,
            "gpu_tflops_fp16": 1000,
        },
    )

    assert not resolution.supported
    assert resolution.kind == "unsupported"
    assert math.isinf(resolution.distance)


def test_larger_memory_or_distant_proxy_is_rejected():
    larger_memory = resolve_gpu(
        "L4",
        backend="solver",
        available=frozenset({"L40S"}),
    )
    distant = resolve_gpu(
        "tiny-future-gpu",
        backend="solver",
        available=frozenset({"L4", "L40S"}),
        requested_profile={
            "gpu_vendor": "nvidia",
            "gpu_mem_gb": 8,
            "gpu_bandwidth_gbps": 10,
            "gpu_tflops_fp16": 2,
        },
    )

    assert not larger_memory.supported
    assert not distant.supported


def test_dtype_alias_and_directional_fallback_rules():
    assert canonicalize_dtype("bfloat16") == "bf16"
    assert canonicalize_dtype("float8_e4m3fn") == "fp8_e4m3"

    alias = resolve_dtype(
        "bfloat16",
        backend="aic",
        component="fmha",
        available=frozenset({"bf16"}),
    )
    conservative = resolve_dtype(
        "fp8_e4m3",
        backend="perfdb",
        component="kv_cache",
        available=frozenset({"bf16"}),
    )
    unsupported_fp32 = resolve_dtype(
        "fp32",
        backend="solver",
        component="weights",
        available=frozenset({"bf16"}),
    )

    assert alias.kind == "alias"
    assert alias.backend_value == "bfloat16"
    assert conservative.resolved == "bf16"
    assert conservative.throughput_scale == 1.0
    assert "higher_precision_conservative_proxy" in conservative.reasons
    assert not unsupported_fp32.supported


def test_unsupported_dtype_and_resolution_fingerprint_are_stable():
    unsupported = resolve_dtype(
        "fp32",
        backend="aic",
        component="fmha",
        available=frozenset({"fp8"}),
    )
    resolution = resolve_gpu(
        "nvidia-a10g",
        backend="aic",
        available=frozenset({"A30", "L40S"}),
    )
    restored = CompatibilityResolution(**resolution.to_dict())

    assert not unsupported.supported
    assert restored == resolution
    assert restored.fingerprint() == resolution.fingerprint()
