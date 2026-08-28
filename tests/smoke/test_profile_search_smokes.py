from src.prediction.aic_support import load_aic_support_profiles
from src.prediction.compatibility import GPUProfile
from src.prediction.profile_search import (
    ModelProfile,
    PredictionProfile,
    SupportedProfile,
    TopologyProfile,
    WorkloadProfile,
    build_operation_signature,
    model_profile_from_values,
    rank_profiles,
)


def _model(model_id="target", params=8e9, hidden=4096, layers=32, dtype="bf16"):
    return ModelProfile(
        model_id=model_id,
        architecture="LlamaForCausalLM",
        layers=layers,
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        attention_heads=32,
        kv_heads=8,
        head_dim=128,
        vocab_size=128000,
        parameter_count=params,
        is_moe=False,
        routed_experts=0,
        active_experts=0,
        weight_dtype=dtype,
        max_context=8192,
    )


def _gpu(name="H100", compute=989, bandwidth=3350, memory=80, vendor="nvidia"):
    return GPUProfile(
        name,
        vendor=vendor,
        architecture="hopper",
        memory_gb=memory,
        memory_bandwidth_gbps=bandwidth,
        fp16_tflops=compute,
        nvlink_bandwidth_gbps=900,
    )


def _requested(gpu=None, model=None, workload=None):
    return PredictionProfile(
        gpu=gpu or _gpu(),
        model=model or _model(),
        workload=workload or WorkloadProfile("online", 512, 128, 8, 4.0),
        topology=TopologyProfile(tp=1, pp=1, ep=1, dp=2),
        engine_name="vllm",
        engine_version="0.22.0",
    )


def _supported(profile_id, gpu=None, model=None):
    return SupportedProfile(
        profile_id=profile_id,
        gpu=gpu or _gpu(),
        model=model or _model(model_id="proxy"),
        engine_name="vllm",
        engine_version="0.22.0",
        aic_system=(gpu or _gpu()).canonical_name.lower(),
    )


def test_operation_signature_increases_with_model_and_workload_size():
    topology = TopologyProfile()
    small = build_operation_signature(
        _model(params=8e9), WorkloadProfile("online", 512, 128, 4), topology
    )
    large = build_operation_signature(
        _model(params=70e9, hidden=8192, layers=80),
        WorkloadProfile("online", 4096, 512, 16),
        topology,
    )

    assert large.weight_bytes_per_gpu > small.weight_bytes_per_gpu
    assert large.prefill_flops > small.prefill_flops
    assert large.decode_flops_per_token > small.decode_flops_per_token
    assert large.kv_bytes_per_token_per_gpu > small.kv_bytes_per_token_per_gpu
    assert small.decode_memory_bytes_per_token > small.weight_bytes_per_gpu

    tp8 = build_operation_signature(
        _model(params=8e9),
        WorkloadProfile("online", 512, 128, 4),
        TopologyProfile(tp=8),
    )
    assert tp8.prefill_flops < small.prefill_flops / 7
    assert tp8.decode_flops_per_token < small.decode_flops_per_token / 7


def test_missing_architecture_uses_structural_model_profile_instead_of_failing():
    profile = model_profile_from_values(
        "acme/unknown-8b",
        {
            "model_params_b": 8,
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attn_heads": 32,
            "weight_dtype": "bf16",
        },
    )

    assert profile is not None
    assert profile.architecture == "generic_dense"
    assert profile.intermediate_size == 16384
    assert profile.kv_heads == 32


def test_moe_signature_uses_active_experts_for_compute_and_total_experts_for_memory():
    dense = _model(params=30e9, hidden=2048, layers=48)
    moe = ModelProfile(
        **{
            **dense.__dict__,
            "model_id": "moe",
            "is_moe": True,
            "routed_experts": 128,
            "active_experts": 8,
            "moe_intermediate_size": 768,
        }
    )
    workload = WorkloadProfile("online", 512, 128, 4)

    dense_signature = build_operation_signature(dense, workload, TopologyProfile())
    moe_signature = build_operation_signature(moe, workload, TopologyProfile())
    moe_ep8 = build_operation_signature(
        moe,
        workload,
        TopologyProfile(ep=8),
    )

    assert moe_signature.weight_bytes_per_gpu == dense_signature.weight_bytes_per_gpu
    assert moe_signature.decode_flops_per_token < dense_signature.decode_flops_per_token
    assert moe_ep8.weight_bytes_per_gpu < moe_signature.weight_bytes_per_gpu
    assert moe_ep8.decode_flops_per_token < moe_signature.decode_flops_per_token
    assert moe_ep8.communication_bytes_per_token > moe_signature.communication_bytes_per_token


def test_joint_knn_ranks_exact_profile_first_and_is_deterministic():
    requested = _requested()
    exact = _supported("exact", model=_model(model_id="exact"))
    far_gpu = _supported("far-gpu", gpu=_gpu("L4", compute=121, bandwidth=300, memory=24))
    far_model = _supported("far-model", model=_model("large", params=70e9, hidden=8192, layers=80))

    first = rank_profiles(requested, [far_model, far_gpu, exact], limit=3)
    second = rank_profiles(requested, [exact, far_gpu, far_model], limit=3)

    assert [match.supported.profile_id for match in first] == [
        match.supported.profile_id for match in second
    ]
    assert first[0].supported.profile_id == "exact"
    assert first[0].distance == 0.0
    assert first[0].prefill_speed_ratio == 1.0
    assert first[0].decode_speed_ratio == 1.0


def test_joint_knn_normalizes_slower_target_before_queueing():
    requested = _requested(gpu=_gpu("slow", compute=100, bandwidth=500, memory=80))
    proxy = _supported("fast-proxy", gpu=_gpu())

    match = rank_profiles(requested, [proxy], limit=1)[0]

    assert 0 < match.prefill_speed_ratio < 1
    assert 0 < match.decode_speed_ratio < 1
    assert match.confidence < 1


def test_joint_knn_accounts_for_kv_dtype_separately_from_weight_dtype():
    requested_model = ModelProfile(**{**_model().__dict__, "kv_cache_dtype": "fp8"})
    requested = _requested(model=requested_model)
    proxy = _supported("bf16-kv")

    match = rank_profiles(requested, [proxy], limit=1)[0]
    target_signature = build_operation_signature(
        requested_model, requested.workload, requested.topology
    )
    proxy_signature = build_operation_signature(proxy.model, requested.workload, requested.topology)

    assert target_signature.kv_bytes_per_token_per_gpu < proxy_signature.kv_bytes_per_token_per_gpu
    assert "higher_precision_dtype_proxy" in match.reasons


def test_joint_knn_keeps_requested_workload_and_penalizes_context_extrapolation():
    requested = _requested(workload=WorkloadProfile("online", 9000, 1000, 8, 2.0))
    short_context = _supported("short", model=_model("short"))
    long_context = _supported(
        "long",
        model=ModelProfile(**{**_model("long").__dict__, "max_context": 32768}),
    )

    matches = rank_profiles(requested, [short_context, long_context], limit=2)

    assert matches[0].supported.profile_id == "long"
    assert requested.workload.input_tokens == 9000
    assert "context_extrapolation" in matches[1].reasons


def test_joint_knn_filters_model_that_cannot_fit_requested_vram():
    requested = _requested(
        gpu=_gpu(memory=24),
        model=_model(params=70e9, hidden=8192, layers=80),
    )

    assert rank_profiles(requested, [_supported("proxy")]) == ()


def test_joint_knn_allows_cross_vendor_only_with_penalty():
    requested = _requested(gpu=_gpu("MI300X", vendor="amd"))
    match = rank_profiles(requested, [_supported("H100")], limit=1)[0]

    assert "cross_vendor_penalty" in match.reasons
    assert match.confidence < 0.2


def test_aic_support_index_uses_public_support_matrix():
    profiles = load_aic_support_profiles("vllm", "0.22.0")

    assert len(profiles) > 100
    assert {profile.aic_system for profile in profiles} >= {"h100_sxm", "l40s"}
    assert all(profile.model.layers > 0 for profile in profiles)
    assert all(profile.gpu.memory_gb and profile.gpu.memory_gb > 0 for profile in profiles)
    qwen_moe = next(
        profile.model for profile in profiles if profile.model.model_id == "Qwen/Qwen3-30B-A3B"
    )
    assert 20e9 < qwen_moe.parameter_count < 40e9
