from src.agent.tools.agent_tools import _generated_tp_options, config_runnable
from src.config.policy import load_config_policy
from src.infra.resource_map import ClusterResourceSnapshot
from src.validation.validator import Validator


def test_policy_resolves_phi_rules_and_global_tp_ceiling():
    policy = load_config_policy()

    assert policy.rule_for("H100", "microsoft/phi-4").allowed_tp == (1, 2)
    assert policy.rule_for("A100_80GB", "microsoft/phi-4").allowed_tp == (1, 2)
    assert policy.rule_for("MI300", "microsoft/phi-4").allowed_tp == (1, 2)

    ok, reason = config_runnable(
        {"model_id": "microsoft/phi-4", "tp": 8, "pp": 1, "gpu_count": 8},
        {"model_id": "microsoft/phi-4"},
        gpu_type="H100",
    )

    assert not ok
    assert "limits TP to 2" in reason


def test_policy_matches_servingsim_quantization_availability():
    policy = load_config_policy()

    expected_h100 = {
        "meta-llama/Llama-3.2-1B-Instruct": "fp16",
        "microsoft/phi-4": "fp16",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "bf16",
        "Qwen/Qwen2.5-Coder-32B-Instruct": "fp16",
    }
    for model_id, precision in expected_h100.items():
        assert policy.rule_for("H100", model_id).precision == precision

    assert policy.rule_for("L40S", "moonshotai/Kimi-K2-Instruct") is None
    assert policy.rule_for("L40S", "deepseek-ai/DeepSeek-V3") is None


def test_policy_accepts_allowed_phi_tp_and_enforces_known_gpu_rule():
    allowed, _ = config_runnable(
        {"model_id": "microsoft/phi-4", "tp": 2, "pp": 1, "gpu_count": 2},
        {"model_id": "microsoft/phi-4"},
        gpu_type="H100",
    )
    disallowed, reason = config_runnable(
        {"model_id": "microsoft/phi-4", "tp": 1, "pp": 1, "gpu_count": 1},
        {"model_id": "microsoft/phi-4"},
        gpu_type="A10G",
    )

    assert allowed
    assert not disallowed
    assert "allows TP [2]" in reason


def test_policy_precision_is_checked_when_catalog_launch_config_supplies_it():
    ok, reason = config_runnable(
        {
            "model_id": "microsoft/phi-4",
            "tp": 2,
            "pp": 1,
            "gpu_count": 2,
            "weight_dtype": "float16",
        },
        {"model_id": "microsoft/phi-4"},
        gpu_type="H100",
    )
    wrong_precision, wrong_reason = config_runnable(
        {
            "model_id": "microsoft/phi-4",
            "tp": 2,
            "pp": 1,
            "gpu_count": 2,
            "weight_dtype": "bfloat16",
        },
        {"model_id": "microsoft/phi-4"},
        gpu_type="H100",
    )

    assert ok
    assert reason == ""
    assert not wrong_precision
    assert "requires fp16 precision" in wrong_reason

    quantized, quantized_reason = config_runnable(
        {
            "model_id": "microsoft/phi-4",
            "tp": 2,
            "pp": 1,
            "gpu_count": 2,
            "weight_dtype": "float16",
            "weight_quantization_method": "fp8",
        },
        {"model_id": "microsoft/phi-4"},
        gpu_type="H100",
    )
    assert not quantized
    assert "requires fp16 precision" in quantized_reason


def test_expert_parallelism_is_fixed_to_one():
    valid, _ = config_runnable(
        {"tp": 4, "pp": 1, "ep": 1, "gpu_count": 4},
        {"is_moe": True, "num_routed_experts": 64},
    )

    assert valid
    for ep in (2, 8, "1", 1.5, True):
        with_ep, reason = config_runnable(
            {"tp": 4, "pp": 1, "ep": ep, "gpu_count": 4},
            {"is_moe": True, "num_routed_experts": 64},
        )
        assert not with_ep
        assert "ep must be exactly 1" in reason


def test_gpu_count_must_equal_engine_parallelism():
    valid, _ = config_runnable({"tp": 2, "pp": 1, "gpu_count": 2})
    oversized, reason = config_runnable({"tp": 2, "pp": 1, "gpu_count": 8})

    assert valid
    assert not oversized
    assert "gpu_count must equal tp*pp=2" in reason


def test_validator_rejects_phi_tp8_from_snapshot_model_identity():
    snapshot = ClusterResourceSnapshot(
        1,
        {},
        [],
        [{"job_id": "phi", "job_features": {"model_id": "microsoft/phi-4"}}],
    )
    plan = {
        "actions": [
            {
                "job_id": "phi",
                "type": "place",
                "ladder": [
                    {
                        "role": "aggregate",
                        "env": ["reserved", "aws", "us-east-1", "zone-a", "H100"],
                        "config": {
                            "instance_type": "p5.48xlarge",
                            "gpu_count": 8,
                            "tp": 8,
                            "pp": 1,
                        },
                        "n_replicas": 1,
                    }
                ],
            }
        ]
    }

    result = Validator().val_plan(plan, snapshot)

    assert not result.feasible
    assert any("C6 policy" in violation for violation in result.violations)


def test_generated_phi_frames_use_policy_tp_values_on_eight_gpu_instance():
    assert _generated_tp_options(
        heads=40,
        gpu_cap=8,
        gpu_type="H100",
        model_id="microsoft/phi-4",
        allocation_kind="instance",
    ) == [1, 2]
