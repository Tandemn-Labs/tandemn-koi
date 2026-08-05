from types import SimpleNamespace

import pytest
from src.prediction.calibration import (
    build_prediction_context,
    calibrate_prediction,
    learn_throughput_fusion,
)

VERSION = "koi-surrogate-v3:test"
CONFIG = {
    "dp": 2,
    "effective_batch_size": 8,
    "engine_name": "vllm",
    "engine_version": "0.22.0",
    "max_num_seq": 8,
    "pp": 1,
    "tp": 1,
}
FEATURES = {
    "gpu_type": "H100",
    "isl_token_avg": 128,
    "model_id": "model-a",
    "osl_token_avg": 64,
    "type": "batch",
    "weight_dtype": "bf16",
}


class _Store:
    def __init__(self, rows):
        self.rows = list(rows)

    def get_all_rows(self, limit=None):
        return list(self.rows)


def _calibration_row(
    index,
    *,
    available=90.0,
    deployment_id=None,
    version=VERSION,
    scenario="peak",
    dp=2,
):
    config = {**CONFIG, "dp": dp}
    context = build_prediction_context(config, FEATURES, scenario=scenario)
    return SimpleNamespace(
        row_id=f"row-{index}",
        deployment_id=deployment_id or f"deploy-{index}",
        evidence_available_timestamp_utc=available,
        y_observed_mean={
            "throughput_token_per_sec": 120.0,
            "p99_ttft_ms": 90.0,
            "cost_per_token": 99.0,
            "slo_margin": -99.0,
        },
        y_observed_trajectory={},
        V_observed_trajectory={"sm_utilization": [0.6]},
        prediction_lineage={
            "schema_version": 3,
            "composite_version": version,
            "context": context,
            "pre_calibration": {
                "y_hat": {
                    "throughput_token_per_sec": 100.0,
                    "p99_ttft_ms": 100.0,
                    "cost_per_token": 0.01,
                    "slo_margin": 1.0,
                },
                "v_hat": {"sm_utilization": 0.5},
            },
        },
    )


def test_residual_calibration_y_v_and_never_cost_or_slo():
    result = calibrate_prediction(
        {
            "throughput_token_per_sec": 100.0,
            "p99_ttft_ms": 100.0,
            "cost_per_token": 0.01,
            "slo_margin": 1.0,
        },
        {"sm_utilization": 0.5},
        CONFIG,
        FEATURES,
        _Store([_calibration_row(index) for index in range(5)]),
        VERSION,
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )

    assert result.status == "learned"
    assert result.y_hat["throughput_token_per_sec"] == pytest.approx(120.0)
    assert result.y_hat["p99_ttft_ms"] == pytest.approx(90.0)
    assert result.v_hat["sm_utilization"] == pytest.approx(0.6)
    assert result.y_hat["cost_per_token"] == 0.01
    assert result.y_hat["slo_margin"] == 1.0
    assert "cost_per_token" not in result.offsets_y
    assert "slo_margin" not in result.offsets_y


def test_calibration_blocks_future_version_context_scenario_dp_and_same_deployment():
    incompatible = [
        _calibration_row(0, available=100.0),
        _calibration_row(1, version="other"),
        _calibration_row(2, scenario="mean"),
        _calibration_row(3, dp=1),
    ]
    repeated = [_calibration_row(index + 4, deployment_id="same-deployment") for index in range(8)]
    result = calibrate_prediction(
        {"throughput_token_per_sec": 100.0},
        {},
        CONFIG,
        FEATURES,
        _Store([*incompatible, *repeated]),
        VERSION,
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )
    assert result.status == "insufficient_evidence"
    assert result.y_hat["throughput_token_per_sec"] == 100.0

    missing_cutoff = calibrate_prediction(
        {"throughput_token_per_sec": 100.0},
        {},
        CONFIG,
        FEATURES,
        _Store([_calibration_row(index) for index in range(5)]),
        VERSION,
        scenario="peak",
        as_of_timestamp_utc=None,
    )
    assert missing_cutoff.status == "disabled"

    missing_version = calibrate_prediction(
        {"throughput_token_per_sec": 100.0},
        {},
        CONFIG,
        FEATURES,
        _Store([_calibration_row(index) for index in range(5)]),
        None,
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )
    assert missing_version.status == "disabled"

    incomplete_context = calibrate_prediction(
        {"throughput_token_per_sec": 100.0},
        {},
        {key: value for key, value in CONFIG.items() if key != "engine_version"},
        FEATURES,
        _Store([_calibration_row(index) for index in range(5)]),
        VERSION,
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )
    assert incomplete_context.status == "disabled"


def _fusion_row(index, *, deployment_id=None, available=90.0, solver_version="solver-v1"):
    observed = 150.0 + index
    return SimpleNamespace(
        row_id=f"fusion-{index}",
        deployment_id=deployment_id or f"fusion-deploy-{index}",
        evidence_available_timestamp_utc=available,
        y_observed_mean={"throughput_token_per_sec": observed},
        y_observed_trajectory={},
        prediction_lineage={
            "schema_version": 3,
            "context": build_prediction_context(CONFIG, FEATURES, scenario="peak"),
            "backends": {
                "primary": {
                    "version": "aic-v1",
                    "y_hat": {"throughput_token_per_sec": 100.0 + index},
                },
                "solver": {
                    "version": solver_version,
                    "y_hat": {"throughput_token_per_sec": 200.0 + index},
                },
            },
        },
    )


def test_fusion_requires_five_deployments_versions_and_cross_fit_lower_bound():
    learned = learn_throughput_fusion(
        {"primary": 110.0, "solver": 210.0},
        {"primary": "aic-v1", "solver": "solver-v1"},
        CONFIG,
        FEATURES,
        _Store([_fusion_row(index) for index in range(6)]),
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )
    assert learned.status == "learned"
    assert learned.sample_count == 6
    assert 110.0 < learned.throughput < 210.0
    assert learned.lower_throughput is not None
    assert learned.lower_throughput <= learned.throughput
    assert sum(learned.weights.values()) == pytest.approx(1.0)

    median_lower = learn_throughput_fusion(
        {"primary": 110.0, "solver": 210.0},
        {"primary": "aic-v1", "solver": "solver-v1"},
        CONFIG,
        FEATURES,
        _Store([_fusion_row(index) for index in range(6)]),
        scenario="peak",
        as_of_timestamp_utc=100.0,
        lower_quantile=0.5,
    )
    assert median_lower.lower_throughput >= learned.lower_throughput

    with pytest.raises(ValueError, match="lower_quantile"):
        learn_throughput_fusion(
            {"primary": 110.0, "solver": 210.0},
            {"primary": "aic-v1", "solver": "solver-v1"},
            CONFIG,
            FEATURES,
            _Store([]),
            scenario="peak",
            as_of_timestamp_utc=100.0,
            lower_quantile=0.75,
        )

    one_deployment = learn_throughput_fusion(
        {"primary": 110.0, "solver": 210.0},
        {"primary": "aic-v1", "solver": "solver-v1"},
        CONFIG,
        FEATURES,
        _Store([_fusion_row(index, deployment_id="same") for index in range(8)]),
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )
    assert one_deployment.status == "insufficient_evidence"
    assert one_deployment.sample_count == 1

    wrong_version = learn_throughput_fusion(
        {"primary": 110.0, "solver": 210.0},
        {"primary": "aic-v1", "solver": "solver-v2"},
        CONFIG,
        FEATURES,
        _Store([_fusion_row(index) for index in range(6)]),
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )
    assert wrong_version.sample_count == 0

    future = learn_throughput_fusion(
        {"primary": 110.0, "solver": 210.0},
        {"primary": "aic-v1", "solver": "solver-v1"},
        CONFIG,
        FEATURES,
        _Store([_fusion_row(index, available=100.0) for index in range(6)]),
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )
    assert future.sample_count == 0
