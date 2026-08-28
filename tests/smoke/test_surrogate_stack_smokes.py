import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from src.core.models import RankSpec
from src.prediction.backends.aic import AICBackend
from src.prediction.backends.base import SurrogateEstimate
from src.prediction.calibration import EVIDENCE_SCAN_LIMIT, build_prediction_context
from src.prediction.composer import SurrogateComposer, compact_prediction_lineage
from src.prediction.surrogate import SurrogateUnsupportedConfig


class _NoFitError(Exception):
    pass


class _Primary:
    def __init__(self):
        self.calls = []

    def compose_prediction(self, job_config, job_features, candidate_graph, method, scenario):
        self.calls.append((job_config, job_features, candidate_graph, method, scenario))
        return (
            {
                "throughput_token_per_sec": 100.0,
                "p99_ttft_ms": 10.0,
                "p99_tpot_ms": 1.0,
                "cost_per_token": 0.01,
                "slo_margin": 9.0,
            },
            {"kv_cache_util": 0.0, "gpu_mem_used_fraction": 0.0},
        )


class _UnsupportedBackend:
    name = "primary"

    def provides(self):
        return set()

    def estimate(self, *_args, **_kwargs):
        return SurrogateEstimate(
            status="unsupported",
            version="aic-v1",
            source="primary",
            metadata={"error": "missing AIC performance slice"},
        )


class _Peers:
    def __init__(self):
        self.calls = []

    def predict(self, job_config, job_features, *, scenario):
        self.calls.append((job_config, job_features, scenario))
        return {
            "solver": {
                "status": "success",
                "backend_version": "solver-v1",
                "request_hash": "request-1",
                "task": "capacity",
                "output_tps": 200.0,
            },
            "blis": {
                "status": "success",
                "backend_version": "blis-v1",
                "output_tps": 150.0,
                "ttft_ms_p99": 12.0,
                "tpot_ms_p99": 1.5,
            },
        }


class _PerfDB:
    version = "perfdb-v1"

    def estimate(self, candidate, **kwargs):
        return SurrogateEstimate(
            y_hat={
                "throughput_token_per_sec": 300.0,
                "p99_ttft_ms": 15.0,
                "p99_tpot_ms": 2.0,
            },
            v_hat={"sm_utilization": 0.7},
            status="success",
            version=self.version,
            coverage={
                "throughput_token_per_sec": 1.0,
                "p99_ttft_ms": 1.0,
                "p99_tpot_ms": 1.0,
                "sm_utilization": 1.0,
            },
            spread={"throughput_token_per_sec": 3.0},
            source="perfdb",
        )


class _Evidence:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.limits = []

    def get_all_rows(self, limit=None):
        self.limits.append(limit)
        return list(self.rows[-limit:]) if limit is not None else list(self.rows)


class _Graph:
    v = (
        "gpu_mem_used_fraction",
        "kv_cache_util",
        "vram_headroom_gb",
        "pipeline_bubble_fraction",
        "per_tok_comm_bytes",
        "comm_overhead_pct",
    )
    y = (
        "throughput_token_per_sec",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "cost_per_token",
        "slo_margin",
    )


CONFIG = {
    "model_id": "model-a",
    "gpu_type": "H100",
    "weight_dtype": "bf16",
    "kvcache_dtype": "fp16",
    "engine_name": "vllm",
    "engine_version": "0.22.0",
    "model_params_b": 8,
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attn_heads": 32,
    "num_kv_heads": 8,
    "gpu_mem_gb": 80,
    "gpu_per_node": 8,
    "nvlink_bandwidth_gbps": 900,
    "internode_bandwidth_gbps": 400,
    "activation_dtype": "bf16",
    "tp": 1,
    "pp": 1,
    "dp": 2,
    "max_num_seq": 8,
    "max_num_batched_tokens": 2048,
}
FEATURES = {
    "type": "batch",
    "isl_token_avg": 128,
    "osl_token_avg": 128,
    "target_p99_ttft_ms": 20,
    "target_p99_tpot_ms": 10,
}


def test_aic_backend_forces_direct_method_and_preserves_inputs_scenario_and_failures():
    primary = _Primary()
    backend = AICBackend(primary)
    config = {"dp": 3}
    features = {"type": "batch"}
    graph = object()

    estimate = backend.estimate(
        SimpleNamespace(job_config=config, job_features=features),
        candidate_graph=graph,
        method=("custom",),
        scenario="peak",
    )

    assert primary.calls[0] == (config, features, graph, ("AIC_Direct",), "peak")
    assert primary.calls[0][0] is config
    assert primary.calls[0][1] is features
    assert estimate.metadata["method"] == ["AIC_Direct"]
    assert estimate.metadata["prediction_semantics"] == {
        "basis": "aic_direct_point",
        "throughput_token_per_sec": "point_capacity",
        "p99_ttft_ms": "base_service_latency",
        "p99_tpot_ms": "base_service_latency",
        "slo_margin": "base_service_latency_margin",
        "queue_model": "none",
        "queue_slo_verified": False,
    }

    class _Fail(_Primary):
        def compose_prediction(self, *args, **kwargs):
            raise _NoFitError("does not fit")

    try:
        AICBackend(_Fail()).estimate(
            SimpleNamespace(job_config={}, job_features={}), candidate_graph=graph
        )
    except _NoFitError as exc:
        assert str(exc) == "does not fit"
    else:
        raise AssertionError("structured primary failure was swallowed")

    class _Unsupported(_Primary):
        def compose_prediction(self, *args, **kwargs):
            self.last_metadata = {
                "compatibility": {"gpu": {"requested": "MI300", "kind": "unsupported"}}
            }
            raise SurrogateUnsupportedConfig("no AIC estimate")

    unsupported = AICBackend(_Unsupported()).estimate(
        SimpleNamespace(job_config={}, job_features={}), candidate_graph=graph
    )
    assert unsupported.status == "unsupported"
    assert unsupported.metadata["error"] == "no AIC estimate"
    assert unsupported.metadata["compatibility"]["gpu"]["requested"] == "MI300"
    assert unsupported.metadata["prediction_semantics"]["queue_slo_verified"] is False


def test_aic_backend_serializes_stateful_compatibility_metadata():
    class StatefulPrimary:
        def __init__(self):
            self.last_metadata = {}

        def compose_prediction(self, job_config, **_kwargs):
            marker = job_config["gpu_type"]
            self.last_metadata = {
                "compatibility": {
                    "gpu": {
                        "requested": marker,
                        "resolved": marker,
                        "kind": "exact",
                        "confidence": 1.0,
                    }
                }
            }
            time.sleep(0.01)
            return {"throughput_token_per_sec": 1.0}, {}

    backend = AICBackend(StatefulPrimary())

    def estimate(marker):
        result = backend.estimate(
            SimpleNamespace(job_config={"gpu_type": marker}, job_features={}),
            candidate_graph=object(),
        )
        return result.metadata["compatibility"]["gpu"]["requested"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(estimate, ("A10G", "L4", "H100", "L40S")))

    assert results == ["A10G", "L4", "H100", "L40S"]


def test_composer_forces_direct_primary_trace_and_preserves_inputs_scenario():
    primary = _Primary()
    peers = _Peers()
    composer = SurrogateComposer(primary, peer_client=peers, peer_mode="shadow")

    y_hat, v_hat, trace = composer.compose_prediction_with_trace(
        {**CONFIG, "pp": 4},
        FEATURES,
        _Graph(),
        method=("AIC_DynoSim",),
        scenario="peak",
    )

    assert primary.calls[0][0]["dp"] == 2
    assert primary.calls[0][3] == ("AIC_Direct",)
    assert primary.calls[0][4] == "peak"
    assert peers.calls[0][0]["dp"] == 2
    assert peers.calls[0][2] == "peak"
    assert y_hat["throughput_token_per_sec"] == 100.0
    assert y_hat["p99_tpot_ms"] == 1.0
    assert "throughput_tokens_per_sec" not in str(trace)
    assert v_hat["gpu_mem_used_fraction"] != 0.0
    assert v_hat["kv_cache_util"] != 0.0
    assert v_hat["pipeline_bubble_fraction"] > 0
    assert trace["schema_version"] == 3
    assert trace["method"] == ["AIC_Direct"]
    assert trace["components"]["primary"]["status"] == "success"
    assert trace["backends"]["solver"]["y_hat"]["throughput_token_per_sec"] == 200.0
    assert trace["fusion"]["applied"] is False


def test_perfdb_shadow_records_coverage_spread_without_changing_primary():
    composer = SurrogateComposer(
        _Primary(),
        perfdb_backend=_PerfDB(),
        perfdb_mode="shadow",
        peer_mode="off",
    )
    y_hat, _, trace = composer.compose_prediction_with_trace(CONFIG, FEATURES, _Graph())
    assert y_hat["throughput_token_per_sec"] == 100.0
    assert trace["components"]["perfdb"]["status"] == "success"
    assert trace["components"]["perfdb"]["spread"]["throughput_token_per_sec"] == 3.0


def test_perfdb_enabled_rederives_cost_and_slo_from_changed_throughput_and_latency():
    composer = SurrogateComposer(
        _Primary(),
        perfdb_backend=_PerfDB(),
        perfdb_mode="enabled",
        peer_mode="off",
    )
    y_hat, _, _ = composer.compose_prediction_with_trace(CONFIG, FEATURES, _Graph())
    assert y_hat["throughput_token_per_sec"] == 300.0
    assert y_hat["cost_per_token"] == 0.01 * 100.0 / 300.0
    assert y_hat["slo_margin"] == min((20.0 - 15.0) / 20.0, (10.0 - 2.0) / 10.0)


def test_primary_failure_short_circuits_and_records_compact_failure():
    class _Fail(_Primary):
        def compose_prediction(self, *args, **kwargs):
            raise _NoFitError("does not fit")

    peers = _Peers()
    composer = SurrogateComposer(_Fail(), peer_client=peers)
    try:
        composer.compose_prediction(CONFIG, FEATURES, _Graph(), scenario="peak")
    except _NoFitError:
        pass
    else:
        raise AssertionError("primary failure did not propagate")

    assert peers.calls == []
    assert composer.last_trace["failure"] == {
        "stage": "primary",
        "error_type": "_NoFitError",
        "message": "does not fit",
    }


def test_unsupported_primary_uses_roofline_peer_best_effort_in_shadow_mode():
    for gpu_type in (
        "nvidia-a10g",
        "nvidia-RTXPRO6000",
        "nvidia-L4",
        "H100",
        "unexpected-future-gpu",
    ):
        peers = _Peers()
        composer = SurrogateComposer(
            _UnsupportedBackend(),
            peer_client=peers,
            peer_mode="shadow",
            perfdb_mode="off",
        )

        y_hat, _, trace = composer.compose_prediction_with_trace(
            {**CONFIG, "gpu_type": gpu_type}, FEATURES, _Graph()
        )

        assert y_hat["throughput_token_per_sec"] == 200.0
        assert y_hat["p99_ttft_ms"] == 12.0
        assert y_hat["p99_tpot_ms"] == 1.5
        assert trace["components"]["fallback"]["status"] == "success"
        assert trace["metadata"]["fallback_sources"] == ["solver", "blis"]


def test_unsupported_primary_uses_perfdb_best_effort_in_shadow_mode():
    composer = SurrogateComposer(
        _UnsupportedBackend(),
        perfdb_backend=_PerfDB(),
        perfdb_mode="shadow",
        peer_mode="off",
    )

    y_hat, v_hat, trace = composer.compose_prediction_with_trace(CONFIG, FEATURES, _Graph())

    assert y_hat["throughput_token_per_sec"] == 300.0
    assert y_hat["p99_ttft_ms"] == 15.0
    assert v_hat["sm_utilization"] == 0.7
    assert trace["metadata"]["fallback_sources"] == ["perfdb"]


def test_incomplete_successful_primary_uses_available_fallback_outputs():
    class IncompletePrimary:
        name = "primary"

        @staticmethod
        def provides():
            return {"throughput_token_per_sec"}

        @staticmethod
        def estimate(*_args, **_kwargs):
            return SurrogateEstimate(
                y_hat={"throughput_token_per_sec": 100.0},
                status="success",
                version="aic-v1",
                source="primary",
            )

    composer = SurrogateComposer(
        IncompletePrimary(),
        perfdb_backend=_PerfDB(),
        perfdb_mode="shadow",
        peer_mode="off",
    )

    y_hat, _, trace = composer.compose_prediction_with_trace(CONFIG, FEATURES, _Graph())

    assert y_hat["throughput_token_per_sec"] == 100.0
    assert y_hat["p99_ttft_ms"] == 15.0
    assert y_hat["p99_tpot_ms"] == 2.0
    assert trace["components"]["fallback"]["status"] == "success"
    assert trace["metadata"]["fallback_sources"] == ["perfdb"]
    assert trace["prediction_semantics"]["basis"] == "composed_point_estimate"
    assert trace["prediction_semantics"]["queue_slo_verified"] is False


def test_unsupported_primary_ignores_very_low_coverage_perfdb_values():
    class LowCoveragePerfDB(_PerfDB):
        def estimate(self, candidate, **kwargs):
            estimate = super().estimate(candidate, **kwargs)
            estimate.coverage = dict.fromkeys(estimate.coverage, 0.05)
            return estimate

    composer = SurrogateComposer(
        _UnsupportedBackend(),
        perfdb_backend=LowCoveragePerfDB(),
        perfdb_mode="shadow",
        peer_mode="off",
    )

    y_hat, _, trace = composer.compose_prediction_with_trace(CONFIG, FEATURES, _Graph())

    assert y_hat == {}
    assert trace["components"]["fallback"]["status"] == "partial"


def test_unsupported_primary_returns_partial_analytic_result_when_others_are_off():
    composer = SurrogateComposer(
        _UnsupportedBackend(),
        perfdb_mode="off",
        peer_mode="off",
    )

    y_hat, v_hat, trace = composer.compose_prediction_with_trace(CONFIG, FEATURES, _Graph())

    assert y_hat == {}
    assert v_hat["gpu_mem_used_fraction"] > 0
    assert trace["components"]["fallback"]["status"] == "partial"
    assert trace["components"]["fallback"]["metadata"]["missing_core_y"] == [
        "p99_tpot_ms",
        "p99_ttft_ms",
        "throughput_token_per_sec",
    ]


def test_learned_fusion_trace_skips_residual_throughput_calibration():
    evidence = _Evidence()
    peers = _Peers()
    composer = SurrogateComposer(
        _Primary(),
        peer_client=peers,
        peer_mode="enabled",
        evidence_store=evidence,
    )
    _, _, cold_trace = composer.compose_prediction_with_trace(
        CONFIG,
        FEATURES,
        _Graph(),
        scenario="peak",
        as_of_timestamp_utc=50.0,
    )
    assert cold_trace["normalized_candidate"]["job_config"]
    assert cold_trace["normalized_candidate"]["job_features"]
    context = build_prediction_context(CONFIG, FEATURES, scenario="peak")
    primary_version = composer.primary.version
    rows = []
    for index in range(6):
        observed = 150.0 + index
        rows.append(
            SimpleNamespace(
                row_id=f"row-{index}",
                deployment_id=f"deploy-{index}",
                evidence_available_timestamp_utc=40.0,
                y_observed_mean={"throughput_token_per_sec": observed},
                y_observed_trajectory={},
                V_observed_trajectory={},
                prediction_lineage={
                    "schema_version": 3,
                    "composite_version": cold_trace["composite_version"],
                    "context": context,
                    "backends": {
                        "primary": {
                            "version": primary_version,
                            "y_hat": {"throughput_token_per_sec": 100.0 + index},
                        },
                        "solver": {
                            "version": "solver-v1",
                            "y_hat": {"throughput_token_per_sec": 200.0 + index},
                        },
                        "blis": {
                            "version": "blis-v1",
                            "y_hat": {"throughput_token_per_sec": 150.0 + index},
                        },
                    },
                    "pre_calibration": {
                        "y_hat": {"throughput_token_per_sec": 25.0},
                        "v_hat": {},
                    },
                },
            )
        )
    evidence.rows = rows
    evidence.limits.clear()

    y_hat, _, trace = composer.compose_prediction_with_trace(
        CONFIG,
        FEATURES,
        _Graph(),
        scenario="peak",
        as_of_timestamp_utc=100.0,
    )

    assert trace["fusion"]["status"] == "learned"
    assert evidence.limits == [EVIDENCE_SCAN_LIMIT]
    assert trace["fusion"]["applied"] is True
    assert trace["fusion"]["lower_quantile"] == 0.05
    assert trace["fusion"]["lower_throughput"] <= y_hat["throughput_token_per_sec"]
    assert y_hat["cost_per_token"] == 0.01 * 100.0 / y_hat["throughput_token_per_sec"]
    assert trace["calibration"]["skipped_y_nodes"] == ["throughput_token_per_sec"]
    assert "throughput_token_per_sec" not in trace["calibration"]["offsets_y"]
    assert trace["timings_ms"]["total"] >= 0


def test_stress_scenario_does_not_run_peers():
    primary = _Primary()
    peers = _Peers()
    composer = SurrogateComposer(primary, peer_client=peers, peer_mode="enabled")
    _, _, trace = composer.compose_prediction_with_trace(
        CONFIG,
        FEATURES,
        _Graph(),
        method=("AIC_DynoSim",),
        scenario="peak_all_multiturn_stress",
    )
    assert peers.calls == []
    assert primary.calls[0][3] == ("AIC_Direct",)
    assert trace["method"] == ["AIC_Direct"]


def test_peer_failure_warns_once_and_falls_back():
    class MissingPeers:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("optional tandemn-predictors package is unavailable")

    composer = SurrogateComposer(_Primary(), peer_client=MissingPeers(), peer_mode="shadow")
    composer.compose_prediction_with_trace(CONFIG, FEATURES, _Graph())
    _, _, trace = composer.compose_prediction_with_trace(CONFIG, FEATURES, _Graph())

    assert len(composer._peer_warnings) == 1
    assert trace["backends"]["peer_client"]["status"] == "failed"


def test_rank_lineage_round_trips_without_entering_config_x():
    rank = RankSpec.from_dict(
        {
            "role": "aggregate",
            "env": ["reserved", "aws", "us-east-1", "zone-a", "H100"],
            "config": {"tp": 1, "pp": 1},
            "prediction_lineage": {"schema_version": 3, "deployment_id": "deploy-a"},
        }
    )
    assert rank.prediction_lineage["deployment_id"] == "deploy-a"
    assert "prediction_lineage" not in rank.config
    assert rank.to_dict()["prediction_lineage"]["schema_version"] == 3


def test_compact_prediction_lineage_excludes_debug_payloads():
    trace = {
        "schema_version": 3,
        "context": {"hard": {"model_id": "model"}},
        "scenario": "peak",
        "prediction_semantics": {
            "basis": "aic_direct_point",
            "queue_slo_verified": False,
        },
        "compatibility": {
            "primary": {
                "gpu": {
                    "requested": "A10G",
                    "resolved": "A30",
                    "kind": "nearest",
                }
            }
        },
        "backends": {
            "primary": {
                "status": "success",
                "version": "aic-v1",
                "y_hat": {"throughput_token_per_sec": 10.0},
                "metadata": {
                    "large": "debug-only",
                    "error_type": "ExampleError",
                    "error": "useful failure",
                },
            }
        },
        "pre_calibration": {"y_hat": {"throughput_token_per_sec": 10.0}},
        "composite_version": "v1",
        "timings_ms": {"total": 1.0},
        "components": {"primary": {"metadata": {"large": "debug-only"}}},
        "normalized_candidate": {"job_config": {"secret": True}},
    }

    lineage = compact_prediction_lineage(trace)

    assert lineage["backends"]["primary"]["version"] == "aic-v1"
    assert lineage["compatibility"]["primary"]["gpu"]["resolved"] == "A30"
    assert lineage["prediction_semantics"]["queue_slo_verified"] is False
    assert lineage["backends"]["primary"]["diagnostics"] == {
        "error_type": "ExampleError",
        "error": "useful failure",
    }
    assert "metadata" not in lineage["backends"]["primary"]
    assert "timings_ms" not in lineage
    assert "components" not in lineage
    assert "normalized_candidate" not in lineage
