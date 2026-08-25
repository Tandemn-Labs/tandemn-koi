from copy import deepcopy

import pytest
from src.orchestrator.debug_logging import _surrogate_summary
from src.prediction.backends.aic import AICBackend
from src.prediction.backends.base import Candidate
from src.prediction.composer import SurrogateComposer
from src.prediction.surrogate import (
    SurrogateExecutionError,
    SurrogateMemoryNoFit,
    SurrogateUnsupportedConfig,
)


class _Graph:
    def __init__(self, *, x=("config",), v=("utilization",), y=("throughput",)):
        self.x = x
        self.v = v
        self.y = y


class _CountingSurrogate:
    def __init__(self, metadata=None):
        self.calls = []
        self.last_metadata = {}
        self.metadata = metadata or {}

    def compose_prediction(self, job_config, job_features, candidate_graph, method, scenario):
        self.calls.append((job_config, job_features, candidate_graph, method, scenario))
        self.last_metadata = deepcopy(self.metadata)
        marker = float(job_config.get("marker", 1.0))
        return (
            {
                "throughput_token_per_sec": marker,
                "p99_ttft_ms": 10.0,
                "p99_tpot_ms": 1.0,
            },
            {"gpu_mem_used_fraction": 0.5},
        )


class _RaisingSurrogate:
    def __init__(self, error_type):
        self.calls = 0
        self.last_metadata = {}
        self.error_type = error_type

    def compose_prediction(self, *_args, **_kwargs):
        self.calls += 1
        self.last_metadata = {}
        raise self.error_type("prediction failed")


def test_aic_raw_cache_canonicalizes_reordered_inputs_and_scope():
    surrogate = _CountingSurrogate()
    backend = AICBackend(surrogate)
    first_candidate = Candidate(
        job_config={"marker": 7, "nested": {"b": 2, "a": 1}},
        job_features={"z": 3, "a": {"second": 2, "first": 1}},
        env=("reserved", "aws", "us-east-1", "zone-a", "H100"),
    )
    reordered_candidate = Candidate(
        job_config={"nested": {"a": 1, "b": 2}, "marker": 7},
        job_features={"a": {"first": 1, "second": 2}, "z": 3},
        env=("reserved", "aws", "us-east-1", "zone-a", "H100"),
    )

    miss = backend.estimate(
        first_candidate,
        candidate_graph=_Graph(x=("b", "a"), v=("d", "c"), y=("f", "e")),
        method=["AIC_Direct"],
        scenario="peak",
    )
    hit = backend.estimate(
        reordered_candidate,
        candidate_graph=_Graph(x=("a", "b"), v=("c", "d"), y=("e", "f")),
        method=("AIC_Direct",),
        scenario="peak",
    )

    assert len(surrogate.calls) == 1
    assert miss.metadata["aic_raw_cache"] == {
        "hit": False,
        "key_version": "aic-primary-cache-v1",
        "entries": 1,
        "max_entries": 512,
    }
    assert hit.metadata["aic_raw_cache"]["hit"] is True
    assert "key" not in hit.metadata["aic_raw_cache"]
    assert surrogate.calls[0][3] == ("AIC_Direct",)
    assert backend.cache_info() == {
        "hits": 1,
        "misses": 1,
        "bypasses": 0,
        "evictions": 0,
        "entries": 1,
        "max_entries": 512,
    }


def test_aic_raw_cache_deep_copies_insertions_and_returns():
    surrogate = _CountingSurrogate(metadata={"nested": {"values": [1]}})
    backend = AICBackend(surrogate)
    candidate = Candidate({"marker": 4}, {})
    graph = _Graph()

    first = backend.estimate(candidate, candidate_graph=graph)
    first.y_hat["throughput_token_per_sec"] = -1.0
    first.metadata["nested"]["values"].append(2)
    first.metadata["aic_raw_cache"]["entries"] = 999

    second = backend.estimate(candidate, candidate_graph=graph)
    assert second.y_hat["throughput_token_per_sec"] == 4.0
    assert second.metadata["nested"]["values"] == [1]
    assert second.metadata["aic_raw_cache"]["entries"] == 1

    second.metadata["nested"]["values"].append(3)
    third = backend.estimate(candidate, candidate_graph=graph)
    assert third.metadata["nested"]["values"] == [1]
    assert len(surrogate.calls) == 1


def test_aic_raw_cache_separates_method_scenario_graph_version_and_inputs():
    surrogate = _CountingSurrogate(metadata={"aic_database_mode": "SILICON"})
    backend = AICBackend(surrogate)
    graph = _Graph(x=("x",), v=("v",), y=("y",))
    candidate = Candidate({"marker": 1, "config": "a"}, {"feature": "a"}, env=("env-a",))

    initial = backend.estimate(candidate, candidate_graph=graph)
    backend.estimate(candidate, candidate_graph=graph, method=("AIC_DynoSim",))
    backend.estimate(candidate, candidate_graph=graph, scenario="peak")
    backend.estimate(candidate, candidate_graph=_Graph(x=("other",), v=("v",), y=("y",)))
    backend.estimate(
        Candidate({"marker": 1, "config": "b"}, {"feature": "a"}, env=("env-a",)),
        candidate_graph=graph,
    )
    backend.estimate(
        Candidate({"marker": 1, "config": "a"}, {"feature": "b"}, env=("env-a",)),
        candidate_graph=graph,
    )
    backend.estimate(
        Candidate({"marker": 1, "config": "a"}, {"feature": "a"}, env=("env-b",)),
        candidate_graph=graph,
    )
    backend.version = f"{backend.version}:next"
    version_miss = backend.estimate(candidate, candidate_graph=graph)
    version_hit = backend.estimate(candidate, candidate_graph=graph)

    assert len(surrogate.calls) == 8
    assert initial.version.endswith(":silicon")
    assert version_miss.version == version_hit.version
    assert version_hit.version == f"{backend.version}:silicon"
    assert version_hit.metadata["aic_raw_cache"]["hit"] is True


def test_aic_raw_cache_is_bounded_lru_and_hits_refresh_recency():
    surrogate = _CountingSurrogate()
    backend = AICBackend(surrogate)
    graph = _Graph()

    for marker in range(1, 513):
        backend.estimate(Candidate({"marker": marker}, {}), candidate_graph=graph)

    backend.estimate(Candidate({"marker": 1}, {}), candidate_graph=graph)
    backend.estimate(Candidate({"marker": 513}, {}), candidate_graph=graph)
    refreshed = backend.estimate(Candidate({"marker": 1}, {}), candidate_graph=graph)
    evicted = backend.estimate(Candidate({"marker": 2}, {}), candidate_graph=graph)

    assert refreshed.metadata["aic_raw_cache"]["hit"] is True
    assert evicted.metadata["aic_raw_cache"]["hit"] is False
    assert len(surrogate.calls) == 514
    assert backend.cache_info() == {
        "hits": 2,
        "misses": 514,
        "bypasses": 0,
        "evictions": 2,
        "entries": 512,
        "max_entries": 512,
    }


def test_aic_raw_cache_does_not_store_failures_exceptions_or_degraded_results():
    graph = _Graph()
    candidate = Candidate({"marker": 1}, {})

    for error_type, status in (
        (SurrogateUnsupportedConfig, "unsupported"),
        (SurrogateExecutionError, "failed"),
    ):
        surrogate = _RaisingSurrogate(error_type)
        backend = AICBackend(surrogate)
        first = backend.estimate(candidate, candidate_graph=graph)
        second = backend.estimate(candidate, candidate_graph=graph)
        assert first.status == second.status == status
        assert second.metadata["aic_raw_cache"]["hit"] is False
        assert surrogate.calls == 2
        assert backend.cache_info()["entries"] == 0

    for error_type in (SurrogateMemoryNoFit, RuntimeError):
        surrogate = _RaisingSurrogate(error_type)
        backend = AICBackend(surrogate)
        for _ in range(2):
            with pytest.raises(error_type, match="prediction failed"):
                backend.estimate(candidate, candidate_graph=graph)
        assert surrogate.calls == 2
        assert backend.cache_info()["entries"] == 0

    for metadata in (
        {"aic_fallback": {"reason": "silicon unavailable"}},
        {"model_profile_enrichment": {"status": "unavailable"}},
        {"degraded": True},
    ):
        surrogate = _CountingSurrogate(metadata=metadata)
        backend = AICBackend(surrogate)
        first = backend.estimate(candidate, candidate_graph=graph)
        second = backend.estimate(candidate, candidate_graph=graph)
        assert first.status == second.status == "success"
        assert second.metadata["aic_raw_cache"]["hit"] is False
        assert len(surrogate.calls) == 2
        assert backend.cache_info()["entries"] == 0


def test_aic_raw_cache_accepts_healthy_direct_omissions_but_requires_positive_throughput():
    graph = _Graph()
    healthy = _CountingSurrogate(
        metadata={"aic_fallback_omitted_nodes": ["p99_tpot_ms", "p99_ttft_ms"]}
    )
    backend = AICBackend(healthy)
    candidate = Candidate({"marker": 10}, {})

    assert backend.has_cached(candidate, candidate_graph=graph, method=["AIC_Direct"]) is False
    backend.estimate(candidate, candidate_graph=graph, method="AIC_Direct")
    assert backend.has_cached(candidate, candidate_graph=graph, method=("AIC_Direct",)) is True
    hit = backend.estimate(candidate, candidate_graph=graph)
    assert hit.metadata["aic_raw_cache"]["hit"] is True
    assert len(healthy.calls) == 1

    for marker in (0, -1, "inf"):
        nonpositive = _CountingSurrogate()
        backend = AICBackend(nonpositive)
        candidate = Candidate({"marker": marker}, {})
        first = backend.estimate(candidate, candidate_graph=graph)
        second = backend.estimate(candidate, candidate_graph=graph)
        assert first.metadata["aic_raw_cache"]["hit"] is False
        assert second.metadata["aic_raw_cache"]["hit"] is False
        assert len(nonpositive.calls) == 2
        assert backend.cache_info()["entries"] == 0


def test_aic_raw_cache_bypasses_noncanonical_inputs_and_clear_resets_info():
    surrogate = _CountingSurrogate()
    backend = AICBackend(surrogate)
    graph = _Graph()
    candidate = Candidate({"marker": 2}, {})

    backend.estimate(candidate, candidate_graph=graph)
    backend.estimate(candidate, candidate_graph=graph)
    unserializable = Candidate({"marker": 3, "value": object()}, {})
    nonfinite = Candidate({"marker": 4, "value": float("nan")}, {})
    for bypassed in (unserializable, unserializable, nonfinite, nonfinite):
        result = backend.estimate(bypassed, candidate_graph=graph)
        assert result.metadata["aic_raw_cache"]["hit"] is False

    assert len(surrogate.calls) == 5
    assert backend.cache_info() == {
        "hits": 1,
        "misses": 1,
        "bypasses": 4,
        "evictions": 0,
        "entries": 1,
        "max_entries": 512,
    }

    backend.clear_cache()
    assert backend.cache_info() == {
        "hits": 0,
        "misses": 0,
        "bypasses": 0,
        "evictions": 0,
        "entries": 0,
        "max_entries": 512,
    }
    after_clear = backend.estimate(candidate, candidate_graph=graph)
    assert after_clear.metadata["aic_raw_cache"]["hit"] is False
    assert len(surrogate.calls) == 6


def test_aic_raw_cache_metadata_flows_through_composer_component_trace():
    surrogate = _CountingSurrogate()
    backend = AICBackend(surrogate)
    composer = SurrogateComposer(backend, perfdb_mode="off", peer_mode="off")
    config = {
        "model_id": "model-a",
        "model_params_b": 8,
        "gpu_mem_gb": 80,
        "tp": 1,
        "pp": 1,
        "dp": 1,
    }
    features = {"type": "batch", "isl_token_avg": 128, "osl_token_avg": 64}
    graph = _Graph(
        x=("model_id",),
        v=("gpu_mem_used_fraction",),
        y=("throughput_token_per_sec", "p99_ttft_ms", "p99_tpot_ms"),
    )

    assert composer.primary_cache_contains(config, features, graph) is False
    _, _, miss_trace = composer.compose_prediction_with_trace(config, features, graph)
    assert composer.primary_cache_contains(config, features, graph) is True
    assert composer.primary_cache_contains(config, features, graph, scenario="peak") is False
    _, _, hit_trace = composer.compose_prediction_with_trace(config, features, graph)

    miss_metadata = miss_trace["components"]["primary"]["metadata"]["aic_raw_cache"]
    hit_metadata = hit_trace["components"]["primary"]["metadata"]["aic_raw_cache"]
    assert miss_metadata["hit"] is False
    assert hit_metadata == {
        "hit": True,
        "key_version": "aic-primary-cache-v1",
        "entries": 1,
        "max_entries": 512,
    }
    assert len(surrogate.calls) == 1


def test_compact_surrogate_summary_exposes_only_safe_raw_cache_fields():
    summary = _surrogate_summary(
        {
            "schema_version": 3,
            "normalized_candidate": {"job_config": {"secret": "candidate-secret"}},
            "components": {
                "primary": {
                    "status": "success",
                    "version": "aic-v1",
                    "metadata": {
                        "aic_raw_cache": {
                            "hit": True,
                            "key_version": "aic-primary-cache-v1",
                            "entries": 12,
                            "max_entries": 512,
                            "key": "cache-key-secret",
                            "candidate_inputs": {"secret": "cache-candidate-secret"},
                        },
                        "other": "backend-secret",
                    },
                }
            },
        }
    )

    assert summary["components"]["primary"]["metadata"] == {
        "aic_raw_cache": {
            "hit": True,
            "key_version": "aic-primary-cache-v1",
            "entries": 12,
            "max_entries": 512,
        }
    }
    rendered = repr(summary)
    assert "cache-key-secret" not in rendered
    assert "candidate-secret" not in rendered
    assert "backend-secret" not in rendered
