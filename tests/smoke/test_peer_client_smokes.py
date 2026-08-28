import sys
from types import SimpleNamespace

import pytest
from src.prediction.peer_client import PeerPredictorClient


def test_peer_client_preserves_dp_and_scenario_in_shared_query():
    captured = {}

    def fake_predict(query, predictors):
        captured["query"] = query
        captured["predictors"] = predictors
        return {
            "blis": SimpleNamespace(
                to_dict=lambda: {
                    "status": "success",
                    "backend_version": "blis-v1",
                    "output_tps": 1000.0,
                }
            )
        }

    client = PeerPredictorClient(
        ("solver", "blis"),
        predict_fn=fake_predict,
        query_cls=SimpleNamespace,
    )
    result = client.predict(
        {
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "gpu_type": "H100",
            "instance_type": "p5.48xlarge",
            "tp": 2,
            "pp": 1,
            "dp": 4,
            "max_num_seq": 32,
            "max_num_batched_tokens": 8192,
            "weight_dtype": "bf16",
        },
        {
            "type": "online",
            "isl_token_avg": 1024,
            "osl_token_avg": 256,
            "request_arrival_rate": 8.0,
        },
        scenario="peak",
    )

    query = captured["query"]
    assert captured["predictors"] == ("solver", "blis")
    assert query.num_replicas == 4
    assert query.task == "online"
    assert query.context["scenario"] == "peak"
    assert result["blis"]["output_tps"] == 1000.0


def test_peer_client_reports_missing_optional_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "predictor_compare", None)

    with pytest.raises(RuntimeError, match="optional tandemn-predictors"):
        PeerPredictorClient._query(
            {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "gpu_type": "H100"},
            {"type": "online", "isl_token_avg": 128, "osl_token_avg": 32},
            scenario="peak",
        )


def test_peer_client_resolves_unexpected_gpu_and_model_to_roofline(monkeypatch):
    class Query:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "predictor_compare", SimpleNamespace(Query=Query))
    common = {
        "model_id": "acme/unknown-12b",
        "model_params_b": 12,
        "type": "batch",
        "isl_token_avg": 512,
        "osl_token_avg": 170,
        "max_num_seq": 3,
        "max_num_batched_tokens": 2048,
        "weight_dtype": "bf16",
        "tp": 2,
    }

    a10g = PeerPredictorClient._query({**common, "gpu_type": "nvidia-a10g"}, {}, scenario="peak")
    l4 = PeerPredictorClient._query({**common, "gpu_type": "nvidia-L4"}, {}, scenario="peak")
    rtx = PeerPredictorClient._query(
        {**common, "gpu_type": "nvidia-RTXPRO6000"}, {}, scenario="peak"
    )
    future = PeerPredictorClient._query(
        {
            **common,
            "gpu_type": "unexpected-future-gpu",
            "gpu_mem_gb": 48,
            "gpu_bandwidth_gbps": 432,
            "gpu_tflops_fp16": 181,
            "gpu_generation": "ada",
        },
        {},
        scenario="peak",
    )

    assert a10g.device == "A10G"
    assert a10g.solver_instance_family == "g5.48xlarge"
    assert l4.device == "L4"
    assert l4.solver_instance_family == "g6.48xlarge"
    assert rtx.device == "L40S"
    assert rtx.solver_instance_family == "g6e.48xlarge"
    assert rtx.context["compatibility"]["gpu"]["requested"] == "nvidia-RTXPRO6000"
    assert rtx.context["model_approximation"]["resolved_peer_model"] == "llama3-8b"
    assert rtx.context["model_approximation"]["throughput_scale"] == pytest.approx(8 / 12)
    assert future.device == "L4"
    assert future.context["compatibility"]["gpu"]["kind"] == "nearest"
    assert future.context["compatibility"]["gpu"]["throughput_scale"] == 1.0


def test_peer_client_retries_nearest_supported_roofline_config(monkeypatch):
    class Query:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "predictor_compare", SimpleNamespace(Query=Query))
    attempts = []

    def fake_predict(query, predictors):
        attempts.append((query.task, query.num_replicas, query.tp, query.pp, predictors))
        success = query.task == "capacity" and query.num_replicas == 1 and query.pp == 1
        return {
            "solver": SimpleNamespace(
                to_dict=lambda: {
                    "status": "success" if success else "unsupported",
                    "output_tps": 10.0 if success else None,
                    "ttft_ms_p99": 10.0 if success else None,
                    "backend_version": "solver-v1",
                }
            )
        }

    result = PeerPredictorClient(("solver",), predict_fn=fake_predict).predict(
        {
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "gpu_type": "unexpected-future-gpu",
            "tp": 1,
            "pp": 2,
            "dp": 3,
            "max_num_seq": 4,
            "max_num_batched_tokens": 2048,
            "weight_dtype": "fp16",
            "gpu_mem_gb": 48,
            "gpu_bandwidth_gbps": 432,
            "gpu_tflops_fp16": 181,
            "gpu_generation": "ada",
        },
        {
            "type": "online",
            "isl_token_avg": 512,
            "osl_token_avg": 170,
            "request_arrival_rate": 2.0,
        },
        scenario="peak",
    )["solver"]

    assert result["status"] == "success"
    assert result["output_tps"] == 30.0
    assert result.get("ttft_ms_p99") is None
    assert result["approximation"]["requested"]["pp"] == 2
    assert result["approximation"]["resolved"]["pp"] == 1
    assert attempts[-1][-1] == ("solver",)


def test_peer_client_rejects_proxy_model_that_cannot_fit_requested_gpu(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "predictor_compare",
        SimpleNamespace(Query=lambda **kwargs: SimpleNamespace(**kwargs)),
    )

    query = PeerPredictorClient._query(
        {
            "model_id": "acme/unknown-70b",
            "model_params_b": 70,
            "gpu_type": "A10G",
            "gpu_mem_gb": 24,
            "gpu_mem_util": 0.9,
            "weight_dtype": "bf16",
            "tp": 1,
            "pp": 1,
        },
        {"type": "batch", "isl_token_avg": 512, "osl_token_avg": 170},
        scenario="peak",
    )

    assert query is None
