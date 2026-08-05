from types import SimpleNamespace

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

    client = PeerPredictorClient(("solver", "blis"), predict_fn=fake_predict)
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
