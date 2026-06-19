import contextlib
import io
import math
import unittest

from src.prediction.surrogate import SurrogatePrediction
from src.prediction.tchebycheff import (
    DEFAULT_MAXIMIZE,
    compute_tchebycheff,
    compute_tchebycheff_dro,
)


class PredictionSmokeTests(unittest.TestCase):
    def test_tchebycheff_and_dro_scores_are_finite(self):
        y_hat = {"throughput_tokens_per_sec": 1000, "slo_margin": 100}
        w_t = {"throughput_tokens_per_sec": 0.5, "slo_margin": 0.5}
        z_star_t = {"throughput_tokens_per_sec": 1000, "slo_margin": 100}
        normalization_range = {"throughput_tokens_per_sec": 1000, "slo_margin": 100}
        dro_band = {
            "throughput_tokens_per_sec": {"upper": 1100, "lower": 900},
            "slo_margin": {"upper": 110, "lower": 90},
        }

        j = compute_tchebycheff(y_hat, w_t, z_star_t, normalization_range, 1e-3, DEFAULT_MAXIMIZE)
        j_dro = compute_tchebycheff_dro(
            y_hat, dro_band, w_t, z_star_t, normalization_range, 1e-3, DEFAULT_MAXIMIZE
        )

        self.assertTrue(math.isfinite(j))
        self.assertTrue(math.isfinite(j_dro))
        self.assertEqual(j, -0.0)
        self.assertLess(j_dro, j)

    def test_surrogate_compose_prediction_runs_dynosim(self):
        class Graph:
            x = (
                "gpu_type",
                "engine_name",
                "engine_version",
                "tp",
                "ep",
                "block_size",
                "max_num_seq",
                "max_num_batched_tokens",
                "prefix_cache_enabled",
                "chunked_prefill_enable",
                "pd_enabled",
                "prefill_worker_count",
                "decode_worker_count",
                "isl_token_avg",
                "osl_token_avg",
                "request_arrival_rate",
                "workload_prefix_concentration",
                "is_session_affinity",
            )
            v = ("kv_cache_util", "kv_pressure_score", "pd_inbalance")
            y = (
                "cost_per_token",
                "p99_ttft_ms",
                "p99_tpot_ms",
                "throughput_tokens_per_sec",
                "slo_margin",
            )

        predictor = SurrogatePrediction(objective="batched")
        direct_x, _derive_x, _direct_v, _derive_v, _direct_y, _derive_y = (
            predictor.resolve_prediction_scope(Graph(), "AIC_DynoSim")
        )
        job_config = {
            "model_id": "nvidia/Llama-3.1-8B-Instruct-FP8",
            "engine_name": "vllm",
            "engine_version": "0.19.0",
            "tp": 1,
            "ep": 1,
            "block_size": 64,
            "max_num_seq": 256,
            "max_num_batched_tokens": 8192,
            "prefix_cache_enabled": True,
            "chunked_prefill_enable": True,
            "pd_enabled": False,
            "prefill_worker_count": 1,
            "decode_worker_count": 1,
        }
        job_features = {
            "cloud": "aws",
            "region": "us-east-1",
            "market": "reserved",
            "zone": "use1-az1",
            "gpu_type": "H200",
            "instance_type": "p5e.48xlarge",
            "num_nodes_per_chain": 1,
            "interconnect_type": "nvlink",
            "isl_token_avg": 4000,
            "osl_token_avg": 500,
            "request_arrival_rate": 100,
            "workload_prefix_concentration": 0.20,
            "target_p99_ttft_ms": 200,
            "target_p99_tpot_ms": 10,
        }

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            y_hat, v_hat = predictor.compose_prediction(
                job_config=job_config,
                job_features=job_features,
                candidate_graph=Graph(),
                method=("AIC_DynoSim",),
            )

        self.assertIn("p99_ttft_ms", y_hat)
        self.assertIn("p99_tpot_ms", y_hat)
        self.assertIn("throughput_tokens_per_sec", y_hat)
        self.assertIn("cost_per_token", y_hat)
        self.assertIn("slo_margin", y_hat)
        self.assertIn("input_length_observed", v_hat)
        self.assertIn("output_length_observed", v_hat)
        self.assertIn("kv_pressure_score", v_hat)
        self.assertGreater(len(direct_x), 0)


if __name__ == "__main__":
    unittest.main()
