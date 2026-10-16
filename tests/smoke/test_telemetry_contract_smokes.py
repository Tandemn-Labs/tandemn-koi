"""Record the legacy telemetry contract before adding the federated path."""

import unittest

import numpy as np
from src.core.causal_graph_store_adapter import (
    DEFAULT_EDGE_TABLE_PATH,
    parse_seed_candidate_graph,
)
from src.infra.telemetry import RankTelemetry, StoreTelemetry


class TelemetryContractSmokeTests(unittest.TestCase):
    def test_default_graph_has_the_current_observation_names(self):
        graph = parse_seed_candidate_graph(DEFAULT_EDGE_TABLE_PATH)

        self.assertEqual(
            set(graph.v),
            {
                "gpu_mem_used_fraction",
                "kv_cache_util",
                "activation_mem_pressure",
                "vram_headroom_gb",
                "live_batch_size",
                "depth_req_q",
                "input_length_observed",
                "output_length_observed",
                "sm_utilization",
                "mem_bandwidth_utilization",
                "nvlink_tput_observed",
                "pcie_tput_observed",
                "kvcache_hit_rate",
                "prefill_iteration_counts_per_second",
                "decode_itr_counts_per_second",
                "pd_inbalance",
                "expert_inbalance",
                "comm_overhead_pct",
                "pipeline_bubble_fraction",
                "per_tok_comm_bytes",
                "kv_pressure_score",
                "dispatch_overhead_ms",
            },
        )
        self.assertEqual(
            set(graph.y),
            {
                "cost_per_token",
                "p99_ttft_ms",
                "p99_tpot_ms",
                "throughput_token_per_sec",
                "slo_margin",
            },
        )

    def test_rank_output_keeps_numeric_series_and_predictions_separate(self):
        rank = RankTelemetry(
            job_id="job_1",
            rank_id="rank_1",
            v_observed={"depth_req_q": np.array([0.0, 2.0])},
            v_predicted={"depth_req_q": 1.0},
            y_observed={"throughput_token_per_sec": np.array([0.0, 20.0])},
            y_predicted={"throughput_token_per_sec": 10.0},
        )

        self.assertEqual(rank.job_id, "job_1")
        self.assertEqual(rank.rank_id, "rank_1")
        self.assertIsInstance(rank.v_observed, dict)
        self.assertIsInstance(rank.y_observed, dict)
        np.testing.assert_array_equal(rank.v_observed["depth_req_q"], [0.0, 2.0])
        np.testing.assert_array_equal(rank.y_observed["throughput_token_per_sec"], [0.0, 20.0])
        self.assertNotIn("kv_pressure_score", rank.v_observed)
        self.assertNotIn("cost_per_token", rank.y_observed)
        self.assertEqual(rank.v_predicted, {"depth_req_q": 1.0})
        self.assertEqual(rank.y_predicted, {"throughput_token_per_sec": 10.0})

    def test_legacy_rank_combination_rules(self):
        chain_values = {"chain_0": 0.2, "chain_1": 0.8}
        chain_throughput = {"chain_0": 10.0, "chain_1": 30.0}
        cases = (
            ("throughput_token_per_sec", 1.0),
            ("depth_req_q", 1.0),
            ("p99_ttft_ms", 0.8),
            ("p99_tpot_ms", 0.8),
            ("slo_margin", 0.2),
            ("kvcache_hit_rate", 0.5),
            ("cost_per_token", 0.65),
        )

        for metric, expected in cases:
            with self.subTest(metric=metric):
                self.assertAlmostEqual(
                    StoreTelemetry._rank_value(metric, chain_values, chain_throughput),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
