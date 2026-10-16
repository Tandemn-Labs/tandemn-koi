"""Check query profiles define labelled, rank-level source measurements."""

import json
import unittest
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parents[2] / "src" / "infra" / "metrics" / "profiles"

EXPECTED_QUERIES = {
    "vllm_promql.json": {
        "ttft_p99_seconds",
        "tpot_p99_seconds",
        "generated_output_tokens_per_second",
        "running_requests",
        "waiting_requests",
        "kv_cache_usage_fraction",
        "prefix_cache_hits_per_second",
        "prefix_cache_queries_per_second",
        "prompt_tokens_per_request",
        "generated_tokens_per_request",
        "prefill_observations_per_second",
        "decode_observations_per_second",
        "prefill_time_seconds_per_second",
        "decode_time_seconds_per_second",
        "cache_config_chains_by_block_shape",
    },
    "dcgm_promql.json": {
        "gpu_memory_used_mib_total",
        "gpu_memory_free_mib_total",
        "gpu_memory_reserved_mib_total",
        "gpu_memory_free_mib_mean",
        "gpu_reporting_device_count",
        "sm_active_fraction_mean",
        "dram_active_fraction_mean",
        "pcie_bytes_per_second_mean",
        "nvlink_bytes_per_second_mean",
    },
}


class MetricProfileSmokeTests(unittest.TestCase):
    def test_each_profile_has_a_unit_and_rank_query_for_every_input(self):
        for filename, expected_queries in EXPECTED_QUERIES.items():
            with self.subTest(profile=filename):
                profile = json.loads((PROFILE_DIR / filename).read_text())
                self.assertEqual(profile["version"], 1)
                self.assertEqual(profile["language"], "promql")
                self.assertEqual(profile["rank_labels"], ["job_id", "rank_id"])
                self.assertEqual(set(profile["queries"]), expected_queries)
                for name, item in profile["queries"].items():
                    with self.subTest(profile=filename, input=name):
                        self.assertTrue(item["unit"])
                        self.assertTrue(item["aggregation"])
                        self.assertIn("{target}", item["expression"])
                        self.assertIn("job_id", item["expression"])
                        self.assertIn("rank_id", item["expression"])

    def test_vllm_counters_get_rates_before_rank_sums(self):
        profile = json.loads((PROFILE_DIR / "vllm_promql.json").read_text())
        for name in (
            "generated_output_tokens_per_second",
            "prefix_cache_hits_per_second",
            "prefix_cache_queries_per_second",
        ):
            with self.subTest(input=name):
                expression = profile["queries"][name]["expression"]
                self.assertIn("sum by (job_id, rank_id) (rate(", expression)
        self.assertIn(
            "sum by (le, job_id, rank_id)",
            profile["queries"]["ttft_p99_seconds"]["expression"],
        )
        self.assertNotIn("chain_id", profile["queries"]["ttft_p99_seconds"]["expression"])

    def test_vllm_cache_metadata_drops_chain_id_from_the_result(self):
        profile = json.loads((PROFILE_DIR / "vllm_promql.json").read_text())
        expression = profile["queries"]["cache_config_chains_by_block_shape"]["expression"]

        self.assertIn("num_gpu_blocks", expression)
        self.assertIn("block_size", expression)
        self.assertIn("chain_id", expression)
        self.assertIn(
            "count by (job_id, rank_id, num_gpu_blocks, block_size)",
            expression,
        )

    def test_dcgm_uses_reserved_memory_and_does_not_redifferentiate_rates(self):
        profile = json.loads((PROFILE_DIR / "dcgm_promql.json").read_text())
        self.assertTrue(
            {
                "gpu_memory_used_mib_total",
                "gpu_memory_free_mib_total",
                "gpu_memory_reserved_mib_total",
            }.issubset(profile["queries"])
        )
        for name in ("pcie_bytes_per_second_mean", "nvlink_bytes_per_second_mean"):
            with self.subTest(input=name):
                expression = profile["queries"][name]["expression"]
                self.assertIn("TX_BYTES", expression)
                self.assertIn("RX_BYTES", expression)
                self.assertNotIn("rate(", expression)


if __name__ == "__main__":
    unittest.main()
