"""Source adapters translate database-aggregated series without Store access."""

import importlib
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from src.infra.metrics.adapters.dcgm import DCGMAdapter
from src.infra.metrics.adapters.vllm import VLLMAdapter
from src.infra.metrics.backends.prometheus import collect_measurements
from src.infra.metrics.base import (
    MetricQueryError,
    QueryResult,
    RankContext,
    TimeSeries,
    adapters_for_rank,
    load_plugins,
)

START = datetime(2026, 10, 16, tzinfo=UTC)
END = START + timedelta(seconds=60)
TIMES = np.array([START.timestamp(), END.timestamp()])
PROFILE_DIR = Path(__file__).resolve().parents[2] / "src/infra/metrics/profiles"


def context(job="job_1", rank="rank_1", **kwargs):
    return RankContext(job, rank, "cluster_1", "vllm", "dcgm", **kwargs)


def series(values, job="job_1", rank="rank_1", timestamps=TIMES, **labels):
    return TimeSeries(
        {"job_id": job, "rank_id": rank, **labels},
        np.asarray(timestamps, dtype=float),
        np.asarray(values, dtype=float),
    )


class FakeReader:
    def __init__(self, source, results=None):
        self.queries = json.loads((PROFILE_DIR / f"{source}_promql.json").read_text())["queries"]
        self.results = results or {}
        self.calls = []

    def query_range(self, query, *, start, end, step_seconds):
        self.calls.append((query, start, end, step_seconds))
        # A single rank makes the rendered expression directly comparable to its profile.
        for name, item in self.queries.items():
            expected = item["expression"].replace(
                "__KOI_RANK_FILTER__", 'job_id="job_1",rank_id="rank_1"'
            )
            if query == expected:
                return self.results.get(name, QueryResult(()))
        return QueryResult(())


class MetricsAdapterSmokeTests(unittest.TestCase):
    def collect(self, adapter, ranks=None):
        return adapter.collect(ranks or [context()], start=START, end=END, step_seconds=15)

    def test_shared_query_helper_is_in_the_prometheus_backend(self):
        backend = importlib.import_module("src.infra.metrics.backends.prometheus")
        self.assertTrue(callable(getattr(backend, "collect_measurements", None)))

    def test_query_without_rank_filter_marker_is_rejected(self):
        reader = FakeReader("vllm")
        with self.assertRaisesRegex(ValueError, "missing __KOI_RANK_FILTER__"):
            collect_measurements(
                reader,
                [context()],
                {"unscoped": "up"},
                start=START,
                end=END,
                step_seconds=15,
            )
        self.assertEqual(reader.calls, [])

    def test_vllm_translates_units_and_preserves_zero(self):
        measurements = {
            "ttft_p99_seconds": [0, 0.12],
            "tpot_p99_seconds": [0.01, 0.02],
            "generated_output_tokens_per_second": [0, 50],
            "running_requests": [0, 4],
            "waiting_requests": [0, 2],
            "kv_cache_usage_fraction": [0, 0.8],
            "prefix_cache_hit_fraction": [0, 0.5],
            "prompt_tokens_per_request": [10, 20],
            "generated_tokens_per_request": [5, 15],
            "prefill_observations_per_second": [0, 3],
            "decode_observations_per_second": [0, 2],
            "prefill_decode_time_ratio": [0, 1.5],
        }
        reader = FakeReader(
            "vllm", {name: QueryResult((series(values),)) for name, values in measurements.items()}
        )
        output = self.collect(VLLMAdapter(reader))[("job_1", "rank_1")]
        expected_v = {
            "live_batch_size": [0, 4],
            "depth_req_q": [0, 2],
            "kv_cache_util": [0, 0.8],
            "kvcache_hit_rate": [0, 0.5],
            "input_length_observed": [10, 20],
            "output_length_observed": [5, 15],
            "prefill_iteration_counts_per_second": [0, 3],
            "decode_itr_counts_per_second": [0, 2],
            "pd_inbalance": [0, 1.5],
        }
        self.assertEqual(set(output.v), set(expected_v))
        for name, values in expected_v.items():
            np.testing.assert_allclose(output.v[name], values)
            np.testing.assert_array_equal(output.timestamps[name], TIMES)
        np.testing.assert_allclose(output.y["p99_ttft_ms"], [0, 120])
        np.testing.assert_allclose(output.y["p99_tpot_ms"], [10, 20])
        np.testing.assert_array_equal(output.y["throughput_token_per_sec"], [0, 50])
        self.assertNotIn("cost_per_token", output.y)
        self.assertEqual(len(reader.calls), len(measurements) + 1)
        for _, start, end, step in reader.calls:
            self.assertEqual((start, end, step), (START, END, 15))

    def test_slo_margin_uses_the_existing_ttft_target(self):
        reader = FakeReader("vllm", {"ttft_p99_seconds": QueryResult((series([0.1, 0.3]),))})
        output = self.collect(
            VLLMAdapter(reader), [context(job_features={"target_p99_ttft_ms": 200})]
        )[("job_1", "rank_1")]
        np.testing.assert_allclose(output.y["slo_margin"], [0.5, -0.5])
        for target in (None, 0, -1, float("nan"), "200", True):
            with self.subTest(target=target):
                output = self.collect(
                    VLLMAdapter(reader), [context(job_features={"target_p99_ttft_ms": target})]
                )[("job_1", "rank_1")]
                self.assertNotIn("slo_margin", output.y)

    def test_dcgm_keeps_fraction_and_byte_units_and_converts_mib_to_gib(self):
        reader = FakeReader(
            "dcgm",
            {
                "gpu_memory_used_fraction_mean": QueryResult((series([0, 0.75]),)),
                "gpu_memory_free_mib_mean": QueryResult((series([0, 2048]),)),
                "sm_active_fraction_mean": QueryResult((series([0, 0.5]),)),
                "dram_active_fraction_mean": QueryResult((series([0, 0.6]),)),
                "pcie_bytes_per_second_mean": QueryResult((series([0, 1e9]),)),
                "nvlink_bytes_per_second_mean": QueryResult((series([0, 2e9]),)),
            },
        )
        output = self.collect(DCGMAdapter(reader))[("job_1", "rank_1")]
        for name, values in {
            "gpu_mem_used_fraction": [0, 0.75],
            "vram_headroom_gb": [0, 2],
            "sm_utilization": [0, 0.5],
            "mem_bandwidth_utilization": [0, 0.6],
            "pcie_tput_observed": [0, 1e9],
            "nvlink_tput_observed": [0, 2e9],
        }.items():
            np.testing.assert_allclose(output.v[name], values)
        self.assertEqual(output.y, {})
        self.assertNotIn("activation_mem_pressure", output.v)
        self.assertEqual(len(reader.calls), 6)

    def test_empty_results_do_not_manufacture_measurements(self):
        for source, adapter_type in (("vllm", VLLMAdapter), ("dcgm", DCGMAdapter)):
            with self.subTest(source=source):
                reader = FakeReader(source)
                output = self.collect(adapter_type(reader))[("job_1", "rank_1")]
                self.assertEqual((output.v, output.y, output.timestamps), ({}, {}, {}))
                self.assertEqual(output.features, {})
                self.assertEqual(adapter_type(reader).collect([], start=START, end=END), {})

    def test_warnings_infos_and_sample_timestamps_survive_translation(self):
        reader = FakeReader(
            "vllm",
            {
                "ttft_p99_seconds": QueryResult(
                    (series([0.2], timestamps=TIMES[1:]),), ("partial response",), ("notice",)
                )
            },
        )
        output = self.collect(VLLMAdapter(reader))[("job_1", "rank_1")]
        np.testing.assert_array_equal(output.timestamps["p99_ttft_ms"], TIMES[1:])
        self.assertTrue(any("partial response" in warning for warning in output.warnings))
        self.assertTrue(any("notice" in info for info in output.infos))
        self.assertEqual(reader.results["ttft_p99_seconds"].series[0].values[0], 0.2)

    def test_one_query_per_measurement_scopes_exact_job_rank_pairs(self):
        reader = FakeReader("vllm")
        ranks = [context(), context(rank="rank_2"), context(job="job_2", rank="rank_3")]
        output = self.collect(VLLMAdapter(reader), ranks)
        self.assertEqual(set(output), {(rank.job_id, rank.rank_id) for rank in ranks})
        self.assertEqual(len(reader.calls), 13)
        for query, *_ in reader.calls:
            self.assertIn('job_id="job_1",rank_id=~"rank_1|rank_2"', query)
            self.assertIn('job_id="job_2",rank_id="rank_3"', query)
            self.assertNotIn("__KOI_RANK_FILTER__", query)

    def test_job_and_regex_labels_are_escaped(self):
        reader = FakeReader("vllm")
        job = 'job"\\\n'
        self.collect(VLLMAdapter(reader), [context(job, "rank.1"), context(job, "rank+2")])
        query = reader.calls[0][0]
        self.assertIn("job_id=" + json.dumps(job), query)
        self.assertIn('rank_id=~"rank\\\\.1|rank\\\\+2"', query)

    def test_results_are_separated_by_job_and_rank(self):
        class Reader:
            def query_range(self, query, **kwargs):
                if "vllm:generation_tokens_total" in query:
                    return QueryResult((series([1, 2]), series([3, 4], job="job_2")))
                return QueryResult(())

        output = self.collect(VLLMAdapter(Reader()), [context(), context(job="job_2")])
        np.testing.assert_array_equal(
            output[("job_1", "rank_1")].y["throughput_token_per_sec"], [1, 2]
        )
        np.testing.assert_array_equal(
            output[("job_2", "rank_1")].y["throughput_token_per_sec"], [3, 4]
        )

    def test_unscoped_duplicate_and_non_rank_results_are_rejected(self):
        invalid = (
            (series([1, 2], job="other_job"),),
            (series([1, 2]), series([3, 4])),
            (series([1, 2], source_id="replica_1"),),
            (TimeSeries({"rank_id": "rank_1"}, TIMES, np.array([1.0, 2.0])),),
        )
        for samples in invalid:
            with self.subTest(samples=samples):
                reader = FakeReader("vllm", {"running_requests": QueryResult(samples)})
                with self.assertRaises(MetricQueryError):
                    self.collect(VLLMAdapter(reader))

    def test_query_failures_propagate_without_store_fallback(self):
        class Reader:
            def query_range(self, query, **kwargs):
                raise MetricQueryError("database unavailable")

        with self.assertRaisesRegex(MetricQueryError, "database unavailable"):
            self.collect(VLLMAdapter(Reader()))

    def test_real_adapters_use_the_existing_registry(self):
        vllm = VLLMAdapter(FakeReader("vllm"))
        dcgm = DCGMAdapter(FakeReader("dcgm"))
        self.assertEqual(adapters_for_rank(context(), load_plugins([vllm, dcgm])), (vllm, dcgm))

    def test_kv_blocks_are_a_separate_timestamped_capacity_feature(self):
        raw = series([1, 1], num_gpu_blocks="4096")
        reader = FakeReader(
            "vllm", {"num_kv_blocks_mean": QueryResult((raw,), ("partial config",), ("notice",))}
        )
        output = self.collect(VLLMAdapter(reader))[("job_1", "rank_1")]
        np.testing.assert_array_equal(output.features["num_kv_blocks_mean"], [4096, 4096])
        np.testing.assert_array_equal(output.timestamps["num_kv_blocks_mean"], TIMES)
        self.assertEqual((output.v, output.y), ({}, {}))
        self.assertTrue(any("partial config" in warning for warning in output.warnings))
        self.assertTrue(any("notice" in info for info in output.infos))
        np.testing.assert_array_equal(raw.values, [1, 1])

    def test_kv_capacity_can_change_over_time_without_multiplying_replicas(self):
        raw = QueryResult(
            (
                series([1], timestamps=TIMES[1:], num_gpu_blocks="8192"),
                series([1], timestamps=TIMES[:1], num_gpu_blocks="4096"),
                series([1], timestamps=TIMES[:1], num_gpu_blocks="4096"),
            )
        )
        reader = FakeReader("vllm", {"num_kv_blocks_mean": raw})
        output = self.collect(VLLMAdapter(reader))[("job_1", "rank_1")]
        np.testing.assert_array_equal(output.features["num_kv_blocks_mean"], [4096, 8192])
        np.testing.assert_array_equal(output.timestamps["num_kv_blocks_mean"], TIMES)

    def test_kv_capacity_disagreement_is_not_an_invented_average(self):
        raw = QueryResult(
            (
                series([1, 1], num_gpu_blocks="4096"),
                series([1, 1], num_gpu_blocks="8192"),
            )
        )
        reader = FakeReader("vllm", {"num_kv_blocks_mean": raw})
        with self.assertRaisesRegex(MetricQueryError, "capacity disagreement"):
            self.collect(VLLMAdapter(reader))

    def test_zero_kv_capacity_is_preserved_and_unknown_capacity_is_absent(self):
        for label in ("0", "None", None):
            with self.subTest(label=label):
                labels = {} if label is None else {"num_gpu_blocks": label}
                raw = QueryResult((series([1], timestamps=TIMES[:1], **labels),))
                reader = FakeReader("vllm", {"num_kv_blocks_mean": raw})
                output = self.collect(VLLMAdapter(reader))[("job_1", "rank_1")]
                if label == "0":
                    np.testing.assert_array_equal(output.features["num_kv_blocks_mean"], [0])
                    np.testing.assert_array_equal(
                        output.timestamps["num_kv_blocks_mean"], TIMES[:1]
                    )
                else:
                    self.assertEqual((output.features, output.timestamps), ({}, {}))
                    self.assertTrue(output.warnings)

    def test_kv_capacity_still_checks_identity_and_invalid_counts(self):
        invalid = (
            series([1, 1], num_gpu_blocks="-1"),
            series([1, 1], num_gpu_blocks="3.5"),
            series([1, 1], num_gpu_blocks="NaN"),
            series([1, 1], num_gpu_blocks="4096", job="other_job"),
            series([1, 1], num_gpu_blocks="None", job="other_job"),
            series([1, 1], num_gpu_blocks="4096", source_id="unexpected"),
            TimeSeries({"num_gpu_blocks": "4096"}, TIMES, np.ones(2)),
        )
        for raw in invalid:
            with self.subTest(labels=raw.labels):
                reader = FakeReader("vllm", {"num_kv_blocks_mean": QueryResult((raw,))})
                with self.assertRaises(MetricQueryError):
                    self.collect(VLLMAdapter(reader))

    def test_kv_capacity_is_separated_by_job_and_rank(self):
        class Reader:
            def query_range(self, query, **kwargs):
                if "vllm:cache_config_info" in query:
                    return QueryResult(
                        (
                            series([1, 1], num_gpu_blocks="4096"),
                            series([1, 1], job="job_2", num_gpu_blocks="8192"),
                        )
                    )
                return QueryResult(())

        outputs = self.collect(VLLMAdapter(Reader()), [context(), context(job="job_2")])
        np.testing.assert_array_equal(
            outputs[("job_1", "rank_1")].features["num_kv_blocks_mean"], [4096, 4096]
        )
        np.testing.assert_array_equal(
            outputs[("job_2", "rank_1")].features["num_kv_blocks_mean"], [8192, 8192]
        )

    def test_profiles_keep_existing_rank_aggregation_meanings(self):
        vllm = json.loads((PROFILE_DIR / "vllm_promql.json").read_text())["queries"]
        dcgm = json.loads((PROFILE_DIR / "dcgm_promql.json").read_text())["queries"]
        for name in ("ttft_p99_seconds", "tpot_p99_seconds"):
            self.assertIn("sum by (le, job_id, rank_id)", vllm[name]["expression"])
            self.assertTrue(vllm[name]["expression"].startswith("histogram_quantile"))
        for name in (
            "prefix_cache_hit_fraction",
            "prompt_tokens_per_request",
            "generated_tokens_per_request",
            "prefill_decode_time_ratio",
        ):
            self.assertTrue(vllm[name]["expression"].startswith("sum by (job_id, rank_id)"))
            self.assertEqual(vllm[name]["expression"].count("sum by (job_id, rank_id)"), 2)
        memory = dcgm["gpu_memory_used_fraction_mean"]["expression"]
        self.assertTrue(memory.startswith("avg by (job_id, rank_id)"))
        self.assertIn("DCGM_FI_DEV_FB_RESERVED", memory)


if __name__ == "__main__":
    unittest.main()
