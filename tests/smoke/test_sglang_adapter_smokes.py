"""SGLang v0.5.21 metrics translate into the shared rank observation contract."""

import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from src.infra.metrics.adapters.dcgm import DCGMAdapter
from src.infra.metrics.adapters.sglang import SGLangAdapter
from src.infra.metrics.base import (
    QueryResult,
    RankContext,
    TimeSeries,
    adapters_for_rank,
    load_plugins,
)

START = datetime(2026, 10, 16, tzinfo=UTC)
END = START + timedelta(seconds=60)
TIMES = np.array([START.timestamp(), END.timestamp()])
PROFILE = Path(__file__).resolve().parents[2] / "src/infra/metrics/profiles/sglang_promql.json"


def context(job="job_1", rank="rank_1", **kwargs):
    return RankContext(job, rank, "cluster_1", "sglang", "dcgm", **kwargs)


def series(values, job="job_1", rank="rank_1", timestamps=TIMES, **labels):
    return TimeSeries(
        {"job_id": job, "rank_id": rank, **labels},
        np.asarray(timestamps, dtype=float),
        np.asarray(values, dtype=float),
    )


class FakeReader:
    def __init__(self, results=None):
        self.queries = json.loads(PROFILE.read_text())["queries"]
        self.results = results or {}
        self.calls = []

    def query_range(self, query, *, start, end, step_seconds):
        self.calls.append((query, start, end, step_seconds))
        for name, item in self.queries.items():
            selector = 'job_id="job_1",rank_id="rank_1"'
            expected = item["expression"].replace("__KOI_RANK_FILTER__", selector)
            if query == expected:
                return self.results.get(name, QueryResult(()))
        return QueryResult(())


class SGLangAdapterSmokeTests(unittest.TestCase):
    def collect(self, adapter, contexts=None):
        return adapter.collect(contexts or [context()], start=START, end=END, step_seconds=15)

    def test_translates_v0521_metrics_and_keeps_the_generation_gauge_as_is(self):
        values = {
            "ttft_p99_seconds": [0.1, 0.25],
            "tpot_p99_seconds": [0.005, 0.02],
            "generation_throughput": [0, 50],
            "running_requests": [0, 4],
            "waiting_requests": [3, 0],
            "full_token_usage": [0, 0.75],
            "prefix_cache_hit_fraction": [0, 0.25],
            "prompt_tokens_per_request": [0, 32.5],
            "generated_tokens_per_request": [0, 40],
            "prefill_observations_per_second": [0, 4],
            "decode_observations_per_second": [0, 3],
        }
        reader = FakeReader(
            {name: QueryResult((series(samples),)) for name, samples in values.items()}
        )
        output = self.collect(SGLangAdapter(reader))[("job_1", "rank_1")]
        expected_v = {
            "live_batch_size": [0, 4],
            "depth_req_q": [3, 0],
            "kv_cache_util": [0, 0.75],
            "kvcache_hit_rate": [0, 0.25],
            "input_length_observed": [0, 32.5],
            "output_length_observed": [0, 40],
            "prefill_iteration_counts_per_second": [0, 4],
            "decode_itr_counts_per_second": [0, 3],
        }
        self.assertEqual(set(output.v), set(expected_v))
        for field, samples in expected_v.items():
            np.testing.assert_array_equal(output.v[field], samples)
            np.testing.assert_array_equal(output.timestamps[field], TIMES)
        np.testing.assert_array_equal(output.y["p99_ttft_ms"], [100, 250])
        np.testing.assert_array_equal(output.y["p99_tpot_ms"], [5, 20])
        np.testing.assert_array_equal(output.y["throughput_token_per_sec"], [0, 50])
        self.assertEqual(len(reader.calls), 11)
        self.assertTrue(all(call[3] == 15 for call in reader.calls))

    def test_ttft_target_derives_slo_margin(self):
        reader = FakeReader({"ttft_p99_seconds": QueryResult((series([0.1, 0.3]),))})
        output = self.collect(
            SGLangAdapter(reader), [context(job_features={"target_p99_ttft_ms": 200})]
        )[("job_1", "rank_1")]
        np.testing.assert_allclose(output.y["slo_margin"], [0.5, -0.5])
        np.testing.assert_array_equal(output.timestamps["slo_margin"], TIMES)

    def test_missing_metrics_stay_missing_and_zero_samples_stay_present(self):
        reader = FakeReader(
            {
                "generation_throughput": QueryResult((series([0, 0]),)),
                "waiting_requests": QueryResult((series([0, 1]),)),
            }
        )
        output = self.collect(SGLangAdapter(reader))[("job_1", "rank_1")]
        np.testing.assert_array_equal(output.y["throughput_token_per_sec"], [0, 0])
        np.testing.assert_array_equal(output.v["depth_req_q"], [0, 1])
        self.assertNotIn("p99_ttft_ms", output.y)
        self.assertNotIn("kv_cache_util", output.v)
        self.assertNotIn("input_length_observed", output.v)
        self.assertNotIn("output_length_observed", output.v)
        self.assertNotIn("prefill_iteration_counts_per_second", output.v)
        self.assertNotIn("decode_itr_counts_per_second", output.v)
        self.assertNotIn("pd_inbalance", output.v)
        self.assertEqual(SGLangAdapter(FakeReader()).collect([], start=START, end=END), {})

    def test_query_profiles_scope_multiple_jobs_and_ranks_once_per_measurement(self):
        reader = FakeReader()
        contexts = [context(), context(rank="rank_2"), context(job="job_2", rank="rank_1")]
        outputs = self.collect(SGLangAdapter(reader), contexts)
        self.assertEqual(set(outputs), {(c.job_id, c.rank_id) for c in contexts})
        self.assertEqual(len(reader.calls), 11)
        for query, *_ in reader.calls:
            self.assertIn('job_id="job_1",rank_id=~"rank_1|rank_2"', query)
            self.assertIn('job_id="job_2",rank_id="rank_1"', query)
            self.assertNotIn("chain_id", query)

    def test_adapter_selects_through_the_existing_registry(self):
        adapter = SGLangAdapter(FakeReader())
        plugins = load_plugins([adapter, DCGMAdapter(FakeReader())])
        self.assertIs(adapters_for_rank(context(), plugins)[0], adapter)


if __name__ == "__main__":
    unittest.main()
