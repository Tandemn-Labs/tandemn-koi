import io
import json
import unittest
import urllib.parse
from datetime import UTC, datetime

from src.infra.metric_adapters import (
    DcgmAdapter,
    MetricQueryError,
    MetricQueryProfile,
    MetricTarget,
    PromQLExecutor,
    TimeSeries,
    VllmAdapter,
    load_builtin_profile,
)

START = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
END = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)


def _target():
    return MetricTarget(
        job_id="job-1",
        rank_id="rank-1",
        chain_indexes=(0, 1),
        selectors={"cluster": "cluster-a"},
    )


class MetricAdapterSmokeTests(unittest.TestCase):
    def test_vllm_collects_requested_variables_for_live_chains(self):
        expressions = []

        def query_range(expression, target, start, end, step):
            expressions.append(expression)
            self.assertEqual(target, _target())
            return [
                TimeSeries(
                    labels={"job_id": "job-1", "rank_id": "rank-1", "chain_index": "0"},
                    points=[(1.0, 10.0), (2.0, 20.0)],
                ),
                TimeSeries(
                    labels={"job_id": "job-1", "rank_id": "rank-1", "chain_index": "9"},
                    points=[(1.0, 999.0)],
                ),
            ]

        observations = VllmAdapter(load_builtin_profile("vllm", "promql")).collect(
            [_target()],
            START,
            END,
            query_range,
            requested_variables={"throughput_token_per_sec"},
        )

        self.assertEqual([item.value for item in observations], [10.0, 20.0])
        self.assertTrue(all(item.variable == "throughput_token_per_sec" for item in observations))
        self.assertTrue(all(item.source == "vllm" for item in observations))
        self.assertEqual(len(expressions), 1)
        self.assertIn("{target}", expressions[0])
        self.assertIn("vllm:generation_tokens_total", expressions[0])

    def test_dcgm_uses_vendor_neutral_output_names_and_keeps_gpu_labels(self):
        def query_range(expression, target, start, end, step):
            self.assertIn("DCGM_FI_PROF_NVLINK_TX_BYTES", expression)
            return [
                TimeSeries(
                    labels={
                        "job_id": "job-1",
                        "rank_id": "rank-1",
                        "chain_index": "1",
                        "UUID": "GPU-1",
                    },
                    points=[(1.0, 1234.0)],
                )
            ]

        observations = DcgmAdapter(load_builtin_profile("dcgm", "promql")).collect(
            [_target()],
            START,
            END,
            query_range,
            requested_variables={"inter_gpu_tput_bytes_per_sec"},
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].variable, "inter_gpu_tput_bytes_per_sec")
        self.assertEqual(observations[0].labels["UUID"], "GPU-1")

    def test_vllm_adapter_accepts_a_non_promql_profile(self):
        expressions = []
        profile = MetricQueryProfile(
            source="vllm",
            language="sql",
            queries={"throughput_token_per_sec": "SELECT throughput FROM metrics WHERE {target}"},
        )

        def execute_query(expression, target, start, end, step):
            expressions.append((expression, target))
            return [
                TimeSeries(
                    labels={"job_id": "job-1", "rank_id": "rank-1", "chain_index": "0"},
                    points=[(1.0, 12.0)],
                )
            ]

        observations = VllmAdapter(profile).collect(
            [_target()],
            START,
            END,
            execute_query,
        )

        self.assertEqual(
            expressions,
            [("SELECT throughput FROM metrics WHERE {target}", _target())],
        )
        self.assertEqual(observations[0].value, 12.0)

    def test_target_rejects_invalid_chain_indexes(self):
        with self.assertRaises(ValueError):
            MetricTarget("job-1", "rank-1", (-1,))
        with self.assertRaises(ValueError):
            MetricTarget("job-1", "rank-1", (True,))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class PromQLExecutorSmokeTests(unittest.TestCase):
    def test_query_range_posts_and_parses_finite_matrix_samples(self):
        captured = {}
        payload = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {
                            "job_id": "job-1",
                            "rank_id": "rank-1",
                            "chain_index": "0",
                        },
                        "values": [[1, "2.5"], [2, "NaN"]],
                    }
                ],
            },
        }

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = urllib.parse.parse_qs(request.data.decode())
            captured["authorization"] = request.headers.get("Authorization")
            captured["timeout"] = timeout
            return _Response(json.dumps(payload).encode())

        executor = PromQLExecutor(
            "https://prometheus.example/",
            timeout_seconds=3,
            bearer_token="secret",
            opener=opener,
        )
        series = executor.execute("up{target}", _target(), START, END, 10)

        self.assertEqual(captured["url"], "https://prometheus.example/api/v1/query_range")
        self.assertEqual(
            captured["body"]["query"],
            ['up{cluster="cluster-a",job_id="job-1",rank_id="rank-1",chain_index=~"0|1"}'],
        )
        self.assertEqual(captured["body"]["step"], ["10"])
        self.assertEqual(captured["authorization"], "Bearer secret")
        self.assertEqual(captured["timeout"], 3.0)
        self.assertEqual(series[0].points, [(1.0, 2.5)])

    def test_query_range_rejects_prometheus_errors(self):
        def opener(request, timeout):
            return _Response(json.dumps({"status": "error", "error": "bad query"}).encode())

        executor = PromQLExecutor("https://prometheus.example", opener=opener)

        with self.assertRaisesRegex(MetricQueryError, "bad query"):
            executor.execute("bad{target}", _target(), START, END, 10)


if __name__ == "__main__":
    unittest.main()
