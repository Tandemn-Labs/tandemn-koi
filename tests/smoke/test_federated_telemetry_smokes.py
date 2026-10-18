"""Federated collection, adapter grouping, and node-price calculations without Store reads."""

import io
import json
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs

import numpy as np
from src.infra.federated_telemetry import FederatedTelemetry
from src.infra.metrics.adapters.dcgm import DCGMAdapter
from src.infra.metrics.adapters.vllm import VLLMAdapter
from src.infra.metrics.backends.prometheus import PrometheusReader
from src.infra.metrics.base import MetricQueryError, RankObservations, load_plugins
from src.infra.telemetry import RankTelemetry

END = datetime(2026, 10, 17, 12, tzinfo=UTC)
START = END - timedelta(seconds=60)


def node(name, price, cluster="cluster_1"):
    return {"node_id": name, "cluster_id": cluster, "price_per_node_hour": price}


def rank(name="rank_1", **fields):
    return {
        "rank_id": name,
        "cluster_id": "cluster_1",
        "engine_name": "vllm",
        "device_source": "dcgm",
        "expected_replicas": 2,
        **fields,
    }


def snapshot(*jobs):
    return SimpleNamespace(
        active_jobs_summary=lambda: [{"job_id": job_id, "ranks": ranks} for job_id, ranks in jobs]
    )


def observations(*, v=None, y=None, times=None, **kwargs):
    v = {key: np.asarray(values, dtype=float) for key, values in (v or {}).items()}
    y = {key: np.asarray(values, dtype=float) for key, values in (y or {}).items()}
    timestamps = {
        key: np.asarray(times, dtype=float)
        if times is not None
        else START.timestamp() + np.arange(len(values)) * 10
        for key, values in {**v, **y}.items()
    }
    return RankObservations(v=v, y=y, timestamps=timestamps, **kwargs)


class Adapter:
    def __init__(self, kind, name, results=None):
        self.kind, self.name = kind, name
        self.results = results or {}
        self.calls = []
        self.error = None

    def collect(self, contexts, *, start, end, step_seconds):
        self.calls.append(([(c.job_id, c.rank_id) for c in contexts], start, end, step_seconds))
        if self.error:
            raise self.error
        return self.results


class FederatedTelemetrySmokeTests(unittest.TestCase):
    def collector(self, *adapters, **kwargs):
        return FederatedTelemetry(
            plugins=load_plugins(adapters),
            tick_interval_sec=60,
            sample_bucket_sec=15,
            now_fn=kwargs.pop("now_fn", lambda: END),
            **kwargs,
        )

    def collect_one(self, nodes=(), tps=(100, 200), **kwargs):
        raw = observations(
            v={"depth_req_q": [0, 2]},
            y={} if tps is None else {"throughput_token_per_sec": tps},
        )
        engine = Adapter("engine", "vllm", {("job_1", "rank_1"): raw})
        collector = self.collector(engine, Adapter("device", "dcgm"), **kwargs)
        bundle = collector.collect_telemetry(0, 1, snapshot(("job_1", [rank(nodes=nodes)])))
        return collector, bundle, next(collector.iter_per_rank(bundle))

    def test_real_adapters_combine_multiple_ranks_with_one_request_per_measurement(self):
        keys = [("job_1", "rank_1"), ("job_1", "rank_2"), ("job_2", "rank_1")]
        calls = []

        def open_response(request, timeout):
            params = parse_qs(request.data.decode())
            calls.append(params)
            query = params["query"][0]
            values = None
            labels = {}
            if "vllm:generation_tokens_total" in query:
                values = [100, 200]
            elif "vllm:time_to_first_token_seconds_bucket" in query:
                values = [0.1, 0.2]
            elif "vllm:num_requests_waiting" in query:
                values = [0, 2]
            elif "DCGM_FI_PROF_SM_ACTIVE" in query:
                values = [0.5, 0.75]
            elif "vllm:cache_config_info" in query:
                values, labels = [1, 1], {"num_gpu_blocks": "4096"}
            result = (
                []
                if values is None
                else [
                    {
                        "metric": {"job_id": job, "rank_id": rid, **labels},
                        "values": [[START.timestamp(), values[0]], [END.timestamp(), values[1]]],
                    }
                    for job, rid in keys
                ]
            )
            payload = {
                "status": "success",
                "data": {"resultType": "matrix", "result": result},
                "warnings": ["partial response"] if "vllm:generation_tokens_total" in query else [],
                "infos": ["query notice"] if labels else [],
            }
            return io.BytesIO(json.dumps(payload).encode())

        reader = PrometheusReader("http://unused.local", opener=open_response)
        collector = self.collector(VLLMAdapter(reader), DCGMAdapter(reader))
        view = snapshot(
            (
                "job_1",
                [
                    rank(
                        nodes=[node("a", 12), node("b", 24)],
                        committed_mechanism_id="mech_1",
                        deploy_timestamp_utc=10,
                    ),
                    rank("rank_2", nodes=[node("free", 0)]),
                ],
            ),
            ("job_2", [rank()]),
        )
        bundle = collector.collect_telemetry(20, 21, view)
        outputs = {(r.job_id, r.rank_id): r for r in collector.iter_per_rank(bundle)}
        self.assertEqual(set(outputs), set(keys))
        self.assertEqual(len(calls), 19)
        for params in calls:
            self.assertEqual(
                (params["start"], params["end"], params["step"]),
                ([str(START.timestamp())], [str(END.timestamp())], ["15"]),
            )
            self.assertIn('job_id="job_1",rank_id=~"rank_1|rank_2"', params["query"][0])
            self.assertIn('job_id="job_2",rank_id="rank_1"', params["query"][0])
        for output in outputs.values():
            self.assertIsInstance(output, RankTelemetry)
            self.assertEqual((output.v_predicted, output.y_predicted), ({}, {}))
            np.testing.assert_array_equal(output.v_observed["depth_req_q"], [0, 2])
            np.testing.assert_array_equal(output.v_observed["sm_utilization"], [0.5, 0.75])
            np.testing.assert_array_equal(output.y_observed["p99_ttft_ms"], [100, 200])
            self.assertEqual(output.expected_replicas, 2)
            self.assertEqual(output.health_observed_replicas, 0)
        first = outputs[("job_1", "rank_1")]
        self.assertEqual((first.committed_mechanism_id, first.deploy_timestamp_utc), ("mech_1", 10))
        np.testing.assert_allclose(first.y_observed["cost_per_token"], [0.0001, 0.00005])
        np.testing.assert_array_equal(
            outputs[("job_1", "rank_2")].y_observed["cost_per_token"], [0, 0]
        )
        self.assertNotIn("cost_per_token", outputs[("job_2", "rank_1")].y_observed)
        observed = bundle.observations[("job_1", "rank_1")]
        np.testing.assert_array_equal(observed.features["num_kv_blocks_mean"], [4096, 4096])
        np.testing.assert_array_equal(
            observed.timestamps["cost_per_token"], [START.timestamp(), END.timestamp()]
        )
        self.assertTrue(any("partial response" in message for message in observed.warnings))
        self.assertTrue(any("query notice" in message for message in observed.infos))

    def test_selects_mixed_sources_and_groups_each_adapter_across_jobs(self):
        vllm, other = Adapter("engine", "vllm"), Adapter("engine", "other_engine")
        dcgm, device = Adapter("device", "dcgm"), Adapter("device", "other_device")
        collector = self.collector(vllm, other, dcgm, device)
        collector.collect_telemetry(
            0,
            1,
            snapshot(
                ("job_1", [rank(), rank("rank_2", engine_name="other_engine")]),
                ("job_2", [rank(device_source="other_device")]),
            ),
        )
        self.assertEqual(vllm.calls[0][0], [("job_1", "rank_1"), ("job_2", "rank_1")])
        self.assertEqual(other.calls[0][0], [("job_1", "rank_2")])
        self.assertEqual(dcgm.calls[0][0], [("job_1", "rank_1"), ("job_1", "rank_2")])
        self.assertEqual(device.calls[0][0], [("job_2", "rank_1")])
        self.assertTrue(all(len(adapter.calls) == 1 for adapter in (vllm, other, dcgm, device)))

    def test_unknown_sources_fail_before_any_collection(self):
        engine, device = Adapter("engine", "vllm"), Adapter("device", "dcgm")
        collector = self.collector(engine, device)
        with self.assertRaisesRegex(ValueError, "no engine adapter"):
            collector.collect_telemetry(
                0, 1, snapshot(("job_1", [rank(), rank("bad", engine_name="missing")]))
            )
        self.assertEqual(engine.calls + device.calls, [])

    def test_cost_requires_all_prices_and_a_positive_throughput(self):
        for nodes, tps in (
            ([], [100]),
            ([node("a", None)], [100]),
            ([node("a", 12), node("b", None)], [100]),
            ([node("a", 12)], None),
            ([node("a", 12)], [0, -1]),
        ):
            with self.subTest(nodes=nodes, tps=tps):
                _, _, output = self.collect_one(nodes, tps)
                self.assertNotIn("cost_per_token", output.y_observed)
        _, bundle, output = self.collect_one([node("a", 36)], [0, 100, -1, 200, np.nan, np.inf])
        np.testing.assert_allclose(output.y_observed["cost_per_token"], [0.0001, 0.00005])
        np.testing.assert_array_equal(
            bundle.observations[("job_1", "rank_1")].timestamps["cost_per_token"],
            START.timestamp() + np.array([10, 30]),
        )
        self.assertEqual(output.y_observed["throughput_token_per_sec"][0], 0)

    def test_node_deduplication_and_cluster_identity(self):
        _, _, output = self.collect_one([node("a", 12), node("a", 12), node("a", 24, "cluster_2")])
        np.testing.assert_allclose(output.y_observed["cost_per_token"], [0.0001, 0.00005])
        with self.assertRaisesRegex(ValueError, "conflicting node price"):
            self.collect_one([node("a", 12), node("a", 24)])

    def test_shared_nodes_have_no_cost_without_an_allocation_rule(self):
        results = {
            (job, "rank_1"): observations(y={"throughput_token_per_sec": [100]})
            for job in ("job_1", "job_2")
        }
        engine = Adapter("engine", "vllm", results)
        collector = self.collector(engine, Adapter("device", "dcgm"))
        bundle = collector.collect_telemetry(
            0,
            1,
            snapshot(
                ("job_1", [rank(nodes=[node("shared", 12)])]),
                ("job_2", [rank(nodes=[node("shared", 12)])]),
            ),
        )
        outputs = list(collector.iter_per_rank(bundle))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all("cost_per_token" not in output.y_observed for output in outputs))

    def test_empty_ranks_skip_and_graph_filtering_happens_after_cost_calculation(self):
        graph = SimpleNamespace(v=[], y=["cost_per_token"])
        collector, bundle, output = self.collect_one([node("a", 36)], candidate_graph=graph)
        self.assertEqual(output.v_observed, {})
        self.assertEqual(set(output.y_observed), {"cost_per_token"})
        self.assertIn("throughput_token_per_sec", bundle.observations[("job_1", "rank_1")].y)
        empty = collector.collect_telemetry(1, 2, snapshot())
        self.assertEqual(list(collector.iter_per_rank(empty)), [])
        missing = self.collector(Adapter("engine", "vllm"), Adapter("device", "dcgm"))
        empty = missing.collect_telemetry(0, 1, snapshot(("job_1", [rank()])))
        self.assertEqual(list(missing.iter_per_rank(empty)), [])

    def test_adapter_results_are_not_mutated_and_duplicate_fields_are_rejected(self):
        raw = observations(v={"depth_req_q": [0]}, y={"throughput_token_per_sec": [100]})
        engine = Adapter("engine", "vllm", {("job_1", "rank_1"): raw})
        device = Adapter("device", "dcgm")
        collector = self.collector(engine, device)
        view = snapshot(("job_1", [rank(nodes=[node("a", 36)])]))
        collector.collect_telemetry(0, 1, view)
        self.assertNotIn("cost_per_token", raw.y)
        self.assertNotIn("cost_per_token", raw.timestamps)
        device.results = {("job_1", "rank_1"): observations(v={"depth_req_q": [5]})}
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            self.collector(engine, device).collect_telemetry(0, 1, view)

    def test_failed_collection_does_not_advance_the_window_or_fall_back(self):
        engine, device = Adapter("engine", "vllm"), Adapter("device", "dcgm")
        now = iter([END, END + timedelta(seconds=60), END + timedelta(seconds=120)])
        collector = self.collector(engine, device, now_fn=lambda: next(now))
        view = snapshot(("job_1", [rank()]))
        collector.collect_telemetry(0, 1, view)
        engine.error = MetricQueryError("offline")
        with self.assertRaisesRegex(MetricQueryError, "offline"):
            collector.collect_telemetry(1, 2, view)
        engine.error = None
        bundle = collector.collect_telemetry(1, 2, view)
        self.assertEqual(bundle.start, END)
        self.assertEqual(bundle.end, END + timedelta(seconds=120))
        self.assertEqual([call[1] for call in engine.calls], [START, END, END])

    def test_next_window_does_not_repeat_the_previous_end_sample(self):
        engine = Adapter("engine", "vllm")
        now = iter([END, END + timedelta(seconds=30)])
        collector = self.collector(engine, Adapter("device", "dcgm"), now_fn=lambda: next(now))
        view = snapshot(("job_1", [rank()]))
        collector.collect_telemetry(0, 1, view)
        raw = observations(
            y={"throughput_token_per_sec": [999, 2, 3]},
            times=END.timestamp() + np.array([0, 15, 30]),
        )
        engine.results = {("job_1", "rank_1"): raw}
        output = next(collector.iter_per_rank(collector.collect_telemetry(1, 2, view)))
        np.testing.assert_array_equal(output.y_observed["throughput_token_per_sec"], [2, 3])
        np.testing.assert_array_equal(raw.y["throughput_token_per_sec"], [999, 2, 3])

    def test_invalid_timing_and_unrequested_rank_results_fail(self):
        for kwargs in (
            {"tick_interval_sec": 0},
            {"sample_bucket_sec": -1},
            {"sample_bucket_sec": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FederatedTelemetry(plugins={}, **kwargs)
        bad = Adapter(
            "engine", "vllm", {("other_job", "rank_1"): observations(v={"depth_req_q": [1]})}
        )
        with self.assertRaisesRegex(ValueError, "unexpected job/rank"):
            self.collector(bad, Adapter("device", "dcgm")).collect_telemetry(
                0, 1, snapshot(("job_1", [rank()]))
            )


if __name__ == "__main__":
    unittest.main()
