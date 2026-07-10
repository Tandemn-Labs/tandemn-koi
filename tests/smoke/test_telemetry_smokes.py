import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from src.core.candidate_graph import CandidateGraph
from src.core.models import Node
from src.infra.resource_map import ClusterResourceSnapshot
from src.infra.telemetry import StoreTelemetry


def _graph():
    nodes = {
        "kv_cache_util": Node("kv_cache_util", "V"),
        "p99_ttft_ms": Node("p99_ttft_ms", "Y"),
        "throughput_token_per_sec": Node("throughput_token_per_sec", "Y"),
        "cost_per_token": Node("cost_per_token", "Y"),
        "slo_margin": Node("slo_margin", "Y"),
    }
    return CandidateGraph(nodes, {}, {})


def _snapshot():
    shape = {"rank_id": "rank_0", "mechanism_id": "mech_1", "model_id": "m"}
    return ClusterResourceSnapshot(
        tick=1,
        resources={},
        active_jobs=[
            {
                "job_id": "job_1",
                "job_features": {"type": "online"},
                "active_chains": [
                    {"chain_id": "chain_a", "shape_json": dict(shape)},
                    {"chain_id": "chain_b", "shape_json": dict(shape)},
                ],
            }
        ],
        pending_jobs=[],
    )


def _row(chain_id, rank_id="rank_0", **metrics):
    return SimpleNamespace(
        ts=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        chain_id=chain_id,
        rank_id=rank_id,
        **metrics,
    )


class _Store:
    def __init__(self, rows):
        self.rows = rows

    def rows_for_job_window(self, user_id, job_id, start, end):
        return list(self.rows)


class StoreTelemetrySmokeTests(unittest.TestCase):
    def test_aggregates_rank_trajectories_without_gpu_double_counting(self):
        telemetry = StoreTelemetry(
            user_id="user_1",
            gpu_metric_store=_Store(
                [
                    _row(
                        "chain_a",
                        throughput_token_per_sec=10.0,
                        kv_cache_util=0.2,
                        p99_ttft_ms=100.0,
                        slo_margin=5.0,
                        cost_per_token=0.01,
                        depth_req_q=999.0,
                    ),
                    _row(
                        "chain_a",
                        throughput_token_per_sec=10.0,
                        kv_cache_util=0.4,
                        p99_ttft_ms=100.0,
                        slo_margin=5.0,
                        cost_per_token=0.01,
                    ),
                    _row(
                        "chain_b",
                        throughput_token_per_sec=20.0,
                        kv_cache_util=0.9,
                        p99_ttft_ms=120.0,
                        slo_margin=2.0,
                        cost_per_token=0.02,
                    ),
                    _row("stale_chain", throughput_token_per_sec=999.0),
                ]
            ),
            candidate_graph=_graph(),
            now_fn=lambda: datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC),
        )

        bundle = telemetry.collect_telemetry(0, 1, _snapshot())
        rank = next(telemetry.iter_per_rank(bundle))

        self.assertEqual(rank.job_id, "job_1")
        self.assertEqual(rank.rank_id, "rank_0")
        self.assertEqual(rank.committed_mechanism_id, "mech_1")
        self.assertAlmostEqual(rank.v_observed["kv_cache_util"][0], 0.6)
        self.assertAlmostEqual(rank.y_observed["throughput_token_per_sec"][0], 30.0)
        self.assertAlmostEqual(rank.y_observed["p99_ttft_ms"][0], 120.0)
        self.assertAlmostEqual(rank.y_observed["slo_margin"][0], 2.0)
        self.assertAlmostEqual(rank.y_observed["cost_per_token"][0], 0.5 / 30.0)
        self.assertNotIn("depth_req_q", rank.v_observed)
        self.assertEqual(rank.v_predicted, {})
        self.assertEqual(rank.y_predicted, {})

    def test_rank_mismatch_is_contract_error(self):
        telemetry = StoreTelemetry(
            user_id="user_1",
            gpu_metric_store=_Store([_row("chain_a", rank_id="rank_bad", kv_cache_util=0.2)]),
            candidate_graph=_graph(),
            now_fn=lambda: datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC),
        )

        bundle = telemetry.collect_telemetry(0, 1, _snapshot())
        with self.assertRaises(ValueError):
            list(telemetry.iter_per_rank(bundle))


if __name__ == "__main__":
    unittest.main()
