import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from src.core.candidate_graph import CandidateGraph
from src.core.models import Node
from src.infra.resource_map import ClusterResourceSnapshot, ResourceMapManager
from src.infra.telemetry import StoreTelemetry


def _graph():
    nodes = {
        "kv_cache_util": Node("kv_cache_util", "V"),
        "kvcache_hit_rate": Node("kvcache_hit_rate", "V"),
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
                    {"chain_id": "rank_0_chain_0", "shape_json": dict(shape)},
                    {"chain_id": "rank_0_chain_1", "shape_json": dict(shape)},
                ],
            }
        ],
        pending_jobs=[],
    )


def _row(chain_index, rank_id="rank_0", job_id="job_1", ts=None, **metrics):
    return SimpleNamespace(
        ts=ts or datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        job_id=job_id,
        rank_id=rank_id,
        chain_index=chain_index,
        **metrics,
    )


class _Store:
    def __init__(self, rows):
        self.rows = rows

    def rows_for_job_window(self, user_id, job_id, start, end):
        return list(self.rows)


class StoreTelemetrySmokeTests(unittest.TestCase):
    def test_tp2_gpu_rows_form_one_multi_sample_rank_trajectory(self):
        snapshot = ClusterResourceSnapshot(
            tick=1,
            resources={},
            active_jobs=[
                {
                    "job_id": "job_1",
                    "job_features": {"type": "online"},
                    "active_chains": [
                        {
                            "chain_id": "rank_0_chain_0",
                            "shape_json": {
                                "rank_id": "rank_0",
                                "mechanism_id": "mech_1",
                                "model_id": "m",
                            },
                        }
                    ],
                }
            ],
            pending_jobs=[],
        )
        first = datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC)
        second = datetime(2026, 1, 1, 0, 0, 20, tzinfo=UTC)
        telemetry = StoreTelemetry(
            user_id="user_1",
            gpu_metric_store=_Store(
                [
                    _row(
                        0,
                        ts=first,
                        local_rank="0",
                        throughput_token_per_sec=10.0,
                        kv_cache_util=0.2,
                        kvcache_hit_rate=0.25,
                    ),
                    _row(
                        0,
                        ts=first,
                        local_rank="1",
                        throughput_token_per_sec=10.0,
                        kv_cache_util=0.2,
                        kvcache_hit_rate=0.25,
                    ),
                    _row(
                        0,
                        ts=second,
                        local_rank="0",
                        throughput_token_per_sec=20.0,
                        kv_cache_util=0.4,
                        kvcache_hit_rate=0.75,
                    ),
                    _row(
                        0,
                        ts=second,
                        local_rank="1",
                        throughput_token_per_sec=20.0,
                        kv_cache_util=0.4,
                        kvcache_hit_rate=0.75,
                    ),
                ]
            ),
            candidate_graph=_graph(),
            now_fn=lambda: datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC),
        )

        rank = next(telemetry.iter_per_rank(telemetry.collect_telemetry(0, 1, snapshot)))

        self.assertEqual(rank.y_observed["throughput_token_per_sec"].tolist(), [10.0, 20.0])
        self.assertEqual(rank.v_observed["kv_cache_util"].tolist(), [0.2, 0.4])
        self.assertEqual(rank.v_observed["kvcache_hit_rate"].tolist(), [0.25, 0.75])

    def test_aggregates_rank_trajectories_without_gpu_double_counting(self):
        telemetry = StoreTelemetry(
            user_id="user_1",
            gpu_metric_store=_Store(
                [
                    _row(
                        0,
                        throughput_token_per_sec=10.0,
                        kv_cache_util=0.2,
                        p99_ttft_ms=100.0,
                        slo_margin=5.0,
                        cost_per_token=0.01,
                        depth_req_q=999.0,
                    ),
                    _row(
                        0,
                        throughput_token_per_sec=10.0,
                        kv_cache_util=0.4,
                        p99_ttft_ms=100.0,
                        slo_margin=5.0,
                        cost_per_token=0.01,
                    ),
                    _row(
                        1,
                        throughput_token_per_sec=20.0,
                        kv_cache_util=0.9,
                        p99_ttft_ms=120.0,
                        slo_margin=2.0,
                        cost_per_token=0.02,
                    ),
                    _row(99, throughput_token_per_sec=999.0),
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

    def test_job_mismatch_is_contract_error(self):
        telemetry = StoreTelemetry(
            user_id="user_1",
            gpu_metric_store=_Store([_row(0, job_id="job_bad", kv_cache_util=0.2)]),
            candidate_graph=_graph(),
            now_fn=lambda: datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC),
        )

        bundle = telemetry.collect_telemetry(0, 1, _snapshot())
        with self.assertRaises(ValueError):
            list(telemetry.iter_per_rank(bundle))

    def test_rank_replicas_and_metric_indexes_share_chain_ids(self):
        class Model:
            def __init__(self, **values):
                self.values = values

            def model_dump(self, mode="json"):
                return self.values

        running_job = SimpleNamespace(
            job=Model(
                job_id="job_1",
                user_id="user_1",
                kind="inference",
                status="running",
                created_at=None,
                finished_at=None,
                finish_reason=None,
                spec_json={},
            ),
            ranks=[
                Model(
                    rank_id="rank_0",
                    plan_id="plan_1",
                    role="aggregate",
                    status="running",
                    shape_json={"mechanism_id": "mech_1"},
                    n_replicas=2,
                )
            ],
        )

        job = ResourceMapManager._running_job_to_summary(running_job)
        self.assertEqual(
            [chain["chain_id"] for chain in job["active_chains"]],
            ["rank_0_chain_0", "rank_0_chain_1"],
        )
        self.assertTrue(
            all(chain["shape_json"]["rank_id"] == "rank_0" for chain in job["active_chains"])
        )

        telemetry = StoreTelemetry(
            user_id="user_1",
            gpu_metric_store=_Store(
                [
                    _row(0, throughput_token_per_sec=10.0),
                    _row(1, throughput_token_per_sec=20.0),
                ]
            ),
            candidate_graph=_graph(),
            now_fn=lambda: datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC),
        )
        snapshot = ClusterResourceSnapshot(1, {}, [job], [])
        rank = next(telemetry.iter_per_rank(telemetry.collect_telemetry(0, 1, snapshot)))
        self.assertEqual(rank.y_observed["throughput_token_per_sec"].tolist(), [30.0])


if __name__ == "__main__":
    unittest.main()
