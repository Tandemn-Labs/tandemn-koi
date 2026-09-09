"""Footprint-cap and growth-swap smokes: the per-job GPU cap math, fleet
capacity accounting, composite trimming, the observe-side traffic-share
fallback, predicted-throughput baselines, and the joint solver's delta
charge for growth swaps on a fully committed fleet."""

import unittest
from unittest.mock import patch

import src.agent.tools.agent_tools as agent_tools
from src.infra import deployment_x

ENV = "reserved|aws|r1|z1|H100"


class PerJobGpuCapSmokeTests(unittest.TestCase):
    def test_divides_free_capacity_among_waiting_jobs(self):
        # koi_debug_run_v4 replay: 96 free GPUs, 3 waiting, 104-GPU fleet.
        self.assertEqual(agent_tools._per_job_gpu_cap(96, 3, 104), 32)

    def test_singleton_job_hits_the_fleet_fraction_ceiling(self):
        self.assertEqual(agent_tools._per_job_gpu_cap(96, 1, 104), 37)

    def test_no_free_capacity_returns_the_ceiling_for_growth_bounds(self):
        self.assertEqual(agent_tools._per_job_gpu_cap(0, 5, 104), 37)

    def test_floor_is_one_gpu(self):
        self.assertEqual(agent_tools._per_job_gpu_cap(5, 10, 104), 1)


class FleetGpuCapacitySmokeTests(unittest.TestCase):
    def test_totals_from_merged_pool_sources_with_free_fallback(self):
        resources = {
            "reserved|gcp|r|z|H100": {
                "free": 24,
                "pools": [
                    {"free": 24, "merged_pool_sources": [{"total": 24}]},
                ],
            },
            "reserved|gcp|r|z|A100": {
                "free": 8,
                "pools": [{"free": 8}],  # no sources -> contributes free
            },
        }
        self.assertEqual(agent_tools._fleet_gpu_capacity(resources), (32, 32))

    def test_empty_resources(self):
        self.assertEqual(agent_tools._fleet_gpu_capacity({}), (0, 0))


class CapCompositeRanksSmokeTests(unittest.TestCase):
    def _rank(self, gpus: int, replicas: int) -> dict:
        return {"config": {"gpu_count": gpus}, "n_replicas": replicas}

    def test_first_rank_is_shrunk_to_fit_the_cap(self):
        ranks = [self._rank(8, 5), self._rank(8, 2), self._rank(4, 2)]
        capped = agent_tools._cap_composite_ranks(ranks, 32)
        self.assertEqual([(r["config"]["gpu_count"], r["n_replicas"]) for r in capped], [(8, 4)])

    def test_later_ranks_fill_remaining_room(self):
        ranks = [self._rank(8, 5), self._rank(8, 2), self._rank(4, 2)]
        capped = agent_tools._cap_composite_ranks(ranks, 48)
        self.assertEqual(
            [(r["config"]["gpu_count"], r["n_replicas"]) for r in capped],
            [(8, 5), (8, 1)],
        )

    def test_inputs_are_not_mutated(self):
        ranks = [self._rank(8, 5)]
        agent_tools._cap_composite_ranks(ranks, 16)
        self.assertEqual(ranks[0]["n_replicas"], 5)


class RankTrafficShareFallbackSmokeTests(unittest.TestCase):
    def test_explicit_share_is_honored(self):
        self.assertAlmostEqual(
            deployment_x._rank_traffic_share({"rank_traffic_share": 0.3}, 2, 11), 0.3
        )

    def test_single_rank_defaults_to_full_share(self):
        self.assertEqual(deployment_x._rank_traffic_share({}, 3, 3), 1.0)

    def test_multi_rank_missing_share_falls_back_to_replica_proportion(self):
        # koi_debug_run_v4 regression: this raised and wedged S1_OBSERVE.
        self.assertAlmostEqual(deployment_x._rank_traffic_share({}, 2, 11), 2 / 11)

    def test_degenerate_counts_fall_back_to_full_share(self):
        self.assertEqual(deployment_x._rank_traffic_share({}, 0, 11), 1.0)


class _FakeSnapshot:
    def __init__(self, jobs):
        self._jobs = jobs

    def active_jobs_summary(self):
        return self._jobs


class PredictedActiveTpsSmokeTests(unittest.TestCase):
    def _chain(self, tps):
        return {
            "chain_id": "r1_chain_x",
            "shape_json": {"rank_id": "r1", "predicted_y": {"throughput_token_per_sec": tps}},
        }

    def test_sums_per_chain_predicted_throughput(self):
        snap = _FakeSnapshot(
            [{"job_id": "j1", "current_ladder": [self._chain(100.0), self._chain(150.0)]}]
        )
        self.assertAlmostEqual(agent_tools._predicted_active_tps(snap, "j1"), 250.0)

    def test_missing_predictions_return_none(self):
        snap = _FakeSnapshot([{"job_id": "j1", "current_ladder": [{"shape_json": {}}]}])
        self.assertIsNone(agent_tools._predicted_active_tps(snap, "j1"))
        self.assertIsNone(agent_tools._predicted_active_tps(None, "j1"))


def _chain_row(rank_id: str = "r1", tp: int = 8) -> dict:
    return {
        "chain_id": f"{rank_id}_chain_x",
        "chain_status": "running",
        "shape_json": {
            "rank_id": rank_id,
            "env": ENV.split("|"),
            "instance_type": "p5",
            "gpu_count": tp,
            "count": tp,
            "tp": tp,
            "pp": 1,
        },
    }


def _growth_swap(job_id: str, replicas: int, tagged: bool = True) -> dict:
    cand = {
        "job_id": job_id,
        "type": "swap",
        "rehabilitation_status": None,
        "ladder": [
            {
                "role": "aggregate",
                "env": ENV.split("|"),
                "config": {"instance_type": "p5", "gpu_count": 8, "tp": 8, "pp": 1},
                "n_replicas": replicas,
            }
        ],
        "target_tps": 100.0,
        "achieved_tps": 90.0,
        "served_fraction": 0.9,
        "sigma": -1.0,
        "keep_baseline_sigma": -3.0,
        "swap_gain_over_keep": 2.0,
        "queue_state": "stable",
        "prediction_assessment": {
            "basis": "aic_direct_point",
            "kind": "point",
            "status": "success",
            "queue_slo_verified": False,
        },
    }
    if tagged:
        cand["growth_swap"] = True
    return cand


def _joint(candidates, free_instances=0, free_gpus=0, actives=()):
    resources = {ENV: {"free": free_gpus, "gpu_type": "H100"}}
    specs = {ENV: {"p5": {"gpus_per_instance": 8, "free_instances": free_instances}}}
    slow_loop = type("SlowLoop", (), {"get_sss_swap_budget_t": lambda self: 2})()
    with (
        patch.object(agent_tools._CTX, "resource_map", object()),
        patch.object(agent_tools._CTX, "slow_loop", slow_loop),
        patch.object(agent_tools, "get_resource_map", return_value=resources),
        patch.object(agent_tools, "instance_catalog", return_value=specs),
        patch.object(agent_tools, "get_pending_jobs", return_value=[]),
        patch.object(agent_tools, "get_priority", return_value=[]),
        patch.object(agent_tools, "get_active_jobs", return_value=list(actives)),
    ):
        return agent_tools.jointly_select_placements(candidates)


class GrowthSwapJointSmokeTests(unittest.TestCase):
    """A growth swap is charged only the delta above the job's current
    deployment, so growing 2 -> 3 replicas needs one free instance, not
    three."""

    def _active(self):
        return {
            "job_id": "busy",
            "status": "running",
            "current_ladder": [_chain_row(), _chain_row()],
        }

    def test_growth_swap_charged_only_the_delta(self):
        result = _joint(
            [_growth_swap("busy", 3)], free_instances=1, free_gpus=8, actives=[self._active()]
        )
        chosen = [c for c in result["chosen"] if c.get("job_id") == "busy"]
        self.assertEqual(len(chosen), 1)
        self.assertTrue(chosen[0].get("growth_swap"))

    def test_untagged_swap_pays_full_footprint_and_is_dropped(self):
        result = _joint(
            [_growth_swap("busy", 3, tagged=False)],
            free_instances=1,
            free_gpus=8,
            actives=[self._active()],
        )
        self.assertEqual([c.get("job_id") for c in result["chosen"]], [])


if __name__ == "__main__":
    unittest.main()
