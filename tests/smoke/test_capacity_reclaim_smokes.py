"""Capacity-reclaim smokes: negative-gain shrink swaps are admitted at the
work-conserving floor (at most one per tick), never displace real gains, and
the joint solver's stranded-slot tie-break prefers tighter packing."""

import math
import unittest
from unittest.mock import patch

import src.agent.tools.agent_tools as agent_tools

ENV = "reserved|aws|r1|z1|H100"


def _rank(instance_type: str, gpu_count: int = 1, n_replicas: int = 1) -> dict:
    return {
        "role": "aggregate",
        "env": ENV.split("|"),
        "config": {
            "instance_type": instance_type,
            "gpu_count": gpu_count,
            "tp": gpu_count,
            "pp": 1,
        },
        "n_replicas": n_replicas,
    }


def _assessment() -> dict:
    return {
        "basis": "aic_direct_point",
        "kind": "point",
        "status": "success",
        "queue_slo_verified": False,
    }


def _reclaim(job_id: str, gain: float) -> dict:
    return {
        "job_id": job_id,
        "type": "swap",
        "capacity_reclaim": True,
        "rehabilitation_status": None,
        "ladder": [_rank("p5")],
        "target_tps": 100.0,
        "achieved_tps": 80.0,
        "served_fraction": 0.8,
        "sigma": -1.0,
        "keep_baseline_sigma": -1.0 - gain,
        "swap_gain_over_keep": gain,
        "queue_state": "stable",
        "prediction_assessment": _assessment(),
    }


def _joint(candidates, free_instances=8, free_gpus=8, swap_budget=2):
    resources = {ENV: {"free": free_gpus, "gpu_type": "H100"}}
    specs = {ENV: {"p5": {"gpus_per_instance": 8, "free_instances": free_instances}}}
    slow_loop = type(
        "SlowLoop", (), {"get_sss_swap_budget_t": lambda self: swap_budget}
    )()
    with (
        patch.object(agent_tools._CTX, "resource_map", object()),
        patch.object(agent_tools._CTX, "slow_loop", slow_loop),
        patch.object(agent_tools, "get_resource_map", return_value=resources),
        patch.object(agent_tools, "instance_catalog", return_value=specs),
        patch.object(agent_tools, "get_pending_jobs", return_value=[]),
        patch.object(agent_tools, "get_priority", return_value=[]),
    ):
        return agent_tools.jointly_select_placements(candidates)


class CapacityReclaimJointSmokeTests(unittest.TestCase):
    def test_negative_gain_reclaim_swap_is_admitted_at_the_floor(self):
        result = _joint([_reclaim("busy", -2.0)])
        chosen = [c for c in result["chosen"] if c.get("job_id") == "busy"]
        self.assertEqual(len(chosen), 1)
        self.assertTrue(chosen[0].get("work_conserving_floor"))
        self.assertEqual(chosen[0]["service_class"], "partial")
        self.assertEqual(
            chosen[0]["prediction_assessment"].get("selection_mode"), "capacity_reclaim"
        )

    def test_at_most_one_reclaim_swap_per_tick_and_least_loss_wins(self):
        result = _joint([_reclaim("a", -5.0), _reclaim("b", -2.0)])
        reclaims = [c for c in result["chosen"] if c.get("capacity_reclaim")]
        self.assertEqual(len(reclaims), 1)
        self.assertEqual(reclaims[0]["job_id"], "b")

    def test_reclaim_never_displaces_a_positive_gain_placement(self):
        place = {
            "job_id": "waiting",
            "type": "place",
            "ladder": [_rank("p5", gpu_count=8)],
            "target_tps": 100.0,
            "achieved_tps": 100.0,
            "served_fraction": 1.0,
            "sigma": 5.0,
            "queue_state": "stable",
            "prediction_assessment": _assessment(),
        }
        result = _joint([place, _reclaim("busy", -2.0)], free_instances=1, free_gpus=8)
        chosen_ids = [c.get("job_id") for c in result["chosen"]]
        self.assertIn("waiting", chosen_ids)
        self.assertNotIn("busy", chosen_ids)


class StrandedSlotSmokeTests(unittest.TestCase):
    def test_ladder_stranded_gpus_counts_unused_box_slots(self):
        specs = {ENV: {"p5": {"gpus_per_instance": 8, "free_instances": 4}}}
        ladder = [_rank("p5", gpu_count=1, n_replicas=2)]
        self.assertEqual(agent_tools._ladder_stranded_gpus(ladder, specs), 14)
        full = [_rank("p5", gpu_count=8, n_replicas=3)]
        self.assertEqual(agent_tools._ladder_stranded_gpus(full, specs), 0)
        self.assertEqual(agent_tools._ladder_stranded_gpus(None, specs), 0)

    def test_equal_gain_candidates_resolve_to_the_tighter_packing(self):
        loose = {
            "job_id": "j",
            "type": "place",
            "marker": "loose",
            "ladder": [_rank("p5", gpu_count=1)],
            "target_tps": 100.0,
            "achieved_tps": 100.0,
            "served_fraction": 1.0,
            "sigma": 5.0,
            "queue_state": "stable",
            "prediction_assessment": _assessment(),
        }
        tight = dict(loose, marker="tight", ladder=[_rank("p5", gpu_count=8)])
        result = _joint([loose, tight])
        chosen = [c for c in result["chosen"] if c.get("job_id") == "j"]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["marker"], "tight")


class ReclaimFullFleetSmokeTests(unittest.TestCase):
    """Run-16 regression: on a fully committed fleet (0 free instances, 0 free
    GPUs, and even a spent swap budget) a reclaim shrink must still be chosen -
    it occupies a strict subset of GPUs its job already holds."""

    def test_reclaim_swap_survives_a_fully_committed_fleet(self):
        result = _joint([_reclaim("busy", -2.0)], free_instances=0, free_gpus=0)
        chosen = [c for c in result["chosen"] if c.get("job_id") == "busy"]
        self.assertEqual(len(chosen), 1)
        self.assertTrue(chosen[0].get("capacity_reclaim"))

    def test_reclaim_swap_is_exempt_from_the_churn_swap_budget(self):
        result = _joint(
            [_reclaim("busy", -2.0)], free_instances=0, free_gpus=0, swap_budget=0
        )
        self.assertEqual(
            [c.get("job_id") for c in result["chosen"] if c.get("capacity_reclaim")],
            ["busy"],
        )


class ReclaimRankReconstructionSmokeTests(unittest.TestCase):
    """_reclaim_current_rank must invert both ladder encodings: the real
    executor's one-chain-per-replica rows and sim-style n_replicas ranks."""

    def _chain(self, rank_id: str = "r1", status: str = "running",
               instance: str = "p5", tp: int = 8) -> dict:
        return {
            "chain_id": f"{rank_id}_chain_x",
            "chain_status": status,
            "shape_json": {
                "rank_id": rank_id,
                "env": ENV.split("|"),
                "instance_type": instance,
                "gpu_count": tp,
                "count": tp,
                "tp": tp,
                "pp": 1,
            },
        }

    def test_chain_rows_reconstruct_dp_from_multiplicity(self):
        current = agent_tools._reclaim_current_rank([self._chain(), self._chain()])
        self.assertIsNotNone(current)
        self.assertEqual(current["n_replicas"], 2)
        self.assertEqual(current["config"]["instance_type"], "p5")
        self.assertEqual(current["config"]["tp"], 8)

    def test_rank_style_ladder_keeps_explicit_n_replicas(self):
        current = agent_tools._reclaim_current_rank(
            [_rank("p5", gpu_count=8, n_replicas=3)]
        )
        self.assertIsNotNone(current)
        self.assertEqual(current["n_replicas"], 3)

    def test_single_replica_multi_rank_and_launching_are_not_subjects(self):
        single = agent_tools._reclaim_current_rank([self._chain()])
        self.assertEqual(single["n_replicas"], 1)  # caller rejects < 2
        two_ranks = [self._chain("r1"), self._chain("r2", instance="p4", tp=4)]
        self.assertIsNone(agent_tools._reclaim_current_rank(two_ranks))
        launching = [self._chain(), self._chain(status="launching")]
        self.assertIsNone(agent_tools._reclaim_current_rank(launching))
        self.assertIsNone(agent_tools._reclaim_current_rank([]))


if __name__ == "__main__":
    unittest.main()
