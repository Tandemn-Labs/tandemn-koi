import unittest

from src.core.models import Plan
from src.executor.executor import Executor, StorePlanExecutor


class _PlanStore:
    def create(self, plan):
        self.plan = plan
        return plan


class StorePlanExecutorSmokeTests(unittest.TestCase):
    def test_base_executor_requires_subclass(self):
        with self.assertRaises(NotImplementedError):
            Executor().send_to_executor({})

    def test_writes_store_plan_and_preserves_rank_id(self):
        store = _PlanStore()
        plan = Plan.from_raw(_raw_place_plan(), tick=7)

        ack = StorePlanExecutor("user_1", plan_store=store).send_to_executor(plan)

        self.assertEqual(ack, [{"plan_id": store.plan.plan_id, "status": "created"}])
        self.assertEqual(store.plan.user_id, "user_1")
        self.assertEqual(store.plan.tick_rationale, "place one rank")
        self.assertEqual(
            store.plan.actions[0].ladder[0]["rank_id"], "rank_01K00000000000000000000000"
        )
        self.assertEqual(store.plan.actions[0].ladder[0]["predicted_y"], {"p99_ttft_ms": 120.0})
        self.assertEqual(store.plan.actions[0].ladder[0]["predicted_v"], {"kv_cache_util": 0.4})

    def test_rejects_raw_plan_without_rank_id(self):
        store = _PlanStore()
        raw = _raw_place_plan()
        raw["actions"][0]["ladder"][0].pop("rank_id")
        with self.assertRaisesRegex(ValueError, "executor requires rank_<ULID>"):
            StorePlanExecutor("user_1", plan_store=store).send_to_executor(raw)

    def test_rank_ulid_timestamp_boundary_matches_orca(self):
        for first in ("0", "7"):
            raw = _raw_place_plan()
            raw["actions"][0]["ladder"][0]["rank_id"] = f"rank_{first}{'Z' * 25}"
            StorePlanExecutor("user_1", plan_store=_PlanStore()).send_to_executor(raw)

        raw = _raw_place_plan()
        raw["actions"][0]["ladder"][0]["rank_id"] = f"rank_8{'0' * 25}"
        with self.assertRaisesRegex(ValueError, "executor requires rank_<ULID>"):
            StorePlanExecutor("user_1", plan_store=_PlanStore()).send_to_executor(raw)


def _raw_place_plan():
    return {
        "tick_rationale": "place one rank",
        "actions": [
            {
                "job_id": "job_1",
                "type": "place",
                "target_tps": 10.0,
                "target_p99_ttft_ms": 500.0,
                "target_p99_tpot_ms": 50.0,
                "ladder": [
                    {
                        "role": "aggregate",
                        "rank_id": "rank_01K00000000000000000000000",
                        "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                        "config": {"gpu_count": 1},
                        "n_replicas": 1,
                        "predicted_y": {"p99_ttft_ms": 120.0},
                        "predicted_v": {"kv_cache_util": 0.4},
                    }
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
