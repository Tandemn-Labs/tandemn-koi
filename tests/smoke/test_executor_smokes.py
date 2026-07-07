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
        self.assertEqual(store.plan.actions[0].ladder[0]["rank_id"], "rank_0")
        self.assertEqual(store.plan.actions[0].ladder[0]["predicted_y"], {"p99_ttft_ms": 120.0})
        self.assertEqual(store.plan.actions[0].ladder[0]["predicted_v"], {"kv_cache_util": 0.4})

    def test_accepts_raw_plan_input(self):
        store = _PlanStore()

        StorePlanExecutor("user_1", plan_store=store).send_to_executor(_raw_place_plan())

        self.assertEqual(store.plan.actions[0].job_id, "job_1")
        self.assertEqual(store.plan.actions[0].ladder[0]["rank_id"], "rank_0")


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
