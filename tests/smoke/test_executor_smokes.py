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
        self.assertTrue(store.plan.actions[0].ladder[0]["rank_id"].startswith("rank_"))
        self.assertEqual(store.plan.actions[0].ladder[0]["predicted_y"], {"p99_ttft_ms": 120.0})
        self.assertEqual(store.plan.actions[0].ladder[0]["predicted_v"], {"kv_cache_util": 0.4})
        self.assertEqual(
            store.plan.actions[0].ladder[0]["prediction_lineage"]["deployment_id"],
            "deploy-0",
        )
        self.assertEqual(store.plan.actions[0].target_tps, 10.0)

    def test_partial_plan_sends_admitted_target_and_rank_share(self):
        store = _PlanStore()
        raw = _raw_place_plan()
        action = raw["actions"][0]
        action.update(
            {
                "target_tps": 100.0,
                "admitted_tps": 60.0,
                "achieved_tps": 60.0,
                "unmet_tps": 40.0,
                "meets_target": False,
                "served_fraction": 0.6,
                "admission_mode": "advisory",
            }
        )
        action["ladder"][0]["rank_traffic_share"] = 0.6

        plan = Plan.from_raw(raw, tick=7)
        StorePlanExecutor("user_1", plan_store=store).send_to_executor(plan)

        stored_action = store.plan.actions[0]
        self.assertEqual(plan.actions[0].target_tps, 100.0)
        self.assertEqual(stored_action.target_tps, 60.0)
        self.assertEqual(stored_action.ladder[0]["rank_traffic_share"], 0.6)

    def test_full_admission_metadata_keeps_required_target(self):
        store = _PlanStore()
        raw = _raw_place_plan()
        raw["actions"][0].update(
            {
                "admitted_tps": 10.0,
                "achieved_tps": 10.0,
                "unmet_tps": 0.0,
                "meets_target": True,
                "served_fraction": 1.0,
                "admission_mode": "enforced",
            }
        )

        StorePlanExecutor("user_1", plan_store=store).send_to_executor(Plan.from_raw(raw, tick=7))

        self.assertEqual(store.plan.actions[0].target_tps, 10.0)

    def test_accepts_raw_plan_input(self):
        store = _PlanStore()

        StorePlanExecutor("user_1", plan_store=store).send_to_executor(_raw_place_plan())

        self.assertEqual(store.plan.actions[0].job_id, "job_1")
        self.assertTrue(store.plan.actions[0].ladder[0]["rank_id"].startswith("rank_"))


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
                        "prediction_lineage": {
                            "schema_version": 3,
                            "deployment_id": "deploy-0",
                        },
                    }
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
