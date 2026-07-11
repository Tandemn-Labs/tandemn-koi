import unittest

from src.agent.agent import KoiAgentHarness, PlanMaterializationError, SpecialistRunner
from src.agent.tools import agent_tools
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Mechanism, PlanAction, RankSpec

ENV = "reserved|aws|us-east-1|us-east-1b|L40S"


def _slice():
    return {"user_id": "usr_test", "env_budget": {ENV: 4}}


def _valid_place():
    return {
        "job_id": "job_1",
        "user_id": "usr_test",
        "type": "place",
        "ladder": [
            {
                "role": "aggregate",
                "env": ENV.split("|"),
                "config": {
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "instance_type": "g6e.xlarge",
                    "gpu_type": "L40S",
                    "gpu_count": 1,
                    "tp": 1,
                    "pp": 1,
                    "engine_name": "vllm",
                },
                "n_replicas": 4,
                "mechanism_id": "M_demo",
            }
        ],
        "predicted_y": {},
        "predicted_sigma": 0.0,
        "budget_utilization": {ENV: 4},
        "used_capacity": {ENV: 4},
        "fitness": "happy",
        "marginal_value_of_more": {},
        "unused_capacity": {},
        "mechanism_ids": ["M_demo"],
        "new_mechanism_proposals": [],
        "reasoning": "canonical placement",
    }


class SpecialistSchemaSmokeTests(unittest.TestCase):
    def test_unknown_mechanism_fails_before_scoring(self):
        class Registry:
            def get_mechanism(self, mechanism_id):
                raise KeyError(mechanism_id)

        saved = agent_tools._CTX.mechanism_registry
        agent_tools._CTX.mechanism_registry = Registry()
        action = PlanAction.from_dict(_valid_place())
        try:
            with self.assertRaisesRegex(PlanMaterializationError, "unknown mechanism_id"):
                KoiAgentHarness.__new__(KoiAgentHarness)._validate_ladder(action, None)
        finally:
            agent_tools._CTX.mechanism_registry = saved

    def test_rank_spec_strips_engine_knobs_from_both_forms(self):
        forbidden = {
            "max_num_seq": 1,
            "max_num_batched_tokens": 2,
            "block_size": 3,
        }
        explicit = RankSpec.from_dict({"role": "aggregate", "config": {"tp": 1, **forbidden}})
        shorthand = RankSpec.from_dict({"aggregate": {"tp": 1, **forbidden}})

        self.assertEqual(explicit.config, {"tp": 1})
        self.assertEqual(shorthand.config, {"tp": 1})

    def test_ladder_rejects_false_scope_but_accepts_partial(self):
        registry = MechanismRegistry()
        burst_id = registry.add_mechanism(
            Mechanism(
                edge_ids=["peak_to_mean_ratio->depth_req_q"],
                scope={
                    "x": ["peak_to_mean_ratio"],
                    "v": ["depth_req_q"],
                    "workload_type": "online",
                    "conditions": [{"feature": "peak_to_mean_ratio", "op": ">", "value": 2}],
                },
                narrative="Bursts build queues.",
            )
        )
        partial_id = registry.add_mechanism(
            Mechanism(
                edge_ids=["tp->comm_overhead_pct"],
                scope={
                    "x": ["tp", "unknown_knob"],
                    "v": ["comm_overhead_pct"],
                    "workload_type": "online",
                    "conditions": [{"feature": "unknown_knob", "op": ">", "value": 0}],
                },
                narrative="Partially known communication mechanism.",
            )
        )

        class Snapshot:
            @staticmethod
            def pending_jobs_summary():
                return [
                    {
                        "job_id": "job_1",
                        "job_features": {"type": "online", "peak_to_mean_ratio": 2},
                    }
                ]

        saved = (agent_tools._CTX.mechanism_registry, agent_tools._CTX.resource_map)
        harness = KoiAgentHarness.__new__(KoiAgentHarness)
        try:
            agent_tools._CTX.mechanism_registry = registry
            agent_tools._CTX.resource_map = None

            false_scope = _valid_place()
            false_scope["ladder"][0]["mechanism_id"] = burst_id
            with self.assertRaisesRegex(PlanMaterializationError, "does not apply"):
                harness._validate_ladder(PlanAction.from_dict(false_scope), None, Snapshot())

            partial = _valid_place()
            partial["ladder"][0]["mechanism_id"] = partial_id
            harness._validate_ladder(PlanAction.from_dict(partial), None, Snapshot())
        finally:
            agent_tools._CTX.mechanism_registry, agent_tools._CTX.resource_map = saved

    def test_valid_canonical_place_passes(self):
        self.assertEqual(SpecialistRunner._validate(_valid_place(), "job_1", _slice()), [])

    def test_shorthand_ladder_fails(self):
        result = _valid_place()
        result["ladder"] = [{"env": ENV, "count": 4}]
        violations = SpecialistRunner._validate(result, "job_1", _slice())
        self.assertIn("ladder[0].role must be 'aggregate'", violations)
        self.assertIn("ladder[0].config must be a dict", violations)

    def test_four_part_env_fails(self):
        result = _valid_place()
        result["used_capacity"] = {"aws|us-east-1|us-east-1b|L40S": 4}
        result["ladder"][0]["env"] = ["aws", "us-east-1", "us-east-1b", "L40S"]
        violations = SpecialistRunner._validate(result, "job_1", _slice())
        self.assertTrue(any("used_capacity env" in violation for violation in violations))
        self.assertTrue(any("ladder[0].env" in violation for violation in violations))

    def test_defer_must_not_include_ladder(self):
        result = _valid_place()
        result["type"] = "defer"
        violations = SpecialistRunner._validate(result, "job_1", _slice())
        self.assertIn("defer must not include ladder", violations)


if __name__ == "__main__":
    unittest.main()
