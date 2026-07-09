import unittest

from src.agent.agent import SpecialistRunner

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
