import unittest

from src.agent.tools import agent_tools


class _DRO:
    def compute_dro_band(self, y_hat):
        return {}


class _ResourceMap:
    def snapshot(self):
        return _Snapshot()

    def resources_summary(self):
        return {
            "reserved|aws|us-east-1|use1-az1|H100": {
                "free": 8,
                "gpu_type": "H100",
            }
        }

    def rank_allocation_summary(self, rank, resources=None):
        gpus = rank.gpus_per_chain()
        return {
            "allocation_kind": "gpu",
            "instance_type": rank.config.get("instance_type"),
            "gpus_per_unit": gpus,
            "price_per_unit_hour": None,
            "capacity_per_replica": gpus,
        }

    def model_catalog(self, model_id):
        return {"model_id": model_id, "model_params_b": 70.0, "hidden_size": 8192}


class _EvidenceStore:
    def get_rows_for_job(self, job_id):
        return []

    def retrieve_similar_rows(self, job_features, top_k=10):
        return []


class _MechanismRegistry:
    def filter_by_scope(self, subset_x, subset_v):
        return []


class _ConfidenceService:
    pass


class _RecordingSurrogate:
    def __init__(self):
        self.calls = []

    def compose_prediction(
        self, job_config, job_features, candidate_graph, method=("AIC_DynoSim",)
    ):
        self.calls.append((dict(job_config), dict(job_features)))
        return (
            {
                "p99_ttft_ms": 10.0,
                "p99_tpot_ms": 1.0,
                "throughput_token_per_sec": 1000.0,
            },
            {"kv_cache_util": 0.4},
        )


class AgentToolsSmokeTests(unittest.TestCase):
    def test_size_ladder_threads_rank_env_and_job_model_to_surrogate(self):
        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in ("resource_map", "surrogate", "candidate_graph", "dro")
        }
        surrogate = _RecordingSurrogate()
        try:
            agent_tools.bind_tools(
                resource_map=_ResourceMap(),
                surrogate=surrogate,
                candidate_graph=object(),
                dro=_DRO(),
            )
            result = agent_tools.size_ladder(
                ranks=[
                    {
                        "role": "aggregate",
                        "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                        "config": {"instance_type": "p5.48xlarge", "tp": 1, "pp": 1},
                    }
                ],
                job_features={
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "type": "online",
                    "request_arrival_rate": 1.0,
                    "output_len_tokens_avg": 100.0,
                    "headroom_factor": 1.0,
                    "target_p99_ttft_ms": 100.0,
                    "target_p99_tpot_ms": 10.0,
                },
            )

            self.assertTrue(result["meets_target"])
            job_config, job_features = surrogate.calls[0]
            self.assertEqual(job_config["model_id"], "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(job_features["gpu_type"], "H100")
            self.assertEqual(job_features["market"], "reserved")
            self.assertEqual(job_features["cloud"], "aws")
            self.assertEqual(job_features["region"], "us-east-1")
            self.assertEqual(job_features["zone"], "use1-az1")
            self.assertEqual(job_features["instance_type"], "p5.48xlarge")
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

    def test_get_job_brief_includes_model_catalog(self):
        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in (
                "resource_map",
                "evidence_store",
                "mechanism_registry",
                "confidence_service",
            )
        }
        try:
            agent_tools.bind_tools(
                resource_map=_ResourceMap(),
                evidence_store=_EvidenceStore(),
                mechanism_registry=_MechanismRegistry(),
                confidence_service=_ConfidenceService(),
            )
            brief = agent_tools.get_job_brief("job_1")

            self.assertEqual(brief["job_features"]["model_id"], "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(brief["model_catalog"]["model_params_b"], 70.0)
            self.assertNotIn("model_params_b", brief["job_features"])
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)

    def test_stamp_plan_predictions_writes_raw_rank_predictions(self):
        saved = {
            name: getattr(agent_tools._CTX, name)
            for name in ("surrogate", "candidate_graph", "dro")
        }
        try:
            agent_tools.bind_tools(
                surrogate=_RecordingSurrogate(),
                candidate_graph=object(),
                dro=_DRO(),
            )
            snapshot = _Snapshot()
            plan = agent_tools.stamp_plan_predictions(
                {
                    "actions": [
                        {
                            "job_id": "job_1",
                            "type": "place",
                            "ladder": [
                                {
                                    "role": "aggregate",
                                    "env": ["reserved", "aws", "us-east-1", "use1-az1", "H100"],
                                    "config": {"instance_type": "p5.48xlarge", "gpu_count": 1},
                                    "n_replicas": 1,
                                }
                            ],
                        }
                    ]
                },
                snapshot,
            )

            rank = plan.actions[0].ladder[0]
            self.assertEqual(rank.predicted_y["p99_ttft_ms"], 10.0)
            self.assertEqual(rank.predicted_v, {"kv_cache_util": 0.4})
        finally:
            for name, value in saved.items():
                setattr(agent_tools._CTX, name, value)


class _Snapshot:
    def pending_jobs_summary(self):
        return [
            {
                "job_id": "job_1",
                "job_features": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
            }
        ]


if __name__ == "__main__":
    unittest.main()
