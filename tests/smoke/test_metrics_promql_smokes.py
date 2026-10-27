"""Evaluate profile aggregation rules with promtool in a local container."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parents[2] / "src/infra/metrics/profiles"
IMAGE = os.environ.get("KOI_TEST_PROMETHEUS_IMAGE")


@unittest.skipUnless(IMAGE, "set KOI_TEST_PROMETHEUS_IMAGE to run local promtool checks")
class MetricsPromQLSmokeTests(unittest.TestCase):
    def test_profiles_aggregate_rank_series_and_ratio_counter_totals(self):
        vllm = json.loads((PROFILE_DIR / "vllm_promql.json").read_text())["queries"]
        dcgm = json.loads((PROFILE_DIR / "dcgm_promql.json").read_text())["queries"]
        sglang = json.loads((PROFILE_DIR / "sglang_promql.json").read_text())["queries"]
        inputs = []

        def add(metric, replica, values, **labels):
            tags = {"job_id": "job_1", "rank_id": "rank_1", "instance": replica, **labels}
            selector = ",".join(f"{key}={json.dumps(value)}" for key, value in tags.items())
            inputs.append({"series": metric + "{" + selector + "}", "values": values})

        for replica, hits, lookups, prompt, requests, generated in (
            ("a", 5, 10, 100, 10, 150),
            ("b", 5, 10, 200, 20, 300),
            ("idle", 0, 0, 0, 0, 0),
        ):
            for metric, increment in (
                ("prefix_cache_hits_total", hits),
                ("prefix_cache_queries_total", lookups),
                ("request_prompt_tokens_sum", prompt),
                ("request_prompt_tokens_count", requests),
                ("request_generation_tokens_sum", generated),
                ("request_generation_tokens_count", requests),
                ("request_prefill_time_seconds_sum", hits),
                ("request_decode_time_seconds_sum", lookups),
                ("generation_tokens_total", hits),
            ):
                add("vllm:" + metric, replica, f"0+{increment}x10")
        for replica, buckets in (("a", (990, 1000, 1000)), ("b", (990, 1000, 1000))):
            for le, increment in zip(("1", "2", "+Inf"), buckets, strict=True):
                add("vllm:time_to_first_token_seconds_bucket", replica, f"0+{increment}x10", le=le)
        for replica, gpu, used, free, reserved in (
            ("a", "0", 1, 2, 1),
            ("a", "1", 6, 2, 0),
            ("b", "0", 2, 2, 0),
        ):
            for metric, value in (("USED", used), ("FREE", free), ("RESERVED", reserved)):
                add("DCGM_FI_DEV_FB_" + metric, replica, f"{value}+0x10", gpu_index=gpu)
        for replica in ("a", "b"):
            add("vllm:cache_config_info", replica, "1+0x10", num_gpu_blocks="4096", block_size="16")

        for replica, running, waiting, throughput, usage in (
            ("a", 2, 1, 2, 0.25),
            ("b", 3, 2, 4, 0.75),
        ):
            for tp_rank in ("0", "1"):
                add("sglang:num_running_reqs", replica, f"{running}+0x10", tp_rank=tp_rank)
                add("sglang:gen_throughput", replica, f"{throughput}+0x10", tp_rank=tp_rank)
                add("sglang:full_token_usage", replica, f"{usage}+0x10", tp_rank=tp_rank)
            add("sglang:num_running_reqs", replica, f"{running // 2}+0x10", priority="1")
            add("sglang:num_queue_reqs", replica, f"{waiting}+0x10")
            for metric, increment in (
                ("prompt_tokens_histogram_sum", 1000 if replica == "a" else 300),
                ("generation_tokens_histogram_sum", 100 if replica == "a" else 1500),
                ("prompt_tokens_histogram_count", 10 if replica == "a" else 30),
                ("generation_tokens_histogram_count", 10 if replica == "a" else 30),
                ("time_to_first_token_seconds_count", 1000),
                ("request_time_per_output_token_seconds_count", 1000),
            ):
                add(f"sglang:{metric}", replica, f"0+{increment}x10")
            for mode, increment in (
                ("input", 80 if replica == "a" else 10),
                ("device_hit", 20 if replica == "a" else 10),
                ("host_hit", 0),
                ("storage_hit", 0),
            ):
                add(
                    "sglang:prefill_effective_tokens_total",
                    replica,
                    f"0+{increment}x10",
                    mode=mode,
                )
            for metric, buckets in (
                ("time_to_first_token_seconds_bucket", (990, 1000, 1000)),
                ("request_time_per_output_token_seconds_bucket", (990, 1000, 1000)),
            ):
                for le, increment in zip(("0.01", "0.02", "+Inf"), buckets, strict=True):
                    add(f"sglang:{metric}", replica, f"0+{increment}x10", le=le)

        for rank_id, count in (("rank_empty", 0), ("rank_zero", 10)):
            for metric in ("prompt_tokens_histogram", "generation_tokens_histogram"):
                add(f"sglang:{metric}_sum", "empty", "0+0x10", rank_id=rank_id)
                add(f"sglang:{metric}_count", "empty", f"0+{count}x10", rank_id=rank_id)

        expected = [
            ("prefix_cache_hit_fraction", vllm, 0.5),
            ("prompt_tokens_per_request", vllm, 10),
            ("generated_tokens_per_request", vllm, 15),
            ("prefill_decode_time_ratio", vllm, 0.5),
            ("generated_output_tokens_per_second", vllm, 1),
            ("ttft_p99_seconds", vllm, 1),
            ("gpu_memory_used_fraction_mean", dcgm, 0.5),
            ("gpu_memory_free_mib_mean", dcgm, 2),
            ("ttft_p99_seconds", sglang, 0.01),
            ("tpot_p99_seconds", sglang, 0.01),
            ("generation_throughput", sglang, 6),
            ("running_requests", sglang, 5),
            ("waiting_requests", sglang, 3),
            ("full_token_usage", sglang, 0.5),
            ("prefix_cache_hit_fraction", sglang, 0.25),
            ("prompt_tokens_per_request", sglang, 32.5),
            ("generated_tokens_per_request", sglang, 40),
            ("prefill_observations_per_second", sglang, 200),
            ("decode_observations_per_second", sglang, 200),
        ]
        checks = []
        checks.append(
            {
                "expr": vllm["num_kv_blocks_mean"]["expression"].replace(
                    "__KOI_RANK_FILTER__", 'job_id="job_1",rank_id="rank_1"'
                ),
                "eval_time": "1m",
                "exp_samples": [
                    {
                        "labels": '{job_id="job_1",rank_id="rank_1",num_gpu_blocks="4096"}',
                        "value": 1,
                    }
                ],
            }
        )
        for name, profile, value in expected:
            expression = profile[name]["expression"].replace(
                "__KOI_RANK_FILTER__", 'job_id="job_1",rank_id="rank_1"'
            )
            checks.append(
                {
                    "expr": f"abs(({expression}) - {value}) < bool 0.000001",
                    "eval_time": "1m",
                    "exp_samples": [{"labels": '{job_id="job_1",rank_id="rank_1"}', "value": 1}],
                }
            )
        for rank_id in ("rank_empty", "rank_zero"):
            for name in ("prompt_tokens_per_request", "generated_tokens_per_request"):
                expression = sglang[name]["expression"].replace(
                    "__KOI_RANK_FILTER__", f'job_id="job_1",rank_id="{rank_id}"'
                )
                samples = (
                    []
                    if rank_id == "rank_empty"
                    else [{"labels": f'{{job_id="job_1",rank_id="{rank_id}"}}', "value": 0}]
                )
                checks.append({"expr": expression, "eval_time": "1m", "exp_samples": samples})
        for profile in (vllm, dcgm, sglang):
            for item in profile.values():
                first = item["expression"].replace(
                    "__KOI_RANK_FILTER__", 'job_id="job_1",rank_id=~"rank_1|rank_2"'
                )
                second = item["expression"].replace(
                    "__KOI_RANK_FILTER__", 'job_id="job_2",rank_id="rank_3"'
                )
                # Discard results after evaluation to check every rendered query's syntax.
                checks.append(
                    {
                        "expr": f"(({first}) or ({second})) and on (job_id, rank_id) vector(0)",
                        "eval_time": "1m",
                        "exp_samples": [],
                    }
                )
        # JSON is a YAML subset, so promtool can read this without a YAML dependency.
        spec = {
            "evaluation_interval": "10s",
            "tests": [{"interval": "10s", "input_series": inputs, "promql_expr_test": checks}],
        }
        with tempfile.TemporaryDirectory(prefix="koi-promql-", dir="/tmp/opencode") as directory:
            path = Path(directory) / "rules.yaml"
            path.write_text(json.dumps(spec))
            completed = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--entrypoint",
                    "/bin/promtool",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "--volume",
                    f"{directory}:/checks:ro",
                    IMAGE,
                    "test",
                    "rules",
                    "/checks/rules.yaml",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
