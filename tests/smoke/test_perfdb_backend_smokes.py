import csv
import json

from src.prediction.backends.base import Candidate
from src.prediction.backends.perfdb import PerfDBBackend

FIELDS = [
    "avg_mem_bw_util_pct",
    "avg_mem_util_pct",
    "avg_sm_util_pct",
    "benchmark_target_concurrency",
    "dp",
    "exp_id",
    "gpu_count_total",
    "gpu_model",
    "input_len_tokens_avg",
    "kv_cache_util_pct_avg",
    "max_num_seqs",
    "model_architecture",
    "model_config_json",
    "num_preemptions",
    "output_len_tokens_avg",
    "output_tokens_per_sec",
    "pp",
    "precision",
    "status",
    "task_type",
    "tpot_ms_p99",
    "tp",
    "ttft_ms_p99",
]


def _write_perfdb(path, throughput=600.0):
    model = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }
    row = {
        "avg_mem_bw_util_pct": 35,
        "avg_mem_util_pct": 80,
        "avg_sm_util_pct": 60,
        "benchmark_target_concurrency": 8,
        "dp": 1,
        "exp_id": "row-1",
        "gpu_count_total": 2,
        "gpu_model": "H100",
        "input_len_tokens_avg": 128,
        "kv_cache_util_pct_avg": 20,
        "max_num_seqs": 8,
        "model_architecture": "LlamaForCausalLM",
        "model_config_json": json.dumps(model),
        "num_preemptions": 0,
        "output_len_tokens_avg": 64,
        "output_tokens_per_sec": throughput,
        "pp": 1,
        "precision": "bf16",
        "status": "success",
        "task_type": "batch",
        "tpot_ms_p99": 12,
        "tp": 2,
        "ttft_ms_p99": 100,
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _candidate(**overrides):
    config = {
        "dp": 1,
        "effective_batch_size": 8,
        "max_num_seq": 8,
        "pp": 1,
        "tp": 2,
        **overrides,
    }
    features = {
        "gpu_type": "H100",
        "hidden_size": 4096,
        "isl_token_avg": 128,
        "model_architecture": "LlamaForCausalLM",
        "num_attn_heads": 32,
        "num_hidden_layers": 32,
        "num_kv_heads": 8,
        "osl_token_avg": 64,
        "type": "batch",
        "weight_dtype": "bf16",
    }
    return Candidate(config, features)


def test_perfdb_strict_scope_coverage_spread_dp_and_hash_version(tmp_path):
    path = tmp_path / "perfdb.csv"
    _write_perfdb(path)
    backend = PerfDBBackend(path, enforce_readiness=False)

    estimate = backend.estimate(_candidate())

    assert estimate.status == "success"
    assert estimate.y_hat["throughput_token_per_sec"] == 600.0
    assert estimate.coverage["throughput_token_per_sec"] == 1.0
    assert estimate.spread["throughput_token_per_sec"] == 0.0
    assert backend.estimate(_candidate(dp=2)).status == "unsupported"
    for field, value in (
        ("gpu_type", "A100"),
        ("weight_dtype", "fp16"),
        ("type", "online"),
        ("model_architecture", "OtherArchitecture"),
    ):
        mismatch = _candidate()
        mismatch.job_features[field] = value
        assert backend.estimate(mismatch).status == "no_coverage"

    old_version = backend.version
    _write_perfdb(path, throughput=700.0)
    assert PerfDBBackend(path, enforce_readiness=False).version != old_version


def test_perfdb_enabled_readiness_requires_five_distinct_points(tmp_path):
    path = tmp_path / "perfdb.csv"
    _write_perfdb(path)
    estimate = PerfDBBackend(path).estimate(_candidate())
    assert estimate.status == "insufficient_coverage"
    assert estimate.metadata["required_points"] == 5
