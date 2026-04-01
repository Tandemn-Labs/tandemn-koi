# Koi — Automatic LLM Inference Placement System

Koi is an intelligent orchestration layer for LLM inference jobs. Given a model, a workload description, and a map of available GPU resources, it determines the optimal placement: which GPU type, how many, and exactly how to configure tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP), and the vLLM engine — then tells the tandemn CLI where and how to launch the job.

It does not launch jobs itself. It decides. The tandemn system does the launching.

---

## How It Works — End to End

```
tandemn launch Qwen/Qwen2.5-72B-Instruct dataset.jsonl --hours 8 --cheapest
                          │
                          ▼
              ┌───────────────────────┐
              │     JobRequest        │  model, task_type, avg_input_tokens,
              │     ResourceMap       │  avg_output_tokens, num_requests, SLO,
              └─────────┬─────────────┘  objective + VPC GPU inventory
                        │
                        ▼
              ┌───────────────────────┐
              │       Oracle          │  1. Feasibility pruning (memory, TP head
              │   oracle.py           │     divisibility, PP layer constraints)
              │                       │  2. Enumerate all valid (gpu, TP, PP, DP)
              │                       │     combinations against available resources
              │                       │  3. Predict throughput / latency / cost
              │                       │     using 4-layer interpolation stack
              │                       │  4. Check SLO compliance, sort by cost
              └─────────┬─────────────┘
                        │  List[OracleCandidate] — feasible configs, sorted cheapest first
                        ▼
              ┌───────────────────────┐
              │    LLM Ensemble       │  3 thinkers run in parallel (asyncio):
              │   ensemble.py         │    Sagan   — cost minimizer
              │                       │    Turing  — SLO guardian
              │                       │    Hopper  — hardware efficiency
              │                       │  Each picks one candidate + reasoning
              │                       │
              │                       │  Judge synthesizes all three → final pick
              └─────────┬─────────────┘
                        │
                        ▼
              ┌───────────────────────┐
              │   PlacementDecision   │  gpu_type, instance_type, num_gpus,
              │                       │  TP / PP / DP, vLLM engine config,
              │                       │  predicted metrics, reasoning, confidence,
              │                       │  thinker proposals, top alternatives
              └───────────────────────┘
```

Total placement time: typically **30–60 seconds** (Oracle <1s, 3 parallel LLM calls ~10–15s, judge ~5–10s).

---

## Quick Start

```bash
# 1. Install dependencies
pip install anthropic pydantic python-dotenv aiohttp

# 2. Set API key
cp .env.example .env
# edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the demo
python demo.py
```

The demo uses `results.json` (real Qwen-72B profiling data from L40S) as the performance database and runs two placement scenarios.

---

## File Reference

### `koi/schemas.py` — All Data Models

Every interface in the system is defined here as a Pydantic model. Nothing flows between modules as a raw dict.

**Input models:**

| Model | Description |
|---|---|
| `JobRequest` | Incoming job from tandemn CLI. Has `model_name`, `task_type` (batch/online), `avg_input_tokens`, `avg_output_tokens`, `num_requests`, SLO fields (`slo_deadline_hours`, `slo_tpot_ms`, `slo_ttft_ms`), and `objective` (cheapest/fastest/balanced). |
| `ResourceMap` | Snapshot of the VPC GPU inventory. List of `GPUResource` entries, each with `gpu_type`, `instance_type`, `total_gpus`, `allocated_gpus`, `cost_per_instance_hour_usd`, `gpu_memory_gb`, `interconnect`. |
| `GPUResource` | One GPU type in the VPC. Has `.available_gpus` and `.cost_per_gpu_hour_usd` as computed properties. |

**Oracle outputs:**

| Model | Description |
|---|---|
| `EngineConfig` | vLLM launch parameters: `tensor_parallel_size`, `pipeline_parallel_size`, `max_num_seqs`, `max_model_len`, `gpu_memory_utilization`, `dtype`, `enable_chunked_prefill`. Has `.to_vllm_args()` for CLI rendering. |
| `PlacementConfig` | Complete hardware spec: `gpu_type`, `instance_type`, `num_gpus`, `num_instances`, `tp`, `pp`, `dp`, `region`, `engine_config`. |
| `PredictedMetrics` | Performance forecast: `throughput_tokens_per_sec`, `estimated_runtime_hours`, `total_cost_usd`, `tpot_ms`, `ttft_ms`, `cost_per_hour_usd`, `confidence` (0–1), `data_source` (exact_match/interpolated/cross_gpu/analytical). |
| `OracleCandidate` | One feasible config + its predicted metrics + `meets_slo` flag + `slo_margin_pct`. |

**Decision output:**

| Model | Description |
|---|---|
| `ThinkerProposal` | One LLM thinker's pick: `thinker_name`, `config`, `metrics`, `reasoning`, `key_concerns`, `confidence_in_choice`. |
| `PlacementDecision` | Final output. Has `recommendation` (PlacementConfig), `predicted_metrics`, `reasoning` (judge's synthesis), `confidence`, `thinker_proposals` (all three), `alternatives` (top 3 other options), and `.display_summary()` for pretty CLI output. |

**Refinement models (Phase 2):**

| Model | Description |
|---|---|
| `RuntimeMetrics` | Live metrics snapshot fetched from the metrics API: throughput, TPOT, TTFT, GPU util, memory, queue depth. |
| `DeltaRecord` | Prediction vs actual error record stored in SQLite. The ground truth dataset that the system learns from. |
| `PESComponents` | Placement Efficiency Score breakdown: `cer` (cost efficiency), `per` (physical efficiency), `ss` (stability). `composite = α×CER + β×PER + γ×SS`. |

---

### `koi/oracle.py` — Numerical Prediction Engine

The Oracle runs entirely before any LLM is invoked. It answers: *"given this model and these GPUs, what configs are even possible, and how well does each one perform?"*

**Feasibility pruning (eliminates impossible configs):**
- Memory: model weights in fp16 must fit per GPU at the given TP. E.g. Qwen-72B (144GB fp16) needs TP≥4 on L40S (45.5GB VRAM). Requires ≥8GB headroom for KV cache.
- TP validity: TP must evenly divide `num_attention_heads` AND `num_kv_heads`.
- PP validity: PP must evenly divide `num_layers`.
- Resource availability: `tp × pp × dp ≤ available_gpus`.

**Candidate enumeration:** For every available GPU type, tries TP ∈ {1,2,4,8} × PP ∈ {1,2,4} × DP ∈ {1..max_available}. Skips any combination that fails the checks above.

**4-layer interpolation (first hit wins):**

| Layer | Condition | Confidence |
|---|---|---|
| 1. Exact match | Same model + GPU type + TP + PP in perf DB | 0.80 |
| 2. Interpolated | Same GPU + TP + PP, different model (scale by param ratio) | 0.55 |
| 3. Cross-GPU | Same model + TP + PP, different GPU (scale by bandwidth/FLOPS ratio) | 0.45 |
| 4. Analytical | No DB data → pure roofline model (bandwidth / model weight bytes) | 0.35 |

I/O length scaling: longer sequences reduce throughput via a dampened square-root scaling (`scale = (base_work/target_work)^0.5`, clamped to [0.5, 1.4]). Work units = `0.3 × input_len + 1.0 × output_len` (decode is ~3× heavier than prefill per token).

Cross-GPU scaling: blends bandwidth scaling (for decode-bound) and FLOPS scaling (for prefill-bound) weighted by the job's prefill/decode ratio.

**Output:** `List[OracleCandidate]` sorted SLO-meeting first, then cheapest first within each group.

**Hardware specs table** (in `oracle.py`, `GPU_SPECS`):

| GPU | BW (GB/s) | FP16 TFLOPS | VRAM (GB) |
|---|---|---|---|
| H100 SXM | 3350 | 989 | 79 |
| H200 | 4800 | 989 | 140 |
| A100 | 2000 | 312 | 79 |
| L40S | 864 | 733 | 45.5 |
| A10G | 600 | 125 | 23 |
| L4 | 300 | 121 | 23 |

**Known model architectures** (in `oracle.py`, `MODEL_ARCH`): Qwen2.5-72B, Qwen3-32B, Qwen3-235B-A22B, DeepSeek-R1-Distill-70B, Llama-3-70B/8B. Unknown models fall back to regex param extraction from the model name.

---

### `koi/ensemble.py` — Multi-LLM Thinker + Judge

Three instances of `claude-opus-4-6` run in parallel with different system-prompt personas. Each independently reviews the Oracle's candidate table and proposes one configuration.

**Thinkers:**

| Name | Persona | Optimization bias |
|---|---|---|
| **Sagan** | Cost optimizer | Cheapest config that meets SLO. Right-size everything, SLO margin beyond minimum is waste. |
| **Turing** | SLO guardian | ≥20% headroom, safety buffers, prefer empirically validated configs over analytical estimates. |
| **Hopper** | HW efficiency | Maximize GPU utilization (tokens/GPU/sec). Avoid PP bubbles, communication overhead, bandwidth waste. |

Each thinker returns JSON:
```json
{
  "chosen_candidate_idx": 3,
  "reasoning": "...",
  "key_concerns": ["PCIe overhead at TP=8", "..."],
  "confidence_in_choice": 0.82
}
```

**Judge** (`claude-opus-4-6`, different system prompt): receives all three proposals, the full candidate table, and the job context. Synthesizes into one final JSON decision with `chosen_candidate_idx`, `reasoning`, `confidence`, and `advisor_agreement` (full/majority/split).

**Fallback behavior:** If any API call fails, the affected thinker falls back to the cheapest SLO-meeting candidate. If the judge fails, majority vote among thinkers is used.

**Context injected into every LLM call:**
- Job details (model, task type, token lengths, SLO, objective)
- Available GPU inventory with costs and interconnect
- Full candidate table (up to 15 rows, pre-formatted) with throughput, TPOT, runtime, cost, confidence, SLO margin
- (Phase 2) VPC delta history from the refinement engine as few-shot correction examples

---

### `koi/placement.py` — Main Orchestrator

`KoiPlacement` is the only class the calling code needs. It wires the Oracle and Ensemble together.

```python
koi = KoiPlacement(
    api_key="sk-ant-...",
    perfdb_path="./perfdb",
    llm_model="claude-opus-4-6",
    max_candidates_to_llm=15,
)
decision = koi.decide(request, resource_map)
print(decision.display_summary())
```

`decide()` is synchronous. `decide_async()` is the async version for FastAPI/server contexts.

Steps:
1. Oracle generates all feasible candidates and sorts them
2. Filters to SLO-meeting candidates only (unless `include_non_slo_candidates=True`)
3. Warns if no SLO-meeting candidates exist (passes all to LLM anyway)
4. Runs the ensemble (`run_sync` wraps `asyncio.run` around the async calls)
5. Builds top-3 alternatives by excluding the chosen config from the front of the sorted list
6. Returns `PlacementDecision`

---

### Orca Integration

Koi runs as an HTTP service that the [Orca](https://github.com/Tandemn-Labs/Tandemn-orca) CLI calls alongside its roofline solver.

**Setup:**
```bash
pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-... python -m koi.server
# Koi listens on :8090 (configure with KOI_PORT)
```

**Orca side:** Set `KOI_SERVICE_URL=http://localhost:8090` and the CLI will call `POST /decide` in parallel with roofline during `orca plan` / `orca deploy`.

**Data flow:**
1. Orca's `GET /resources` returns raw AWS data: instance catalog (GPU specs, pricing, vCPUs, quota family) + quota pools (per-family/region/market vCPU limits)
2. CLI forwards this as the `resource_map` in `POST /decide` along with the `job_request`
3. Koi's Oracle joins instances to quota pools, computes per-instance-type availability, enumerates TP/PP/DP candidates, predicts metrics
4. LLM ensemble picks the best config
5. CLI shows both Koi and roofline results; user chooses which to deploy

**`POST /decide` request:**
```json
{
  "job_request": {
    "model_name": "Qwen/Qwen2.5-72B-Instruct",
    "task_type": "batch",
    "avg_input_tokens": 512,
    "avg_output_tokens": 256,
    "num_requests": 100000,
    "slo_deadline_hours": 8.0,
    "objective": "cheapest"
  },
  "resource_map": {
    "instances": [...],
    "quotas": [...]
  }
}
```

**Response:** Full `PlacementDecision` (recommendation, predicted metrics, reasoning, thinker proposals, alternatives).

---

### `koi/monitor.py` — Runtime Monitoring (Phase 2)

Koi does not instrument jobs itself. It fetches metrics that are already being collected. This file provides the processing layer on top of fetched data.

**`MetricsSource`** (abstract base class): implement `async fetch(job_id) → RuntimeMetrics` for any data source. Concrete implementations are in `metrics_api.py`.

**`KalmanFilter1D`**: smooths raw noisy metric readings into reliable state estimates. Parameters:
- `R` (measurement noise): higher = trust the model more, react to measurements less. Default 25.0.
- `Q` (process noise): higher = track changes faster. Default 1.0.
- For TPOT: low Q (slowly changing), higher R (noisy per-request variance)
- For throughput: higher Q (can spike quickly)

**`DeadbandController`**: two-threshold hysteresis prevents oscillation from small metric fluctuations.
- GREEN: metric < 80% of SLO → system is healthy, do nothing
- YELLOW_HIGH: metric > 90% of SLO → approaching limit, MPC considers action
- RED: metric > 105% of SLO → SLO violated, act immediately
- YELLOW_LOW: metric < 50% of SLO → overprovisioned, consider scaling down

The hysteresis means the state only exits GREEN when the outer threshold is crossed, and only returns to GREEN when the metric drops below the inner threshold. This prevents the system from oscillating between states on small fluctuations.

**Anti-windup** (`JobMonitorState.freeze()`): when a reconfiguration is in progress (weights loading, autoscale spinning up), error tracking is suspended for that job. Prevents compounding corrections while a change is already being applied.

**`KoiMonitor`**: manages per-job polling loops. Call `start_job(decision)` on deployment, `stop_job(job_id)` on completion. `compute_delta()` returns a `DeltaRecord` ready for the refinement engine.

---

### `koi/metrics_api.py` — Metrics Data Sources

Two concrete `MetricsSource` implementations:

**`tandemnMetricsAPISource`**: hits `GET {base_url}/jobs/{job_id}/metrics`. Reads `tandemn_METRICS_API_URL` and `tandemn_METRICS_API_KEY` from environment. Parses both flat JSON and nested JSON response formats. This is the primary source — implement the endpoint on the tandemn side and this just works.

**`VLLMPrometheusSource`**: reads vLLM's built-in `/metrics` Prometheus endpoint. Accepts a dict of `{job_id: endpoint_url}` so multiple jobs on different ports are handled. Parses Prometheus text format, extracts throughput, concurrent requests, GPU cache utilization. TPOT is estimated from throughput (histogram parsing is a TODO).

To add a new source: subclass `MetricsSource` from `koi.monitor` and implement `async fetch(job_id) → Optional[RuntimeMetrics]`.

---

### `koi/refinement.py` — Evolutionary Learning Engine (Phase 2)

This is what makes Koi get better over time. Three independent learning channels:

**Channel 1 — `DeltaStore` (SQLite at `./data/delta_store.db`):**
Stores every completed job's prediction error: `(model, gpu_type, tp, pp, predicted_tps, actual_tps, delta_pct, predicted_tpot, actual_tpot, delta_tpot)`. One row per job.

Query pattern: `find_similar(gpu_type, tp, pp, model_name, k=8)` returns the k most recent similar runs. This is the RAG corpus — the Oracle correction layer retrieves from here to generate prediction adjustments specific to this VPC's actual performance characteristics.

Why per-VPC: the same instance type performs differently across AWS regions, NVLink vs PCIe availability, thermal conditions, co-tenancy patterns. The delta store learns the "personality" of the specific cluster.

**Channel 2 — `PolicyMemory` (ChromaDB at `./data/policy_memory`):**
Stores natural-language summaries of past job outcomes as vector embeddings. Example: *"Qwen-72B batch on L40S TP=4 PP=4. Predicted 1180 tok/s, actual 1197 tok/s. PES=0.87. Key lesson: Oracle accurate for this config class."*

At placement time, `retrieve_similar(model_name, gpu_type, k=3)` returns the most semantically similar past decisions. These are injected into the LLM ensemble prompt as few-shot examples. The LLMs see what worked and what didn't for similar workloads without any fine-tuning.

Falls back to in-memory list if ChromaDB is not installed.

**Channel 3 — `EfficiencyFrontier` (SQLite at `./data/frontier.db`):**
Tracks the Pareto frontier of `(cost, throughput)` for each workload class (model family × task type × I/O length bucket). Used to compute the CER denominator in PES: the cheapest known SLO-meeting configuration for this workload class. As the system discovers cheaper configs, the frontier tightens and CER scores from earlier jobs retroactively look worse — the system is graded on an increasingly hard curve.

**`PolicyLearner`**: builds the RAG correction context string from Channel 1. Given a new Oracle prediction, retrieves similar delta records and formats them into a natural-language block that describes historical prediction errors on this cluster. This block is injected into the LLM ensemble prompt so thinkers can apply learned corrections.

**`compute_pes()`**: computes the scalar PES for a completed job.
```
PES = α×CER + β×PER + γ×SS

CER = best_known_cost / actual_cost      (0 if SLO missed)
PER = actual_throughput / roofline_peak
SS  = time_in_final_config / total_time

Weights (batch): α=0.50, β=0.35, γ=0.15
Weights (online): α=0.40, β=0.30, γ=0.30
```

**`KoiRefinement`**: top-level class that wires all three channels. Call `record_completion()` after each job finishes to update all stores simultaneously.

---

### `koi/arbiter.py` — Multi-Job Swap Arbiter (Phase 2)

Handles the global scheduling problem: given all running jobs, should any resources be moved between them?

**`RunningJob`**: tracks a live job's current metrics, SLO pressure, and priority. `is_donor_candidate()` returns True if the job has >30% SLO headroom and could spare resources.

**`compute_nbs()`**: Net Benefit Score for a proposed swap.
```
NBS = benefit_victim - cost_donor - transition_cost - opportunity_cost

benefit_victim   = SLO_pressure_reduction × victim_priority × 2.0
cost_donor       = donor_SLO_headroom_loss × donor_priority × 1.5
transition_cost  = (downtime_minutes / 60) × gpu_cost × resource_delta × 0.5
opportunity_cost = 0.1 (placeholder; scales with cluster utilization)
```
Positive NBS = swap is beneficial. NBS < 0 = harmful, do not swap.

**`SwapArbiter.evaluate_swaps(victim_job_id)`**: scans all running jobs for donor candidates, computes NBS for each, returns proposals sorted by NBS descending. Proposals with `requires_llm_review=True` (high-priority jobs, cross-model swaps) are routed through LLM reasoning before execution.

**Fairness floor**: no job can be reduced to a point where its SLO is at risk. The minimum allowed state is the config that meets SLO with ≥0% margin.

**`ResourceAdder`**: separate from the Swap Arbiter. Decides whether to acquire *new* capacity (cloud autoscale) rather than rebalancing existing resources. Uses newsvendor framing: `should_add_capacity()` provisions now if projected TPOT (accounting for load trend and provision lead time) will reach the SLO before new instances are ready. `should_remove_capacity()` scales down if headroom >40% and load is not growing.

---

### `koi/exploration.py` — Active Exploration Loop (Phase 2)

A separate background process that runs every 30–60 minutes (much slower than the main placement loop). Its job is to discover new efficient configurations by occasionally trying uncertain regions of the config space.

**UCB acquisition function**: `UCB = estimated_PES + β × uncertainty`
- `estimated_PES`: Oracle confidence adjusted for SLO margin
- `uncertainty`: estimated from the number of similar past runs in the DeltaStore. 0 runs = 1.0, ≥10 runs = 0.1.
- `β` controls exploration aggressiveness. Starts at 0.5, decays as knowledge accumulates, floors at 0.05.

**`ExplorationBudget`**: tracks what fraction of recent decisions have been exploratory. Target budget starts at 10%, decays toward 2% as uncertainty drops. Resets when a new GPU type or model family is added to the cluster.

**Exploration safety rules** (both must hold):
- Job priority ≤ 5 (never explore on production/high-priority jobs)
- SLO headroom ≥ 50% (there's enough slack to absorb a suboptimal config)

**`get_exploration_override()`**: called by the placement pipeline before the LLM ensemble. If the job qualifies, returns a UCB-selected candidate (which may not be the cheapest) instead of the standard top candidate. The LLM ensemble then reasons about this candidate. If it fails SLO, the system falls back to the standard recommendation.

---

## Performance Database (`perfdb/`)

The Oracle loads all JSON and CSV files from this directory at startup.

**Supported formats:**

`results.json` format (from vLLM benchmarking runs):
```json
[{
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "tp": 4, "pp": 4,
  "max_input_length": 128, "max_output_length": 128,
  "total_tokens_per_sec": 1196.8,
  "instance_type": "4x g6e.12xlarge",
  "price_per_hour": 18.72,
  "total_gpus": 16,
  "benchmark_target_concurrency": 81
}]
```

`data.csv` format (canonical schema from `schema.md` — columns: `model_name`, `gpu_model`, `tp`, `pp`, `input_len_tokens_fixed`, `output_len_tokens_fixed`, `tokens_per_sec_total`, etc.)

If `perfdb/` has no data, the Oracle falls back to `./results.json` in the project root.

**Current data coverage:** Qwen-2.5-72B on 16× L40S (`4x g6e.12xlarge`), TP=4 PP=4, various I/O lengths. All other predictions are interpolated or analytical until more profiling data is added.

---

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Phase 2: metrics fetching
tandemn_METRICS_API_URL=http://localhost:8080
tandemn_METRICS_API_KEY=

# Optional: direct vLLM Prometheus scraping
VLLM_METRICS_URL=http://localhost:8000
```

Copy `.env.example` to `.env` and fill in values.

---

## What Is and Is Not Implemented

### Phase 1 — Fully Implemented
- `schemas.py`: all data models
- `oracle.py`: feasibility pruning, 4-layer interpolation, candidate enumeration
- `ensemble.py`: 3 parallel thinkers + judge, async via `asyncio.gather`, JSON structured output, fallback handling
- `placement.py`: synchronous and async orchestrator
- `demo.py`: end-to-end demo with 3 scenarios

### Phase 2 — Framework Present, Not Yet Wired
- `monitor.py`: Kalman filter, deadband, anti-windup, per-job state — needs `tandemnMetricsAPISource.fetch()` implemented
- `metrics_api.py`: `tandemnMetricsAPISource` is written — needs tandemn API endpoint on the other side; `VLLMPrometheusSource` is functional
- `refinement.py`: `DeltaStore`, `PolicyMemory`, `EfficiencyFrontier`, `PolicyLearner`, `compute_pes` — all implemented; needs to be called from `placement.py` after job completion
- `arbiter.py`: `SwapArbiter`, `compute_nbs`, `ResourceAdder` — logic implemented; Oracle integration for new config predictions is a TODO
- `exploration.py`: UCB scoring, budget tracking — implemented; not yet triggered from placement pipeline

### Explicitly Not In Scope (Yet)
- GRPO fine-tuning of the LLM ensemble on policy memory
- Actual job launching (tandemn CLI handles this)
- Multi-tenant auth / job ownership
- CloudWatch GPU metrics integration (framework accepts it via custom `MetricsSource`)

---

## Adding New Profiling Data

Drop any JSON file in `perfdb/` matching the `results.json` schema (array of benchmark records). The Oracle picks it up automatically on next startup. For `data.csv` format, place it at `perfdb/data.csv`. Both can coexist.

The more data added — especially for different TP/PP configs, GPU types, and I/O length combinations — the better the Oracle predictions and the lower the confidence penalty. The full target coverage is in `perfdb/README.md`.

---

## Reference: `PlacementDecision.display_summary()` Output

```
============================================================
  KOI PLACEMENT DECISION — job-a1b2c3d4
============================================================
  Model    : Qwen/Qwen2.5-72B-Instruct
  Placement: 4x g6e.12xlarge (L40S) | TP=4 PP=4 DP=1 | 16 GPUs total
  Region   : us-east-1

  Parallelism  : TP=4  PP=4  DP=1
  Throughput   : 1197 tok/s
  Est. Runtime : 6.43 hours
  Est. Cost    : $120.16
  Confidence   : 80%  (source: interpolated)

  vLLM Args:
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 4 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9 \
  --dtype auto

  Reasoning: Sagan and Turing both converged on TP=4 PP=4 on L40S...
============================================================
```
