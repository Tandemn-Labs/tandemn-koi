# Koi v2 — Evolutionary Agentic Cluster Management

**Automated, data-driven GPU placement for batched LLM inference.**

Koi abstracts the GPU/MLOps/deployment layer from the user. User provides a model, a dataset, and an SLO — Koi figures out the rest.

---

## 1. What Koi Does

```
User: "Run Qwen/Qwen2.5-72B-Instruct on this 5000-request dataset, finish in 8 hours"

Koi:  1. Queries PerfDB → finds that A100-80GB TP=8 PP=1 gets 1498 tok/s for this model
      2. Checks live quota → 4× p4de.24xlarge available in us-west-2
      3. Proposes config: p4de TP=4 PP=2, ETA=0.35h, cost=$14.34
      4. Tells Orca to launch it
      5. Monitors throughput via Orca telemetry
      6. If falling behind SLO → adds replicas (A/B tests new GPU chain)
      7. When done → records outcome in memory (prediction error, what worked, what didn't)
      8. Next similar job → Koi already knows the answer
```

**Koi is the brain, Orca is the muscle.** Orca never needs to know Koi exists — it just receives API calls to launch, scale, and monitor jobs.

---

## 2. Design Philosophy: Evolutionary Agentic Systems

Koi draws from the evolutionary AI systems literature:

| System | Core Idea | What Koi Borrows |
|--------|-----------|-----------------|
| [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind) | LLM generates solutions → evaluate → select → evolve | Population of configs evaluated in production; fitness = real $/token |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | MAP-Elites + island model for diversity | Quality-diversity: don't just find the cheapest config, explore the Pareto frontier |
| [ACE](https://arxiv.org/abs/2510.04618) (ICLR 2026) | Evolving contexts as playbooks — structured incremental updates | Koi's memory layer: structured, incremental, never collapses into a summary |
| [GEPA](https://gepa-ai.github.io/gepa/) | Reflective evolution with actionable side-information | Failure diagnosis: when a config fails, Koi reflects on WHY (OOM? quota? slow?) and stores actionable rules |
| [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) | LLM with tools in an autonomous loop | Koi's agent: Claude with tools for PerfDB, Orca API, memory, physics |

**Key principle from AlphaEvolve**: *Every production job is a real evaluation.* Unlike lab settings, Koi can't afford to explore bad configs aggressively — the user's SLO is at stake. But every job (even "normal" ones) generates learning data. The system improves passively through production traffic, and explores actively only when SLO headroom allows.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                            KOI                                      │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     KOI AGENT                                 │  │
│  │              (Claude Agent SDK)                                │  │
│  │                                                               │  │
│  │  Tools:                                                       │  │
│  │    query_perfdb()     → Performance Database                  │  │
│  │    query_memory()     → Agentic Memory                        │  │
│  │    get_resources()    → Live GPU Quota (from Orca)            │  │
│  │    get_gpu_physics()  → GPU Specs + Bottleneck Analysis       │  │
│  │    launch_chain()     → Deploy via Orca API                   │  │
│  │    scale_chain()      → Add/remove replicas via Orca API      │  │
│  │    get_job_metrics()  → Live telemetry from Orca              │  │
│  │    record_outcome()   → Write to Agentic Memory               │  │
│  │                                                               │  │
│  └──────┬──────────┬───────────┬──────────┬─────────────────────┘  │
│         │          │           │          │                         │
│    Context 1  Context 2   Context 3   Actions                      │
│         │          │           │          │                         │
│  ┌──────▼──┐ ┌─────▼────┐ ┌───▼───┐ ┌───▼────────────┐           │
│  │ PerfDB  │ │ Agentic  │ │  SLO  │ │  Orca REST API │           │
│  │         │ │ Memory   │ │Monitor│ │                 │           │
│  │ 687 col │ │          │ │       │ │ POST /deploy    │           │
│  │ bench-  │ │ outcomes │ │ good/ │ │ POST /scale     │           │
│  │ marks   │ │ failures │ │ bad   │ │ GET  /resources │           │
│  │ physics │ │ rules    │ │ live  │ │ GET  /metrics   │           │
│  └─────────┘ └──────────┘ └───────┘ └────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │         ORCA             │
                    │    (Data Plane)          │
                    │                          │
                    │  ┌──────┐ ┌──────┐       │
                    │  │Chain1│ │Chain2│  ...   │
                    │  │L40S  │ │A100  │       │
                    │  │TP=4  │ │TP=8  │       │
                    │  │PP=2  │ │PP=1  │       │
                    │  └──────┘ └──────┘       │
                    │                          │
                    │  Instance Launch          │
                    │  Monitoring (local DB)    │
                    │  Job Completion           │
                    │  Chunk Routing (Redis)    │
                    └──────────────────────────┘
```

---

## 4. The Koi Agent

### Why Claude Agent SDK (not a fixed pipeline)

Koi v1 used a fixed pipeline: Oracle → 3 LLM Thinkers → Judge → Decision. This was rigid — 150-300 seconds per decision, retries on guardrail failures, LLMs proposing physically impossible configs.

Koi v2 replaces this with a **single autonomous agent** that has tools. The agent decides what information to gather and what actions to take based on the situation:

```python
# Pseudocode — what the agent does autonomously
agent = KoiAgent(
    model="claude-sonnet-4-6",
    tools=[query_perfdb, query_memory, get_resources, get_gpu_physics,
           launch_chain, scale_chain, get_job_metrics, record_outcome],
    system_prompt=KOI_SYSTEM_PROMPT,
)

# User request comes in
result = agent.run("""
    Deploy Qwen/Qwen2.5-72B-Instruct for batch inference.
    Dataset: 5000 requests, avg 953 input tokens, avg 1024 output tokens.
    SLO: finish in 8 hours.
    Objective: cheapest $/token.
""")
```

The agent then:
1. Calls `query_perfdb(model="Qwen/Qwen2.5-72B-Instruct")` to find benchmark data
2. Calls `get_resources()` to see what GPUs are available
3. Calls `get_gpu_physics("A100-80GB")` to understand bottlenecks
4. Reasons about the options and picks a config
5. Calls `launch_chain(config)` to deploy

If the agent needs more data, it queries more. If the perfdb has an exact match with high confidence, it decides in one tool call. If it's a novel model with no data, it does more exploration. **The agent adapts its strategy to the situation**, unlike the fixed pipeline.

### Agent Tools

| Tool | Input | Output | When Used |
|------|-------|--------|-----------|
| `query_perfdb` | model, gpu_type, tp, pp, io_ratio filters | Matching benchmark records with throughput, latency, cost, physics features | Every decision — find what we know about this config |
| `get_model_arch` | model_name | Model architecture: layers, heads, kv_heads, params, is_moe, hidden_dim | Every decision — especially for unknown models (fetches from HF Hub if needed) |
| `find_similar_models` | model_name (fetches HF config internally) | Ranked list of PerfDB models by physics-vector distance + their benchmark records | When model has NO perfdb data — finds architecturally similar models by inference physics, not name |
| `query_memory` | model, instance_type, status filters | Past job outcomes, failure reasons, learned rules | Every decision — check if we've tried this before |
| `get_resources` | none | Live GPU quota per region: available vCPUs, instance types, pricing | Every decision — what can we actually launch? |
| `get_gpu_physics` | gpu_type | Bandwidth, TFLOPS, VRAM, interconnect, roofline analysis | When reasoning about WHY a config is fast/slow |
| `launch_chain` | job config (gpu, tp, pp, dp, instance_type) | job_id, status | After deciding on a config |
| `scale_chain` | job_id, new config, count | replica_ids | During monitoring — add/remove capacity |
| `get_job_metrics` | job_id | Live throughput, TPOT, GPU util, KV cache, queue depth | Monitoring loop — check if SLO on track |
| `record_outcome` | job_id, outcome (success/fail), metrics, failure_reason | confirmation | After job completes — feed memory layer |

### PerfDB Retrieval: Agent with SQL Tools vs RAG

**Why agent-driven queries beat RAG for structured data:**

| | RAG (Koi v1) | Agent + SQL (Koi v2) |
|---|---|---|
| **Query** | Embedding similarity: "find records similar to this text" | Structured: "SELECT * WHERE model='Qwen-72B' AND gpu='L40S' AND io_ratio BETWEEN 3 AND 5" |
| **Precision** | Fuzzy — can miss relevant records, surface irrelevant ones (L40S gap in our testing) | Exact — gets exactly what's asked for |
| **Multi-step** | Single FAISS query, top-k | Agent can query, inspect results, query again ("show me cross-GPU scaling for this TP") |
| **Aggregation** | Can't compute "avg throughput for this model on L40S" | Agent can ask for aggregates, comparisons, trends |
| **Failure diagnosis** | Can't query "show me all failed configs for this model" | Agent queries memory for failures + reasons |
| **Speed** | Fast (50ms FAISS) | Slower (LLM tool calls) but can be cached for repeat workloads |

**Hybrid approach for speed:**

1. Pre-compute a **context packet** when a job arrives (fast, no LLM):
   - `get_model_arch(model_name)` → HF config → physics vector
   - Query PerfDB: exact model match first, physics-similar models as fallback
   - Filter by available GPU types (from `get_resources()`)
   - Sort by io_ratio closeness to the job's workload
   - Top 20 records as initial context, tagged with source:
     - `"direct"` — same model, high confidence
     - `"proxy:Llama-70B:dist=0.02"` — similar model, distance-scaled confidence
   - Query memory for past outcomes on this model (or similar models)

2. **Pre-compute a cost table** (critical — LLMs are bad at arithmetic):
   - For each config in memory outcomes + PerfDB, compute:
     `total_cost = ($/hr × total_tokens) / (tps × 3600)`
   - Sort by total cost, cheapest first
   - Tag each row with source: `VERIFIED` (ground truth from memory) or `PerfDB`
   - Inject the table into the prompt — agent reads costs instead of computing them
   - Example table the agent sees:
     ```
     Source       GPU          Config             TPS     $/hr    ETA    Total$  SLO
     VERIFIED     L40S         TP=4 PP=4 DP=1    1100    18.72   2.50   $46.74   ✓
     PerfDB       A100-80GB    TP=8 PP=1 DP=1    2186    40.96   1.26   $51.45   ✓
     PerfDB       L40S         TP=4 PP=2 DP=1     528    20.98   5.20  $109.11   ✓
     ```
   - Agent picks the cheapest ✓ row and verifies with tools

3. Give the agent this packet + tools to query more if needed:
   - Agent sees the cost table + can call tools to verify or explore alternatives
   - If cost table has a VERIFIED row → high confidence, minimal tool calls
   - If only PerfDB rows → agent verifies physics, may adjust
   - If no data at all → agent uses physics/roofline + explores conservatively

4. For repeat workloads, **memory IS the primary context** — already vetted by real production runs. Agent checks memory first before touching PerfDB.

---

## 5. Three Context Channels

Every agent decision is informed by three independent channels:

### Context 1: Performance Database (PerfDB)

**What:** 687-column CSV (future: proper database) of real vLLM benchmark results.

**Key columns the agent cares about:**

| Category | Columns | Agent uses for |
|----------|---------|---------------|
| Config | `tp, pp, dp, instance_type, gpu_model, total_gpus` | What was the config? |
| Throughput (batch) | `tokens_per_sec_total, tokens_per_sec_per_gpu` | **Primary metric for batch** — directly determines ETA and $/job. TPOT/TTFT don't matter for batch; nobody cares about per-token latency when processing a dataset. |
| Latency (online, future) | `tpot_ms_p50, ttft_ms_p50, e2e_ms_p50` | Only matters for online serving where a user waits for each token. Irrelevant for batch placement decisions. |
| Cost | `price_per_hour, cost_per_1m_tokens_total_usd, total_cost_usd` | How much did it cost? `cost_per_1m_tokens` is the real optimization target for `objective: cheapest`. |
| Physics | `gpu_bandwidth_gbps, gpu_tflops_fp16, params_per_gpu, vram_headroom_gb, bandwidth_per_param, crosses_node_boundary` | WHY was it fast/slow? |
| Model | `model_name, params_billion, model_size_gb, is_moe, model_architecture` | What model? |
| Workload | `input_len_tokens_avg, output_len_tokens_avg, prefill_decode_ratio, num_requests, batch_size` | What was the workload shape? |
| GPU Metrics | `gpu_sm_util_pct, gpu_mem_bw_util_pct, kv_cache_util_pct_avg` | How saturated was the hardware? |

**Physics features are critical.** They let the agent reason causally:
- `bandwidth_per_param` low + `gpu_mem_bw_util_pct` high → memory-bandwidth bottleneck → more TP or faster GPU
- `vram_headroom_gb` near zero → KV cache pressure → risk of OOM at higher concurrency
- `crosses_node_boundary` = true → inter-node latency penalty → avoid if possible
- `prefill_decode_ratio` > 4 → compute-bound prefill → TFLOPS matters more than bandwidth

**Unknown model handling: physics-vector similarity search.**

When a model has zero PerfDB records, the agent calls `find_similar_models()` which:

1. Fetches the unknown model's HF `config.json` (layers, heads, kv_heads, hidden_dim, vocab, MoE fields)
2. Computes a **physics vector** — the features that actually determine inference performance:

```python
physics_vector = {
    "model_size_gb":        num_params * dtype_bytes / 1e9,   # VRAM, bandwidth requirement
    "kv_bytes_per_token":   2 * num_layers * kv_dim * 2,      # concurrent request capacity
    "flops_per_fwd":        attention_flops + ffn_flops,      # compute cost
    "gqa_ratio":            num_attention_heads / num_kv_heads,# KV cache efficiency
    "num_layers":           num_layers,                       # valid PP values
    "num_attention_heads":  num_attention_heads,               # valid TP values
    "is_moe":               int(num_experts > 1),             # dense vs MoE
}
```

3. Computes weighted distance against every model in PerfDB (fast — one row per distinct model):

```python
distance = (
    0.35 * |Δmodel_size_gb| / model_size_gb +    # most important: memory footprint
    0.25 * |Δkv_bytes_per_token| / kv_bytes +     # KV cache sizing
    0.15 * |ΔFLOPs| / FLOPs +                     # compute profile
    0.10 * |Δgqa_ratio| / gqa_ratio +             # KV efficiency
    0.05 * |Δlayers| / layers +                   # PP compatibility
    0.05 * |Δheads| / heads +                     # TP compatibility
    0.05 * |Δis_moe|                              # dense/MoE mismatch
)
```

4. Returns ranked list: `[("Llama-70B", dist=0.02, records=[...]), ("DeepSeek-70B", dist=0.04, records=[...])]`

**Why physics distance, not embeddings?** Two models with identical `model_size_gb`, `kv_bytes_per_token`, and `flops_per_fwd` will have identical inference throughput on the same GPU — regardless of what the model is called or what it was trained on. Embedding similarity captures text similarity ("Qwen" is close to "Qwen2" in embedding space) but misses physics. Physics distance captures what actually matters for placement.

**Confidence scaling:** Agent discounts proxy data based on distance. `dist=0.02` → 95% confidence in proxy benchmarks. `dist=0.15` → 70% confidence. `dist>0.30` → too different, use analytical roofline instead.

**Future: PerfDB as a real database.** The CSV works for 300 records but won't scale. Plan:
- SQLite or Postgres for structured queries
- Same schema, indexed by (model_name, gpu_type, tp, pp, input_len, output_len)
- Agent's `query_perfdb` tool runs SQL queries
- Ingest pipeline from Orca's profiling runs

### Context 2: Agentic Memory

**What:** A structured, persistent store of everything Koi has tried, observed, and learned.

**This is NOT a summary or embedding store.** Inspired by [ACE (Agentic Context Engineering)](https://arxiv.org/abs/2510.04618), the memory uses structured, incremental updates that preserve detail. No lossy compression, no embedding-space collapse.

**Schema:**

```sql
-- Every decision Koi makes (predicted)
CREATE TABLE decisions (
    decision_id     TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_name      TEXT NOT NULL,
    
    -- Config chosen
    instance_type   TEXT NOT NULL,
    gpu_type        TEXT NOT NULL,
    tp              INTEGER NOT NULL,
    pp              INTEGER NOT NULL,
    dp              INTEGER NOT NULL,
    num_gpus        INTEGER NOT NULL,
    quantization    TEXT,
    
    -- What Koi predicted
    predicted_tps           REAL,
    predicted_cost_per_hour REAL,
    predicted_total_cost    REAL,
    predicted_runtime_hours REAL,
    prediction_confidence   REAL,     -- 0-1
    prediction_source       TEXT,     -- "perfdb_exact", "perfdb_interpolated", "analytical", "memory"
    
    -- Context at decision time
    slo_deadline_hours      REAL,
    objective               TEXT,     -- "cheapest", "fastest", "balanced"
    avg_input_tokens        INTEGER,
    avg_output_tokens       INTEGER,
    num_requests            INTEGER,
    
    -- Quota snapshot at decision time
    quota_snapshot          JSON,     -- {"A100-80GB": {"available": 32, "region": "us-west-2"}, ...}
    other_jobs_running      JSON,     -- [{"job_id": "x", "gpu_type": "L40S", "gpus": 16}, ...]
    
    -- Agent reasoning (structured, not free text)
    why_this_config         TEXT,     -- "PerfDB record #47: L40S TP=4 PP=2 gets 833 TPS at io_ratio=2"
    alternatives_considered JSON      -- [{"config": "A100 TP=8", "rejected_because": "2x cost for 1.8x speed"}]
);

-- What actually happened (ground truth, written on job completion)
CREATE TABLE outcomes (
    outcome_id      TEXT PRIMARY KEY,
    decision_id     TEXT REFERENCES decisions(decision_id),
    job_id          TEXT NOT NULL,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    status          TEXT NOT NULL,     -- "succeeded", "failed", "slo_met", "slo_missed"
    
    -- Actual performance
    actual_tps              REAL,
    actual_cost_per_hour    REAL,
    actual_total_cost       REAL,
    actual_runtime_hours    REAL,
    actual_tpot_ms          REAL,
    actual_cost_per_m_tokens REAL,
    
    -- Prediction error (the learning signal)
    delta_tps_pct           REAL,     -- (actual - predicted) / predicted * 100
    delta_cost_pct          REAL,
    slo_met                 BOOLEAN,
    slo_headroom_pct        REAL,     -- (slo_hours - actual_hours) / slo_hours * 100
    
    -- Failure analysis (CRITICAL — this is what makes the system learn)
    failure_reason          TEXT,     -- NULL if succeeded
    failure_category        TEXT,     -- "quota_exceeded", "oom", "throughput_below_estimate",
                                     -- "instance_unavailable", "launch_timeout", "spot_preemption",
                                     -- "kv_cache_pressure", "pipeline_bubble", "pcie_saturation"
    failure_detail          TEXT,     -- "A100-40GB TP=4: 144GB/4=36GB > 40GB VRAM"
    corrective_action       TEXT      -- "Use TP=8 or switch to A100-80GB"
);

-- Learned rules (extracted from patterns in outcomes)
CREATE TABLE rules (
    rule_id         TEXT PRIMARY KEY,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    rule_text       TEXT NOT NULL,     -- "Qwen-72B on A100-40GB requires TP>=8 (OOM at TP=4)"
    rule_type       TEXT NOT NULL,     -- "constraint", "preference", "causal"
    confidence      REAL,              -- 0-1, increases with confirming evidence
    evidence_count  INTEGER DEFAULT 1, -- how many outcomes support this rule
    
    -- Scope: when does this rule apply?
    model_pattern   TEXT,              -- "Qwen/Qwen2.5-72B*" or NULL for all
    gpu_pattern     TEXT,              -- "A100-40GB" or NULL for all
    workload_pattern TEXT,             -- "batch_prefill_heavy" or NULL for all
    
    -- Provenance
    derived_from    JSON               -- ["outcome_id_1", "outcome_id_2", ...]
);
```

**Memory is per-CHAIN, not per-job.**

A job can go through multiple GPU configurations (chains) during its lifetime. Each chain is a separate learning data point:

```
Job job-abc (SLO=8h, Qwen-72B):
  Chain 1: L40S TP=4 PP=2 DP=2    (0h → 3h, 833 TPS, then FALLING_BEHIND)
  Chain 2: A100-80GB TP=4 PP=2     (3h → 3.1h, A/B test probe)
  Chain 3: A100-80GB TP=4 PP=2     (3.1h → 5.5h, won A/B, completed)

Memory records:
  decision-1 → outcome: "L40S 833 TPS, fell behind at 3h" (chain 1)
  decision-2 → outcome: "A100-80GB 2400 TPS, A/B winner" (chain 2)
  decision-3 → outcome: "A100-80GB finished job, SLO met" (chain 3)
```

Each decision in the `decisions` table maps to one chain. The `outcomes` table records what happened for that specific chain, not the whole job. A `job_id` can have multiple `decision_id`s.

**Chain snapshots — periodic memory updates DURING execution:**

In addition to recording chain outcomes at end, Koi writes periodic snapshots of how each chain is performing. This creates a time series in memory:

```sql
CREATE TABLE chain_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    decision_id     TEXT REFERENCES decisions(decision_id),
    job_id          TEXT NOT NULL,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Performance at this moment
    throughput_tps          REAL,
    tokens_completed        INTEGER,
    tokens_remaining        INTEGER,
    elapsed_hours           REAL,
    slo_headroom_pct        REAL,

    -- GPU health at this moment
    gpu_cache_usage_pct     REAL,
    gpu_sm_util_pct         REAL,
    gpu_mem_bw_util_pct     REAL,
    num_requests_waiting    INTEGER
);
```

This is NOT the same as Orca's raw telemetry (1Hz, in-memory). Chain snapshots are sampled every **2-5 minutes** and written to SQLite. They persist across restarts and give the agent historical context: "L40S performance degraded from 850→600 TPS over 2 hours as KV cache filled up."

**How the agent uses memory:**

Memory returns TWO tiers of information, with different trust levels:

**Tier 1 — Outcomes (ground truth, highest trust):**
Completed jobs with actual measured performance. These are verified data points.
> "Qwen-72B on L40S TP=4 PP=2: **actual** 833 TPS, $33 total, SLO met ✓ (delta from prediction: -3.9%)"

**Tier 2 — Decisions without outcomes (predictions, moderate trust):**
Past decisions Koi made but jobs haven't completed yet (or no feedback received). Better than nothing — they show what Koi previously chose and at what confidence — but unverified.
> "Qwen-72B on L40S TP=4 PP=2: **predicted** 528 TPS @ $20.98/hr, confidence=82% — no outcome yet"

The agent should:
- If Tier 1 outcome exists → reuse that config with HIGH confidence (90%+)
- If only Tier 2 decisions exist → consider them but still verify against PerfDB/physics (same confidence as before)
- If no memory at all → full exploration from PerfDB + physics (lowest confidence)

**Agent workflow with memory:**

1. **Before deciding**: `query_memory(model="Qwen-72B")` → returns both outcomes AND decisions
2. **If outcomes exist**: "Last time this succeeded at 833 TPS on L40S TP=4" → reuse config, skip PerfDB
3. **If only decisions exist**: "I predicted 528 TPS but no verification yet" → check PerfDB to confirm, same confidence
4. **If failed outcomes exist**: "A100-40GB TP=4 OOMed" → avoid that config, check rules
5. **After chain ends**: `record_outcome(decision_id, actual_tps=...)` → promotes decision to ground truth
6. **Rule extraction**: After N outcomes with similar patterns, agent proposes a rule

**Launch outcome tracking (per-attempt, not per-job):**

The memory layer tracks not just "what config ran" but "what we tried to launch and whether it actually started." This is critical because quota ≠ availability. You can have 32 A100 vCPUs in quota but still fail to launch because the capacity isn't physically available in that region at that moment.

```sql
-- Every launch attempt, regardless of success
CREATE TABLE launch_attempts (
    attempt_id      TEXT PRIMARY KEY,
    decision_id     TEXT REFERENCES decisions(decision_id),
    job_id          TEXT NOT NULL,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    instance_type   TEXT NOT NULL,
    gpu_type        TEXT NOT NULL,
    region          TEXT NOT NULL,
    market          TEXT NOT NULL,     -- "on_demand", "spot"
    count           INTEGER NOT NULL,  -- how many instances requested
    
    -- Outcome
    launched        BOOLEAN NOT NULL,  -- did it actually start?
    time_to_launch  REAL,             -- seconds from request to running (NULL if failed)
    failure_reason  TEXT,             -- "InsufficientCapacity", "SpotInterrupted", "QuotaExceeded", "Timeout"
    
    -- Context: what else was happening
    quota_available INTEGER,          -- vCPUs available in quota at request time
    other_jobs_in_region JSON         -- what else was running in this region
);
```

This gives the agent a **soft availability signal** per (instance_type, region, time_of_day):

```
Agent queries: "In the last 24h, how often did p5.48xlarge actually launch in us-west-2?"

Results:
  6 attempts, 2 succeeded (33% success rate)
  Failures: 4× InsufficientCapacity (all between 10am-4pm EST)
  Successes: 2× launched (both after 8pm EST)

Agent reasons: "H100 capacity is scarce during business hours in us-west-2.
  It's 2pm now — don't propose H100, go with A100-80GB instead.
  Memory rule: 'H100 us-west-2 on-demand: ~30% availability daytime, ~80% evening'"
```

This is different from quota (which says "you CAN launch 16 H100 GPUs") vs availability (which says "but you probably WON'T get them right now"). Quota is static, availability is temporal and soft.

**Future: Orca streams launch analytics to Koi.** Currently Koi would only know about its own launch attempts. But Orca sees ALL launches across all users — including non-Koi launches. Orca should expose an analytics endpoint:

```
GET /analytics/launch_success_rate?instance_type=p5.48xlarge&region=us-west-2&window=24h
→ {"attempts": 14, "succeeded": 5, "rate": 0.36, "avg_time_to_launch": 180}
```

This turns Orca into a real-time availability oracle. Koi's agent uses this to discount configs on scarce instance types without wasting time on launch failures.

**Why not use mem0 or ChromaDB?**

Both are embedding stores — they compress structured data into vectors, losing precision. Koi's memory is relational: "was A100-40GB TP=4 tried for Qwen-72B? → yes, it OOMed." This is a SQL query, not a similarity search. The agent needs exact recall, not fuzzy matching.

ChromaDB is fine as an optional secondary store for natural-language lessons (e.g., "L40S pipeline bubbles are worse than NVLink at PP>2"). But the primary memory is structured SQL.

### Context 3: SLO Monitoring (Live Feedback)

**What:** Real-time signal — is the current deployment meeting the SLO?

**Source:** Orca's telemetry endpoints:
- `GET /job/{id}/metrics/stream` — SSE, 1Hz, throughput + TPOT + GPU metrics
- `GET /job/{id}/replicas/{rid}/metrics` — per-replica metrics (for A/B tests)
- `GET /job/{id}` — job status (running/succeeded/failed)
- `GET /job/{id}/chunks/progress` — chunk completion: total/pending/inflight/completed/failed

**The monitoring loop is NOT the agent.** The agent is expensive (LLM call per invocation). You don't want an LLM evaluating metrics every second. Instead:

```
┌─────────────────────────────────────────────────────┐
│              MONITORING INFRASTRUCTURE               │
│                  (no LLM, pure code)                 │
│                                                      │
│  Job Tracker (in-memory)                             │
│  ┌─────────────────────────────────────────────┐     │
│  │ job-abc123:                                  │     │
│  │   config: A100-80GB TP=4 PP=2               │     │
│  │   slo_deadline: 8.0h                         │     │
│  │   started_at: 2026-04-05T10:00:00           │     │
│  │   total_tokens: 7,500,000                    │     │
│  │   predicted_tps: 2590                        │     │
│  │   predicted_eta: 0.8h                        │     │
│  │                                              │     │
│  │   --- live (updated every 10s) ---           │     │
│  │   actual_tps: [2400, 2380, 2410, 2395, ...]  │     │
│  │   tokens_completed: 4,200,000                │     │
│  │   tokens_remaining: 3,300,000                │     │
│  │   elapsed_hours: 0.49                        │     │
│  │   projected_eta: 0.87h                       │     │
│  │   slo_headroom: 89%                          │     │
│  │   status: ON_TRACK                           │     │
│  │                                              │     │
│  │   --- GPU health ---                         │     │
│  │   gpu_cache_usage: 0.62                      │     │
│  │   gpu_sm_util: 78%                           │     │
│  │   gpu_mem_bw_util: 85%                       │     │
│  │   num_requests_waiting: 3                    │     │
│  └─────────────────────────────────────────────┘     │
│                                                      │
│  Polling loop (every 10s, per tracked job):          │
│    1. GET /job/{id}/metrics → update actual_tps      │
│    2. GET /job/{id}/chunks/progress → tokens done    │
│    3. Compute projected ETA:                         │
│       remaining_tokens / smoothed_tps / 3600         │
│    4. Compute SLO headroom:                          │
│       (slo_deadline - elapsed - projected) / slo     │
│    5. Classify status:                               │
│       headroom > 30%  → ON_TRACK (green)             │
│       headroom 10-30% → AT_RISK  (yellow)            │
│       headroom < 10%  → FALLING_BEHIND (red)         │
│                                                      │
│  Trigger rules (NO LLM needed):                      │
│    ON_TRACK → do nothing                             │
│    AT_RISK  → log warning, continue watching         │
│    FALLING_BEHIND → wake the agent                   │
│    COMPLETED → wake agent for record_outcome()       │
│    FAILED → wake agent for failure analysis          │
│                                                      │
└────────────────┬────────────────────────────────────┘
                 │
                 │ FALLING_BEHIND or COMPLETED or FAILED
                 ▼
┌─────────────────────────────────────────────────────┐
│                    KOI AGENT                         │
│              (only invoked when needed)              │
│                                                      │
│  On FALLING_BEHIND:                                  │
│    Agent receives: job state + last 60s of metrics   │
│    + GPU health + cluster state + memory             │
│                                                      │
│    Agent reasons:                                    │
│    "TPS dropped from 2400→1800 over last 5 min.     │
│     KV cache at 85% → cache pressure building.       │
│     gpu_mem_bw_util at 95% → bandwidth saturated.    │
│     This is a memory bandwidth bottleneck, not       │
│     compute. Adding same-GPU replicas won't help     │
│     (they'll hit the same BW ceiling).               │
│                                                      │
│     Options:                                         │
│     1. Scale out DP=2 (double throughput, double $)  │
│     2. A/B test H100 replica (3350 GB/s vs 2000)     │
│     3. Accept: even at 1800 TPS, ETA=1.16h < 8h SLO │
│                                                      │
│     Decision: Accept — still meeting SLO with 85%    │
│     headroom. Not worth the cost of scaling."        │
│                                                      │
│  On COMPLETED:                                       │
│    record_outcome(actual_tps, cost, slo_met, ...)   │
│    Extract rules if patterns detected                │
│                                                      │
│  On FAILED:                                          │
│    record_outcome(status=failed, reason=...)        │
│    Agent diagnoses: OOM? Spot preemption? Timeout?  │
│    Stores corrective action in rules table           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**How it tracks multiple jobs:**

The monitoring infrastructure is a simple Python process (part of the Koi server) that maintains an in-memory dict of tracked jobs. When `/decide` returns a config and Orca launches it, the job gets registered:

```python
# In Koi server, after /decide
tracked_jobs[job_id] = {
    "config": decision.config,
    "slo_deadline_hours": request.slo_deadline_hours,
    "total_tokens": request.num_requests * (request.avg_input_tokens + request.avg_output_tokens),
    "predicted_tps": decision.predicted_tps,
    "started_at": datetime.now(),
    "metrics_buffer": deque(maxlen=360),  # 1hr of 10s samples
}
```

Every 10 seconds, a background asyncio task polls Orca for each tracked job, updates the state, and checks the trigger rules. **No LLM calls in the loop** — just arithmetic (`remaining_tokens / smoothed_tps`). The agent only wakes up when something needs a decision.

**Three separate async loops with DIFFERENT periods:**

These are architecturally distinct. Do NOT conflate them.

```
┌─────────────────────────────────────────────────────────────────┐
│ LOOP 1: TELEMETRY POLLING (10s)                                  │
│   Pure code, no LLM, no DB writes.                               │
│   Fetches raw metrics from Orca, updates in-memory JobTracker.   │
│   Computes SLO headroom. Checks trigger thresholds.              │
│   Cost: ~0 (one HTTP GET per job per 10s)                        │
│                                                                   │
│ LOOP 2: MEMORY SNAPSHOT (2-5 min)                                │
│   Pure code, no LLM. Writes current chain performance to         │
│   chain_snapshots table in SQLite. Creates time series per chain. │
│   Also records launch failures immediately (not periodic).        │
│   Cost: ~0 (SQLite write)                                        │
│                                                                   │
│ LOOP 3: AGENT FEEDBACK (event-driven, NOT periodic)              │
│   LLM invocation. ONLY fires on triggers:                        │
│   - FALLING_BEHIND → agent diagnoses + proposes scale/swap       │
│   - OVER_PROVISIONED → agent suggests replicas to kill           │
│   - CHAIN_END → agent records chain outcome in memory            │
│   - JOB_COMPLETE → agent records final outcome + extracts rules  │
│   - LAUNCH_FAILED → agent records failure + adjusts strategy     │
│   Cost: ~$0.05 per invocation (only when needed)                 │
│                                                                   │
│ KEY: Loops 1 and 2 run CONTINUOUSLY at their intervals.          │
│      Loop 3 runs ONLY on events. They do NOT share a timer.      │
└─────────────────────────────────────────────────────────────────┘

Timeline for a healthy job:
  t=0s    [L1] poll metrics → WARMING_UP
  t=10s   [L1] poll metrics → WARMING_UP
  ...
  t=300s  [L1] poll metrics → ON_TRACK (warmup complete)
  t=300s  [L2] snapshot chain → write to chain_snapshots
  t=310s  [L1] poll metrics → ON_TRACK
  ...
  t=600s  [L2] snapshot chain → write to chain_snapshots
  ...
  t=3600s [L1] all chunks done → COMPLETED
  t=3600s [L3] AGENT WAKES → record_outcome, extract rules ← ONLY LLM call

Timeline for a struggling job:
  t=300s  [L1] poll → ON_TRACK
  t=600s  [L2] snapshot → TPS=800
  t=900s  [L2] snapshot → TPS=650 (degrading!)
  t=1000s [L1] poll → FALLING_BEHIND (headroom < 10%)
  t=1000s [L3] AGENT WAKES → diagnoses KV cache pressure → proposes A/B test
  t=1000s [L2] record: chain 1 ending (agent initiated scale-up)
  t=1020s [L1] new replica detected → track both chains
  t=1200s [L2] snapshot chain 1 (old): 500 TPS | snapshot chain 2 (new): 2400 TPS
  t=1300s [L3] AGENT WAKES → A/B complete → kill chain 1, keep chain 2
  t=1300s [L2] record: chain 1 outcome (lost A/B), chain 2 promoted
```

**Chain lifecycle events that update memory (Loop 2 + immediate):**

| Event | When | What's written | Loop |
|-------|------|----------------|------|
| Chain starts | Replica(s) running, first metrics arrive | `decisions` row (predicted config) | Immediate |
| Chain snapshot | Every 2-5 min while chain is alive | `chain_snapshots` row (TPS, headroom, GPU health) | Loop 2 |
| Launch failed | Instance couldn't start | `launch_attempts` row (failure_reason) | Immediate |
| Chain ends (swap/kill) | Replicas terminated | `outcomes` row (actual TPS, delta, per-CHAIN) | Immediate |
| Job completes | All chunks done | `outcomes` row for final chain + job-level summary | Loop 3 (agent) |

**Orca needs to tell Koi about chain lifecycle events.** Specifically:
- When replicas launch successfully → chain started (Koi gets this from first metrics ingest)
- When replicas are killed/swapped → chain ended (Koi detects this from replica state change in telemetry)
- When job completes → already handled via webhook `POST /job/complete`

No additional Orca API needed — Koi infers chain start/end from the replica state changes it already monitors in Loop 1.

**When does the agent (Loop 3) wake up?**

| Trigger | What happens | Agent cost |
|---------|-------------|------------|
| `FALLING_BEHIND` | Agent reads metrics + physics, decides whether to scale/swap/accept | ~$0.05 per invocation |
| `OVER_PROVISIONED` | Agent suggests replicas to kill | ~$0.03 |
| `CHAIN_END` | Agent records chain outcome in memory (may not need LLM — can be rule-based) | ~$0.02 or $0 |
| `JOB_COMPLETE` | Agent records final outcome, extracts rules | ~$0.02 |
| `LAUNCH_FAILED` | Agent records failure, adjusts strategy for retry | ~$0.03 |
| `ON_TRACK` | Nothing. No agent call. | $0 |

For a typical 8-hour batch job that runs smoothly: 0 agent calls during monitoring. One call at completion. Total monitoring cost: $0.02.

For a job that hits trouble twice: 2 agent calls during monitoring + 1 at completion = ~$0.12. Still trivial compared to the GPU cost.

---

## 6. The Decision Loop

### Initial Placement

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 0. IDENTIFY MODEL                                │
│    get_model_arch(model_name)                    │
│    → Known model? Use arch from registry/perfdb  │
│    → Unknown? Fetch HF config.json               │
│    → Can't fetch? Estimate from name (70B→Llama) │
│                                                  │
│    query_perfdb(model=model_name)                │
│    → Has data? Great, use it directly            │
│    → No data? find_similar_models(model_name)     │
│      Fetches HF config → computes physics vector: │
│        model_size_gb, kv_bytes/token, FLOPs/fwd,  │
│        gqa_ratio, num_layers, num_heads, is_moe   │
│      Ranks PerfDB models by physics distance:      │
│        "Llama-70B: dist=0.02 (144GB, GQA=8, 80L)  │
│         DeepSeek-R1-70B: dist=0.04 (same class)    │
│         Qwen3-32B: dist=0.38 (too small, skip)"   │
│      Agent uses Llama-70B benchmarks as proxy      │
│      with confidence scaled by distance            │
│                                                  │
│ 1. GATHER CONTEXT                                │
│    query_perfdb(model, gpu_types, io_ratio)      │
│    query_memory(model, status=*)                 │
│    get_resources()                               │
│    get_gpu_physics(available_gpus)               │
│                                                  │
│ 2. REASON                                        │
│    Agent analyzes:                               │
│    - PerfDB records → what configs work          │
│    - Memory → what we've tried, what failed      │
│    - Resources → what's available                │
│    - Physics → why certain configs are better    │
│    - SLO → what throughput we need               │
│                                                  │
│ 3. DECIDE                                        │
│    Agent proposes ranked configs:                │
│    [1] A100-80GB TP=4 PP=2 — $1.46/M tokens     │
│    [2] A100-40GB TP=8 PP=1 — $1.51/M tokens     │
│    [3] L40S TP=4 PP=2 DP=2 — $4.54/M tokens     │
│                                                  │
│ 4. LAUNCH                                        │
│    launch_chain(config[0])                       │
│    → Orca deploys instances                      │
│                                                  │
│ 5. MONITOR                                       │
│    Loop: get_job_metrics(job_id)                 │
│    If bad → go to Adaptation                     │
│    If done → go to Learning                      │
└─────────────────────────────────────────────────┘
```

### Adaptation (A/B Testing)

When monitoring detects the SLO is at risk:

```
SLO at risk (throughput below target)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. DIAGNOSE                                      │
│    Agent reads live metrics + physics:            │
│    "GPU mem BW util at 95% → bandwidth-bound     │
│     L40S only has 864 GB/s, need more bandwidth  │
│     A100 has 2000 GB/s"                          │
│                                                  │
│ 2. PROPOSE REPAIR                                │
│    query_perfdb(model, gpu="A100-80GB")          │
│    → A100-80GB TP=8 PP=1 gets 1498 TPS           │
│                                                  │
│ 3. A/B TEST (not hard swap)                      │
│    scale_chain(job_id, new_config, count=1)      │
│    → Orca launches A100 replica alongside L40S   │
│    → Both process chunks from same Redis queue   │
│                                                  │
│ 4. COMPARE (2 min observation window)            │
│    get_job_metrics(job_id, per_replica=True)     │
│    L40S replica: 450 tok/s                       │
│    A100 replica: 1320 tok/s  ← winner            │
│                                                  │
│ 5. RESOLVE                                       │
│    Kill L40S replicas                            │
│    record_outcome(L40S: underperformed)          │
│    record_outcome(A100: confirmed at 1320 TPS)   │
│    → Memory now has ground truth for BOTH configs │
└─────────────────────────────────────────────────┘
```

**Key insight from AlphaEvolve:** The A/B test is a real-world evaluation of two population members. Both get measured on the same fitness function (chunk completion throughput). The winner survives, the loser is culled. Double the learning data per intervention.

### Opportunistic Exploration (Future — gated by SLO headroom)

*Inspired by [AdaEvolve](https://arxiv.org/html/2602.20133): when you're making progress, use the surplus to explore.*

Adaptation fires when things go wrong. Exploration fires when things go **right**. If the primary chains are crushing the SLO with 50%+ headroom, that surplus is wasted capacity — or it's free exploration budget.

```
Monitoring detects: SLO headroom > 50%
(primary chains finishing way ahead of deadline)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ OPPORTUNISTIC EXPLORATION                            │
│ (ONLY when SLO is comfortably met)                   │
│                                                      │
│ 1. WHAT TO EXPLORE                                   │
│    Agent picks an untested config based on:           │
│                                                      │
│    Momentum (direction of recent improvements):      │
│    ┌──────────────────────────────────────────────┐  │
│    │ Recent exploration history:                   │  │
│    │   t=1: L40S TP=4 → 631 TPS ($3/hr)          │  │
│    │   t=2: L40S TP=8 → 863 TPS ($13/hr)         │  │
│    │   t=3: A100-40GB TP=8 → 1498 TPS ($33/hr)   │  │
│    │                                               │  │
│    │ Momentum signal: "throughput improved each    │  │
│    │ time we moved toward higher-bandwidth GPUs.   │  │
│    │ Direction = more bandwidth."                  │  │
│    │                                               │  │
│    │ Next exploration: A100-80GB TP=4 PP=2         │  │
│    │ (same bandwidth direction, but TP=4 instead   │  │
│    │  of 8 — can we get better $/token?)           │  │
│    │                                               │  │
│    │ Anti-momentum: "3 L4 explorations all had     │  │
│    │ vram_headroom=0.00 → stop exploring L4 for    │  │
│    │ 72B models"                                   │  │
│    └──────────────────────────────────────────────┘  │
│                                                      │
│    Curiosity (unexplored regions):                   │
│    "Never tried H100 for this model — high           │
│     uncertainty, high potential. Worth a probe."     │
│                                                      │
│ 2. HOW TO EXPLORE (cheap, disposable)                │
│    Launch exploratory chain on SPOT instances:       │
│    - 60-90% cheaper than on-demand                   │
│    - If spot preempted → fine, it was exploratory    │
│    - Primary chains on on-demand never touched       │
│    - Only need 2-5 minutes of data for benchmarks   │
│                                                      │
│    scale_chain(job_id, {                             │
│      gpu: "A100-80GB", tp: 4, pp: 2,                │
│      market: "spot",   ← cheap, disposable          │
│      count: 1                                        │
│    })                                                │
│                                                      │
│ 3. EVALUATE (2 min observation)                      │
│    Compare exploratory replica vs primary replicas:  │
│    - Exploratory A100-80GB TP=4: 2400 TPS, $5.12/hr │
│    - Primary A100-40GB TP=8:     1498 TPS, $4.10/hr │
│    - $/M tokens: exploratory=$0.59 vs primary=$0.76  │
│                                                      │
│ 4. PROMOTE OR KILL                                   │
│    Exploratory wins on $/token?                      │
│    → PROMOTE: make it a primary chain                │
│      (switch from spot to on-demand if needed)       │
│      Kill the more expensive primary chain           │
│      Net effect: same throughput, lower cost         │
│                                                      │
│    Exploratory loses?                                │
│    → KILL: terminate, record in memory               │
│      "A100-80GB TP=4 PP=2: 2400 TPS but $0.59/M    │
│       vs current $0.76/M — actually better, but     │
│       spot preemption risk for primary is too high.  │
│       Keep for future spot-tolerant jobs."           │
│                                                      │
│    Either way → learning data for memory             │
│                                                      │
│ 5. UPDATE MOMENTUM                                   │
│    If exploration improved $/token:                  │
│    → strengthen momentum in that direction           │
│    → next exploration goes further (try H100?)       │
│                                                      │
│    If exploration was worse:                         │
│    → dampen momentum in that direction               │
│    → try orthogonal direction next (PP instead of    │
│      GPU swap? quantization? different DP?)          │
│                                                      │
│    If region exhausted (3 failures in same area):    │
│    → anti-momentum: mark region as explored,         │
│      stop wasting budget there                       │
└─────────────────────────────────────────────────────┘
```

**Momentum in configuration space:**

There's no gradient, no loss function. But there IS a signal in the trajectory of exploration outcomes. Momentum tracks which *direction* in configuration space has been productive:

| Dimension | What "moving in this direction" means | Momentum signal |
|-----------|--------------------------------------|-----------------|
| GPU bandwidth | L4 → L40S → A100 → H100 | "each step up improved $/token" |
| TP degree | TP=2 → TP=4 → TP=8 | "TP=8 saturated PCIe, momentum dies" |
| PP degree | PP=1 → PP=2 → PP=4 | "PP=2 was fine, PP=4 added bubble overhead" |
| DP (scale-out) | DP=1 → DP=2 → DP=4 | "linear scaling, keep going if quota allows" |
| Quantization | User-specified (fp16 default). Koi does NOT auto-quantize. | Orca needs `--quantization fp8/int8` flag. Koi respects user's choice, uses it to filter PerfDB records. |
| Market | on-demand → spot | "spot is 70% cheaper, 3/10 jobs get preempted" |

The agent maintains a simple momentum table in memory:

```sql
CREATE TABLE exploration_momentum (
    model_class     TEXT,           -- "dense_70b", "moe_200b"
    dimension       TEXT,           -- "gpu_bandwidth", "tp_degree", "dp_scale"
    direction       TEXT,           -- "increase", "decrease"
    recent_delta    REAL,           -- avg improvement from last 3 explorations in this direction
    explore_count   INTEGER,        -- how many times explored
    last_explored   DATETIME,
    status          TEXT            -- "promising", "saturated", "avoid"
);
```

**Why this matters for multi-tenancy:** In a shared cluster, Koi serves many users running different models. Each job that explores on spots generates learning data that benefits ALL future jobs with similar models. The exploration cost ($0.50 for 5 minutes on a spot A100) is amortized across hundreds of future decisions. The more jobs Koi handles, the faster it fills memory, the less it needs to explore, the better it gets.

**Piggyback exploration — heterogeneous replicas from day one:**

Even before SLO headroom gating, there's a cheaper form of exploration: when the agent decides to launch DP=N replicas, make N-1 the "safe" config and 1 the "probe."

```
Agent decides: A100-80GB TP=4 PP=2 DP=4 (4 replicas)

Instead of:  4× A100-80GB TP=4 PP=2 (all identical)
Launch:      3× A100-80GB TP=4 PP=2 (safe majority)
           + 1× L40S TP=4 PP=2      (probe — also meets SLO solo, cheaper)

All 4 pull chunks from the same Redis queue.
After 5 minutes:
  A100 replicas: 2400 TPS each → 7200 TPS combined
  L40S replica:  833 TPS
  Total: 8033 TPS (need 260 TPS for SLO → massive headroom)

Learning: L40S gets 833 TPS at $13.35/hr → $4.46/M tokens
          A100 gets 2400 TPS at $40.96/hr → $4.74/M tokens
          L40S is actually CHEAPER per token!
→ Next job: agent knows to use L40S, not A100
```

**Rules for the heterogeneous replica:**
- Must individually meet SLO at DP=1 (or close enough that the safe majority absorbs the gap)
- Prefer configs the memory has LOW confidence on (high learning value)
- Prefer SPOT for the probe replica (cheapest exploration)
- The safe majority guarantees the SLO; the probe is bonus learning
- FIFO chunk routing self-balances: slow probe gets fewer chunks, fast primaries compensate

**When to piggyback vs when not to:**
- `DP >= 3` → always piggyback 1 replica (safe majority still ≥ 2)
- `DP = 2` → piggyback only if SLO headroom > 40% (one failure replica = 50% throughput loss)
- `DP = 1` → never piggyback (no safety net)

**The SLO gate is non-negotiable:** Exploration ONLY fires when:
- `slo_headroom > 50%` (projected to finish with 2× time to spare)
- Spot quota is available (doesn't steal on-demand from primary chains)
- Exploration budget not exhausted (max 2 exploratory chains per job)
- The config hasn't been explored in the last 24h for this model class

### Learning (Post-Job)

```
Job completes (succeeded or failed)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. RECORD OUTCOME                                │
│    record_outcome(                               │
│      decision_id, actual_tps, actual_cost,       │
│      slo_met, failure_reason_if_any              │
│    )                                             │
│                                                  │
│ 2. COMPUTE PREDICTION ERROR                      │
│    delta = actual_tps - predicted_tps            │
│    → "Oracle overestimated by 12% for this       │
│       model on A100-40GB TP=8"                   │
│                                                  │
│ 3. EXTRACT RULES (if pattern detected)           │
│    Agent reviews recent outcomes:                │
│    "3 of last 5 A100-40GB jobs overestimated     │
│     by 10-15% → rule: discount A100-40GB         │
│     analytical estimates by 12%"                 │
│                                                  │
│ 4. UPDATE FRONTIER                               │
│    If this config is cheapest SLO-meeting seen   │
│    → new Pareto frontier point                   │
│    Next job knows "bar to beat = $1.51/M tokens" │
└─────────────────────────────────────────────────┘
```

**This is the evolutionary loop.** Job N's outcome enriches the context for Job N+1. After 50 jobs, Koi has a rich memory of what works for each model × GPU × workload class. The agent barely needs to think — it just looks up the answer from memory.

**Completed jobs feed Memory, not PerfDB.** PerfDB is the controlled profiling corpus (isolated benchmarks, reproducible conditions). Memory captures production reality (shared cluster, variable load, spot preemptions). The agent sees both and reasons about the gap:
- PerfDB: "A100-80GB TP=4 PP=2 gets 2590 TPS (isolated)"
- Memory: "In production we got 2100 TPS (3 other jobs on cluster)"
- Agent: "22% contention penalty — adjust estimate for current cluster state"

Exception: jobs that ran in isolation (no other jobs on cluster) can optionally be backfilled into PerfDB as `source: "production_isolated"` — they're equivalent to controlled benchmarks.

---

## 7. GPU Physics Layer

The agent needs to understand WHY configs perform the way they do. Raw benchmark numbers aren't enough — the agent needs causal reasoning.

**GPU specs the agent can query:**

```python
# get_gpu_physics("A100-80GB") returns:
{
    "gpu_type": "A100-80GB",
    "vram_gb": 80.0,
    "bandwidth_gbps": 2000,        # HBM2e
    "fp16_tflops": 312,
    "interconnect": "NVLink",
    "nvlink_bw_gbps": 600,
    "pcie_bw_gbps": 31.5,
    "generation": "Ampere",
    "fp8_native": False,           # No FP8 on Ampere
    "cost_per_gpu_hour": 5.12,
    
    # Derived for a specific model
    "for_model": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "model_size_gb": 144.0,
        "weight_per_gpu_tp4": 36.0,   # 144 / 4
        "weight_per_gpu_tp8": 18.0,   # 144 / 8
        "vram_headroom_tp4": 44.0,    # 80 - 36 = 44GB for KV cache
        "vram_headroom_tp8": 62.0,    # 80 - 18 = 62GB
        "bandwidth_per_param_tp4": 110.0,  # (2000 * 4) / 72
        "bandwidth_per_param_tp8": 220.0,  # (2000 * 8) / 72
        "bottleneck_tp4": "compute",   # bandwidth_per_param < ridge_point
        "bottleneck_tp8": "memory",    # bandwidth_per_param > ridge_point
        "roofline_tps_tp4": 3500,
        "roofline_tps_tp8": 6200,
    }
}
```

**This enables causal reasoning:**
- "L40S TP=4 only gets 833 TPS while A100-80GB TP=4 gets 2590 TPS → why? L40S bandwidth=864 GB/s vs A100=2000 GB/s, and this workload is bandwidth-bound at TP=4 (decode phase). That's a 2.3× bandwidth ratio explaining the ~3× throughput gap."
- "A100-40GB TP=4 OOMs but A100-80GB TP=4 works → 144GB/4 = 36GB per GPU, A100-40GB only has 40GB → 4GB headroom isn't enough for KV cache at concurrency=256."

---

## 8. Multi-Tenancy (Next Version)

Single-user Koi picks the globally cheapest config. Multi-tenant Koi must solve a harder problem: **resource allocation across competing jobs on a shared cluster.**

### The Problem

```
User A: "Run Qwen-72B, 100K requests, 8hr SLO, cheapest"
User B: "Run Llama-70B, 10K requests, 1hr SLO, fastest"
User C: "Run Qwen-32B, 500K requests, 24hr SLO, cheapest"

Available: 160 L40S GPUs, 64 A100 GPUs, 16 H100 GPUs
```

Optimal for A alone: A100-80GB TP=4 PP=2 (8 GPUs)
Optimal for B alone: H100 TP=8 PP=1 (8 GPUs) — fastest
Optimal for C alone: L40S TP=4 PP=2 DP=4 (32 GPUs) — cheapest over 24h

But if B takes 8 H100 GPUs, they're all gone. And A and C compete for A100s.

### The Approach

**Cluster-aware agent.** The agent sees not just available quota but also:
- What jobs are currently running and their resource usage
- Each job's SLO urgency (how much slack remains)
- Predicted completion times for each job

**The agent reasons about allocation:**

```
"User B has 1hr SLO with 10K requests — urgent. H100 TP=8 finishes in 0.2h.
 User A has 8hr SLO — plenty of slack. L40S TP=4 PP=2 finishes in 2.5h.
 User C has 24hr SLO — very flexible. Can use whatever's left.
 
 Allocate: B→H100 (8 GPUs), A→L40S (8 GPUs), C→remaining L40S (32 GPUs).
 Total cluster utilization: 56/240 GPUs active. If A finishes first, 
 reassign its L40S GPUs to C to speed up."
```

**The memory layer enables this.** After 50 multi-tenant runs, Koi learns patterns like:
- "When cluster is >70% utilized, L40S jobs get 15% lower throughput (PCIe contention from co-located jobs)"
- "H100 jobs should be short and high-priority — the NVLink fabric doesn't share well"
- "L4 is good for background/low-priority work — cheapest and doesn't compete for A100/H100 quota"

---

## 9. Comparison to Koi v1

| Aspect | Koi v1 | Koi v2 |
|--------|--------|--------|
| **Decision engine** | Fixed pipeline: Oracle → 3 LLM Thinkers → Judge | Single agent with tools (Claude Agent SDK) |
| **PerfDB retrieval** | FAISS embedding similarity (RAG) | Agent-driven SQL queries (structured) |
| **Memory** | ChromaDB embeddings + SQLite delta store | Structured SQL: decisions, outcomes, rules |
| **Monitoring** | Kalman filter + deadband controller | Agent reads raw metrics, reasons about cause |
| **Adaptation** | Hard swap (linear) | A/B test: launch repair alongside, compare, cull loser |
| **Physics** | Embedded in prompt text | First-class tool: `get_gpu_physics()` |
| **Latency** | 150-300s (multiple LLM calls) | 5-30s (single agent, cached for repeats) |
| **Multi-tenancy** | Not supported | Cluster-aware agent with job-level arbitration |
| **Learning** | Passive (stores deltas) | Active: extracts rules, adjusts predictions, explores |

---

## 10. Implementation Plan

### Phase 1: Core Agent + Monitoring + Scale Up/Down (THIS VERSION)

**Koi:**
- [ ] `koi/agent.py` — KoiAgent built on Claude Agent SDK with tool definitions
- [ ] `koi/tools/perfdb.py` — PerfDB query tool (pandas over CSV)
- [ ] `koi/tools/memory.py` — Agentic Memory (decisions, outcomes, rules, launch_attempts tables)
- [ ] `koi/tools/resources.py` — Live resource map from Orca
- [ ] `koi/tools/physics.py` — GPU specs + bottleneck analysis + physics-vector similarity
- [ ] `koi/tools/orca_api.py` — Launch, scale, kill, metrics from Orca
- [ ] `koi/monitor.py` — Background polling loop (10s), SLO tracking, agent triggers
- [ ] `koi/server.py` — HTTP endpoints: `/decide`, `/health`, `/job/complete`
- [ ] `koi/schemas.py` — Pydantic models for all data structures
- [ ] Scale UP: A/B test when FALLING_BEHIND (launch repair replica, compare, cull loser)
- [ ] Scale DOWN: shed excess replicas when over-provisioned (headroom > 70%)
- [ ] Spot preemption recovery: detect dead replica → decide whether to replace based on SLO math
- [ ] Warm-up detection: ignore first 5 min of metrics before SLO projections
- [ ] Verbose evolutionary trace logging (`--verbose`)

**Orca (new branch `koi-compat` off `main`):**
- [ ] `GET /resources` endpoint — instance catalog + quota pools (fix A100 40/80 VRAM)
- [ ] Fix A100 40GB vs 80GB in `gpu_specs.py` + `config.py`
- [ ] `call_koi` timeout to 300s (thread join + HTTP)
- [ ] Per-replica metrics for heterogeneous replica types (verify A/B test works)
- [ ] Final metrics fallback in `_assemble_output()` — ring buffer if summaries missing
- [ ] `POST /job/complete` webhook to `KOI_SERVICE_URL`
- [ ] `--quantization` flag in CLI (fp16 default)

### Future (NOT this version)

**Multi-tenancy:**
- [ ] Cluster state tracker (all jobs, all resources, all SLOs)
- [ ] Priority-based allocation in agent reasoning
- [ ] Resource rebalancing (reassign GPUs when jobs complete)

**Opportunistic exploration:**
- [ ] SLO headroom-gated exploration on spot instances (AdaEvolve-inspired)
- [ ] Momentum tracking in configuration space
- [ ] Piggyback exploration (heterogeneous replica in DP≥3)

**Budget enforcement:**
- [ ] User specifies max spend, agent tracks cost accrual and scales down to stay under

**Launch availability analytics:**
- [ ] Orca `GET /analytics/launch_success_rate` endpoint
- [ ] Soft availability model per (instance_type, region, time_of_day)

**Agent framework swap:**
- [ ] Evaluate LangGraph, OpenAI Agents SDK, OpenClaw for model-agnostic agent layer

**PerfDB upgrade:**
- [ ] Migrate from CSV to SQLite when >10K records

---

## 11. Batch-Specific Details

### Scale DOWN, not just up (Phase 1)

Scaling down is as important as scaling up for `objective: cheapest`. When over-provisioned, shed excess replicas.

Scenario: Agent launches DP=4 to meet an 8hr SLO. After 1 hour, 40% of tokens are done. Projected ETA: 2.5h. That's 5.5 hours of headroom — you're burning 4× GPU cost for no reason.

```
Monitoring detects: slo_headroom > 70% AND elapsed > 20% of SLO
    │
    ▼
Agent reasons: "At current 4×A100 rate, finishing in 2.5h.
  SLO is 8h. I can kill 2 replicas, run on 2×A100,
  finish in ~5h, and save 50% on GPU cost."
    │
    ▼
scale_chain(job_id, count=-2)  → kill 2 replicas
Record: "Learned: for 5K requests on Qwen-72B, DP=2 is sufficient for 8hr SLO"
```

This is the **cheapest** objective in action — not just picking the cheapest config upfront, but actively shedding excess capacity mid-job.

### Spot preemption recovery (Phase 1)

Batch jobs on spots WILL get preempted. When a replica dies, agent decides whether to replace based on SLO math.

```
Monitoring detects: replica-3 stopped reporting metrics for 30s
    │
    ▼
Check: is it spot preemption? (Orca watchdog reports "dead" + market="spot")
    │
    ▼
Agent decides:
  - Remaining tokens: 2M. Current TPS with surviving replicas: 1800.
  - ETA without recovery: 0.31h. SLO: 8h. Headroom: 96%.
  - Don't recover — 3 replicas are enough.
  OR
  - Remaining tokens: 5M. Surviving replicas: 1. TPS: 600. ETA: 2.3h. SLO: 3h.
  - MUST recover. Launch replacement (try same spot, fallback to on-demand).
```

The Orca watchdog already handles replica death and chunk reclamation. Koi's role is deciding WHETHER to replace (based on SLO math) and WHAT to replace with (same GPU? different? on-demand fallback?).

### Cost tracking and budget enforcement

The doc tracks $/token after the fact but doesn't enforce a BUDGET. User should be able to say "finish in 8h, spend at most $50."

```
User: model=Qwen-72B, 5K requests, SLO=8h, budget=$50

Agent: "A100-80GB TP=4 PP=2 gets 2590 TPS → ETA=0.8h → cost=$32.77.
  Under budget. But if I use L40S TP=4 PP=2 → ETA=2.5h → cost=$33.37.
  Both under budget, but L40S is $0.60 more total and 3× slower.
  Pick A100 — faster AND cheaper."

Monitoring: if cost accrual exceeds pace (tracking $/hr × elapsed),
  agent can scale down or swap to cheaper GPU to stay under budget.
```

### Warm-up period handling

The first 2-5 minutes of a vLLM job are misleading — CUDA compilation, model loading, KV cache warming. Throughput ramps from 0 → steady state over this period. The monitoring loop needs to know:
- Don't trigger FALLING_BEHIND during warm-up
- Don't use warm-up throughput for SLO projection
- Wait for `steady_state_detected` (throughput variance < 10% over 60s window) before making decisions

```python
# In monitoring loop
if elapsed_minutes < 5 and throughput_variance > 0.15:
    status = "WARMING_UP"  # don't trigger agent
```

---

## 12. Open Questions

1. **Claude Agent SDK for now, abstracted for swap.** Claude's tool use and retrieval quality is unmatched today — use it. But wrap the agent behind an `AgentLLM` interface so we can swap to open-source models (Llama, Qwen) or other SDKs later without rewriting the tools. The tools themselves are model-agnostic (SQL queries, HTTP calls) — only the reasoning layer cares which LLM is driving. Future: evaluate LangGraph (production-mature, model-agnostic, persistent state), OpenAI Agents SDK (supports 100+ models since v0.10), and OpenClaw (fast-growing OSS agent, but more consumer-facing — needs assessment for structured tool dispatch). Key criteria: tool use quality on structured data, latency, cost, and ability to run locally.

2. **PerfDB stays as CSV for now.** 300 records is fine for pandas. Agent's `query_perfdb` tool reads CSV via pandas filters. Migrate to SQLite when we hit 10K+ records or need joins across tables.

3. **Memory retention: 30-day rolling window + monthly summary.** Raw outcomes kept for 30 days (full detail for recent decisions). Every 30 days, agent summarizes old outcomes into rules and frontier updates, then archives raw records. Rules persist indefinitely but carry a `last_confirmed` timestamp — agent weights recent confirmations higher. Stale rules (not confirmed in 90 days) get demoted, not deleted.

4. **Agent cost per decision: acceptable.** ~$0.05-0.10 per decision, $50-100/day at 1000 jobs. Trivial compared to GPU savings from better placement (a single $40/hr → $14/hr improvement on one job pays for a day of agent costs).

5. **Observability: verbose evolutionary trace for demos.** Every decision must be traceable — not just the final config, but the full reasoning chain. `--verbose` mode prints:
   ```
   [Koi] Job job-abc123: Qwen/Qwen2.5-72B-Instruct, 5K reqs, SLO=8h, objective=cheapest
   [Koi] Memory hit: 3 past outcomes for this model
   [Koi]   outcome-1: A100-80GB TP=8 PP=1 → 1498 TPS, $45 total, SLO met ✓
   [Koi]   outcome-2: L40S TP=4 PP=2 → 833 TPS, $33 total, SLO met ✓ ← cheapest known
   [Koi]   outcome-3: A100-40GB TP=4 → FAILED (OOM) — rule: "TP>=8 on A100-40GB for 72B"
   [Koi] PerfDB: 12 direct records, 8 proxy (Llama-70B, dist=0.02)
   [Koi] Resources: 80 L40S, 32 A100-80GB, 16 H100 available
   [Koi] Physics: io_ratio=0.93 → balanced workload, bandwidth matters
   [Koi] Agent decision: L40S TP=4 PP=2 DP=1 — $33 total (cheapest from memory)
   [Koi]   confidence: 92% (memory-backed, 2 prior successes)
   [Koi]   alternative rejected: A100-80GB TP=8 ($45, 36% more expensive)
   [Koi] Launching via Orca...
   [Koi] Monitoring: ON_TRACK (2400 TPS, headroom=85%)
   [Koi] Piggyback probe: launched 1× A10G TP=4 PP=4 (spot, $5.67/hr)
   [Koi] Probe result: A10G gets 280 TPS at $5.67/hr → $5.63/M tokens (worse than L40S $4.46)
   [Koi] Probe killed. Memory updated: "A10G too slow for 72B batch at this io_ratio"
   [Koi] Job completed: 833 TPS actual, $33.12 total, SLO met ✓ (2.5h / 8h)
   [Koi] Evolution: memory now has 4 outcomes for Qwen-72B. Frontier: L40S TP=4 PP=2 @ $4.46/M
   ```
   This trace is the demo. It shows memory recall, physics reasoning, exploration, and learning in one job. Saved to a log file per job for post-hoc analysis.

6. **Exploration budget: user-configurable.** Default 10% of jobs can be exploratory. User can set `--explore-budget 0.2` (aggressive learning) or `--explore-budget 0` (no exploration, pure exploit). Exposed as a parameter in the `/decide` request, stored per-user in memory.
