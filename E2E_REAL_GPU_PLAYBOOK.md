# Real GPU Test Playbook: Koi + Orca

## Setup

### Environment Variables
```bash
# Required
export ANTHROPIC_API_KEY="sk-ant-..."    # Koi now fails fast if missing (unless KOI_TEST_FAKE_DECIDE=1)
export KOI_EXCLUDE_GPUS="A100-40GB,A100-80GB,H100,H200"
export ORCA_URL="<tunnel-url>"
export KOI_PORT=8090
export HF_TOKEN="hf_..."

# Outbox (new, Phase 4). Default is ./state/outbox.db. Set to empty-string
# to disable the durable path (legacy fire-and-forget fallback kicks in).
export ORCA_OUTBOX_DB_PATH="./state/outbox.db"

# Optional tuning
export KOI_DECIDE_TIMEOUT=300
export KOI_TRIGGER_TIMEOUT=180
export KOI_WARMUP_MINUTES=5
export KOI_LLM_MODEL=claude-sonnet-4-6
export KOI_INBOX_RETENTION_DAYS=14      # new, Phase 2b; must be ≥ outbox retention
```

### Available GPUs (after exclusion)

| GPU | Instance | VRAM | $/hr/gpu | PerfDB Records |
|-----|----------|------|----------|----------------|
| L40S | g6e.12xlarge | 48 GB | ~$4.10 | 92 (Qwen3-32B=44, Qwen2.5-72B=36, Qwen3-235B=12) |
| L4 | g6.12xlarge | 24 GB | ~$1.50 | 88 (Qwen3-32B=56, Qwen2.5-72B=32) |
| A10G | g5.12xlarge | 24 GB | ~$1.30 | 0 (roofline only) |

### 3-Terminal Launch

```bash
# Terminal 1: Orca (in Tandemn-orca repo, orca/contract-hardening branch)
KOI_SERVICE_URL=http://localhost:8090 python server.py --tunnel

# Terminal 2: Koi (this repo, koi/contract-hardening branch)
KOI_EXCLUDE_GPUS=A100-40GB,A100-80GB,H100,H200 \
ANTHROPIC_API_KEY=sk-ant-... \
ORCA_URL=<tunnel-url-from-terminal-1> \
HF_TOKEN=hf_... \
KOI_PORT=8090 \
.venv/bin/python -m koi.server

# Terminal 3: sim_ctl + curl
python simulation/sim_ctl.py --koi http://localhost:8090 --orca <tunnel-url>
```

### Monitoring Commands
```bash
# Koi: overall health + hardening counters
curl -s localhost:8090/health | python3 -m json.tool
curl -s localhost:8090/jobs | python3 -m json.tool
curl -s localhost:8090/resources | python3 -m json.tool

# Orca: outbox backlog visibility (new in Phase 4d)
curl -s <orca-url>/health | python3 -m json.tool
```

---

## Hardening Invariants (Phase 1–6)

The changes on `koi/contract-hardening` and `orca/contract-hardening` introduce invariants
the scenarios below check for. These are the signals that tell you the new delivery
pipeline is working correctly — not just that the old flow still completes.

### What should always be true (steady state)

On **Orca `/health`**:
- `outbox_enabled: true`
- `outbox_pending: 0` within seconds of any lifecycle event
- `outbox_oldest_undelivered_age_secs: 0` or small

On **Koi `/health`**:
- `status: "ok"` (never `"fatal"` — if you see fatal, a monitor task died)
- `inbox_processed` grows monotonically through a run
- `inbox_processing == 0` at steady state (non-zero means a handler is mid-flight right now)
- `stale_inbox_claims == 0` (non-zero means a handler crashed and the claim is stuck)

### What to inspect on-disk after a run

```bash
# Orca outbox — no undelivered rows should remain after a completed run
sqlite3 ./state/outbox.db \
  "SELECT event_type, COUNT(*) FROM outbox WHERE delivered_at IS NULL GROUP BY event_type"
# → empty

# Orca outbox — audit trail of the run
sqlite3 ./state/outbox.db \
  "SELECT event_id, event_type, attempts, last_status_code FROM outbox ORDER BY created_at DESC LIMIT 20"
# → each event_id has dedup_key shape (e.g. 'replica_failed:mo-abc-r0'),
#   most attempts=1, status 200

# Koi inbox — every event processed once
sqlite3 ./data/koi_runtime.db \
  "SELECT status, COUNT(*) FROM inbox GROUP BY status"
# → all 'processed', none 'processing'

# Koi outcomes — unique by (decision_id, job_id, status)
sqlite3 ./data/koi_memory.db \
  "SELECT decision_id, job_id, status, COUNT(*) AS n FROM outcomes
   GROUP BY decision_id, job_id, status HAVING n > 1"
# → empty (unique index enforces this; WHERE NOT EXISTS guards legacy DBs)
```

---

## Scenario R1: Happy Path Revalidation (after hardening)
**Purpose**: Verify the full delivery pipeline works end-to-end with real GPUs.
**Cost**: ~$1.50 | **Time**: 20 min | **Model**: Qwen3-32B

```bash
orca batch --model Qwen/Qwen3-32B \
  --input examples/workloads/stress_5000.jsonl \
  --slo 2 --use-koi
```

### Legacy checkpoints (from before hardening)
- [ ] Koi picks L40S (not A100/H100)
- [ ] `/job/started` webhook fires after model_ready (~8 min)
- [ ] `/jobs` shows ON_TRACK, smoothed_tps > 0
- [ ] Job completes, outcome recorded
- [ ] `/health` shows memory_outcomes incremented

### NEW — Hardening invariants
- [ ] **Koi startup validates ANTHROPIC_API_KEY** — if you set an empty
      or malformed key, Koi fails fast with a clear RuntimeError. With
      a valid `sk-ant-` prefixed key, startup proceeds.
- [ ] **Envelope on every webhook**: `tail -f /tmp/koi_e2e.log` should
      show entries with `event_id=...` like
      `event_id=job_launching:<rid>:0`, `event_id=job_started:<rid>`,
      `event_id=job_complete:<group_id>`.
- [ ] **Outbox drains after each lifecycle event**:
      ```bash
      curl -s <orca-url>/health | jq .outbox_pending
      # Expect: 0 within ~2s of each /job/* webhook being emitted
      ```
- [ ] **Inbox processes events**: Koi `/health` shows `inbox_processed`
      growing through the run:
      - after /job/config-attempted: +1 (per attempt)
      - after /job/launching: +1
      - after /job/launch-heartbeat: +N (every 45s)
      - after /job/started: +1
      - after /job/complete: +1
- [ ] **No stuck handlers**: `stale_inbox_claims == 0` throughout.
- [ ] **Per-chain outcome idempotency**: after completion, the
      `outcomes` table has exactly 1 row per chain (or 1 total for a
      single-chain job). Run the `GROUP BY ... HAVING n > 1` query above
      — must be empty.
- [ ] **Truthful /health through the run**: Koi `/health` status stays
      `"ok"` — never flips to `"fatal"`. If it does, a monitor task died.

### Phase coverage
R1 exercises: Phase 1 (contract), 2a (schemas), 2b (inbox claim), 2c
(handler wrap), 2d (outcome idempotency), 2e (truthful /health), 2f
(API-key gate), 4a/4b/4c/4d (outbox + publisher + migration + health).
Does **not** exercise: scale correlation (→ R3), replica-failed dedup (→
R4), long outage recovery (→ fault-injection unit tests).

---

## Scenario R2: Memory Learning (2 Sequential Jobs)
**Purpose**: First-ever test of outcome-to-decision feedback loop
**Cost**: ~$3 | **Time**: 40 min | **Model**: Qwen3-32B

```bash
# Job 1
orca batch --model Qwen/Qwen3-32B \
  --input examples/workloads/stress_5000.jsonl \
  --slo 2 --use-koi
# Wait for completion...

# Job 2 (same model, same workload)
orca batch --model Qwen/Qwen3-32B \
  --input examples/workloads/stress_5000.jsonl \
  --slo 2 --use-koi
```

### Checkpoints
- [ ] Job 1 outcome recorded with actual TPS
- [ ] Job 2 decision logs mention "MEMORY" or prior outcome
- [ ] Job 2 predicted_tps closer to Job 1 actual TPS
- [ ] Decision confidence higher on Job 2

### NEW — Hardening invariants
- [ ] **Retry idempotency**: if you stop Koi between jobs and restart,
      any outbox events buffered during the restart drain cleanly, and
      Koi's inbox dedups them (no duplicate outcomes for Job 1 after
      restart).

---

## Scenario R3: FALLING_BEHIND → Scale-Up
**Purpose**: Force monitor trigger, verify agent scales up with correct
decision correlation.
**Cost**: ~$3 | **Time**: 30 min | **Model**: Qwen3-32B

```bash
# Tight SLO: 0.3h = 18 min for 6M tokens → needs ~5556 TPS
# Single L40S does ~3000-5000 TPS → falls behind
orca batch --model Qwen/Qwen3-32B \
  --input examples/workloads/stress_5000.jsonl \
  --slo 0.3 --use-koi
```

### Legacy checkpoints
- [ ] Status: WARMING_UP → ON_TRACK → AT_RISK → FALLING_BEHIND
- [ ] Trigger fires (trigger_type=FALLING_BEHIND)
- [ ] Agent proposes scale-up via `scale_chain_tool`
- [ ] Orca `/job/{id}/scale` returns 200 with `new_replicas` list
- [ ] New replica appears in `/jobs`
- [ ] Aggregate TPS improves

### NEW — Hardening invariants (Phase 3)
- [ ] **Per-replica decision mapping**: Koi's scale_chain_tool logs
      `register_pending_replica_decision(replica_id=<rid>, decision_id=<did>)`
      for each ID Orca returned — not a single FIFO queue entry.
- [ ] **Exact lookup on /job/started**: when the new replica fires
      `/job/started`, Koi's log shows `consume_pending_replica_decision`
      called with the exact replica_id, matching the decision_id from
      scale. No FIFO misattribution possible.
- [ ] **Runtime state survives restart mid-scale**: stop Koi between
      scale_chain_tool call and the new replica's /job/started; restart
      Koi; the pending replica decision is restored from
      `koi_runtime.db → pending_replica_decisions`. Verify:
      ```bash
      sqlite3 ./data/koi_runtime.db "SELECT * FROM pending_replica_decisions"
      ```

---

## Scenario R4: Replica Kill + Recovery
**Purpose**: Test failure detection → agent recovery, with the Phase 5
guarantee-fire + dedup under real failures.
**Cost**: ~$5 | **Time**: 45 min | **Model**: Qwen3-32B

```bash
# Launch with DP=2 (2 replicas)
orca batch --model Qwen/Qwen3-32B \
  --input examples/workloads/stress_50000.jsonl \
  --slo 6 --dp 2 --use-koi
```

### Phase 1: Steady state (~15 min)
- [ ] Both replicas tracked in `/jobs`
- [ ] Aggregate TPS = sum of 2 replicas (~6000-10000 TPS)
- [ ] Status: ON_TRACK, headroom > 50%
- [ ] Orca `/health`: `outbox_pending: 0`

### Phase 2: Kill a replica (sim_ctl or manual)
```bash
sim> kill <replica_id>
# OR: manually terminate EC2 instance via AWS console
```

### Legacy checkpoints
- [ ] Orca watchdog detects heartbeat timeout
- [ ] `/job/replica-failed` webhook fires to Koi
- [ ] Dead replica TPS drops to 0
- [ ] Headroom drops, status changes to AT_RISK or FALLING_BEHIND
- [ ] FAILED trigger fires
- [ ] Agent proposes scale-up replacement
- [ ] New replica launches, tracked by Koi

### NEW — Hardening invariants (Phase 5)
- [ ] **Exactly one `/job/replica-failed` event in Orca outbox** with
      `event_id=replica_failed:<replica_id>` — even though both
      watchdog and `monitor_replica` detect the death independently.
      `INSERT OR IGNORE` collapses them:
      ```bash
      sqlite3 ./state/outbox.db \
        "SELECT event_id, attempts FROM outbox WHERE event_type='replica_failed'"
      # → One row per dead replica, not two.
      ```
- [ ] **Structured reason_code on the failure**: Koi logs show
      `reason_code=heartbeat_timeout` or `log_stream_error` or
      `clean_exit_pending_chunks`, not just free-text `reason`.
- [ ] **Exactly one processed inbox row per replica death**:
      ```bash
      sqlite3 ./data/koi_runtime.db \
        "SELECT event_id, status FROM inbox WHERE event_type='replica_failed'"
      # → All 'processed', one per dead replica.
      ```
- [ ] **Guarantee-fire fallback (if triggered)**: if `monitor_replica`
      exits an unforeseen code path, Koi receives a failure with
      `reason_code=monitor_thread_exited`. This shouldn't fire in a
      normal kill scenario, but if it does, the system still recovers
      (Koi scales up a replacement).

### Phase 3: Verify recovery (~15 min)
- [ ] New replica reaches model_ready, TPS flowing
- [ ] Aggregate TPS recovers to pre-kill level
- [ ] Status returns to ON_TRACK
- [ ] Outcome records failure + recovery chain

---

## Scenario R5: Spot Preemption → On-Demand Fallback
**Purpose**: Real spot preemption handling with Beta-prior learning
**Cost**: ~$4 | **Time**: 60 min | **Model**: Qwen3-32B

```bash
orca batch --model Qwen/Qwen3-32B \
  --input examples/workloads/stress_5000.jsonl \
  --slo 2 --use-koi --spot
# Wait for AWS preemption OR manually kill spot instance
```

### Checkpoints
- [ ] Initial launch on spot instances
- [ ] Preemption detected (heartbeat timeout)
- [ ] `/job/replica-failed` webhook with `reason_code=spot_preemption`
      (NEW: Phase 5 structured reason code)
- [ ] Beta prior updates (availability drops below 50%)
- [ ] Agent proposes on_demand fallback
- [ ] Recovery on on-demand instance
- [ ] Cost table in next decision shows degraded spot availability

---

## Scenario R6: Concurrent Multi-Job
**Purpose**: Two jobs competing for same GPU pool
**Cost**: ~$6 | **Time**: 40 min | **Models**: Qwen3-32B + Qwen2.5-72B

```bash
# Job 1
orca batch --model Qwen/Qwen3-32B \
  --input examples/workloads/stress_5000.jsonl \
  --slo 2 --use-koi

# Job 2 (immediately after)
orca batch --model Qwen/Qwen2.5-72B-Instruct \
  --input examples/workloads/stress_5000.jsonl \
  --slo 4 --use-koi
```

### Checkpoints
- [ ] ResourceLedger shows pending GPUs for both jobs
- [ ] No resource double-booking
- [ ] Both jobs tracked independently in `/jobs`
- [ ] Both complete with outcomes recorded

### NEW — Hardening invariants
- [ ] **Event IDs distinct across jobs**: `sqlite3 ./state/outbox.db
      "SELECT DISTINCT job_id FROM outbox"` shows both job IDs;
      no cross-contamination.
- [ ] **Per-job outcome isolation**: memory `outcomes` table has two
      distinct decision_ids, one per job.

---

## Scenario R7: L4 Cheapest Path
**Purpose**: Validate L4 GPU path (never tested on real hardware)
**Cost**: ~$1 | **Time**: 30 min | **Model**: Qwen3-32B

```bash
# Restart Koi with L40S also excluded
KOI_EXCLUDE_GPUS=A100-40GB,A100-80GB,H100,H200,L40S \
ANTHROPIC_API_KEY=sk-ant-... ORCA_URL=<url> HF_TOKEN=hf_... \
KOI_PORT=8090 .venv/bin/python -m koi.server
```

```bash
orca batch --model Qwen/Qwen3-32B \
  --input examples/workloads/stress_5000.jsonl \
  --slo 4 --use-koi  # relaxed SLO, L4 is slow
```

### Checkpoints
- [ ] Agent picks L4, TP=4, PP=4
- [ ] Model loads on L4 (24GB VRAM, needs 4+ GPUs)
- [ ] TPS in 500-1200 range
- [ ] Job completes
- [ ] (hardening) All R1-style invariants also pass on the L4 path

---

## Quick hardening verification (any scenario)

Run this after any scenario completes. Green output = all Phase 1–6
invariants held on the real wire.

```bash
#!/bin/bash
# verify_hardening.sh — sanity check after an E2E run
set -u
KOI_URL="${KOI_URL:-http://localhost:8090}"
ORCA_URL="${ORCA_URL:?set ORCA_URL}"
OUTBOX_DB="${ORCA_OUTBOX_DB_PATH:-./state/outbox.db}"
KOI_RUNTIME_DB="${KOI_RUNTIME_DB:-./data/koi_runtime.db}"
KOI_MEMORY_DB="${KOI_MEMORY_DB:-./data/koi_memory.db}"

echo "=== Koi /health ==="
curl -sf "$KOI_URL/health" | python3 -m json.tool || { echo "FAIL: Koi /health unreachable"; exit 1; }

echo
echo "=== Orca /health ==="
curl -sf "$ORCA_URL/health" | python3 -m json.tool || { echo "FAIL: Orca /health unreachable"; exit 1; }

echo
echo "=== Outbox drain check ==="
PENDING=$(sqlite3 "$OUTBOX_DB" "SELECT COUNT(*) FROM outbox WHERE delivered_at IS NULL")
echo "outbox undelivered: $PENDING"
[ "$PENDING" = "0" ] || echo "⚠ non-zero backlog — check Orca /health"

echo
echo "=== Inbox stuck-handler check ==="
STUCK=$(sqlite3 "$KOI_RUNTIME_DB" "SELECT COUNT(*) FROM inbox WHERE status='processing'")
echo "inbox processing (stuck): $STUCK"
[ "$STUCK" = "0" ] || echo "⚠ non-zero stuck handlers — some event is mid-flight or crashed"

echo
echo "=== Per-chain outcome uniqueness ==="
DUPES=$(sqlite3 "$KOI_MEMORY_DB" "SELECT COUNT(*) FROM (
  SELECT decision_id, job_id, status, COUNT(*) n FROM outcomes
  GROUP BY decision_id, job_id, status HAVING n > 1
)")
echo "duplicate outcome groups: $DUPES"
[ "$DUPES" = "0" ] || { echo "FAIL: duplicate outcomes found"; exit 1; }

echo
echo "=== Monitor task liveness ==="
FATAL=$(curl -sf "$KOI_URL/health" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('fatal','none'))")
echo "monitor _fatal: $FATAL"
[ "$FATAL" = "none" ] || { echo "FAIL: monitor task died: $FATAL"; exit 1; }

echo
echo "✓ All hardening invariants held."
```

Save as `tools/verify_hardening.sh`, `chmod +x`, run after each scenario.

---

## Recommended Execution Order

For verifying the hardening:

| Phase | Scenarios | Cost | Time | Hardening coverage |
|-------|-----------|------|------|---------------------|
| A | R1 | ~$1.50 | 20 min | Envelope + outbox + inbox + /health |
| B | R1 → R4 | ~$6.50 | ~65 min | Add dedup + guarantee-fire |
| C | R1 → R3 → R4 | ~$9.50 | ~95 min | Add scale correlation |
| D (full) | R1 → R2 → R3 → R4 → R5 → R7 | ~$19 | ~4 hr | Everything |

**Minimum to validate the hardening in the wild: Phase A (R1)** at $1.50.
**Maximum confidence before shipping: Phase C** at ~$9.50.

---

## Prior Test History

| Test | Date | Outcome | Key Finding |
|------|------|---------|-------------|
| Test 1-2 | Apr 2-4 | Failed | Spot preemption during model load, HF 504 |
| Test 3 | Apr 5 | Pass | Full loop closed: decision → launch → complete → outcome |
| Test 4 | Apr 5 | Partial | HF timeout, fixed with HF_TOKEN + S3 weights |
| Test 5 | Apr 6 | Pass | Fallback PP=4→PP=2, on-demand, GPU exclusion |
| Test 6 | Apr 6-7 | Pass | Multi-replica DP=2, manual kill (no auto-recovery) |
| **Hardening** | **Apr 19** | **Pending R1** | 690 tests green; ready to validate on real GPUs |

### Hardening pass (Phase 1–6, Apr 19)

- Phase 1 — shared contract (EventEnvelope, ReasonCode, TERMINAL_PHASES)
- Phase 2a — Koi webhook schemas accept envelope (Optional fields)
- Phase 2b — Inbox state machine with atomic claim_event
- Phase 2c — Wrap handlers in _run_with_inbox (mark_processed only on success)
- Phase 2d — Append-only outcome idempotency via WHERE NOT EXISTS
- Phase 2e — Truthful /health (catches clean task exits, not just exceptions)
- Phase 2f — Conditional ANTHROPIC_API_KEY validation
- Phase 3 — Per-replica scale correlation (exact replica_id map, no FIFO)
- Phase 4a/b/c/d — Orca outbox + publisher + 10 call-site migration + /health
- Phase 5 — monitor_replica guarantee-fire + terminal-phase normalization
- Phase 6 — Cross-repo integration tests + AgenticMemory thread-safety fix

Branches: `koi/contract-hardening`, `orca/contract-hardening`.
