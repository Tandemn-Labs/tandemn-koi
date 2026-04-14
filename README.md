# Koi

**Optional intelligence and runtime-control layer for Tandemn Orca batch inference jobs.**

Koi is not the data plane. Orca still launches, runs, and completes jobs on its own. If you do **not** set `KOI_SERVICE_URL`, Orca continues to work standalone with its normal placement stack.

When Koi is enabled, it adds three things on top of Orca:

1. **Placement decisions** via `POST /decide`
2. **Runtime control** for chunked jobs via Orca webhooks and Orca telemetry
3. **Learning** from real launches, failures, and completed runs

## What Koi Does Today

Current `rewrite` branch behavior:

- Orca CLI can call Koi for a placement recommendation and ranked alternatives.
- Orca carries Koi metadata into the **chunked** launch path.
- Orca calls back into Koi on:
  - `/job/config-attempted`
  - `/job/launching`
  - `/job/launch-heartbeat`
  - `/job/started`
  - `/job/launch-failed`
  - `/job/replica-failed`
  - `/job/complete`
- Koi tracks pending launches, renews pending GPU leases while launch is still active, monitors running jobs, and can scale existing chunked jobs through Orca.
- Koi persists deterministic runtime state to disk so restart does not wipe pending reservations or tracked jobs.

What Koi does **not** do yet:

- It does **not** replace Orca.
- It does **not** directly launch new jobs autonomously. Initial launch is still CLI-driven through Orca.
- It does **not** yet reconcile restored runtime state against Orca on startup.

## Orca Integration

There are two valid ways to run Tandemn today:

### 1. Orca standalone

Do nothing special. Leave `KOI_SERVICE_URL` unset.

Orca still supports:

- roofline placement
- optional advisor / performance-db placement
- chunked multi-replica execution
- dashboard, analytics, watchdogs, and lifecycle management

### 2. Orca with Koi enabled

Set `KOI_SERVICE_URL` for the Orca CLI / server side and point Koi back at Orca with `ORCA_URL`.

Typical flow:

1. Orca CLI calls `POST /decide`
2. Koi returns:
   - `config`
   - `predicted_tps`
   - `predicted_cost_per_hour`
   - `predicted_runtime_hours`
   - `predicted_total_cost`
   - `alternatives`
   - `_decision_id`
3. Orca launches the chosen config, optionally retrying Koi-provided alternatives in the chunked path
4. Orca reports progress and outcomes back to Koi through webhooks
5. Koi updates memory, resource leases, and monitoring state

## Runtime State vs Agentic Memory

Koi has **two different persistence layers**, on purpose:

### `AgenticMemory`

Append-only learning history:

- decisions
- outcomes
- launch attempts
- availability priors

This is where Koi learns from production.

### `RuntimeStateStore`

Overwrite-style control-plane state:

- tracked jobs
- pending launches
- pending scale decisions
- pending GPU reservations

This is **not** LLM memory. It exists so deterministic server/monitor/ledger state survives restart.

Only deterministic code writes to `RuntimeStateStore`. The Anthropic agent may influence decisions, but it should not directly mutate this control-plane state.

## Launch Lease Model

Koi uses a `ResourceLedger` to reserve GPUs between `/decide` and successful launch.

That reservation is now a **renewable lease**, not a one-shot timeout:

- `/decide` creates a pending reservation
- `/job/launch-heartbeat` refreshes the lease while Orca is still searching / provisioning / bootstrapping
- `/job/started` releases the reservation and registers the running replica
- `/job/launch-failed` releases the reservation if the launch never succeeded

This prevents Koi from reopening capacity too early during slow launches.

## Quick Start

### 1. Install

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Configure

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # required for real /decide decisions
export ORCA_URL=http://localhost:26336     # required for live monitoring / scale actions
export KOI_PORT=8090
```

Useful optional paths:

```bash
export KOI_MEMORY_PATH=./data/koi_memory.db
export KOI_RUNTIME_STATE_PATH=./data/koi_runtime.db
export KOI_PERFDB_PATH=./perfdb/perfdb_all.csv
```

### 3. Run Koi

```bash
python -m koi.server
```

### 4. Point Orca at Koi

In the Orca shell / environment:

```bash
export KOI_SERVICE_URL=http://localhost:8090
```

If `KOI_SERVICE_URL` is not set, Orca stays standalone.

## HTTP API

Core endpoints:

- `POST /decide`
- `POST /job/config-attempted`
- `POST /job/launching`
- `POST /job/launch-heartbeat`
- `POST /job/started`
- `POST /job/launch-failed`
- `POST /job/replica-failed`
- `POST /job/complete`
- `GET /health`
- `GET /jobs`

Operationally:

- `/job/launching` creates or updates pending launch state
- `/job/launch-heartbeat` refreshes the pending lease and launch phase
- `/job/started` converts pending launch state into tracked runtime state
- `/job/complete` records the final outcome and unregisters the job

## Key Files

- [koi/server.py](/home/orange/Desktop/tandemn/koi/koi/server.py): FastAPI service, webhook handling, startup restore
- [koi/agent.py](/home/orange/Desktop/tandemn/koi/koi/agent.py): Claude-driven decision and scale logic
- [koi/monitor.py](/home/orange/Desktop/tandemn/koi/koi/monitor.py): runtime polling, trigger generation, tracked-job state
- [koi/resource_ledger.py](/home/orange/Desktop/tandemn/koi/koi/resource_ledger.py): pending GPU lease tracking
- [koi/runtime_state.py](/home/orange/Desktop/tandemn/koi/koi/runtime_state.py): deterministic runtime-state persistence
- [koi/tools/memory.py](/home/orange/Desktop/tandemn/koi/koi/tools/memory.py): append-only learning memory
- [ARCHITECTURE.md](/home/orange/Desktop/tandemn/koi/ARCHITECTURE.md): longer architecture notes and current implementation caveats
- [simulation/run_sim_tests.py](/home/orange/Desktop/tandemn/koi/simulation/run_sim_tests.py): integration and control-loop simulation

## Tests

Run the normal test suite:

```bash
.venv/bin/pytest tests/ -q
```

Run the simulation suite:

```bash
.venv/bin/python simulation/run_sim_tests.py
```

LLM-driven simulation tiers require `ANTHROPIC_API_KEY`.

## Current Caveats

- Orca can still run perfectly fine without Koi.
- Koi runtime state now survives restart, but startup reconciliation with Orca is still a separate project.
- Durable Orca webhook outbox / retry is still not implemented.
- Initial launch is still CLI-driven; agent-driven `launch_chain` for new jobs is not the current production path.
