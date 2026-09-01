# Koi paper ablations

Three arms, one codebase. Both ablations are boot-time flags on the normal
runner; the full system is the default. The flags (and the resolved ablation
state) are stamped into every run's `run_manifest` event in
`logs/koi/<run-id>/events.jsonl`, so every trace is self-describing.

| Arm | Command addition | Branch with the default pre-set |
| --- | --- | --- |
| Full Koi | *(nothing)* | `main` / `tandemn_ablations` |
| Koi without DAG/mechanisms | `--mechanism-mode inert` (or `KOI_MECHANISM_MODE=inert`) | `ablation_no_dag` |
| Koi without online learning | `--learning-mode frozen` (or `KOI_LEARNING_MODE=frozen`) | `ablation_no_learning` |

The convenience branches differ from `tandemn_ablations` by exactly one line
(the flag's default), so a collaborator can check one out and run the standard
command from the README unchanged:

```bash
OPENAI_API_KEY='...' TANDEMN_USER_ID='usr_...' \
.venv/bin/python -m src.orchestrator.runner \
  --ticks 0 --tick-interval-sec 300 --telemetry-window-sec 300 \
  --trace all --rust-log warn \
  --mechanism-mode inert        # or: --learning-mode frozen
```

## ⚠️ Experimental hygiene: fresh Store user per run

`init_causal_graph` imports the seed tables only when the Store is **empty**
for that `user_id`; otherwise it loads whatever is there — including Beta
confidences mutated by a previous full-Koi run. **Every ablation run needs a
fresh `TANDEMN_USER_ID`** (or a documented Store reset). Without this, the
"no online learning" arm silently starts from another run's *learned*
posteriors and the ablation is contaminated.

Do not mix arms under one user id, ever.

## What `--mechanism-mode inert` disables (the causal DAG/mechanisms)

Mechanism identity carries **no decision content**; the graph is inert, not
absent (the planner's hard gates require a resolvable `mechanism_id`, so a
single pass-through sentinel mechanism is registered at boot and stamped on
every rank).

- **EIG ≡ 0.** `compute_eig` and the inline `beta * eig` term of sigma return
  exactly 0 — sigma becomes pure exploitation
  (`agent_tools.compute_eig`, `_compute_sigma`).
- **Mechanism selection replaced by the sentinel.** `_applicable_mechanism_id`
  returns the pass-through id; scope matching, confidence ranking, and the
  `no_mechanism` candidate veto never run
  (`agent_tools._applicable_mechanism_id`, `agent.py:_validate_ladder`).
- **Confidence is never read.** Knob ranking, mechanism briefs, and Q-label
  summaries are withheld.
- **DAG tools hidden from the LLM**: `get_edge_confidence`,
  `get_mechanism_confidence`, `get_influencing_knobs`, `get_scope`,
  `get_applicable_mechanisms`, `get_recent_q_histogram`, `compute_eig`,
  `check_past_failure`, `set_new_mechanisms`, `val_new_mechanisms`
  (`agent_tools._MECHANISM_INERT_HIDDEN_TOOLS`).
- **Prompts stripped.** Root and specialist prompts carry no mechanism/EIG
  vocabulary; job briefs and similar-deployment briefs omit mechanism fields.
- **No mechanism is ever created**: the admission tool is hidden *and* refuses.

Untouched: candidate enumeration (`build_scored_candidates`), the surrogate,
Tchebycheff/DRO/switch-cost scoring, C0–C6 validation, budgets, joint
placement, dead-shape memory, and all of S2/S3 learning (which keeps running
but is never read — keeping this arm orthogonal to the learning arm).

## What `--learning-mode frozen` disables (online learning)

Nothing updates from observations; every knob and prior keeps its boot value.

- **S3 LEARN is skipped entirely**: no DRO residual ingestion, no Beta
  confidence updates (and no Store flush), no slow-loop knob updates
  (`beta_t`, `B_t`, `lambda_swit`, `epsilon_dro`, `w_t`, `z_star_t`), no
  target annealing, no CUSUM recalibration (`fsm_states.TickRunner.S3`).
- **Surrogate calibration/fusion severed.** The composer is built without an
  evidence store and `bind_tools` never rebinds one, so `calibrate_prediction`
  and `learn_throughput_fusion` no-op — including the catastrophic-miss fast
  path (`runner.build_runner`, `agent_tools.bind_tools`).
- **Dead-shape memory disabled.** Starved-streak tracking, dead-shape
  recording, and `observed_dead_shapes` annotation are all off — an observed
  failure never biases future candidates (`fsm_states.S0`/`S2`).
- **No mechanism admission** (growing the causal model mid-run is learning).
- **EIG still runs, from the frozen seed priors.** The exploration bonus
  exists but never updates — this is deliberate, and is what keeps this arm
  distinct from the inert-DAG arm (which zeroes EIG).

Kept running: `evidence_store.append_row` (traces stay analyzable; nothing
consumes the rows for learning), reactive health/SWAP eligibility, deployment
retry/backoff bookkeeping (control, not learning).

Known consequence to note in the paper: the C4 swap-budget validator reads
`slow_state.B_t`, which stays at its boot default `B_MAX = 10` all run, and
`z_star_t`/`epsilon_dro` keep their seed values.

## Verifying an arm

- The `run_manifest` event carries `config.mechanism_mode`,
  `config.learning_mode`, and an `ablation` block (with the sentinel id in
  inert mode).
- Inert arm: every place/swap rank carries the same pass-through
  `mechanism_id`; per-job sigma diagnostics show `eig = 0`.
- Frozen arm: `slow_update_diagnostics` is `{"learning_frozen": true}` every
  tick, and edge/mechanism alpha/beta in the Store never move off their seeds.
- Smoke tests: `tests/smoke/test_ablation_modes_smokes.py` (the agent-layer
  cases need the full dependency stack: `aiconfigurator`, `tandemn-store`).
