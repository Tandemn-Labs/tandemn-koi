# Tandemn Intelligence (Koi)

**Turning heterogeneous GPU fleets across clouds and environments into a single, self-optimizing inference cluster.**

> Koi is a causal, self-calibrating planner for LLM inference fleets.
Koi is our centralized algorithmic planner, creating an execution plan for the system to deploy via Kubernetes, Ray, Dynamo, Skypilot, etc
> 

---

> The initial draft of our white paper outlining the algorithm is available here: https://www.tandemn.com/blog/koi
> 

---

## What is Tandemn?

Modern inference fleets span many clouds, regions, and GPU types across **reserved, on-demand, and spot** capacity. Yet the way each job actually runs — hardware, engine, parallelism, quantization — is still chosen **by hand**.

**Tandemn places those jobs into one logical cluster and adds the missing piece: the optimizer.**

For every job, the central algorithm:

1. **predicts** how each candidate configuration will behave,
2. **scores** it against per-job SLOs and cost,
3. **explores** its own uncertainty where it matters, and
4. **continually improves** with every single deployment.

---

## The problem we're solving

A real inference fleet looks like this:

- dozens to hundreds of concurrent jobs, some online (latency SLOs), some batch (deadline SLOs);
- many tenants with different quotas, priorities, budgets, and data-isolation rules;
- heterogeneous hardware across clouds, regions, markets, GPU types, instance types, engines, and network topologies;
- volatile spot capacity, launch failures, chain deaths, and degraded replicas;
- and a constant pressure to hit SLOs at the lowest possible cost.

The hard part isn't picking a GPU for one model. It's **simultaneous cluster control under constraints** — moving tenant A's low-priority batch job off scarce H100s so tenant B's latency-critical job can meet its SLO, without thrashing the fleet.

Doing that by hand is not feasible. Koi makes it a continuous, learned, auditable decision.

---

## Reasoning Logic

Koi runs as a **deterministic control loop with a reasoning planner at its core**. The principle is strict:

> **The planner proposes. Deterministic code validates and executes.** The model never performs a side effect directly.
> 

Every tick (default ~5 minutes, or on demand) walks along a fixed path `S0 → S7`:

```
events / telemetry / scheduler signals
                │  (recorded as facts, never as reflex prompts)
        ┌───────▼─────────┐
        │  Runtime State  │   clusters · tenants · jobs · resource map · evidence
        └───────┬─────────┘
                │
 ┌──────────────▼──────────────────────────────────────────────┐
 │            S0 → S7  ·  Deterministic Tick Spine             │
 │                                                             │
 │  S1 OBSERVE  →  S2 VALIDATE  →  S3 LEARN (slow loop)        │
 │      →  S4 PLAN  (Koi reasoning planner, budget-first)      │
 │          →  S5 CHECK (C0–C7 validators)                     │
 │              →  S6 DEPLOY  →  S7 EXIT                       │
 └──────────────┬──────────────────────────────────────────────┘
                │  one validated, typed Plan
        ┌───────▼──────────┐
        │  Tandemn-System  │   executes on  Ray · Dynamo · Kubernetes
        └──────────────────┘
```

### The planning loop: predict → score → explore → validate → learn

Koi is **causal** and **self-calibrating** by construction:

- **Predict.** A surrogate model predicts each candidate config's outcomes — TTFT, TPOT, throughput, cost — *before* anything is deployed.
- **Score.** A **Tchebycheff scalarization** collapses multi-objective trade-offs (per-job SLOs + cost) into a single, comparable score. A **distributionally-robust (DRO)** check keeps SLO promises honest under prediction uncertainty, and a **switch-cost** term discourages needless churn.
- **Explore.** **Expected Information Gain (EIG)** steers limited canary/exploration capacity toward the configs that will teach the planner the most — so uncertainty shrinks where it actually matters.
- **Validate.** A deterministic constraint hierarchy (`C0–C7`: tenant policy → resource capacity → physical feasibility → SLO chance → swap budget → admission → repair → side-effect gating) gates every plan. Nothing reaches the executor unvalidated.
- **Learn (self-calibrate).** After deployment, observed-vs-predicted trajectories flow through **CUSUM** (drift detection), **ICP** (per-edge conformal invariance), and a **four-quadrant** classifier. These update **Beta confidence** over a causal **mechanism graph** (decision variables → mediators → outcomes) and the slow-loop knobs — exploration weight, swap budget, DRO radius, objective weights. **Koi gets measurably better with every deployment.**

The reasoning core is a **budget-first** planner: it allocates tenant envelopes and per-job budgets *first*, then runs bounded per-job specialists *inside* those budgets. This is the deliberate anti-"split-brain" design — parallel reasoning without sub-agents fighting over scarce GPUs.

---

## Repository layout

This repo only contains the code for the algorithm, which is the system’s intelligence planner. **Tandemn-System** is the execution layer that carries out the plans on **Ray, Dynamo, or Kubernetes**.

```
src/
  core/          data model · causal candidate graph · confidence · evidence · mechanism registry
  prediction/    surrogate predictor (DynoSim/AIC) · Tchebycheff scalarization
  cost/          DRO chance constraints · switch cost
  exploration/   EIG (expected information gain)
  validation/    CUSUM · ICP · four-quadrant classifier · C0–C7 plan validator
  learning/      slow loop (self-calibration) · regret
  orchestrator/  S0–S7 deterministic tick spine
  agent/         root reasoning planner · bounded per-job specialists · tool registry
  infra/         resource map (clouds/regions/markets/GPUs) · telemetry
  bootstrap/     seed tables · initialization
```

---

## Roadmap

We are building Koi **in the open and moving fast.** Here's what's in the repo today and what's landing next.

### Now : the v0 core

- [x]  Self-calibrating planner loop: **predict → score → explore → validate → learn**
- [x]  Surrogate prediction, **Tchebycheff** scoring, **EIG** exploration, **DRO**robust SLO checks, **switch-cost**aware swaps
- [x]  Causal **mechanism graph** with Beta confidence; **CUSUM / ICP / four-quadrant** self-calibration
- [x]  Deterministic **S0–S7** tick spine with **C0–C7** validation — *Koi plans, Tandemn-System runs*
- [x]  Multi-cloud / multi-region / spot-aware resource model with **TP / PP / DP / EP** parallelism

### Next : 2 weeks (in active development, shipping fast)

- [ ]  **Prefill/Decode (PD) disaggregation** in the planner — separate prefill and decode deployments, with Koi choosing the optimal **P:D ratio** per job
- [ ]  First-class **SP / PP / CP** planning — sequence, pipeline, and **context parallelism** as native search dimensions
- [ ]  **Upgraded performance database + sharper surrogate stack** for higher-fidelity prediction
- [ ]  **Adaptive objective weights via mirror descent** — automatically walking the Pareto front per fleet and per tenant, instead of fixed weights

> **PD, full SP/PP/CP planning, and adaptive (mirror-descent) weights all land in roughly two weeks.** This is our current sprint and it's moving.
> 

### Near Future

- [ ]  **Pause & preempt jobs** — checkpoint and resume so lower-priority or spot-backed work yields gracefully and finishes later (deadline- and preemption-aware)
- [ ]  **Faster, richer performance-database search** — similarity retrieval over every past deployment, so the planner reasons from the closest real evidence
- [ ]  **Deeper cloud coverage + smarter on-demand & spot scheduling** — more cloud integrations, preemption-aware placement, spot-reclaim resilience

### Horizon

- [ ]  **Multi-engine support beyond vLLM** — SGLang, TensorRT-LLM, and more inference engines as first-class, modeled targets (Tandemn-System already runs on **Ray, Dynamo, or Kubernetes** underneath)

---

## Why now

Inference is eating compute budgets, and the gap between "we have GPUs everywhere" and "every job runs optimally" is widening — because the optimization is still manual. We think the right answer is a planner that **learns the physics of your fleet from your own deployments** and keeps getting better.

We're shipping this quickly and in public:

- **PD, SP/PP/CP, and adaptive weights: ~2 weeks.** Then preemption, deeper spot/on-demand scheduling, and multi-engine.
- Every piece is designed to be **auditable and replayable** — the planner proposes, deterministic code decides.
- The hard math (causal mechanisms, DRO, conformal calibration, Tchebycheff fronts) is real and already in the tree.

If turning a sprawling, heterogeneous GPU fleet into one self-optimizing cluster sounds like your kind of problem — **come build with us.**

- ⭐ **Star the repo** to follow the momentum.
- 🐛 **Open issues / discussions** — tell us about your fleet, your engines, your SLOs.
- 🔧 **Send PRs** — new clouds, new engines, better surrogates, and the parallelism frontiers above are all fair game.
