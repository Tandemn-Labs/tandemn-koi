"""Flat tool registry for the S4 root RLM planner.

Every tool is a free function the root LLM can call from its REPL. State
handles (resource_map, evidence_store, mechanism_registry, ...) are injected
once at boot via bind_tools(); after that the LLM sees a flat function
namespace, mirroring how Claude Code tools read their session context.

Design rules (from realactualopencodeagentic.md):
    - Read and compute tools only. No tool here submits launches, kills
      chains, or mutates cloud state. The deterministic S6 executor owns
      side effects.
    - Mechanism proposals are the one allowed mutation, and they pass
      through deterministic validation (val_new_mechanisms) before the
      registry admits them.
    - Budget-first planning: the root must build and validate a BudgetBook
      before running per-job specialists. validate_budget_book and
      run_job_specialists enforce that order - specialists refuse to run
      without a validated book.
    - Tools return plain Python data (dicts, lists, tuples, scalars). The
      LLM works in a REPL, so tuples are fine; nothing opaque crosses the
      boundary.

v0 market scope: reserved/on-demand instances only. Resource summaries may
carry spot fields (spot_preemption_rate, beta_launch_success) for later;
no tool branches on them yet.

Tool catalog:

    cluster / context:
        get_cluster_state           compact snapshot: jobs, resources, slow state
        get_resource_map            free/total per env
        get_active_jobs             active job descriptors
        get_pending_jobs            pending job descriptors
        get_slow_state_summary      w_t, z_star_t, beta_t, B_t, lambda_swit, epsilon_dro
        get_recent_q_histogram      Q1-Q4 counts over a window (optional per-mechanism)
        get_recent_theory_blobs     NL retrospectives from EvidenceStore
        get_strategy_history        recent cluster-level strategy decisions
        get_priority                deterministic priority table for jobs
        get_regret_slope            mean recent (1 - Q1 rate)
        get_gpu_capacity            free GPUs per env for one gpu_type
        get_job_brief               assembled specialist input for one job

    tenant / budget:
        build_tenant_envelopes      deterministic envelopes per tenant
        get_tenant_envelopes        cached envelopes for this tick
        validate_budget_book        deterministic BudgetBook validation
        run_job_specialists         bounded per-job specialist calls (post-validation)

    resource simulation:
        simulate_allocation         counterfactual resources after a plan
        simulate_resource_free      counterfactual resources if a job released
        simulate_future_state       alias of simulate_allocation
        enumerate_ladder            feasible chain configs under constraints
        required_throughput_enumerator  required tokens/sec from workload + SLO
        size_ladder                 derive n_replicas per rank from y_hat + capacity

    mechanism / confidence:
        get_scope                   mechanisms whose scope matches job features
        get_edge_confidence         c(e) + counters for one or many edges
        get_mechanism_confidence    c(M) + counters for one or many mechanisms
        get_similar_deployments     kNN-ish briefs over EvidenceStore
        set_new_mechanisms          validate + admit a mechanism proposal
        val_new_mechanisms          pre-admission validation only

    prediction / scoring:
        predict_outcome             surrogate prediction + DRO band
        compute_tchebycheff         augmented Tchebycheff J
        compute_eig                 proxy causal EIG for a ladder
        compute_switching_cost      4-component switch cost bundle
        compute_slo_dro             DRO-bounded SLO violation probabilities
        compute_cusum               two-sided CUSUM on one variable
        compute_icp                 ICP invariance test on one edge
        c_d_classification          (v_verdict, y_verdict) -> Q1..Q4

    plan-level:
        compute_sigma               per-job and aggregate sigma for a plan
        check_feasibility           plan validation via the bound validator
        swap_counter                active-job ladder changes in a plan
        check_coverage              Pareto-coverage diagnostic
        check_canary_sanity         canary size / risk heuristics
        check_past_failure          recent Q3/Q4 matches for plan choices
        simulate_outcome_trajectory predicted outcomes for each plan action
"""

import logging
import math
from typing import Any

from src.config.hyperparameters import GAMMA_SLO, UTILIZATION_TARGET_ONLINE
from src.core.models import LADDER_ACTIONS, SWAP_BUDGET_ACTIONS, Plan, RankSpec

log = logging.getLogger("koi.agent_tools")


class _ToolContext:
    """References to every component the tools wrap. Bound once at boot."""

    slow_loop = None
    dro = None
    evidence_store = None
    mechanism_registry = None
    confidence_service = None
    candidate_graph = None
    resource_map = None
    surrogate = None
    telemetry = None
    cusum = None
    icp = None
    quadrant_validator = None
    eig_module = None
    tchebycheff_module = None
    switchcost_module = None
    plan_validator = None
    regret_calculator = None
    tenant_registry = None
    specialist_runner = None

    # Per-tick caches written by the budget tools.
    tenant_envelopes = None
    validated_budget_book = None


_CTX: Any = _ToolContext()


def bind_tools(**components) -> None:
    """Bind components into the module context. Call once at boot.

    Args:
        **components: Any subset of the attribute names on _ToolContext
            (slow_loop, dro, evidence_store, mechanism_registry,
            confidence_service, candidate_graph, resource_map, surrogate,
            telemetry, cusum, icp, quadrant_validator, eig_module,
            tchebycheff_module, switchcost_module, plan_validator,
            regret_calculator, tenant_registry, specialist_runner).
            None values are ignored so partial rebinds are safe.

    Raises:
        ValueError: If a name is not a known context attribute.
    """
    for name, value in components.items():
        if not hasattr(_ToolContext, name):
            raise ValueError(f"bind_tools: unknown component {name!r}")
        if value is not None:
            setattr(_CTX, name, value)


def _require(*names: str) -> None:
    """Raise a clear error if any required dependency is unbound."""
    missing = [n for n in names if getattr(_CTX, n, None) is None]
    if missing:
        raise RuntimeError(f"agent_tools needs {missing} bound. Call bind_tools(...) at boot.")


# Components every planning run needs. Asserted once at the start of the S4
# loop so a wiring gap surfaces at tick start with the full list, not one
# tool at a time deep inside a trajectory. tenant_registry is intentionally
# absent (a single "default" tenant is synthesized when it is unbound), and
# plan_validator is absent (K_P pre-screen is optional; S5 is authoritative).
_PLANNING_DEPENDENCIES = (
    "slow_loop",
    "dro",
    "evidence_store",
    "mechanism_registry",
    "confidence_service",
    "candidate_graph",
    "eig_module",
    "tchebycheff_module",
    "switchcost_module",
    "surrogate",
    "resource_map",
    "specialist_runner",
)


def assert_planning_ready() -> None:
    """Fail fast if any component the S4 planner needs is unbound.

    Converts a late mid-trajectory RuntimeError (raised one tool at a time
    by _require, after the model has already burned turns) into one clear
    error at tick start listing every missing binding.

    Raises:
        RuntimeError: If any name in _PLANNING_DEPENDENCIES is unbound.
    """
    missing = [n for n in _PLANNING_DEPENDENCIES if getattr(_CTX, n, None) is None]
    if missing:
        raise RuntimeError(
            "agent_tools is not fully wired for planning; unbound: "
            f"{missing}. Bind these via bind_tools(...) at boot (or pass "
            "tool_dependencies to KoiAgentHarness) before the agent runs."
        )


def reset_tick_caches() -> None:
    """Clear per-tick caches: tenant envelopes and the validated BudgetBook.

    Must run at every tick boundary (S0 wires it via the TickRunner's
    on_tick_start hook). Without this, run_job_specialists' default-book
    path could reuse a book validated against LAST tick's capacity -
    a stale-budget hole in the anti-split-brain ordering.
    """
    _CTX.tenant_envelopes = None
    _CTX.validated_budget_book = None


# Public module functions that are NOT LLM tools (infrastructure/boot).
_NON_TOOL_NAMES = frozenset(
    {
        "bind_tools",
        "all_callables",
        "assert_planning_ready",
        "reset_tick_caches",
    }
)


def all_callables() -> dict[str, Any]:
    """Return every public LLM tool as a name -> callable dict.

    The harness binds these into the root REPL namespace in one shot.
    The __module__ filter drops imported callables (e.g. the Plan class)
    so only tool functions defined here are exposed; _NON_TOOL_NAMES drops
    the boot/infra functions (notably reset_tick_caches, which the model
    must never call mid-trajectory).
    """
    return {
        name: fn
        for name, fn in globals().items()
        if callable(fn)
        and not name.startswith("_")
        and name not in _NON_TOOL_NAMES
        and getattr(fn, "__module__", None) == __name__
    }


def _env_key(env) -> str:
    """Normalize an env identifier (tuple or string) to a flat string key."""
    if isinstance(env, (tuple, list)):
        return "|".join(str(part) for part in env)
    return str(env)


def _snapshot():
    _require("resource_map")
    return _CTX.resource_map.snapshot()


def _as_plan(plan, tick: int = 0) -> Plan:
    """Normalize whatever a plan tool receives into a typed Plan.

    The harness passes an already-typed Plan; the LLM may pass the raw
    dict it built in the REPL. Plan.from_raw handles both, so every
    plan-level tool can call this and then work against plan.actions.
    """
    if isinstance(plan, Plan):
        return plan
    return Plan.from_raw(plan, tick=tick)


def _ranks_as_dicts(action) -> list:
    """A PlanAction's ladder as the dict list the EIG/switchcost adapters take."""
    if not action.ladder:
        return []
    return [rank.to_dict() for rank in action.ladder]


def _prev_ladder_for(snapshot, job_id: str) -> list:
    """The job's current ladder from the snapshot, as a rank-dict list.

    Empty when the snapshot has no such accessor or the job is new -
    switch cost then sees an all-additions transition, which is correct
    for a first placement.
    """
    if snapshot is None:
        return []
    if hasattr(snapshot, "current_ladder"):
        return list(snapshot.current_ladder(job_id) or [])
    return []


def _slo_thresholds_for(snapshot, job_id: str) -> dict:
    """The job's per-objective SLO thresholds from the snapshot.

    Empty when unavailable - dro_chance_constraint then returns no
    violation, which is the correct no-signal default.
    """
    if snapshot is None:
        return {}
    if hasattr(snapshot, "slo_thresholds"):
        return dict(snapshot.slo_thresholds(job_id) or {})
    return {}


def _derive_y_hat(action, snapshot) -> dict:
    """Outcome prediction for an action's ladder.

    Prefers the action's advisory predicted_y (the LLM's predict_outcome
    result) but does NOT depend on it: when absent, predicts on the
    ladder's representative rank (the 'aggregate' rank if present, else
    the first). Multi-rank composition into a single job outcome is a
    surrogate concern; this is the v0 proxy.
    """
    if action.predicted_y:
        return dict(action.predicted_y)
    if not action.ladder:
        return {}
    rep = next((r for r in action.ladder if r.role == "aggregate"), action.ladder[0])
    try:
        return predict_outcome({"job_config": rep.config})["y_hat"]
    except Exception:
        log.exception("y_hat derivation failed for job %s", action.job_id)
        return {}


# ----------------------------------------------------------------------
# Cluster / context tools
# ----------------------------------------------------------------------


def get_cluster_state() -> dict[str, Any]:
    """Return a compact cluster snapshot for orientation.

    Returns:
        Dict with tick, active_jobs, pending_jobs, resources, slow_state.
        Summaries only - inspect specific jobs with get_job_brief.
    """
    _require("resource_map", "slow_loop")
    snap = _snapshot()
    return {
        "tick": _CTX.slow_loop.state.tick,
        "active_jobs": snap.active_jobs_summary() if hasattr(snap, "active_jobs_summary") else [],
        "pending_jobs": snap.pending_jobs_summary()
        if hasattr(snap, "pending_jobs_summary")
        else [],
        "resources": snap.resources_summary() if hasattr(snap, "resources_summary") else {},
        "slow_state": get_slow_state_summary(),
    }


def get_resource_map() -> dict[str, Any]:
    """Return free/total capacity per environment.

    Returns:
        Dict env_key -> {"free": int, "total": int, "gpu_type": str, ...}.
        May include reserve fields (reserved_recovery_gpus,
        reserved_canary_gpus) and spot fields for later market support.
    """
    snap = _snapshot()
    return snap.resources_summary() if hasattr(snap, "resources_summary") else {}


def get_active_jobs() -> list[dict[str, Any]]:
    """Return descriptors for currently running jobs.

    Returns:
        List of dicts with at least job_id, tenant_id, current ladder
        summary, and recent Q label where available.
    """
    snap = _snapshot()
    return snap.active_jobs_summary() if hasattr(snap, "active_jobs_summary") else []


def get_pending_jobs() -> list[dict[str, Any]]:
    """Return descriptors for jobs waiting for placement."""
    snap = _snapshot()
    return snap.pending_jobs_summary() if hasattr(snap, "pending_jobs_summary") else []


def get_slow_state_summary() -> dict[str, Any]:
    """Return the current slow-loop knobs in one dict.

    Returns:
        Dict with tick, w_t, z_star_t, lambda_swit, beta_t, B_t,
        epsilon_dro, regret_slope, q1_rate, observed_swap_rate,
        observed_coverage.
    """
    _require("slow_loop")
    s = _CTX.slow_loop.state
    return {
        "tick": s.tick,
        "w_t": dict(s.w_t),
        "z_star_t": dict(s.z_star_t),
        "lambda_swit": s.lambda_swit,
        "beta_t": s.beta_t,
        "B_t": s.B_t,
        "epsilon_dro": s.epsilon_dro,
        "regret_slope": s.regret_slope,
        "q1_rate": s.q1_rate,
        "observed_swap_rate": s.observed_swap_rate,
        "observed_coverage": s.observed_coverage,
    }


def get_recent_q_histogram(
    window: int = 20,
    mechanism_id: str | None = None,
) -> dict[str, int]:
    """Return Q1-Q4 counts over recent decided (row, mechanism) pairs.

    Args:
        window: Ticks to look back.
        mechanism_id: If given, count only that mechanism's labels.

    Returns:
        Dict {"Q1": int, "Q2": int, "Q3": int, "Q4": int}.
    """
    _require("quadrant_validator", "evidence_store")
    hist = _CTX.quadrant_validator.aggregate_quadrant_histogram(
        _CTX.evidence_store, int(window), mechanism_id=mechanism_id
    )
    return {(q.value if hasattr(q, "value") else str(q)): n for q, n in hist.items()}


def get_recent_theory_blobs(window: int = 20) -> list[dict[str, Any]]:
    """Return recent NL retrospectives logged on evidence rows.

    Args:
        window: Ticks to look back.

    Returns:
        List of {"tick", "job_id", "mechanism_ids", "q_labels",
        "theory_blob"} for rows that carry a theory_blob.
    """
    _require("evidence_store")
    store = _CTX.evidence_store
    current = store.current_tick()
    rows = store.get_rows_in_window((max(0, current - int(window)), current))
    out = []
    for r in rows:
        blob = getattr(r, "theory_blob", None)
        if not blob:
            continue
        q_labels = {
            mid: (q.value if hasattr(q, "value") else q)
            for mid, q in getattr(r, "q_label_per_mechanism", {}).items()
        }
        out.append(
            {
                "tick": r.tick,
                "job_id": r.job_id,
                "mechanism_ids": list(getattr(r, "mechanism_ids", [])),
                "q_labels": q_labels,
                "theory_blob": blob,
            }
        )
    return out


def get_strategy_history(window: int = 10) -> list[dict[str, Any]]:
    """Return recent cluster-level strategy decisions, newest last.

    Args:
        window: Ticks to look back.

    Returns:
        List of {"tick", "strategy", "headline"} dicts, or [] when the
        store does not track strategy decisions.
    """
    _require("evidence_store")
    if hasattr(_CTX.evidence_store, "get_recent_strategy_decisions"):
        return _CTX.evidence_store.get_recent_strategy_decisions(int(window))
    return []


def get_priority(jobs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Build a deterministic priority table for jobs.

    Combines tenant priority, job class, online/batch, deadline pressure,
    SLO margin, queue age, and recent failure signals into one score.
    The root reads this table instead of raw job data, then inspects
    specific jobs near decision boundaries.

    Args:
        jobs: Job descriptor dicts. Defaults to pending + active jobs.

    Returns:
        List of {"job_id", "tenant_id", "priority_score", "signals"}
        sorted by descending score.
    """
    if jobs is None:
        jobs = list(get_pending_jobs()) + list(get_active_jobs())
    scored = []
    for j in jobs:
        signals = {
            "tenant_priority": float(j.get("tenant_priority", 1.0)),
            "priority_class": float(j.get("priority_class", 0)),
            "is_online": 1.0 if j.get("type", "online") == "online" else 0.0,
            "deadline_pressure": float(j.get("deadline_pressure", 0.0)),
            "slo_margin_deficit": max(0.0, -float(j.get("slo_margin", 0.0))),
            "queue_age_ticks": float(j.get("queue_age_ticks", 0)),
            "recent_failures": float(j.get("recent_failures", 0)),
        }
        score = (
            signals["tenant_priority"] * 10.0
            + signals["priority_class"] * 10.0
            + signals["is_online"] * 3.0
            + signals["deadline_pressure"] * 5.0
            + signals["slo_margin_deficit"] * 8.0
            + signals["queue_age_ticks"] * 0.5
            + signals["recent_failures"] * 2.0
        )
        scored.append(
            {
                "job_id": j.get("job_id", j.get("id")),
                "tenant_id": j.get("tenant_id", "default"),
                "priority_score": score,
                "signals": signals,
            }
        )
    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored


def get_regret_slope(window: int = 20) -> float:
    """Return the mean recent (1 - Q1 rate). High means still learning."""
    _require("slow_loop")
    return float(_CTX.slow_loop.get_sss_regret_slope(int(window)))


def get_gpu_capacity(gpu_type: str) -> dict[str, int]:
    """Return free GPU count per env for one gpu_type.

    Args:
        gpu_type: For example "H100", "A100", "L40S".

    Returns:
        Dict env_key -> free count, only envs matching gpu_type.
    """
    resources = get_resource_map()
    return {
        env: info.get("free", 0)
        for env, info in resources.items()
        if info.get("gpu_type") == gpu_type
    }


def get_job_brief(job_id: str) -> dict[str, Any]:
    """Assemble the specialist input brief for one job.

    Pulls the job descriptor, recent evidence, applicable mechanisms,
    and similar deployments into the JobSpecialistBrief shape the
    specialist prompt expects.

    Args:
        job_id: The job to brief.

    Returns:
        Dict with job_id, tenant_id, job_features, current_ladder,
        recent_q_labels, recent_theory_blobs, similar_deployments,
        applicable_mechanisms.
    """
    _require("evidence_store", "mechanism_registry", "confidence_service")
    descriptor = None
    for j in list(get_active_jobs()) + list(get_pending_jobs()):
        if j.get("job_id", j.get("id")) == job_id:
            descriptor = j
            break

    rows = _CTX.evidence_store.get_rows_for_job(job_id)
    recent_rows = rows[-5:]
    recent_q = []
    blobs = []
    for r in recent_rows:
        q_labels = {
            mid: (q.value if hasattr(q, "value") else q)
            for mid, q in getattr(r, "q_label_per_mechanism", {}).items()
        }
        recent_q.append({"tick": r.tick, "q_labels": q_labels})
        if getattr(r, "theory_blob", None):
            blobs.append({"tick": r.tick, "theory_blob": r.theory_blob})

    features = dict(descriptor.get("job_features", {})) if descriptor else {}
    subset_x = list(features.get("subset_x", features.keys()))
    mechanisms = get_scope({"subset_x": subset_x, "subset_v": []})

    return {
        "job_id": job_id,
        "tenant_id": (descriptor or {}).get("tenant_id", "default"),
        "job_features": features,
        "current_ladder": (descriptor or {}).get("current_ladder"),
        "recent_q_labels": recent_q,
        "recent_theory_blobs": blobs,
        "similar_deployments": get_similar_deployments(features, top_k=5),
        "applicable_mechanisms": mechanisms,
    }


# ----------------------------------------------------------------------
# Tenant / budget tools
# ----------------------------------------------------------------------


def build_tenant_envelopes() -> dict[str, dict[str, Any]]:
    """Build deterministic tenant envelopes for this tick.

    Envelopes are the legal resource boundary per tenant: floors,
    ceilings, quotas, and env allow/deny lists. The root reasons over
    them but cannot exceed them. With no tenant_registry bound, a single
    "default" tenant owns all capacity (the v0 single-tenant case).

    Returns:
        Dict tenant_id -> envelope dict. Also cached for
        get_tenant_envelopes and validate_budget_book.
    """
    resources = get_resource_map()
    capacity = {_env_key(env): int(info.get("free", 0)) for env, info in resources.items()}

    if _CTX.tenant_registry is None:
        envelopes = {
            "default": {
                "tenant_id": "default",
                "priority_tier": "standard",
                "fairness_weight": 1.0,
                "guaranteed_floor": {},
                "burst_ceiling": dict(capacity),
                "hard_quota": dict(capacity),
                "allowed_envs": list(capacity.keys()),
                "denied_envs": [],
                "budget_usd_remaining": None,
                "can_use_spot": False,
            }
        }
    else:
        tenants = _CTX.tenant_registry.list_tenants()
        total_weight = sum(float(t.get("fairness_weight", 1.0)) for t in tenants) or 1.0
        envelopes = {}
        for t in tenants:
            weight = float(t.get("fairness_weight", 1.0))
            share = {env: int(free * weight / total_weight) for env, free in capacity.items()}
            envelopes[t["tenant_id"]] = {
                "tenant_id": t["tenant_id"],
                "priority_tier": t.get("priority_tier", "standard"),
                "fairness_weight": weight,
                "guaranteed_floor": dict(t.get("guaranteed_floor", {})),
                "burst_ceiling": dict(t.get("burst_ceiling", share)),
                "hard_quota": dict(t.get("hard_quota", share)),
                "allowed_envs": list(t.get("allowed_envs", capacity.keys())),
                "denied_envs": list(t.get("denied_envs", [])),
                "budget_usd_remaining": t.get("budget_usd_remaining"),
                "can_use_spot": bool(t.get("can_use_spot", False)),
            }

    _CTX.tenant_envelopes = envelopes
    return envelopes


def get_tenant_envelopes() -> dict[str, dict[str, Any]]:
    """Return the cached tenant envelopes, building them if needed."""
    if _CTX.tenant_envelopes is None:
        return build_tenant_envelopes()
    return _CTX.tenant_envelopes


def validate_budget_book(budget_book: dict[str, Any]) -> dict[str, Any]:
    """Deterministically validate a BudgetBook before specialists run.

    Checks, in order:
        1. Every job budget references a known tenant envelope.
        2. No job budget uses an env denied to its tenant.
        3. Per-tenant env sums stay within the tenant hard quota.
        4. Cluster-wide env sums stay within free capacity minus reserves.
        5. Implied active-job swaps stay within the swap budget B_t.

    On success the book is cached so run_job_specialists can verify it
    was validated. Any change to the book requires re-validation.

    Args:
        budget_book: {"tick": int, "job_budgets": {job_id: slice},
            "reserves": {env_key: int}, "rationale": str}. Each slice is
            {"tenant_id", "job_id", "env_budget": {env_key: gpus},
            "allowed_actions", "strategy_hint", "canary_cap",
            "priority_score", "notes"}.

    Returns:
        {"ok": bool, "violations": List[str]}.
    """
    _require("slow_loop")
    violations: list[str] = []
    envelopes = get_tenant_envelopes()
    resources = get_resource_map()
    capacity = {_env_key(env): int(info.get("free", 0)) for env, info in resources.items()}
    reserves = {_env_key(env): int(n) for env, n in (budget_book.get("reserves") or {}).items()}

    job_budgets = budget_book.get("job_budgets") or {}
    cluster_totals: dict[str, int] = {}
    tenant_totals: dict[str, dict[str, int]] = {}
    implied_swaps = 0
    active_ids = {j.get("job_id", j.get("id")) for j in get_active_jobs()}

    for job_id, slice_ in job_budgets.items():
        tenant_id = slice_.get("tenant_id", "default")
        envelope = envelopes.get(tenant_id)
        if envelope is None:
            violations.append(f"job {job_id}: unknown tenant {tenant_id!r}")
            continue

        denied = {_env_key(e) for e in envelope.get("denied_envs", [])}
        for env, gpus in (slice_.get("env_budget") or {}).items():
            key = _env_key(env)
            gpus = int(gpus)
            if gpus < 0:
                violations.append(f"job {job_id}: negative budget in {key}")
                continue
            if key in denied:
                violations.append(f"job {job_id}: env {key} denied for tenant {tenant_id}")
            cluster_totals[key] = cluster_totals.get(key, 0) + gpus
            tenant_totals.setdefault(tenant_id, {})
            tenant_totals[tenant_id][key] = tenant_totals[tenant_id].get(key, 0) + gpus

        hint = str(slice_.get("strategy_hint", "")).lower()
        if job_id in active_ids and any(
            word in hint for word in ("swap", "migrate", "replace", "move")
        ):
            implied_swaps += 1

    for tenant_id, totals in tenant_totals.items():
        quota = {_env_key(e): int(n) for e, n in envelopes[tenant_id].get("hard_quota", {}).items()}
        for env, used in totals.items():
            limit = quota.get(env)
            if limit is not None and used > limit:
                violations.append(f"tenant {tenant_id}: {used} GPUs in {env} exceeds quota {limit}")

    for env, used in cluster_totals.items():
        allocatable = capacity.get(env, 0) - reserves.get(env, 0)
        if used > allocatable:
            violations.append(f"env {env}: budgets sum to {used} but allocatable is {allocatable}")

    b_t = _CTX.slow_loop.get_sss_swap_budget_t()
    if implied_swaps > b_t:
        violations.append(f"implied swaps {implied_swaps} exceed swap budget B_t={b_t}")

    ok = len(violations) == 0
    _CTX.validated_budget_book = budget_book if ok else None
    return {"ok": ok, "violations": violations}


def run_job_specialists(
    jobs: list[str],
    budget_book: dict[str, Any] | None = None,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Run bounded per-job specialists under a validated BudgetBook.

    Refuses to run when the supplied book is not the one most recently
    validated by validate_budget_book - that ordering is the
    anti-split-brain invariant. Each specialist optimizes one job inside
    its BudgetSlice and reports a fitness signal; it cannot allocate
    outside its slice or see the cluster plan.

    Args:
        jobs: Job ids to process. Each must have a slice in the book.
        budget_book: The validated book. Defaults to the cached one.
        max_workers: Parallel specialist calls.

    Returns:
        List of JobSpecialistResult dicts ({"job_id", "type", "ladder",
        "predicted_y", "predicted_sigma", "budget_utilization",
        "fitness", "marginal_value_of_more", "unused_capacity",
        "mechanism_ids", "new_mechanism_proposals", "reasoning"}).

    Raises:
        RuntimeError: If no validated book exists or no specialist
            runner is bound.
    """
    _require("specialist_runner")
    book = budget_book if budget_book is not None else _CTX.validated_budget_book
    if book is None or book is not _CTX.validated_budget_book:
        raise RuntimeError(
            "run_job_specialists requires the BudgetBook most recently "
            "validated by validate_budget_book. Validate first."
        )
    missing = [j for j in jobs if j not in (book.get("job_budgets") or {})]
    if missing:
        raise RuntimeError(f"jobs {missing} have no BudgetSlice in the book")
    return _CTX.specialist_runner.run_many(
        jobs=jobs, budget_book=book, max_workers=int(max_workers)
    )


# ----------------------------------------------------------------------
# Resource simulation tools
# ----------------------------------------------------------------------


def simulate_allocation(plan) -> dict[str, Any]:
    """Return counterfactual resource state after applying a plan.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts. Normalized
            so the resource map always receives a typed Plan.

    Returns:
        Dict env_key -> {"free_now", "free_after", "delta"}.
    """
    _require("resource_map")
    return _CTX.resource_map.simulate_resource_state_after(_as_plan(plan))


def simulate_resource_free(job_id: str) -> dict[str, int]:
    """Return capacity freed per env if job_id released its chains."""
    _require("resource_map")
    if hasattr(_CTX.resource_map, "simulate_resource_free"):
        return _CTX.resource_map.simulate_resource_free(job_id)
    return {}


def simulate_future_state(plan: dict[str, Any]) -> dict[str, Any]:
    """Alias of simulate_allocation for prompt readability."""
    return simulate_allocation(plan)


def enumerate_ladder(constraints: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate feasible chain configs under structural constraints.

    Args:
        constraints: {"model_id", "gpu_types", "tp_options", "pp_options",
            "engines", ...} - whatever the resource map's enumerator
            supports.

    Returns:
        List of candidate chain config dicts. [] when the resource map
        has no enumerator.
    """
    _require("resource_map")
    if hasattr(_CTX.resource_map, "enumerate_chain_configs"):
        return _CTX.resource_map.enumerate_chain_configs(constraints)
    return []


def required_throughput_enumerator(job_features: dict[str, Any]) -> float:
    """Compute required tokens/sec from workload features and SLO type.

    Online jobs: arrival_rate * output_len_avg * headroom.
    Batch jobs: total_token_budget / deadline_seconds * headroom.

    Args:
        job_features: Workload dict with type, arrival rate, output
            length, token budget, deadline, headroom_factor.

    Returns:
        Required throughput in tokens/sec.
    """
    job_type = job_features.get("type", "online")
    headroom = float(job_features.get("headroom_factor", 1.5))
    if job_type == "batch":
        budget = float(job_features.get("total_token_budget", 0.0))
        deadline_s = float(job_features.get("deadline_hours", 24.0)) * 3600.0
        return budget / max(1.0, deadline_s) * headroom
    rate = float(job_features.get("request_arrival_rate", 0.0))
    out_avg = float(job_features.get("output_len_tokens_avg", 0.0))
    return rate * out_avg * headroom


def _y_value(y_hat: dict[str, Any], *keys: str) -> float:
    """First present y_hat value across spelling variants, else 0.0."""
    for key in keys:
        value = y_hat.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _feature_value(features: dict[str, Any], *keys: str) -> float | None:
    """First present feature value across spelling variants, else None."""
    for key in keys:
        value = features.get(key)
        if value is not None:
            return float(value)
    return None


def size_ladder(
    ranks: list[dict[str, Any]],
    job_features: dict[str, Any],
    target_tps: float | None = None,
    utilization_target: float | None = None,
) -> dict[str, Any]:
    """Derive each rank's replica/dp count, regime-aware.

    The LLM proposes rank CONFIGS (gpu, tp, pp, engine, quant); this
    computes the replica/dp count rather than leaving it to a guess.

    Batch (deadline-bound, no per-request SLO): size purely on throughput
    at full utilization.

        target_tps    = total_token_budget / deadline_seconds * headroom
        per_chain_eff = per_chain_raw
        n_replicas    = ceil(target_tps / per_chain_eff)

    Online (rate-bound UNDER latency SLOs): two extra constraints.
      1. Latency gate: a config whose predicted p99 TTFT or p99 TPOT
         already exceeds target is INFEASIBLE for online. TPOT especially
         cannot be fixed by adding replicas (it is per-replica decode
         latency), so an SLO-violating config must be replaced, not scaled.
      2. Utilization derating: per-replica throughput is multiplied by
         utilization_target (< 1) so the replica runs below saturation and
         queue wait stays bounded (wait ~ rho/(1-rho)), keeping p99 TTFT.

        target_tps    = request_arrival_rate * output_len_tokens_avg * headroom
        per_chain_eff = per_chain_raw * utilization_target
        n_replicas    = ceil(target_tps / per_chain_eff)

    Capped by capacity in both regimes: n_replicas = min(needed,
    floor(free_gpus / gpus_per_chain)). A capacity-bound rank reports its
    GPU shortfall in marginal_value (the starved-fitness signal for budget
    reallocation). An SLO-infeasible online rank is flagged and forces
    meets_target False.

    For a disaggregated ladder each role is sized independently to
    target_tps; achieved_tps is the bottleneck (min) across ranks.
    Cross-role unit matching (prefill input-tps vs decode output-tps) is a
    v1 refinement.

    Args:
        ranks: rank dicts (RankSpec.from_dict form) with role, env, config.
        job_features: the job's W features - type ("online"/"batch"),
            request_arrival_rate, output_len_tokens_avg, target_p99_ttft_ms,
            target_p99_tpot_ms, total_token_budget, deadline_hours,
            headroom_factor.
        target_tps: override; default computed from job_features via
            required_throughput_enumerator.
        utilization_target: override; default UTILIZATION_TARGET_ONLINE for
            online, 1.0 for batch.

    Returns:
        {"ranks": [sized rank dicts], "regime", "target_tps",
         "achieved_tps", "meets_target", "per_rank": [...],
         "marginal_value": {env_key: extra_gpus_to_meet_target}}.
    """
    _require("resource_map", "surrogate", "candidate_graph", "dro")

    regime = str(job_features.get("type", "online")).lower()
    is_online = regime != "batch"
    target = (
        float(target_tps)
        if target_tps is not None
        else float(required_throughput_enumerator(job_features))
    )
    util = (
        float(utilization_target)
        if utilization_target is not None
        else (UTILIZATION_TARGET_ONLINE if is_online else 1.0)
    )
    ttft_target = _feature_value(job_features, "target_p99_ttft_ms", "target_p99_TTFT_ms")
    tpot_target = _feature_value(job_features, "target_p99_tpot_ms", "target_p99_TPOT_ms")

    sized: list[dict[str, Any]] = []
    per_rank: list[dict[str, Any]] = []
    marginal: dict[str, int] = {}
    achieved_values: list[float] = []
    all_slo_ok = True

    for raw in ranks:
        rank = RankSpec.from_dict(raw)
        gpus_per_chain = rank.gpus_per_chain()
        gpu_type = rank.env[3] if rank.env and len(rank.env) >= 4 else None
        free = _CTX.resource_map.get_avail_capacity(rank.env, gpu_type) if gpu_type else 0
        max_by_cap = free // gpus_per_chain if gpus_per_chain > 0 else 0

        y_hat = predict_outcome({"job_config": rank.config}).get("y_hat", {})
        per_chain_raw = _y_value(y_hat, "throughput_tokens_per_sec", "throughput_token_per_sec")
        per_chain_eff = per_chain_raw * util

        slo_violations: list[str] = []
        if is_online:
            ttft_pred = _y_value(y_hat, "p99_ttft_ms", "p99_TTFT_ms")
            tpot_pred = _y_value(y_hat, "p99_tpot_ms", "p99_TPOT_ms")
            if ttft_target is not None and ttft_pred > ttft_target:
                slo_violations.append(f"p99_ttft {ttft_pred:.0f}ms > {ttft_target:.0f}ms target")
            if tpot_target is not None and tpot_pred > tpot_target:
                slo_violations.append(f"p99_tpot {tpot_pred:.1f}ms > {tpot_target:.1f}ms target")
        slo_ok = not slo_violations
        all_slo_ok = all_slo_ok and slo_ok

        if per_chain_eff > 0:
            needed = max(1, math.ceil(target / per_chain_eff))
            no_prediction = False
        else:
            needed = max_by_cap  # no throughput signal: use what capacity allows
            no_prediction = True

        n_replicas = min(needed, max_by_cap) if max_by_cap >= 1 else 0
        rank.n_replicas = n_replicas
        sized.append(rank.to_dict())

        achieved_values.append(n_replicas * per_chain_eff)
        capacity_bound = needed > n_replicas
        if capacity_bound:
            env_key = _env_key(rank.env)
            marginal[env_key] = marginal.get(env_key, 0) + (needed - n_replicas) * gpus_per_chain

        per_rank.append(
            {
                "role": rank.role,
                "env": list(rank.env) if rank.env else None,
                "per_chain_tps_raw": per_chain_raw,
                "per_chain_tps_effective": per_chain_eff,
                "utilization_target": util,
                "gpus_per_chain": gpus_per_chain,
                "needed_replicas": needed,
                "max_replicas_by_capacity": max_by_cap,
                "n_replicas": n_replicas,
                "capacity_bound": capacity_bound,
                "slo_ok": slo_ok,
                "slo_violations": slo_violations,
                "no_throughput_prediction": no_prediction,
            }
        )

    achieved_tps = min(achieved_values) if achieved_values else 0.0
    meets_target = (
        achieved_tps >= target and all(r["n_replicas"] >= 1 for r in per_rank) and all_slo_ok
    )
    return {
        "ranks": sized,
        "regime": regime,
        "target_tps": target,
        "achieved_tps": achieved_tps,
        "meets_target": meets_target,
        "per_rank": per_rank,
        "marginal_value": marginal,
    }


# ----------------------------------------------------------------------
# Mechanism / confidence tools
# ----------------------------------------------------------------------


def get_edge_confidence(edge_or_list) -> Any:
    """Return c(e) and counters for one edge id or a list of them.

    Args:
        edge_or_list: edge_id string or list of edge_id strings.

    Returns:
        For one id: {"c", "visit_count", "envs_seen", "q_histogram"}.
        For a list: dict edge_id -> that shape.
    """
    _require("confidence_service")
    cs = _CTX.confidence_service

    def one(edge_id: str) -> dict[str, Any]:
        return {
            "c": cs.get_edge_confidence(edge_id),
            "visit_count": cs.get_edge_visit_count(edge_id),
            "envs_seen": sorted(_env_key(e) for e in cs.get_edge_environment_seen(edge_id)),
            "q_histogram": dict(cs.get_edge_q_histogram(edge_id)),
        }

    if isinstance(edge_or_list, list):
        return {eid: one(eid) for eid in edge_or_list}
    return one(str(edge_or_list))


def get_mechanism_confidence(m_id) -> Any:
    """Return c(M) and counters for one mechanism id or a list of them.

    Args:
        m_id: mechanism_id string or list of mechanism_id strings.

    Returns:
        For one id: {"c", "visit_count", "envs_seen", "q_histogram"}.
        For a list: dict mechanism_id -> that shape.
    """
    _require("confidence_service")
    cs = _CTX.confidence_service

    def one(mid: str) -> dict[str, Any]:
        return {
            "c": cs.get_mechanism_confidence(mid),
            "visit_count": cs.get_mechanism_visit_count(mid),
            "envs_seen": sorted(_env_key(e) for e in cs.get_mechanism_environment_seen(mid)),
            "q_histogram": dict(cs.get_mechanism_q_histogram(mid)),
        }

    if isinstance(m_id, list):
        return {mid: one(mid) for mid in m_id}
    return one(str(m_id))


def get_scope(job_features: dict[str, Any]) -> list[dict[str, Any]]:
    """Return active mechanisms whose scope matches job features.

    Args:
        job_features: Dict with subset_x and subset_v lists (or any
            feature names; they are matched against mechanism scopes).

    Returns:
        List of mechanism briefs with confidence and visit counts.
    """
    _require("mechanism_registry", "confidence_service")
    subset_x = job_features.get("subset_x", []) if isinstance(job_features, dict) else []
    subset_v = job_features.get("subset_v", []) if isinstance(job_features, dict) else []
    mechs = _CTX.mechanism_registry.filter_by_scope(subset_x, subset_v)
    return [
        {
            "mechanism_id": m.mechanism_id,
            "edge_ids": list(m.edge_ids),
            "scope": dict(m.scope),
            "narrative": m.narrative,
            "c": _CTX.confidence_service.get_mechanism_confidence(m.mechanism_id),
            "visit_count": _CTX.confidence_service.get_mechanism_visit_count(m.mechanism_id),
        }
        for m in mechs
        if m.status == "active"
    ]


def get_similar_deployments(
    job_features: dict[str, Any],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return briefs of past deployments similar to the given features.

    Uses the store's retrieval method when available; otherwise falls
    back to a naive scan of recent rows matched on workload type and
    gpu_type.

    Args:
        job_features: Feature dict for similarity.
        top_k: Maximum briefs to return.

    Returns:
        List of {"tick", "job_id", "rank_id", "env_label",
        "mechanism_ids", "q_labels", "y_observed_mean"}.
    """
    _require("evidence_store")
    store = _CTX.evidence_store

    if hasattr(store, "retrieve_similar_rows"):
        rows = store.retrieve_similar_rows(job_features, top_k=int(top_k))
    elif hasattr(store, "rag_retrieve_similar"):
        rows = store.rag_retrieve_similar(job_features, top_k=int(top_k))
    else:
        current = store.current_tick()
        rows = store.get_rows_in_window((max(0, current - 50), current))
        wanted_gpu = job_features.get("gpu_type")
        wanted_type = job_features.get("type") or job_features.get("workload_type")
        wanted_type = str(wanted_type).lower() if wanted_type is not None else None
        if wanted_gpu or wanted_type:
            rows = [
                r
                for r in rows
                if (wanted_gpu is None or r.env_label[3] == wanted_gpu)
                and (
                    wanted_type is None
                    or str(r.W_observed.get("type") or r.W_observed.get("workload_type")).lower()
                    == wanted_type
                )
            ]
        rows = rows[-int(top_k) :]

    return [
        {
            "tick": r.tick,
            "job_id": r.job_id,
            "rank_id": r.rank_id,
            "env_label": r.env_label,
            "mechanism_ids": list(getattr(r, "mechanism_ids", [])),
            "q_labels": {
                mid: (q.value if hasattr(q, "value") else q)
                for mid, q in getattr(r, "q_label_per_mechanism", {}).items()
            },
            "y_observed_mean": dict(getattr(r, "y_observed_mean", {}) or {}),
        }
        for r in rows
    ]


def set_new_mechanisms(
    edges: list[str],
    applicable_to: dict[str, Any],
    llm_blurb: str,
) -> dict[str, Any]:
    """Validate and admit a new mechanism proposal.

    The only mutation tool. Validation is deterministic; the registry
    admits the mechanism only when every edge exists, the topology is
    legal, and the proposal is not a duplicate.

    Args:
        edges: edge_id strings, all present in CandidateGraph.
        applicable_to: Scope dict, e.g. {"x": [...], "v": [...],
            "workload_type": "online"}.
        llm_blurb: One-paragraph narrative for the mechanism.

    Returns:
        {"ok": bool, "mechanism_id": str | None, "violations": list}.
    """
    _require("mechanism_registry", "candidate_graph")
    from src.core.models import Mechanism

    candidate = Mechanism(
        edge_ids=list(edges),
        scope=dict(applicable_to),
        narrative=str(llm_blurb),
    )
    check = val_new_mechanisms(candidate)
    if not check["ok"]:
        return {"ok": False, "mechanism_id": None, "violations": check["violations"]}
    mid = _CTX.mechanism_registry.add_mechanism(candidate)
    return {"ok": True, "mechanism_id": mid, "violations": []}


def val_new_mechanisms(m_new) -> dict[str, Any]:
    """Run pre-admission validation on a mechanism proposal.

    Checks that every edge_id exists in CandidateGraph, that the bundle
    topology only uses X->V and V->Y edges, and that the proposal is not
    a duplicate of an existing mechanism.

    Args:
        m_new: Mechanism object or dict with edge_ids, scope, narrative.

    Returns:
        {"ok": bool, "violations": List[str]}.
    """
    _require("mechanism_registry", "candidate_graph")
    from src.core.models import Mechanism

    if isinstance(m_new, dict):
        m_new = Mechanism(
            edge_ids=list(m_new.get("edge_ids", [])),
            scope=dict(m_new.get("scope", {})),
            narrative=str(m_new.get("narrative", "")),
        )

    violations: list[str] = []
    cg = _CTX.candidate_graph
    edge_objs = []
    for eid in m_new.edge_ids:
        if eid not in cg.edge_table:
            violations.append(f"edge {eid!r} not in CandidateGraph")
            continue
        edge_objs.append(cg.edge_table[eid])

    if edge_objs and not cg.val_topology(edge_objs):
        violations.append("topology violation - only X->V and V->Y edges allowed")

    is_dup, existing_id = _CTX.mechanism_registry.is_duplicate_mechanism(m_new)
    if is_dup:
        violations.append(f"duplicate of existing mechanism {existing_id}")

    return {"ok": len(violations) == 0, "violations": violations}


# ----------------------------------------------------------------------
# Prediction / scoring tools
# ----------------------------------------------------------------------


def predict_outcome(
    config: dict[str, Any],
    mechanism: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the surrogate and attach the DRO uncertainty band.

    Args:
        config: X variables for the candidate. May embed job_config and
            job_features sub-dicts; otherwise the whole dict is treated
            as job_config.
        mechanism: Optional mechanism context, informational only.

    Returns:
        {"y_hat": dict, "v_hat": dict, "dro_band": dict}.
    """
    _require("candidate_graph", "dro", "surrogate")
    job_features = dict(config.get("job_features", {}))
    job_config = dict(config.get("job_config", config))
    result = _CTX.surrogate.compose_prediction(
        job_config=job_config,
        job_features=job_features,
        candidate_graph=_CTX.candidate_graph,
        method=("AIC_DynoSim",),
    )
    if isinstance(result, tuple) and len(result) == 2:
        y_hat, v_hat = result
    else:
        y_hat = getattr(result, "y_hat", {}) or {}
        v_hat = getattr(result, "v_hat", {}) or {}
    dro_band = _CTX.dro.compute_dro_band(y_hat or {})
    return {"y_hat": y_hat or {}, "v_hat": v_hat or {}, "dro_band": dro_band}


def compute_tchebycheff(
    y_hat: dict[str, float],
    wt: dict[str, float] | None = None,
    z_star: dict[str, float] | None = None,
) -> float:
    """Compute the augmented Tchebycheff scalar J for one prediction.

    Sign-flipped so larger J means closer to the ideal point.

    Args:
        y_hat: objective -> predicted value.
        wt: Objective weights. Defaults to the slow loop's current w_t.
        z_star: Reference point. Defaults to the slow loop's current z_star_t.

    Returns:
        J <= 0.
    """
    _require("tchebycheff_module", "slow_loop")
    weights = wt if wt is not None else _CTX.slow_loop.get_sss_wt()
    reference = z_star if z_star is not None else _CTX.slow_loop.get_sss_z_star_t()
    return float(
        _CTX.tchebycheff_module.compute_tchebycheff(
            y_hat=y_hat,
            w_t=weights,
            z_star_t=reference,
            normalization_range=_CTX.slow_loop.typical_ranges,
        )
    )


def compute_eig(candidate_ladder: dict[str, Any]) -> float:
    """Compute the proxy causal EIG for a candidate ladder.

    Higher means the ladder tests more uncertain, less-visited edges
    and mechanisms.

    Args:
        candidate_ladder: Canonical ladder list of rank dicts. Each rank
            carries mechanism_id, config, n_replicas, and env.

    Returns:
        Non-negative EIG value.
    """
    _require(
        "eig_module",
        "candidate_graph",
        "mechanism_registry",
        "confidence_service",
        "evidence_store",
    )
    ladder = _materialize_ladder(candidate_ladder)
    return float(
        _CTX.eig_module.compute_eig(
            L_prime=ladder,
            candidate_graph=_CTX.candidate_graph,
            mechanism_registry=_CTX.mechanism_registry,
            confidence_service=_CTX.confidence_service,
            evidence_store=_CTX.evidence_store,
        )
    )


def compute_switching_cost(ladder_prev: Any, ladder_new: Any) -> dict[str, float]:
    """Compute the 4-component switch cost between two ladders.

    Args:
        ladder_prev: Current ladder (chain entry dicts or objects).
        ladder_new: Proposed ladder.

    Returns:
        {"c_coldstart", "c_parallel", "c_kill", "c_risk", "total"}.
    """
    _require("switchcost_module", "dro", "slow_loop")
    L_prev = _materialize_chain_list(ladder_prev)
    L_new = _materialize_chain_list(ladder_new)
    bundle = _CTX.switchcost_module.compute_switch_cost(
        L_prev=L_prev,
        L_new=L_new,
        residual_history=_CTX.dro,
        epsilon_dro=_CTX.slow_loop.get_sss_radius_dro(),
    )
    return bundle.as_dict()


def compute_slo_dro(
    slo_thresholds: dict[str, float],
    y_hat: dict[str, float],
) -> dict[str, float]:
    """Compute DRO-bounded SLO violation probabilities.

    Args:
        slo_thresholds: objective -> threshold.
        y_hat: Point prediction per objective.

    Returns:
        Dict objective -> probability plus "_any_violated".
    """
    _require("dro")
    return _CTX.dro.dro_chance_constraint(pred_y=y_hat, slo_thresholds=slo_thresholds)


def compute_cusum(
    observed: list[float],
    predicted,
    delta: float,
    h: float,
) -> dict[str, Any]:
    """Run two-sided CUSUM on one variable's residual sequence.

    Args:
        observed: Observed sub-tick samples.
        predicted: Predicted scalar (broadcast) or same-length series.
        delta: Drift tolerance.
        h: Fire threshold.

    Returns:
        {"direction": str, "fired": bool, "fire_tick": int | None}.
    """
    _require("cusum")
    direction, fired, fire_tick = _CTX.cusum.cusum_per_v(
        observed=observed, predicted=predicted, delta=float(delta), h=float(h)
    )
    return {
        "direction": direction.value if hasattr(direction, "value") else str(direction),
        "fired": bool(fired),
        "fire_tick": int(fire_tick) if fire_tick is not None else None,
    }


def compute_icp(edge_id: str) -> str:
    """Run the ICP invariance test on one edge.

    Args:
        edge_id: Edge to test.

    Returns:
        "accept", "reject", or "undecided".
    """
    _require("icp", "candidate_graph", "evidence_store")
    edge = _CTX.candidate_graph.edge_table.get(edge_id)
    if edge is None:
        return "undecided"
    result = _CTX.icp.compute_icp_per_edge(edge=edge, evidence_store=_CTX.evidence_store)
    return result.value if hasattr(result, "value") else str(result)


def c_d_classification(v_cusum_matched: bool, y_cusum_matched: bool) -> str:
    """Map V and Y CUSUM verdicts to a quadrant label.

    Args:
        v_cusum_matched: True iff the V bundle CUSUM said matched.
        y_cusum_matched: True iff the Y bundle CUSUM said matched.

    Returns:
        "Q1", "Q2", "Q3", or "Q4".
    """
    _require("quadrant_validator")
    label = _CTX.quadrant_validator.classify_quadrant(
        "matched" if v_cusum_matched else "diverged",
        "matched" if y_cusum_matched else "diverged",
    )
    return label.value if hasattr(label, "value") else str(label)


# ----------------------------------------------------------------------
# Plan-level tools
# ----------------------------------------------------------------------


def compute_sigma(plan) -> dict[str, Any]:
    """Score a plan: per-job sigma and the cluster aggregate.

    sigma = J + beta_t * eig - gamma * Pr_DRO - lambda_swit * switch_cost,
    over every ladder-bearing action (place/swap/retry/resume). The
    scoring inputs are DERIVED, not trusted from the LLM: prev_ladder and
    slo_thresholds come from the snapshot, y_hat from the action's
    advisory predicted_y or a fresh surrogate call. Non-ladder actions
    (keep/defer/preempt/terminate/diagnose) deploy nothing new and score 0.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.

    Returns:
        {"per_job": dict, "aggregate_sigma": float, "swap_count": int}.
    """
    _require(
        "slow_loop",
        "tchebycheff_module",
        "eig_module",
        "switchcost_module",
        "dro",
        "candidate_graph",
        "mechanism_registry",
        "confidence_service",
        "evidence_store",
    )
    typed = _as_plan(plan)
    snapshot = _snapshot()
    per_job: dict[str, dict[str, float]] = {}
    aggregate = 0.0

    w_t = _CTX.slow_loop.get_sss_wt()
    z_star = _CTX.slow_loop.get_sss_z_star_t()
    beta = _CTX.slow_loop.get_sss_eig_incentive_t()
    lam = _CTX.slow_loop.get_sss_lambda_switch()
    eps_dro = _CTX.slow_loop.get_sss_radius_dro()

    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        job_id = action.job_id
        ladder_dicts = _ranks_as_dicts(action)
        y_hat = _derive_y_hat(action, snapshot)
        if not y_hat:
            continue

        J = float(
            _CTX.tchebycheff_module.compute_tchebycheff(
                y_hat=y_hat,
                w_t=w_t,
                z_star_t=z_star,
                normalization_range=_CTX.slow_loop.typical_ranges,
            )
        )
        eig_value = float(
            _CTX.eig_module.compute_eig(
                L_prime=_materialize_ladder(ladder_dicts),
                candidate_graph=_CTX.candidate_graph,
                mechanism_registry=_CTX.mechanism_registry,
                confidence_service=_CTX.confidence_service,
                evidence_store=_CTX.evidence_store,
            )
        )
        switch_bundle = _CTX.switchcost_module.compute_switch_cost(
            L_prev=_materialize_chain_list(_prev_ladder_for(snapshot, job_id)),
            L_new=_materialize_chain_list(ladder_dicts),
            residual_history=_CTX.dro,
            epsilon_dro=eps_dro,
        )
        pr_slo = float(
            _CTX.dro.dro_chance_constraint(
                pred_y=y_hat,
                slo_thresholds=_slo_thresholds_for(snapshot, job_id),
            ).get("_any_violated", 0.0)
        )

        sigma_i = J + beta * eig_value - GAMMA_SLO * pr_slo - lam * switch_bundle.total
        per_job[job_id] = {
            "J": J,
            "eig": eig_value,
            "switch_cost_total": switch_bundle.total,
            "pr_slo_dro": pr_slo,
            "sigma": sigma_i,
        }
        aggregate += sigma_i

    return {
        "per_job": per_job,
        "aggregate_sigma": aggregate,
        "swap_count": swap_counter(typed),
    }


def check_feasibility(plan) -> dict[str, Any]:
    """Validate a plan with the bound plan validator.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.

    Returns:
        {"feasible": bool, "violations": List[str]}.
    """
    _require("plan_validator", "resource_map", "slow_loop")
    result = _CTX.plan_validator.val_plan(
        plan=_as_plan(plan),
        cluster_snapshot=_snapshot(),
        slow_state=_CTX.slow_loop.state,
    )
    return {
        "feasible": bool(getattr(result, "feasible", False)),
        "violations": list(getattr(result, "violations", [])),
    }


def swap_counter(plan) -> int:
    """Count active-job churn against the C4 swap budget B_t.

    Counts actions in SWAP_BUDGET_ACTIONS (swap, preempt, retry). PLACE
    and DEFER are admission, not churn; KEEP/DIAGNOSE/TERMINATE move no
    running workload.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.

    Returns:
        Number of churning actions.
    """
    typed = _as_plan(plan)
    return sum(1 for a in typed.actions if a.type in SWAP_BUDGET_ACTIONS)


def check_coverage(plan) -> dict[str, Any]:
    """Score how close the plan's predicted outcomes sit to z_star.

    A rough Pareto-coverage diagnostic, not the R2 indicator itself.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.

    Returns:
        Dict objective -> score in [0, 1] plus "aggregate".
    """
    _require("slow_loop")
    typed = _as_plan(plan)
    snapshot = _snapshot()
    z_star = _CTX.slow_loop.get_sss_z_star_t()
    ranges = _CTX.slow_loop.typical_ranges
    objectives = list(z_star.keys())
    scores = dict.fromkeys(objectives, 0.0)
    n = 0
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS:
            continue
        y_hat = _derive_y_hat(action, snapshot)
        if not y_hat:
            continue
        for obj in objectives:
            if obj not in y_hat:
                continue
            gap = abs(float(y_hat[obj]) - float(z_star[obj])) / max(ranges.get(obj, 1.0), 1e-9)
            scores[obj] += max(0.0, 1.0 - gap)
        n += 1
    if n > 0:
        scores = {k: v / n for k, v in scores.items()}
    aggregate = sum(scores.values()) / max(1, len(scores))
    return {**scores, "aggregate": aggregate}


def check_canary_sanity(plan) -> dict[str, Any]:
    """Heuristic canary-size check on each ladder-bearing action.

    A swap/place launches the new ladder's ranks as canaries alongside
    production; flag any whose total replica count looks large.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.

    Returns:
        {"ok": bool, "warnings": List[str]}.
    """
    typed = _as_plan(plan)
    warnings: list[str] = []
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        total = sum(rank.n_replicas for rank in action.ladder)
        if total > 10:
            warnings.append(
                f"job {action.job_id}: ladder launches {total} chains (> 10 canary heuristic)"
            )
    return {"ok": len(warnings) == 0, "warnings": warnings}


def check_past_failure(plan, window: int = 20) -> dict[str, Any]:
    """Match plan (mechanism, env) choices against recent Q3/Q4 evidence.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.
        window: Ticks to look back.

    Returns:
        {"matched_failures": List[dict], "warnings": List[str]}.
    """
    _require("evidence_store")
    typed = _as_plan(plan)
    store = _CTX.evidence_store
    cutoff = store.current_tick() - int(window)
    failures: list[dict[str, Any]] = []
    warnings: list[str] = []

    for action in typed.actions:
        for rank in action.ladder or []:
            mech_id = rank.mechanism_id or action.mechanism_id
            if mech_id is None:
                continue
            bad = 0
            for row in store.get_rows_for_mechanism(mech_id, limit=200):
                if row.tick <= cutoff:
                    continue
                if rank.env is not None and _env_key(row.env_label) != _env_key(rank.env):
                    continue
                q = row.q_label_per_mechanism.get(mech_id)
                q_text = q.value if hasattr(q, "value") else q
                if q_text in ("Q3", "Q4"):
                    bad += 1
            if bad:
                failures.append(
                    {
                        "job_id": action.job_id,
                        "mechanism_id": mech_id,
                        "env": list(rank.env) if rank.env else None,
                        "n": bad,
                    }
                )
                warnings.append(
                    f"job {action.job_id}: {bad} recent Q3/Q4 rows for "
                    f"mechanism {mech_id} in {rank.env}"
                )
    return {"matched_failures": failures, "warnings": warnings}


def simulate_outcome_trajectory(plan) -> dict[str, Any]:
    """Predict outcomes for each ladder-bearing action in the plan.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.

    Returns:
        Dict job_id -> {"y_hat", "v_hat", "dro_band"} for the ladder's
        representative rank (aggregate rank if present, else the first).
    """
    typed = _as_plan(plan)
    out: dict[str, Any] = {}
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        rep = next((r for r in action.ladder if r.role == "aggregate"), action.ladder[0])
        out[action.job_id] = predict_outcome({"job_config": rep.config})
    return out


# ----------------------------------------------------------------------
# Internal adapters
# ----------------------------------------------------------------------


def _materialize_ladder(ladder_ranks):
    """Adapt a canonical rank-list ladder into the object shape eig.py consumes.

    eig.py expects .ranks (each .mechanism_id, .config, .n_replicas),
    .envs(), .duration_minutes, and .applicable_mechanisms. The
    applicable set is resolved from the registry: the committed
    mechanisms of every rank plus scope matches on the rank configs.
    An empty applicable set would zero the relevance gate and silently
    kill EIG, so committed mechanisms are always included.
    """

    class _Ladder:
        pass

    class _Rank:
        pass

    ladder = _Ladder()
    ladder.ranks = []
    ladder.duration_minutes = 5.0
    config_keys: set = set()

    for r in ladder_ranks:
        if hasattr(r, "to_dict"):
            r = r.to_dict()
        rank = _Rank()
        rank.mechanism_id = r.get("mechanism_id")
        rank.config = r.get("config", {})
        rank.n_replicas = int(r.get("n_replicas", 1))
        rank.is_canary = bool(r.get("is_canary", False))
        env = r.get("env")
        # env arrives as a list from RankSpec.to_dict; envs() puts these in
        # a set, so coerce to a hashable tuple here.
        rank.env = tuple(env) if isinstance(env, (list, tuple)) else env
        ladder.ranks.append(rank)
        config_keys |= set(rank.config.keys())

    applicable = {}
    if _CTX.mechanism_registry is not None:
        for rank in ladder.ranks:
            if rank.mechanism_id is None:
                continue
            try:
                mech = _CTX.mechanism_registry.get_mechanism(rank.mechanism_id)
                applicable[mech.mechanism_id] = mech
            except KeyError:
                log.warning("ladder references unknown mechanism %s", rank.mechanism_id)
        for mech in _CTX.mechanism_registry.filter_by_scope(sorted(config_keys), []):
            if mech.status == "active":
                applicable[mech.mechanism_id] = mech

    ladder.applicable_mechanisms = list(applicable.values())
    ladder.envs = lambda: {r.env for r in ladder.ranks if r.env is not None}
    return ladder


def _materialize_chain_list(chain_list):
    """Adapt rank dicts into ChainEntry objects for switchcost.py.

    Synthesizes a stable chain_id when one is absent (role + env +
    sorted config), because switch cost matches ΔL+/ΔL- by chain_id -
    None ids would collapse every distinct rank into one and break the
    add/kill diff. env is coerced to a hashable tuple for pricing lookups.
    """
    if not chain_list:
        return []
    if hasattr(chain_list[0], "chain_id"):
        return list(chain_list)
    from src.cost.switchcost import ChainEntry  # type: ignore[import-not-found]

    out = []
    for c in chain_list:
        env = c.get("env")
        env = tuple(env) if isinstance(env, (list, tuple)) else env
        config = c.get("config", {})
        chain_id = c.get("chain_id")
        if not chain_id:
            # repr over key-sorted config tolerates unhashable values
            # (nested lists/dicts) while staying deterministic per tick.
            fingerprint = repr(
                (c.get("role", ""), env, sorted(config.items(), key=lambda kv: kv[0]))
            )
            chain_id = "auto_" + str(abs(hash(fingerprint)))
        out.append(
            ChainEntry(
                chain_id=chain_id,
                config=config,
                env=env,
                n_replicas=int(c.get("n_replicas", 1)),
            )
        )
    return out
