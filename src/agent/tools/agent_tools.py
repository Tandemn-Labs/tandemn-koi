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

v0 market scope: reserved instances only. Resource summaries carry
market|cloud|region|zone|gpu_type env keys.

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

    user / budget:
        build_user_envelopes        deterministic envelopes per user
        get_user_envelopes          cached envelopes for this tick
        allocate_budget_book        default BudgetBook from priority + free GPUs
        validate_budget_book        deterministic BudgetBook validation
        run_job_specialists         bounded per-job specialist calls (post-validation)

    resource simulation:
        simulate_allocation         counterfactual resources after a plan
        simulate_resource_free      counterfactual resources if a job released
        enumerate_ladder            feasible chain configs under constraints
        required_throughput_enumerator  required tokens/sec from workload + SLO
        size_ladder                 derive n_replicas per rank from y_hat + capacity

    mechanism / confidence:
        get_scope                   mechanisms whose scope matches job features
        get_edge_confidence         c(e) + counters for one or many edges
        get_mechanism_confidence    c(M) + counters for one or many mechanisms
        get_influencing_knobs       X knobs that drive an objective, by confidence
        get_similar_deployments     kNN-ish briefs over EvidenceStore
        set_new_mechanisms          validate + admit a mechanism proposal
        val_new_mechanisms          pre-admission validation only

    prediction / scoring:
        predict_outcome             calibrated surrogate prediction + DRO band
        get_z_star                  current ideal-point reference (z_star_t)
        compute_tchebycheff         augmented Tchebycheff J
        optimize_config             LLM-steered coordinate descent over candidates
        compute_eig                 proxy causal EIG for a ladder
        compute_switching_cost      4-component switch cost bundle
        compute_slo_dro             DRO-bounded SLO violation probabilities

    plan-level:
        compute_sigma               per-job and aggregate sigma for a plan
        check_feasibility           plan validation via the bound validator
        swap_counter                active-job ladder changes in a plan
        check_coverage              Pareto-coverage diagnostic
        check_canary_sanity         canary size / risk heuristics
        check_past_failure          recent Q3/Q4 matches for plan choices
        simulate_outcome_trajectory predicted outcomes for each plan action
"""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any

import numpy as np
from src.config.hyperparameters import GAMMA_SLO, UTILIZATION_TARGET_ONLINE
from src.core.models import (
    LADDER_ACTIONS,
    SWAP_BUDGET_ACTIONS,
    ActionType,
    Plan,
    PlanAction,
    RankSpec,
    env_gpu_type,
)
from src.infra.deployment_x import build_rank_x

# Residual calibration: debias the surrogate with observed (observed-predicted)
# residuals from similar past deployments, so scoring uses reality-corrected
# predictions as the performance database grows.
CALIBRATION_WINDOW = 50  # ticks of evidence to draw similar rows from
CALIBRATION_MIN_SAMPLES = 5  # below this, leave the objective uncorrected
_NONNEGATIVE_Y = frozenset(
    {
        "throughput_token_per_sec",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "cost_per_token",
    }
)

# The ONLY knobs the planner may propose: placement (where), topology, model
# parallelism, and the disaggregated prefill/decode worker split. Everything
# else (engine, router, quantization, cache flags, scheduling, batch autotune)
# is engine/catalog-owned - see _ENGINE_OWNED_X.
AGENT_TUNABLE_X = frozenset(
    {
        # placement / environment
        "market",
        "cloud",
        "region",
        "gpu_type",
        "instance_type",
        # topology
        "num_nodes_per_chain",
        "interconnect_type",
        # model parallelism
        "tp",
        "pp",
        "sp",
        "dp",
        "ep",
        "cp",
        # disaggregated prefill/decode worker split
        "prefill_worker_count",
        "decode_worker_count",
    }
)

# Engine-AUTOTUNED batch knobs: never valid from the agent OR from workload
# features. Stripped from BOTH config and features.
_ENGINE_AUTOTUNED_X = frozenset({"max_num_seq", "max_num_batched_tokens", "block_size"})

# Engine/catalog-owned CONFIG knobs the agent must NOT set (not in the allowed
# proposal set). The engine/catalog supplies them, and the valid workload value
# for something like router_policy lives in the job's FEATURES - so these are
# dropped from the agent's CONFIG only and kept in features. This is what stops
# an invented value like router_policy='latency' from ever reaching the surrogate.
_ENGINE_OWNED_X = frozenset(
    {
        "engine_name",
        "engine_version",
        "weight_dtype",
        "kvcache_dtype",
        "weight_quantization_bits",
        "prefix_cache_enabled",
        "chunked_prefill_enable",
        "router_policy",
        "scheduling_policy",
        "preemption_policy",
        "gpu_mem_util",
        "kv_transfer_method",
    }
)


def _sanitize_agent_config(config: dict[str, Any]) -> dict[str, Any]:
    """Strip engine-owned + engine-autotuned knobs from an agent-proposed CONFIG.

    The agent may only propose the placement/topology/parallelism/PD knobs in
    AGENT_TUNABLE_X; everything else is engine/catalog-owned and is removed here
    so an invalid invented value (router_policy='latency', an unsupported tp,
    etc.) can never reach the surrogate. The catalog, workload features, and
    surrogate defaults supply the real values.
    """
    drop = _ENGINE_AUTOTUNED_X | _ENGINE_OWNED_X
    return {key: value for key, value in (config or {}).items() if key not in drop}


def _sanitize_agent_features(features: dict[str, Any]) -> dict[str, Any]:
    """Strip only engine-AUTOTUNED batch knobs from WORKLOAD features.

    Features legitimately carry workload-owned values the surrogate needs
    (router_policy='kv_router', isl/osl, arrival rate, ...), so we keep the
    engine-owned set here and only remove the three batch-autotune knobs the
    agent must never smuggle in through features.
    """
    return {key: value for key, value in (features or {}).items() if key not in _ENGINE_AUTOTUNED_X}


# --- Scoring priors (P1) --------------------------------------------------
# The slow loop OWNS z*/typical_ranges, but at cold start it hands z*=0 and a
# {name: 1.0} range stub. That makes the Tchebycheff gap = raw magnitude and
# collapses J to ~ -50, so every placement loses to defer=0. Until the slow
# loop is seeded, the scorer defends itself by substituting these domain priors
# for degenerate values. Objectives: p99_ttft_ms / p99_tpot_ms / cost_per_token
# are MINIMIZED; throughput_token_per_sec / slo_margin are MAXIMIZED.
# cost_per_token values are order-of-magnitude - set them to YOUR cost units.
DEFAULT_TYPICAL_RANGES = {
    "p99_ttft_ms": 500.0,
    "p99_tpot_ms": 50.0,
    "cost_per_token": 5e-6,
    "throughput_token_per_sec": 1000.0,
    "slo_margin": 1.0,
}
DEFAULT_COLD_START_Z_STAR = {
    "p99_ttft_ms": 100.0,
    "p99_tpot_ms": 20.0,
    "cost_per_token": 1e-6,
    "throughput_token_per_sec": 3000.0,
    "slo_margin": 1.0,
}
# Opportunity cost charged per WAITING job the plan leaves unserved, so a
# feasible placement (sigma ~ J<=0 + EIG) beats defer (0). Scaled by priority.
UNSERVED_PENALTY = 1.0


def _seeded_ranges(ranges: dict[str, Any] | None) -> dict[str, float]:
    """Replace degenerate (missing / 0 / 1.0-stub) per-objective ranges with
    domain priors, so a Tchebycheff gap is O(1) instead of raw magnitude."""
    merged = dict(ranges or {})
    for obj, default in DEFAULT_TYPICAL_RANGES.items():
        if merged.get(obj) in (None, 0, 0.0, 1.0):
            merged[obj] = default
    return merged


# The slow loop initializes z_star_t to a UNIFORM placeholder before it has any
# evidence - historically all-zero, and in current builds all-99999 (a five-nines
# "unset" sentinel, seen in live traces). Neither is a real ideal: fed to
# Tchebycheff the gap becomes a raw sentinel magnitude and J blows up to +/-
# millions, inverting the contract (J <= 0). The primary detector is direction-
# and value-agnostic (a z* that is uniform across every objective is a
# placeholder, whatever the constant); the explicit sentinel set is documented
# insurance for a PARTIALLY-degenerate vector.
_Z_STAR_UNSET_SENTINELS = frozenset({0.0, 99999.0})


def _is_placeholder_z_star(values: list[float]) -> bool:
    """True when a z* vector is empty or uniform across all objectives - i.e. the
    slow loop's cold-start placeholder, not an evidence-derived ideal (real z*
    values are heterogeneous: cost ~1e-6, latency ~100, throughput ~thousands)."""
    if not values:
        return True
    return len(values) >= 2 and len(set(values)) == 1


def _seeded_z_star(z_star: dict[str, Any] | None) -> dict[str, float]:
    """Replace a missing / non-physical / placeholder z* with domain priors, so
    Tchebycheff distance is O(1) and J stays <= 0. A degenerate value is None,
    non-numeric, non-finite, <= 0 (dimensionally wrong for every objective), the
    slow loop's unset sentinel, or a member of a uniform placeholder vector.
    Evidence-set, heterogeneous, positive values are left untouched."""
    merged = dict(z_star or {})
    numeric = [
        v
        for v in merged.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    placeholder = _is_placeholder_z_star(numeric)
    for obj, default in DEFAULT_COLD_START_Z_STAR.items():
        v = merged.get(obj)
        degenerate = (
            v is None
            or isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
            or float(v) in _Z_STAR_UNSET_SENTINELS
            or placeholder
        )
        if degenerate:
            merged[obj] = default
    return merged


log = logging.getLogger("koi.agent_tools")
_PREDICT_RAW_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}


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
    user_registry = None
    specialist_runner = None
    tool_call_logger = None
    cluster_snapshot = None

    # Per-tick caches written by the budget tools.
    user_envelopes = None
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
            regret_calculator, user_registry, specialist_runner).
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
# tool at a time deep inside a trajectory. user_registry is intentionally
# absent (the Store user_id owns all capacity in v0), and
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


# Per-tick surrogate-call budget: a runaway BACKSTOP, not a quality limiter.
# The planner may explore freely (grid / coordinate search) up to this many
# DISTINCT surrogate simulations per tick; beyond it a sim raises
# SurrogateBudgetExceeded, which size_ladder / optimize_config / the specialist
# eval loops catch and treat as an infeasible/skipped frame - so hitting it late
# just means "commit the best scored so far", never a crash or forced defer.
# Cache hits are free and do NOT count. Raise it for deeper search; it is
# generous by design (a normal tick uses far fewer).
SURROGATE_CALL_BUDGET = 100
_surrogate_calls = 0


class SurrogateBudgetExceeded(RuntimeError):
    """Raised when a tick exceeds SURROGATE_CALL_BUDGET distinct surrogate sims."""


def reset_tick_caches() -> None:
    """Clear per-tick caches: user envelopes and the validated BudgetBook.

    Must run at every tick boundary (S0 wires it via the TickRunner's
    on_tick_start hook). Without this, run_job_specialists' default-book
    path could reuse a book validated against LAST tick's capacity -
    a stale-budget hole in the anti-split-brain ordering.
    """
    global _surrogate_calls
    _CTX.user_envelopes = None
    _CTX.validated_budget_book = None
    _PREDICT_RAW_CACHE.clear()
    _surrogate_calls = 0


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
    tools = {
        name: fn
        for name, fn in globals().items()
        if callable(fn)
        and not name.startswith("_")
        and name not in _NON_TOOL_NAMES
        and getattr(fn, "__module__", None) == __name__
    }
    logger = getattr(_CTX, "tool_call_logger", None)
    if logger is None:
        return tools
    return {name: _logged_tool(name, fn, logger) for name, fn in tools.items()}


def _short(value: Any, limit: int = 500) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _logged_tool(name, fn, logger):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.time()
        logger(
            {
                "kind": "tool_call_started",
                "name": name,
                "args": [_short(arg) for arg in args],
                "kwargs": {key: _short(value) for key, value in kwargs.items()},
            }
        )
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            logger(
                {
                    "kind": "tool_call_error",
                    "name": name,
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": repr(exc),
                }
            )
            raise
        logger(
            {
                "kind": "tool_call_finished",
                "name": name,
                "elapsed_sec": round(time.time() - started, 3),
                "result": _short(result),
            }
        )
        return result

    return wrapper


def _env_key(env) -> str:
    """Normalize an env identifier (tuple or string) to a flat string key."""
    if isinstance(env, (tuple, list)):
        return "|".join(str(part) for part in env)
    return str(env)


def _snapshot():
    _require("resource_map")
    return _CTX.cluster_snapshot or _CTX.resource_map.snapshot()


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
    ranks = []
    for rank in action.ladder:
        raw = rank.to_dict()
        raw["mechanism_id"] = raw.get("mechanism_id") or action.mechanism_id
        if raw["mechanism_id"] is None:
            raise ValueError(f"job {action.job_id}: ladder rank requires mechanism_id")
        ranks.append(raw)
    return ranks


def _job_features_for(snapshot, job_id: str) -> dict[str, Any]:
    """Return the workload features for a job in the current snapshot."""
    if snapshot is None:
        return {}
    for accessor in ("active_jobs_summary", "pending_jobs_summary"):
        if not hasattr(snapshot, accessor):
            continue
        for job in getattr(snapshot, accessor)() or []:
            if job.get("job_id", job.get("id")) == job_id:
                return dict(job.get("job_features") or {})
    return {}


def _rank_prediction_payload(rank: RankSpec, job_features: dict[str, Any] | None = None) -> dict:
    """Build the surrogate payload for one rank without mutating the rank."""
    features = _sanitize_agent_features(dict(job_features or {}))
    env = None
    if rank.env is not None:
        env = list(rank.env) if isinstance(rank.env, (list, tuple)) else str(rank.env).split("|")
        if len(env) >= 5:
            features.update(
                {
                    "market": env[0],
                    "cloud": env[1],
                    "region": env[2],
                    "zone": env[3],
                    "gpu_type": env[4],
                }
            )
    if rank.config.get("instance_type") is not None:
        features["instance_type"] = rank.config["instance_type"]

    config = _sanitize_agent_config(dict(rank.config))
    if "model_id" not in config and features.get("model_id") is not None:
        config["model_id"] = features["model_id"]
    resource_map = getattr(_CTX, "resource_map", None)
    model_id = config.get("model_id") or features.get("model_id")
    if env and model_id and resource_map is not None:
        try:
            count = rank.gpus_per_chain()
            shape = {**config, "env": list(env), "count": count, "gpu_count": count}
            compiled_x = build_rank_x(
                job_values=features,
                shape=shape,
                env=(str(env[0]), str(env[1]), str(env[2]), str(env[3]), str(env[4])),
                resources=resource_map.resources_summary(),
                hardware_catalog=resource_map.hardware_catalog(),
                model_catalog=resource_map.model_catalog(str(model_id)),
                replica_count=max(1, int(rank.n_replicas or 1)),
            )
            config.update(compiled_x)
        except Exception:
            log.exception("rank prediction X assembly failed; using rank config only")
    return {"job_config": config, "job_features": features}


def _rank_mechanism_context(rank: RankSpec, job_features: dict[str, Any]) -> dict[str, Any]:
    """Return the same enriched rank context used for prediction."""
    payload = _rank_prediction_payload(rank, job_features)
    return {**payload["job_features"], **payload["job_config"], "dp": rank.n_replicas}


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


# Job-outcome composition across a (possibly heterogeneous) multi-rank ladder.
# y_hat is per RANK (per config); the JOB's outcome composes the ranks. The
# composed values stay on a per-chain scale so they remain comparable to
# z_star / typical_ranges, which are computed from per-rank evidence:
#   latency (ttft/tpot) -> max across ranks (a request hits one rank; the SLO
#                          must hold for every serving rank)
#   throughput          -> replica-weighted mean per-chain (intensive). Total
#                          throughput vs target is size_ladder's job, not J's.
#   cost_per_token      -> throughput-weighted mean (= total$ / total tokens)
#   slo_margin          -> min across ranks (worst headroom)
_LATENCY_OBJS = frozenset({"p99_ttft_ms", "p99_tpot_ms", "p99_TTFT_ms", "p99_TPOT_ms"})
_THROUGHPUT_OBJ = "throughput_token_per_sec"
_COST_OBJS = frozenset({"cost_per_token"})
_MARGIN_OBJS = frozenset({"slo_margin"})


def _compose_job_y_hat(action, job_features: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compose a job-level y_hat from a ladder's per-rank predictions.

    y_hat is predicted per rank (per config); this rolls the ranks up to the
    single job-level outcome that J and Pr_DRO are scored on. Honors the
    action's advisory predicted_y when the planner attached one (it already
    composed). A single-rank ladder returns that rank's y_hat unchanged, so
    homogeneous ladders are unaffected.
    """
    # TODO - I can debate this as we don't need the LLM to pass the predicted_y
    # we want it to CALL The SUrrogate ALWAYS
    # so i am, for now, removing this call.
    # if action.predicted_y:
    #     return dict(action.predicted_y)
    samples: list[tuple[int, dict]] = []
    for rank in action.ladder or []:
        try:
            payload = _rank_prediction_payload(rank, job_features)
            y = _predict_outcome_core(payload["job_config"], payload["job_features"]).get(
                "y_hat", {}
            )
        except Exception:
            log.exception("rank y_hat failed for job %s", action.job_id)
            y = {}
        if y:
            samples.append((max(1, int(rank.n_replicas or 1)), y))
    if not samples:
        return {}
    # Always roll up (even a single rank) so DP is applied: a lone rank with
    # n_replicas=N must report N * per_chain throughput, not per_chain.
    return _roll_up_ranks(samples)


def _roll_up_ranks(samples: list[tuple[int, dict]]) -> dict[str, Any]:
    """Roll up per-rank (n_replicas, y_hat) samples into one job y_hat."""

    def _tput(y: dict) -> float:
        return float(y.get(_THROUGHPUT_OBJ) or 0.0)

    objectives = set().union(*[set(y) for _, y in samples])
    tput_weight_total = sum(n * _tput(y) for n, y in samples)
    composed: dict[str, Any] = {}
    for obj in objectives:
        present = [(n, float(y[obj]), _tput(y)) for n, y in samples if y.get(obj) is not None]
        if not present:
            continue
        if obj == _THROUGHPUT_OBJ:
            # Data-parallel replicas serve IN PARALLEL, so aggregate throughput is
            # the SUM over replicas (n_replicas * per_chain_tps), NOT an average.
            # Scoring per-chain here under-counted delivered throughput by the
            # replica factor - it made every multi-replica job's J far too negative
            # (big models, which need the most replicas, hit hardest -> deferred).
            # size_ladder already scales by replicas for meets_target; J must match.
            composed[obj] = sum(n * v for n, v, _ in present)
        elif obj in _LATENCY_OBJS:
            composed[obj] = max(v for _, v, _ in present)
        elif obj in _MARGIN_OBJS:
            composed[obj] = min(v for _, v, _ in present)
        elif obj in _COST_OBJS and tput_weight_total > 0:
            composed[obj] = sum(n * t * v for n, v, t in present) / tput_weight_total
        else:
            weight = sum(n for n, _, _ in present) or 1
            composed[obj] = sum(n * v for n, v, _ in present) / weight
    return composed


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
        Env keys use market|cloud|region|zone|gpu_type.
    """
    snap = _snapshot()
    return snap.resources_summary() if hasattr(snap, "resources_summary") else {}


def get_active_jobs() -> list[dict[str, Any]]:
    """Return descriptors for currently running jobs.

    Returns:
        List of dicts with at least job_id, user_id, current ladder
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


def get_priority() -> list[dict[str, Any]]:
    """Build a deterministic priority table for jobs.

    Combines user priority, job class, online/batch, deadline pressure,
    SLO margin, queue age, and recent failure signals into one score.
    The root reads this table instead of raw job data, then inspects
    specific jobs near decision boundaries.

    Returns:
        List of {"job_id", "user_id", "priority_score", "signals"}
        sorted by descending score.
    """
    jobs = list(get_pending_jobs()) + list(get_active_jobs())
    scored: list[dict[str, Any]] = []
    for j in jobs:
        signals = {
            "user_priority": float(j.get("user_priority", 1.0)),
            "priority_class": float(j.get("priority_class", 0)),
            "is_online": 1.0 if j.get("type", "online") == "online" else 0.0,
            "deadline_pressure": float(j.get("deadline_pressure", 0.0)),
            "slo_margin_deficit": max(0.0, -float(j.get("slo_margin", 0.0))),
            "queue_age_ticks": float(j.get("queue_age_ticks", 0)),
            "recent_failures": float(j.get("recent_failures", 0)),
        }
        score = (
            signals["user_priority"] * 10.0
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
                "user_id": j.get("user_id"),
                "priority_score": score,
                "signals": signals,
            }
        )
    scored.sort(key=lambda x: float(x["priority_score"]), reverse=True)
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
        Dict with job_id, user_id, job_features, current_ladder,
        recent_q_labels, recent_theory_blobs, similar_deployments,
        mechanism_candidates.
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
    spec = dict((descriptor or {}).get("spec_json") or {})
    model_id = features.get("model_id") or spec.get("model_id")
    model_catalog = {}
    if model_id and hasattr(_CTX.resource_map, "model_catalog"):
        model_catalog = dict(_CTX.resource_map.model_catalog(str(model_id)) or {})
    mechanism_context = {**model_catalog, **features}
    mechanism_context["workload_type"] = (
        features.get("workload_type") or features.get("type") or (descriptor or {}).get("kind")
    )
    mechanisms = get_scope(mechanism_context)

    return {
        "job_id": job_id,
        "user_id": (descriptor or {}).get("user_id"),
        "job_features": features,
        "model_catalog": model_catalog,
        "current_ladder": (descriptor or {}).get("current_ladder"),
        "recent_q_labels": recent_q,
        "recent_theory_blobs": blobs,
        "similar_deployments": get_similar_deployments(features, top_k=5),
        "mechanism_candidates": mechanisms,
        "instance_catalog": instance_catalog(),
    }


def instance_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    """Per-env INSTANCE facts from the resource map, so the planner/specialist
    can size correctly instead of guessing hardware it cannot know from training.

    For each env_key: {instance_type: {gpus_per_instance, free_instances,
    gpu_type, price_per_instance_hour}}. gpus_per_instance is the key fact: a
    rank's config.gpu_count is the GPUs used per replica on ONE instance and must
    be <= gpus_per_instance * num_nodes_per_chain. So the right instance for a
    tp=8 frame is an 8-GPU box (e.g. p5.48xlarge), NOT eight 1-GPU boxes
    (p5.4xlarge). pool_budget UNIT counts are how many instances are free, which
    is NOT the same as GPUs per instance - do not confuse them.
    """
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    for env_key, info in get_resource_map().items():
        env_map: dict[str, dict[str, Any]] = {}
        for pool in info.get("pools") or []:
            instance_type = pool.get("instance_type")
            if not instance_type:
                continue
            env_map[str(instance_type)] = {
                "gpus_per_instance": int(pool.get("gpus_per_instance", 0) or 0),
                "free_instances": int(pool.get("free_instances", 0) or 0),
                "gpu_type": info.get("gpu_type") or str(env_key).split("|")[-1],
                "price_per_instance_hour": pool.get("price_per_instance_hour"),
            }
        if env_map:
            catalog[str(env_key)] = env_map
    return catalog


# ----------------------------------------------------------------------
# User / budget tools
# ----------------------------------------------------------------------


def _pool_limits(resources: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    resource_map = getattr(_CTX, "resource_map", None)
    if resource_map is None or not hasattr(resource_map, "pool_capacity"):
        return {}
    return resource_map.pool_capacity(resources)


def _pool_budget_by_env(resources: dict[str, Any]) -> dict[str, dict[str, int]]:
    budget: dict[str, dict[str, int]] = {}
    for (env, instance_type), limit in _pool_limits(resources).items():
        budget.setdefault(env, {})[instance_type] = int(limit["available_units"])
    return budget


def _parse_pool_budget(slice_: dict[str, Any]) -> tuple[dict[str, dict[str, int]], list[str]]:
    raw = slice_.get("pool_budget") or {}
    if not isinstance(raw, dict):
        return {}, ["pool_budget must be a dict"]
    budget: dict[str, dict[str, int]] = {}
    violations = []
    for env, pools in raw.items():
        env_key = _env_key(env)
        if not isinstance(pools, dict):
            violations.append(f"pool_budget[{env_key}] must be a dict")
            continue
        for instance_type, units in pools.items():
            if isinstance(units, bool) or not isinstance(units, int) or units < 0:
                violations.append(f"pool budget for {instance_type} in {env_key} must be >= 0")
                continue
            budget.setdefault(env_key, {})[str(instance_type)] = units
    return budget, violations


def build_user_envelopes() -> dict[str, dict[str, Any]]:
    """Build deterministic user envelopes for this tick.

    Envelopes are the legal resource boundary per user: floors,
    ceilings, quotas, and env allow/deny lists. The root reasons over
    them but cannot exceed them. With no user_registry bound, each Store
    user owns all capacity exposed by their resource map.

    Returns:
        Dict user_id -> envelope dict. Also cached for get_user_envelopes
        and validate_budget_book.
    """
    resources = get_resource_map()
    capacity = {_env_key(env): int(info.get("free", 0)) for env, info in resources.items()}
    current_user_id = getattr(getattr(_CTX, "resource_map", None), "user_id", None)
    if not isinstance(current_user_id, str) or not current_user_id:
        raise ValueError("resource_map.user_id is required to build user envelopes")

    if _CTX.user_registry is None:
        envelopes: dict[str, dict[str, Any]] = {
            current_user_id: {
                "user_id": current_user_id,
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
        users = _CTX.user_registry.list_users()
        total_weight = sum(float(u.get("fairness_weight", 1.0)) for u in users) or 1.0
        envelopes = {}
        for u in users:
            user_id = str(u["user_id"])
            weight = float(u.get("fairness_weight", 1.0))
            share = {env: int(free * weight / total_weight) for env, free in capacity.items()}
            envelopes[user_id] = {
                "user_id": user_id,
                "priority_tier": u.get("priority_tier", "standard"),
                "fairness_weight": weight,
                "guaranteed_floor": dict(u.get("guaranteed_floor", {})),
                "burst_ceiling": dict(u.get("burst_ceiling", share)),
                "hard_quota": dict(u.get("hard_quota", share)),
                "allowed_envs": list(u.get("allowed_envs", capacity.keys())),
                "denied_envs": list(u.get("denied_envs", [])),
                "budget_usd_remaining": u.get("budget_usd_remaining"),
                "can_use_spot": bool(u.get("can_use_spot", False)),
            }

    _CTX.user_envelopes = envelopes
    return envelopes


def get_user_envelopes() -> dict[str, dict[str, Any]]:
    """Return the cached user envelopes, building them if needed."""
    if _CTX.user_envelopes is None:
        return build_user_envelopes()
    return _CTX.user_envelopes


def allocate_budget_book() -> dict[str, Any]:
    """Build the default BudgetBook expected by ``validate_budget_book``.

    v0 hands every job a PERMISSIVE upper-bound budget: each job may see the FULL
    free pool in every env it is allowed (capped by the user's hard quota), NOT
    an exclusive slice. Budgets deliberately OVERLAP across jobs. The old
    pre-partition was the split-brain bug: a fair-share split stranded H100 with
    a job that did not need it and boxed the one that did out of the 8-GPU frame
    it wanted, so both under-served and deferred. The real cross-job capacity
    decision is now made GLOBALLY, after specialists propose, by
    jointly_select_placements (joint GPU selection) + check_feasibility. So a
    specialist is free to propose the best GPU for its job, and the root
    reconciles the ONE shared pool jointly instead of guessing the split up front.
    """
    _require("slow_loop")
    resources = get_resource_map()
    envelopes = get_user_envelopes()
    pending = list(get_pending_jobs())
    active = list(get_active_jobs())
    pending_ids = {j.get("job_id", j.get("id")) for j in pending}
    active_ids = {j.get("job_id", j.get("id")) for j in active}
    by_id = {j.get("job_id", j.get("id")): j for j in pending + active}
    free_env = {_env_key(env): int(info.get("free", 0)) for env, info in resources.items()}
    free_pools = {env: dict(pools) for env, pools in _pool_budget_by_env(resources).items()}
    priorities = get_priority()
    job_budgets: dict[str, dict[str, Any]] = {}

    for entry in priorities:
        job_id = entry.get("job_id")
        if not job_id or job_id not in by_id:
            continue
        job = by_id[job_id]
        user_id = job.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"job {job_id!r} missing user_id")
        envelope = envelopes.get(user_id, {})
        allowed = {_env_key(env) for env in envelope.get("allowed_envs", free_env.keys())}
        denied = {_env_key(env) for env in envelope.get("denied_envs", [])}
        quota = {_env_key(e): int(n) for e, n in envelope.get("hard_quota", {}).items()}
        # Full free pool per allowed env, capped only by the user's hard quota.
        # Overlaps other jobs on purpose - it is an upper bound, not a reservation.
        env_budget: dict[str, int] = {}
        for env in allowed - denied:
            cap = free_env.get(env, 0)
            if env in quota:
                cap = min(cap, quota[env])
            if cap > 0:
                env_budget[env] = cap
        pool_budget = {
            env: dict(free_pools.get(env, {})) for env in env_budget if free_pools.get(env)
        }

        is_pending = job_id in pending_ids or job.get("status") == "waiting"
        is_active = job_id in active_ids or job.get("status") == "running"
        job_budgets[job_id] = {
            "slice_id": job_id,
            "user_id": user_id,
            "job_id": job_id,
            "env_budget": env_budget,
            "pool_budget": pool_budget,
            "allowed_actions": ["place", "defer"] if is_pending else ["keep", "swap"],
            "strategy_hint": "place"
            if is_pending and env_budget
            else "keep"
            if is_active
            else "defer",
            "canary_cap": 1,
            "priority_score": entry.get("priority_score", 0.0),
            "notes": "permissive upper-bound budget (full free pool; joint selector reconciles)",
        }

    return {
        "tick": int(getattr(_CTX.slow_loop.state, "tick", 0)),
        "job_budgets": job_budgets,
        "reserves": {},
        "rationale": "permissive upper-bound budgets: each job sees the full free pool (capped by quota); jointly_select_placements + check_feasibility reconcile across jobs",
    }


def validate_budget_book(budget_book: dict[str, Any]) -> dict[str, Any]:
    """Deterministically validate a BudgetBook before specialists run.

    Checks, in order:
        1. Every job budget references a known user envelope.
        2. No job budget uses an env denied to its user.
        3. Each job's per-env budget stays within the user hard quota.
        4. Each job's per-env budget stays within free capacity minus reserves.
        5. Implied active-job swaps stay within the swap budget B_t.

    Budgets are PERMISSIVE upper bounds that overlap across jobs (see
    allocate_budget_book), so each is validated as a per-job cap, NOT a
    cross-job partition sum; the real shared-pool fit is enforced later by
    jointly_select_placements + check_feasibility.

    On success the book is cached so run_job_specialists can verify it
    was validated. Any change to the book requires re-validation.

    Args:
        budget_book: {"tick": int, "job_budgets": {job_id: slice},
            "reserves": {env_key: int}, "rationale": str}. Each slice is
            {"user_id", "job_id", "env_budget": {env_key: gpus},
            "pool_budget": {env_key: {instance_type: units}},
            "allowed_actions", "strategy_hint", "canary_cap",
            "priority_score", "notes"}.

    Returns:
        {"ok": bool, "violations": List[str]}.
    """
    _require("slow_loop")
    violations: list[str] = []
    envelopes = get_user_envelopes()
    resources = get_resource_map()
    capacity = {_env_key(env): int(info.get("free", 0)) for env, info in resources.items()}
    pool_limits = _pool_limits(resources)
    reserves = {_env_key(env): int(n) for env, n in (budget_book.get("reserves") or {}).items()}

    job_budgets = budget_book.get("job_budgets") or {}
    allocatable = {env: capacity.get(env, 0) - reserves.get(env, 0) for env in capacity}
    implied_swaps = 0
    active_ids = {j.get("job_id", j.get("id")) for j in get_active_jobs()}

    # Budgets are permissive per-job UPPER BOUNDS (they overlap across jobs), so
    # each is validated as a cap - never a cross-job sum. The shared-pool fit is
    # decided globally by jointly_select_placements + check_feasibility.
    for job_id, slice_ in job_budgets.items():
        user_id = slice_.get("user_id")
        envelope = envelopes.get(user_id)
        if envelope is None:
            violations.append(f"job {job_id}: unknown user {user_id!r}")
            continue

        denied = {_env_key(e) for e in envelope.get("denied_envs", [])}
        quota = {_env_key(e): int(n) for e, n in envelope.get("hard_quota", {}).items()}
        env_budget = {
            _env_key(env): int(gpus) for env, gpus in (slice_.get("env_budget") or {}).items()
        }
        for key, gpus in env_budget.items():
            if gpus < 0:
                violations.append(f"job {job_id}: negative budget in {key}")
                continue
            if key in denied:
                violations.append(f"job {job_id}: env {key} denied for user {user_id}")
            if gpus > allocatable.get(key, 0):
                violations.append(
                    f"job {job_id}: {gpus} GPUs in {key} exceeds free capacity "
                    f"{allocatable.get(key, 0)}"
                )
            limit = quota.get(key)
            if limit is not None and gpus > limit:
                violations.append(f"job {job_id}: {gpus} GPUs in {key} exceeds user quota {limit}")

        pool_budget, pool_errors = _parse_pool_budget(slice_)
        violations.extend(f"job {job_id}: {error}" for error in pool_errors)
        pool_envs = {env for env, _ in pool_limits}
        for env, gpus in env_budget.items():
            if gpus > 0 and env in pool_envs and env not in pool_budget:
                violations.append(f"job {job_id}: pool_budget is required for env {env}")
        for env, pools in pool_budget.items():
            env_key = _env_key(env)
            if env_key not in env_budget:
                violations.append(f"job {job_id}: pool budget env {env_key} has no env budget")
            for instance_type, units in pools.items():
                pool_limit = pool_limits.get((env_key, str(instance_type)))
                if pool_limit is None:
                    violations.append(
                        f"job {job_id}: pool {instance_type} is not available in env {env_key}"
                    )
                elif int(units) > int(pool_limit["available_units"]):
                    violations.append(
                        f"job {job_id}: {units} units of {instance_type} in {env_key} exceed "
                        f"{pool_limit['available_units']} free"
                    )

        hint = str(slice_.get("strategy_hint", "")).lower()
        if job_id in active_ids and any(
            word in hint for word in ("swap", "migrate", "replace", "move")
        ):
            implied_swaps += 1

    b_t = _CTX.slow_loop.get_sss_swap_budget_t()
    if implied_swaps > b_t:
        violations.append(f"implied swaps {implied_swaps} exceed swap budget B_t={b_t}")

    ok = len(violations) == 0
    _CTX.validated_budget_book = budget_book if ok else None
    # "feasible" mirrors "ok" so a planner that standardizes on either key reads
    # both validation tools consistently (see check_feasibility).
    return {"ok": ok, "feasible": ok, "violations": violations}


def _budget_violations(
    action,
    slice_: dict[str, Any],
    resources: dict[str, Any] | None = None,
) -> list[str]:
    """Check a ladder's actual reserved capacity against its BudgetSlice."""
    resource_map = getattr(_CTX, "resource_map", None)
    if resource_map is None or not hasattr(resource_map, "requested_capacity"):
        return []
    resources = resources if resources is not None else get_resource_map()
    by_env, by_pool = resource_map.requested_capacity(Plan(tick=0, actions=[action]), resources)
    env_budget = {
        _env_key(env): int(value) for env, value in (slice_.get("env_budget") or {}).items()
    }
    pool_budget, violations = _parse_pool_budget(slice_)
    violations.extend(
        f"reserved capacity {used} in {env} exceeds slice budget {env_budget.get(env, 0)}"
        for env, used in by_env.items()
        if used > env_budget.get(env, 0)
    )
    for (env, instance_type), demand in by_pool.items():
        allowed = pool_budget.get(env, {}).get(instance_type, 0)
        if demand["units"] > allowed:
            violations.append(
                f"pool {instance_type} in {env} needs {demand['units']} units, "
                f"slice allows {allowed}"
            )
    return violations


def run_job_specialists(
    max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    """Run bounded per-job specialists under a validated BudgetBook.

    Refuses to run when the supplied book is not the one most recently
    validated by validate_budget_book - that ordering is the
    anti-split-brain invariant. Each specialist optimizes one job inside
    its BudgetSlice and reports a fitness signal; it cannot allocate
    outside its slice or see the cluster plan.

    Args:
        max_workers: Parallel specialist calls.

    Returns:
        Dict job_id -> JobSpecialistResult ({"job_id", "type", "ladder",
        "predicted_y", "predicted_sigma", "budget_utilization",
        "fitness", "marginal_value_of_more", "unused_capacity",
        "mechanism_ids", "new_mechanism_proposals", "reasoning"}).

    Raises:
        RuntimeError: If no validated book exists or no specialist
            runner is bound.
    """
    _require("specialist_runner")
    book = _CTX.validated_budget_book
    if book is None:
        raise RuntimeError(
            "run_job_specialists requires the BudgetBook most recently "
            "validated by validate_budget_book. Validate first."
        )
    job_ids = list((book.get("job_budgets") or {}).keys())
    results = _CTX.specialist_runner.run_many(
        jobs=job_ids, budget_book=book, max_workers=int(max_workers)
    )
    return {str(result.get("job_id")): result for result in results}


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
        deadline_s = (
            float(job_features.get("deadline_hours", job_features.get("deadline_hrs", 24.0)))
            * 3600.0
        )
        return budget / max(1.0, deadline_s) * headroom
    rate = float(job_features.get("request_arrival_rate", 0.0))
    out_avg = float(
        job_features.get("output_len_tokens_avg", job_features.get("osl_token_avg", 0.0))
    )
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


def _rank_allocation_summary(rank, resources=None) -> dict[str, Any]:
    resource_map = getattr(_CTX, "resource_map", None)
    if resource_map is not None and hasattr(resource_map, "rank_allocation_summary"):
        return resource_map.rank_allocation_summary(rank, resources)
    engine_gpus = rank.gpus_per_chain()
    return {
        "allocation_kind": "gpu",
        "instance_type": rank.config.get("instance_type"),
        "gpus_per_unit": engine_gpus,
        "price_per_unit_hour": None,
        "capacity_per_replica": engine_gpus,
        "engine_gpus": engine_gpus,
    }


def _model_num_heads(config: dict[str, Any], job_features: dict[str, Any] | None) -> int | None:
    """Best-effort attention-head count for the rank's model, from the rank
    config, the job features, or the model catalog. None when unknown (then
    head-divisibility is not enforced here and the surrogate stays the backstop).
    """
    for source in (config or {}), (job_features or {}):
        for key in (
            "num_attn_heads",
            "num_attention_heads",
            "n_heads",
            "num_heads",
            "attention_heads",
        ):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    model_id = (config or {}).get("model_id") or (job_features or {}).get("model_id")
    resource_map = getattr(_CTX, "resource_map", None)
    if model_id and resource_map is not None and hasattr(resource_map, "model_catalog"):
        try:
            catalog = dict(resource_map.model_catalog(str(model_id)) or {})
        except Exception:
            return None
        for key in (
            "num_attn_heads",
            "num_attention_heads",
            "n_heads",
            "num_heads",
            "attention_heads",
        ):
            value = catalog.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def config_runnable(
    config: dict[str, Any],
    job_features: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Deterministic physical-validity pre-check for a rank config.

    Enforces the HARD constraints the model/hardware impose - tp*pp must fit the
    engine's GPU demand, and tp must divide the model's attention-head count - in
    CODE, so an unrunnable config (e.g. tp=8 on a 28-head model) is rejected with
    a clear reason instead of being nagged about in the prompt or crashing the
    surrogate. Checks it cannot evaluate (missing catalog arch) are skipped, not
    failed - the surrogate stays the backstop for those. Returns (ok, reason).
    """
    config = config or {}
    try:
        tp = int(config.get("tp") or 1)
        pp = int(config.get("pp") or 1)
    except (TypeError, ValueError):
        return True, ""  # non-numeric parallelism - let the schema/validator handle it
    if tp < 1 or pp < 1:
        return False, f"tp={tp} and pp={pp} must both be >= 1"
    gpu_count = config.get("gpu_count")
    if isinstance(gpu_count, int) and not isinstance(gpu_count, bool) and tp * pp > gpu_count:
        return False, f"tp*pp={tp * pp} exceeds gpu_count={gpu_count} (need one GPU per shard)"
    heads = _model_num_heads(config, job_features)
    if heads and heads % tp != 0:
        return False, f"tp={tp} does not divide the model's {heads} attention heads (cannot shard)"
    return True, ""


def size_ladder(
    ranks: list[dict[str, Any]],
    job_features: dict[str, Any],
    target_tps: float | None = None,
    utilization_target: float | None = None,
) -> dict[str, Any]:
    """Size each rank's replica count to meet ONE shared throughput target.

    The planner proposes rank CONFIGS (gpu, tp, pp, engine, quant), possibly
    HETEROGENEOUS - e.g. an H100 rank and an A100 rank for the same job; this
    derives each rank's replica/dp count instead of leaving it to a guess.
    Ranks are filled in the order given, each covering the REMAINING target,
    so the ladder's achieved throughput SUMS across its ranks (parallel
    replicas, not a series). v0 is aggregate-only; the disaggregated
    prefill->decode SERIES case (achieved = min across roles) is deferred.

        target_tps    = required throughput (batch: budget/deadline*headroom;
                        online: arrival_rate * output_len * headroom)
        per_chain_eff = per_chain_raw * utilization_target  (online derates
                        below saturation so queue wait ~ rho/(1-rho) and thus
                        p99 TTFT stay bounded; batch uses 1.0)
        per rank      : needed = ceil(remaining / per_chain_eff), capped by
                        the selected pool's free capacity;
                        remaining -= achieved
        achieved_tps  = SUM over ranks of n_replicas * per_chain_eff

    Online latency gate: a rank whose predicted p99 TTFT or TPOT exceeds
    target is EXCLUDED (n_replicas = 0) - latency is per-replica and replicas
    cannot fix it - and its share spills to the remaining ranks. A
    capacity-bound rank reports its GPU shortfall in marginal_value; unmet_tps
    reports any target the whole ladder could not cover.

    Args:
        ranks: rank dicts (RankSpec.from_dict form) with role, env, config.
            Heterogeneous ranks are allowed; order them by preference.
        job_features: the job's W features - type ("online"/"batch"),
            request_arrival_rate, output_len_tokens_avg, target_p99_ttft_ms,
            target_p99_tpot_ms, total_token_budget, deadline_hours,
            headroom_factor.
        target_tps: override; default from required_throughput_enumerator.
        utilization_target: override; default UTILIZATION_TARGET_ONLINE for
            online, 1.0 for batch.

    Returns:
        {"ranks": [deployable rank dicts, n_replicas >= 1], "regime",
         "target_tps", "achieved_tps" (summed), "unmet_tps", "meets_target",
         "per_rank": [...all ranks incl. dropped/excluded...],
         "marginal_value": {env_key: extra_gpus_to_meet_target}}.
    """
    _require("resource_map", "surrogate", "candidate_graph", "dro")

    job_features = _sanitize_agent_features(dict(job_features or {}))
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
    achieved_total = 0.0
    remaining = target
    remaining_by_pool: dict[tuple[str, str | None], int] = {}
    resources = (
        _CTX.resource_map.resources_summary()
        if hasattr(_CTX.resource_map, "resources_summary")
        else None
    )

    # Fill ranks in the order given, each covering the REMAINING target, so a
    # heterogeneous parallel ladder shares one target and achieved throughput
    # SUMS across ranks.
    for raw in ranks:
        rank = RankSpec.from_dict(raw)
        gpus_per_chain = rank.gpus_per_chain()
        gpu_type = env_gpu_type(rank.env)
        env_key = _env_key(rank.env)
        if resources is not None:
            info = resources.get(env_key)
            env_free = int(info.get("free", 0)) if info and info.get("gpu_type") == gpu_type else 0
        else:
            env_free = _CTX.resource_map.get_avail_capacity(rank.env, gpu_type) if gpu_type else 0
        allocation_error = None
        try:
            allocation = _rank_allocation_summary(rank, resources)
            capacity_per_replica = int(allocation["capacity_per_replica"])
        except Exception as exc:
            allocation_error = str(exc)
            allocation = {
                "allocation_kind": None,
                "instance_type": None,
                "gpus_per_unit": None,
                "price_per_unit_hour": None,
            }
            capacity_per_replica = max(1, gpus_per_chain)
        pool_key = (env_key, allocation["instance_type"])
        free = remaining_by_pool.setdefault(
            pool_key, int(allocation.get("free_capacity_gpus", env_free))
        )
        max_by_cap = free // capacity_per_replica if capacity_per_replica > 0 else 0

        payload = _rank_prediction_payload(rank, job_features)
        pred_error = None
        runnable, validity_reason = config_runnable(dict(rank.config), payload["job_features"])
        if not runnable:
            # Physically unrunnable (parallelism vs GPUs, model sharding): reject
            # in code with a reason and SKIP the surrogate - no wasted sim, no
            # crash. Replaces hand-checking divisibility in the prompt.
            pred_error = validity_reason
            y_hat = {}
        else:
            try:
                y_hat = _predict_outcome_core(payload["job_config"], payload["job_features"]).get(
                    "y_hat", {}
                )
            except Exception as exc:
                # Backstop for anything the pre-check couldn't evaluate (missing
                # catalog arch, other engine rejection): infeasible, don't crash.
                log.warning("size_ladder: surrogate rejected rank config (%s)", exc)
                pred_error = str(exc)
                y_hat = {}
        per_chain_raw = _y_value(y_hat, "throughput_token_per_sec")
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
        physical_violations: list[str] = []
        if allocation_error:
            physical_violations.append(allocation_error)
        elif (
            allocation["allocation_kind"] != "gpu" and gpus_per_chain > allocation["gpus_per_unit"]
        ):
            physical_violations.append(
                f"engine needs {gpus_per_chain} GPUs but {allocation['instance_type']} "
                f"has {allocation['gpus_per_unit']}"
            )
        if pred_error:
            physical_violations.append(f"config not runnable: {pred_error}")
        physical_ok = not physical_violations

        # An SLO-infeasible online config is not viable at ANY replica count
        # (p99 TPOT is per-replica; replicas cannot fix it): give it 0 and let
        # its share spill to the remaining ranks.
        excluded = (is_online and not slo_ok) or not physical_ok
        if excluded:
            needed = 0
            n_replicas = 0
            no_prediction = per_chain_eff <= 0
        elif per_chain_eff > 0:
            needed = max(0, math.ceil(remaining / per_chain_eff)) if remaining > 0 else 0
            n_replicas = min(needed, max_by_cap) if max_by_cap >= 1 else 0
            no_prediction = False
        else:
            needed = max_by_cap  # no throughput signal: use what capacity allows
            n_replicas = max_by_cap
            no_prediction = True

        rank.n_replicas = n_replicas
        remaining_by_pool[pool_key] = max(0, free - n_replicas * capacity_per_replica)
        # "ranks" is the DEPLOYABLE ladder: a 0-replica rank (excluded for SLO,
        # or capacity-starved) is dropped here and shows only in per_rank.
        if n_replicas >= 1:
            sized.append(rank.to_dict())

        contributed = n_replicas * per_chain_eff
        achieved_total += contributed
        remaining = max(0.0, remaining - contributed)

        capacity_bound = (not excluded) and needed > n_replicas
        if capacity_bound:
            marginal[env_key] = (
                marginal.get(env_key, 0) + (needed - n_replicas) * capacity_per_replica
            )

        per_rank.append(
            {
                "role": rank.role,
                "env": list(rank.env) if rank.env else None,
                "per_chain_tps_raw": per_chain_raw,
                "per_chain_tps_effective": per_chain_eff,
                "utilization_target": util,
                "gpus_per_chain": gpus_per_chain,
                "allocation_kind": allocation["allocation_kind"] if not allocation_error else None,
                "instance_type": allocation["instance_type"] if not allocation_error else None,
                "gpus_per_allocation_unit": allocation["gpus_per_unit"]
                if not allocation_error
                else None,
                "capacity_gpus_per_replica": capacity_per_replica,
                "free_capacity_gpus": free,
                "price_per_unit_hour": allocation["price_per_unit_hour"]
                if not allocation_error
                else None,
                "needed_replicas": needed,
                "max_replicas_by_capacity": max_by_cap,
                "n_replicas": n_replicas,
                "capacity_bound": capacity_bound,
                "slo_ok": slo_ok,
                "slo_violations": slo_violations,
                "physical_ok": physical_ok,
                "physical_violations": physical_violations,
                "excluded_slo_infeasible": excluded,
                "no_throughput_prediction": no_prediction,
            }
        )

    # Achieved throughput SUMS across parallel ranks; only ranks actually
    # serving (n_replicas >= 1) gate the latency SLO (excluded ranks serve none).
    achieved_tps = achieved_total
    served_slo_ok = all(r["slo_ok"] for r in per_rank if r["n_replicas"] >= 1)
    meets_target = achieved_tps >= target and served_slo_ok
    return {
        "ranks": sized,
        "regime": regime,
        "target_tps": target,
        "achieved_tps": achieved_tps,
        "unmet_tps": remaining,
        "meets_target": meets_target,
        "per_rank": per_rank,
        "marginal_value": marginal,
    }


# ----------------------------------------------------------------------
# Mechanism / confidence tools
# ----------------------------------------------------------------------


def get_edge_confidence(edge_or_list) -> Any:
    """Confidence record(s) for one edge id or a list of them.

    ConfidenceService owns the numeric state access; this tool builds the
    JSON-friendly record shape exposed to the planner.

    Args:
        edge_or_list: edge_id string or list of edge_id strings.

    Returns:
        One id -> the confidence record dict; a list -> {edge_id: record}.
    """
    _require("confidence_service")
    cs = _CTX.confidence_service

    def one(edge_id: str) -> dict[str, Any]:
        alpha, beta = cs.get_edge_alpha_beta(edge_id)
        return {
            "c": cs.get_edge_confidence(edge_id),
            "alpha": alpha,
            "beta": beta,
            "visit_count": cs.get_edge_visit_count(edge_id),
            "envs_seen": sorted(_env_key(e) for e in cs.get_edge_environment_seen(edge_id)),
            "last_touched_tick": cs.get_edge_last_touched(edge_id),
            "q_histogram": dict(cs.get_edge_q_histogram(edge_id)),
        }

    if isinstance(edge_or_list, list):
        return {eid: one(eid) for eid in edge_or_list}
    return one(str(edge_or_list))


def get_mechanism_confidence(m_id) -> Any:
    """Confidence record(s) for one mechanism id or a list of them.

    ConfidenceService owns the numeric state access; this tool builds the
    JSON-friendly record shape exposed to the planner.

    Args:
        m_id: mechanism_id string or list of mechanism_id strings.

    Returns:
        One id -> the confidence record dict; a list -> {mid: record}.
    """
    _require("confidence_service")
    cs = _CTX.confidence_service

    def one(mid: str) -> dict[str, Any]:
        alpha, beta = cs.get_mechanism_alpha_beta(mid)
        return {
            "c": cs.get_mechanism_confidence(mid),
            "alpha": alpha,
            "beta": beta,
            "visit_count": cs.get_mechanism_visit_count(mid),
            "envs_seen": sorted(_env_key(e) for e in cs.get_mechanism_environment_seen(mid)),
            "last_touched_tick": cs.get_mechanism_last_touched(mid),
            "q_histogram": dict(cs.get_mechanism_q_histogram(mid)),
        }

    if isinstance(m_id, list):
        return {mid: one(mid) for mid in m_id}
    return one(str(m_id))


def get_influencing_knobs(
    job_features: dict[str, Any],
    objective: str | None = None,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    """Reverse lookup: which X knobs drive an objective, by path confidence.

    The closed-world graph holds every X->V and V->Y edge. This walks
    BACKWARD from the objective Y, through each mediator V that feeds it,
    to the X knobs that feed those mediators, scoring each knob by the
    strongest causal path confidence c(X->V) * c(V->Y). It answers the
    planner's question "to move this objective, which knobs are worth
    tuning and how sure are we?" - the input side of optimize_config.

    Mechanisms applicable to the job's scope are attached per knob so the
    planner can cite a mechanism_id on the RankSpec it ends up tuning.
    Confidence comes straight from ConfidenceService (single owner); this
    tool only traverses and ranks.

    Args:
        job_features: Workload values used for mechanism annotation. Does not
            restrict which knobs are returned (the graph is closed-world).
        objective: a Y variable name to trace. None traces every Y node.
        top_k: max knobs to return, highest path confidence first.

    Returns:
        List of {"knob", "score", "paths": [{"v","y","c_xv","c_vy",
        "path_c"}...], "mechanisms": [mechanism_id...]} sorted by score.
    """
    _require("candidate_graph", "confidence_service")
    cg = _CTX.candidate_graph
    cs = _CTX.confidence_service

    if objective is not None:
        objectives = [objective]
    else:
        objectives = [n for n, node in cg.node_table.items() if node.node_type == "Y"]

    edge_to_mechs: dict[str, set] = {}
    if _CTX.mechanism_registry is not None and isinstance(job_features, dict):
        matches = _CTX.mechanism_registry.find_applicable(
            job_features,
            require_x_overlap=False,
        )
        for mechanism, _ in matches:
            for edge_id in mechanism.edge_ids:
                edge_to_mechs.setdefault(edge_id, set()).add(mechanism.mechanism_id)

    knobs: dict[str, dict[str, Any]] = {}
    for y in objectives:
        for vy in cg.get_edges_to(y):  # V -> Y edges into the objective
            c_vy = cs.get_edge_confidence(vy.edge_id)
            for xv in cg.get_edges_to(vy.src):  # X -> V edges into that mediator
                c_xv = cs.get_edge_confidence(xv.edge_id)
                path_c = c_xv * c_vy
                rec = knobs.setdefault(
                    xv.src, {"knob": xv.src, "score": 0.0, "paths": [], "mechanisms": set()}
                )
                rec["score"] = max(rec["score"], path_c)
                rec["paths"].append(
                    {"v": vy.src, "y": y, "c_xv": c_xv, "c_vy": c_vy, "path_c": path_c}
                )
                rec["mechanisms"].update(edge_to_mechs.get(xv.edge_id, ()))

    ranked = sorted(
        (record for record in knobs.values() if record["knob"] in AGENT_TUNABLE_X),
        key=lambda record: record["score"],
        reverse=True,
    )[: int(top_k)]
    for rec in ranked:
        rec["paths"].sort(key=lambda p: p["path_c"], reverse=True)
        rec["paths"] = rec["paths"][:5]
        rec["mechanisms"] = sorted(rec["mechanisms"])
    return ranked


def _mechanism_briefs(matches) -> list[dict[str, Any]]:
    return [
        {
            "mechanism_id": mechanism.mechanism_id,
            "name": mechanism.name,
            "edge_ids": list(mechanism.edge_ids),
            "scope": dict(mechanism.scope),
            "narrative": mechanism.narrative,
            "c": _CTX.confidence_service.get_mechanism_confidence(mechanism.mechanism_id),
            "visit_count": _CTX.confidence_service.get_mechanism_visit_count(
                mechanism.mechanism_id
            ),
            "match_quality": match["quality"],
            "matched_x": match["matched_x"],
            "missing_x": match["missing_x"],
            "condition_results": match["condition_results"],
        }
        for mechanism, match in matches
    ]


def get_scope(job_features: dict[str, Any]) -> list[dict[str, Any]]:
    """Return existing mechanism candidates for a pre-rank job context.

    Args:
        job_features: Workload and model values known before a rank exists.

    Returns:
        Exact and partial mechanism briefs with confidence and match details.
    """
    _require("mechanism_registry", "confidence_service")
    context = dict(job_features or {})
    matches = _CTX.mechanism_registry.find_applicable(context, require_x_overlap=False)
    return _mechanism_briefs(matches)


def get_applicable_mechanisms(
    rank: dict[str, Any],
    job_features: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return exact and partial mechanisms for one concrete candidate rank."""
    _require("mechanism_registry", "confidence_service")
    typed_rank = RankSpec.from_dict(rank)
    context = _rank_mechanism_context(typed_rank, job_features)
    return _mechanism_briefs(_CTX.mechanism_registry.find_applicable(context))


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
                if (wanted_gpu is None or env_gpu_type(r.env_label) == wanted_gpu)
                and (
                    wanted_type is None
                    or str(r.X.get("type") or r.X.get("workload_type")).lower() == wanted_type
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

    The proposer does NOT set the mechanism's confidence. On admission the
    mechanism is seeded NEUTRAL (Beta(1,1), c=0.5) by ConfidenceService -
    an unproven theory starts agnostic and earns confidence only from
    evidence. An offline seeding pass may later assign it a deliberate bin
    and promote it into the seed table.

    Args:
        edges: edge_id strings, all present in CandidateGraph.
        applicable_to: Scope dict, e.g. {"x": [...], "v": [...],
            "workload_type": "online"}.
        llm_blurb: One-paragraph narrative for the mechanism.

    Returns:
        {"ok": bool, "mechanism_id": str | None, "seed_confidence": float,
         "violations": list}.
    """
    _require("mechanism_registry", "candidate_graph", "confidence_service")
    from src.core.models import Mechanism

    candidate = Mechanism(
        edge_ids=list(edges),
        scope=dict(applicable_to),
        narrative=str(llm_blurb),
    )
    check = val_new_mechanisms(candidate)
    if not check["ok"]:
        return {
            "ok": False,
            "mechanism_id": None,
            "seed_confidence": None,
            "violations": check["violations"],
        }
    mid = _CTX.mechanism_registry.add_mechanism(candidate)
    # Confidence is set by the single writer, not the proposer: neutral prior.
    c0 = _CTX.confidence_service.seed_new_mechanism_confidence(mid)
    return {"ok": True, "mechanism_id": mid, "seed_confidence": c0, "violations": []}


def val_new_mechanisms(m_new) -> dict[str, Any]:
    """Run pre-admission validation on a mechanism proposal.

    Uses the canonical proposal validator for edges, topology, duplicates,
    scope variables, qualifiers, and conditions.

    Args:
        m_new: Mechanism object or dict with edge_ids, scope, narrative.

    Returns:
        {"ok": bool, "violations": List[str]}.
    """
    _require("mechanism_registry", "candidate_graph")
    from src.validation.validator import Validator

    ok, violations = Validator(
        candidate_graph=_CTX.candidate_graph,
        mechanism_registry=_CTX.mechanism_registry,
    ).val_mechanism_proposal(m_new)
    return {"ok": ok, "violations": violations}


# ----------------------------------------------------------------------
# Prediction / scoring tools
# ----------------------------------------------------------------------


def _similar_rows(
    job_features: dict[str, Any], window: int = CALIBRATION_WINDOW, top_k: int = 80
) -> list:
    """Evidence rows from deployments similar to this job (for calibration).

    Prefers the store's retrieval helper when present;
    otherwise scans the recent window and filters by gpu_type / workload
    type. Returns raw EvidenceRow objects (so callers can read residuals).
    """
    store = _CTX.evidence_store
    if store is None:
        return []
    if hasattr(store, "retrieve_similar_rows"):
        try:
            return list(store.retrieve_similar_rows(job_features, top_k=top_k))
        except Exception:
            log.exception("retrieve_similar_rows failed; falling back to scan")
    if not (hasattr(store, "get_rows_in_window") and hasattr(store, "current_tick")):
        return []
    current = store.current_tick()
    rows = store.get_rows_in_window((max(0, current - window), current))
    gpu = job_features.get("gpu_type")
    typ = job_features.get("type")
    if gpu or typ:
        rows = [
            r
            for r in rows
            if (gpu is None or env_gpu_type(r.env_label) == gpu)
            and (typ is None or r.X.get("type") == typ or r.X.get("workload_type") == typ)
        ]
    return rows[-top_k:]


def _residual_offsets(job_features: dict[str, Any], objectives) -> dict[str, float]:
    """Mean observed residual (observed - predicted) per objective over
    similar deployments. Objectives with < CALIBRATION_MIN_SAMPLES are
    omitted (too noisy to correct)."""
    rows = _similar_rows(job_features)
    offsets: dict[str, float] = {}
    for obj in objectives:
        samples = []
        for r in rows:
            arr = getattr(r, "residuals_per_y", {}).get(obj)
            if arr is not None and len(arr) > 0:
                samples.append(float(np.mean(np.asarray(arr, dtype=float))))
        if len(samples) >= CALIBRATION_MIN_SAMPLES:
            offsets[obj] = float(np.mean(samples))
    return offsets


def _calibrate_y_hat(y_hat: dict[str, float], job_features: dict[str, Any]):
    """Debias the surrogate's y_hat with empirical residual offsets.

    calibrated = surrogate + mean(observed - predicted over similar rows),
    clamped >= 0 for physically non-negative objectives. Returns
    (calibrated_y_hat, offsets_applied).
    """
    offsets = _residual_offsets(job_features, list(y_hat.keys()))
    calibrated: dict[str, float] = {}
    for obj, val in y_hat.items():
        if val is None:
            calibrated[obj] = val
            continue
        corrected = float(val) + offsets.get(obj, 0.0)
        if obj in _NONNEGATIVE_Y:
            corrected = max(0.0, corrected)
        calibrated[obj] = corrected
    return calibrated, offsets


def predict_outcome(
    config: dict[str, Any],
    mechanism: dict[str, Any] | None = None,
    env: list[str] | tuple[str, ...] | None = None,  # TODO - Added direct env to this tool for v0
    calibrate: bool = True,
) -> dict[str, Any]:
    """Run the surrogate, debias it with evidence, attach the DRO band.

    The mechanistic surrogate (Calculon/DynoSim) is kept pure; the
    empirical residual correction is applied HERE as a thin calibration
    layer so every scoring path (compute_sigma, size_ladder,
    optimize_config) optimizes against reality-corrected numbers. As the
    evidence database grows the surrogate's systematic error is learned
    away. Cold start (no similar residuals) returns the raw surrogate.

    Args:
        config: X variables for the candidate. May embed job_config and
            job_features sub-dicts; otherwise the whole dict is job_config.
        mechanism: Optional mechanism context, informational only.
        env: Optional canonical [market, cloud, region, zone, gpu_type].
        calibrate: Apply the residual correction (default True).

    Returns:
        {"y_hat": calibrated dict, "y_hat_raw": surrogate dict,
         "calibration_offsets": dict, "v_hat": dict, "dro_band": dict}.
    """
    if not isinstance(config, dict):
        raise ValueError(
            "predict_outcome scores ONE config dict, not a ladder/list. Pass "
            "{'job_config': {...}, 'job_features': {...}} (or a flat X config). "
            "To score a whole ladder: assemble ranks into an action, build a "
            "plan, then call compute_sigma(plan)."
        )
    job_features = _sanitize_agent_features(dict(config.get("job_features", {})))
    job_config = _sanitize_agent_config(dict(config.get("job_config", config)))
    if env and len(env) == 5 and not job_config.get("gpu_type"):
        job_config["gpu_type"] = env[4]
    return _predict_outcome_core(job_config, job_features, calibrate=calibrate)


def _predict_outcome_core(
    job_config: dict[str, Any],
    job_features: dict[str, Any],
    calibrate: bool = True,
) -> dict[str, Any]:
    """Run prediction with already trusted or sanitized inputs."""
    _require("candidate_graph", "dro", "surrogate")
    cache_key = json.dumps(
        {"job_config": job_config, "job_features": job_features}, sort_keys=True, default=str
    )
    if cache_key in _PREDICT_RAW_CACHE:
        y_hat_raw, v_hat = _PREDICT_RAW_CACHE[cache_key]
    else:
        global _surrogate_calls
        _surrogate_calls += 1
        if _surrogate_calls > SURROGATE_CALL_BUDGET:
            raise SurrogateBudgetExceeded(
                f"surrogate-call budget {SURROGATE_CALL_BUDGET} reached this tick; "
                "narrow to your best few candidate configs and reuse scored results."
            )
        result = _CTX.surrogate.compose_prediction(
            job_config=job_config,
            job_features=job_features,
            candidate_graph=_CTX.candidate_graph,
            method=("AIC_DynoSim",),
        )
        if isinstance(result, tuple) and len(result) == 2:
            y_hat_raw, v_hat = result
        else:
            y_hat_raw = getattr(result, "y_hat", {}) or {}
            v_hat = getattr(result, "v_hat", {}) or {}
        _PREDICT_RAW_CACHE[cache_key] = (dict(y_hat_raw or {}), dict(v_hat or {}))
    y_hat_raw = y_hat_raw or {}

    if calibrate and y_hat_raw:
        y_hat, offsets = _calibrate_y_hat(y_hat_raw, job_features)
    else:
        y_hat, offsets = dict(y_hat_raw), {}

    dro_band = _CTX.dro.compute_dro_band(y_hat or {})
    return {
        "y_hat": y_hat or {},
        "y_hat_raw": y_hat_raw,
        "calibration_offsets": offsets,
        "v_hat": v_hat or {},
        "dro_band": dro_band,
    }


def stamp_plan_predictions(plan, cluster_snapshot=None):
    """Attach raw per-rank predictions to the plan that will be deployed."""
    typed = _as_plan(plan)
    snapshot = cluster_snapshot if cluster_snapshot is not None else _snapshot()
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        job_features = _job_features_for(snapshot, action.job_id)
        for rank in action.ladder:
            payload = _rank_prediction_payload(rank, job_features)
            pred = _predict_outcome_core(
                payload["job_config"], payload["job_features"], calibrate=False
            )
            rank.predicted_y = dict(pred.get("y_hat_raw") or pred.get("y_hat") or {})
            rank.predicted_v = dict(pred.get("v_hat") or {})
    return typed


def get_z_star(job_features: dict[str, Any] | None = None) -> dict[str, float]:
    """Current ideal-point reference z_star_t for Tchebycheff scoring.

    z_star_t is the slow loop's running best-achievable value per
    objective, maintained in the slow loop from the performance database
    (the kNN/quantile-of-observed-bests reference, updated each tick). It
    is what compute_tchebycheff measures distance FROM, so "good" for an
    objective means "close to z_star_t". Exposed read-only so the planner
    can see the current target per objective before it scores configs.

    The slow loop is the single owner of z_star_t; this tool never
    recomputes it (residual calibration lives in predict_outcome, the
    reference point lives in the slow loop - they are separate concerns).

    Args:
        job_features: accepted for a future per-scope reference; today the
            cluster-level z_star_t is returned regardless.

    Returns:
        Dict objective -> reference value.
    """
    _require("slow_loop")
    if job_features is not None and hasattr(_CTX.slow_loop, "get_sss_z_star_t_for_scope"):
        return dict(_CTX.slow_loop.get_sss_z_star_t_for_scope(job_features))
    return dict(_CTX.slow_loop.get_sss_z_star_t())


def _scoreable_y_hat(y_hat, weights, reference, ranges) -> dict:
    return {
        k: v
        for k, v in (y_hat or {}).items()
        if v is not None and k in weights and k in reference and k in ranges
    }


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
    reference = _seeded_z_star(z_star if z_star is not None else _CTX.slow_loop.get_sss_z_star_t())
    ranges = _seeded_ranges(_CTX.slow_loop.typical_ranges)
    y_score = _scoreable_y_hat(y_hat, weights, reference, ranges)
    if not y_score:
        return 0.0
    return float(
        _CTX.tchebycheff_module.compute_tchebycheff(
            y_hat=y_score,
            w_t=weights,
            z_star_t=reference,
            normalization_range=ranges,
        )
    )


def optimize_config(
    base_config: dict[str, Any],
    candidates: dict[str, list],
    job_features: dict[str, Any] | None = None,
    env: list[str] | tuple[str, ...] | None = None,  # TODO - Added direct env to this tool for v0
    objective_weights: dict[str, float] | None = None,
    max_passes: int = 2,
) -> dict[str, Any]:
    """LLM-steered coordinate descent over candidate knob values.

    An OPTIONAL inner optimizer. The planner reasons its way to a config
    and a small set of values worth trying per knob (from
    get_influencing_knobs / enumerate_ladder); this does the mechanical
    local refinement the planner would otherwise do by hand - try each
    candidate value for one knob, keep whichever maximizes the calibrated
    Tchebycheff J, sweep again until a pass makes no improvement or
    max_passes is hit. It does NOT replace the planner's search or pick the
    knob domains; it only polishes within the box the planner hands it, so
    the LLM's free reasoning stays in charge of WHAT to explore.

    Scoring uses predict_outcome (calibrated against the evidence database)
    and the slow loop's current w_t / z_star_t, so the local optimum chases
    reality-corrected outcomes rather than raw surrogate numbers.

    Args:
        base_config: starting config. May embed job_config / job_features
            sub-dicts, or be a flat X config; the flat dict is the config.
        candidates: {knob_name: [value, ...]}. Only these knobs vary;
            everything else in base_config stays fixed.
        job_features: W features for calibration and weighting. Defaults to
            base_config["job_features"] when present.
        env: Optional canonical [market, cloud, region, zone, gpu_type].
        objective_weights: override w_t; defaults to the slow loop's w_t.
        max_passes: coordinate-descent sweeps over the knob set.

    Returns:
        {"config": best config, "j": best J, "y_hat": calibrated
         prediction, "improved": bool, "n_evaluated": int,
         "trace": [{"knob","chosen","j"}...]}.
    """
    _require("surrogate", "tchebycheff_module", "slow_loop")
    features = _sanitize_agent_features(
        dict(job_features if job_features is not None else base_config.get("job_features", {}))
    )
    weights = objective_weights if objective_weights is not None else _CTX.slow_loop.get_sss_wt()
    reference = _seeded_z_star(_CTX.slow_loop.get_sss_z_star_t())
    core = _sanitize_agent_config(dict(base_config.get("job_config", base_config)))
    core.pop("job_features", None)
    candidates = {key: values for key, values in candidates.items() if key in AGENT_TUNABLE_X}

    def _score(cfg: dict[str, Any]):
        try:
            pred = predict_outcome({"job_config": cfg, "job_features": features}, env=env)
        except Exception as exc:
            # Skip a config the surrogate rejects instead of crashing the sweep.
            log.warning("optimize_config: surrogate rejected config, skipping (%s)", exc)
            return float("-inf"), None
        j = compute_tchebycheff(pred["y_hat"], weights, reference)
        return j, pred

    best_cfg = dict(core)
    best_j, best_pred = _score(best_cfg)
    n_eval = 1
    trace: list[dict[str, Any]] = []

    for _ in range(max(1, int(max_passes))):
        improved_pass = False
        for knob, values in candidates.items():
            local_best = None
            for value in values:
                if best_cfg.get(knob) == value:
                    continue
                trial = dict(best_cfg)
                trial[knob] = value
                j, pred = _score(trial)
                n_eval += 1
                if j > best_j:
                    best_j, best_pred, best_cfg = j, pred, trial
                    local_best = value
                    improved_pass = True
            if local_best is not None:
                trace.append({"knob": knob, "chosen": local_best, "j": best_j})
        if not improved_pass:
            break

    return {
        "config": best_cfg,
        "j": best_j,
        "y_hat": best_pred["y_hat"] if best_pred else {},
        "improved": bool(trace),
        "n_evaluated": n_eval,
        "trace": trace,
    }


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


def _switch_pricing_map() -> dict:
    resource_map = getattr(_CTX, "resource_map", None)
    if resource_map is not None and hasattr(resource_map, "switch_pricing_map"):
        return resource_map.switch_pricing_map()
    return {}


def compute_switching_cost(
    ladder_prev: Any,
    ladder_new: Any,
    pred_y_new: dict[str, float] | None = None,
    slo_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute the 4-component switch cost between two ladders.

    Args:
        ladder_prev: Current ladder (chain entry dicts or objects).
        ladder_new: Proposed ladder.
        pred_y_new: Optional proposed ladder prediction for DRO risk.
        slo_thresholds: Optional per-objective SLO thresholds for DRO risk.

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
        pricing_map=_switch_pricing_map(),
        slo_thresholds=slo_thresholds,
        pred_y_new=pred_y_new,
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


# NOTE: compute_cusum / compute_icp / c_d_classification were intentionally
# removed from the agent tool surface. They are evidence-time VALIDATION
# primitives that the FSM runs in S2 via the cusum / icp / quadrant_validator
# modules directly. The planning agent consumes their RESULTS (via
# get_edge_confidence / get_mechanism_confidence / get_recent_q_histogram) and
# cannot meaningfully run them on a hypothetical config that has no observed
# trajectory yet, so exposing them here was dead weight.


# ----------------------------------------------------------------------
# Plan-level tools
# ----------------------------------------------------------------------


def _target_reference(job_features: dict[str, Any]) -> dict[str, float]:
    """Per-job scoring reference = the job's OWN SLO / throughput TARGET, not the
    absolute best-achievable z* ideal. Meeting the target is "good enough" (see
    _clamp_to_reference), so J stops rewarding over-service. Empty when the job
    declares no targets (caller then falls back to z*)."""
    ref: dict[str, float] = {}
    ttft = _feature_value(job_features, "target_p99_ttft_ms", "target_p99_TTFT_ms")
    tpot = _feature_value(job_features, "target_p99_tpot_ms", "target_p99_TPOT_ms")
    if ttft:
        ref["p99_ttft_ms"] = float(ttft)
    if tpot:
        ref["p99_tpot_ms"] = float(tpot)
    try:
        req_tps = float(required_throughput_enumerator(job_features))
        if req_tps > 0:
            ref[_THROUGHPUT_OBJ] = req_tps
    except Exception:
        pass
    return ref


def _clamp_to_reference(y_hat: dict[str, Any], reference: dict[str, float]) -> dict[str, float]:
    """One-sided distance: a value that MEETS its target (latency <= target, or
    throughput >= target) is snapped TO the target so its Tchebycheff gap is 0 -
    exceeding a target earns no extra credit. Missing it keeps the real value, so
    only the shortfall is penalized."""
    out: dict[str, float] = {}
    for obj, value in (y_hat or {}).items():
        if value is None:
            continue
        target = reference.get(obj)
        v = float(value)
        if target is not None and (
            (obj in _LATENCY_OBJS and v < target) or (obj == _THROUGHPUT_OBJ and v > target)
        ):
            v = target
        out[obj] = v
    return out


def compute_sigma(plan) -> dict[str, Any]:
    """Score a plan: per-job sigma and the cluster aggregate.

    sigma = J + beta_t * eig - gamma * Pr_DRO - lambda_swit * switch_cost,
    over every ladder-bearing action (place/swap). The
    scoring inputs are DERIVED, not trusted from the LLM: prev_ladder and
    slo_thresholds come from the snapshot, y_hat from the action's
    advisory predicted_y or a fresh surrogate call. Non-ladder actions
    (keep/defer/terminate/diagnose) deploy nothing new and score 0.

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
    # z_star is fetched PER JOB in the loop; z*/ranges are seeded against domain
    # priors so an unseeded slow loop (z*=0, range=1.0) cannot collapse J to ~-50.
    ranges = _seeded_ranges(_CTX.slow_loop.typical_ranges)
    beta = _CTX.slow_loop.get_sss_eig_incentive_t()
    lam = _CTX.slow_loop.get_sss_lambda_switch()
    eps_dro = _CTX.slow_loop.get_sss_radius_dro()
    pricing_map = _switch_pricing_map()

    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        job_id = action.job_id
        ladder_dicts = _ranks_as_dicts(action)
        job_features = _job_features_for(snapshot, job_id)
        y_hat = _compose_job_y_hat(action, job_features)
        if not y_hat:
            continue
        # Score against the job's OWN SLO/throughput TARGET, not the absolute z*
        # ideal: meeting the target is "good enough" and earns 0 gap (no reward for
        # over-service), so the optimizer stops lavishing scarce GPUs on a job that
        # already meets its SLO and starving the one that needs them. Fall back to
        # z* only when a job declares no targets.
        target_ref = _target_reference(job_features)
        if target_ref:
            reference, y_for_score = target_ref, _clamp_to_reference(y_hat, target_ref)
            # Normalize by the TARGET itself, so a gap is a FRACTION of the
            # requirement (|y-target|/target), not |y-target|/typical_range. The
            # seeded range (~1000 tps) dwarfs a target (~150), which flattened a
            # 73%-of-target miss into a ~0.1 gap - the solver then could not tell
            # the 72B needs H100 far more than a 7B does, and mis-gave the H100.
            norm = {obj: abs(t) for obj, t in target_ref.items() if t}
        else:
            reference, y_for_score, norm = _seeded_z_star(get_z_star(job_features)), y_hat, ranges
        y_score = _scoreable_y_hat(y_for_score, w_t, reference, norm)
        if not y_score:
            continue
        slo_thresholds = _slo_thresholds_for(snapshot, job_id)

        J = float(
            _CTX.tchebycheff_module.compute_tchebycheff(
                y_hat=y_score,
                w_t=w_t,
                z_star_t=reference,
                normalization_range=norm,
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
            pricing_map=pricing_map,
            slo_thresholds=slo_thresholds,
            pred_y_new=y_hat,
        )
        pr_slo = float(
            _CTX.dro.dro_chance_constraint(
                pred_y=y_hat,
                slo_thresholds=slo_thresholds,
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

    # Serve-value: leaving a waiting job unserved is NOT free. Charge an
    # opportunity cost per pending job the plan does not place, over snapshot
    # demand (not just explicit defer actions, so omitting a job cannot dodge
    # it), SCALED by the job's priority score.
    #
    # The scale matters: placing job i beats deferring it iff sigma_i + penalty_i
    # > 0, i.e. penalty_i > |sigma_i|. A config's Tchebycheff distance |sigma| is
    # empirically ~10-30 even for an SLO-crushing frame (J measures distance to
    # the IDEAL z*, not to the SLO), so a flat penalty of 1.0 could NEVER offset
    # it and defer always won. The bug: priority was read from the raw pending-job
    # dict (`priority_class` / `user_priority`), which carries no COMPOSED score,
    # so it fell back to 1.0. Use the real priority_score from get_priority() (the
    # same table the budget book uses; scale ~10-50). Now a serveable job is
    # placed, while a job whose only feasible frame is far from ideal (e.g. a 72B
    # stuck on L40S, |sigma|~33) correctly stays deferred until a better frame
    # (e.g. H100) is available. Raise UNSERVED_PENALTY above 1.0 to bias harder
    # toward serving.
    served = {a.job_id for a in typed.actions if a.type in LADDER_ACTIONS and a.ladder}
    priority_by_job = {
        p.get("job_id"): float(p.get("priority_score", 1.0) or 1.0) for p in get_priority()
    }
    unserved_penalty = 0.0
    for job in get_pending_jobs():
        jid = job.get("job_id", job.get("id"))
        if jid and jid not in served:
            unserved_penalty += UNSERVED_PENALTY * max(1.0, priority_by_job.get(jid, 1.0))
    aggregate -= unserved_penalty

    return {
        "per_job": per_job,
        "aggregate_sigma": aggregate,
        "swap_count": swap_counter(typed),
        "unserved_penalty": unserved_penalty,
    }


def _cap_key_str(key: tuple) -> str:
    """Readable capacity key: ('gpu', env) -> 'gpu:env'; ('pool', env, it) -> 'pool:env:it'."""
    return ":".join(str(part) for part in key)


def _ladder_capacity_cost(
    ladder: list[Any], instance_specs: dict[str, dict[str, dict[str, Any]]]
) -> dict[tuple, int]:
    """Resource cost of a ladder as {capacity_key: amount}, in BOTH dimensions the
    validator enforces:
      ('gpu', env_key)            -> GPUs used = sum(gpu_count * n_replicas)
      ('pool', env_key, instance) -> whole INSTANCES used
                                     = sum(n_replicas * ceil(gpu_count / gpus_per_instance))
    Per-pool instances is the constraint the old env-GPU-only check missed: eight
    1-GPU g6e.xlarge replicas need 8 INSTANCES even though they are only 8 of the
    L40S env's 16 GPUs, and only 4 g6e.xlarge may be free (validator C5) - so the
    solver picked an infeasible set and the planner deferred everything. Pool cost
    is added only when instance_specs knows the instance's gpus_per_instance;
    otherwise the coarse env-GPU dimension still bounds it."""
    cost: dict[tuple, int] = {}
    for rank in ladder or []:
        if not isinstance(rank, dict):
            continue
        env = rank.get("env")
        if env is None:
            continue
        env_key = _env_key(env)
        cfg = rank.get("config") or {}
        try:
            gpus = max(0, int(cfg.get("gpu_count", 0) or 0))
            reps = max(0, int(rank.get("n_replicas", 1) or 1))
        except (TypeError, ValueError):
            continue
        cost[("gpu", env_key)] = cost.get(("gpu", env_key), 0) + gpus * reps
        instance_type = cfg.get("instance_type")
        spec = (
            (instance_specs.get(env_key) or {}).get(str(instance_type)) if instance_type else None
        )
        gpi = int(spec.get("gpus_per_instance", 0) or 0) if spec else 0
        if gpi > 0:
            per_replica = max(1, -(-gpus // gpi))  # ceil(gpus / gpi) instances per replica
            key = ("pool", env_key, str(instance_type))
            cost[key] = cost.get(key, 0) + reps * per_replica
    return cost


def _largest_pow2_divisor_leq(heads: int | None, cap: int) -> int:
    """Largest power of 2 that divides `heads` and is <= cap; 1 if heads unknown.
    Picks a tp that both shards the model (divides attention heads) and fits the
    instance's GPU count."""
    if not heads or int(heads) <= 0 or cap < 1:
        return 1
    tp, power = 1, 2
    while power <= cap and int(heads) % power == 0:
        tp, power = power, power * 2
    return tp


def _normalize_candidate_rank(raw: Any) -> dict[str, Any] | None:
    """Clean any proposed rank to the canonical shape, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    env = raw.get("env")
    cfg = raw.get("config")
    if not (isinstance(env, (list, tuple)) and len(env) == 5 and isinstance(cfg, dict)):
        return None
    keep = (
        "instance_type",
        "gpu_count",
        "tp",
        "pp",
        "sp",
        "ep",
        "cp",
        "num_nodes_per_chain",
        "interconnect_type",
    )
    config = {k: cfg[k] for k in keep if k in cfg and cfg[k] is not None}
    if not config.get("instance_type"):
        return None
    for knob, default in (
        ("gpu_count", 1),
        ("tp", 1),
        ("pp", 1),
        ("sp", 1),
        ("ep", 1),
        ("cp", 1),
        ("num_nodes_per_chain", 1),
    ):
        config.setdefault(knob, default)
    try:
        n_replicas = max(1, int(raw.get("n_replicas", 1) or 1))
    except (TypeError, ValueError):
        n_replicas = 1
    return {"role": "aggregate", "env": list(env), "config": config, "n_replicas": n_replicas}


def _rank_shape_key(rank: dict[str, Any]) -> tuple:
    cfg = rank.get("config") or {}
    return (
        tuple(rank.get("env") or []),
        cfg.get("instance_type"),
        cfg.get("tp"),
        cfg.get("pp"),
        cfg.get("gpu_count"),
        cfg.get("num_nodes_per_chain"),
    )


def _score_one_frame(
    jid: str, user_id: Any, slice_id: Any, rank: dict[str, Any], features: dict[str, Any]
) -> dict[str, Any]:
    """The proven per-frame pipeline: runnable -> mechanism -> size_ladder ->
    feasibility -> per-job sigma. Returns {candidate|None, meets_target, diag}."""
    env = rank.get("env")
    cfg = dict(rank.get("config") or {})
    gpu_type = env_gpu_type(env) if env else None
    diag: dict[str, Any] = {
        "env": gpu_type,
        "instance_type": cfg.get("instance_type"),
        "tp": cfg.get("tp"),
        "status": None,
        "reason": None,
        "meets_target": False,
        "achieved_tps": None,
        "target_tps": None,
        "sigma": None,
    }
    runnable, reason = config_runnable(cfg, features)
    if not runnable:
        diag.update(status="unrunnable", reason=reason)
        return {"candidate": None, "meets_target": False, "diag": diag}
    mid = None
    try:
        apps = get_applicable_mechanisms(rank, features)
        if isinstance(apps, dict):
            mid = apps.get("exact") or apps.get("mechanism_id")
            if not mid:
                vals = apps.get("mechanisms") or apps.get("applicable") or []
                if vals:
                    mid = vals[0] if isinstance(vals[0], str) else vals[0].get("mechanism_id")
        elif isinstance(apps, (list, tuple)) and apps:
            mid = apps[0] if isinstance(apps[0], str) else apps[0].get("mechanism_id")
    except Exception:
        mid = None
    if not mid:
        diag.update(status="no_mechanism", reason="no applicable mechanism")
        return {"candidate": None, "meets_target": False, "diag": diag}
    scored_rank = dict(rank)
    scored_rank["mechanism_id"] = mid
    try:
        sized = size_ladder([scored_rank], features)
    except Exception as exc:
        diag.update(status="size_error", reason=f"size_ladder failed: {exc}")
        return {"candidate": None, "meets_target": False, "diag": diag}
    ranks = sized.get("ranks") or []
    meets = bool(sized.get("meets_target"))
    diag.update(
        achieved_tps=sized.get("achieved_tps"),
        target_tps=sized.get("target_tps"),
        meets_target=meets,
    )
    if not ranks:
        diag.update(
            status="no_fit",
            reason=f"does not fit/meet SLO (achieved {sized.get('achieved_tps')} of "
            f"{sized.get('target_tps')} tps)",
        )
        return {"candidate": None, "meets_target": False, "diag": diag}
    act = {
        "job_id": jid,
        "type": "place",
        "user_id": user_id,
        "ladder": ranks,
        "target_tps": sized.get("target_tps"),
        "mechanism_id": mid,
        "budget_ref": slice_id,
        "rationale": f"Deterministic {gpu_type} candidate "
        f"({'full-service' if meets else 'under-target'}).",
    }
    one = {"tick_rationale": "candidate scoring", "actions": [act]}
    try:
        feas = check_feasibility(one)
        if not feas.get("feasible"):
            diag.update(status="infeasible", reason="; ".join(feas.get("violations", []))[:200])
            return {"candidate": None, "meets_target": meets, "diag": diag}
        act["sigma"] = compute_sigma(one)["per_job"][jid]["sigma"]
    except Exception as exc:
        diag.update(status="score_error", reason=f"scoring failed: {exc}")
        return {"candidate": None, "meets_target": meets, "diag": diag}
    diag.update(status="ok", sigma=act["sigma"])
    return {"candidate": act, "meets_target": meets, "diag": diag}


def build_scored_candidates(
    budget_book: dict[str, Any] | None = None,
    specialist_results: Any = None,
) -> dict[str, Any]:
    """Deterministic candidate pipeline for all waiting jobs: normalize specialist
    ladders (HINTS), generate a frame on each OTHER viable GPU type they skipped
    (largest free instance of that type, tp = a head-dividing power of 2 that fits,
    from instance_catalog + the model's head count), size each, and score per-job
    sigma via the proven chain (config_runnable -> get_applicable_mechanisms ->
    size_ladder -> check_feasibility -> compute_sigma).

    EVERY physically-runnable, feasible frame is returned as a candidate - INCLUDING
    under-target ones - because placing a job beats deferring it; the joint solver
    decides serve-vs-defer from the scored sigma, not a hard meets_target gate here.
    A job appears in `exhausted` only when it has NO runnable, feasible frame at all.

    Returns {"candidates": [...for jointly_select_placements],
             "exhausted": {job_id: reason},
             "diagnostics": {job_id: [per-frame diag incl. meets_target/achieved_tps]}}.
    """
    _require("resource_map", "surrogate")
    snapshot = _snapshot()
    specs = instance_catalog()
    gpu_type_env: dict[str, list[str]] = {}
    for env_key, info in get_resource_map().items():
        try:
            if int(info.get("free", 0) or 0) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        gpu_type = info.get("gpu_type") or str(env_key).split("|")[-1]
        gpu_type_env.setdefault(str(gpu_type), str(env_key).split("|"))

    spec_by_job: dict[str, dict[str, Any]] = {}
    if isinstance(specialist_results, dict):
        if specialist_results.get("job_id"):
            spec_by_job[specialist_results["job_id"]] = specialist_results
        else:
            spec_by_job = {k: v for k, v in specialist_results.items() if isinstance(v, dict)}
    elif isinstance(specialist_results, (list, tuple)):
        for result in specialist_results:
            if isinstance(result, dict) and result.get("job_id"):
                spec_by_job[result["job_id"]] = result
    budgets = (
        ((budget_book or {}).get("job_budgets") or {}) if isinstance(budget_book, dict) else {}
    )

    # Phase 1 (cheap, no surrogate): build the candidate FRAME list per job -
    # specialist ladders (hints) plus one generated frame per viable GPU type the
    # specialist skipped (largest free instance of that type; big models need it,
    # small models keep their cheaper specialist frame and the solver picks by sigma).
    frames_by_job: dict[str, list[dict[str, Any]]] = {}
    ctx_by_job: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for job in get_pending_jobs():
        jid = job.get("job_id", job.get("id"))
        if not jid:
            continue
        features = _job_features_for(snapshot, jid) or dict(job.get("job_features") or {})
        model_id = features.get("model_id") or job.get("model_id")
        user_id = job.get("user_id") or features.get("user_id")
        slice_id = (budgets.get(jid) or {}).get("slice_id", jid)
        heads = _model_num_heads({"model_id": model_id}, features)

        frames: list[dict[str, Any]] = []
        seen: set = set()
        for raw in (spec_by_job.get(jid) or {}).get("ladder") or []:
            rank = _normalize_candidate_rank(raw)
            if rank is not None and _rank_shape_key(rank) not in seen:
                seen.add(_rank_shape_key(rank))
                frames.append(rank)
        covered = {env_gpu_type(r["env"]) for r in frames if r.get("env")}
        for gpu_type, env in gpu_type_env.items():
            if gpu_type in covered:
                continue
            best_instance = None
            for instance_type, spec in (specs.get(_env_key(env)) or {}).items():
                gpi = int(spec.get("gpus_per_instance", 0) or 0)
                if gpi <= 0 or int(spec.get("free_instances", 0) or 0) <= 0:
                    continue
                if best_instance is None or gpi > best_instance[1]:
                    best_instance = (instance_type, gpi)
            if best_instance is None:
                continue
            instance_type, gpi = best_instance
            tp = _largest_pow2_divisor_leq(heads, gpi)
            rank = _normalize_candidate_rank(
                {
                    "role": "aggregate",
                    "env": list(env),
                    "config": {"instance_type": instance_type, "gpu_count": tp, "tp": tp, "pp": 1},
                    "n_replicas": 1,
                }
            )
            if rank is not None and _rank_shape_key(rank) not in seen:
                seen.add(_rank_shape_key(rank))
                frames.append(rank)
        frames_by_job[jid] = frames
        ctx_by_job[jid] = (user_id, slice_id, features)

    # Phase 2 (surrogate-heavy): score EVERY frame across all jobs CONCURRENTLY.
    # Each _score_one_frame runs size_ladder + check_feasibility + compute_sigma -
    # tens of seconds of surrogate each - so scoring them serially blew the tick
    # (~7 min). Fan the leaf scorings out on a thread pool: the surrogate runs its
    # own worker pool and concurrent tool calls are already used safely in the REPL.
    tasks = [
        (jid, ctx_by_job[jid][0], ctx_by_job[jid][1], rank, ctx_by_job[jid][2])
        for jid, frames in frames_by_job.items()
        for rank in frames
    ]
    scored_by_job: dict[str, list[dict[str, Any]]] = {jid: [] for jid in frames_by_job}
    if tasks:

        def _run(task: tuple) -> tuple:
            jid, user_id, slice_id, rank, features = task
            return jid, _score_one_frame(jid, user_id, slice_id, rank, features)

        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
            for jid, scored in pool.map(_run, tasks):
                scored_by_job[jid].append(scored)

    # Phase 3 (cheap): group into candidates / exhausted / diagnostics. Every
    # feasible frame is a candidate - INCLUDING under-target ones (placing beats
    # deferring; the joint solver decides serve-vs-defer from sigma). `exhausted` =
    # a job with NO runnable, feasible frame at all, not merely "under target".
    candidates: list[dict[str, Any]] = []
    exhausted: dict[str, str] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for jid, scored_list in scored_by_job.items():
        job_diag = [s["diag"] for s in scored_list]
        job_candidates = [s["candidate"] for s in scored_list if s["candidate"] is not None]
        diagnostics[jid] = job_diag
        if job_candidates:
            candidates.extend(job_candidates)
        else:
            reasons = [d["reason"] for d in job_diag if d.get("reason")]
            exhausted[jid] = (
                "; ".join(dict.fromkeys(reasons)) if reasons else "no runnable, feasible frame"
            )

    return {"candidates": candidates, "exhausted": exhausted, "diagnostics": diagnostics}


def jointly_select_placements(
    candidates: list[dict[str, Any]],
    reserves: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Joint GPU selection across ALL waiting jobs against the one shared pool.

    Chooses at most one candidate frame per job (or defers it) to MAXIMIZE the
    cluster objective - the sum of placed per-job sigma minus the unserved-demand
    penalty for every waiting job left unplaced - subject to per-env free-GPU
    capacity. This is the joint decision the greedy per-job loop cannot make: it
    weighs every job's GPU options together, so a scarce type (e.g. H100) goes to
    whichever job it helps most instead of being pre-split blindly. It ARBITRATES
    the frames you pass; it does NOT invent them. Proposing the right GPU types
    (an L40S frame and an H100 frame for a big model) is the planner's
    domain-knowledge job - this tool just picks the joint optimum among them.

    Capacity is enforced in TWO dimensions - env GPU totals AND per-pool whole-
    INSTANCE limits (validator C5) - so the returned assignment actually fits the
    instance pools, not just the env GPU count (two 4x g6e.xlarge jobs need 8
    such instances but only 4 may be free). Still run check_feasibility on the
    assembled plan for the full C0-C7 checks (SLO, quota, swap budget, ...).

    Args:
        candidates: scored, already check_feasibility-passed frames. Each is a
            dict:
              {"job_id": str,
               "sigma": float,       # the PER-JOB sigma, i.e.
                                     # compute_sigma(one_job_plan)["per_job"][job_id]["sigma"]
                                     # (NOT aggregate_sigma - the serve penalty is applied here),
               "ladder": [rank, ...] # each rank carries env + config.gpu_count + n_replicas,
               ...}                  # any other keys (type, user_id, target_tps, mechanism_id,
                                     # rationale, ...) pass through on the winner, so the chosen
                                     # entries are ready to drop into plan["actions"].
        reserves: optional {env_key: int} GPUs to hold back.

    Returns:
        {"chosen": [candidate, ...],   # <=1 per job, the joint-optimal set
         "deferred": [job_id, ...],    # waiting jobs no candidate served
         "objective": float,           # gain over the all-defer baseline (>= 0)
         "used": {cap_key: int}, "capacity": {cap_key: int}}
    (cap_key is 'gpu:<env>' or 'pool:<env>:<instance_type>')
    """
    _require("resource_map")
    reserve_map = {_env_key(env): int(n) for env, n in (reserves or {}).items()}
    resources = get_resource_map()
    specs = instance_catalog()
    # Capacity is two-dimensional: env GPU totals AND per-pool whole-instance
    # limits. The pool dimension is what the old env-GPU-only check missed.
    capacity: dict[tuple, int] = {}
    for env, info in resources.items():
        env_key = _env_key(env)
        capacity[("gpu", env_key)] = max(0, int(info.get("free", 0)) - reserve_map.get(env_key, 0))
    for env_key, pools in specs.items():
        for instance_type, spec in pools.items():
            capacity[("pool", env_key, str(instance_type))] = int(
                spec.get("free_instances", 0) or 0
            )
    priority_by_job = {
        p.get("job_id"): float(p.get("priority_score", 1.0) or 1.0) for p in get_priority()
    }

    def penalty(jid: str) -> float:
        return UNSERVED_PENALTY * max(1.0, priority_by_job.get(jid, 1.0))

    # Group scored candidates by job, attaching each frame's per-env GPU cost and
    # its GAIN over deferring that job: placing adds sigma AND avoids the -penalty,
    # so the marginal value of placing vs deferring is sigma + penalty. A frame
    # whose gain <= 0 (sigma more negative than the serve penalty) is never worth
    # placing over a defer and is dropped up front.
    by_job: dict[str, list[dict[str, Any]]] = {}
    for cand in candidates or []:
        jid = cand.get("job_id")
        if not jid:
            continue
        cost = _ladder_capacity_cost(cand.get("ladder") or [], specs)
        if not cost:
            continue  # no real GPU footprint -> not a placeable frame
        gain = float(cand.get("sigma", 0.0)) + penalty(jid)
        if gain <= 0:
            continue
        by_job.setdefault(jid, []).append({"cand": cand, "cost": cost, "gain": gain})
    jobs = [jid for jid in by_job if by_job[jid]]

    best: dict[str, Any] = {"objective": 0.0, "chosen": []}  # all-defer baseline == 0 gain

    space = 1
    for jid in jobs:
        space *= 1 + len(by_job[jid])

    if space <= 200_000:
        # Exact branch-and-bound: every node is a capacity-feasible assignment
        # (deferring the remaining jobs), so its accumulated gain is a valid
        # objective; keep the best. Place-branches that overflow a pool are pruned.
        def dfs(i: int, used: dict[str, int], gain: float, chosen: list[dict[str, Any]]) -> None:
            if gain > best["objective"]:
                best["objective"] = gain
                best["chosen"] = list(chosen)
            if i >= len(jobs):
                return
            dfs(i + 1, used, gain, chosen)  # defer job i
            for entry in by_job[jobs[i]]:
                new_used = dict(used)
                over = False
                for key, need in entry["cost"].items():
                    new_used[key] = new_used.get(key, 0) + need
                    if new_used[key] > capacity.get(key, 0):
                        over = True
                        break
                if over:
                    continue
                chosen.append(entry["cand"])
                dfs(i + 1, new_used, gain + entry["gain"], chosen)
                chosen.pop()

        dfs(0, {}, 0.0, [])
    else:
        # Greedy fallback for a large choice space: best-gain frame per job in
        # priority order, taking each only if it still fits. Bounded, never over
        # capacity, not guaranteed optimal.
        log.warning("jointly_select_placements: %d combos, using greedy fallback", space)
        used: dict[str, int] = {}
        chosen: list[dict[str, Any]] = []
        total = 0.0
        for jid in sorted(jobs, key=lambda j: priority_by_job.get(j, 1.0), reverse=True):
            for entry in sorted(by_job[jid], key=lambda e: e["gain"], reverse=True):
                trial = dict(used)
                over = False
                for key, need in entry["cost"].items():
                    trial[key] = trial.get(key, 0) + need
                    if trial[key] > capacity.get(key, 0):
                        over = True
                        break
                if not over:
                    used, total = trial, total + entry["gain"]
                    chosen.append(entry["cand"])
                    break
        best = {"objective": total, "chosen": chosen}

    chosen = best["chosen"]
    placed_ids = {c.get("job_id") for c in chosen}
    used_final: dict[str, int] = {}
    for c in chosen:
        for key, need in _ladder_capacity_cost(c.get("ladder") or [], specs).items():
            used_final[_cap_key_str(key)] = used_final.get(_cap_key_str(key), 0) + need
    deferred = [
        jid
        for jid in (j.get("job_id", j.get("id")) for j in get_pending_jobs())
        if jid and jid not in placed_ids
    ]
    return {
        "chosen": chosen,
        "deferred": deferred,
        "objective": best["objective"],
        "used": used_final,
        "capacity": {_cap_key_str(k): v for k, v in capacity.items()},
    }


def check_feasibility(plan) -> dict[str, Any]:
    """Validate a plan with the bound plan validator.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.

    Returns:
        {"feasible": bool, "violations": List[str]}.
    """
    _require("plan_validator", "resource_map", "slow_loop")
    # Materialize omitted jobs (active -> keep, waiting -> defer) BEFORE
    # validation, exactly as the harness does to the committed plan. Otherwise
    # C2 coverage rejects any plan that legitimately relies on auto-defer, the
    # place-vs-defer baseline (an empty/defer plan) reads as INFEASIBLE, and the
    # planner defers everything because it can never establish a baseline sigma.
    typed = _as_plan(plan)
    covered = {a.job_id for a in typed.actions}
    for job in list(get_active_jobs()):
        jid = job.get("job_id", job.get("id"))
        if jid and jid not in covered:
            typed.actions.append(PlanAction(job_id=str(jid), type=ActionType.KEEP))
            covered.add(jid)
    for job in list(get_pending_jobs()):
        jid = job.get("job_id", job.get("id"))
        if jid and jid not in covered:
            typed.actions.append(PlanAction(job_id=str(jid), type=ActionType.DEFER))
            covered.add(jid)
    result = _CTX.plan_validator.val_plan(
        plan=typed,
        cluster_snapshot=_snapshot(),
        slow_state=_CTX.slow_loop.state,
    )
    feasible = bool(getattr(result, "feasible", False))
    violations = list(getattr(result, "violations", []))
    # Physical-validity of each proposed config (tp*pp vs GPUs, model sharding),
    # enforced in CODE not the prompt: a config the model cannot shard is
    # infeasible regardless of what the C0-C7 validator checked.
    snapshot = _snapshot()
    for action in typed.actions:
        if action.type in LADDER_ACTIONS and action.ladder:
            for i, rank in enumerate(action.ladder):
                ok_cfg, reason = config_runnable(
                    dict(getattr(rank, "config", {}) or {}),
                    _job_features_for(snapshot, action.job_id),
                )
                if not ok_cfg:
                    feasible = False
                    violations.append(f"job {action.job_id} rank {i}: {reason}")
    # Return BOTH keys (ok + feasible) so either planner convention reads it
    # right - check_feasibility historically used "feasible" while every other
    # validation tool uses "ok"; exposing both removes that footgun.
    return {
        "feasible": feasible,
        "ok": feasible,
        "violations": violations,
    }


def swap_counter(plan) -> int:
    """Count active-job churn against the C4 swap budget B_t.

    Counts actions in SWAP_BUDGET_ACTIONS (swap). PLACE
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
    snapshot = _snapshot() if getattr(_CTX, "resource_map", None) is not None else None
    z_star = _CTX.slow_loop.get_sss_z_star_t()
    ranges = _CTX.slow_loop.typical_ranges
    objectives = list(z_star.keys())
    scores = dict.fromkeys(objectives, 0.0)
    n = 0
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS:
            continue
        y_hat = _compose_job_y_hat(action, _job_features_for(snapshot, action.job_id))
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
    """Predict outcomes per ladder-bearing action: each rank, plus composed.

    y_hat/v_hat are predicted per RANK (per config). For a heterogeneous
    ladder this returns every rank's prediction AND the composed job-level
    y_hat that compute_sigma scores (throughput as a replica-weighted mean,
    worst-case latency, blended cost) - the parts and the whole.

    Args:
        plan: A typed Plan or any raw form Plan.from_raw accepts.

    Returns:
        Dict job_id -> {"per_rank": [{"env", "n_replicas", "mechanism_id",
        "y_hat", "v_hat", "dro_band"}, ...], "job_y_hat": composed y_hat}.
    """
    typed = _as_plan(plan)
    snapshot = _snapshot() if getattr(_CTX, "resource_map", None) is not None else None
    out: dict[str, Any] = {}
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        job_features = _job_features_for(snapshot, action.job_id)
        per_rank = []
        for rank in action.ladder:
            payload = _rank_prediction_payload(rank, job_features)
            pred = _predict_outcome_core(payload["job_config"], payload["job_features"])
            per_rank.append(
                {
                    "env": list(rank.env) if rank.env else None,
                    "n_replicas": rank.n_replicas,
                    "mechanism_id": rank.mechanism_id or action.mechanism_id,
                    "y_hat": pred.get("y_hat", {}),
                    "v_hat": pred.get("v_hat", {}),
                    "dro_band": pred.get("dro_band", {}),
                }
            )
        out[action.job_id] = {
            "per_rank": per_rank,
            "job_y_hat": _compose_job_y_hat(action, job_features),
        }
    return out


# ----------------------------------------------------------------------
# Internal adapters
# ----------------------------------------------------------------------


def _materialize_ladder(ladder_ranks):
    """Adapt a canonical rank-list ladder into the object shape eig.py consumes.

    eig.py expects .ranks (each .mechanism_id, .config, .n_replicas),
    .envs(), .duration_minutes, and .applicable_mechanisms. The
    applicable set contains the mechanisms committed by the ranks. An empty
    set would zero the relevance gate and silently kill EIG.
    """

    class _Ladder:
        pass

    class _Rank:
        pass

    ladder = _Ladder()
    ladder.ranks = []
    ladder.duration_minutes = 5.0

    for r in ladder_ranks:
        if hasattr(r, "to_dict"):
            r = r.to_dict()
        rank = _Rank()
        rank.mechanism_id = r.get("mechanism_id")
        if rank.mechanism_id is None:
            raise ValueError("ladder rank requires mechanism_id")
        rank.config = _sanitize_agent_config(r.get("config", {}))
        rank.n_replicas = int(r.get("n_replicas", 1))
        rank.is_canary = bool(r.get("is_canary", False))
        env = r.get("env")
        # env arrives as a list from RankSpec.to_dict; envs() puts these in
        # a set, so coerce to a hashable tuple here.
        rank.env = tuple(env) if isinstance(env, (list, tuple)) else env
        ladder.ranks.append(rank)

    applicable = {}
    if _CTX.mechanism_registry is not None:
        for rank in ladder.ranks:
            try:
                mech = _CTX.mechanism_registry.get_mechanism(rank.mechanism_id)
                applicable[mech.mechanism_id] = mech
            except KeyError:
                raise ValueError(f"unknown mechanism_id {rank.mechanism_id!r}") from None

    ladder.applicable_mechanisms = list(applicable.values())
    ladder.envs = lambda: {r.env for r in ladder.ranks if r.env is not None}
    return ladder


def _materialize_chain_list(chain_list):
    """Adapt rank/store-chain dicts into ChainEntry objects for switchcost.py.

    Synthesizes a stable chain_id when one is absent (role + env +
    sorted config), because switch cost matches delta_L+/delta_L- by chain_id -
    None ids would collapse every distinct rank into one and break the
    add/kill diff. Store rows use shape_json/target_node; planner ranks use
    config/env. env is coerced to a hashable tuple for pricing lookups.
    """
    if not chain_list:
        return []
    if hasattr(chain_list[0], "chain_id"):
        return list(chain_list)
    from src.cost.switch_cost import ChainEntry

    out = []
    for c in chain_list:
        shape = c.get("shape_json") or {}
        config = c.get("config") or shape
        env = c.get("env") or c.get("target_node") or shape.get("env") or shape.get("target_node")
        env = tuple(env) if isinstance(env, (list, tuple)) else env
        chain_id = c.get("chain_id")
        if not chain_id:
            import hashlib

            # repr over key-sorted config tolerates unhashable values
            # (nested lists/dicts) while staying deterministic per tick.
            fingerprint = repr(
                (c.get("role", ""), env, sorted(config.items(), key=lambda kv: kv[0]))
            )
            chain_id = "auto_" + hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
        out.append(
            ChainEntry(
                chain_id=chain_id,
                config=config,
                env=env,
                n_replicas=int(c.get("n_replicas") or c.get("chains") or 1),
            )
        )
    return out
