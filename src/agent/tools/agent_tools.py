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
        get_surrogate_budget_status current tick search-budget accounting
        get_partial_online_admission_status  guarded admission mode/counters
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

import copy
import json
import logging
import math
import time
from functools import wraps
from threading import Lock
from typing import Any

from src.config.hyperparameters import GAMMA_SLO
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
from src.prediction.composer import compact_prediction_lineage
from src.prediction.surrogate import (
    SurrogateExecutionError,
    SurrogateMemoryNoFit,
    SurrogateUnsupportedConfig,
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
_INTERNAL_AGENT_CONFIG_X = frozenset({"_arrival_share_rps"})

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
    """Strip engine-owned, autotuned, and Koi-internal values from CONFIG.

    The agent may only propose the placement/topology/parallelism/PD knobs in
    AGENT_TUNABLE_X; everything else is engine/catalog-owned and is removed here
    so an invalid invented value (router_policy='latency', an unsupported tp,
    etc.) can never reach the surrogate. The catalog, workload features, and
    surrogate defaults supply the real values.
    """
    drop = _ENGINE_AUTOTUNED_X | _ENGINE_OWNED_X | _INTERNAL_AGENT_CONFIG_X
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
    # Divisor that normalizes the cost gap (cost - ideal). It should be ~the observed
    # cost SPREAD (~1e-6 cheap to ~2e-5 pricey), so gaps land O(0.1-2) and cost
    # DIFFERENTIATES frames. COST_PENALTY_CAP bounds the top regardless, so a small
    # divisor no longer risks cost dominating J - that's what the cap is for. (An
    # earlier 5e-2 made gaps ~1e-5 -> cost went silent -> the latency bonus
    # over-provisioned onto premium GPUs.) Slow loop overwrites this with the learned
    # spread once it has cost evidence.
    "cost_per_token": 1e-5,
    "throughput_token_per_sec": 1000.0,
    "slo_margin": 1.0,
}
DEFAULT_COLD_START_Z_STAR = {
    "p99_ttft_ms": 300.0,
    "p99_tpot_ms": 50.0,
    # IDEAL (best-case) $/token the cost objective measures the gap ABOVE. It must sit
    # BELOW the CHEAPEST achievable real cost, or those frames read as "at/below ideal"
    # -> gap 0 -> cost cannot differentiate them. This bit us: 7B frames predicted
    # ~9e-7, BELOW the old 1e-6 ideal, so cost went silent and the latency opt_bonus
    # over-provisioned them onto premium H100. 1e-8 is a near-zero floor below any
    # realistic per-token cost, so EVERY frame carries a positive, differentiating
    # (bounded, capped) cost gap -> cheapest-that-meets wins. Slow loop learns the min.
    "cost_per_token": 1e-8,
    "throughput_token_per_sec": 3000.0,
    "slo_margin": 1.0,
}
# Opportunity cost charged per WAITING job the plan leaves unserved, so a
# feasible placement (sigma ~ J<=0 + EIG) beats defer (0). Scaled by priority.
UNSERVED_PENALTY = 1.0
# COST is a weighted OPTIMIZE objective (see compute_sigma), governed by the slow
# loop's weight w_t["cost_per_token"] - the SAME code serves any market, only the
# weight changes. Reserved sets it ~0 (fleet is sunk -> cost inert, a small job may
# use a free faster H100); a pay-per-use market sets it > 0 (then cheaper-$/token
# wins, which can be H100 since it is cheaper PER TOKEN when fast enough). SLO stays
# PRIMARY (target-relative, saturating); cost is a secondary additive term, so it
# only optimizes AMONG SLO-meeting frames and never overrides a real SLO need.
# HARD INVARIANT enforced in code (not trusted to the slow loop): the cost term is
# soft-capped at COST_PENALTY_CAP in compute_sigma, so no matter how TIGHT a cost
# range the slow loop learns (it learns the observed spread, which can be ~1e-5 and
# would otherwise blow cost_penalty up to ~0.4 - rivaling J), cost can never outrank
# a meaningful SLO gap. J for a clearly under-target frame is ~-0.15..-0.20; the cap
# sits well below that, so "satisfice SLO, THEN minimize cost" holds structurally.
# The cap is a soft saturation (monotonic), so cheaper-per-token frames still win
# AMONG meeters - it only bites when a tiny learned range would make cost dominate.
COST_PENALTY_CAP = 0.05
# Soft-cap on the beyond-target OPTIMIZATION bonus (see compute_sigma). Same role as
# COST_PENALTY_CAP: once a job MEETS its SLO/demand target, Koi may still be rewarded
# for BEATING it on the mode-relevant axis (batch -> more throughput, online -> lower
# latency), but that reward saturates here so it can NEVER outrank another job's
# target MISS. That IS the fairness guard: satisfice every target first (primary J),
# then optimize the mode axis with leftover capacity only (bounded secondary).
OPT_BONUS_CAP = 0.05


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
_Z_STAR_UNSET_SENTINELS = frozenset({0.0, 99999.0, 999999.0})


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
_SURROGATE_EXECUTION_LOCK = Lock()


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
    trace_logger = None
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
    surrogate_changed = components.get("surrogate") is not None and components.get(
        "surrogate"
    ) is not getattr(_CTX, "surrogate", None)
    specialist_changed = components.get("specialist_runner") is not None and components.get(
        "specialist_runner"
    ) is not getattr(_CTX, "specialist_runner", None)
    context_changed = any(
        value is not None and value is not getattr(_CTX, name, None)
        for name, value in components.items()
    )
    for name, value in components.items():
        if not hasattr(_ToolContext, name):
            raise ValueError(f"bind_tools: unknown component {name!r}")
        if value is not None:
            setattr(_CTX, name, value)
    if surrogate_changed:
        with _SURROGATE_EXECUTION_LOCK:
            _prediction_cache.clear()
    if specialist_changed:
        _specialist_results_cache.clear()
    if context_changed:
        _scored_candidates_cache.clear()
    surrogate = getattr(_CTX, "surrogate", None)
    evidence_store = getattr(_CTX, "evidence_store", None)
    if (
        surrogate is not None
        and evidence_store is not None
        and hasattr(surrogate, "bind_evidence_store")
    ):
        surrogate.bind_evidence_store(evidence_store)


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


# Per-tick surrogate SEARCH-call budget: a runaway BACKSTOP, not a quality limiter.
# The planner may explore freely (grid / coordinate search) up to this many
# surrogate simulations per tick; beyond it a sim raises SurrogateBudgetExceeded,
# which candidate evaluation reports separately from physical failures, so hitting
# it late means "commit the best scored so far", never a crash or forced defer.
# Raise it for deeper search; it is generous by design (a normal tick uses far fewer).
SURROGATE_CALL_BUDGET = 100
PARTIAL_ONLINE_ADMISSION_MODE = "off"
_AIC_DIRECT_METHOD = ("AIC_Direct",)
_AIC_DYNOSIM_METHOD = ("AIC_DynoSim",)
_surrogate_calls = 0
_surrogate_cache_hits = 0
_surrogate_raw_cache_hits = 0
_surrogate_budget_rejections = 0
_surrogate_finalization_calls = 0
_surrogate_stress_calls = 0
_partial_online_searches = 0
_partial_online_queue_aware_probes = 0
_partial_online_safe_probes = 0
_partial_online_admissions = 0
_partial_online_truncated_searches = 0
# Per-tick memo of RAW surrogate output keyed on (job_config, job_features,
# scenario, calibration, method, accounting mode). DynoSim is deterministic, so
# re-probing a config the LLM already
# evaluated THIS tick returns the identical numbers - we serve them from here
# instead of re-running the surrogate. Access happens under
# _SURROGATE_EXECUTION_LOCK (see _predict_outcome_core).
# Cleared every tick by reset_tick_caches so calibration / z* / evidence updates are
# never stale, and so a cache hit never leaks across capacity/telemetry boundaries.
_prediction_cache: dict[str, dict[str, Any]] = {}
_specialist_results_cache: dict[str, dict[str, dict[str, Any]]] = {}
_scored_candidates_cache: dict[str, dict[str, Any]] = {}


class SurrogateBudgetExceeded(RuntimeError):
    """Raised when a search call would exceed the tick's surrogate budget."""


def configure_surrogate_call_budget(limit: int) -> None:
    """Set the per-tick search-call limit without resetting current metrics."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("surrogate call budget must be an integer")
    if limit < 0:
        raise ValueError("surrogate call budget must be >= 0")
    global SURROGATE_CALL_BUDGET
    with _SURROGATE_EXECUTION_LOCK:
        SURROGATE_CALL_BUDGET = limit
    # A result truncated under the old limit is not reusable after reconfiguration.
    _scored_candidates_cache.clear()


def configure_partial_online_admission(mode: str) -> None:
    """Configure guarded partial admission without resetting tick metrics."""
    if not isinstance(mode, str):
        raise TypeError("partial online admission mode must be a string")
    if mode not in {"off", "advisory"}:
        raise ValueError("partial online admission mode must be exactly 'off' or 'advisory'")
    global PARTIAL_ONLINE_ADMISSION_MODE
    with _SURROGATE_EXECUTION_LOCK:
        PARTIAL_ONLINE_ADMISSION_MODE = mode
    _scored_candidates_cache.clear()


def get_partial_online_admission_status() -> dict[str, Any]:
    """Return configured mode and current-tick guarded-search accounting."""
    with _SURROGATE_EXECUTION_LOCK:
        return {
            "mode": PARTIAL_ONLINE_ADMISSION_MODE,
            "searches": _partial_online_searches,
            "queue_aware_probes": _partial_online_queue_aware_probes,
            "safe_probes": _partial_online_safe_probes,
            "admissions": _partial_online_admissions,
            "truncated_searches": _partial_online_truncated_searches,
        }


def get_surrogate_budget_status() -> dict[str, int]:
    """Return current tick search usage and non-search surrogate accounting."""
    with _SURROGATE_EXECUTION_LOCK:
        return {
            "limit": SURROGATE_CALL_BUDGET,
            "calls_executed": _surrogate_calls,
            "cache_hits": _surrogate_cache_hits,
            "raw_cache_hits": _surrogate_raw_cache_hits,
            "budget_rejections": _surrogate_budget_rejections,
            "finalization_calls": _surrogate_finalization_calls,
            "stress_calls": _surrogate_stress_calls,
            "remaining": max(0, SURROGATE_CALL_BUDGET - _surrogate_calls),
        }


def _tick_cache_key(*values: Any) -> str | None:
    """Build a stable key for plain per-tick tool inputs."""
    try:
        return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return None


def reset_tick_caches() -> None:
    """Clear per-tick tool caches and surrogate accounting.

    Must run at every tick boundary (S0 wires it via the TickRunner's
    on_tick_start hook). Without this, run_job_specialists' default-book
    path could reuse a book validated against LAST tick's capacity -
    a stale-budget hole in the anti-split-brain ordering.
    """
    global _surrogate_budget_rejections, _surrogate_cache_hits, _surrogate_calls
    global _surrogate_raw_cache_hits
    global _surrogate_finalization_calls, _surrogate_stress_calls
    global _partial_online_admissions, _partial_online_queue_aware_probes
    global _partial_online_safe_probes, _partial_online_searches
    global _partial_online_truncated_searches
    _CTX.user_envelopes = None
    _CTX.validated_budget_book = None
    with _SURROGATE_EXECUTION_LOCK:
        _surrogate_calls = 0
        _surrogate_cache_hits = 0
        _surrogate_raw_cache_hits = 0
        _surrogate_budget_rejections = 0
        _surrogate_finalization_calls = 0
        _surrogate_stress_calls = 0
        _partial_online_searches = 0
        _partial_online_queue_aware_probes = 0
        _partial_online_safe_probes = 0
        _partial_online_admissions = 0
        _partial_online_truncated_searches = 0
        _prediction_cache.clear()
    _specialist_results_cache.clear()
    _scored_candidates_cache.clear()


# Public module functions that are NOT LLM tools (infrastructure/boot).
_NON_TOOL_NAMES = frozenset(
    {
        "bind_tools",
        "all_callables",
        "assert_planning_ready",
        "compute_sigma_for_commit",
        "configure_partial_online_admission",
        "configure_surrogate_call_budget",
        "reset_tick_caches",
        "stamp_plan_predictions",
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


def _rank_prediction_payload(
    rank: RankSpec,
    job_features: dict[str, Any] | None = None,
    *,
    arrival_rate_rps: float | None = None,
) -> dict:
    """Build the surrogate payload for one rank without mutating the rank."""
    features = _sanitize_agent_features(dict(job_features or {}))
    arrival_share = arrival_rate_rps
    if arrival_share is None:
        arrival_share = rank.config.get("_arrival_share_rps")
    if arrival_share is not None:
        features["request_arrival_rate"] = float(arrival_share)
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
            resources = resource_map.resources_summary()
            compiled_x = build_rank_x(
                job_values=features,
                shape=shape,
                env=(str(env[0]), str(env[1]), str(env[2]), str(env[3]), str(env[4])),
                resources=resources,
                hardware_catalog=resource_map.hardware_catalog(),
                model_catalog=resource_map.model_catalog(str(model_id)),
                replica_count=max(1, int(rank.n_replicas or 1)),
            )
            config.update(compiled_x)
            allocation = _rank_allocation_summary(rank, resources)
            price = allocation.get("price_per_unit_hour")
            if price is not None:
                config["price_per_hour"] = float(price)
        except Exception:
            log.exception("rank prediction X assembly failed; using rank config only")
    config.pop("_arrival_share_rps", None)
    return {"job_config": config, "job_features": features}


def _public_rank_arrival_rps(action, rank: RankSpec, job_features: dict[str, Any]) -> float | None:
    """Derive a rank's request rate from public action accounting when available."""
    target_tps = getattr(action, "target_tps", None)
    traffic_share = rank.rank_traffic_share
    output_length = _feature_value(job_features, "osl_token_avg", "output_len_tokens_avg")
    try:
        target = float(target_tps)
        share = float(traffic_share)
        output = float(output_length)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (target, share, output)):
        return None
    if target <= 0.0 or share <= 0.0 or output <= 0.0:
        return None
    return target * share / output


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
    thresholds = (
        dict(snapshot.slo_thresholds(job_id) or {}) if hasattr(snapshot, "slo_thresholds") else {}
    )
    features = _job_features_for(snapshot, job_id)
    for outcome, names in {
        "p99_ttft_ms": ("target_p99_ttft_ms", "target_p99_TTFT_ms"),
        "p99_tpot_ms": ("target_p99_tpot_ms", "target_p99_TPOT_ms"),
    }.items():
        value = next((features[name] for name in names if features.get(name) is not None), None)
        if value is not None and float(value) > 0:
            thresholds.setdefault(outcome, float(value))
    return thresholds


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


def _compose_job_y_hat(
    action,
    job_features: dict[str, Any] | None = None,
    *,
    method: tuple[str, ...] = _AIC_DIRECT_METHOD,
    scenario: str = "mean",
    finalization: bool = False,
) -> dict[str, Any]:
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
        # Public action accounting is authoritative. The private config field remains
        # readable only for plans created before rank_traffic_share was persisted.
        share_rps = _public_rank_arrival_rps(action, rank, job_features or {})
        try:
            payload = _rank_prediction_payload(
                rank,
                job_features,
                arrival_rate_rps=share_rps,
            )
            y = _predict_outcome_core(
                payload["job_config"],
                payload["job_features"],
                scenario=scenario,
                method=method,
                _finalization=finalization,
            ).get("y_hat", {})
        except SurrogateBudgetExceeded:
            raise
        except (SurrogateMemoryNoFit, SurrogateUnsupportedConfig) as exc:
            log.warning("rank y_hat rejected for job %s (%s)", action.job_id, exc)
            y = {}
        except SurrogateExecutionError:
            raise
        except Exception:
            log.exception("rank y_hat failed for job %s", action.job_id)
            y = {}
        if not y:
            return {}
        samples.append((max(1, int(rank.n_replicas or 1)), y))
    if not samples:
        return {}
    # Always roll up (even a single rank) so DP is applied: a lone rank with
    # n_replicas=N must report N * per_chain throughput, not per_chain.
    return _roll_up_ranks(samples)


def _roll_up_ranks(samples: list[tuple[int, dict]]) -> dict[str, Any]:
    """Roll up per-rank y_hat samples into one job y_hat.

    Each rank's y_hat is ALREADY the DP-aggregate the surrogate simulated for that
    rank (num_workers = n_replicas), so we compose ACROSS RANKS only - we do NOT
    scale by n_replicas again (that was the throughput double-count that surfaced
    once the surrogate started honoring dp):
      throughput -> SUM across ranks (parallel ranks add up)
      latency    -> MAX across ranks (a request hits one rank; every serving rank
                    must clear its SLO)
      cost/token -> throughput-weighted mean (= total $ / total tokens)
      slo_margin -> MIN across ranks (worst headroom)
      other      -> throughput-weighted mean
    The n_replicas in each sample is ignored for scaling; it is left in the tuple
    only for backward compatibility with callers.
    """

    def _tput(y: dict) -> float:
        return float(y.get(_THROUGHPUT_OBJ) or 0.0)

    ys = [y for _, y in samples]
    if not ys:
        return {}
    objectives = set().union(*[set(y) for y in ys])
    composed: dict[str, Any] = {}
    for obj in objectives:
        present = [(float(y[obj]), _tput(y)) for y in ys if y.get(obj) is not None]
        if not present:
            continue
        if obj == _THROUGHPUT_OBJ:
            composed[obj] = sum(v for v, _ in present)
        elif obj in _LATENCY_OBJS:
            composed[obj] = max(v for v, _ in present)
        elif obj in _MARGIN_OBJS:
            composed[obj] = min(v for v, _ in present)
        else:
            weight = sum(t for _, t in present)
            composed[obj] = (
                sum(v * t for v, t in present) / weight
                if weight > 0
                else sum(v for v, _ in present) / len(present)
            )
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

    def field(job: dict[str, Any], features: dict[str, Any], *names: str, default=None):
        for source in (job, features):
            for name in names:
                value = source.get(name)
                if value is not None:
                    return value
        return default

    def number(value: Any, default: float) -> float:
        if isinstance(value, bool):
            return default
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return default
        return converted if math.isfinite(converted) else default

    priority_classes = {
        "LOW": 0.0,
        "STANDARD": 1.0,
        "HIGH": 2.0,
        "CRITICAL": 3.0,
    }
    jobs = list(get_pending_jobs()) + list(get_active_jobs())
    scored: list[dict[str, Any]] = []
    for j in jobs:
        job_features = j.get("job_features")
        if not isinstance(job_features, dict):
            job_features = {}

        raw_priority_class = field(j, job_features, "priority_class", default=0.0)
        if isinstance(raw_priority_class, str):
            class_name = raw_priority_class.strip().upper()
            priority_class = priority_classes.get(class_name, number(raw_priority_class, 0.0))
        else:
            priority_class = number(raw_priority_class, 0.0)

        workload_type = field(j, job_features, "workload_type", "type")
        if workload_type is None:
            workload_type = j.get("kind")
        if workload_type is None:
            workload_type = job_features.get("kind", "online")
        signals = {
            "user_priority": number(field(j, job_features, "user_priority"), 1.0),
            "priority_class": priority_class,
            "is_online": 1.0 if str(workload_type).strip().lower() == "online" else 0.0,
            "deadline_pressure": number(field(j, job_features, "deadline_pressure"), 0.0),
            "slo_margin_deficit": max(0.0, -number(field(j, job_features, "slo_margin"), 0.0)),
            "queue_age_ticks": number(field(j, job_features, "queue_age_ticks"), 0.0),
            "recent_failures": number(field(j, job_features, "recent_failures"), 0.0),
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
    scored.sort(key=lambda x: (-float(x["priority_score"]), str(x["job_id"] or "")))
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
    include_active: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run bounded per-job specialists under a validated BudgetBook.

    Refuses to run when the supplied book is not the one most recently
    validated by validate_budget_book - that ordering is the
    anti-split-brain invariant. Each specialist optimizes one job inside
    its BudgetSlice and reports a fitness signal; it cannot allocate
    outside its slice or see the cluster plan.

    Args:
        max_workers: Parallel specialist calls.
        include_active: Include running jobs in addition to waiting jobs. Defaults
            to False because the candidate builder currently consumes waiting jobs.

    Returns:
        Dict job_id -> JobSpecialistResult ({"job_id", "type", "ladder",
        "budget_utilization", "used_capacity", "fitness",
        "marginal_value_of_more", "unused_capacity", "mechanism_ids",
        "new_mechanism_proposals", "reasoning"}).

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
    cache_key = _tick_cache_key(book, int(max_workers), bool(include_active))
    if cache_key is not None and cache_key in _specialist_results_cache:
        return copy.deepcopy(_specialist_results_cache[cache_key])
    pending_ids = {
        job.get("job_id", job.get("id"))
        for job in get_pending_jobs()
        if job.get("job_id", job.get("id"))
    }
    eligible_ids = set(pending_ids)
    if include_active:
        eligible_ids.update(
            job.get("job_id", job.get("id"))
            for job in get_active_jobs()
            if job.get("job_id", job.get("id"))
        )
    job_ids = [job_id for job_id in (book.get("job_budgets") or {}) if job_id in eligible_ids]
    results = (
        _CTX.specialist_runner.run_many(
            jobs=job_ids, budget_book=book, max_workers=int(max_workers)
        )
        if job_ids
        else []
    )
    by_job = {str(result.get("job_id")): result for result in results}
    if cache_key is not None:
        _specialist_results_cache[cache_key] = copy.deepcopy(by_job)
        return copy.deepcopy(_specialist_results_cache[cache_key])
    return copy.deepcopy(by_job)


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

        target        = required throughput (batch: budget/deadline*headroom;
                        online: arrival_rate * output_len * headroom)
        per rank      : SEARCH replica count (DP) d = 1,2,4,...,cap. Direct cheaply
                        screens capacity. For online jobs with latency SLOs, DynoSim
                        verifies only a DP that can carry the share, plus max DP for
                        a possible SLO-safe partial contribution. Take the first DP
                        whose queue-aware TTFT/TPOT clear the SLO and keeps up; the
                        rest of the demand spills to the next rank.
        achieved_tps  = demand actually served within SLO (SUM across ranks).

    Online latency GATES the replica count (queueing latency FALLS with more
    replicas), it does not veto the rank. In advisory mode only, a rank that cannot
    provide full service may contribute a separately tested SLO-safe partial load;
    otherwise the legacy fallback remains diagnostic-only for online candidates.
    meets_target requires the whole demand covered AND every serving rank in SLO.

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
    global _partial_online_admissions, _partial_online_queue_aware_probes
    global _partial_online_safe_probes, _partial_online_searches
    global _partial_online_truncated_searches

    job_features = _sanitize_agent_features(dict(job_features or {}))
    regime = str(job_features.get("type", "online")).lower()
    is_online = regime != "batch"
    # utilization_target is accepted for backward compatibility but no longer used.
    # Direct screens each DP's capacity, and queue-aware verification supplies the
    # online p99 latency; no linear per-chain utilization estimate is needed.
    _ = utilization_target
    target = (
        float(target_tps)
        if target_tps is not None
        else float(required_throughput_enumerator(job_features))
    )
    osl = _feature_value(job_features, "osl_token_avg", "output_len_tokens_avg") or 0.0
    ttft_target = _feature_value(job_features, "target_p99_ttft_ms", "target_p99_TTFT_ms")
    tpot_target = _feature_value(job_features, "target_p99_tpot_ms", "target_p99_TPOT_ms")
    verify_online_latency = is_online and (ttft_target is not None or tpot_target is not None)

    def _finite_latency(y: dict, *keys: str) -> float | None:
        for key in keys:
            value = y.get(key)
            if value is None:
                continue
            try:
                latency = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return latency if math.isfinite(latency) and latency >= 0 else None
        return None

    def _slo_ok(y: dict) -> bool:
        # Online latency gate. Batch has no per-request latency SLO -> always passes
        # here and is sized on throughput alone.
        if not is_online:
            return True
        if not y:
            return False
        ttft = _finite_latency(y, "p99_ttft_ms", "p99_TTFT_ms")
        tpot = _finite_latency(y, "p99_tpot_ms", "p99_TPOT_ms")
        if ttft_target is not None and (ttft is None or ttft > ttft_target):
            return False
        return not (tpot_target is not None and (tpot is None or tpot > tpot_target))

    def _slo_prediction_complete(y: dict) -> bool:
        if not is_online:
            return True
        return (
            bool(y)
            and (
                ttft_target is None
                or any(y.get(key) is not None for key in ("p99_ttft_ms", "p99_TTFT_ms"))
            )
            and (
                tpot_target is None
                or any(y.get(key) is not None for key in ("p99_tpot_ms", "p99_TPOT_ms"))
            )
        )

    def _capacity_tps(y: dict) -> float:
        try:
            point = _y_value(y, "throughput_token_per_sec")
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(point) or point <= 0:
            return 0.0
        lower = y.get("_throughput_token_per_sec_lower")
        if lower is None:
            return point
        try:
            lower_value = float(lower)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(lower_value) or lower_value <= 0:
            return 0.0
        return min(point, lower_value)

    def _covers_offered_load(capacity_tps: float, offered_tps: float) -> bool:
        return capacity_tps >= offered_tps or math.isclose(
            capacity_tps,
            offered_tps,
            rel_tol=1e-3,
            abs_tol=1e-6,
        )

    def _dp_candidates(cap: int, preferred: int | None = None) -> list[int]:
        ds, d = [], 1
        while d <= cap:
            ds.append(d)
            d *= 2
        if cap >= 1 and (not ds or ds[-1] != cap):
            ds.append(cap)
        if preferred is not None and 1 <= preferred <= cap:
            larger = [candidate for candidate in ds if candidate > preferred]
            smaller = [candidate for candidate in ds if candidate < preferred]
            ds = [preferred, *larger, *smaller]
        return ds

    physical_rejections: list[str] = []

    def _run_at_load(
        rank: RankSpec,
        d: int,
        share_tps: float,
        method: tuple[str, ...],
    ) -> dict | None:
        feats = dict(job_features)
        arrival_rate_rps = None
        if is_online and osl > 0:
            arrival_rate_rps = float(share_tps) / osl
            feats["request_arrival_rate"] = arrival_rate_rps
        r = RankSpec.from_dict(rank.to_dict())
        r.n_replicas = int(d)
        payload = _rank_prediction_payload(r, feats, arrival_rate_rps=arrival_rate_rps)
        if not config_runnable(dict(r.config), payload["job_features"])[0]:
            return None
        try:
            prediction = _predict_outcome_core(
                payload["job_config"],
                payload["job_features"],
                scenario="peak",
                method=method,
            )
            y_hat = dict(prediction.get("y_hat", {}))
            lower = prediction.get("throughput_token_per_sec_lower")
            if lower is not None:
                y_hat["_throughput_token_per_sec_lower"] = lower
            return y_hat
        except (SurrogateMemoryNoFit, SurrogateUnsupportedConfig) as exc:
            log.warning("size_ladder: surrogate rejected rank config (%s)", exc)
            physical_rejections.append(str(exc))
            return None
        except SurrogateExecutionError as exc:
            message = str(exc)
            if message.startswith("completed ") and message.endswith(" requests"):
                log.warning("size_ladder: surrogate overload at dp=%d (%s)", d, exc)
                return {}
            raise

    def _predict_at(
        rank: RankSpec,
        d: int,
        share_tps: float,
        max_dp: int,
        direct_predictions: dict[int, dict | None],
    ) -> dict | None:
        # Predict this rank at DP=d workers carrying `share_tps` tokens/s of demand.
        # Direct screens capacity first. DynoSim runs only when an online DP can
        # carry the share, or at max DP to characterize a possible partial rank.
        # Returns:
        #   dict y_hat -> the DP-aggregate prediction,
        #   {}         -> overloaded/incomplete at this DP (try more replicas),
        #   None       -> physical/memory no-fit (no replica count fixes it).
        direct_y = _run_at_load(rank, d, share_tps, _AIC_DIRECT_METHOD)
        direct_predictions[d] = direct_y
        if direct_y is None or not verify_online_latency:
            return direct_y
        direct_can_carry = bool(direct_y) and _covers_offered_load(
            _capacity_tps(direct_y), share_tps
        )
        if not direct_can_carry and d != max_dp:
            return {}
        return _run_at_load(rank, d, share_tps, _AIC_DYNOSIM_METHOD)

    sized: list[dict[str, Any]] = []
    per_rank: list[dict[str, Any]] = []
    marginal: dict[str, int] = {}
    remaining = target
    remaining_by_pool: dict[tuple[str, str | None], int] = {}
    resources = (
        _CTX.resource_map.resources_summary()
        if hasattr(_CTX.resource_map, "resources_summary")
        else None
    )

    # Cover the demand rank by rank. Each rank tries to serve the REMAINING demand;
    # size_ladder SEARCHES its replica count (DP) against the surrogate's own p99, so
    # latency GATES the replica count instead of vetoing the rank up front (queueing
    # latency DOES fall with more replicas). A rank that clears TTFT+TPOT+keep-up at
    # some DP takes the whole remaining share; one that cannot even at max capacity
    # takes the throughput it can push and the shortfall spills to the next rank
    # (heterogeneous ladder). meets_target needs the whole demand covered AND every
    # serving rank within SLO.
    for raw in ranks:
        physical_rejection_start = len(physical_rejections)
        rank = RankSpec.from_dict(raw)
        preferred_replicas = max(1, int(rank.n_replicas or 1))
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

        runnable, validity_reason = config_runnable(
            dict(rank.config), _rank_prediction_payload(rank, job_features)["job_features"]
        )
        share = remaining
        n_replicas = 0
        served = 0.0
        slo_ok = False
        reason: str | None = None
        y_hat: dict[str, Any] = {}
        dp_tried = 0
        direct_predictions: dict[int, dict | None] = {}
        partial_search_attempted = False
        partial_search_probes = 0
        partial_search_truncated = False
        partial_search_upper_tps = 0.0
        partial_admission = False

        if not runnable:
            reason = f"config not runnable: {validity_reason}"
        elif max_by_cap < 1:
            reason = "no free capacity in pool"
        elif share <= 0:
            reason = "demand already covered by earlier ranks"
        else:
            met = False
            fallback_d = 0
            fallback_y: dict[str, Any] = {}
            for d in _dp_candidates(max_by_cap, preferred_replicas):
                dp_tried = max(dp_tried, d)
                y = _predict_at(rank, d, share, max_by_cap, direct_predictions)
                if y is None:
                    reason = "does not fit (memory/physical)"
                    break
                if not y:
                    # overloaded / requests did not all complete at this DP - add
                    # replicas and retry (this is the queue that DP relieves).
                    reason = "overloaded at max DP" if d == max_by_cap else reason
                    continue
                y_hat = y
                if d >= fallback_d:
                    fallback_d, fallback_y = d, y
                tp = _capacity_tps(y)
                keeps_up = _covers_offered_load(tp, share) if share > 0 else tp > 0
                if _slo_ok(y) and keeps_up and tp >= share:
                    n_replicas, served, slo_ok, reason, met = d, share, True, "ok", True
                    break
            advisory_search = (
                not met and PARTIAL_ONLINE_ADMISSION_MODE == "advisory" and verify_online_latency
            )
            if advisory_search:
                partial_search_attempted = True
                with _SURROGATE_EXECUTION_LOCK:
                    _partial_online_searches += 1
                usable_dp = [
                    d
                    for d, direct_y in direct_predictions.items()
                    if direct_y and _capacity_tps(direct_y) > 0
                ]
                if usable_dp and osl > 0:
                    partial_d = max(usable_dp)
                    partial_search_upper_tps = min(
                        share,
                        _capacity_tps(direct_predictions[partial_d] or {}),
                    )
                    low = 0.0
                    high = partial_search_upper_tps
                    safe_load = 0.0
                    safe_y: dict[str, Any] = {}
                    last_probe_y: dict[str, Any] = {}
                    for _probe_index in range(7):
                        probe_load = (low + high) / 2.0
                        if probe_load <= 0 or probe_load <= low:
                            break
                        partial_search_probes += 1
                        with _SURROGATE_EXECUTION_LOCK:
                            _partial_online_queue_aware_probes += 1
                        try:
                            probe_y = _run_at_load(
                                rank,
                                partial_d,
                                probe_load,
                                _AIC_DYNOSIM_METHOD,
                            )
                        except SurrogateBudgetExceeded:
                            if not safe_y:
                                raise
                            partial_search_truncated = True
                            with _SURROGATE_EXECUTION_LOCK:
                                _partial_online_truncated_searches += 1
                            break
                        last_probe_y = probe_y or {}
                        probe_safe = (
                            probe_y is not None
                            and _slo_prediction_complete(probe_y)
                            and _slo_ok(probe_y)
                            and _covers_offered_load(_capacity_tps(probe_y), probe_load)
                        )
                        if probe_safe:
                            low = probe_load
                            safe_load = min(probe_load, _capacity_tps(probe_y))
                            safe_y = probe_y
                            with _SURROGATE_EXECUTION_LOCK:
                                _partial_online_safe_probes += 1
                        else:
                            high = probe_load
                    if safe_y and safe_load > 0:
                        n_replicas = partial_d
                        served = safe_load
                        y_hat = safe_y
                        slo_ok = True
                        partial_admission = True
                        reason = (
                            "SLO-safe partial (search truncated by budget)"
                            if partial_search_truncated
                            else "SLO-safe partial admission"
                        )
                        with _SURROGATE_EXECUTION_LOCK:
                            _partial_online_admissions += 1
                    else:
                        y_hat = last_probe_y or fallback_y
                        slo_ok = _slo_ok(y_hat)
                        reason = "no positive SLO-safe partial load at max usable DP"
                elif osl <= 0:
                    reason = "cannot test partial load without a positive output length"
                else:
                    reason = "no positive conservative direct capacity at max usable DP"
            elif not met and fallback_y:
                # No DP up to capacity cleared SLO+keep-up at the full share. Take max
                # capacity and serve the throughput it can push; the rest spills to
                # the next rank. slo_ok=False here keeps meets_target honest.
                y_hat = fallback_y
                tp = _capacity_tps(y_hat)
                partial = min(share, tp)
                if partial > 0:
                    n_replicas = fallback_d
                    served = partial
                    slo_ok = _slo_ok(y_hat)
                    reason = "capacity-bound (partial share)" if slo_ok else "under-SLO at max DP"
            elif not met and reason is None:
                reason = "cannot meet SLO/keep-up at any replica count"

        rank.n_replicas = n_replicas
        if n_replicas >= 1:
            rank.rank_traffic_share = served / target if target > 0 else 1.0
            rank.config.pop("_arrival_share_rps", None)
            sized.append(rank.to_dict())
            remaining = max(0.0, remaining - served)
            remaining_by_pool[pool_key] = max(0, free - n_replicas * capacity_per_replica)
        if dp_tried > n_replicas and not slo_ok:
            marginal[env_key] = (
                marginal.get(env_key, 0) + max(0, dp_tried - n_replicas) * capacity_per_replica
            )

        per_rank.append(
            {
                "role": rank.role,
                "env": list(rank.env) if rank.env else None,
                "instance_type": allocation["instance_type"] if not allocation_error else None,
                "gpus_per_chain": gpus_per_chain,
                "capacity_gpus_per_replica": capacity_per_replica,
                "free_capacity_gpus": free,
                "price_per_unit_hour": allocation["price_per_unit_hour"]
                if not allocation_error
                else None,
                "max_replicas_by_capacity": max_by_cap,
                "n_replicas": n_replicas,
                "share_tps": share,
                "served_tps": served,
                "slo_ok": slo_ok,
                "prediction_received": bool(y_hat),
                "prediction_complete": _slo_prediction_complete(y_hat),
                "partial_search_attempted": partial_search_attempted,
                "partial_search_probes": partial_search_probes,
                "partial_search_truncated": partial_search_truncated,
                "partial_search_upper_tps": partial_search_upper_tps,
                "partial_admission": partial_admission,
                "admitted_tps": served if partial_admission else None,
                "reason": reason,
                "physical_violations": physical_rejections[physical_rejection_start:],
            }
        )
        if partial_search_truncated:
            break

    # achieved = demand actually served; meets_target needs the WHOLE demand covered
    # AND every serving rank inside its latency SLO.
    achieved_tps = max(0.0, target - remaining)
    serving = [r for r in per_rank if r["n_replicas"] >= 1]
    served_slo_ok = bool(serving) and all(r["slo_ok"] for r in serving)
    meets_target = remaining <= max(1e-6, 1e-3 * target) and served_slo_ok
    partial_online_admission = (
        PARTIAL_ONLINE_ADMISSION_MODE == "advisory"
        and verify_online_latency
        and achieved_tps > 0
        and achieved_tps < target
        and any(r["partial_admission"] for r in serving)
    )
    return {
        "ranks": sized,
        "regime": regime,
        "target_tps": target,
        "achieved_tps": achieved_tps,
        "unmet_tps": max(0.0, remaining),
        "meets_target": meets_target,
        "partial_online_admission": partial_online_admission,
        "admission_mode": "advisory" if partial_online_admission else None,
        "partial_search_probes": sum(r["partial_search_probes"] for r in per_rank),
        "partial_search_truncated": any(r["partial_search_truncated"] for r in per_rank),
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


def predict_outcome(
    config: dict[str, Any],
    mechanism: dict[str, Any] | None = None,
    env: list[str] | tuple[str, ...] | None = None,  # TODO - Added direct env to this tool for v0
    calibrate: bool = True,
    scenario: str = "mean",
    queue_aware: bool = False,
) -> dict[str, Any]:
    """Run the composed surrogate and attach its prediction lineage and DRO band.

    Args:
        config: X variables for the candidate. May embed job_config and
            job_features sub-dicts; otherwise the whole dict is job_config.
        mechanism: Optional mechanism context, informational only.
        env: Optional canonical [market, cloud, region, zone, gpu_type].
        calibrate: Apply the residual correction (default True).
        queue_aware: Use DynoSim queue verification instead of the default
            inexpensive Direct prediction.

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
    is_online = _job_mode(job_features) == "online"
    method = (
        _AIC_DYNOSIM_METHOD
        if is_online and (queue_aware or scenario == "peak_all_multiturn_stress")
        else _AIC_DIRECT_METHOD
    )
    return _predict_outcome_core(
        job_config,
        job_features,
        calibrate=calibrate,
        scenario=scenario,
        method=method,
    )


def _prediction_cache_key(
    job_config: dict[str, Any],
    job_features: dict[str, Any],
    scenario: str,
    calibrate: bool,
    method: tuple[str, ...],
    finalization: bool,
) -> str | None:
    """Canonical, order-stable key for the composed-prediction memo.

    Calibration and accounting modes are part of the key so finalization cannot
    silently reuse a search-budgeted prediction without finalization accounting.
    """
    try:
        return json.dumps(
            [job_config, job_features, scenario, bool(calibrate), method, bool(finalization)],
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        return None


def _aic_raw_cache_hit(prediction_lineage: Any) -> bool:
    """Whether the primary AIC backend reports a successful cross-tick raw hit."""
    if not isinstance(prediction_lineage, dict):
        return False
    components = prediction_lineage.get("components")
    primary = components.get("primary") if isinstance(components, dict) else None
    metadata = primary.get("metadata") if isinstance(primary, dict) else None
    raw_cache = metadata.get("aic_raw_cache") if isinstance(metadata, dict) else None
    return isinstance(raw_cache, dict) and raw_cache.get("hit") is True


def _predict_outcome_core(
    job_config: dict[str, Any],
    job_features: dict[str, Any],
    calibrate: bool = True,
    scenario: str = "mean",
    *,
    method: str | tuple[str, ...] = _AIC_DIRECT_METHOD,
    _finalization: bool = False,
) -> dict[str, Any]:
    """Run or reuse one composed prediction with trusted, sanitized inputs.

    Cache misses remain serialized because the composed surrogate is not thread-safe.
    The composer owns calibration; callers receive final values plus the raw and
    calibration details carried in its prediction lineage.
    """
    _require("candidate_graph", "dro", "surrogate")
    global _surrogate_budget_rejections, _surrogate_cache_hits, _surrogate_calls
    global _surrogate_raw_cache_hits
    global _surrogate_finalization_calls, _surrogate_stress_calls
    selected_method = (method,) if isinstance(method, str) else tuple(method)
    key = _prediction_cache_key(
        job_config,
        job_features,
        scenario,
        calibrate,
        selected_method,
        _finalization,
    )

    with _SURROGATE_EXECUTION_LOCK:
        cached = _prediction_cache.get(key) if key is not None else None
        if cached is not None:
            _surrogate_cache_hits += 1
        else:
            is_stress = scenario == "peak_all_multiturn_stress"
            is_search = not _finalization and not is_stress
            preflight_raw_hit = False
            charged_search = False
            if _finalization:
                _surrogate_finalization_calls += 1
            elif is_stress:
                _surrogate_stress_calls += 1
            else:
                if _surrogate_calls >= SURROGATE_CALL_BUDGET:
                    cache_contains = getattr(_CTX.surrogate, "primary_cache_contains", None)
                    if callable(cache_contains):
                        try:
                            preflight_raw_hit = bool(
                                cache_contains(
                                    job_config=job_config,
                                    job_features=job_features,
                                    candidate_graph=_CTX.candidate_graph,
                                    method=selected_method,
                                    scenario=scenario,
                                )
                            )
                        except Exception:
                            preflight_raw_hit = False
                    if not preflight_raw_hit:
                        _surrogate_budget_rejections += 1
                        raise SurrogateBudgetExceeded(
                            f"surrogate-call budget {SURROGATE_CALL_BUDGET} reached this tick; "
                            "narrow to your best few candidate configs and reuse scored results."
                        )
                else:
                    _surrogate_calls += 1
                    charged_search = True
            if hasattr(_CTX.surrogate, "compose_prediction_with_trace"):
                # Point-in-time cutoff prevents later evidence leaking into replayed predictions.
                try:
                    y_hat, v_hat, prediction_lineage = _CTX.surrogate.compose_prediction_with_trace(
                        job_config=job_config,
                        job_features=job_features,
                        candidate_graph=_CTX.candidate_graph,
                        method=selected_method,
                        scenario=scenario,
                        as_of_timestamp_utc=time.time() if calibrate else None,
                    )
                except Exception:
                    _persist_surrogate_trace(getattr(_CTX.surrogate, "last_trace", None))
                    raise
                _persist_surrogate_trace(prediction_lineage)
            else:
                result = _CTX.surrogate.compose_prediction(
                    job_config=job_config,
                    job_features=job_features,
                    candidate_graph=_CTX.candidate_graph,
                    method=selected_method,
                    scenario=scenario,
                )
                prediction_lineage = None
                if isinstance(result, tuple) and len(result) == 2:
                    y_hat, v_hat = result
                else:
                    y_hat = getattr(result, "y_hat", {}) or {}
                    v_hat = getattr(result, "v_hat", {}) or {}
            if is_search and (preflight_raw_hit or _aic_raw_cache_hit(prediction_lineage)):
                if charged_search:
                    _surrogate_calls = max(0, _surrogate_calls - 1)
                _surrogate_raw_cache_hits += 1
            cached = copy.deepcopy(
                {
                    "y_hat": dict(y_hat or {}),
                    "v_hat": dict(v_hat or {}),
                    "prediction_lineage": copy.deepcopy(prediction_lineage),
                }
            )
            if key is not None:
                _prediction_cache[key] = copy.deepcopy(cached)

    cached = copy.deepcopy(cached)
    y_hat = dict(cached["y_hat"])
    v_hat = dict(cached["v_hat"])
    prediction_lineage = cached.get("prediction_lineage")
    y_hat_raw = ((prediction_lineage or {}).get("raw") or {}).get("y_hat") or dict(y_hat)
    offsets = ((prediction_lineage or {}).get("calibration") or {}).get("offsets_y") or {}
    lower_throughput = ((prediction_lineage or {}).get("fusion") or {}).get("lower_throughput")

    dro_band = _CTX.dro.compute_dro_band(y_hat or {})
    return {
        "y_hat": y_hat or {},
        "y_hat_raw": y_hat_raw,
        "calibration_offsets": offsets,
        "v_hat": v_hat or {},
        "prediction_lineage": prediction_lineage,
        "throughput_token_per_sec_lower": lower_throughput,
        "dro_band": dro_band,
    }


def _persist_surrogate_trace(trace: dict[str, Any] | None) -> None:
    """Send one actual composer execution to the runner's fail-open debug logger."""
    persist = getattr(getattr(_CTX, "trace_logger", None), "persist_surrogate_prediction", None)
    if not callable(persist) or not trace:
        return
    snapshot = getattr(_CTX, "cluster_snapshot", None)
    tick = getattr(snapshot, "tick", None)
    try:
        persist(trace, tick=tick)
    except Exception:
        log.exception("surrogate debug trace persistence failed")


def _attach_peak_multiturn_stress(action: dict[str, Any], job_features: dict[str, Any]) -> None:
    try:
        if float(job_features.get("multi_turn_ratio") or 0.0) <= 0.0:
            return
    except (TypeError, ValueError):
        return

    diagnostic: dict[str, Any] = {
        "p99_ttft_ms": None,
        "p99_tpot_ms": None,
        "throughput_token_per_sec": None,
        "completed_requests": None,
        "error": None,
    }
    try:
        samples: list[tuple[int, dict]] = []
        completed_requests = 0
        for raw_rank in action.get("ladder") or []:
            rank = RankSpec.from_dict(raw_rank)
            payload = _rank_prediction_payload(rank, job_features)
            pred = _predict_outcome_core(
                payload["job_config"],
                payload["job_features"],
                calibrate=False,
                scenario="peak_all_multiturn_stress",
                method=(
                    _AIC_DYNOSIM_METHOD
                    if _job_mode(job_features) == "online"
                    else _AIC_DIRECT_METHOD
                ),
            )
            y = dict(pred.get("y_hat_raw") or pred.get("y_hat") or {})
            v = dict(pred.get("v_hat") or {})
            replicas = max(1, int(rank.n_replicas or 1))
            if y:
                samples.append((replicas, y))
            completed = v.get("completed_requests") or v.get("expected_completed_requests")
            if completed is not None:
                completed_requests += replicas * int(completed)
        if samples:
            y_hat = _roll_up_ranks(samples)
            for key in ("p99_ttft_ms", "p99_tpot_ms", "throughput_token_per_sec"):
                diagnostic[key] = y_hat.get(key)
        diagnostic["completed_requests"] = completed_requests or None
    except Exception as exc:
        diagnostic["error"] = str(exc)

    action.setdefault("selection_diagnostics", {})["peak_all_multiturn_stress"] = diagnostic


def _decision_required_objectives(snapshot, action, job_features: dict[str, Any]) -> list[str]:
    """Hard objectives whose decision-time bands must be evaluated later."""
    required = {
        str(objective)
        for objective, threshold in _slo_thresholds_for(snapshot, action.job_id).items()
        if threshold is not None
    }
    try:
        target_tps = action.target_tps or required_throughput_enumerator(job_features)
        if float(target_tps or 0.0) > 0:
            required.add(_THROUGHPUT_OBJ)
    except (TypeError, ValueError):
        pass
    return sorted(required)


def stamp_plan_predictions(plan, cluster_snapshot=None):
    """Attach raw per-rank predictions to the plan that will be deployed."""
    typed = _as_plan(plan)
    snapshot = cluster_snapshot if cluster_snapshot is not None else _snapshot()
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        job_features = _job_features_for(snapshot, action.job_id)
        method = _AIC_DYNOSIM_METHOD if _job_mode(job_features) == "online" else _AIC_DIRECT_METHOD
        required_objectives = _decision_required_objectives(snapshot, action, job_features)
        partial_admission = None
        if action.admission_mode == "advisory" and action.served_fraction is not None:
            partial_admission = {
                "mode": "advisory",
                "requested_tps": action.target_tps,
                "admitted_tps": action.admitted_tps,
                "served_fraction": action.served_fraction,
                "enforced": False,
            }
        for rank in action.ladder:
            payload = _rank_prediction_payload(
                rank,
                job_features,
                arrival_rate_rps=_public_rank_arrival_rps(action, rank, job_features),
            )
            pred = _predict_outcome_core(
                payload["job_config"],
                payload["job_features"],
                calibrate=False,
                scenario="peak",
                method=method,
                _finalization=True,
            )
            rank.predicted_y = dict(pred.get("y_hat") or pred.get("y_hat_raw") or {})
            rank.predicted_v = dict(pred.get("v_hat") or {})
            lineage = compact_prediction_lineage(pred.get("prediction_lineage"))
            lineage["decision_dro_band"] = copy.deepcopy(pred.get("dro_band") or {})
            lineage["decision_required_objectives"] = list(required_objectives)
            lineage["deployment_id"] = (
                f"deploy:{typed.tick}:{action.job_id}:{rank.rank_id or 'rank'}"
            )
            lineage["evidence_baseline"] = "pre_calibration"
            if partial_admission is not None:
                lineage["partial_admission"] = copy.deepcopy(partial_admission)
            rank.prediction_lineage = lineage
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
        except (SurrogateMemoryNoFit, SurrogateUnsupportedConfig) as exc:
            log.warning("optimize_config: surrogate rejected config, skipping (%s)", exc)
            return float("-inf"), None
        except SurrogateExecutionError:
            raise
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


# Per-mode OPTIMIZE axis: which objective Koi pushes PAST its target once the target
# is met. Batch maximizes aggregate throughput; online minimizes latency (online
# throughput is pinned to demand, so there is nothing to maximize there). Cost is
# always optimized separately (the cost term), for both modes.
_BATCH_OPTIMIZE_AXES: tuple[str, ...] = (_THROUGHPUT_OBJ,)
_ONLINE_OPTIMIZE_AXES: tuple[str, ...] = ("p99_ttft_ms", "p99_tpot_ms")


def _job_mode(job_features: dict[str, Any] | None) -> str:
    """Normalized workload mode: 'batch' or 'online' (default)."""
    jf = job_features or {}
    mode = jf.get("type") or jf.get("workload_type") or jf.get("kind") or "online"
    return "batch" if str(mode).lower() == "batch" else "online"


def _optimize_axes_for_mode(job_mode: str) -> tuple[str, ...]:
    return _BATCH_OPTIMIZE_AXES if job_mode == "batch" else _ONLINE_OPTIMIZE_AXES


def _axis_headroom(y_value: float, target: float, axis: str) -> float:
    """Fraction by which y BEATS its target on `axis` (>0 beating, <=0 missing).

    Maximized axis (throughput): (y - target)/|target|.
    Minimized axis (latency):    (target - y)/|target|.
    """
    if not target:
        return 0.0
    beat = (float(y_value) - target) if axis == _THROUGHPUT_OBJ else (target - float(y_value))
    return beat / abs(target)


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
    return _compute_sigma(plan, finalization=False)


def compute_sigma_for_commit(plan) -> dict[str, Any]:
    """Score a materialized plan without consuming the search-call budget."""
    return _compute_sigma(plan, finalization=True)


def _compute_sigma(plan, finalization: bool) -> dict[str, Any]:
    """Shared plan scorer with an explicit surrogate-accounting mode."""
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
    served_fraction_by_job: dict[str, float] = {}
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        raw_fraction = getattr(action, "served_fraction", None)
        if raw_fraction is None:
            served_fraction_by_job[action.job_id] = 1.0
            continue
        if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, int | float):
            raise ValueError(
                f"job {action.job_id}: served_fraction must be a finite number in [0, 1]"
            )
        served_fraction = float(raw_fraction)
        if not math.isfinite(served_fraction) or not 0.0 <= served_fraction <= 1.0:
            raise ValueError(
                f"job {action.job_id}: served_fraction must be a finite number in [0, 1]"
            )
        served_fraction_by_job[action.job_id] = served_fraction

    # w_t (objective weights) is fetched PER JOB in the loop below - it is per-workload-
    # mode (batch favors throughput+cost, online favors latency), so it is NOT global.
    # z_star is fetched PER JOB in the loop; z*/ranges are seeded against domain
    # priors so an unseeded slow loop (z*=0, range=1.0) cannot collapse J to ~-50.
    ranges = _seeded_ranges(_CTX.slow_loop.typical_ranges)
    beta = _CTX.slow_loop.get_sss_eig_incentive_t()
    lam = _CTX.slow_loop.get_sss_lambda_switch()
    eps_dro = _CTX.slow_loop.get_sss_radius_dro()
    pricing_map = _switch_pricing_map()
    # Cost as a weighted OPTIMIZE objective: w_cost is the MARKET knob (0 reserved,
    # > 0 pay-per-use). Kept SECONDARY to the SLO by scoring the $/token gap ABOVE
    # the ideal as a FRACTION of the typical cost RANGE (like every other objective),
    # NOT the raw ratio cost/ideal (unbounded ~1-55x, which swamped the target-
    # relative J and made a cheap under-target frame beat a pricier one that MEETS
    # target). cost_ref = ideal $/token; cost_range = typical spread.
    # w_cost is per-job (per-mode weights) -> computed inside the loop alongside w_t.
    cost_ref = float(_seeded_z_star(get_z_star()).get("cost_per_token", 0.0) or 0.0)
    cost_range = float(ranges.get("cost_per_token", 0.0) or 0.0)

    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        job_id = action.job_id
        ladder_dicts = _ranks_as_dicts(action)
        job_features = _job_features_for(snapshot, job_id)
        # Per-workload-mode objective weights: batch favors throughput+cost, online
        # favors latency (get_sss_wt honors the mode; falls back to global if unset).
        job_mode = _job_mode(job_features)
        w_t = _CTX.slow_loop.get_sss_wt(job_type=job_mode)
        w_cost = float((w_t or {}).get("cost_per_token", 0.0) or 0.0)
        prediction_method = _AIC_DYNOSIM_METHOD if job_mode == "online" else _AIC_DIRECT_METHOD
        prediction_scenario = "peak" if job_mode == "online" else "mean"
        y_hat = _compose_job_y_hat(
            action,
            job_features,
            method=prediction_method,
            scenario=prediction_scenario,
            finalization=finalization,
        )
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
        prev_ladder = _prev_ladder_for(snapshot, job_id)
        switch_bundle = _CTX.switchcost_module.compute_switch_cost(
            L_prev=_materialize_chain_list(prev_ladder),
            L_new=_materialize_chain_list(ladder_dicts),
            residual_history=_CTX.dro,
            epsilon_dro=eps_dro,
            pricing_map=pricing_map,
            slo_thresholds=slo_thresholds,
            pred_y_new=y_hat,
        )
        # Churn cost applies only to CHANGING a running deployment. A fresh placement
        # (waiting -> running) has no prior ladder - nothing to churn - so its
        # size-proportional spin-up must NOT be charged against the serve decision, or
        # it dominates the (weak, target-relative) J. Swaps keep the real churn cost.
        switch_total = switch_bundle.total if prev_ladder else 0.0
        pr_slo = float(
            _CTX.dro.dro_chance_constraint(
                pred_y=y_hat,
                slo_thresholds=slo_thresholds,
            ).get("_any_violated", 0.0)
        )

        # Weighted cost objective (secondary to SLO): the $/token gap ABOVE the ideal
        # as a FRACTION of the typical cost range - bounded and on J's scale, so it
        # only ranks frames and can never flip a target-MEETING frame below an
        # under-target one. w_cost=0 (reserved) -> inert; w_cost>0 (pay-per-use) ->
        # cheaper-per-token frames score higher.
        cost_pred = _y_value(y_hat, "cost_per_token")
        raw_cost = (
            w_cost * max(0.0, float(cost_pred or 0.0) - cost_ref) / cost_range
            if (w_cost and cost_range > 0)
            else 0.0
        )
        # Soft-cap to COST_PENALTY_CAP: monotonic (cheaper-per-token still wins AMONG
        # meeters), ~= raw_cost when raw_cost << cap (a well-scaled range passes through
        # untouched), and saturates toward the cap as raw_cost grows - so however tight
        # a range the slow loop learns, cost can never outrank a meaningful SLO gap.
        cost_penalty = (
            COST_PENALTY_CAP * raw_cost / (raw_cost + COST_PENALTY_CAP) if raw_cost > 0.0 else 0.0
        )
        # Beyond-target OPTIMIZATION bonus (bounded secondary, like cost): once the
        # job MEETS its target (J saturates at 0), reward BEATING it on the mode axis -
        # batch maximizes throughput, online minimizes latency. One-sided (no credit
        # below target; the shortfall is already penalized in J) and soft-capped at
        # OPT_BONUS_CAP so it can NEVER outrank another job's target MISS: satisfice
        # every target first (fairness), then optimize the mode axis with leftover.
        opt_raw = 0.0
        if target_ref:
            for axis in _optimize_axes_for_mode(job_mode):
                tgt = target_ref.get(axis)
                yv = y_hat.get(axis)
                if tgt and yv is not None:
                    hr = _axis_headroom(float(yv), float(tgt), axis)
                    if hr > 0.0:
                        opt_raw += float((w_t or {}).get(axis, 0.0) or 0.0) * hr
        opt_bonus = OPT_BONUS_CAP * opt_raw / (opt_raw + OPT_BONUS_CAP) if opt_raw > 0.0 else 0.0
        # Net VALUE secondary = beating the mode target (opt_bonus) MINUS cost, FLOORED
        # at 0. A target-MEETING frame (J=0) must NEVER be pushed negative by the
        # cost/opt tradeoff: a forced-expensive meeter (e.g. the 72B, whose only fit is
        # 8xH100) would otherwise score cost_penalty > opt_bonus -> sigma < 0 -> read as
        # defer-worthy -> the whole plan collapsed to defer. Floored, cheaper+faster
        # frames still rank higher (they live in the positive band), while an
        # expensive-but-meeting frame bottoms out at 0 (met, no extra value) instead of
        # looking worse than defer. cost and opt_bonus stay in the diagnostics.
        value_bonus = max(0.0, opt_bonus - cost_penalty)
        sigma_i = J + value_bonus + beta * eig_value - GAMMA_SLO * pr_slo - lam * switch_total
        per_job[job_id] = {
            "J": J,
            "eig": eig_value,
            "switch_cost_total": switch_total,
            "pr_slo_dro": pr_slo,
            "cost_penalty": cost_penalty,
            "opt_bonus": opt_bonus,
            "value_bonus": value_bonus,
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
    priority_by_job = {
        p.get("job_id"): float(p.get("priority_score", 1.0) or 1.0) for p in get_priority()
    }
    unserved_penalty = 0.0
    for job in get_pending_jobs():
        jid = job.get("job_id", job.get("id"))
        if not jid:
            continue
        served_fraction = served_fraction_by_job.get(jid, 0.0)
        unserved_penalty += (
            UNSERVED_PENALTY * max(1.0, priority_by_job.get(jid, 1.0)) * (1.0 - served_fraction)
        )
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
    """Fill-tp for ONE instance: the largest power of 2 that divides `heads` and is
    <= cap (the instance's GPU count); 1 if heads unknown. Under instance-atomic
    accounting a rank reserves the WHOLE instance, so a smaller tp would just idle
    the rest of the box - use as many of its GPUs as can shard the model. Scaling
    THROUGHPUT past one instance comes from n_replicas and extra (heterogeneous)
    ranks sized by size_ladder, NOT from a smaller tp inside one box."""
    if not heads or int(heads) <= 0 or cap < 1:
        return 1
    tp, power = 1, 2
    while power <= cap and int(heads) % power == 0:
        tp, power = power, power * 2
    return tp


def _applicable_mechanism_id(rank: dict[str, Any], features: dict[str, Any]) -> str | None:
    """Best applicable mechanism id for a rank (exact, then partial), or None."""
    try:
        apps = get_applicable_mechanisms(rank, features)
    except Exception:
        return None
    if isinstance(apps, dict):
        mid = apps.get("exact") or apps.get("mechanism_id")
        if mid:
            return mid
        vals = apps.get("mechanisms") or apps.get("applicable") or []
        if vals:
            return vals[0] if isinstance(vals[0], str) else vals[0].get("mechanism_id")
    elif isinstance(apps, (list, tuple)) and apps:
        return apps[0] if isinstance(apps[0], str) else apps[0].get("mechanism_id")
    return None


def _online_slo_targets(features: dict[str, Any]) -> dict[str, Any]:
    """Online latency SLO targets to carry ON the emitted action (None for batch).
    compute_sigma reads these from job_features, but the deployed action must also
    carry them - Orca/Dynamo route on them - so we copy them onto every place act."""
    return {
        "target_p99_ttft_ms": _feature_value(features, "target_p99_ttft_ms", "target_p99_TTFT_ms"),
        "target_p99_tpot_ms": _feature_value(features, "target_p99_tpot_ms", "target_p99_TPOT_ms"),
    }


def _online_sizing_rejection(
    sized: dict[str, Any], features: dict[str, Any]
) -> tuple[str, str] | None:
    """Reject online service lacking complete, SLO-safe admission evidence."""
    if _job_mode(features) != "online":
        return None
    targets = _online_slo_targets(features)
    has_latency_target = any(value is not None for value in targets.values())
    per_rank = sized.get("per_rank") or []
    serving = [rank for rank in per_rank if int(rank.get("n_replicas") or 0) >= 1]
    checked_ranks = serving or per_rank
    if has_latency_target and any(
        rank.get("prediction_received") and rank.get("prediction_complete") is False
        for rank in checked_ranks
    ):
        return "prediction_incomplete", "surrogate prediction omitted a declared TTFT/TPOT SLO"
    if has_latency_target and any(
        rank.get("slo_ok") is False
        and (rank.get("prediction_received") or "under-SLO" in str(rank.get("reason") or ""))
        for rank in checked_ranks
    ):
        return "under_slo", "predicted TTFT/TPOT does not meet the declared online SLO"
    if sized.get("ranks") and not sized.get("meets_target"):
        try:
            achieved_tps = float(sized.get("achieved_tps") or 0.0)
        except (TypeError, ValueError, OverflowError):
            achieved_tps = 0.0
        if not math.isfinite(achieved_tps) or achieved_tps <= 0:
            return "no_fit", "online frame has no positive SLO-safe admitted throughput"
        try:
            safe_partial = bool(checked_ranks) and all(
                rank.get("prediction_received") is True
                and rank.get("prediction_complete") is True
                and rank.get("slo_ok") is True
                and (
                    rank.get("served_tps") is None
                    or (math.isfinite(float(rank["served_tps"])) and float(rank["served_tps"]) > 0)
                )
                for rank in checked_ranks
            )
        except (TypeError, ValueError, OverflowError):
            safe_partial = False
        guarded_partial = (
            has_latency_target
            and PARTIAL_ONLINE_ADMISSION_MODE == "advisory"
            and sized.get("partial_online_admission") is True
        )
        if guarded_partial and safe_partial:
            return None
        return "under_target", "online frame does not provide full-service throughput"
    return None


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
        rank.get("n_replicas", 1),
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
    mid = _applicable_mechanism_id(rank, features)
    if not mid:
        diag.update(status="no_mechanism", reason="no applicable mechanism")
        return {"candidate": None, "meets_target": False, "diag": diag}
    scored_rank = dict(rank)
    scored_rank["mechanism_id"] = mid
    try:
        sized = size_ladder([scored_rank], features)
    except SurrogateBudgetExceeded as exc:
        diag.update(status="budget_exhausted", reason=str(exc))
        return {"candidate": None, "meets_target": False, "diag": diag}
    except Exception as exc:
        diag.update(status="size_error", reason=f"size_ladder failed: {exc}")
        return {"candidate": None, "meets_target": False, "diag": diag}
    ranks = sized.get("ranks") or []
    meets = bool(sized.get("meets_target"))
    target_tps = float(sized.get("target_tps") or 0.0)
    achieved_tps = float(sized.get("achieved_tps") or 0.0)
    unmet_tps = max(0.0, target_tps - achieved_tps) if target_tps > 0 else 0.0
    served_fraction = min(1.0, achieved_tps / target_tps) if target_tps > 0 else 1.0
    diag.update(
        achieved_tps=achieved_tps,
        target_tps=target_tps,
        meets_target=meets,
        unmet_tps=unmet_tps,
        served_fraction=served_fraction,
        partial_search_probes=int(sized.get("partial_search_probes") or 0),
        partial_search_truncated=bool(sized.get("partial_search_truncated")),
    )
    online_rejection = _online_sizing_rejection(sized, features)
    if online_rejection is not None:
        status, reason = online_rejection
        diag.update(status=status, reason=reason)
        return {
            "candidate": None,
            "composite_eligible": status == "under_target",
            "meets_target": False,
            "diag": diag,
        }
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
        "target_tps": target_tps,
        "achieved_tps": achieved_tps,
        "unmet_tps": unmet_tps,
        "meets_target": meets,
        "served_fraction": served_fraction,
        "mechanism_id": mid,
        "budget_ref": slice_id,
        **_online_slo_targets(features),
        "rationale": f"Deterministic {gpu_type} candidate "
        f"({'full-service' if meets else 'under-target'}).",
    }
    partial_online_candidate = sized.get("partial_online_admission") is True
    if partial_online_candidate:
        act["admitted_tps"] = achieved_tps
        act["admission_mode"] = "advisory"
    one = {"tick_rationale": "candidate scoring", "actions": [act]}
    try:
        feas = check_feasibility(one)
        if not feas.get("feasible"):
            diag.update(status="infeasible", reason="; ".join(feas.get("violations", []))[:200])
            return {"candidate": None, "meets_target": meets, "diag": diag}
        score = compute_sigma(one)["per_job"][jid]
        act["sigma"] = score["sigma"]
    except SurrogateBudgetExceeded as exc:
        diag.update(status="budget_exhausted", reason=str(exc))
        return {"candidate": None, "meets_target": meets, "diag": diag}
    except Exception as exc:
        diag.update(status="score_error", reason=f"scoring failed: {exc}")
        return {"candidate": None, "meets_target": meets, "diag": diag}
    diag.update(status="ok", **score)
    return {"candidate": act, "meets_target": meets, "diag": diag}


def _score_composite(
    jid: str, user_id: Any, slice_id: Any, ranks: list[dict[str, Any]], features: dict[str, Any]
) -> dict[str, Any]:
    """Score ONE heterogeneous, data-parallel multi-rank ladder for a job.

    size_ladder fills the given ranks in order, each covering the REMAINING
    throughput target and capped by its own pool's free capacity, so achieved SUMS
    across pools - the way a big job reaches a target NO single pool can (e.g. all
    H100 across p5.48xlarge + p5.4xlarge, or an H100 rank plus an A100 rank). The
    caller orders the ranks best-pool-first. ALWAYS returns a {candidate,
    meets_target, diag} dict (same shape as _score_one_frame); candidate is None -
    with a diagnostic status/reason - when the ladder can't be built or scored, so
    the attempt is always visible in diagnostics (never a silent drop)."""
    diag: dict[str, Any] = {
        "env": "composite",
        "instance_type": None,
        "tp": None,
        "status": None,
        "reason": None,
        "meets_target": False,
        "achieved_tps": None,
        "target_tps": None,
        "sigma": None,
    }
    scored_ranks: list[dict[str, Any]] = []
    for rank in ranks:
        runnable, _ = config_runnable(dict(rank.get("config") or {}), features)
        if not runnable:
            continue
        mid = _applicable_mechanism_id(rank, features)
        if not mid:
            continue
        sr = dict(rank)
        sr["mechanism_id"] = mid
        scored_ranks.append(sr)
    if len(scored_ranks) < 2:
        diag.update(
            status="no_composite", reason=f"only {len(scored_ranks)} runnable rank(s) to combine"
        )
        return {"candidate": None, "meets_target": False, "diag": diag}
    try:
        sized = size_ladder(scored_ranks, features)
    except SurrogateBudgetExceeded as exc:
        diag.update(status="budget_exhausted", reason=str(exc))
        return {"candidate": None, "meets_target": False, "diag": diag}
    except Exception as exc:
        diag.update(status="size_error", reason=f"size_ladder failed: {exc}")
        return {"candidate": None, "meets_target": False, "diag": diag}
    sized_ranks = sized.get("ranks") or []
    meets = bool(sized.get("meets_target"))
    target_tps = float(sized.get("target_tps") or 0.0)
    achieved_tps = float(sized.get("achieved_tps") or 0.0)
    unmet_tps = max(0.0, target_tps - achieved_tps) if target_tps > 0 else 0.0
    served_fraction = min(1.0, achieved_tps / target_tps) if target_tps > 0 else 1.0
    label = "+".join(str((r.get("config") or {}).get("instance_type")) for r in sized_ranks) or None
    diag.update(
        instance_type=label,
        meets_target=meets,
        achieved_tps=achieved_tps,
        target_tps=target_tps,
        unmet_tps=unmet_tps,
        served_fraction=served_fraction,
        partial_search_probes=int(sized.get("partial_search_probes") or 0),
        partial_search_truncated=bool(sized.get("partial_search_truncated")),
    )
    online_rejection = _online_sizing_rejection(sized, features)
    if online_rejection is not None:
        status, reason = online_rejection
        diag.update(status=status, reason=reason)
        return {"candidate": None, "meets_target": False, "diag": diag}
    if len(sized_ranks) < 2:
        # size_ladder covered the target (or ran out) on ONE rank - no composite;
        # the single-frame candidate already represents it.
        diag.update(
            status="no_composite",
            reason=f"size_ladder used {len(sized_ranks)} rank(s) of {len(scored_ranks)}",
        )
        return {"candidate": None, "meets_target": meets, "diag": diag}
    act = {
        "job_id": jid,
        "type": "place",
        "user_id": user_id,
        "ladder": sized_ranks,
        "target_tps": target_tps,
        "achieved_tps": achieved_tps,
        "unmet_tps": unmet_tps,
        "meets_target": meets,
        "served_fraction": served_fraction,
        "mechanism_id": sized_ranks[0].get("mechanism_id"),
        "budget_ref": slice_id,
        **_online_slo_targets(features),
        "rationale": f"Deterministic composite candidate ({label}) "
        f"({'full-service' if meets else 'under-target'}).",
    }
    partial_online_candidate = sized.get("partial_online_admission") is True
    if partial_online_candidate:
        act["admitted_tps"] = achieved_tps
        act["admission_mode"] = "advisory"
    one = {"tick_rationale": "candidate scoring", "actions": [act]}
    try:
        feas = check_feasibility(one)
        if not feas.get("feasible"):
            diag.update(status="infeasible", reason="; ".join(feas.get("violations", []))[:200])
            return {"candidate": None, "meets_target": meets, "diag": diag}
        score = compute_sigma(one)["per_job"][jid]
        act["sigma"] = score["sigma"]
    except SurrogateBudgetExceeded as exc:
        diag.update(status="budget_exhausted", reason=str(exc))
        return {"candidate": None, "meets_target": meets, "diag": diag}
    except Exception as exc:
        diag.update(status="score_error", reason=f"scoring failed: {exc}")
        return {"candidate": None, "meets_target": meets, "diag": diag}
    diag.update(status="ok", **score)
    return {"candidate": act, "meets_target": meets, "diag": diag}


def _budget_skipped_frame(rank: dict[str, Any]) -> dict[str, Any]:
    """Diagnostic placeholder for a frame not attempted after cap exhaustion."""
    env = rank.get("env")
    cfg = rank.get("config") or {}
    return {
        "candidate": None,
        "meets_target": False,
        "diag": {
            "env": env_gpu_type(env) if env else None,
            "instance_type": cfg.get("instance_type"),
            "tp": cfg.get("tp"),
            "status": "budget_skipped",
            "reason": "not attempted after surrogate search budget exhaustion",
            "meets_target": False,
            "achieved_tps": None,
            "target_tps": None,
            "sigma": None,
        },
    }


def build_scored_candidates(
    budget_book: dict[str, Any] | None = None,
    specialist_results: Any = None,
) -> dict[str, Any]:
    """Deterministic candidate pipeline for all waiting jobs: normalize specialist
    ladders (HINTS), then generate the right-sized menu - for every (gpu_type,
    instance_type) with free capacity, ONE frame at fill-tp (the largest power of 2
    that shards the model's heads and fits the instance's GPUs). Resource accounting
    is instance-atomic (a rank reserves the whole instance), so a partial tp would
    just idle the box - fill it, and scale throughput with n_replicas, not smaller
    tp. Size and score each via the proven chain (config_runnable ->
    get_applicable_mechanisms -> size_ladder -> check_feasibility -> compute_sigma).
    Frames the model can't fit fall out when the surrogate rejects them.

    Then, for any job whose target NO single pool can reach (each single frame is
    under-target), add ONE heterogeneous COMPOSITE candidate: a data-parallel ladder
    spanning pools (best-throughput pool first) that size_ladder fills so achieved
    SUMS across ranks. This is how a big job that exceeds one pool's capacity gets
    served, and it lets the joint solver pick a heterogeneous placement (e.g. spill
    part of a job to A100 so another job keeps the H100) when that is the best
    cluster outcome.

    EVERY physically-runnable, feasible batch frame is returned as a candidate.
    Online jobs that declare latency SLOs require complete SLO-meeting predictions;
    under-target online candidates are emitted only in advisory mode after guarded
    load probing. A job appears in `exhausted` only when it has NO runnable, feasible
    frame; budget-truncated jobs are reported separately.

    Returns {"candidates": [...for jointly_select_placements],
             "exhausted": {job_id: reason},
             "budget_limited": {job_id: reason},
             "diagnostics": {job_id: [per-frame diag incl. meets_target/achieved_tps]}}.
    """
    _require("resource_map", "surrogate")
    cache_key = _tick_cache_key(budget_book, specialist_results)
    if cache_key is not None and cache_key in _scored_candidates_cache:
        return copy.deepcopy(_scored_candidates_cache[cache_key])
    snapshot = _snapshot()
    specs = instance_catalog()
    free_envs: list[tuple[str, list[str]]] = []
    for raw_env_key, info in sorted(get_resource_map().items(), key=lambda item: _env_key(item[0])):
        try:
            if int(info.get("free", 0) or 0) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        env_key = _env_key(raw_env_key)
        free_envs.append((env_key, env_key.split("|")))

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
    # specialist ladders (hints) plus the full generated menu: one frame per
    # (gpu_type, instance_type, tp width) with free capacity. Small instances give
    # small-tp frames (right for small models), big instances give the tp ladder
    # (tp2*DP4 .. tp8*DP1) for big ones; the solver picks the best-sigma survivor.
    frames_by_job: dict[str, list[dict[str, Any]]] = {}
    ctx_by_job: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    pending_jobs = sorted(
        get_pending_jobs(), key=lambda job: str(job.get("job_id", job.get("id", "")))
    )
    for job in pending_jobs:
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
        # For EVERY (gpu_type, instance_type) with free capacity, one fill-tp frame:
        # a 1-GPU box -> tp=1 (right for small models), an 8-GPU box -> tp=8 (right
        # for big ones). Accounting is instance-atomic, so a partial tp just idles
        # the box - fill it; size_ladder scales n_replicas across instances for
        # throughput, and Phase 2.5 spans pools when one is not enough. Specialist
        # ladders above stay as hints (deduped by shape).
        for env_key, env in free_envs:
            for instance_type, spec in sorted((specs.get(env_key) or {}).items()):
                gpi = int(spec.get("gpus_per_instance", 0) or 0)
                if gpi <= 0 or int(spec.get("free_instances", 0) or 0) <= 0:
                    continue
                tp = _largest_pow2_divisor_leq(heads, gpi)
                rank = _normalize_candidate_rank(
                    {
                        "role": "aggregate",
                        "env": list(env),
                        "config": {
                            "instance_type": instance_type,
                            "gpu_count": tp,
                            "tp": tp,
                            "pp": 1,
                        },
                        "n_replicas": 1,
                    }
                )
                if rank is not None and _rank_shape_key(rank) not in seen:
                    seen.add(_rank_shape_key(rank))
                    frames.append(rank)
        frames_by_job[jid] = frames
        ctx_by_job[jid] = (user_id, slice_id, features)

    # Phase 2 (surrogate-heavy): deterministic round-robin over jobs. The surrogate
    # is globally serialized, so threads add no throughput and obscure which jobs
    # received the final calls. Specialist frame 0 leads for every job before any
    # job receives frame 1.
    scored_by_job: dict[str, list[dict[str, Any]]] = {jid: [] for jid in frames_by_job}
    budget_exhausted = False
    max_frames = max((len(frames) for frames in frames_by_job.values()), default=0)
    for frame_index in range(max_frames):
        for jid, frames in frames_by_job.items():
            if frame_index >= len(frames):
                continue
            user_id, slice_id, features = ctx_by_job[jid]
            scored = _score_one_frame(jid, user_id, slice_id, frames[frame_index], features)
            scored_by_job[jid].append(scored)
            if scored.get("diag", {}).get("status") == "budget_exhausted":
                budget_exhausted = True
                break
        if budget_exhausted:
            break

    if budget_exhausted:
        for jid, frames in frames_by_job.items():
            results = scored_by_job[jid]
            for rank in frames[len(results) :]:
                results.append(_budget_skipped_frame(rank))

    # Phase 2.5 (heterogeneous composites). Two motivations:
    #  - CAPACITY: if NO single pool meets the target, span pools (fill the
    #    highest-throughput ones first) so a big job is served across pools.
    #  - COST (on-demand market only, w_cost>0): ALSO try a cheapest-$/token-first
    #    mix even when a single pool already meets, so a heterogeneous placement
    #    (cheap pool + top-up) can beat the single-pool winner on cost. In reserved
    #    (w_cost=0) the fleet is sunk - a "cheaper" mix saves nothing - so we only do
    #    the capacity fallback.
    # size_ladder fills each ordering and sums achieved; the joint solver ranks EVERY
    # candidate by sigma (which includes the bounded cost term), so if no mix is both
    # cheaper AND SLO-meeting, the single pool still wins. A cheapest-first ordering
    # whose cheapest pool already meets alone collapses to <2 ranks -> no composite.
    # scored_by_job[jid] aligns with frames_by_job[jid].
    try:
        w_cost = float((_CTX.slow_loop.get_sss_wt() or {}).get("cost_per_token", 0.0) or 0.0)
    except Exception:
        w_cost = 0.0
    if not budget_exhausted:
        for jid, frames in frames_by_job.items():
            results = scored_by_job.get(jid) or []
            composable = [
                (f, r)
                for f, r in zip(frames, results, strict=False)
                if r.get("candidate") is not None or r.get("composite_eligible")
            ]
            if not composable:
                continue
            user_id, slice_id, features = ctx_by_job[jid]
            single_meets = any(r.get("meets_target") for _, r in composable)
            orders: list[list[dict[str, Any]]] = []
            if not single_meets:
                orders.append(
                    [
                        f
                        for f, _ in sorted(
                            composable,
                            key=lambda fr: fr[1]["diag"].get("achieved_tps") or 0.0,
                            reverse=True,
                        )
                    ]
                )
            if w_cost > 0:
                orders.append(
                    [
                        f
                        for f, _ in sorted(
                            composable, key=lambda fr: fr[1]["diag"].get("cost_penalty") or 0.0
                        )
                    ]
                )
            for order in orders:
                composite = _score_composite(jid, user_id, slice_id, order, features)
                scored_by_job[jid].append(composite)
                if composite.get("diag", {}).get("status") == "budget_exhausted":
                    budget_exhausted = True
                    break
            if budget_exhausted:
                break

    if budget_exhausted:
        for _jid, scored_list in scored_by_job.items():
            composable = [
                item
                for item in scored_list
                if item.get("candidate") is not None or item.get("composite_eligible")
            ]
            single_meets = any(item.get("meets_target") for item in composable)
            composite_was_relevant = len(composable) >= 2 and (not single_meets or w_cost > 0)
            has_composite_result = any(
                item.get("diag", {}).get("env") == "composite" for item in scored_list
            )
            has_budget_status = any(
                item.get("diag", {}).get("status") in {"budget_exhausted", "budget_skipped"}
                for item in scored_list
            )
            if composite_was_relevant and not has_composite_result and not has_budget_status:
                scored_list.append(
                    {
                        "candidate": None,
                        "meets_target": False,
                        "diag": {
                            "env": "composite",
                            "instance_type": None,
                            "tp": None,
                            "status": "budget_skipped",
                            "reason": (
                                "composite not attempted after surrogate search budget exhaustion"
                            ),
                            "meets_target": False,
                            "achieved_tps": None,
                            "target_tps": None,
                            "sigma": None,
                        },
                    }
                )

    # Phase 3 (cheap): group into candidates / exhausted / diagnostics. Batch may
    # retain under-target frames; online partials pass only through the advisory
    # safety gate above. Budget-truncated jobs never become exhausted.
    candidates: list[dict[str, Any]] = []
    exhausted: dict[str, str] = {}
    budget_limited: dict[str, str] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for jid, scored_list in scored_by_job.items():
        job_diag = [s["diag"] for s in scored_list]
        job_candidates = [s["candidate"] for s in scored_list if s["candidate"] is not None]
        diagnostics[jid] = job_diag
        limited = [
            diag
            for diag in job_diag
            if diag.get("status") in {"budget_exhausted", "budget_skipped"}
        ]
        if limited:
            reasons = [diag["reason"] for diag in limited if diag.get("reason")]
            budget_limited[jid] = (
                "; ".join(dict.fromkeys(reasons))
                if reasons
                else "surrogate search budget exhausted"
            )
        if job_candidates:
            candidates.extend(job_candidates)
        elif not limited:
            reasons = [d["reason"] for d in job_diag if d.get("reason")]
            exhausted[jid] = (
                "; ".join(dict.fromkeys(reasons)) if reasons else "no runnable, feasible frame"
            )

    result = {
        "candidates": candidates,
        "exhausted": exhausted,
        "budget_limited": budget_limited,
        "diagnostics": diagnostics,
    }
    if cache_key is not None:
        _scored_candidates_cache[cache_key] = copy.deepcopy(result)
        return copy.deepcopy(_scored_candidates_cache[cache_key])
    return copy.deepcopy(result)


def jointly_select_placements(
    candidates: list[dict[str, Any]],
    reserves: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Joint GPU selection across ALL waiting jobs against the one shared pool.

    Chooses at most one candidate frame per job (or defers it) to MAXIMIZE the
    cluster objective - the sum of placed per-job sigma plus avoided unserved
    demand, credited by served_fraction for under-target frames - subject to
    per-env free-GPU capacity. This is the joint decision the greedy per-job loop cannot make: it
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
               "served_fraction": float, # 0..1 target throughput covered
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

    def candidate_served_fraction(cand: dict[str, Any]) -> float | None:
        def finite_number(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            number = float(value)
            return number if math.isfinite(number) else None

        accounting_fields = {
            "target_tps",
            "admitted_tps",
            "achieved_tps",
            "unmet_tps",
            "meets_target",
            "served_fraction",
            "admission_mode",
        }
        if not any(field in cand for field in accounting_fields):
            return 1.0
        if not all(field in cand for field in ("target_tps", "achieved_tps", "served_fraction")):
            return None

        target = finite_number(cand["target_tps"])
        achieved = finite_number(cand["achieved_tps"])
        fraction = finite_number(cand["served_fraction"])
        if target is None or target <= 0.0 or achieved is None or achieved <= 0.0:
            return None
        if fraction is None or fraction <= 0.0 or fraction > 1.0:
            return None
        if achieved > target and not math.isclose(
            achieved,
            target,
            rel_tol=1e-3,
            abs_tol=1e-6,
        ):
            return None
        derived = min(1.0, achieved / target)
        if not math.isclose(fraction, derived, rel_tol=1e-3, abs_tol=1e-6):
            return None

        if "admitted_tps" in cand:
            admitted = finite_number(cand["admitted_tps"])
            if admitted is None or admitted <= 0.0:
                return None
            if not math.isclose(admitted, achieved, rel_tol=1e-3, abs_tol=1e-6):
                return None
        return fraction

    # Group scored candidates by job, attaching each frame's per-env GPU cost and
    # its GAIN over deferring that job. Under-target frames only avoid the defer
    # penalty in proportion to delivered throughput; a 50%-served frame is not the
    # same as a full-service placement.
    by_job: dict[str, list[dict[str, Any]]] = {}
    for cand in candidates or []:
        jid = cand.get("job_id")
        if not jid:
            continue
        cost = _ladder_capacity_cost(cand.get("ladder") or [], specs)
        if not cost:
            continue  # no real GPU footprint -> not a placeable frame
        served_fraction = candidate_served_fraction(cand)
        if served_fraction is None:
            continue
        served_credit = penalty(jid) * served_fraction
        gain = float(cand.get("sigma", 0.0)) + served_credit
        if gain <= 0:
            continue
        cand["served_fraction"] = served_fraction
        cand["served_credit"] = served_credit
        cand["solver_gain"] = gain
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


def plan_tick() -> dict[str, Any]:
    """Run the full deterministic pipeline as a one-call planning recommendation.

    The root may commit this result directly or make a reasoned, feasible adjustment.
    This shortcut must not be combined with a second execution of the same pipeline in
    one tick. Negative sigma is expected and is not, by itself, a reason to defer.

    Sequence is byte-for-byte the documented MANDATORY ORDER:
      build_user_envelopes -> get_priority -> allocate_budget_book ->
      validate_budget_book -> run_job_specialists -> build_scored_candidates ->
      jointly_select_placements -> plan from chosen (feasibility-checked).

    Jobs the solver leaves out (``deferred`` / ``exhausted``) are omitted; the harness
    auto-defers omitted waiting jobs. Returns ``{"tick_rationale", "actions"}``.
    """
    build_user_envelopes()
    get_priority()
    budget_book = allocate_budget_book()
    validation = validate_budget_book(budget_book)
    if not validation.get("ok", False):
        return {
            "tick_rationale": (
                "plan_tick: budget book failed validation "
                f"({validation.get('violations')}); no placements this tick."
            ),
            "actions": [],
        }
    specialist_results = run_job_specialists()
    scored = build_scored_candidates(budget_book, specialist_results)
    joint = jointly_select_placements(scored.get("candidates", []) or [])
    chosen = list(joint.get("chosen", []) or [])
    deferred = list(joint.get("deferred", []) or [])
    exhausted = scored.get("exhausted", {}) or {}
    budget_limited = scored.get("budget_limited", {}) or {}
    plan: dict[str, Any] = {
        "tick_rationale": (
            f"plan_tick: joint solver placed {len(chosen)} job(s) "
            f"(objective={joint.get('objective')}); deferred={deferred}; "
            f"candidate_exhausted={list(exhausted.keys())}; "
            f"budget_limited={list(budget_limited.keys())}. Recommended by "
            "jointly_select_placements; negative sigma is normal, NOT a defer reason."
        ),
        "actions": chosen,
    }
    feas = check_feasibility(plan)
    if not feas.get("feasible", False):
        # Surface the violation in the rationale but keep the solver's placements:
        # the solver already enforced capacity, so a violation here is a config/
        # physical-shard issue worth seeing, not a reason to silently defer all.
        plan["tick_rationale"] += (
            f" WARNING: feasibility violations on chosen plan: {feas.get('violations')}."
        )
    return plan


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
