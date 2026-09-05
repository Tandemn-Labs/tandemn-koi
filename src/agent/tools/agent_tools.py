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
        size_ladder                 evaluate fixed-rank Direct point capacity

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

from src.config import ablation
from src.config.hyperparameters import GAMMA_SLO
from src.config.policy import MODEL_MAX_TP, SUPPORTED_EP, VALID_TP_DEGREES, load_config_policy
from src.core.models import (
    LADDER_ACTIONS,
    SWAP_BUDGET_ACTIONS,
    ActionType,
    Plan,
    PlanAction,
    RankSpec,
    deployment_ladder_identity,
    deployment_rank_identity,
    env_gpu_type,
)
from src.infra.deployment_x import (
    build_rank_x,
    hardware_gpu_memory_gb,
    materialize_launch_config,
)
from src.prediction.analytic_v import target_memory_fit
from src.prediction.composer import compact_prediction_lineage
from src.prediction.queue_model import estimate_queue_shadow
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
_WORK_CONSERVING_GAIN_FLOOR = 1e-9
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
        # Frozen learning keeps the surrogate evidence-blind: binding the store
        # here would silently re-enable calibration/fusion on every S4 rebind.
        and not ablation.learning_frozen()
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
PARTIAL_ONLINE_ADMISSION_MODE = "advisory"
_AIC_DIRECT_METHOD = ("AIC_Direct",)
_HARD_SIZING_FAILURES = frozenset(
    {"invalid_config", "no_pool_capacity", "physical_no_fit", "resource_budget"}
)
_SOFT_PREDICTION_FAILURES = frozenset(
    {
        "unsupported_prediction",
        "prediction_failed",
        "prediction_empty",
        "prediction_incomplete",
        "zero_predicted_capacity",
        "demand_unmodeled",
    }
)
_DIRECT_PREDICTION_SEMANTICS = {
    "basis": "aic_direct_point",
    "throughput_token_per_sec": "point_capacity",
    "p99_ttft_ms": "base_service_latency",
    "p99_tpot_ms": "base_service_latency",
    "slo_margin": "base_service_latency_margin",
    "queue_model": "none",
    "queue_slo_verified": False,
}
_POINT_PREDICTION_BASES = frozenset({"aic_direct_point", "composed_point_estimate"})


def _is_exploratory_assessment(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("basis") in _POINT_PREDICTION_BASES
        and value.get("kind") == "exploratory"
        and value.get("status") in _SOFT_PREDICTION_FAILURES
    )


_surrogate_calls = 0
_surrogate_cache_hits = 0
_surrogate_budget_rejections = 0
_surrogate_finalization_calls = 0
_surrogate_stress_calls = 0
_partial_online_searches = 0
_partial_online_queue_aware_probes = 0
_partial_online_safe_probes = 0
_partial_online_admissions = 0
_partial_online_truncated_searches = 0
_placement_decision_sequence = 0
# Per-tick memo of RAW surrogate output keyed on (job_config, job_features,
# scenario, calibration, method, accounting mode). Direct AIC is deterministic,
# so re-probing a config the LLM already evaluated THIS tick returns identical
# numbers. Serve them from here instead of re-running the surrogate. Access happens under
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
    """Retain the legacy admission-mode setting for compatibility and reporting."""
    if not isinstance(mode, str):
        raise TypeError("partial online admission mode must be a string")
    if mode not in {"off", "advisory"}:
        raise ValueError("partial online admission mode must be exactly 'off' or 'advisory'")
    global PARTIAL_ONLINE_ADMISSION_MODE
    with _SURROGATE_EXECUTION_LOCK:
        PARTIAL_ONLINE_ADMISSION_MODE = mode
    _scored_candidates_cache.clear()


def get_partial_online_admission_status() -> dict[str, Any]:
    """Return point-capacity admission accounting and legacy zeroed counters."""
    with _SURROGATE_EXECUTION_LOCK:
        return {
            "mode": PARTIAL_ONLINE_ADMISSION_MODE,
            "prediction_basis": "aic_direct_point",
            "queue_model": "none",
            "queue_slo_verified": False,
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
    global _surrogate_finalization_calls, _surrogate_stress_calls
    global _partial_online_admissions, _partial_online_queue_aware_probes
    global _partial_online_safe_probes, _partial_online_searches
    global _partial_online_truncated_searches
    global _placement_decision_sequence
    _CTX.user_envelopes = None
    _CTX.validated_budget_book = None
    with _SURROGATE_EXECUTION_LOCK:
        _surrogate_calls = 0
        _surrogate_cache_hits = 0
        _surrogate_budget_rejections = 0
        _surrogate_finalization_calls = 0
        _surrogate_stress_calls = 0
        _partial_online_searches = 0
        _partial_online_queue_aware_probes = 0
        _partial_online_safe_probes = 0
        _partial_online_admissions = 0
        _partial_online_truncated_searches = 0
        _placement_decision_sequence = 0
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

# Tools withheld from the LLM when the causal DAG is ablated (mechanism-mode
# inert): everything that reads or writes mechanisms, edges, or confidence.
_MECHANISM_INERT_HIDDEN_TOOLS = frozenset(
    {
        "get_edge_confidence",
        "get_mechanism_confidence",
        "get_influencing_knobs",
        "get_scope",
        "get_applicable_mechanisms",
        "get_recent_q_histogram",
        "compute_eig",
        "check_past_failure",
        "set_new_mechanisms",
        "val_new_mechanisms",
    }
)

# Tools withheld when learning is frozen: admitting a mechanism mid-run grows
# the causal model from observations, which is exactly what "frozen" forbids.
_LEARNING_FROZEN_HIDDEN_TOOLS = frozenset({"set_new_mechanisms", "val_new_mechanisms"})


def all_callables() -> dict[str, Any]:
    """Return every public LLM tool as a name -> callable dict.

    The harness binds these into the root REPL namespace in one shot.
    The __module__ filter drops imported callables (e.g. the Plan class)
    so only tool functions defined here are exposed; _NON_TOOL_NAMES drops
    the boot/infra functions (notably reset_tick_caches, which the model
    must never call mid-trajectory). Ablation modes additionally hide the
    tool families their disabled subsystem would have served.
    """
    hidden = set(_NON_TOOL_NAMES)
    if ablation.mechanism_inert():
        hidden |= _MECHANISM_INERT_HIDDEN_TOOLS
    if ablation.learning_frozen():
        hidden |= _LEARNING_FROZEN_HIDDEN_TOOLS
    tools = {
        name: fn
        for name, fn in globals().items()
        if callable(fn)
        and not name.startswith("_")
        and name not in hidden
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
        if raw["mechanism_id"] is None and ablation.mechanism_inert():
            raw["mechanism_id"] = ablation.passthrough_mechanism_id()
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


def _active_health_for(snapshot, job_id: str) -> dict[str, Any]:
    """Return deterministic health evidence for an active job, when available."""
    if snapshot is None or not hasattr(snapshot, "active_jobs_summary"):
        return {}
    for job in snapshot.active_jobs_summary() or []:
        if job.get("job_id", job.get("id")) == job_id:
            return dict(job.get("health") or {})
    return {}


def _rank_prediction_payload(
    rank: RankSpec,
    job_features: dict[str, Any] | None = None,
    *,
    arrival_rate_rps: float | None = None,
) -> dict:
    """Build a fixed-frame Direct payload without synthetic offered-load overrides."""
    features = _sanitize_agent_features(dict(job_features or {}))
    # Kept in the signature for compatibility with older callers. Direct is a point
    # estimator, so changing this value must not be used to synthesize queue behavior.
    _ = arrival_rate_rps
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
            model_catalog = resource_map.model_catalog(str(model_id))
            compiled_x = build_rank_x(
                job_values=features,
                shape=shape,
                env=(str(env[0]), str(env[1]), str(env[2]), str(env[3]), str(env[4])),
                resources=resources,
                hardware_catalog=resource_map.hardware_catalog(),
                model_catalog=model_catalog,
                replica_count=max(1, int(rank.n_replicas or 1)),
            )
            config.update(compiled_x)
            config.update(materialize_launch_config(model_catalog, str(env[4])))
            allocation = _rank_allocation_summary(rank, resources)
            price = allocation.get("price_per_unit_hour")
            if price is not None:
                config["price_per_hour"] = float(price)
        except Exception:
            log.exception("rank prediction X assembly failed; using rank config only")
    config.pop("_arrival_share_rps", None)
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
    homogeneous ladders are unaffected. The method input is retained for
    compatibility; production prediction is always Direct AIC.
    """
    method = _AIC_DIRECT_METHOD
    # TODO - I can debate this as we don't need the LLM to pass the predicted_y
    # we want it to CALL The SUrrogate ALWAYS
    # so i am, for now, removing this call.
    # if action.predicted_y:
    #     return dict(action.predicted_y)
    samples: list[tuple[int, dict]] = []
    for rank in action.ladder or []:
        try:
            payload = _rank_prediction_payload(rank, job_features)
            y = _predict_outcome_core(
                payload["job_config"],
                payload["job_features"],
                scenario=scenario,
                method=method,
                _finalization=finalization,
            ).get("y_hat", {})
        except SurrogateBudgetExceeded:
            raise
        except SurrogateMemoryNoFit:
            raise
        except SurrogateUnsupportedConfig as exc:
            log.warning("rank y_hat rejected for job %s (%s)", action.job_id, exc)
            y = {}
        except SurrogateExecutionError:
            raise
        except Exception:
            log.exception("rank y_hat failed for job %s", action.job_id)
            raise
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
        if not ablation.mechanism_inert():
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
    mechanisms = [] if ablation.mechanism_inert() else get_scope(mechanism_context)
    available_instances = instance_catalog()
    hardware = sorted(
        {
            str(info.get("gpu_type") or str(env).split("|")[-1])
            for env, pools in available_instances.items()
            for info in pools.values()
        }
    )
    policy_rules = load_config_policy().rules_for_model(str(model_id or ""), hardware)

    brief = {
        "job_id": job_id,
        "user_id": (descriptor or {}).get("user_id"),
        "job_features": features,
        "model_catalog": model_catalog,
        "current_ladder": (descriptor or {}).get("current_ladder"),
        "recent_q_labels": recent_q,
        "recent_theory_blobs": blobs,
        "similar_deployments": get_similar_deployments(features, top_k=5),
        "mechanism_candidates": mechanisms,
        "placement_policy": {
            gpu: {"precision": rule.precision, "allowed_tp": list(rule.allowed_tp)}
            for gpu, rule in policy_rules.items()
        },
        "instance_catalog": available_instances,
    }
    if ablation.mechanism_inert():
        # The specialist prompt embeds this brief verbatim; an inert-DAG run
        # must not surface mechanism vocabulary to the LLM at all.
        del brief["mechanism_candidates"]
        del brief["recent_q_labels"]
    return brief


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
            allocation_kind = str(
                pool.get("allocation_kind") or pool.get("allocation_unit") or "instance"
            ).lower()
            raw_gpus_per_unit = int(pool.get("gpus_per_instance") or pool.get("gpus_per_unit") or 1)
            free_gpus = int(pool.get("free", 0) or 0)
            available_units = (
                free_gpus if allocation_kind == "gpu" else int(pool.get("free_instances", 0) or 0)
            )
            env_map[str(instance_type)] = {
                "gpus_per_instance": (1 if allocation_kind == "gpu" else raw_gpus_per_unit),
                "gpus_per_unit": 1 if allocation_kind == "gpu" else raw_gpus_per_unit,
                "candidate_gpu_cap": (
                    available_units if allocation_kind == "gpu" else raw_gpus_per_unit
                ),
                "free_instances": available_units,
                "available_units": available_units,
                "allocation_kind": allocation_kind,
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
    include_active: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run bounded per-job specialists under a validated BudgetBook.

    Refuses to run when the supplied book is not the one most recently
    validated by validate_budget_book - that ordering is the
    anti-split-brain invariant. Each specialist optimizes one job inside
    its BudgetSlice and reports a fitness signal; it cannot allocate
    outside its slice or see the cluster plan.

    Args:
        max_workers: Parallel specialist calls.
        include_active: Include rehabilitation-eligible running jobs in addition to
            waiting jobs. Enabled by default because the candidate builder scores SWAPs.

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

    def retry_actionable(job: dict[str, Any]) -> bool:
        status = job.get("deployment_status")
        if status == "deployment_pending":
            return False
        if status == "deployment_not_materialized":
            return job.get("deployment_retry_allowed") is True
        return True

    pending_ids = {
        job.get("job_id", job.get("id"))
        for job in get_pending_jobs()
        if job.get("job_id", job.get("id")) and retry_actionable(job)
    }
    eligible_ids = set(pending_ids)
    if include_active:
        eligible_ids.update(
            job.get("job_id", job.get("id"))
            for job in get_active_jobs()
            if job.get("job_id", job.get("id"))
            and (job.get("health") or {}).get("rehabilitation_eligible") is True
            and (job.get("deployment_action_type") != "swap" or retry_actionable(job))
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
    job_type = _job_mode(job_features)
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


def _model_catalog_for(model_id: Any) -> dict[str, Any]:
    """Return the Store model catalog row for a model, or {} when unavailable."""
    resource_map = getattr(_CTX, "resource_map", None)
    if not model_id or resource_map is None or not hasattr(resource_map, "model_catalog"):
        return {}
    try:
        return dict(resource_map.model_catalog(str(model_id)) or {})
    except Exception:
        return {}


def _model_num_layers(catalog: dict[str, Any], job_features: dict[str, Any] | None) -> int | None:
    """Best-effort transformer layer count, from the catalog or job features.

    PP partitions layers, so a PP degree must divide this. None when unknown
    (then no generated PP variant is emitted and the surrogate stays the backstop).
    """
    for source in (catalog or {}), (job_features or {}):
        for key in ("num_hidden_layers", "num_layers", "n_layers", "num_hidden_layer"):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


# Catalog facts that decide whether a model's WEIGHTS fit one GPU. Scalars only:
# per-GPU list fields (max_num_seq, kvcache_dtype) are deliberately left out so the
# fit check stays a weight check and cannot trip on an unresolved list.
_WEIGHT_FIT_FIELDS = (
    "model_params_b",
    "model_size_gb",
    "weight_dtype",
    "activation_dtype",
    "weight_quantization_bits",
    "weight_quantization_method",
    "is_moe",
)


def _model_weight_fit_values(
    catalog: dict[str, Any], job_features: dict[str, Any] | None
) -> dict[str, Any]:
    """Return the inputs target_memory_fit needs to judge weight fit for a model."""
    values: dict[str, Any] = {}
    for source in (job_features or {}), (catalog or {}):
        for key in _WEIGHT_FIT_FIELDS:
            value = source.get(key) if isinstance(source, dict) else None
            if value is not None and key not in values:
                values[key] = value
    return values


def config_runnable(
    config: dict[str, Any],
    job_features: dict[str, Any] | None = None,
    gpu_type: str | None = None,
) -> tuple[bool, str]:
    """Deterministic physical-validity pre-check for a rank config.

    Enforces the HARD constraints the model/hardware impose - gpu_count must equal
    the tp*pp engine demand, and tp must divide the model's attention-head count - in
    CODE, so an unrunnable config (e.g. tp=8 on a 28-head model) is rejected with
    a clear reason instead of being nagged about in the prompt or crashing the
    surrogate. Checks it cannot evaluate (missing catalog arch) are skipped, not
    failed - the surrogate stays the backstop for those. Returns (ok, reason).
    """
    config = config or {}
    raw_tp = config.get("tp", 1)
    raw_pp = config.get("pp", 1)
    if type(raw_tp) is not int or type(raw_pp) is not int:
        return False, f"tp and pp must be positive integers (got tp={raw_tp!r}, pp={raw_pp!r})"
    tp = raw_tp
    pp = raw_pp
    if tp < 1 or pp < 1:
        return False, f"tp={tp} and pp={pp} must both be >= 1"
    ep = config.get("ep", SUPPORTED_EP)
    if type(ep) is not int or ep != SUPPORTED_EP:
        return False, f"ep must be exactly {SUPPORTED_EP} in this Koi version (got {ep!r})"
    gpu_count = config.get("gpu_count", config.get("count"))
    if type(gpu_count) is not int or gpu_count <= 0:
        return False, f"gpu_count must be a positive integer (got {gpu_count!r})"
    if gpu_count != tp * pp:
        return False, f"gpu_count must equal tp*pp={tp * pp} (got {gpu_count})"
    heads = _model_num_heads(config, job_features)
    if heads and heads % tp != 0:
        return False, f"tp={tp} does not divide the model's {heads} attention heads (cannot shard)"
    model_id = str(config.get("model_id") or (job_features or {}).get("model_id") or "")
    hardware = str(gpu_type or config.get("gpu_type") or (job_features or {}).get("gpu_type") or "")
    policy = load_config_policy()
    if model_id in MODEL_MAX_TP and tp > MODEL_MAX_TP[model_id]:
        return False, f"{model_id} policy limits TP to {MODEL_MAX_TP[model_id]} (got TP={tp})"
    rule = policy.rule_for(hardware, model_id) if hardware and model_id else None
    if rule is not None:
        if tp not in rule.allowed_tp:
            return False, (
                f"{model_id} policy on {hardware} allows TP {list(rule.allowed_tp)} (got TP={tp})"
            )
        precision_ok = policy.precision_matches(
            rule,
            config.get("weight_dtype") or (job_features or {}).get("weight_dtype"),
            config.get("weight_quantization_method")
            or (job_features or {}).get("weight_quantization_method"),
        )
        if precision_ok is False:
            return False, f"{model_id} policy on {hardware} requires {rule.precision} precision"
    return True, ""


def size_ladder(
    ranks: list[dict[str, Any]],
    job_features: dict[str, Any],
    target_tps: float | None = None,
    utilization_target: float | None = None,
) -> dict[str, Any]:
    """Evaluate fixed-rank proposals with Direct point estimates.

    ``n_replicas`` is part of each proposed frame and is never resized here. Direct
    supplies point capacity and base service latency for that exact geometry; it does
    not simulate an offered-load queue. Generated replica alternatives therefore must
    be enumerated by the candidate builder before calling this function.

    Physical/configuration failures reject a rank. Prediction-only failures retain the
    physically valid rank as an exploratory candidate with zero service credit so the
    joint selector may use otherwise-idle capacity without displacing supported work.

    Args:
        ranks: rank dicts (RankSpec.from_dict form) with role, env, config.
            Heterogeneous ranks are allowed; order them by preference.
        job_features: the job's W features - type ("online"/"batch"),
            request_arrival_rate, output_len_tokens_avg, target_p99_ttft_ms,
            target_p99_tpot_ms, total_token_budget, deadline_hours,
            headroom_factor.
        target_tps: override; default from required_throughput_enumerator.
        utilization_target: Deprecated compatibility input; ignored.

    Returns:
        {"ranks": [deployable rank dicts, n_replicas >= 1], "regime",
         "target_tps", "point_capacity_tps", "achieved_tps", "unmet_tps",
         "meets_target", "candidate_kind",
         "per_rank": [...all ranks incl. dropped/excluded...],
         "marginal_value": {env_key: extra_gpus_to_meet_target}}.
    """
    _require("resource_map", "surrogate", "candidate_graph", "dro")
    job_features = _sanitize_agent_features(dict(job_features or {}))
    regime = _job_mode(job_features)
    is_online = regime == "online"
    _ = utilization_target
    target = (
        float(target_tps)
        if target_tps is not None
        else float(required_throughput_enumerator(job_features))
    )
    if not math.isfinite(target) or target < 0:
        raise ValueError("target_tps must be a finite non-negative number")
    ttft_target = _feature_value(job_features, "target_p99_ttft_ms", "target_p99_TTFT_ms")
    tpot_target = _feature_value(job_features, "target_p99_tpot_ms", "target_p99_TPOT_ms")

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

    def _base_latency_ok(y: dict) -> bool:
        """Compare Direct's base latency with targets without claiming queue safety."""
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

    def _point_capacity_tps(y: dict) -> float:
        try:
            point = _y_value(y, "throughput_token_per_sec")
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return point if math.isfinite(point) and point > 0 else 0.0

    physical_rejections: list[str] = []
    prediction_rejections: list[dict[str, str]] = []

    def _predict_fixed_rank(
        rank: RankSpec,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
        payload = _rank_prediction_payload(rank, job_features)
        memory_fit = target_memory_fit({**payload["job_features"], **payload["job_config"]})
        if memory_fit["status"] == "physical_no_fit":
            reason = (
                "requested model memory no-fit: "
                f"requires {memory_fit['required_gb']:.2f} GiB per GPU, "
                f"capacity {memory_fit['capacity_gb']:.2f} GiB"
            )
            log.warning(
                "size_ladder rejected candidate before surrogate: model=%s gpu=%s "
                "instance=%s tp=%s pp=%s dp=%d error=%s",
                payload["job_config"].get("model_id") or payload["job_features"].get("model_id"),
                payload["job_features"].get("gpu_type") or payload["job_config"].get("gpu_type"),
                rank.config.get("instance_type"),
                rank.config.get("tp"),
                rank.config.get("pp"),
                rank.n_replicas,
                reason,
            )
            physical_rejections.append(reason)
            return (
                {},
                {"kind": "hard", "status": "physical_no_fit", "reason": reason},
                "aic_direct_point",
            )
        try:
            prediction = _predict_outcome_core(
                payload["job_config"],
                payload["job_features"],
                scenario="peak" if is_online else "mean",
                method=_AIC_DIRECT_METHOD,
            )
            semantics = prediction.get("prediction_semantics") or {}
            prediction_basis = str(semantics.get("basis") or "aic_direct_point")
            y_hat = dict(prediction.get("y_hat", {}))
            if not y_hat:
                lineage = prediction.get("prediction_lineage") or {}
                primary = (lineage.get("backends") or {}).get("primary") or {}
                primary_status = primary.get("status")
                metadata = primary.get("metadata") or {}
                if primary_status == "failed":
                    status = "prediction_failed"
                    reason = str(metadata.get("error") or "Direct prediction failed")
                elif primary_status == "unsupported":
                    status = "unsupported_prediction"
                    reason = str(metadata.get("error") or "Direct prediction is unsupported")
                else:
                    status = "prediction_empty"
                    reason = "surrogate prediction returned no Y values"
                prediction_rejections.append({"status": status, "reason": reason})
                return {}, {"kind": "soft", "status": status, "reason": reason}, prediction_basis
            throughput = y_hat.get("throughput_token_per_sec")
            if throughput is None:
                issue = {
                    "kind": "soft",
                    "status": "prediction_incomplete",
                    "reason": "surrogate prediction omitted throughput_token_per_sec",
                }
                prediction_rejections.append({k: issue[k] for k in ("status", "reason")})
                return y_hat, issue, prediction_basis
            try:
                throughput_value = float(throughput)
            except (TypeError, ValueError, OverflowError):
                throughput_value = math.nan
            if not math.isfinite(throughput_value) or throughput_value <= 0:
                issue = {
                    "kind": "soft",
                    "status": "zero_predicted_capacity",
                    "reason": f"surrogate predicted unusable throughput {throughput!r}",
                }
                prediction_rejections.append({k: issue[k] for k in ("status", "reason")})
                return y_hat, issue, prediction_basis
            if not _slo_prediction_complete(y_hat):
                issue = {
                    "kind": "soft",
                    "status": "prediction_incomplete",
                    "reason": "surrogate prediction omitted a declared TTFT/TPOT value",
                }
                prediction_rejections.append({k: issue[k] for k in ("status", "reason")})
                return y_hat, issue, prediction_basis
            lower = prediction.get("throughput_token_per_sec_lower")
            if lower is not None:
                y_hat["_throughput_token_per_sec_lower"] = lower
            return y_hat, None, prediction_basis
        except SurrogateMemoryNoFit as exc:
            log.warning(
                "size_ladder rejected candidate: model=%s gpu=%s instance=%s "
                "tp=%s pp=%s dp=%d error=%s",
                payload["job_config"].get("model_id") or payload["job_features"].get("model_id"),
                payload["job_features"].get("gpu_type") or payload["job_config"].get("gpu_type"),
                rank.config.get("instance_type"),
                rank.config.get("tp"),
                rank.config.get("pp"),
                rank.n_replicas,
                exc,
            )
            physical_rejections.append(str(exc))
            return (
                {},
                {"kind": "hard", "status": "physical_no_fit", "reason": str(exc)},
                "aic_direct_point",
            )
        except SurrogateUnsupportedConfig as exc:
            log.warning(
                "size_ladder unsupported prediction: model=%s gpu=%s instance=%s "
                "tp=%s pp=%s dp=%d error=%s",
                payload["job_config"].get("model_id") or payload["job_features"].get("model_id"),
                payload["job_features"].get("gpu_type") or payload["job_config"].get("gpu_type"),
                rank.config.get("instance_type"),
                rank.config.get("tp"),
                rank.config.get("pp"),
                rank.n_replicas,
                exc,
            )
            issue = {
                "kind": "soft",
                "status": "unsupported_prediction",
                "reason": str(exc),
            }
            prediction_rejections.append({k: issue[k] for k in ("status", "reason")})
            return {}, issue, "aic_direct_point"
        except SurrogateExecutionError as exc:
            issue = {"kind": "soft", "status": "prediction_failed", "reason": str(exc)}
            prediction_rejections.append({k: issue[k] for k in ("status", "reason")})
            return {}, issue, "aic_direct_point"

    sized: list[dict[str, Any]] = []
    per_rank: list[dict[str, Any]] = []
    point_capacity_total = 0.0
    prediction_bases: set[str] = set()
    remaining_by_pool: dict[tuple[str, str | None], int] = {}
    resources = (
        _CTX.resource_map.resources_summary()
        if hasattr(_CTX.resource_map, "resources_summary")
        else None
    )

    hard_issue: dict[str, Any] | None = None
    soft_issue: dict[str, Any] | None = None
    for raw in ranks:
        physical_rejection_start = len(physical_rejections)
        prediction_rejection_start = len(prediction_rejections)
        try:
            rank = RankSpec.from_dict(raw)
        except (TypeError, ValueError) as exc:
            parse_issue = {"kind": "hard", "status": "invalid_config", "reason": str(exc)}
            hard_issue = hard_issue or parse_issue
            per_rank.append(
                {
                    "role": raw.get("role") if isinstance(raw, dict) else None,
                    "env": raw.get("env") if isinstance(raw, dict) else None,
                    "n_replicas": 0,
                    "requested_replicas": raw.get("n_replicas") if isinstance(raw, dict) else None,
                    "served_tps": 0.0,
                    "point_capacity_tps": None,
                    "slo_ok": False,
                    "base_latency_within_target": None,
                    "prediction_received": False,
                    "prediction_complete": False,
                    "reason": parse_issue["reason"],
                    "failure_kind": parse_issue["kind"],
                    "failure_status": parse_issue["status"],
                    "physical_violations": [],
                    "prediction_failures": [],
                }
            )
            continue
        requested_replicas = int(rank.n_replicas)
        gpus_per_chain = rank.gpus_per_chain()
        gpu_type = env_gpu_type(rank.env)
        env_key = _env_key(rank.env)
        if resources is not None:
            info = resources.get(env_key)
            env_free = int(info.get("free", 0)) if info and info.get("gpu_type") == gpu_type else 0
        else:
            env_free = _CTX.resource_map.get_avail_capacity(rank.env, gpu_type) if gpu_type else 0
        allocation_error: str | None = None
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
        issue: dict[str, Any] | None = None
        point_capacity: float | None = None
        base_latency_ok: bool | None = None
        reason: str | None = None
        y_hat: dict[str, Any] = {}
        prediction_basis = "aic_direct_point"

        if requested_replicas < 1:
            issue = {
                "kind": "hard",
                "status": "invalid_config",
                "reason": "n_replicas must be >= 1",
            }
        elif not runnable:
            issue = {
                "kind": "hard",
                "status": "invalid_config",
                "reason": f"config not runnable: {validity_reason}",
            }
        elif allocation_error is not None:
            issue = {
                "kind": "hard",
                "status": "invalid_config",
                "reason": f"allocation cannot be resolved: {allocation_error}",
            }
        elif max_by_cap < 1:
            issue = {
                "kind": "hard",
                "status": "no_pool_capacity",
                "reason": "no free capacity in pool",
            }
        elif requested_replicas > max_by_cap:
            issue = {
                "kind": "hard",
                "status": "no_pool_capacity",
                "reason": (
                    f"requested {requested_replicas} replicas but pool capacity allows {max_by_cap}"
                ),
            }
        else:
            y_hat, issue, prediction_basis = _predict_fixed_rank(rank)
            prediction_bases.add(prediction_basis)
            if issue is None:
                point_capacity = _point_capacity_tps(y_hat)
                base_latency_ok = _base_latency_ok(y_hat)
                reason = (
                    "Direct point capacity with base-latency target miss; queue unmodeled"
                    if base_latency_ok is False
                    else "Direct point capacity; queue unmodeled"
                )

        if issue is not None:
            reason = str(issue["reason"])
            if issue["kind"] == "hard":
                hard_issue = hard_issue or issue
            else:
                soft_issue = soft_issue or issue
        if issue is None or issue["kind"] == "soft":
            rank.rank_traffic_share = None
            rank.config.pop("_arrival_share_rps", None)
            sized.append(rank.to_dict())
            remaining_by_pool[pool_key] = max(0, free - requested_replicas * capacity_per_replica)
        if point_capacity is not None:
            point_capacity_total += point_capacity

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
                "requested_replicas": requested_replicas,
                "n_replicas": requested_replicas if issue is None or issue["kind"] == "soft" else 0,
                "share_tps": None,
                "served_tps": 0.0,
                "point_capacity_tps": point_capacity,
                "prediction_basis": prediction_basis,
                "base_p99_ttft_ms": _finite_latency(y_hat, "p99_ttft_ms", "p99_TTFT_ms"),
                "base_p99_tpot_ms": _finite_latency(y_hat, "p99_tpot_ms", "p99_TPOT_ms"),
                "slo_ok": base_latency_ok is True,
                "base_latency_within_target": base_latency_ok,
                "prediction_received": bool(y_hat),
                "prediction_complete": _slo_prediction_complete(y_hat),
                "partial_search_attempted": False,
                "partial_search_probes": 0,
                "partial_search_truncated": False,
                "partial_search_upper_tps": 0.0,
                "partial_admission": False,
                "admitted_tps": None,
                "reason": reason,
                "failure_kind": issue.get("kind") if issue else None,
                "failure_status": issue.get("status") if issue else None,
                "physical_violations": physical_rejections[physical_rejection_start:],
                "prediction_failures": prediction_rejections[prediction_rejection_start:],
            }
        )

    if target <= 0 and hard_issue is None:
        soft_issue = soft_issue or {
            "kind": "soft",
            "status": "demand_unmodeled",
            "reason": "workload has no positive token-throughput target",
        }
    if hard_issue is not None:
        sized = []
        point_capacity_total = 0.0
    exploratory = hard_issue is None and soft_issue is not None
    achieved_tps = 0.0 if exploratory else min(target, point_capacity_total)
    unmet_tps = max(0.0, target - achieved_tps)
    capacity_fraction = min(1.0, achieved_tps / target) if target > 0 else 0.0
    serving = [rank for rank in per_rank if rank["n_replicas"] >= 1]
    base_latency_ok = bool(serving) and all(
        rank["base_latency_within_target"] is True for rank in serving
    )
    meets_target = (
        target > 0
        and not exploratory
        and achieved_tps >= target - max(1e-6, 1e-3 * target)
        and base_latency_ok
    )
    if not exploratory and hard_issue is None and target > 0 and point_capacity_total > 0:
        for rank_dict, detail in zip(sized, serving, strict=False):
            capacity = float(detail.get("point_capacity_tps") or 0.0)
            routing_share = capacity / point_capacity_total
            rank_dict["rank_traffic_share"] = routing_share if len(sized) > 1 else None
            detail["share_tps"] = target * routing_share
            detail["served_tps"] = achieved_tps * routing_share
    failure = hard_issue or soft_issue
    prediction_basis = (
        "composed_point_estimate"
        if "composed_point_estimate" in prediction_bases
        else "aic_direct_point"
    )
    return {
        "ranks": sized,
        "regime": regime,
        "target_tps": target,
        "point_capacity_tps": point_capacity_total if hard_issue is None else None,
        "capacity_fraction": capacity_fraction,
        "base_latency_within_target": base_latency_ok if hard_issue is None else None,
        "achieved_tps": achieved_tps,
        "unmet_tps": unmet_tps,
        "meets_target": meets_target,
        "candidate_kind": "rejected"
        if hard_issue is not None
        else "exploratory"
        if exploratory
        else "service",
        "prediction_semantics": {
            **_DIRECT_PREDICTION_SEMANTICS,
            "basis": prediction_basis,
        },
        "partial_online_admission": False,
        "admission_mode": None,
        "partial_search_probes": 0,
        "partial_search_truncated": False,
        "per_rank": per_rank,
        "failure_kind": failure["kind"] if failure else None,
        "failure_status": failure["status"] if failure else None,
        "failure_reason": failure["reason"] if failure else None,
        "marginal_value": {},
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
        rows = store.retrieve_similar_rows(job_features, top_k=max(int(top_k) * 4, int(top_k)))
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

    wanted_type = job_features.get("workload_type") or job_features.get("type")
    if wanted_type is not None:
        wanted_type = str(wanted_type).lower()
        rows = [
            row
            for row in rows
            if str(row.X.get("workload_type") or row.X.get("type") or "").lower() == wanted_type
        ]
        rows = rows[-int(top_k) :]

    briefs = []
    for r in rows:
        brief = {
            "tick": r.tick,
            "job_id": r.job_id,
            "rank_id": r.rank_id,
            "env_label": r.env_label,
            "y_observed_mean": dict(getattr(r, "y_observed_mean", {}) or {}),
        }
        # Mechanism annotations are DAG vocabulary; an inert-DAG run keeps the
        # observed outcomes but withholds the causal bookkeeping.
        if not ablation.mechanism_inert():
            brief["mechanism_ids"] = list(getattr(r, "mechanism_ids", []))
            brief["q_labels"] = {
                mid: (q.value if hasattr(q, "value") else q)
                for mid, q in getattr(r, "q_label_per_mechanism", {}).items()
            }
        briefs.append(brief)
    return briefs


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
    if ablation.mechanism_inert() or ablation.learning_frozen():
        return {
            "ok": False,
            "mechanism_id": None,
            "seed_confidence": None,
            "violations": [
                "mechanism admission is disabled for this ablation run "
                f"(mechanism_mode={ablation.mechanism_mode()}, "
                f"learning_mode={ablation.learning_mode()})"
            ],
        }
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
        queue_aware: Deprecated compatibility input. It is ignored; production
            predictions always use Direct AIC.

    Returns:
        {"y_hat": calibrated dict, "y_hat_raw": surrogate dict,
         "calibration_offsets": dict, "v_hat": dict, "dro_band": dict,
         "prediction_semantics": dict}. Direct throughput is point capacity;
         Direct latency is base service latency and does not verify queue SLOs.
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
    _ = queue_aware
    return _predict_outcome_core(
        job_config,
        job_features,
        calibrate=calibrate,
        scenario=scenario,
        method=_AIC_DIRECT_METHOD,
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

    Direct is deterministic within one tick, so commit scoring reuses the exact
    selection prediction. Finalization accounting is tracked separately.
    """
    _ = finalization
    try:
        return json.dumps(
            [job_config, job_features, scenario, bool(calibrate), method],
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        return None


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
    calibration details carried in its prediction lineage. The method input remains
    compatible with older callers but is canonicalized to Direct AIC.
    """
    _require("candidate_graph", "dro", "surrogate")
    ep = job_config.get("ep", SUPPORTED_EP)
    if type(ep) is not int or ep != SUPPORTED_EP:
        raise SurrogateUnsupportedConfig(
            f"ep must be exactly {SUPPORTED_EP} in this Koi version (got {ep!r})"
        )
    memory_fit = target_memory_fit({**job_features, **job_config})
    if memory_fit["status"] == "physical_no_fit":
        raise SurrogateMemoryNoFit(
            "requested model memory no-fit: "
            f"requires {float(memory_fit['required_gb']):.2f} GiB per GPU, "
            f"capacity {float(memory_fit['capacity_gb']):.2f} GiB"
        )
    global _surrogate_budget_rejections, _surrogate_cache_hits, _surrogate_calls
    global _surrogate_finalization_calls, _surrogate_stress_calls
    selected_method = _AIC_DIRECT_METHOD
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
            if _finalization:
                _surrogate_finalization_calls += 1
        else:
            is_stress = scenario == "peak_all_multiturn_stress"
            if _finalization:
                _surrogate_finalization_calls += 1
            elif is_stress:
                _surrogate_stress_calls += 1
            else:
                if _surrogate_calls >= SURROGATE_CALL_BUDGET:
                    _surrogate_budget_rejections += 1
                    raise SurrogateBudgetExceeded(
                        f"surrogate-call budget {SURROGATE_CALL_BUDGET} reached this tick; "
                        "narrow to your best few candidate configs and reuse scored results."
                    )
                _surrogate_calls += 1
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
        "prediction_semantics": copy.deepcopy(
            (prediction_lineage or {}).get("prediction_semantics") or _DIRECT_PREDICTION_SEMANTICS
        ),
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
                method=_AIC_DIRECT_METHOD,
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
    """Comparable objectives whose decision-time bands can be evaluated later."""
    assessment = getattr(action, "prediction_assessment", None) or {}
    required: set[str] = set()
    if assessment.get("queue_slo_verified") is True:
        required.update(
            str(objective)
            for objective, threshold in _slo_thresholds_for(snapshot, action.job_id).items()
            if threshold is not None
        )
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
        method = _AIC_DIRECT_METHOD
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
        stamped_queue_states: list[str] = []
        for rank in action.ladder:
            payload = _rank_prediction_payload(rank, job_features)
            try:
                pred = _predict_outcome_core(
                    payload["job_config"],
                    payload["job_features"],
                    calibrate=True,
                    scenario="peak" if _job_mode(job_features) == "online" else "mean",
                    method=method,
                    _finalization=True,
                )
            except (SurrogateUnsupportedConfig, SurrogateExecutionError) as exc:
                if not _is_exploratory_assessment(action.prediction_assessment):
                    raise
                pred = {
                    "y_hat": {},
                    "v_hat": {},
                    "dro_band": {},
                    "prediction_lineage": {
                        "schema_version": 3,
                        "method": list(_AIC_DIRECT_METHOD),
                        "scenario": "peak",
                        "prediction_semantics": dict(_DIRECT_PREDICTION_SEMANTICS),
                        "backends": {
                            "primary": {
                                "status": "unsupported"
                                if isinstance(exc, SurrogateUnsupportedConfig)
                                else "failed",
                                "metadata": {
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                            }
                        },
                    },
                }
            rank.predicted_y = dict(pred.get("y_hat_raw") or pred.get("y_hat") or {})
            rank.predicted_v = dict(pred.get("v_hat") or {})
            lineage = compact_prediction_lineage(pred.get("prediction_lineage"))
            lineage["prediction_semantics"] = copy.deepcopy(
                pred.get("prediction_semantics")
                or lineage.get("prediction_semantics")
                or _DIRECT_PREDICTION_SEMANTICS
            )
            lineage["prediction_assessment"] = copy.deepcopy(action.prediction_assessment or {})
            if action.service_class is not None:
                lineage["service_class"] = action.service_class
            if _job_mode(job_features) == "online":
                try:
                    lineage["queue_shadow"] = estimate_queue_shadow(
                        arrival_rate_rps=_feature_value(job_features, "request_arrival_rate"),
                        input_tokens_per_request=_feature_value(
                            job_features, "isl_token_avg", "input_len_tokens_avg"
                        ),
                        input_tokens_per_request_max=_feature_value(
                            job_features, "isl_token_max", "input_len_tokens_max"
                        ),
                        output_tokens_per_request=_feature_value(
                            job_features, "osl_token_avg", "output_len_tokens_avg"
                        ),
                        output_tokens_per_request_max=_feature_value(
                            job_features, "osl_token_max", "output_len_tokens_max"
                        ),
                        aggregate_capacity_tps=rank.predicted_y.get("throughput_token_per_sec"),
                        replicas=rank.n_replicas,
                        base_ttft_ms=rank.predicted_y.get("p99_ttft_ms"),
                        homogeneous=len(action.ladder) == 1,
                        scenario="peak",
                        peak_to_mean_ratio=_feature_value(job_features, "peak_to_mean_ratio")
                        or 1.0,
                        affects_selection=False,
                    )
                    stamped_queue_states.append(
                        str(lineage["queue_shadow"].get("status") or "unmodeled")
                    )
                except Exception as exc:
                    lineage["queue_shadow"] = {
                        "model": "erlang_c_token_work_v2",
                        "mode": "shadow",
                        "status": "unmodeled",
                        "reason": f"queue shadow failed: {exc}",
                        "affects_selection": False,
                    }
                    stamped_queue_states.append("unmodeled")
            if pred.get("dro_band"):
                lineage["decision_dro_band"] = copy.deepcopy(pred["dro_band"])
            lineage["decision_required_objectives"] = list(required_objectives)
            lineage["deployment_id"] = (
                f"deploy:{typed.tick}:{action.job_id}:{rank.rank_id or 'rank'}"
            )
            lineage["evidence_baseline"] = "pre_calibration"
            if partial_admission is not None:
                lineage["partial_admission"] = copy.deepcopy(partial_admission)
            rank.prediction_lineage = lineage
        if "unstable" in stamped_queue_states:
            action.queue_state = "unstable"
        elif stamped_queue_states and all(state == "stable" for state in stamped_queue_states):
            action.queue_state = "stable"
        elif stamped_queue_states:
            action.queue_state = "unmodeled"
        action.queue_slo_verified = False
    write_event = getattr(getattr(_CTX, "trace_logger", None), "write_event", None)
    if callable(write_event):
        try:
            write_event(
                "placement_decision_committed",
                {
                    "actions": [
                        {
                            "job_id": action.job_id,
                            "type": action.type.value,
                            "ranks": [
                                {
                                    "rank_id": rank.rank_id,
                                    "env": list(rank.env or []),
                                    "instance_type": rank.config.get("instance_type"),
                                    "tp": rank.config.get("tp"),
                                    "n_replicas": rank.n_replicas,
                                }
                                for rank in action.ladder or []
                            ],
                        }
                        for action in typed.actions
                    ]
                },
                tick=typed.tick,
            )
        except Exception:
            log.exception("committed placement decision trace failed")
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
    if ablation.mechanism_inert():
        return 0.0
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


def _keep_baseline_sigma(
    snapshot,
    job_id: str,
    job_features: dict[str, Any],
    weights: dict[str, float],
    ranges: dict[str, float],
    *,
    priority: float = 1.0,
) -> float:
    """Score the active deployment from observed service outcomes."""
    observed = dict(_active_health_for(snapshot, job_id).get("observed") or {})
    observed_tps = observed.get(_THROUGHPUT_OBJ)
    # Under-service floor. The Tchebycheff baseline below normalizes the
    # throughput gap to at most 1.0, so a deployment serving 8% of its
    # requirement with clean latency scored roughly like one serving 92% - a
    # batch job once held the cluster's scarcest pool for 36 ticks that way,
    # and a dead rank scored BETTER than one serving with terrible latency.
    # Price the shortfall like unserved demand, in proportion: at zero service
    # the baseline equals a full defer penalty, so any swap that serves
    # anything wins.
    under_service_floor = 0.0
    demand_visible = (
        _job_mode(job_features) != "online" or float(observed.get("depth_req_q") or 0.0) > 0.0
    )
    if observed_tps is not None and demand_visible:
        try:
            required = float(required_throughput_enumerator(job_features))
        except Exception:
            required = 0.0
        if required > 0:
            shortfall = min(1.0, max(0.0, 1.0 - float(observed_tps) / required))
            under_service_floor = -UNSERVED_PENALTY * max(1.0, float(priority)) * shortfall
    if _job_mode(job_features) == "online" and float(observed.get("depth_req_q") or 0.0) <= 0:
        # At an empty queue, achieved throughput is offered load rather than capacity.
        observed.pop(_THROUGHPUT_OBJ, None)
    target_ref = _target_reference(job_features)
    if target_ref:
        reference = target_ref
        current = _clamp_to_reference(observed, target_ref)
        normalization = {name: abs(value) for name, value in target_ref.items() if value}
    else:
        reference = _seeded_z_star(get_z_star(job_features))
        current = observed
        normalization = ranges
    scoreable = _scoreable_y_hat(current, weights, reference, normalization)
    if not scoreable:
        return min(0.0, under_service_floor)
    return min(
        float(
            _CTX.tchebycheff_module.compute_tchebycheff(
                y_hat=scoreable,
                w_t=weights,
                z_star_t=reference,
                normalization_range=normalization,
            )
        ),
        under_service_floor,
    )


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

    sigma = J + beta_t * eig - gamma * Pr_DRO - lambda_swit * switch_cost.
    PLACE contributes sigma; SWAP contributes sigma minus the observed KEEP
    baseline. The
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
    per_job: dict[str, dict[str, Any]] = {}
    aggregate = 0.0
    served_fraction_by_job: dict[str, float] = {}
    priority_by_job = {
        p.get("job_id"): float(p.get("priority_score", 1.0) or 1.0) for p in get_priority()
    }
    for action in typed.actions:
        if action.type not in LADDER_ACTIONS or not action.ladder:
            continue
        if _is_exploratory_assessment(action.prediction_assessment):
            served_fraction_by_job[action.job_id] = 0.0
            continue
        raw_fraction = getattr(action, "served_fraction", None)
        if raw_fraction is None:
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
        assessment = action.prediction_assessment
        exploratory = _is_exploratory_assessment(assessment)
        prediction_method = _AIC_DIRECT_METHOD
        prediction_scenario = "peak" if job_mode == "online" else "mean"
        try:
            y_hat = _compose_job_y_hat(
                action,
                job_features,
                method=prediction_method,
                scenario=prediction_scenario,
                finalization=finalization,
            )
        except SurrogateExecutionError as exc:
            if not isinstance(assessment, dict):
                raise
            assessment.update(
                kind="exploratory",
                status="prediction_failed",
                reason=str(exc),
            )
            exploratory = True
            y_hat = {}
        if y_hat and isinstance(assessment, dict):
            throughput = y_hat.get(_THROUGHPUT_OBJ)
            try:
                throughput_value = float(throughput) if throughput is not None else math.nan
            except (TypeError, ValueError, OverflowError):
                throughput_value = math.nan
            missing_latency = job_mode == "online" and any(
                _feature_value(job_features, *target_names) is not None
                and not any(y_hat.get(name) is not None for name in outcome_names)
                for target_names, outcome_names in (
                    (
                        ("target_p99_ttft_ms", "target_p99_TTFT_ms"),
                        ("p99_ttft_ms", "p99_TTFT_ms"),
                    ),
                    (
                        ("target_p99_tpot_ms", "target_p99_TPOT_ms"),
                        ("p99_tpot_ms", "p99_TPOT_ms"),
                    ),
                )
            )
            if throughput is None or missing_latency:
                assessment.update(
                    kind="exploratory",
                    status="prediction_incomplete",
                    reason="final Direct prediction omitted required point outputs",
                )
                exploratory = True
            elif not math.isfinite(throughput_value) or throughput_value <= 0:
                assessment.update(
                    kind="exploratory",
                    status="zero_predicted_capacity",
                    reason=f"final Direct prediction returned capacity {throughput!r}",
                )
                exploratory = True
            elif exploratory:
                try:
                    recovered_target = float(
                        action.target_tps or required_throughput_enumerator(job_features)
                    )
                except (TypeError, ValueError, OverflowError):
                    recovered_target = 0.0
                if recovered_target > 0:
                    assessment.update(
                        kind="point",
                        status="success",
                        reason="Direct point prediction recovered during commit scoring",
                    )
                    exploratory = False
                    if action.service_class != "idle_capacity_fallback":
                        recovered_capacity = max(0.0, float(y_hat.get(_THROUGHPUT_OBJ) or 0.0))
                        recovered_achieved = min(recovered_target, recovered_capacity)
                        recovered_latency_ok = all(
                            target is None
                            or (
                                y_hat.get(metric) is not None
                                and float(y_hat[metric]) <= float(target)
                            )
                            for metric, target in (
                                (
                                    "p99_ttft_ms",
                                    _feature_value(
                                        job_features,
                                        "target_p99_ttft_ms",
                                        "target_p99_TTFT_ms",
                                    ),
                                ),
                                (
                                    "p99_tpot_ms",
                                    _feature_value(
                                        job_features,
                                        "target_p99_tpot_ms",
                                        "target_p99_TPOT_ms",
                                    ),
                                ),
                            )
                        )
                        action.achieved_tps = recovered_achieved
                        action.unmet_tps = max(0.0, recovered_target - recovered_achieved)
                        action.served_fraction = min(1.0, recovered_capacity / recovered_target)
                        action.point_capacity_covers_target = recovered_capacity >= recovered_target
                        action.base_latency_within_target = recovered_latency_ok
                        action.meets_target = bool(
                            action.point_capacity_covers_target and recovered_latency_ok
                        )
                        action.service_class = "supported" if action.meets_target else "partial"
        if exploratory or (not y_hat and isinstance(assessment, dict)):
            assessment_dict = assessment if isinstance(assessment, dict) else {}
            if not exploratory:
                assessment_dict.update(
                    kind="exploratory",
                    status="prediction_incomplete",
                    reason=assessment_dict.get("reason")
                    or "final Direct prediction was unavailable",
                )
            action.admitted_tps = None
            action.achieved_tps = None
            action.unmet_tps = None
            action.meets_target = None
            action.served_fraction = None
            action.admission_mode = None
            if action.service_class != "idle_capacity_fallback":
                action.service_class = "exploratory"
            for rank in action.ladder:
                rank.rank_traffic_share = None
            served_fraction_by_job[job_id] = 0.0
            sigma_i = _WORK_CONSERVING_GAIN_FLOOR
            per_job[job_id] = {
                "J": 0.0,
                "eig": 0.0,
                "switch_cost_total": 0.0,
                "pr_slo_dro": 0.0,
                "cost_penalty": 0.0,
                "opt_bonus": 0.0,
                "value_bonus": 0.0,
                "sigma": sigma_i,
                "prediction_available": bool(y_hat),
                "scoring_mode": "exploration_only",
                "prediction_status": assessment_dict.get("status"),
            }
            aggregate += sigma_i
            continue
        if not y_hat:
            continue
        if isinstance(assessment, dict) and assessment.get("basis") in _POINT_PREDICTION_BASES:
            try:
                declared_target = float(
                    action.target_tps or required_throughput_enumerator(job_features)
                )
                predicted_capacity = float(y_hat.get(_THROUGHPUT_OBJ) or 0.0)
            except (TypeError, ValueError, OverflowError):
                declared_target = 0.0
                predicted_capacity = 0.0
            served_fraction_by_job[job_id] = (
                min(1.0, predicted_capacity / declared_target)
                if declared_target > 0 and predicted_capacity > 0
                else 0.0
            )
        else:
            served_fraction_by_job.setdefault(job_id, 1.0)
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
        eig_value = (
            0.0
            if ablation.mechanism_inert()
            else float(
                _CTX.eig_module.compute_eig(
                    L_prime=_materialize_ladder(ladder_dicts),
                    candidate_graph=_CTX.candidate_graph,
                    mechanism_registry=_CTX.mechanism_registry,
                    confidence_service=_CTX.confidence_service,
                    evidence_store=_CTX.evidence_store,
                )
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
        model_sigma = sigma_i
        rehabilitation_credit = 0.0
        if action.type == ActionType.SWAP:
            severity = (
                1.0
                if action.rehabilitation_status == "critical"
                else 0.5
                if action.rehabilitation_status == "degraded"
                else 0.0
            )
            rehabilitation_credit = (
                UNSERVED_PENALTY * max(1.0, priority_by_job.get(job_id, 1.0)) * severity
            )
            sigma_i += rehabilitation_credit
        selection_mode = assessment.get("selection_mode") if isinstance(assessment, dict) else None
        if selection_mode in {"work_conserving", "emergency_recovery"}:
            if selection_mode == "work_conserving":
                action.service_class = "idle_capacity_fallback"
            if selection_mode == "emergency_recovery":
                sigma_i = _WORK_CONSERVING_GAIN_FLOOR
            else:
                fraction = served_fraction_by_job.get(job_id, 0.0)
                avoided_penalty = (
                    UNSERVED_PENALTY * max(1.0, priority_by_job.get(job_id, 1.0)) * fraction
                )
                if action.queue_state == "unstable" or sigma_i + avoided_penalty <= 0:
                    sigma_i = -avoided_penalty + _WORK_CONSERVING_GAIN_FLOOR
        keep_baseline_sigma = 0.0
        swap_gain_over_keep = sigma_i
        if action.type == ActionType.SWAP:
            keep_baseline_sigma = _keep_baseline_sigma(
                snapshot,
                job_id,
                job_features,
                w_t,
                ranges,
                priority=priority_by_job.get(job_id, 1.0),
            )
            swap_gain_over_keep = sigma_i - keep_baseline_sigma
            action.keep_baseline_sigma = keep_baseline_sigma
            action.swap_gain_over_keep = swap_gain_over_keep
        per_job[job_id] = {
            "J": J,
            "eig": eig_value,
            "switch_cost_total": switch_total,
            "pr_slo_dro": pr_slo,
            "cost_penalty": cost_penalty,
            "opt_bonus": opt_bonus,
            "value_bonus": value_bonus,
            "rehabilitation_credit": rehabilitation_credit,
            "keep_baseline_sigma": keep_baseline_sigma,
            "swap_gain_over_keep": swap_gain_over_keep,
            "model_sigma": model_sigma,
            "sigma": sigma_i,
        }
        aggregate += swap_gain_over_keep if action.type == ActionType.SWAP else sigma_i

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
      ('gpu', env_key)            -> GPUs reserved, including idle GPUs in a
                                     whole-instance allocation
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
        instance_type = cfg.get("instance_type")
        spec = (
            (instance_specs.get(env_key) or {}).get(str(instance_type)) if instance_type else None
        )
        gpi = int(spec.get("gpus_per_instance", 0) or 0) if spec else 0
        if gpi > 0:
            assert spec is not None
            allocation_kind = str(spec.get("allocation_kind") or "instance")
            per_replica = gpus if allocation_kind == "gpu" else max(1, -(-gpus // gpi))
            reserved_gpus = reps * gpus if allocation_kind == "gpu" else reps * per_replica * gpi
            cost[("gpu", env_key)] = cost.get(("gpu", env_key), 0) + reserved_gpus
            key = ("pool", env_key, str(instance_type))
            cost[key] = cost.get(key, 0) + reps * per_replica
        else:
            cost[("gpu", env_key)] = cost.get(("gpu", env_key), 0) + gpus * reps
    return cost


def _pending_deployment_capacity(specs: dict[str, Any]) -> dict[tuple, int]:
    """Capacity reserved by acknowledged requests that have not materialized yet."""
    reserved: dict[tuple, int] = {}
    try:
        active_jobs = get_active_jobs()
    except (AttributeError, RuntimeError):
        active_jobs = []
    jobs = {
        str(job.get("job_id")): job
        for job in [*get_pending_jobs(), *active_jobs]
        if job.get("job_id") is not None
    }
    for job in jobs.values():
        if job.get("deployment_status") != "deployment_pending":
            continue
        ranks = []
        for shape in job.get("last_requested_shapes") or []:
            tp = int(shape.get("tp") or 1)
            pp = int(shape.get("pp") or 1)
            env = shape.get("env") or []
            ranks.append(
                {
                    "env": env.split("|") if isinstance(env, str) else list(env),
                    "config": {
                        "instance_type": shape.get("instance_type"),
                        "gpu_count": int(shape.get("gpu_count") or tp * pp),
                        "tp": tp,
                        "pp": pp,
                    },
                    "n_replicas": int(shape.get("n_replicas") or 1),
                }
            )
        for key, amount in _ladder_capacity_cost(ranks, specs).items():
            reserved[key] = reserved.get(key, 0) + int(amount)
    return reserved


def _largest_pow2_divisor_leq(heads: int | None, cap: int) -> int:
    """Fill-tp for ONE instance: the largest power of 2 that divides `heads` and is
    <= cap (the instance's GPU count); 1 if heads unknown. Under instance-atomic
    accounting a rank reserves the WHOLE instance, so a smaller tp would just idle
    the rest of the box - use as many of its GPUs as can shard the model. Scaling
    THROUGHPUT past one instance comes from explicit fixed-DP candidate variants
    and extra heterogeneous ranks, NOT from a smaller tp inside one box."""
    if not heads or int(heads) <= 0 or cap < 1:
        return 1
    tp, power = 1, 2
    while power <= cap and int(heads) % power == 0:
        tp, power = power, power * 2
    return tp


def _replica_options(cap: int) -> list[int]:
    """Bounded fixed-DP alternatives, exact for common small replica pools."""
    options = list(range(1, min(cap, 8) + 1))
    replicas = 16
    while replicas <= cap:
        options.append(replicas)
        replicas *= 2
    if cap > 0 and (not options or options[-1] != cap):
        options.append(cap)
    return options


def _generated_tp_options(
    *,
    heads: int | None,
    gpu_cap: int,
    gpu_type: str,
    model_id: str,
    allocation_kind: str,
) -> list[int]:
    """Return policy-valid TP alternatives for one generated pool frame."""
    rule = load_config_policy().rule_for(gpu_type, model_id)
    if rule is not None:
        return [
            tp for tp in rule.allowed_tp if tp <= gpu_cap and (not heads or int(heads) % tp == 0)
        ]
    max_tp = MODEL_MAX_TP.get(model_id)
    if max_tp is not None:
        return [
            tp
            for tp in sorted(VALID_TP_DEGREES)
            if tp <= min(gpu_cap, max_tp) and (not heads or int(heads) % tp == 0)
        ]
    if allocation_kind == "gpu":
        options = []
        tp = 1
        while tp <= gpu_cap and (not heads or int(heads) % tp == 0):
            options.append(tp)
            tp *= 2
        return options
    return [_largest_pow2_divisor_leq(heads, gpu_cap)]


# pp=2 and pp=4 are generated. Deeper pipelines than 4 remain off: under the
# pre-2026-09 simulator pp>=4 never ran a decode step; the engine fix landed
# 2026-09-01 and pp=4 is now servable, so it is offered when pp=1 provably
# cannot hold the weights and pp=4 provably can.
_GENERATED_PP_DEGREES = (2, 4)


def _frame_shape_key(rank: dict[str, Any]) -> tuple[str, int, int]:
    """(gpu_type, tp, pp) of a candidate rank - the identity a dead-shape record keys on."""
    env = rank.get("env") or []
    config = rank.get("config") or {}
    gpu_type = str(env[4]) if isinstance(env, (list, tuple)) and len(env) >= 5 else ""
    try:
        return (gpu_type, int(config.get("tp") or 1), int(config.get("pp") or 1))
    except (TypeError, ValueError):
        return (gpu_type, 1, 1)


def _dead_shape_keys(job: dict[str, Any]) -> set[tuple[str, int, int]]:
    """Shapes S0 recorded as dead for this job's model: ran and served nothing
    under load, or failed to launch for a deterministic reason. Replica count is
    deliberately not part of the key - retrying a dead shape with more replicas
    is what the retry loop used to do."""
    keys: set[tuple[str, int, int]] = set()
    for entry in job.get("observed_dead_shapes") or []:
        if not isinstance(entry, dict):
            continue
        try:
            keys.add(
                (
                    str(entry.get("gpu_type") or ""),
                    int(entry.get("tp") or 1),
                    int(entry.get("pp") or 1),
                )
            )
        except (TypeError, ValueError):
            continue
    return keys


def _generated_pp_options(
    *,
    tp: int,
    gpu_cap: int,
    layers: int | None,
    weight_fit_values: dict[str, Any],
    gpu_mem_gb: float | None,
) -> list[int]:
    """Return PP degrees to generate for one (pool, tp) frame.

    pp=1 is always kept, so today's frame set is unchanged. Extra PP degrees are
    added ONLY when the model's weights provably do not fit one GPU at pp=1 and
    provably do fit at that pp - the cheap analytic check, no surrogate call.
    For MoE models, TP already shards expert tensors at Koi's fixed EP=1. PP is
    added only when those TP-sharded weights still do not fit. A PP degree must
    divide the layer count and keep tp*pp inside one instance.
    """
    options = [1]
    if not weight_fit_values or gpu_mem_gb is None:
        return options
    base = {**weight_fit_values, "gpu_mem_gb": gpu_mem_gb, "tp": tp, "ep": SUPPORTED_EP}
    if target_memory_fit({**base, "pp": 1}).get("status") != "physical_no_fit":
        return options
    for pp in _GENERATED_PP_DEGREES:
        if tp * pp > gpu_cap or (layers is not None and layers % pp != 0):
            continue
        if target_memory_fit({**base, "pp": pp}).get("status") == "physical_no_fit":
            continue
        options.append(pp)
    return options


def _applicable_mechanism_id(rank: dict[str, Any], features: dict[str, Any]) -> str | None:
    """Best applicable mechanism id for a rank (exact, then partial), or None."""
    if ablation.mechanism_inert():
        # Mechanism identity carries no decision content in this mode; every
        # rank commits to the one pass-through sentinel.
        return ablation.passthrough_mechanism_id()
    try:
        apps = get_applicable_mechanisms(rank, features)
    except Exception:
        return None
    preferred = rank.get("mechanism_id")
    if preferred and isinstance(apps, (list, tuple)):
        applicable_ids = {
            item if isinstance(item, str) else item.get("mechanism_id")
            for item in apps
            if isinstance(item, str | dict)
        }
        if preferred in applicable_ids:
            return str(preferred)
    if isinstance(apps, dict):
        mid = apps.get("exact") or apps.get("mechanism_id")
        if mid:
            return mid
        vals = apps.get("mechanisms") or apps.get("applicable") or []
        if vals:
            return _best_mechanism_id(vals)
    elif isinstance(apps, (list, tuple)) and apps:
        return _best_mechanism_id(apps)
    return None


def _best_mechanism_id(briefs) -> str | None:
    """Pick the most trustworthy applicable mechanism, not merely the first.

    Registry order carries no meaning, so taking briefs[0] pinned every rank to
    whichever mechanism happened to be registered first. That made the committed
    mechanism constant across a run, which in turn kept EIG at zero on every
    accepted candidate and left the exploration term in sigma inert. Rank on the
    signals the brief already carries: exact scope match first, then confidence,
    then visits as a tie-break so a well-evidenced mechanism outranks a fresh one.
    """

    def number(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return parsed if math.isfinite(parsed) else 0.0

    best_id: str | None = None
    best_score: tuple[int, float, int] | None = None
    for brief in briefs:
        mechanism_id: str | None
        if isinstance(brief, str):
            mechanism_id, quality, confidence, visits = brief, "", 0.0, 0
        elif isinstance(brief, dict):
            raw_mechanism_id = brief.get("mechanism_id")
            mechanism_id = str(raw_mechanism_id) if raw_mechanism_id else None
            quality = str(brief.get("match_quality") or "")
            confidence = number(brief.get("c"))
            visits = int(number(brief.get("visit_count")))
        else:
            continue
        if not mechanism_id:
            continue
        score = (1 if quality == "exact" else 0, float(confidence), visits)
        if best_score is None or score > best_score:
            best_id, best_score = str(mechanism_id), score
    return best_id


def _online_slo_targets(features: dict[str, Any]) -> dict[str, Any]:
    """Online latency SLO targets to carry ON the emitted action (None for batch).
    compute_sigma reads these from job_features, but the deployed action must also
    carry them - Orca/Dynamo route on them - so we copy them onto every place act."""
    if _job_mode(features) != "online":
        return {"target_p99_ttft_ms": None, "target_p99_tpot_ms": None}
    return {
        "target_p99_ttft_ms": _feature_value(features, "target_p99_ttft_ms", "target_p99_TTFT_ms"),
        "target_p99_tpot_ms": _feature_value(features, "target_p99_tpot_ms", "target_p99_TPOT_ms"),
    }


def _online_sizing_rejection(
    sized: dict[str, Any], features: dict[str, Any]
) -> tuple[str, str] | None:
    """Return only deterministic hard failures; prediction quality is advisory."""
    del features
    if sized.get("failure_kind") == "hard":
        return str(sized.get("failure_status") or "invalid_config"), str(
            sized.get("failure_reason") or "candidate failed deterministic sizing"
        )
    return None


def _prediction_assessment(sized: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Summarize Direct's point estimate without implying queue verification."""
    status = str(sized.get("failure_status") or "success")
    semantics = dict(sized.get("prediction_semantics") or _DIRECT_PREDICTION_SEMANTICS)
    queue_state = "not_applicable" if _job_mode(features) != "online" else "unmodeled"
    queue_utilization = None
    queue_affects_selection = False
    if _job_mode(features) == "online" and status == "success":
        per_rank = [
            rank for rank in (sized.get("per_rank") or []) if int(rank.get("n_replicas") or 0) > 0
        ]
        try:
            shadow = estimate_queue_shadow(
                arrival_rate_rps=_feature_value(features, "request_arrival_rate"),
                input_tokens_per_request=_feature_value(
                    features, "isl_token_avg", "input_len_tokens_avg"
                ),
                input_tokens_per_request_max=_feature_value(
                    features, "isl_token_max", "input_len_tokens_max"
                ),
                output_tokens_per_request=_feature_value(
                    features, "osl_token_avg", "output_len_tokens_avg"
                ),
                output_tokens_per_request_max=_feature_value(
                    features, "osl_token_max", "output_len_tokens_max"
                ),
                aggregate_capacity_tps=sized.get("point_capacity_tps"),
                replicas=sum(int(rank.get("n_replicas") or 0) for rank in per_rank),
                base_ttft_ms=max(
                    (
                        float(rank["base_p99_ttft_ms"])
                        for rank in per_rank
                        if rank.get("base_p99_ttft_ms") is not None
                    ),
                    default=None,
                ),
                homogeneous=len(per_rank) == 1,
                scenario="peak",
                peak_to_mean_ratio=_feature_value(features, "peak_to_mean_ratio") or 1.0,
                affects_selection=True,
            )
            queue_state = str(shadow.get("status") or "unmodeled")
            queue_utilization = shadow.get("utilization")
            queue_token_work_pressure = shadow.get("combined_token_work_pressure")
            queue_tail_token_work_pressure = shadow.get("tail_token_work_pressure")
            queue_affects_selection = bool(shadow.get("affects_selection"))
        except Exception:
            queue_state = "unmodeled"
            queue_token_work_pressure = None
            queue_tail_token_work_pressure = None
    else:
        queue_token_work_pressure = None
        queue_tail_token_work_pressure = None
    return {
        **semantics,
        "kind": "exploratory" if status in _SOFT_PREDICTION_FAILURES else "point",
        "status": status,
        "reason": sized.get("failure_reason"),
        "point_capacity_tps": sized.get("point_capacity_tps"),
        "capacity_fraction": sized.get("capacity_fraction"),
        "base_latency_within_target": sized.get("base_latency_within_target"),
        "queue_state": queue_state,
        "queue_utilization": queue_utilization,
        "queue_token_work_pressure": queue_token_work_pressure,
        "queue_tail_token_work_pressure": queue_tail_token_work_pressure,
        "queue_affects_selection": queue_affects_selection,
    }


def _sized_online_slo_risk(sized: dict[str, Any], features: dict[str, Any]) -> bool:
    """Whether a usable Direct point has a base-latency target miss."""
    if _job_mode(features) != "online" or sized.get("failure_kind") == "soft":
        return False
    serving = [
        rank for rank in (sized.get("per_rank") or []) if int(rank.get("n_replicas") or 0) >= 1
    ]
    return any(
        rank.get("prediction_complete") is True and rank.get("slo_ok") is False for rank in serving
    )


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
    raw_replicas = raw.get("n_replicas", 1)
    if isinstance(raw_replicas, bool):
        return None
    try:
        n_replicas = int(raw_replicas)
        exact = float(raw_replicas) == n_replicas
    except (TypeError, ValueError, OverflowError):
        return None
    if n_replicas < 1 or not exact:
        return None
    normalized = {
        "role": "aggregate",
        "env": list(env),
        "config": config,
        "n_replicas": n_replicas,
    }
    if isinstance(raw.get("mechanism_id"), str) and raw["mechanism_id"]:
        normalized["mechanism_id"] = raw["mechanism_id"]
    return normalized


def _rank_shape_key(rank: dict[str, Any]) -> tuple:
    cfg = rank.get("config") or {}
    return (
        tuple(rank.get("env") or []),
        cfg.get("instance_type"),
        cfg.get("tp"),
        cfg.get("pp"),
        cfg.get("sp", 1),
        cfg.get("ep", SUPPORTED_EP),
        cfg.get("cp", 1),
        cfg.get("gpu_count"),
        cfg.get("num_nodes_per_chain", 1),
        cfg.get("interconnect_type"),
        rank.get("n_replicas", 1),
        rank.get("rank_traffic_share"),
        rank.get("mechanism_id"),
    )


def _deployment_shape_key(raw: dict[str, Any], replicas: int | None = None) -> tuple:
    return deployment_rank_identity(raw, replicas=replicas)


def _candidate_budget_errors(jid: str, action: dict[str, Any]) -> list[str]:
    """Return BudgetSlice violations when a validated book is active."""
    book = getattr(_CTX, "validated_budget_book", None)
    if not isinstance(book, dict):
        return []
    slice_ = (book.get("job_budgets") or {}).get(jid)
    if not isinstance(slice_, dict):
        return ["validated BudgetBook has no slice for this job"]
    if action.get("budget_ref") != slice_.get("slice_id"):
        return ["candidate does not reference its validated BudgetSlice"]
    return _budget_violations(PlanAction.from_dict(action), slice_)


def _score_one_frame(
    jid: str,
    user_id: Any,
    slice_id: Any,
    rank: dict[str, Any],
    features: dict[str, Any],
    action_type: str = "place",
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
        "slo_risk": False,
    }
    runnable, reason = config_runnable(cfg, features, gpu_type=gpu_type)
    if not runnable:
        diag.update(status="invalid_config", reason=reason)
        return {"candidate": None, "meets_target": False, "diag": diag}
    mid = _applicable_mechanism_id(rank, features)
    if not mid:
        diag.update(status="no_mechanism", reason="no applicable mechanism")
        return {"candidate": None, "meets_target": False, "diag": diag}
    scored_rank = dict(rank)
    scored_rank["mechanism_id"] = mid
    budget_errors = _candidate_budget_errors(
        jid,
        {
            "job_id": jid,
            "type": action_type,
            "user_id": user_id,
            "ladder": [scored_rank],
            "mechanism_id": mid,
            "budget_ref": slice_id,
        },
    )
    if budget_errors:
        diag.update(status="resource_budget", reason="; ".join(budget_errors)[:200])
        return {"candidate": None, "meets_target": False, "diag": diag}
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
    served_fraction = min(1.0, achieved_tps / target_tps) if target_tps > 0 else 0.0
    slo_risk = _sized_online_slo_risk(sized, features)
    diag.update(
        achieved_tps=achieved_tps,
        target_tps=target_tps,
        meets_target=meets,
        unmet_tps=unmet_tps,
        served_fraction=served_fraction,
        slo_risk=slo_risk,
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
        failure_status = sized.get("failure_status")
        diag.update(
            status=failure_status or "prediction_incomplete",
            reason=sized.get("failure_reason")
            or "no deployable rank produced a complete prediction",
        )
        return {"candidate": None, "meets_target": False, "diag": diag}
    assessment = _prediction_assessment(sized, features)
    assessment["proposal_source"] = rank.get("proposal_source", "specialist")
    exploratory = assessment["kind"] == "exploratory"
    if action_type == "swap" and not exploratory:
        health = features.get("_active_health") or {}
        observed = health.get("observed") or {}
        reasons = set(health.get("reasons") or [])
        per_rank = sized.get("per_rank") or []
        predicted_ttft = max(
            (
                float(item["base_p99_ttft_ms"])
                for item in per_rank
                if item.get("base_p99_ttft_ms") is not None
            ),
            default=None,
        )
        predicted_tpot = max(
            (
                float(item["base_p99_tpot_ms"])
                for item in per_rank
                if item.get("base_p99_tpot_ms") is not None
            ),
            default=None,
        )
        predicted_capacity = float(sized.get("point_capacity_tps") or 0.0)
        current_throughput = observed.get("throughput_token_per_sec")
        current_ttft = observed.get("p99_ttft_ms")
        current_tpot = observed.get("p99_tpot_ms")
        capacity_reasons = {
            "zero_throughput",
            "throughput_shortfall",
            "queue_growing",
            "queue_critical",
        }
        needs_capacity_repair = bool(reasons & capacity_reasons)
        capacity_acceptable = (
            predicted_capacity >= target_tps
            if target_tps > 0
            else current_throughput is not None and predicted_capacity >= float(current_throughput)
        )
        baseline_available = any(
            value is not None for value in (current_throughput, current_ttft, current_tpot)
        )
        not_worse = (
            capacity_acceptable
            and (
                current_ttft is None
                or (predicted_ttft is not None and predicted_ttft <= float(current_ttft))
            )
            and (
                current_tpot is None
                or (predicted_tpot is not None and predicted_tpot <= float(current_tpot))
            )
        )
        improves = (
            (
                "ttft_breach" in reasons
                and predicted_ttft is not None
                and current_ttft is not None
                and predicted_ttft < 0.9 * float(current_ttft)
            )
            or (
                "tpot_breach" in reasons
                and predicted_tpot is not None
                and current_tpot is not None
                and predicted_tpot < 0.9 * float(current_tpot)
            )
            or (needs_capacity_repair and capacity_acceptable)
        )
        if reasons and (not baseline_available or not improves or not not_worse):
            diag.update(
                status="no_rehabilitation_gain",
                reason="SWAP is not Pareto-better than the unhealthy KEEP baseline",
                prediction_assessment=assessment,
            )
            return {"candidate": None, "meets_target": False, "diag": diag}
    if exploratory:
        service_label = f"exploratory ({assessment['status']})"
    elif slo_risk:
        service_label = "base-latency target missed; queue behavior unmodeled"
    elif meets:
        service_label = "point capacity covers target; queue behavior unmodeled"
    else:
        service_label = "point capacity below target; queue behavior unmodeled"
    act = {
        "job_id": jid,
        "type": action_type,
        "user_id": user_id,
        "ladder": ranks,
        "target_tps": target_tps,
        "mechanism_id": mid,
        "budget_ref": slice_id,
        "prediction_assessment": assessment,
        "point_capacity_covers_target": bool(target_tps > 0 and achieved_tps >= target_tps),
        "base_latency_within_target": sized.get("base_latency_within_target"),
        "queue_state": assessment.get("queue_state"),
        "queue_slo_verified": False,
        "service_class": (
            "exploratory"
            if exploratory
            else "supported"
            if meets and not slo_risk and assessment.get("queue_state") != "unstable"
            else "partial"
        ),
        **_online_slo_targets(features),
        "rationale": f"Deterministic {gpu_type} candidate ({service_label}).",
    }
    if action_type == "swap":
        act["swap_reason"] = "replace"
        health = features.get("_active_health") or {}
        act["rehabilitation_status"] = health.get("status")
        act["rehabilitation_reasons"] = list(health.get("reasons") or [])
    if not exploratory and target_tps > 0:
        act.update(
            achieved_tps=achieved_tps,
            unmet_tps=unmet_tps,
            meets_target=meets,
            served_fraction=served_fraction,
        )
    one = {"tick_rationale": "candidate scoring", "actions": [act]}
    try:
        budget_errors = _candidate_budget_errors(jid, act)
        if budget_errors:
            diag.update(status="resource_budget", reason="; ".join(budget_errors)[:200])
            return {"candidate": None, "meets_target": meets, "diag": diag}
        if exploratory:
            score = {
                "sigma": 0.0,
                "scoring_mode": "exploration_only",
                "prediction_available": False,
            }
        else:
            score = compute_sigma(one)["per_job"][jid]
        act["sigma"] = score["sigma"]
        if action_type == "swap":
            act["keep_baseline_sigma"] = score.get("keep_baseline_sigma")
            act["swap_gain_over_keep"] = score.get("swap_gain_over_keep")
        feas = check_feasibility(one)
        if not feas.get("feasible"):
            diag.update(status="infeasible", reason="; ".join(feas.get("violations", []))[:200])
            return {"candidate": None, "meets_target": meets, "diag": diag}
    except SurrogateBudgetExceeded as exc:
        diag.update(status="budget_exhausted", reason=str(exc))
        return {"candidate": None, "meets_target": meets, "diag": diag}
    except Exception as exc:
        diag.update(status="score_error", reason=f"scoring failed: {exc}")
        return {"candidate": None, "meets_target": meets, "diag": diag}
    diag.update(status=assessment["status"] if exploratory else "ok", **score)
    diag["prediction_assessment"] = assessment
    if slo_risk:
        diag["reason"] = "Direct base-latency target missed; queue behavior is unmodeled"
    return {"candidate": act, "meets_target": meets, "diag": diag}


def _score_composite(
    jid: str, user_id: Any, slice_id: Any, ranks: list[dict[str, Any]], features: dict[str, Any]
) -> dict[str, Any]:
    """Score ONE heterogeneous, data-parallel multi-rank ladder for a job.

    size_ladder evaluates every supplied fixed-DP rank and sums point capacities
    across pools. This is how a big job can exceed one pool's capacity (for example,
    H100 across p5.48xlarge + p5.4xlarge, or an H100 rank plus an A100 rank). The
    caller orders ranks by preference. ALWAYS returns a {candidate,
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
        "slo_risk": False,
    }
    scored_ranks: list[dict[str, Any]] = []
    for rank in ranks:
        runnable, _ = config_runnable(
            dict(rank.get("config") or {}), features, gpu_type=env_gpu_type(rank.get("env"))
        )
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
    budget_errors = _candidate_budget_errors(
        jid,
        {
            "job_id": jid,
            "type": "place",
            "user_id": user_id,
            "ladder": scored_ranks,
            "mechanism_id": scored_ranks[0].get("mechanism_id"),
            "budget_ref": slice_id,
        },
    )
    if budget_errors:
        diag.update(status="resource_budget", reason="; ".join(budget_errors)[:200])
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
    served_fraction = min(1.0, achieved_tps / target_tps) if target_tps > 0 else 0.0
    slo_risk = _sized_online_slo_risk(sized, features)
    label = "+".join(str((r.get("config") or {}).get("instance_type")) for r in sized_ranks) or None
    diag.update(
        instance_type=label,
        meets_target=meets,
        achieved_tps=achieved_tps,
        target_tps=target_tps,
        unmet_tps=unmet_tps,
        served_fraction=served_fraction,
        slo_risk=slo_risk,
        partial_search_probes=int(sized.get("partial_search_probes") or 0),
        partial_search_truncated=bool(sized.get("partial_search_truncated")),
    )
    online_rejection = _online_sizing_rejection(sized, features)
    if online_rejection is not None:
        status, reason = online_rejection
        diag.update(status=status, reason=reason)
        return {"candidate": None, "meets_target": False, "diag": diag}
    if not sized_ranks:
        diag.update(
            status=sized.get("failure_status") or "prediction_incomplete",
            reason=sized.get("failure_reason")
            or "no composite rank produced a complete prediction",
        )
        return {"candidate": None, "meets_target": False, "diag": diag}
    assessment = _prediction_assessment(sized, features)
    assessment["proposal_source"] = "composite"
    exploratory = assessment["kind"] == "exploratory"
    if exploratory:
        diag.update(
            status=assessment["status"],
            reason=(
                "prediction-uncertain composite has no defensible traffic split; "
                "retain its physically valid single-rank alternatives"
            ),
            prediction_assessment=assessment,
        )
        return {"candidate": None, "meets_target": False, "diag": diag}
    if len(sized_ranks) < 2:
        # size_ladder covered the target (or ran out) on ONE rank - no composite;
        # the single-frame candidate already represents it.
        diag.update(
            status="no_composite",
            reason=f"size_ladder used {len(sized_ranks)} rank(s) of {len(scored_ranks)}",
        )
        return {"candidate": None, "meets_target": meets, "diag": diag}
    if exploratory:
        service_state = f"exploratory ({assessment['status']})"
    elif slo_risk:
        service_state = "base-latency target missed; queue behavior unmodeled"
    elif meets:
        service_state = "point capacity covers target; queue behavior unmodeled"
    else:
        service_state = "point capacity below target; queue behavior unmodeled"
    act = {
        "job_id": jid,
        "type": "place",
        "user_id": user_id,
        "ladder": sized_ranks,
        "target_tps": target_tps,
        "mechanism_id": sized_ranks[0].get("mechanism_id"),
        "budget_ref": slice_id,
        "prediction_assessment": assessment,
        "point_capacity_covers_target": bool(target_tps > 0 and achieved_tps >= target_tps),
        "base_latency_within_target": sized.get("base_latency_within_target"),
        "queue_state": assessment.get("queue_state"),
        "queue_slo_verified": False,
        "service_class": "supported" if meets and not slo_risk else "partial",
        **_online_slo_targets(features),
        "rationale": f"Deterministic composite candidate ({label}) ({service_state}).",
    }
    if not exploratory and target_tps > 0:
        act.update(
            achieved_tps=achieved_tps,
            unmet_tps=unmet_tps,
            meets_target=meets,
            served_fraction=served_fraction,
        )
    one = {"tick_rationale": "candidate scoring", "actions": [act]}
    try:
        budget_errors = _candidate_budget_errors(jid, act)
        if budget_errors:
            diag.update(status="resource_budget", reason="; ".join(budget_errors)[:200])
            return {"candidate": None, "meets_target": meets, "diag": diag}
        feas = check_feasibility(one)
        if not feas.get("feasible"):
            diag.update(status="infeasible", reason="; ".join(feas.get("violations", []))[:200])
            return {"candidate": None, "meets_target": meets, "diag": diag}
        if exploratory:
            score = {
                "sigma": 0.0,
                "scoring_mode": "exploration_only",
                "prediction_available": False,
            }
        else:
            score = compute_sigma(one)["per_job"][jid]
        act["sigma"] = score["sigma"]
    except SurrogateBudgetExceeded as exc:
        diag.update(status="budget_exhausted", reason=str(exc))
        return {"candidate": None, "meets_target": meets, "diag": diag}
    except Exception as exc:
        diag.update(status="score_error", reason=f"scoring failed: {exc}")
        return {"candidate": None, "meets_target": meets, "diag": diag}
    diag.update(status=assessment["status"] if exploratory else "ok", **score)
    diag["prediction_assessment"] = assessment
    if slo_risk:
        diag["reason"] = "Direct base-latency target missed; queue behavior is unmodeled"
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


def _empty_frame_result(status: str, reason: str, **details: Any) -> dict[str, Any]:
    """Return a typed diagnostic for a job intentionally given no frames."""
    return {
        "candidate": None,
        "meets_target": False,
        "diag": {
            "env": None,
            "instance_type": None,
            "tp": None,
            "status": status,
            "reason": reason,
            "meets_target": False,
            "achieved_tps": None,
            "target_tps": None,
            "sigma": None,
            **details,
        },
    }


def build_scored_candidates(
    budget_book: dict[str, Any] | None = None,
    specialist_results: Any = None,
) -> dict[str, Any]:
    """Deterministic candidate pipeline for all waiting jobs: normalize specialist
    ladders (HINTS), then generate fixed-DP alternatives for each available pool at
    fill-tp (the largest power of 2 that shards the model's heads and fits the
    instance's GPUs). Resource accounting is instance-atomic. Size and score each
    exact frame via the proven chain (config_runnable ->
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
    Direct results are point-capacity/base-latency estimates, not queue simulations.
    Prediction-only failures remain exploratory candidates; only deterministic hard
    failures remove a frame. A job appears in `exhausted` only when it has no
    physically valid candidate; budget-truncated jobs are reported separately.

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
    resources = copy.deepcopy(get_resource_map())
    specs = copy.deepcopy(instance_catalog())
    pending_reserved = _pending_deployment_capacity(specs)
    for env_key, info in resources.items():
        info["free"] = max(
            0,
            int(info.get("free", 0) or 0) - pending_reserved.get(("gpu", _env_key(env_key)), 0),
        )
    for env_key, pools in specs.items():
        for instance_type, spec in pools.items():
            pending_units = pending_reserved.get(("pool", _env_key(env_key), str(instance_type)), 0)
            spec["free_instances"] = max(0, int(spec.get("free_instances", 0) or 0) - pending_units)
    free_envs: list[tuple[str, list[str]]] = []
    for raw_env_key, info in sorted(resources.items(), key=lambda item: _env_key(item[0])):
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
    ctx_by_job: dict[str, tuple[Any, Any, dict[str, Any], str]] = {}
    blocked_by_job: dict[str, dict[str, Any]] = {}
    attempted_identities_by_job: dict[str, list[Any]] = {}
    pending_jobs = list(get_pending_jobs())
    active_rehabilitation = [
        job
        for job in get_active_jobs()
        if (job.get("health") or {}).get("rehabilitation_eligible") is True
    ]
    pending_ids = {str(job.get("job_id", job.get("id"))) for job in pending_jobs}
    # Read once per tick: per-GPU memory for the generated PP gate comes from the
    # same catalog deployment X uses, so the pre-rank check and the X check agree.
    try:
        hardware_catalog = _CTX.resource_map.hardware_catalog()
    except Exception:
        hardware_catalog = None

    def job_order(job: dict[str, Any]) -> tuple[float, str]:
        job_id = str(job.get("job_id", job.get("id", "")))
        raw_priority = (budgets.get(job_id) or {}).get("priority_score", 0.0)
        try:
            priority = float(raw_priority)
        except (TypeError, ValueError):
            priority = 0.0
        return -priority, job_id

    candidate_jobs = sorted([*pending_jobs, *active_rehabilitation], key=job_order)
    for job in candidate_jobs:
        jid = job.get("job_id", job.get("id"))
        if not jid:
            continue
        features = _job_features_for(snapshot, jid) or dict(job.get("job_features") or {})
        if str(jid) not in pending_ids:
            features["_active_health"] = copy.deepcopy(job.get("health") or {})
        model_id = features.get("model_id") or job.get("model_id")
        user_id = job.get("user_id") or features.get("user_id")
        slice_ = budgets.get(jid) or {}
        slice_id = slice_.get("slice_id", jid)
        heads = _model_num_heads({"model_id": model_id}, features)
        model_catalog = _model_catalog_for(model_id)
        layers = _model_num_layers(model_catalog, features)
        weight_fit_values = _model_weight_fit_values(model_catalog, features)

        frames: list[dict[str, Any]] = []
        seen: set = set()
        is_pending = str(jid) in pending_ids
        deployment_status = job.get("deployment_status")
        retry_action_type = "place" if is_pending else "swap"
        attempt_gate_applies = is_pending or job.get("deployment_action_type") == "swap"
        if attempt_gate_applies and deployment_status == "deployment_pending":
            frames_by_job[jid] = []
            ctx_by_job[jid] = (user_id, slice_id, features, retry_action_type)
            blocked_by_job[jid] = _empty_frame_result(
                "deployment_pending",
                "prior placement is still inside the materialization grace period",
                deployment_attempts=int(job.get("deployment_attempts") or 1),
            )
            continue
        if attempt_gate_applies and deployment_status == "deployment_not_materialized":
            attempts = int(job.get("deployment_attempts") or 1)
            retry_after = job.get("deployment_retry_after_tick")
            # A repeatedly-failing placement retires the SHAPE, not the job: the
            # descriptor carries attempted_deployment_identities and
            # recent_rank_failures so a retry can pick different hardware. Emptying
            # the frame list here removed the job from planning permanently, which
            # left jobs unserved while the cluster still had free capacity.
            if job.get("deployment_retry_allowed") is not True:
                frames_by_job[jid] = []
                ctx_by_job[jid] = (user_id, slice_id, features, retry_action_type)
                blocked_by_job[jid] = _empty_frame_result(
                    "deployment_retry_backoff",
                    "prior placement timed out; waiting for retry backoff",
                    deployment_attempts=attempts,
                    retry_after_tick=retry_after,
                )
                continue
        for raw in (spec_by_job.get(jid) or {}).get("ladder") or []:
            rank = _normalize_candidate_rank(raw)
            if rank is not None and _deployment_shape_key(rank) not in seen:
                rank["proposal_source"] = "specialist"
                seen.add(_deployment_shape_key(rank))
                frames.append(rank)
        # For every pool, enumerate bounded fixed-DP alternatives. Direct evaluates
        # each exact frame; size_ladder never mutates a specialist's replica count.
        # a 1-GPU box -> tp=1 (right for small models), an 8-GPU box -> tp=8 (right
        # for big ones). Accounting is instance-atomic, so a partial tp just idles
        # the box - fill it. Explicit fixed-DP variants scale across instances, and
        # Phase 2.5 spans pools when one is not enough. Specialist ladders above stay
        # as exact-capacity proposals (deduped by shape).
        generated_groups: list[list[dict[str, Any]]] = []
        generated_seen = set(seen)
        for env_key, env in free_envs:
            for instance_type, spec in sorted((specs.get(env_key) or {}).items()):
                if slice_ and env_key not in (slice_.get("env_budget") or {}):
                    continue
                if slice_ and instance_type not in (
                    (slice_.get("pool_budget") or {}).get(env_key) or {}
                ):
                    continue
                gpi = int(spec.get("gpus_per_instance", 0) or 0)
                if gpi <= 0 or int(spec.get("free_instances", 0) or 0) <= 0:
                    continue
                allocation_kind = spec.get("allocation_kind")
                gpu_cap = (
                    int(spec.get("candidate_gpu_cap", 0) or 0) if allocation_kind == "gpu" else gpi
                )
                tp_options = _generated_tp_options(
                    heads=heads,
                    gpu_cap=gpu_cap,
                    gpu_type=str(env[4]),
                    model_id=str(model_id or ""),
                    allocation_kind=str(allocation_kind or "instance"),
                )
                pool_gpu_mem_gb = (
                    hardware_gpu_memory_gb(hardware_catalog, env, instance_type)
                    if hardware_catalog is not None
                    else None
                )
                geometries = [
                    (tp, pp)
                    for tp in tp_options
                    for pp in _generated_pp_options(
                        tp=tp,
                        gpu_cap=gpu_cap,
                        layers=layers,
                        weight_fit_values=weight_fit_values,
                        gpu_mem_gb=pool_gpu_mem_gb,
                    )
                ]
                for tp, pp in geometries:
                    engine_gpus = tp * pp
                    replica_frames: list[dict[str, Any]] = []
                    max_replicas = int(spec.get("free_instances", 0) or 0)
                    capacity_per_replica = engine_gpus if allocation_kind == "gpu" else gpi
                    if allocation_kind == "gpu":
                        max_replicas //= max(1, engine_gpus)
                    env_budget = (slice_.get("env_budget") or {}).get(env_key)
                    if env_budget is not None:
                        max_replicas = min(
                            max_replicas,
                            int(env_budget) // capacity_per_replica,
                        )
                    pool_budget = ((slice_.get("pool_budget") or {}).get(env_key) or {}).get(
                        instance_type
                    )
                    if pool_budget is not None:
                        allowed_units = int(pool_budget)
                        if allocation_kind == "gpu":
                            allowed_units //= max(1, engine_gpus)
                        max_replicas = min(max_replicas, allowed_units)
                    replica_options = _replica_options(max_replicas) if slice_ else [1]
                    for replicas in replica_options:
                        rank = _normalize_candidate_rank(
                            {
                                "role": "aggregate",
                                "env": list(env),
                                "config": {
                                    "instance_type": instance_type,
                                    "gpu_count": engine_gpus,
                                    "tp": tp,
                                    "pp": pp,
                                },
                                "n_replicas": replicas,
                            }
                        )
                        if rank is not None and _deployment_shape_key(rank) not in generated_seen:
                            rank["proposal_source"] = "generated"
                            generated_seen.add(_deployment_shape_key(rank))
                            replica_frames.append(rank)
                    if replica_frames:
                        generated_groups.append(replica_frames)
        max_generated_variants = max((len(group) for group in generated_groups), default=0)
        for variant_index in range(max_generated_variants):
            for group in generated_groups:
                if variant_index < len(group):
                    rank = group[variant_index]
                    seen.add(_deployment_shape_key(rank))
                    frames.append(rank)
        if not is_pending:
            current_groups: dict[str, list[dict[str, Any]]] = {}
            for index, chain in enumerate(
                job.get("current_ladder") or job.get("active_chains") or []
            ):
                shape = dict(chain.get("shape_json") or chain)
                rank_id = str(shape.get("rank_id") or f"chain-{index}")
                current_groups.setdefault(rank_id, []).append(chain)
            current_shapes = {
                _deployment_shape_key(chains[0], replicas=len(chains))
                for chains in current_groups.values()
            }
            frames = [rank for rank in frames if _deployment_shape_key(rank) not in current_shapes]
        dead_keys = _dead_shape_keys(job)
        if dead_keys:
            before_dead_filter = len(frames)
            frames = [rank for rank in frames if _frame_shape_key(rank) not in dead_keys]
            if before_dead_filter > 0 and not frames:
                blocked_by_job[jid] = _empty_frame_result(
                    "deployment_shape_observed_dead",
                    "every candidate shape already ran for this model this run and "
                    "served nothing under load, or failed to launch",
                )
        if attempt_gate_applies and deployment_status == "deployment_not_materialized":
            attempted = list(job.get("attempted_deployment_identities") or [])
            attempted_identities_by_job[jid] = attempted
            before_retry_filter = len(frames)
            frames = [
                rank
                for rank in frames
                if not any(deployment_ladder_identity([rank]) == identity for identity in attempted)
            ]
            if before_retry_filter > 0 and not frames:
                blocked_by_job[jid] = _empty_frame_result(
                    "deployment_shape_already_attempted",
                    "all generated deployment shapes have already timed out",
                    deployment_attempts=int(job.get("deployment_attempts") or 1),
                    retry_after_tick=job.get("deployment_retry_after_tick"),
                )
        if not frames and jid not in blocked_by_job:
            blocked_by_job[jid] = _empty_frame_result(
                "no_pool_capacity",
                "no policy-valid candidate frame has free pool capacity",
            )
        frames_by_job[jid] = frames
        ctx_by_job[jid] = (
            user_id,
            slice_id,
            features,
            "place" if str(jid) in pending_ids else "swap",
        )

    # Phase 2 (surrogate-heavy): deterministic round-robin over jobs. The surrogate
    # is globally serialized, so threads add no throughput and obscure which jobs
    # received the final calls. Specialist frame 0 leads for every job before any
    # job receives frame 1.
    scored_by_job: dict[str, list[dict[str, Any]]] = {
        jid: [blocked_by_job[jid]] if jid in blocked_by_job else [] for jid in frames_by_job
    }
    budget_exhausted = False
    max_frames = max((len(frames) for frames in frames_by_job.values()), default=0)
    for frame_index in range(max_frames):
        for jid, frames in frames_by_job.items():
            if frame_index >= len(frames):
                continue
            user_id, slice_id, features, action_type = ctx_by_job[jid]
            scored = (
                _score_one_frame(jid, user_id, slice_id, frames[frame_index], features)
                if action_type == "place"
                else _score_one_frame(
                    jid,
                    user_id,
                    slice_id,
                    frames[frame_index],
                    features,
                    action_type="swap",
                )
            )
            scored_by_job[jid].append(scored)
            if scored.get("diag", {}).get("status") == "budget_exhausted":
                budget_exhausted = True
                break
        if budget_exhausted:
            break

    if budget_exhausted:
        for jid, frames in frames_by_job.items():
            results = scored_by_job[jid]
            scored_frame_count = 0 if jid in blocked_by_job else len(results)
            for rank in frames[scored_frame_count:]:
                results.append(_budget_skipped_frame(rank))

    # Phase 2.5 (heterogeneous composites). Two motivations:
    #  - CAPACITY: if NO single pool meets the target, span pools (fill the
    #    highest-throughput ones first) so a big job is served across pools.
    #  - COST (on-demand market only, w_cost>0): ALSO try a cheapest-$/token-first
    #    mix even when a single pool already meets, so a heterogeneous placement
    #    (cheap pool + top-up) can beat the single-pool winner on cost. In reserved
    #    (w_cost=0) the fleet is sunk - a "cheaper" mix saves nothing - so we only do
    #    the capacity fallback.
    # size_ladder evaluates each ordering and sums point capacity; the joint solver ranks every
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
                if r.get("composite_eligible")
                or (
                    r.get("candidate") is not None
                    and not _is_exploratory_assessment(
                        (r.get("candidate") or {}).get("prediction_assessment")
                    )
                )
            ]
            if not composable:
                continue
            # Fixed-DP generation emits several replica counts per pool. A composite
            # may use at most one of them; otherwise it double-books the same pool as
            # separate ranks. Keep the best scored frame for each pool.
            best_by_pool: dict[tuple[str, str | None], tuple[dict[str, Any], dict[str, Any]]] = {}
            for frame, result in composable:
                cfg = frame.get("config") or {}
                key = (_env_key(frame.get("env")), cfg.get("instance_type"))
                current = best_by_pool.get(key)
                rank_key = (
                    bool(result.get("meets_target")),
                    float((result.get("candidate") or {}).get("sigma", float("-inf"))),
                    float(result.get("diag", {}).get("achieved_tps") or 0.0),
                )
                if current is None:
                    best_by_pool[key] = (frame, result)
                    continue
                current_key = (
                    bool(current[1].get("meets_target")),
                    float((current[1].get("candidate") or {}).get("sigma", float("-inf"))),
                    float(current[1].get("diag", {}).get("achieved_tps") or 0.0),
                )
                if rank_key > current_key:
                    best_by_pool[key] = (frame, result)
            composable = list(best_by_pool.values())
            user_id, slice_id, features, action_type = ctx_by_job[jid]
            if action_type == "swap":
                continue
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
                if any(
                    deployment_ladder_identity(order) == identity
                    for identity in attempted_identities_by_job.get(jid, [])
                ):
                    continue
                composite = _score_composite(jid, user_id, slice_id, order, features)
                scored_by_job[jid].append(composite)
                if composite.get("diag", {}).get("status") == "budget_exhausted":
                    budget_exhausted = True
                    break
            if budget_exhausted:
                break

    if budget_exhausted:
        for _jid, scored_list in scored_by_job.items():
            composable_results = [
                item
                for item in scored_list
                if item.get("candidate") is not None or item.get("composite_eligible")
            ]
            single_meets = any(item.get("meets_target") for item in composable_results)
            composite_was_relevant = len(composable_results) >= 2 and (
                not single_meets or w_cost > 0
            )
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
        elif not limited and not any(
            diagnostic.get("status") in {"deployment_pending", "deployment_retry_backoff"}
            for diagnostic in job_diag
        ):
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
    per-env free-GPU capacity. Advisory positive-throughput frames receive a tiny
    positive gain only when their normal gain is non-positive, making otherwise
    idle capacity work-conserving without displacing normally valuable work. This
    is the joint decision the greedy per-job loop cannot make: it weighs every
    job's GPU options together, so a scarce type (e.g. H100) goes to whichever job
    it helps most instead of being pre-split blindly. It ARBITRATES the frames you
    pass; it does NOT invent them. Proposing the right GPU types (an L40S frame and
    an H100 frame for a big model) is the planner's domain-knowledge job - this
    tool just picks the joint optimum among them.

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
    global _placement_decision_sequence
    _placement_decision_sequence += 1
    decision_index = _placement_decision_sequence
    reserve_map = {_env_key(env): int(n) for env, n in (reserves or {}).items()}
    resources = get_resource_map()
    specs = instance_catalog()
    pending_reserved = _pending_deployment_capacity(specs)
    original_candidates = list(candidates or [])

    def selection_score(candidate: dict[str, Any]) -> float:
        field = (
            "swap_gain_over_keep" if str(candidate.get("type") or "").lower() == "swap" else "sigma"
        )
        raw_value = candidate.get(field)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            return float("-inf")
        value = float(raw_value)
        return value if math.isfinite(value) else float("-inf")

    def emergency_recovery(candidate: dict[str, Any]) -> bool:
        # Health already defines critical (zero throughput, throughput under 10% of
        # required, queue depth >= 100, or TTFT >= 3x target). Re-deriving a
        # narrower subset here left jobs at 27x target latency without a swap
        # because their queue sat just under 100.
        return (
            str(candidate.get("type") or "").lower() == "swap"
            and candidate.get("rehabilitation_status") == "critical"
        )

    eligible_candidates = []
    selection_diagnostics: dict[str, dict[str, Any]] = {}
    eligible_swaps_by_job: dict[str, int] = {}
    rescue_by_job: dict[str, tuple[float, dict[str, Any]]] = {}
    for candidate in original_candidates:
        jid = str(candidate.get("job_id") or "")
        diag = selection_diagnostics.setdefault(
            jid,
            {
                "candidates": 0,
                "vetoed_gain_le_0": 0,
                "vetoed_unstable": 0,
                "eligible": 0,
                "best_gain": None,
                "chosen": False,
            },
        )
        diag["candidates"] += 1
        score = selection_score(candidate)
        if math.isfinite(score) and (diag["best_gain"] is None or score > diag["best_gain"]):
            diag["best_gain"] = round(score, 2)
        is_swap = str(candidate.get("type") or "").lower() == "swap"
        if is_swap and score <= 0:
            diag["vetoed_gain_le_0"] += 1
            # A critical job must always be able to TRY somewhere else, even when
            # every alternative scores below its keep baseline - a wrong baseline
            # or a wrong prediction otherwise pins it to a failing GPU for the
            # run. Remember its best-scoring reject; it is admitted below if
            # nothing better survives, floored so it can never outrank real gains.
            if emergency_recovery(candidate) and math.isfinite(score):
                rescue_best = rescue_by_job.get(jid)
                if rescue_best is None or score > rescue_best[0]:
                    rescue_by_job[jid] = (score, candidate)
            continue
        if (
            is_swap
            and candidate.get("queue_state") == "unstable"
            and not emergency_recovery(candidate)
        ):
            diag["vetoed_unstable"] += 1
            continue
        diag["eligible"] += 1
        if is_swap:
            eligible_swaps_by_job[jid] = eligible_swaps_by_job.get(jid, 0) + 1
        eligible_candidates.append(candidate)

    for jid, (_score, candidate) in rescue_by_job.items():
        if eligible_swaps_by_job.get(jid, 0) > 0:
            continue
        candidate["rescue_floor"] = True
        rescue_diagnostic = selection_diagnostics.get(jid)
        if rescue_diagnostic is not None:
            rescue_diagnostic["rescued"] = True
        eligible_candidates.append(candidate)

    def footprint(candidate: dict[str, Any]) -> int:
        return sum(_ladder_capacity_cost(candidate.get("ladder") or [], specs).values())

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_cost = _ladder_capacity_cost(left.get("ladder") or [], specs)
        right_cost = _ladder_capacity_cost(right.get("ladder") or [], specs)
        keys = set(left_cost) | set(right_cost)
        no_more_resource = all(left_cost.get(key, 0) <= right_cost.get(key, 0) for key in keys)
        no_less_capacity = float(left.get("achieved_tps") or 0.0) >= float(
            right.get("achieved_tps") or 0.0
        )
        no_worse_score = selection_score(left) >= selection_score(right)
        queue_quality = {"stable": 2, "unmodeled": 1, "not_applicable": 1, "unstable": 0}
        no_worse_queue = queue_quality.get(str(left.get("queue_state")), 1) >= queue_quality.get(
            str(right.get("queue_state")), 1
        )
        strictly_better = (
            any(left_cost.get(key, 0) < right_cost.get(key, 0) for key in keys)
            or float(left.get("achieved_tps") or 0.0) > float(right.get("achieved_tps") or 0.0)
            or selection_score(left) > selection_score(right)
        )
        return (
            no_more_resource
            and no_less_capacity
            and no_worse_score
            and no_worse_queue
            and strictly_better
        )

    grouped: dict[tuple[str, str, str | None], list[dict[str, Any]]] = {}
    for candidate in eligible_candidates:
        ladder = candidate.get("ladder") or []
        first = ladder[0] if ladder else {}
        cfg = first.get("config") or {}
        key = (
            str(candidate.get("job_id") or ""),
            _env_key(first.get("env")),
            cfg.get("instance_type"),
        )
        grouped.setdefault(key, []).append(candidate)

    pruned_candidates: list[dict[str, Any]] = []
    for entries in grouped.values():
        if len(entries) <= 4:
            pruned_candidates.extend(entries)
            continue
        specialists: list[dict[str, Any]] = [
            entry
            for entry in entries
            if (entry.get("prediction_assessment") or {}).get("proposal_source") == "specialist"
        ]
        service = [
            entry
            for entry in entries
            if not _is_exploratory_assessment(entry.get("prediction_assessment"))
        ]
        explorers = [
            entry
            for entry in entries
            if _is_exploratory_assessment(entry.get("prediction_assessment"))
        ]
        retained = [
            entry
            for entry in service
            if not any(other is not entry and dominates(other, entry) for other in service)
        ]
        retained.extend(specialists)
        if explorers:
            retained.append(min(explorers, key=footprint))
        seen_ids: set[int] = set()
        for entry in retained:
            if id(entry) not in seen_ids:
                seen_ids.add(id(entry))
                pruned_candidates.append(entry)
    candidates = pruned_candidates
    # Capacity is two-dimensional: env GPU totals AND per-pool whole-instance
    # limits. The pool dimension is what the old env-GPU-only check missed.
    capacity: dict[tuple, int] = {}
    for env, info in resources.items():
        env_key = _env_key(env)
        capacity[("gpu", env_key)] = max(
            0,
            int(info.get("free", 0))
            - reserve_map.get(env_key, 0)
            - pending_reserved.get(("gpu", env_key), 0),
        )
    for env_key, pools in specs.items():
        for instance_type, spec in pools.items():
            capacity[("pool", env_key, str(instance_type))] = max(
                0,
                int(spec.get("free_instances", 0) or 0)
                - pending_reserved.get(("pool", env_key, str(instance_type)), 0),
            )
    priority_by_job = {
        p.get("job_id"): float(p.get("priority_score", 1.0) or 1.0) for p in get_priority()
    }
    pending_job_order = [str(job.get("job_id", job.get("id"))) for job in get_pending_jobs()]
    pending_job_ids = set(pending_job_order)
    slow_loop = getattr(_CTX, "slow_loop", None)
    swap_budget = (
        int(slow_loop.get_sss_swap_budget_t())
        if slow_loop is not None and hasattr(slow_loop, "get_sss_swap_budget_t")
        else 0
    )

    def penalty(jid: str) -> float:
        return UNSERVED_PENALTY * max(1.0, priority_by_job.get(jid, 1.0))

    def finite_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def candidate_served_fraction(cand: dict[str, Any]) -> float | None:
        assessment = cand.get("prediction_assessment") or {}
        exploratory = (
            assessment.get("basis") in _POINT_PREDICTION_BASES
            and assessment.get("kind") == "exploratory"
            and assessment.get("status") in _SOFT_PREDICTION_FAILURES
        )
        if exploratory:
            if cand.get("admitted_tps") is not None or cand.get("admission_mode") is not None:
                return None
            return 0.0
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
            return None
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
        cand.pop("work_conserving_floor", None)
        candidate_job_id = str(cand.get("job_id") or "")
        if not candidate_job_id:
            continue
        cost = _ladder_capacity_cost(cand.get("ladder") or [], specs)
        if not cost:
            continue  # no real GPU footprint -> not a placeable frame
        served_fraction = candidate_served_fraction(cand)
        if served_fraction is None:
            continue
        served_credit = (
            penalty(candidate_job_id) * served_fraction
            if candidate_job_id in pending_job_ids
            else 0.0
        )
        gain = float(cand.get("sigma", 0.0)) + served_credit
        assessment = cand.get("prediction_assessment") or {}
        exploratory = (
            assessment.get("basis") in _POINT_PREDICTION_BASES
            and assessment.get("kind") == "exploratory"
            and assessment.get("status") in _SOFT_PREDICTION_FAILURES
        )
        achieved_tps = finite_number(cand.get("achieved_tps"))
        point_capacity_candidate = (
            achieved_tps is not None
            and achieved_tps > 0
            and (
                assessment.get("basis") in _POINT_PREDICTION_BASES
                or cand.get("admission_mode") == "advisory"
                or (
                    PARTIAL_ONLINE_ADMISSION_MODE == "advisory"
                    and any(
                        cand.get(field) is not None
                        for field in ("target_p99_ttft_ms", "target_p99_tpot_ms")
                    )
                )
            )
        )
        queue_unstable = (
            cand.get("queue_state") == "unstable" or assessment.get("queue_state") == "unstable"
        )
        is_swap = str(cand.get("type") or "").lower() == "swap"
        swap_gain_over_keep = finite_number(cand.get("swap_gain_over_keep"))
        is_emergency_recovery = emergency_recovery(cand)
        if is_swap and (
            (queue_unstable and not is_emergency_recovery)
            or swap_gain_over_keep is None
            or swap_gain_over_keep <= 0
        ):
            if not (cand.get("rescue_floor") and swap_gain_over_keep is not None):
                continue
            # A critical job's rescue: admitted at the floor so it can never
            # outrank a real positive-gain candidate, but the job always gets
            # to TRY one alternative instead of staying pinned to a GPU that
            # is not working.
            gain = _WORK_CONSERVING_GAIN_FLOOR
            cand["work_conserving_floor"] = True
            cand["service_class"] = "partial"
            if isinstance(assessment, dict):
                assessment["selection_mode"] = "emergency_recovery"
        elif is_swap:
            assert swap_gain_over_keep is not None
            gain = swap_gain_over_keep
        if queue_unstable:
            # An emergency rescue keeps its real swap_gain_over_keep: floor_tier
            # already ranks it below every stable candidate, and flattening the
            # gain here made every rescue tie with every speculative idle fill,
            # so the solver could not tell a good rescue from a bad one.
            if not is_emergency_recovery:
                gain = _WORK_CONSERVING_GAIN_FLOOR
            cand["work_conserving_floor"] = True
            cand["service_class"] = "partial" if is_emergency_recovery else "idle_capacity_fallback"
            if isinstance(assessment, dict):
                assessment["selection_mode"] = (
                    "emergency_recovery" if is_emergency_recovery else "work_conserving"
                )
        elif gain <= 0:
            if not exploratory and not point_capacity_candidate:
                continue
            gain = _WORK_CONSERVING_GAIN_FLOOR
            cand["work_conserving_floor"] = True
            cand["service_class"] = "idle_capacity_fallback"
            if isinstance(assessment, dict):
                assessment["selection_mode"] = "work_conserving"
        if not exploratory:
            cand["served_fraction"] = served_fraction
        cand["served_credit"] = served_credit
        cand["solver_gain"] = gain
        floor_tier = (
            2
            if cand.get("work_conserving_floor") and (exploratory or queue_unstable)
            else 1
            if cand.get("work_conserving_floor")
            else 0
        )
        cand["solver_tier"] = floor_tier
        by_job.setdefault(candidate_job_id, []).append(
            {
                "cand": cand,
                "cost": cost,
                "gain": gain,
                "normal_gain": 0.0 if cand.get("work_conserving_floor") else gain,
                "stable_floor_value": (penalty(candidate_job_id) if floor_tier == 1 else 0.0),
                "exploratory_floor_value": (penalty(candidate_job_id) if floor_tier == 2 else 0.0),
                "swap_count": 1 if is_swap else 0,
            }
        )
    jobs = [jid for jid in by_job if by_job[jid]]

    best_solution: dict[str, Any] = {
        "normal_objective": 0.0,
        "stable_floor_objective": 0.0,
        "exploratory_floor_objective": 0.0,
        "chosen": [],
    }

    space = 1
    for jid in jobs:
        space *= 1 + len(by_job[jid])

    solver_mode = "exact" if space <= 200_000 else "greedy"
    if solver_mode == "exact":
        # Exact branch-and-bound: every node is a capacity-feasible assignment
        # (deferring the remaining jobs), so its accumulated gain is a valid
        # objective; keep the best. Place-branches that overflow a pool are pruned.
        def dfs(
            i: int,
            used: dict[str, int],
            normal_gain: float,
            stable_floor_value: float,
            exploratory_floor_value: float,
            swaps_used: int,
            chosen: list[dict[str, Any]],
        ) -> None:
            if (normal_gain, stable_floor_value, exploratory_floor_value) > (
                best_solution["normal_objective"],
                best_solution["stable_floor_objective"],
                best_solution["exploratory_floor_objective"],
            ):
                best_solution["normal_objective"] = normal_gain
                best_solution["stable_floor_objective"] = stable_floor_value
                best_solution["exploratory_floor_objective"] = exploratory_floor_value
                best_solution["chosen"] = list(chosen)
            if i >= len(jobs):
                return
            dfs(
                i + 1,
                used,
                normal_gain,
                stable_floor_value,
                exploratory_floor_value,
                swaps_used,
                chosen,
            )  # defer job i
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
                if swaps_used + entry["swap_count"] > swap_budget:
                    continue
                chosen.append(entry["cand"])
                dfs(
                    i + 1,
                    new_used,
                    normal_gain + entry["normal_gain"],
                    stable_floor_value + entry["stable_floor_value"],
                    exploratory_floor_value + entry["exploratory_floor_value"],
                    swaps_used + entry["swap_count"],
                    chosen,
                )
                chosen.pop()

        dfs(0, {}, 0.0, 0.0, 0.0, 0, [])
    else:
        # Greedy fallback for a large choice space: best-gain frame per job in
        # priority order, taking each only if it still fits. Bounded, never over
        # capacity, not guaranteed optimal.
        log.warning("jointly_select_placements: %d combos, using greedy fallback", space)
        used: dict[str, int] = {}
        chosen: list[dict[str, Any]] = []
        total = 0.0
        swaps_used = 0
        placed_greedy: set[str] = set()
        ordered_jobs = sorted(jobs, key=lambda j: priority_by_job.get(j, 1.0), reverse=True)
        for tier in (0, 1, 2):
            for jid in ordered_jobs:
                if jid in placed_greedy:
                    continue
                entries = [
                    entry
                    for entry in by_job[jid]
                    if int(entry["cand"].get("solver_tier") or 0) == tier
                ]
                for entry in sorted(entries, key=lambda item: item["gain"], reverse=True):
                    trial = dict(used)
                    over = False
                    for key, need in entry["cost"].items():
                        trial[key] = trial.get(key, 0) + need
                        if trial[key] > capacity.get(key, 0):
                            over = True
                            break
                    if not over:
                        if swaps_used + entry["swap_count"] > swap_budget:
                            continue
                        used, total = trial, total + entry["gain"]
                        swaps_used += entry["swap_count"]
                        chosen.append(entry["cand"])
                        placed_greedy.add(jid)
                        break
        best_solution = {
            "normal_objective": sum(
                float(candidate.get("solver_gain", 0.0))
                for candidate in chosen
                if not candidate.get("work_conserving_floor")
            ),
            "stable_floor_objective": sum(
                penalty(str(candidate.get("job_id")))
                for candidate in chosen
                if candidate.get("solver_tier") == 1
            ),
            "exploratory_floor_objective": sum(
                penalty(str(candidate.get("job_id")))
                for candidate in chosen
                if candidate.get("solver_tier") == 2
            ),
            "chosen": chosen,
        }

    chosen = best_solution["chosen"]
    objective = sum(float(candidate.get("solver_gain", 0.0)) for candidate in chosen)
    placed_ids = {c.get("job_id") for c in chosen}
    used_final: dict[str, int] = {}
    for c in chosen:
        for key, need in _ladder_capacity_cost(c.get("ladder") or [], specs).items():
            used_final[_cap_key_str(key)] = used_final.get(_cap_key_str(key), 0) + need
    deferred = [jid for jid in pending_job_order if jid and jid not in placed_ids]
    result = {
        "chosen": chosen,
        "deferred": deferred,
        "objective": objective,
        "used": used_final,
        "capacity": {_cap_key_str(k): v for k, v in capacity.items()},
        "solver_mode": solver_mode,
        "candidate_count_before_pruning": len(original_candidates),
        "candidate_count_after_pruning": len(candidates),
        "combination_count": space,
        "decision_index": decision_index,
        "swap_budget": swap_budget,
        "swap_count": sum(1 for candidate in chosen if candidate.get("type") == "swap"),
        "selection_diagnostics": selection_diagnostics,
    }
    for candidate in chosen:
        chosen_diagnostic = selection_diagnostics.get(str(candidate.get("job_id") or ""))
        if chosen_diagnostic is not None:
            chosen_diagnostic["chosen"] = True
    write_event = getattr(getattr(_CTX, "trace_logger", None), "write_event", None)
    if callable(write_event):
        try:
            write_event(
                "placement_decision",
                {
                    "solver_mode": solver_mode,
                    "decision_index": decision_index,
                    "objective": objective,
                    "candidate_count_before_pruning": len(original_candidates),
                    "candidate_count_after_pruning": len(candidates),
                    "combination_count": space,
                    "surrogate_budget": get_surrogate_budget_status(),
                    # Per job: how many candidates existed and where each was lost
                    # (gain <= 0, unstable, capacity/budget), so a job left on KEEP
                    # is explainable from the trace instead of by inference.
                    "selection_diagnostics": selection_diagnostics,
                    "chosen": [
                        {
                            "job_id": candidate.get("job_id"),
                            "solver_gain": candidate.get("solver_gain"),
                            "queue_state": candidate.get("queue_state"),
                            "prediction_status": (candidate.get("prediction_assessment") or {}).get(
                                "status"
                            ),
                            "ladder": [
                                {
                                    "env": rank.get("env"),
                                    "instance_type": (rank.get("config") or {}).get(
                                        "instance_type"
                                    ),
                                    "tp": (rank.get("config") or {}).get("tp"),
                                    "n_replicas": rank.get("n_replicas"),
                                }
                                for rank in candidate.get("ladder") or []
                            ],
                        }
                        for candidate in chosen
                    ],
                    "deferred": deferred,
                },
                tick=getattr(getattr(_CTX, "cluster_snapshot", None), "tick", None),
            )
        except Exception:
            log.exception("placement decision trace failed")
    return result


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
                    gpu_type=env_gpu_type(getattr(rank, "env", None)),
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
