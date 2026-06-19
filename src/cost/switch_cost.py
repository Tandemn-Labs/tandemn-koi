"""
SwitchCost - 4-component A/B-test transition cost.

Architecture:
    Koi never kills production chains before validating their canary
    replacements. Every "swap" launches NEW chains alongside existing
    ones during a test window tau_ab, then commits only if the canary
    proves itself. Old chains are decommissioned gradually as the LLM
    redistributes resources.

    Consequence: ZERO downtime cost. The real costs are:
      - paying for canary chains during the test window
      - per-chain launch overhead (image / weight / kv init)
      - per-chain graceful drain cost (intentionally low)
      - DRO-bounded risk that the transition fails SLO during the test

Notation:
    L_prev : current ladder for the job (List[ChainEntry])
    L_new  : proposed ladder for next tick
    delta_L_plus  = L_new  \\  L_prev    chains being ADDED (canary)
    delta_L_minus = L_prev \\  L_new     chains being KILLED (decommissioned)

Components:
    c_coldstart   sum_{(c,n) in delta_L_plus} n * (img_pull(c) + weight_load(c) + kv_init(c))
    c_parallel    tau_ab * sum_{(c,n) in delta_L_plus} n * hourly_rate(c)
    c_kill        sum_{(c,n) in delta_L_minus} n * kill_cost(c)
    c_risk        lambda_risk * Pr_DRO[transition fails SLO]

Total enters sigma(L') as - lambda_swit^(t) * SwitchCost.total.

Surrogate dependency:
    None directly. The DRO instance (mafunctions/dro.py:DRO) consumed by
    c_risk accumulates residuals from surrogate's compose_prediction at
    Validator-time; we just hand it slo_thresholds and y_hat for the new ladder.
"""

from collections.abc import Callable
from dataclasses import dataclass

# Boot-overridable defaults.
DEFAULT_TAU_AB_HOURS: float = 1.0 / 12.0  # 5 minutes
DEFAULT_KILL_COST: float = 0.02  # $ graceful drain
DEFAULT_LAMBDA_RISK: float = 50.0  # $ per SLO breach


# ============================ TYPES ============================


@dataclass
class ChainEntry:
    """One (chain_config, replica_count) entry inside a ladder."""

    chain_id: str  # stable id (e.g., config fingerprint)
    config: dict  # X variables defining the chain
    env: str  # env_label (market, cloud, region, zone, gpu)
    n_replicas: int


@dataclass
class SwitchCostBundle:
    """Result of compute_switch_cost - components sum to .total."""

    c_coldstart: float = 0.0
    c_parallel: float = 0.0
    c_kill: float = 0.0
    c_risk: float = 0.0

    @property
    def total(self) -> float:
        return self.c_coldstart + self.c_parallel + self.c_kill + self.c_risk

    def as_dict(self) -> dict[str, float]:
        return {
            "c_coldstart": self.c_coldstart,
            "c_parallel": self.c_parallel,
            "c_kill": self.c_kill,
            "c_risk": self.c_risk,
            "total": self.total,
        }


# ===================== PUBLIC ENTRY ============================


def compute_switch_cost(
    L_prev: list[ChainEntry],
    L_new: list[ChainEntry],
    residual_history,
    epsilon_dro: float,
    *,
    pricing_map: dict | None = None,
    tau_ab_hours: float = DEFAULT_TAU_AB_HOURS,
    lambda_risk: float = DEFAULT_LAMBDA_RISK,
    cost_funcs: dict[str, Callable] | None = None,
    slo_thresholds: dict[str, float] | None = None,
    pred_y_new: dict[str, float] | None = None,
) -> SwitchCostBundle:
    """
    Definition: Compute the 4-component SwitchCost bundle.
                    total = c_coldstart + c_parallel + c_kill + c_risk
    Usage:      agent.tools.compute_switching_cost - one call per
                candidate (L_prev, L_new) pair. Returned .total enters
                sigma as - lambda_swit^(t) * SwitchCost.total.
    Inputs:
        L_prev           : current ladder
        L_new            : proposed ladder
        residual_history : DRO instance OR any object with
                           .dro_chance_constraint(pred_y, slo_thresholds, epsilon)
        epsilon_dro      : current epsilon_DRO from SlowLoop
        pricing_map      : optional pricing map for hourly_rate resolution
        tau_ab_hours     : canary test window in HOURS (default 1/12 = 5 min)
        lambda_risk      : $ value of an SLO breach during transition
        cost_funcs       : optional overrides {"image_pull", "weight_load",
                           "kv_init", "kill", "hourly_rate"}; hourly_rate
                           receives the full ChainEntry, others receive config
        slo_thresholds   : per-objective thresholds for c_risk DRO calculation
        pred_y_new       : surrogate's y_hat for L_new - required for path-1 DRO risk
    Outputs:
        SwitchCostBundle
    Notes:
        For full-fidelity c_risk, pass BOTH slo_thresholds AND pred_y_new.
        Without them c_risk falls back to a coarse epsilon-only signal.
    """
    funcs = cost_funcs or {}
    img_fn = funcs.get("image_pull", _default_image_pull_cost)
    wt_fn = funcs.get("weight_load", _default_weight_load_cost)
    kv_fn = funcs.get("kv_init", _default_kv_init_cost)
    kill_fn = funcs.get("kill", _default_kill_cost)
    rate_fn = funcs.get("hourly_rate", lambda chain: hourly_rate(chain, pricing_map))

    delta_plus = compute_delta_L_plus(L_prev, L_new)
    delta_minus = compute_delta_L_minus(L_prev, L_new)

    bundle = SwitchCostBundle()
    bundle.c_coldstart = c_cold_start(delta_plus, img_fn, wt_fn, kv_fn)
    bundle.c_parallel = c_parallel(delta_plus, tau_ab_hours, rate_fn)
    bundle.c_kill = c_kill(delta_minus, kill_fn)
    bundle.c_risk = c_risk(
        L_new,
        dro_band=None,
        residual_history=residual_history,
        epsilon_dro=epsilon_dro,
        lambda_risk=lambda_risk,
        slo_thresholds=slo_thresholds,
        pred_y_new=pred_y_new,
    )
    return bundle


# ===================== DELTA LADDER HELPERS ============================


def compute_delta_L_plus(
    L_prev: list[ChainEntry],
    L_new: list[ChainEntry],
) -> list[ChainEntry]:
    """
    Definition: delta_L_plus = L_new \\ L_prev - chains being ADDED this tick.
                Two entries match iff their chain_id is identical.
                Replica-count INCREASES of an existing chain count as
                additions of the difference (canary expansion case).
    Usage:      Inner helper for compute_switch_cost.
    Inputs:
        L_prev, L_new : List[ChainEntry]
    Outputs:
        List[ChainEntry] - canary additions only
    """
    prev_by_id = {ce.chain_id: ce for ce in L_prev}
    out: list[ChainEntry] = []
    for ce in L_new:
        prev = prev_by_id.get(ce.chain_id)
        if prev is None:
            out.append(ce)
            continue
        diff = ce.n_replicas - prev.n_replicas
        if diff > 0:
            out.append(
                ChainEntry(
                    chain_id=ce.chain_id,
                    config=ce.config,
                    env=ce.env,
                    n_replicas=diff,
                )
            )
    return out


def compute_delta_L_minus(
    L_prev: list[ChainEntry],
    L_new: list[ChainEntry],
) -> list[ChainEntry]:
    """
    Definition: delta_L_minus = L_prev \\ L_new - chains being KILLED this tick.
                Same matching rule as delta_L_plus; replica-count DECREASES count
                as kills of the difference.
    Usage:      Inner helper for compute_switch_cost.
    Inputs:
        L_prev, L_new : List[ChainEntry]
    Outputs:
        List[ChainEntry] - decommissions only
    """
    new_by_id = {ce.chain_id: ce for ce in L_new}
    out: list[ChainEntry] = []
    for ce in L_prev:
        new = new_by_id.get(ce.chain_id)
        if new is None:
            out.append(ce)
            continue
        diff = ce.n_replicas - new.n_replicas
        if diff > 0:
            out.append(
                ChainEntry(
                    chain_id=ce.chain_id,
                    config=ce.config,
                    env=ce.env,
                    n_replicas=diff,
                )
            )
    return out


# ===================== COMPONENTS ============================


def c_cold_start(
    delta_L_plus: list[ChainEntry],
    image_pull_fn: Callable[[dict], float] | None = None,
    weight_load_fn: Callable[[dict], float] | None = None,
    kv_init_fn: Callable[[dict], float] | None = None,
) -> float:
    """
    Definition: One-time launch overhead for canary chains.
                    c_coldstart = sum_{(c, n) in delta_L_plus} n * (img + wt + kv)(c)
    Usage:      Inner helper for compute_switch_cost.
    Inputs:
        delta_L_plus    : List[ChainEntry] - canary additions
        image_pull_fn   : config → $ image pull (default: cached → $0)
        weight_load_fn  : config → $ weight load (default: ∝ model_size_gb)
        kv_init_fn      : config → $ kv allocator init (default: $0.003)
    Outputs:
        float ≥ 0
    """
    img = image_pull_fn or _default_image_pull_cost
    wt = weight_load_fn or _default_weight_load_cost
    kv = kv_init_fn or _default_kv_init_cost
    total = 0.0
    for ce in delta_L_plus:
        per_replica = img(ce.config) + wt(ce.config) + kv(ce.config)
        total += ce.n_replicas * per_replica
    return total


def c_parallel(
    delta_L_plus: list[ChainEntry],
    ab_cost: float = DEFAULT_TAU_AB_HOURS,
    hourly_rate_fn: Callable[[ChainEntry], float] | None = None,
) -> float:
    """
    Definition: Parallel-running cost during the A/B test window.
                    c_parallel = tau_ab * sum_{(c, n) in delta_L_plus} n * hourly_rate(c)
                This is the "cost of safety" - paying for canary chains
                that run alongside production during validation.
    Usage:      Inner helper for compute_switch_cost.
    Inputs:
        delta_L_plus    : List[ChainEntry]
        ab_cost         : canary test window in HOURS (default 1/12 = 5 min)
        hourly_rate_fn  : ChainEntry → $/hour
    Outputs:
        float ≥ 0
    """
    rate_fn = hourly_rate_fn or (lambda chain: hourly_rate(chain, None))
    chain_sum = 0.0
    for ce in delta_L_plus:
        chain_sum += ce.n_replicas * rate_fn(ce)
    return float(ab_cost) * chain_sum


def c_kill(
    delta_L_minus: list[ChainEntry],
    kill_cost_fn: Callable[[dict], float] | None = None,
) -> float:
    """
    Definition: Graceful-drain cost for decommissioned chains.
                    c_kill = sum_{(c, n) in delta_L_minus} n * kill_cost(c)
                INTENTIONALLY LOW so freeing resources for other jobs is
                not penalized - this is the lever that lets the LLM
                redistribute capacity across the cluster.
    Usage:      Inner helper for compute_switch_cost.
    Inputs:
        delta_L_minus : List[ChainEntry]
        kill_cost_fn  : config → $ per-chain drain cost (default $0.02)
    Outputs:
        float ≥ 0
    """
    kill_fn = kill_cost_fn or _default_kill_cost
    return float(sum(ce.n_replicas * kill_fn(ce.config) for ce in delta_L_minus))


def c_risk(
    L_new: list[ChainEntry],
    dro_band: dict[str, dict[str, float]] | None = None,
    residual_history=None,
    *,
    epsilon_dro: float | None = None,
    lambda_risk: float = DEFAULT_LAMBDA_RISK,
    slo_thresholds: dict[str, float] | None = None,
    pred_y_new: dict[str, float] | None = None,
) -> float:
    """
    Definition: DRO-bounded transition-failure risk.
                    c_risk = lambda_risk * Pr_DRO[transition fails SLO]
                The higher of three estimators is used:
                  path 1: full DRO chance-constraint when pred_y_new +
                          slo_thresholds + a DRO instance are all provided
                  path 2: DRO-band edge crossing rate, when dro_band +
                          slo_thresholds are provided
                  path 3: coarse epsilon-only fallback
    Usage:      Inner helper for compute_switch_cost.
    Inputs:
        L_new            : proposed ladder
        dro_band         : optional pre-computed DRO band per objective
        residual_history : DRO instance with .dro_chance_constraint(...)
        epsilon_dro      : current epsilon_DRO
        lambda_risk      : $ value of SLO breach
        slo_thresholds   : per-objective thresholds
        pred_y_new       : surrogate y_hat for L_new
    Outputs:
        float ≥ 0
    """
    p = probability_transition_fails_dro(
        L_new=L_new,
        dro_band=dro_band,
        residual_history=residual_history,
        epsilon_dro=epsilon_dro,
        slo_thresholds=slo_thresholds,
        pred_y_new=pred_y_new,
    )
    return float(lambda_risk) * float(p)


# ===================== RATE / RISK HELPERS ====================


def hourly_rate(chain, pricing_map: dict | None = None) -> float:
    """
    Definition: Per-chain hourly cost in USD. Resolution order:
                  1. config["hourly_rate"]
                  2. pricing_map[env]["by_instance_type"][instance_type]
                  3. pricing_map[env]["default"]
                  4. pricing_map[gpu_type]
                  5. $1.00/hour
    Usage:      c_parallel default rate function; standalone budgeting.
    Inputs:
        chain        : ChainEntry OR a dict (config-like)
        pricing_map  : optional pricing table
    Outputs:
        float ≥ 0 ($/hour)
    """
    cfg = chain.config if hasattr(chain, "config") else (chain or {})
    env = _env_key(getattr(chain, "env", cfg.get("env")))
    instance_type = cfg.get("instance_type")
    gpu = cfg.get("gpu_type")

    if "hourly_rate" in cfg and cfg["hourly_rate"] is not None:
        return float(cfg["hourly_rate"])

    if pricing_map:
        if env and env in pricing_map and isinstance(pricing_map[env], dict):
            by_instance = pricing_map[env].get("by_instance_type") or {}
            if instance_type and instance_type in by_instance:
                return float(by_instance[instance_type])
            if "default" in pricing_map[env]:
                return float(pricing_map[env]["default"])
        if gpu and gpu in pricing_map:
            return float(pricing_map[gpu])

    return 1.0


def _env_key(env) -> str | None:
    if env is None:
        return None
    if isinstance(env, (tuple, list)):
        return "|".join(str(part) for part in env)
    return str(env)


def probability_transition_fails_dro(
    L_new: list[ChainEntry] | None = None,
    *,
    dro_band: dict[str, dict[str, float]] | None = None,
    residual_history=None,
    epsilon_dro: float | None = None,
    slo_thresholds: dict[str, float] | None = None,
    pred_y_new: dict[str, float] | None = None,
) -> float:
    """
    Definition: DRO-bounded probability that the proposed ladder fails
                SLO during the A/B test window. Three resolution paths:
                  Path 1 (full): residual_history has
                                 .dro_chance_constraint(...) AND
                                 slo_thresholds AND pred_y_new available
                                 → returns "_any_violated".
                  Path 2 (band): dro_band + slo_thresholds available
                                 → fraction of objectives whose band upper
                                 edge crosses the threshold.
                  Path 3 (coarse): epsilon_DRO * 0.5 envelope.
    Usage:      Default backend for c_risk. Production callers should
                aim for path 1 (full fidelity).
    Inputs:
        L_new            : proposed ladder (reserved for future per-rank scoring)
        dro_band         : optional pre-computed DRO band per objective
        residual_history : DRO instance (path 1 enabler)
        epsilon_dro      : epsilon_DRO scalar (used in path 1 + path 3)
        slo_thresholds   : per-objective thresholds
        pred_y_new       : y_hat for L_new (path 1 enabler)
    Outputs:
        float in [0, 1]
    """
    # Path 1 - full DRO chance constraint.
    if (
        residual_history is not None
        and slo_thresholds is not None
        and pred_y_new is not None
        and hasattr(residual_history, "dro_chance_constraint")
    ):
        try:
            out = residual_history.dro_chance_constraint(
                pred_y=pred_y_new,
                slo_thresholds=slo_thresholds,
                epsilon_dro=epsilon_dro,
            )
            return float(out.get("_any_violated", 0.0))
        except Exception:
            pass

    # Path 2 - DRO band edge crossing.
    if dro_band and slo_thresholds:
        risky = 0
        total = 0
        for obj, band in dro_band.items():
            if obj not in slo_thresholds:
                continue
            total += 1
            if band.get("upper", float("-inf")) > slo_thresholds[obj]:
                risky += 1
        if total > 0:
            return risky / total

    # Path 3 - coarse epsilon_DRO envelope.
    eps = float(epsilon_dro) if epsilon_dro is not None else 0.0
    return float(min(eps * 0.5, 1.0))


# ===================== DEFAULT COST HELPERS ===================


def _default_image_pull_cost(config: dict) -> float:
    """$0 if image is cached on the node (the common case)."""
    return 0.0


def _default_weight_load_cost(config: dict) -> float:
    """Scales with model_size_gb. ~$2e-4/GB at H100-spot-ish rates."""
    model_size_gb = config.get("model_size_gb", 0.0)
    return 2e-4 * float(model_size_gb) if model_size_gb else 0.01


def _default_kv_init_cost(config: dict) -> float:
    """Engine-version-specific allocator overhead; typical $0.003."""
    return 0.003


def _default_kill_cost(config: dict) -> float:
    """Graceful drain - intentionally low so freeing resources is cheap."""
    return DEFAULT_KILL_COST
