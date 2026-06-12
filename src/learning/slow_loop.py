"""SlowLoop updates the slow knobs consumed by sigma(L').

Knobs owned here:
    w_t           Tchebycheff objective weights on the simplex
    z_star_t      Pareto reference point per objective
    lambda_swit   Switching-cost penalty
    beta_t        EIG exploration weight
    B_t           Swap budget per tick
    epsilon_dro   Wasserstein DRO radius

Signals that drive them:
    regret_slope        -> beta_t, B_t, target_swap_rate anneal
    observed_swap_rate  -> lambda_swit
    observed_coverage   -> epsilon_dro (via DRO controller)
    r2_gradient         -> w_t
    objective bests     -> z_star_t

Meta timescale (slower cadence than slow_update_all):
    Periodic CUSUM (delta, h) recalibration via recalibrate_cusum_params.
    Called every ~100-500 ticks so CUSUM sensitivity tracks the current
    residual stddev for both V (mediators) and Y (outcomes).

Confidence updates for edges and mechanisms are NOT done here. They happen
earlier in ConfidenceService, before slow_update_all runs. SlowLoop owns
scalar/vector knobs; ConfidenceService owns Beta(alpha, beta) updates.

Per-mechanism Q labels are handled inside RegretCalculator, which uses
EvidenceStore.iter_decided_per_mechanism to flatten across (row, mid) pairs.
SlowLoop just delegates Q1 rate and regret slope to RegretCalculator and
caches the latest values on its SlowState.

Timescales:
    fast = deployment decisions (per-tick agentic plan)
    slow = per-tick knob updates (slow_update_all)
    meta = periodic CUSUM recalibration (recalibrate_cusum_params)
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.config.hyperparameters import (
    B_MAX,
    B_MIN,
    EPSILON_DRO_INIT,
    ETA_BETA,
    ETA_LAMBDA,
    ETA_W,
    RHO_STAR_SLOPE_FINAL,
    RHO_STAR_SLOPE_INIT,
    RHO_STAR_SWIT_FINAL,
    RHO_STAR_SWIT_INIT,
    W_Q1,
    W_REGRET,
)

# Keep this in sync with prediction.tchebycheff.DEFAULT_MAXIMIZE so z_star
# and Tchebycheff scoring use the same objective direction.
_MAXIMIZE_OBJECTIVES = frozenset(
    {
        "throughput_tokens_per_sec",
        "slo_margin",
    }
)


def _is_maximize(objective: str) -> bool:
    """Return True if this outcome objective is maximized."""
    return objective in _MAXIMIZE_OBJECTIVES


@dataclass
class SlowState:
    """In-memory cache of current slow-loop values, refreshed atomically each tick.

    Holds both the scalar/vector knobs that sigma(L') consumes and the
    cached signals (q1_rate, regret_slope, observed_swap_rate,
    observed_coverage) for one-shot lookups by the agent. The CUSUM
    parameter tables are refreshed on a slower cadence by
    recalibrate_cusum_params and are read by the telemetry adapter at
    S2 dispatch time.
    """

    w_t: dict[str, float] = field(default_factory=dict)
    z_star_t: dict[str, float] = field(default_factory=dict)
    lambda_swit: float = 0.05
    beta_t: float = 0.5
    B_t: int = B_MAX
    epsilon_dro: float = EPSILON_DRO_INIT

    regret_slope: float = 0.0
    q1_rate: float | None = None
    observed_swap_rate: float = 0.0
    observed_coverage: float = 0.90

    cusum_params_v: dict[str, tuple[float, float]] = field(default_factory=dict)
    cusum_params_y: dict[str, tuple[float, float]] = field(default_factory=dict)

    tick: int = 0


class SlowLoop:
    """Per-tick knob updater and meta-timescale CUSUM recalibrator.

    Exposes:
        get_sss_* readers     O(1) lookups from the cached SlowState used
                              by Tchebycheff, sigma, validators, and tools.
        compute_* updaters    Per-tick math run by slow_update_all.
        slow_update_all       The per-tick orchestrator (called in S3).
        recalibrate_cusum_params  Meta-timescale refresh of (delta, h) for
                                  V and Y residual sequences.

    Beta(alpha, beta) updates to edge and mechanism confidences are NOT
    in this module. They live in ConfidenceService and are applied earlier
    in S3, before slow_update_all runs.
    """

    def __init__(
        self,
        evidence_store,
        dro,
        regret_calculator,
        objectives: Iterable[str],
        typical_ranges: dict[str, float],
        cusum=None,
        tracked_v_variables: Iterable[str] | None = None,
        config: dict | None = None,
    ):
        """Initialize the slow loop.

        Args:
            evidence_store: EvidenceStore. Source of Q-rate triples,
                residual history, and z_star kNN lookups.
            dro: DRO instance. Handles epsilon_dro updates and provides
                its residual ring buffer for Y recalibration.
            regret_calculator: RegretCalculator. Computes Q1 rate and
                regret slope from (row, mechanism) pairs.
            objectives: List of outcome objective names (the 5 Y's).
            typical_ranges: Per-objective scale used for z_star slack
                and for surrogate-residual normalization.
            cusum: Cusum instance. Required for recalibrate_cusum_params;
                optional if you never call that method.
            tracked_v_variables: V names whose CUSUM params should be
                recalibrated periodically. Defaults to empty - callers
                that want V-CUSUM must supply the V universe.
            config: Optional overrides for any of the hyperparameter
                defaults imported from hyperparameters.py.
        """
        self.evidence_store = evidence_store
        self.dro = dro
        self.regret = regret_calculator
        self.cusum = cusum
        self.objectives = list(objectives)
        self.tracked_v_variables = list(tracked_v_variables or [])
        self.typical_ranges = dict(typical_ranges)

        cfg = config or {}
        self.eta_w = float(cfg.get("eta_w", ETA_W))
        self.eta_lambda = float(cfg.get("eta_lambda", ETA_LAMBDA))
        self.eta_beta = float(cfg.get("eta_beta", ETA_BETA))
        self.target_swap_rate = float(cfg.get("target_swap_rate", RHO_STAR_SWIT_INIT))
        self.target_slope = float(cfg.get("target_slope", RHO_STAR_SLOPE_INIT))
        self.target_swap_rate_final = float(cfg.get("target_swap_rate_final", RHO_STAR_SWIT_FINAL))
        self.target_slope_final = float(cfg.get("target_slope_final", RHO_STAR_SLOPE_FINAL))
        self.B_min = int(cfg.get("B_min", B_MIN))
        self.B_max = int(cfg.get("B_max", B_MAX))

        self.beta_min = float(cfg.get("beta_min", 0.05))
        self.beta_max = float(cfg.get("beta_max", 2.0))
        self.lambda_min = float(cfg.get("lambda_min", 0.0))
        self.lambda_max = float(cfg.get("lambda_max", 1.0))

        # Per-objective z_star slack so candidates matching the empirical
        # best still see a positive gap that keeps exploration alive.
        delta_default = {obj: 0.05 * typical_ranges.get(obj, 1.0) for obj in self.objectives}
        self.delta_per_obj: dict[str, float] = dict(cfg.get("delta_per_obj", delta_default))
        self.anneal_T = int(cfg.get("anneal_T", 500))
        self.recalibration_window = int(cfg.get("recalibration_window", 200))
        self.z_star_top_k = int(cfg.get("z_star_top_k", 200))

        n_obj = max(1, len(self.objectives))
        initial_weights = dict.fromkeys(self.objectives, 1.0 / n_obj)
        initial_z_star = dict.fromkeys(self.objectives, 0.0)

        self.state = SlowState(
            w_t=initial_weights,
            z_star_t=initial_z_star,
            lambda_swit=float(cfg.get("lambda_initial", 0.05)),
            beta_t=float(cfg.get("beta_initial", 0.5)),
            B_t=self.B_max,
            epsilon_dro=float(
                self.dro.epsilon if hasattr(self.dro, "epsilon") else EPSILON_DRO_INIT
            ),
        )

    # ------------------------------------------------------------------
    # Readers (O(1) cached lookups)
    # ------------------------------------------------------------------

    def get_sss_wt(self, job_type: str | None = None) -> dict[str, float]:
        """Return the current Tchebycheff weight vector w_t.

        Args:
            job_type: Reserved for forward-compat with per-job-type w_t.
                Currently ignored.

        Returns:
            Dict mapping each objective to its weight. Sums to 1.
        """
        return dict(self.state.w_t)

    def get_sss_z_star_t(self, job_features: dict | None = None) -> dict[str, float]:
        """Return the current per-objective Pareto reference point.

        Args:
            job_features: If provided, compute a per-job reference point
                from similar evidence rows instead of returning the cached
                global z_star_t.

        Returns:
            Dict mapping each objective to its z_star value.
        """
        if job_features is not None:
            return self.compute_z_star_t(job_features=job_features)
        return dict(self.state.z_star_t)

    def get_sss_lambda_switch(self) -> float:
        """Return the current switching penalty lambda_swit."""
        return float(self.state.lambda_swit)

    def get_sss_eig_incentive_t(self) -> float:
        """Return the current EIG exploration weight beta_t.

        Floored at beta_min so EIG never zeroes out (preserves
        regime-shift sensitivity).
        """
        return float(self.state.beta_t)

    def get_sss_swap_budget_t(self) -> int:
        """Return the current per-tick swap budget B_t.

        Read as the C4 hard constraint in plan validation; honored by
        the agent during ladder assembly.
        """
        return int(self.state.B_t)

    def get_sss_radius_dro(self) -> float:
        """Return the current Wasserstein DRO radius epsilon_dro."""
        return float(self.state.epsilon_dro)

    def get_sss_regret_slope(self, window: int | None = None) -> float:
        """Return the cached regret slope, or recompute on a custom window.

        Args:
            window: If given, recomputes the slope over that window.
                If None, returns the cached value from slow_update_all.

        Returns:
            Non-negative slope. Drives beta_t, B_t, and the swap-rate anneal.
        """
        if window is None:
            return float(self.state.regret_slope)
        return self.compute_regret_slope(window=window)

    def get_sss_q1_rate(self, window: int = W_Q1) -> float | None:
        """Return the cached Q1 rate, or recompute on a custom window.

        Args:
            window: Look-back window in ticks.

        Returns:
            Q1 rate in [0, 1], or None if no decided pairs are available.
        """
        if window == W_Q1 and self.state.q1_rate is not None:
            return self.state.q1_rate
        return self.compute_q1_rate(window=window)

    def get_sss_cusum_params_v(self) -> dict[str, tuple[float, float]]:
        """Return the cached V-CUSUM (delta, h) table.

        Refreshed by recalibrate_cusum_params on the meta timescale and
        consumed by the telemetry adapter when dispatching S2 evaluation.
        """
        return dict(self.state.cusum_params_v)

    def get_sss_cusum_params_y(self) -> dict[str, tuple[float, float]]:
        """Return the cached Y-CUSUM (delta, h) table.

        Refreshed by recalibrate_cusum_params on the meta timescale and
        consumed by S2 when running Y-CUSUM on outcome trajectories.
        """
        return dict(self.state.cusum_params_y)

    # ------------------------------------------------------------------
    # Per-tick updaters (called by slow_update_all)
    # ------------------------------------------------------------------

    def compute_q1_rate(
        self,
        window: int = W_Q1,
        tick: int | None = None,
    ) -> float | None:
        """Compute the rolling cluster Q1 fraction.

        Delegates to RegretCalculator, which uses
        evidence_store.iter_decided_per_mechanism to count Q1 events
        across all decided (row, mechanism) pairs in the window.

        Args:
            window: Look-back window in ticks.
            tick: Current tick. Defaults to self.state.tick.

        Returns:
            Q1 rate in [0, 1] or None when no decided pairs are found.
        """
        current = self.state.tick if tick is None else int(tick)
        return self.regret.compute_q1_rate(self.evidence_store, current, window)

    def compute_regret_slope(
        self,
        window: int = W_REGRET,
        tick: int | None = None,
    ) -> float:
        """Compute the mean recent (1 - Q1_rate) over the window.

        This is the operational signal that drives beta_t and B_t.
        Delegates to RegretCalculator.

        Args:
            window: Averaging window in ticks.
            tick: Current tick. Defaults to self.state.tick.

        Returns:
            Non-negative slope.
        """
        current = self.state.tick if tick is None else int(tick)
        return self.regret.compute_regret_slope(current, self.evidence_store, window)

    def compute_eig_incentive_t(
        self,
        beta_prev: float,
        regret_slope: float,
        target_slope: float | None = None,
        eta_beta: float | None = None,
    ) -> float:
        """Anneal beta_t against the regret slope via mirror descent.

            beta_next = beta_prev * exp(eta_beta * (regret_slope - target_slope))

        Clamped to [beta_min, beta_max]. Steep slope raises beta (more
        exploration); flat slope lowers it (more exploit). Never zero,
        so the system stays sensitive to regime shifts.

        Args:
            beta_prev: Previous beta_t.
            regret_slope: Current regret slope.
            target_slope: Target slope. Defaults to self.target_slope.
            eta_beta: Step size. Defaults to self.eta_beta.

        Returns:
            New beta_t clamped to [beta_min, beta_max].
        """
        target = self.target_slope if target_slope is None else float(target_slope)
        step = self.eta_beta if eta_beta is None else float(eta_beta)
        gradient = float(regret_slope) - target
        new_beta = float(beta_prev) * math.exp(step * gradient)
        return float(max(self.beta_min, min(self.beta_max, new_beta)))

    def compute_swap_budget_t(
        self,
        regret_slope: float,
        target_slope: float | None = None,
        B_min: int | None = None,
        B_max: int | None = None,
    ) -> int:
        """Step-function gate on the regret slope.

            B_t = B_max if regret_slope > target_slope else B_min

        Aggressive re-planning while still learning; lock in when
        converged. B_min stays at least 1 so the system can always
        reconsider at least one job per tick (avoids deadlock).

        Args:
            regret_slope: Current regret slope.
            target_slope: Target slope. Defaults to self.target_slope.
            B_min: Optional lower bound override.
            B_max: Optional upper bound override.

        Returns:
            Integer swap budget in [B_min, B_max].
        """
        target = self.target_slope if target_slope is None else float(target_slope)
        lo = self.B_min if B_min is None else int(B_min)
        hi = self.B_max if B_max is None else int(B_max)
        return hi if float(regret_slope) > target else lo

    def compute_lambda_switch(
        self,
        lambda_prev: float,
        observed_swap_rate: float,
        target_rate: float | None = None,
        eta_lambda: float | None = None,
    ) -> float:
        """Mirror-descent update against the target swap rate.

            lambda_next = lambda_prev * exp(eta * (observed - target))

        If actual churn exceeds target, raise lambda (penalize swaps
        more); if below target, lower it. Clamped to [lambda_min, lambda_max].

        Args:
            lambda_prev: Previous lambda_swit.
            observed_swap_rate: Fraction of active jobs swapped in tick t.
            target_rate: Target swap rate. Defaults to self.target_swap_rate.
            eta_lambda: Step size. Defaults to self.eta_lambda.

        Returns:
            New lambda_swit clamped to [lambda_min, lambda_max].
        """
        target = self.target_swap_rate if target_rate is None else float(target_rate)
        step = self.eta_lambda if eta_lambda is None else float(eta_lambda)
        gradient = float(observed_swap_rate) - target
        new_lambda = float(lambda_prev) * math.exp(step * gradient)
        return float(max(self.lambda_min, min(self.lambda_max, new_lambda)))

    def compute_radius_dro(
        self,
        epsilon_prev: float,
        observed_coverage: float,
        target_coverage: float | None = None,
    ) -> float:
        """Delegate epsilon_dro update to the DRO controller.

        Grows epsilon when observed coverage < target (band was too
        tight); shrinks when over (band was too loose).

        Args:
            epsilon_prev: Previous epsilon_dro.
            observed_coverage: Fraction of recent realized y inside the band.
            target_coverage: Target coverage. If None, the DRO instance's
                own default is used.

        Returns:
            New epsilon_dro inside [dro.eps_min, dro.eps_max].
        """
        kwargs = {
            "current_epsilon": epsilon_prev,
            "observed_coverage": observed_coverage,
        }
        # TODO: Make DRO.update_epsilon_dro own its default target so a custom
        # DRO(target_coverage=...) is honored when target_coverage is omitted.
        if target_coverage is not None:
            kwargs["target"] = target_coverage
        return float(self.dro.update_epsilon_dro(**kwargs))

    def compute_wt(
        self,
        w_prev: dict[str, float],
        r2_gradient: dict[str, float] | None = None,
        eta_w: float | None = None,
    ) -> dict[str, float]:
        """Entropic mirror descent on the R2 Pareto-coverage gradient.

            w_next_j = w_prev_j * exp(eta * grad_R2_j)
            normalize so weights sum to 1.

        The gradient points toward objectives whose Pareto region is
        currently under-mapped. Sweeping w_t over many ticks sweeps the
        entire (possibly non-convex) Pareto front.

        Args:
            w_prev: Previous w_t.
            r2_gradient: Per-objective R2 gradient. If None, returns
                w_prev unchanged - the common case at boot before any
                R2 coverage signal exists.
            eta_w: Step size. Defaults to self.eta_w.

        Returns:
            New w_t on the simplex.
        """
        if r2_gradient is None:
            return dict(w_prev)
        step = self.eta_w if eta_w is None else float(eta_w)
        n_obj = max(1, len(self.objectives))
        scaled = {
            obj: float(w_prev.get(obj, 1.0 / n_obj))
            * math.exp(step * float(r2_gradient.get(obj, 0.0)))
            for obj in self.objectives
        }
        total = sum(scaled.values()) or 1.0
        return {obj: v / total for obj, v in scaled.items()}

    def compute_z_star_t(
        self,
        evidence_store=None,
        delta_per_obj: dict[str, float] | None = None,
        job_features: dict | None = None,
    ) -> dict[str, float]:
        """Compute the per-objective Pareto reference point.

            z_star_j = best y_j across H_t - delta_j   (minimized objectives)
                     = best y_j across H_t + delta_j   (maximized objectives)

        Slack delta_j keeps the gradient alive even at empirical optimum.
        Reads r.y_observed_mean (the denormalized per-objective scalar
        stored on EvidenceRow at build time). When job_features is given
        and the store supports retrieve_similar_rows, the "best" is taken
        over similar rows for a per-job z_star.

        Args:
            evidence_store: Override; defaults to self.evidence_store.
            delta_per_obj: Override; defaults to self.delta_per_obj.
            job_features: Optional similarity filter for per-job z_star.

        Returns:
            Dict mapping each objective to its z_star value. Falls back
            to 0.0 per objective if no observed rows exist yet.
        """
        store = evidence_store or self.evidence_store
        slack = delta_per_obj or self.delta_per_obj

        if job_features is not None and hasattr(store, "retrieve_similar_rows"):
            rows = store.retrieve_similar_rows(job_features, top_k=self.z_star_top_k)
        elif hasattr(store, "get_all_rows"):
            rows = store.get_all_rows(limit=self.z_star_top_k)
        else:
            rows = []

        z_star: dict[str, float] = {}
        for obj in self.objectives:
            values = [
                float(r.y_observed_mean[obj])
                for r in rows
                if getattr(r, "y_observed_mean", None)
                and obj in r.y_observed_mean
                and r.y_observed_mean[obj] is not None
            ]
            if not values:
                z_star[obj] = 0.0
                continue
            offset = float(slack.get(obj, 0.0))
            z_star[obj] = (max(values) + offset) if _is_maximize(obj) else (min(values) - offset)
        return z_star

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def slow_update_all(
        self,
        tick: int,
        observed_swap_rate: float = 0.0,
        observed_coverage: float = 0.90,
        r2_gradient: dict[str, float] | None = None,
        target_overrides: dict[str, float] | None = None,
    ) -> SlowState:
        """Run every compute_* updater in dependency order, then persist.

        Order:
            1. Q1 rate                      (refresh cache)
            2. Regret slope                 (drives 3 and 4)
            3. beta_t   <- regret slope     (EIG exploration weight)
            4. B_t      <- regret slope     (swap budget)
            5. lambda_swit <- observed swap rate
            6. epsilon_dro <- observed coverage
            7. w_t      <- R2 gradient      (Tchebycheff weights)
            8. z_star_t <- memory           (Pareto reference)

        Called in S3 of the FSM tick, AFTER ConfidenceService has
        applied Beta(alpha, beta) updates from this tick's evidence rows.

        Args:
            tick: Current tick id (becomes self.state.tick).
            observed_swap_rate: Fraction of active jobs swapped in
                [tick-1, tick].
            observed_coverage: Fraction of recent outcomes inside the
                DRO band.
            r2_gradient: Optional R2 Pareto-coverage gradient per objective.
            target_overrides: Optional overrides for anneal targets this
                tick, e.g. {"target_swap_rate": 0.05, "target_slope": 0.02}.

        Returns:
            The refreshed SlowState (also persisted into self.state).
        """
        self.state.tick = int(tick)

        if target_overrides:
            self.target_swap_rate = float(
                target_overrides.get("target_swap_rate", self.target_swap_rate)
            )
            self.target_slope = float(target_overrides.get("target_slope", self.target_slope))

        q1 = self.compute_q1_rate(window=W_Q1, tick=tick)
        self.state.q1_rate = q1

        slope = self.compute_regret_slope(window=W_REGRET, tick=tick)
        self.state.regret_slope = slope

        self.state.observed_swap_rate = float(observed_swap_rate)
        self.state.observed_coverage = float(observed_coverage)

        self.state.beta_t = self.compute_eig_incentive_t(self.state.beta_t, slope)
        self.state.B_t = self.compute_swap_budget_t(slope)

        self.state.lambda_swit = self.compute_lambda_switch(
            self.state.lambda_swit, observed_swap_rate
        )

        self.state.epsilon_dro = self.compute_radius_dro(self.state.epsilon_dro, observed_coverage)

        self.state.w_t = self.compute_wt(self.state.w_t, r2_gradient)
        self.state.z_star_t = self.compute_z_star_t()

        return self.state

    # ------------------------------------------------------------------
    # Meta timescale: CUSUM recalibration
    # ------------------------------------------------------------------

    def recalibrate_cusum_params(
        self,
        window: int | None = None,
    ) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
        """Refresh (delta, h) tables for V and Y CUSUM from recent residuals.

        Pulls per-V residual history from evidence_store and per-Y
        residual history from the DRO ring (the authoritative outcome
        residual buffer). Runs cusum.update_cusum_from_history on each
        and writes the result into self.state. The returned tables match
        what get_sss_cusum_params_v / _y will subsequently return.

        Args:
            window: Look-back window in ticks. Defaults to
                self.recalibration_window.

        Returns:
            Tuple of (v_params, y_params), each a dict mapping a
            variable name to its (delta, h) pair. Returns empty dicts
            if no cusum instance was supplied at construction.
        """
        if self.cusum is None:
            return {}, {}

        look_back = self.recalibration_window if window is None else int(window)

        v_history: dict[str, Iterable[float]] = {}
        for v_name in self.tracked_v_variables:
            if hasattr(self.evidence_store, "get_residual_history_per_v"):
                v_history[v_name] = self.evidence_store.get_residual_history_per_v(
                    v_name, look_back
                )

        y_history: dict[str, Iterable[float]] = {}
        for y_name in self.objectives:
            if hasattr(self.dro, "get_residual_history"):
                y_history[y_name] = self.dro.get_residual_history(y_name, look_back)
            elif hasattr(self.evidence_store, "get_residual_history_per_y"):
                y_history[y_name] = self.evidence_store.get_residual_history_per_y(
                    y_name, look_back
                )

        v_params = self.cusum.update_cusum_from_history(v_history) if v_history else {}
        y_params = self.cusum.update_cusum_from_history(y_history) if y_history else {}

        self.state.cusum_params_v = v_params
        self.state.cusum_params_y = y_params
        return v_params, y_params

    # ------------------------------------------------------------------
    # Anneal schedule helper
    # ------------------------------------------------------------------

    def anneal_targets(self, tick: int) -> dict[str, float]:
        """Linear interpolation of anneal targets toward their final values.

            progress = clip(tick / anneal_T, 0, 1)
            target = init * (1 - progress) + final * progress

        Used to gradually reduce exploration pressure as the cluster
        matures. The returned dict can be passed as target_overrides to
        slow_update_all.

        Args:
            tick: Current tick.

        Returns:
            Dict with annealed target_swap_rate and target_slope.
        """
        if self.anneal_T <= 0:
            progress = 1.0
        else:
            progress = max(0.0, min(1.0, float(tick) / float(self.anneal_T)))
        # TODO: Use configured initial/final target attributes here instead
        # of module constants so config overrides affect the anneal schedule.
        return {
            "target_swap_rate": RHO_STAR_SWIT_INIT * (1.0 - progress)
            + RHO_STAR_SWIT_FINAL * progress,
            "target_slope": RHO_STAR_SLOPE_INIT * (1.0 - progress)
            + RHO_STAR_SLOPE_FINAL * progress,
        }
