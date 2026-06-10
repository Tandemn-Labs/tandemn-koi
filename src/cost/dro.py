"""
DRO builds a robust uncertainty band around the surrogate prediction.
The surrogate gives a point prediction y_hat, but its residual distribution is
empirical and can drift over time. DRO protects against this by considering the
worst-case residual distribution inside a Wasserstein-epsilon ambiguity ball.
The radius epsilon_DRO adapts from observed coverage:
    coverage too low  -> grow epsilon
    coverage too high -> shrink epsilon
Residuals are logged as:
    residual = observed_y - predicted_y
DRO is used to estimate downside risk, especially SLO violation probability and
transition failure risk. These risks enter sigma as penalties and make candidate
selection more conservative when uncertainty or drift is high.
"""

from collections import deque
from collections.abc import Iterable

import numpy as np

# Defaults - overridable per cluster at boot.
DEFAULT_EPSILON_INIT: float = 0.15
DEFAULT_EPSILON_MIN: float = 0.02
DEFAULT_EPSILON_MAX: float = 0.50
DEFAULT_TARGET_COVERAGE: float = 0.90
DEFAULT_ETA: float = 0.05
DEFAULT_WINDOW: int = 50
DEFAULT_DEAD_BAND: float = 0.02
DEFAULT_INNER_QUANTILE: float = 0.95
DEFAULT_RESIDUAL_CAPACITY: int = 10_000


class DRO:
    def __init__(
        self,
        epsilon_init: float = DEFAULT_EPSILON_INIT,
        target_coverage: float = DEFAULT_TARGET_COVERAGE,
        eta: float = DEFAULT_ETA,
        eps_min: float = DEFAULT_EPSILON_MIN,
        eps_max: float = DEFAULT_EPSILON_MAX,
        window: int = DEFAULT_WINDOW,
        dead_band: float = DEFAULT_DEAD_BAND,
        residual_capacity: int = DEFAULT_RESIDUAL_CAPACITY,
    ):
        """
        Inputs:
            epsilon_init       : starting epsilon_DRO
            target_coverage    : desired fraction of realized y inside the
                                 predicted DRO band (default 0.90)
            eta                : multiplicative step size for epsilon updates
                                 (default +/-5% per update)
            eps_min / eps_max  : clamp range for epsilon_DRO
            window             : rolling window for observed-coverage tracking
            dead_band          : +/-dead_band around target where epsilon is unchanged
            residual_capacity  : ring-buffer capacity per objective
        """
        self.epsilon = float(epsilon_init)
        self.target = float(target_coverage)
        self.eta = float(eta)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.dead_band = float(dead_band)
        # Coverage flags: True iff realized y fell inside predicted band.
        self._coverage_history: deque[bool] = deque(maxlen=window)
        # Per-objective residual ring buffers, the empirical P_hat_t.
        self._residual_capacity = residual_capacity
        self._residuals_per_dim: dict[str, deque[float]] = {}

    # ----------------- DRO BAND -----------------

    def compute_dro_band(
        self,
        pred_y: dict[str, float],
        residual_history: dict[str, np.ndarray] | None = None,
        epsilon_dro: float | None = None,
        inner_quantile: float = DEFAULT_INNER_QUANTILE,
    ) -> dict[str, dict[str, float]]:
        """
        Definition: Wasserstein-epsilon envelope around the point prediction y_hat
                    per objective.
                        upper_j = y_hat_j + Q_{q}(residuals_j) + epsilon * sigma_hat_j * 2
                        lower_j = y_hat_j + Q_{1-q}(residuals_j) - epsilon * sigma_hat_j * 2
                    Q_q = empirical q-quantile of residuals; sigma_hat = empirical
                    std. The epsilon*sigma*2 shift is a practical Wasserstein-1
                    approximation tight for near-symmetric distributions.
        Usage:      Surrogate-stack post-process: after compose_prediction
                    returns y_hat, this attaches per-objective uncertainty
                    bands. Consumed by Validator.check_C3 (SLO chance) and
                    SwitchCost.c_risk.
        Inputs:
            pred_y           : objective -> y_hat_j  (point prediction from surrogate)
            residual_history : optional override (objective -> np.ndarray).
                               If None, uses the internal accumulated
                               residuals from append_residual_history.
            epsilon_dro      : optional override; default self.epsilon
            inner_quantile   : empirical quantile before Wasserstein shift
                               (default 0.95)
        Outputs:
            Dict[obj -> {"point": y_hat_j, "median": y_hat_j + median(r), "upper": ..., "lower": ...}]
        Notes:
            Falls back to y_hat +/- epsilon for objectives with no residual history yet
            (boot / cold-start). Conservative but well-defined.
        """
        eps = self.epsilon if epsilon_dro is None else float(epsilon_dro)
        band: dict[str, dict[str, float]] = {}

        for obj, y_hat in pred_y.items():
            if y_hat is None:
                continue
            residuals = self._get_residuals_for_dim(obj, residual_history)
            if residuals.size == 0:
                band[obj] = {
                    "point": float(y_hat),
                    "median": float(y_hat),
                    "upper": float(y_hat) + eps,
                    "lower": float(y_hat) - eps,
                }
                continue

            sigma = float(np.std(residuals)) + 1e-9
            q_hi = float(np.quantile(residuals, inner_quantile))
            q_lo = float(np.quantile(residuals, 1.0 - inner_quantile))
            median = float(np.median(residuals))
            shift = eps * sigma * 2.0

            band[obj] = {
                "point": float(y_hat),
                "median": float(y_hat) + median,
                "upper": float(y_hat) + q_hi + shift,
                "lower": float(y_hat) + q_lo - shift,
            }
        return band

    # COVERAGE

    def compute_observed_coverage(
        self,
        recent_predictions: Iterable[dict[str, float]],
        recent_outcomes: Iterable[dict[str, float]],
        recent_dro_bands: Iterable[dict[str, dict[str, float]]] | None = None,
    ) -> float:
        """
        Definition: Fraction of recent realized outcomes that fell inside
                    the DRO band predicted at deploy time.
                    "Inside" = every objective y_j in [lower_j, upper_j].
        Usage:      SlowLoop calls this each tick to decide whether to
                    grow / shrink epsilon via update_epsilon_dro.
        Inputs:
            recent_predictions : iterable of y_hat dicts (kept for API symmetry,
                                 not strictly needed if bands are given)
            recent_outcomes    : iterable of realized y dicts
            recent_dro_bands   : iterable of DRO bands predicted at deploy.
                                 REQUIRED for honest coverage; if None we
                                 return self.target (no signal -> no update).
        Outputs:
            float in [0, 1]
        Notes:
            If the band-history length doesn't match outcomes, returns
            self.target (controller treats as no-signal).
        """
        outcomes = list(recent_outcomes)
        if not outcomes:
            return self.target

        bands = list(recent_dro_bands) if recent_dro_bands is not None else None
        if bands is None or len(bands) != len(outcomes):
            return self.target

        inside_count = 0
        for outcome, band in zip(outcomes, bands, strict=False):
            if self._all_objectives_inside(outcome, band):
                inside_count += 1
        return inside_count / len(outcomes)

    @staticmethod
    def _all_objectives_inside(
        outcome: dict[str, float],
        band: dict[str, dict[str, float]],
    ) -> bool:
        for obj, y in outcome.items():
            if y is None:
                continue
            b = band.get(obj)
            if b is None:
                continue
            if y < b["lower"] or y > b["upper"]:
                return False
        return True

    # CONTROLLER

    def update_epsilon_dro(
        self,
        current_epsilon: float,
        observed_coverage: float,
        target: float = DEFAULT_TARGET_COVERAGE,
    ) -> float:
        """
        Definition: Empirical-coverage controller for epsilon_DRO.
                        obs_cov < target - dead_band  ->  epsilon *= (1 + eta)
                        obs_cov > target + dead_band  ->  epsilon *= (1 - eta)
                        else                           ->  epsilon unchanged
                    Clamps to [eps_min, eps_max]. Persists self.epsilon.
        Usage:      SlowLoop.compute_radius_dro each tick.
        Inputs:
            current_epsilon   : previous epsilon_DRO
            observed_coverage : fraction of recent outcomes inside the band
            target            : desired coverage (default 0.90)
        Outputs:
            new epsilon_DRO (also written into self.epsilon)
        Notes:
            Dead band prevents controller oscillation near target.
        """
        eps = float(current_epsilon)
        if observed_coverage < target - self.dead_band:
            eps = min(eps * (1.0 + self.eta), self.eps_max)
        elif observed_coverage > target + self.dead_band:
            eps = max(eps * (1.0 - self.eta), self.eps_min)
        self.epsilon = eps
        return eps

    # RESIDUAL HISTORY

    def get_residual_history(
        self,
        dim: str,
        window: int | None = None,
    ) -> np.ndarray:
        """
        Definition: Return recent residuals for an objective. Last `window`
                    samples if specified; otherwise everything in the ring.
        Usage:      compute_dro_band and dro_chance_constraint internals;
                    also exposed for diagnostics.
        Inputs:
            dim    : objective name (e.g., "p99_ttft_ms")
            window : optional max sample count
        Outputs:
            np.ndarray (oldest first)
        """
        buf = self._residuals_per_dim.get(dim)
        if buf is None:
            return np.array([], dtype=float)
        if window is None:
            return np.array(buf, dtype=float)
        return np.array(list(buf)[-window:], dtype=float)

    def append_residual_history(
        self,
        pred_y: dict[str, float],
        obs_y: dict[str, float],
    ) -> None:
        """
        Definition: Append one (predicted, observed) pair's residuals per
                    objective present in BOTH dicts. Residual:
                        r = obs_y - pred_y
                    Objectives absent from one side are skipped (handles
                    latency objectives missing on batch jobs cleanly).
        Usage:      Validator.s2_validate calls after each rank's telemetry
                    is processed. Feeds DRO band & chance-constraint logic.
        Inputs:
            pred_y : objective -> y_hat_j  (from SurrogatePrediction.compose_prediction)
            obs_y  : objective -> realized y_j (from Telemetry)
        """
        for obj, y_hat in pred_y.items():
            if y_hat is None:
                continue
            y = obs_y.get(obj)
            if y is None:
                continue
            buf = self._residuals_per_dim.setdefault(obj, deque(maxlen=self._residual_capacity))
            buf.append(float(y) - float(y_hat))

    # CHANCE CONSTRAINT

    def dro_chance_constraint(
        self,
        pred_y: dict[str, float],
        slo_thresholds: dict[str, float],
        epsilon_dro: float | None = None,
        lipschitz_per_obj: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Definition: Per-objective Pr_DRO[g_j(y) > 0] using Wasserstein duality:
                        Pr_DRO[g > 0] <= Pr_{P_hat_t}[g > 0] + epsilon * Lip(g) / sigma
                    Plus aggregate "_any_violated" via union bound:
                        1 - prod(1 - p_j)  (treating SLOs as independent)
        Usage:      Validator C3 (SLO chance constraint) and SwitchCost
                    risk computation. Caller passes per-objective SLO
                    thresholds and gets back DRO-bounded violation probs.
        Inputs:
            pred_y            : objective -> y_hat_j (point prediction)
            slo_thresholds    : objective -> threshold (violation iff
                                y_j > thr for minimize objectives)
            epsilon_dro       : optional override; default self.epsilon
            lipschitz_per_obj : objective -> Lipschitz of g_j (default 1.0)
        Outputs:
            Dict[obj -> Pr_DRO[g_j > 0]] plus "_any_violated"
        Notes:
            Returns Pr = 1.0 for objectives without residual history
            (conservative cold-start behavior).
        """
        eps = self.epsilon if epsilon_dro is None else float(epsilon_dro)
        lip = lipschitz_per_obj or {}
        out: dict[str, float] = {}

        for obj, thr in slo_thresholds.items():
            if obj not in pred_y or pred_y[obj] is None:
                continue
            residuals = self.get_residual_history(obj)
            if residuals.size == 0:
                out[obj] = 1.0
                continue

            y_hat = float(pred_y[obj])
            critical = float(thr) - y_hat
            nominal = float(np.mean(residuals > critical))
            sigma = float(np.std(residuals)) + 1e-9
            dro_bonus = (eps * lip.get(obj, 1.0)) / sigma
            out[obj] = float(min(nominal + dro_bonus, 1.0))

        if out:
            out["_any_violated"] = 1.0 - float(np.prod([1.0 - p for p in out.values()]))
        else:
            out["_any_violated"] = 0.0
        return out

    # INTERNALS

    def _get_residuals_for_dim(
        self,
        dim: str,
        residual_history: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        """Resolve residuals from explicit override or internal buffer."""
        if residual_history is not None and dim in residual_history:
            return np.asarray(residual_history[dim], dtype=float)
        return self.get_residual_history(dim)
