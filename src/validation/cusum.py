"""
CUSUM detects persistent drift in each V trajectory.
For each mediator V, it tracks residuals: residual = observed_value - surrogate_predicted_value
Unlike a simple threshold test, CUSUM accumulates small repeated errors, so it can
catch steady drift even when each individual residual looks harmless.
Defaults:
    delta = 0.5 * residual_std   # tolerated drift
    h     = 4.0 * residual_std    # firing threshold
A mechanism is MATCHED if no V fires. It is DIVERGED if any V fires, because one
failing mediator is enough to break the mechanism's causal story.
"""

from enum import Enum

import numpy as np


class CusumResult(Enum):
    MATCHED = "matched"
    DIVERGED = "diverged"


class CusumDirection(Enum):
    UP = "up"  # S+ crossed +h  (residuals systematically above predicted)
    DOWN = "down"  # S- crossed -h  (residuals systematically below)
    NONE = "none"  # never fired


class Cusum:
    def cusum_per_mechanism(
        self,
        mechanism,
        v_actual_traj: dict[str, np.ndarray],
        v_hat_traj: dict[str, np.ndarray],
        H: dict[str, float],
        residual_table: dict[str, float],
    ) -> CusumResult:
        """
        Definition: Run two-sided CUSUM on every V in the mechanism's
                    bundle. Return MATCHED iff no V fires, else DIVERGED.
        Usage:      Validator.s2_validate uses this as the trajectory axis
                    of the four-quadrant label.
        Inputs:
            mechanism      : Mechanism with `.bundle_v_variables: List[str]`
            v_actual_traj  : v_name -> observed time series V_k(t)
            v_hat_traj     : v_name -> predicted time series V_hat_k(t) from surrogate
            H              : v_name -> h_V (fire threshold)
            residual_table : v_name -> delta_V (drift tolerance)
        Outputs: CusumResult
        Notes:
            Mismatched/missing V-name keys raise early, caller functions must
            ensure all bundle Vs have both H and residual_table entries.
        """
        for v_name in mechanism.bundle_v_variables:
            delta = residual_table[v_name]
            h = H[v_name]

            observed = np.asarray(v_actual_traj[v_name], dtype=float)
            predicted = np.asarray(v_hat_traj[v_name], dtype=float)
            if observed.shape != predicted.shape:
                raise ValueError(
                    f"shape mismatch for {v_name}: "
                    f"observed {observed.shape} vs predicted {predicted.shape}"
                )

            residuals = observed - predicted
            _, _, (_, fired, _) = self.compute_cusum_statistic(
                residuals, {"delta": delta}, {"h": h}
            )
            if fired:
                return CusumResult.DIVERGED

        return CusumResult.MATCHED

    def cusum_per_v(
        self,
        v_observed: np.ndarray,
        v_predicted: np.ndarray,
        residual_table: dict[str, tuple[float, float]],
    ) -> tuple[CusumDirection, bool, int | None]:
        """
        Definition: Run two-sided CUSUM on ONE V's residual sequence.
                    Returns direction, fire flag, and sample index of
                    first fire (None if never fires).
        Usage:      Per-V call from cusum_per_mechanism; standalone for
                    debugging.
        Inputs:
            v_observed     : np.ndarray of V_k(t)
            v_predicted    : np.ndarray of V_hat_k(t)
            residual_table : v_name -> (delta_V, h_V)  (single entry expected)
        Outputs: (direction: CusumDirection, fired: bool, fire_tick: int|None)
        """
        if len(residual_table) != 1:
            raise ValueError(
                f"cusum_per_v expects exactly one (delta, h) entry, got {len(residual_table)}"
            )
        ((delta, h),) = residual_table.values()

        residuals = np.asarray(v_observed, dtype=float) - np.asarray(v_predicted, dtype=float)
        _, _, fire_info = self.compute_cusum_statistic(residuals, {"delta": delta}, {"h": h})
        return fire_info

    def compute_cusum_statistic(
        self,
        residuals: np.ndarray,
        residual_table: dict[str, float],
        edge_table: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[CusumDirection, bool, int | None]]:
        """
        Definition: Two-sided CUSUM recursion.
                        S+_t = max(0,  S+_{t-1} + r_t - delta)    upward drift
                        S-_t = min(0,  S-_{t-1} + r_t + delta)    downward drift
                    Fires the FIRST t where S+_t > h or S-_t < -h.
                    Continues to integrate S+/S- past the fire so callers
                    can inspect the full trajectories for debugging.
        Usage:      Inner helper for cusum_per_v / cusum_per_mechanism.
        Inputs:
            residuals      : np.ndarray of residual time series
            residual_table : {"delta": delta}   drift tolerance
            edge_table     : {"h": h}       fire threshold (legacy param name)
        Outputs: (S_plus_array, S_minus_array, (direction, fired, fire_tick))
        """
        delta = residual_table["delta"]
        h = edge_table["h"]
        n = len(residuals)

        if n == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                (CusumDirection.NONE, False, None),
            )

        s_plus = np.zeros(n, dtype=float)
        s_minus = np.zeros(n, dtype=float)
        direction = CusumDirection.NONE
        fired = False
        fire_tick: int | None = None

        for t in range(n):
            prev_p = s_plus[t - 1] if t > 0 else 0.0
            prev_m = s_minus[t - 1] if t > 0 else 0.0
            s_plus[t] = max(0.0, prev_p + residuals[t] - delta)
            s_minus[t] = min(0.0, prev_m + residuals[t] + delta)

            # Record first fire only; continue integrating S+/S-.
            if not fired:
                if s_plus[t] > h:
                    direction, fired, fire_tick = CusumDirection.UP, True, t
                elif s_minus[t] < -h:
                    direction, fired, fire_tick = CusumDirection.DOWN, True, t

        return s_plus, s_minus, (direction, fired, fire_tick)

    def update_cusum_from_history(
        self,
        historical_residuals_per_v: dict[str, np.ndarray],
    ) -> dict[str, tuple[float, float]]:
        """
        Definition: Recalibrate (delta_V, h_V) for every V from accumulated
                    residuals. delta = 0.5 * sigma, h = 4 * sigma - Page-CUSUM-ARL defaults.
        Usage:      SlowLoop calls periodically (e.g. every 100 ticks) so
                    CUSUM sensitivity tracks the current residual std as
                    surrogate quality / non-stationarity shifts.
        Inputs:
            historical_residuals_per_v : v_name -> np.ndarray of past residuals
        Outputs: v_name -> (delta_V, h_V)
        """
        return {
            v_name: self.cusum_params_per_v(v_name, residuals)
            for v_name, residuals in historical_residuals_per_v.items()
        }

    def cusum_params_per_v(
        self,
        V: str,
        historic_residuals: np.ndarray,
    ) -> tuple[float, float]:
        """
        Definition: Compute (delta, h) from per-V residual std.
                        delta = 0.5 * sigma, h = 4 * sigma.
                    Falls back to tiny defaults if sigma is degenerate
                    (empty history, constant predictions, perfect fit).
        Usage:      Inner helper for update_cusum_from_history.
        Inputs:
            V                  : v_name (carried for traceability)
            historic_residuals : np.ndarray of past residuals
        Outputs: (delta_V > 0, h_V > 0)
        """
        residuals = np.asarray(historic_residuals, dtype=float)
        if residuals.size == 0:
            return 0.01, 0.04
        sigma = float(np.std(residuals))
        if sigma < 1e-9:
            return 0.01, 0.04
        return 0.5 * sigma, 4.0 * sigma
