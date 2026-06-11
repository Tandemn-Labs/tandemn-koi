"""
CUSUM - per-variable trajectory drift detector. Bundle-level matched/diverged.

Two-sided Cumulative-Sum control chart (Page 1954) on the residual sequence
r_t = observed(t) - predicted(t) for each tracked variable in a deployed
mechanism's bundle. Applied to BOTH:
    V bundle: mediator variables (kv_cache_util, pipeline_bubble_fraction, ...)
    Y bundle: outcome variables (p99_TTFT_ms, throughput_tokens_per_sec, ...)

Both axes use the same primitive - within a tick V_hat and y_hat are constant
scalars (or per-timestep trajectories from the surrogate), and the observed
trajectories are sub-tick samples. The mediator and outcome axes feed the
four-quadrant classifier separately.

Why CUSUM (not just |residual| > threshold):
    A single residual above a threshold can be noise. CUSUM accumulates
    SMALL PERSISTENT drifts, firing only when integrated drift crosses h.
    Catches "predicted hit-rate 0.78, observed 0.71 every step" - a 0.07
    per-sample drift that point-tests miss but CUSUM detects in ~6-12 samples.

ARL (Average Run Length) defaults - Page-CUSUM theory:
    delta = 0.5*sigma_residual    drift tolerance - half a standard deviation
    h = 4.0*sigma_residual        fire threshold - 4 cumulative sigma
    Yields: detect-1sigma-shift ~= 6-12 samples, false-alarm-interval ~= 600+ samples.

Bundle aggregation:
    Bundle is MATCHED iff NO variable in it fires; DIVERGED if ANY fires.
    Conservative OR - a single mediator or outcome failing falsifies the
    mechanism's causal story even if the others match.

State lifetime:
    CUSUM state (S+, S-) is per-call. Resets every tick. Across-tick
    accumulation is handled by ConfidenceService via Beta(alpha, beta) updates,
    NOT by persisting CUSUM state.
"""

from collections.abc import Iterable
from enum import Enum

import numpy as np


class CusumResult(Enum):
    MATCHED = "matched"
    DIVERGED = "diverged"


class CusumDirection(Enum):
    UP = "up"  # S+ crossed +h  (residuals systematically above predicted)
    DOWN = "down"  # S- crossed -h  (residuals systematically below)
    NONE = "none"  # never fired


# (delta, h) per variable name
ParamTable = dict[str, tuple[float, float]]
# either a trajectory (np.ndarray) or a constant scalar (broadcast inside)
PredictedSeries = np.ndarray | float | int


class Cusum:
    def cusum_per_mechanism(
        self,
        mechanism,
        candidate_graph,
        v_obs_traj: dict[str, np.ndarray],
        v_hat_traj: dict[str, PredictedSeries],
        y_obs_traj: dict[str, np.ndarray],
        y_hat_traj: dict[str, PredictedSeries],
        v_params: ParamTable,
        y_params: ParamTable,
    ) -> tuple[CusumResult, CusumResult]:
        """
        Definition: Run two-sided CUSUM on every variable in the mechanism's
                    V bundle AND Y bundle. Each bundle is OR-aggregated:
                    MATCHED iff no variable fires, else DIVERGED.
        Usage:      S2 of the FSM tick per (job, rank). The two returned
                    verdicts are the trajectory and outcome axes feeding
                    classify_quadrant.
        Inputs:
            mechanism   : Mechanism with `.edge_ids`
            candidate_graph : CandidateGraph with edge_table
            v_obs_traj  : v_name -> observed sub-tick samples V_k(t)
            v_hat_traj  : v_name -> predicted V_hat_k (scalar or trajectory)
            y_obs_traj  : y_name -> observed sub-tick samples Y_k(t)
            y_hat_traj  : y_name -> predicted y_hat_k (scalar or trajectory)
            v_params    : v_name -> (delta_v, h_v) from cusum_params_per_v
            y_params    : y_name -> (delta_y, h_y) from cusum_params_per_y
        Outputs:
            (v_verdict, y_verdict) - both CusumResult
        Notes:
            Caller guarantees v_params/y_params cover every name in the
            respective bundle. Missing keys raise KeyError early.
        """
        edges = [candidate_graph.edge_table[edge_id] for edge_id in mechanism.edge_ids]
        v_variables = {
            edge.dst for edge in edges if edge.src_type == "X" and edge.dst_type == "V"
        } | {edge.src for edge in edges if edge.src_type == "V" and edge.dst_type == "Y"}
        y_outcomes = {edge.dst for edge in edges if edge.src_type == "V" and edge.dst_type == "Y"}

        v_verdict = self.cusum_per_bundle(v_variables, v_obs_traj, v_hat_traj, v_params)
        y_verdict = self.cusum_per_bundle(y_outcomes, y_obs_traj, y_hat_traj, y_params)
        return v_verdict, y_verdict

    def cusum_per_bundle(
        self,
        variables: Iterable[str],
        obs_traj: dict[str, np.ndarray],
        hat_traj: dict[str, PredictedSeries],
        params: ParamTable,
    ) -> CusumResult:
        """
        Definition: OR-aggregate CUSUM verdicts across a set of variables.
                    Variable-agnostic - works for the V bundle or Y bundle.
        Usage:      Called twice from cusum_per_mechanism. Also reusable
                    by debugging tools.
        Inputs:
            variables : iterable of variable names (V or Y)
            obs_traj  : name -> observed np.ndarray
            hat_traj  : name -> predicted scalar or np.ndarray
            params    : name -> (delta, h)
        Outputs:
            CusumResult.MATCHED iff no variable fires, else DIVERGED.
        """
        for name in variables:
            delta, h = params[name]
            _, fired, _ = self.cusum_per_v(
                observed=obs_traj[name],
                predicted=hat_traj[name],
                delta=delta,
                h=h,
            )
            if fired:
                return CusumResult.DIVERGED
        return CusumResult.MATCHED

    def cusum_per_v(
        self,
        observed: np.ndarray,
        predicted: PredictedSeries,
        delta: float,
        h: float,
    ) -> tuple[CusumDirection, bool, int | None]:
        """
        Definition: Run two-sided CUSUM on ONE variable's residual sequence.
                    Returns direction, fire flag, and sample index of first
                    fire (None if never fires). Accepts a scalar predicted
                    value - broadcasts to observed shape - since V_hat and y_hat
                    are typically constant within a tick (one surrogate call
                    at deploy time).
        Usage:      Inner per-variable call from cusum_per_bundle; also
                    exposed to the agent for standalone drift inspection.
        Inputs:
            observed  : np.ndarray of sub-tick samples
            predicted : scalar (broadcast) or np.ndarray matching observed.shape
            delta     : drift tolerance delta
            h         : fire threshold h
        Outputs:
            (direction: CusumDirection, fired: bool, fire_tick: int|None)
        """
        obs = np.asarray(observed, dtype=float)

        if isinstance(predicted, (float, int)):
            pred = np.full_like(obs, float(predicted))
        else:
            pred = np.asarray(predicted, dtype=float)
            if pred.shape != obs.shape:
                raise ValueError(f"shape mismatch: observed {obs.shape} vs predicted {pred.shape}")

        residuals = obs - pred
        _, _, fire_info = self.compute_cusum_statistic(residuals, delta, h)
        return fire_info

    def compute_cusum_statistic(
        self,
        residuals: np.ndarray,
        delta: float,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[CusumDirection, bool, int | None]]:
        """
        Definition: Two-sided CUSUM recursion.
                        S+_t = max(0,  S+_{t-1} + r_t - delta)    upward drift
                        S-_t = min(0,  S-_{t-1} + r_t + delta)    downward drift
                    Fires the FIRST t where S+_t > h or S-_t < -h. Continues
                    to integrate past the fire so callers can inspect the
                    full trajectories for debugging.
        Usage:      Inner numerical primitive for cusum_per_v.
        Inputs:
            residuals : np.ndarray of residual time series
            delta     : drift tolerance delta
            h         : fire threshold h
        Outputs:
            (S_plus_array, S_minus_array, (direction, fired, fire_tick))
        """
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

            if not fired:
                if s_plus[t] > h:
                    direction, fired, fire_tick = CusumDirection.UP, True, t
                elif s_minus[t] < -h:
                    direction, fired, fire_tick = CusumDirection.DOWN, True, t

        return s_plus, s_minus, (direction, fired, fire_tick)

    def update_cusum_from_history(
        self,
        historical_residuals: dict[str, np.ndarray],
    ) -> ParamTable:
        """
        Definition: Recalibrate (delta, h) for every tracked variable from
                    accumulated residuals. delta = 0.5*sigma, h = 4*sigma - Page-CUSUM
                    ARL defaults. Variable-agnostic: works for V and Y.
        Usage:      Called periodically (e.g. every 100 ticks) so CUSUM
                    sensitivity tracks the current residual std as
                    surrogate quality / non-stationarity shifts. Caller
                    typically passes V residuals and Y residuals separately
                    and stores the two ParamTables.
        Inputs:
            historical_residuals : name -> np.ndarray of past residuals
        Outputs:
            name -> (delta, h)
        """
        return {
            name: self.cusum_params_per_v(name, residuals)
            for name, residuals in historical_residuals.items()
        }

    def cusum_params_per_v(
        self,
        name: str,
        historic_residuals: np.ndarray,
    ) -> tuple[float, float]:
        """
        Definition: Compute (delta, h) from per-variable residual std.
                        delta = 0.5*sigma, h = 4*sigma.
                    Falls back to tiny defaults if sigma is degenerate (empty
                    history, constant predictions, perfect fit). Works
                    identically for V and Y - calibration is purely
                    statistical, not type-aware.
        Usage:      Inner helper for update_cusum_from_history.
        Inputs:
            name               : variable name (carried for traceability)
            historic_residuals : np.ndarray of past residuals
        Outputs:
            (delta > 0, h > 0)
        """
        residuals = np.asarray(historic_residuals, dtype=float)
        if residuals.size == 0:
            return 0.01, 0.04
        sigma = float(np.std(residuals))
        if sigma < 1e-9:
            return 0.01, 0.04
        return 0.5 * sigma, 4.0 * sigma


# added smoke test below
# if __name__ == "__main__":
#     from src.core.candidate_graph import CandidateGraph
#     from src.core.models import Edge, Mechanism, Node

#     edge_xv = Edge(
#         edge_id="batch_size->kv_cache_pressure",
#         src="batch_size",
#         dst="kv_cache_pressure",
#         src_type="X",
#         dst_type="V",
#     )
#     edge_vy = Edge(
#         edge_id="kv_cache_pressure->ttft_ms",
#         src="kv_cache_pressure",
#         dst="ttft_ms",
#         src_type="V",
#         dst_type="Y",
#     )
#     mechanism = Mechanism(
#         mechanism_id="M_demo",
#         edge_ids=[edge_xv.edge_id, edge_vy.edge_id],
#         scope={"x": ["batch_size"], "v": ["kv_cache_pressure"]},
#         narrative="KV pressure mediates batch size and TTFT.",
#     )
#     graph = CandidateGraph(
#         node_table={
#             "batch_size": Node("batch_size", "X"),
#             "kv_cache_pressure": Node("kv_cache_pressure", "V"),
#             "ttft_ms": Node("ttft_ms", "Y"),
#         },
#         edge_table={
#             edge_xv.edge_id: edge_xv,
#             edge_vy.edge_id: edge_vy,
#         },
#     )

#     cusum = Cusum()
#     v_verdict, y_verdict = cusum.cusum_per_mechanism(
#         mechanism=mechanism,
#         candidate_graph=graph,
#         v_obs_traj={"kv_cache_pressure": np.array([0.21, 0.20, 0.22])},
#         v_hat_traj={"kv_cache_pressure": 0.20},
#         y_obs_traj={"ttft_ms": np.array([110.0, 111.0, 112.0])},
#         y_hat_traj={"ttft_ms": 100.0},
#         v_params={"kv_cache_pressure": (0.05, 0.20)},
#         y_params={"ttft_ms": (1.0, 5.0)},
#     )

#     direction, fired, fire_tick = cusum.cusum_per_v(
#         observed=np.array([110.0, 111.0, 112.0]),
#         predicted=100.0,
#         delta=1.0,
#         h=5.0,
#     )

#     print("v_verdict:", v_verdict)
#     print("y_verdict:", y_verdict)
#     print("single_y_direction:", direction)
#     print("single_y_fired:", fired)
#     print("single_y_fire_tick:", fire_tick)

#     assert v_verdict == CusumResult.MATCHED
#     assert y_verdict == CusumResult.DIVERGED
#     assert direction == CusumDirection.UP
#     assert fired is True
#     assert fire_tick == 0
#     print("All CUSUM smoke tests passed.")
