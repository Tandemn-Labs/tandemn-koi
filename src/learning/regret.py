"""
Mechanism regret tracks progress using Q1 rate rather than noisy outcome regret.
Outcome regret is high-variance because workload jitter, neighbor noise, and
allocator drift affect J. Mechanism regret is more stable because Q1 is measured
as a frequency over many deployments.
    R_mech = sum_t (Q1_star - Q1_rate_t)
The regret slope drives slow-loop control: higher slope increases EIG exploration
weight beta_t, swap budget B_t, and the target swap rate rho*_swit. Default Q1_star = 1.0.
"""

from collections.abc import Iterable

import numpy as np
from src.config.hyperparameters import DEFAULT_Q1_STAR


class RegretCalculator:
    def compute_q1_rate(
        self,
        evidence_store,
        tick: int,
        window: int = 20,
    ) -> float | None:
        """
        Definition: Rolling Q1 fraction over the last `window` ticks,
                    counting only decided deployments.
        Usage:  agent; SlowLoop signals; regret-slope building block.
        Inputs:
            evidence_store : EvidenceStore with get_rows_in_window(t0, t1)
            tick           : current tick (inclusive upper bound)
            window         : look-back window in ticks (default 20)
        Outputs: float in [0, 1] or None if no decided rows in window.
        """
        start = max(0, tick - window)
        rows = [
            r
            for r in evidence_store.get_rows_in_window(start, tick)
            if r.quadrant is not None and r.quadrant != "undecided"
        ]
        if not rows:
            return None
        q1 = sum(1 for r in rows if r.quadrant == "Q1")
        return q1 / len(rows)

    def compute_q1_rate_per_mechanism(
        self,
        evidence_store,
        mechanism_id: str,
        window: int,
    ) -> float | None:
        """
        Definition: Q1 rate restricted to deployments using `mechanism_id`,
                    over the last `window` ticks.
        Usage:  Diagnostic - flag mechanisms whose c(M) is high but
                whose recent Q1 rate is degrading (early scope drift).
        Inputs:
            evidence_store : EvidenceStore with get_rows_for_mechanism(id) and current_tick()
            mechanism_id   : str
            window         : ticks
        Outputs: float in [0, 1] or None.
        """
        cur = evidence_store.current_tick()
        cutoff = cur - window
        rows = [
            r
            for r in evidence_store.get_rows_for_mechanism(mechanism_id)
            if r.tick > cutoff and r.quadrant is not None and r.quadrant != "undecided"
        ]
        if not rows:
            return None
        return sum(1 for r in rows if r.quadrant == "Q1") / len(rows)

    def compute_q1_rate_per_env(
        self,
        evidence_store,
        env: str,
        window: int,
    ) -> float | None:
        """
        Definition: Q1 rate restricted to one ICP env, over `window` ticks.
        Usage:      Diagnostic - surface envs where mechanisms degrade
                    (engine regression, driver issue, hardware quirk).
        Inputs:
            evidence_store : EvidenceStore with get_rows_for_environment(env)
            env            : env_label (cloud, region, market, gpu)
            window         : ticks
        Outputs: float in [0, 1] or None.
        """
        cur = evidence_store.current_tick()
        cutoff = cur - window
        rows = [
            r
            for r in evidence_store.get_rows_for_environment(env)
            if r.tick > cutoff and r.quadrant is not None and r.quadrant != "undecided"
        ]
        if not rows:
            return None
        return sum(1 for r in rows if r.quadrant == "Q1") / len(rows)

    def compute_inst_regret(
        self,
        tick: int,
        evidence_store,
        window: int = 20,
        q1_star: float = DEFAULT_Q1_STAR,
    ) -> float:
        """
        Definition: Instantaneous mechanism regret at tick t.
                        inst_regret_t = max(0, Q1* - Q1_rate_t)
                    Default Q1* = 1.0.
        Usage:      Building block for cumulative regret and regret slope.
        Inputs:
            tick           : current tick
            evidence_store : EvidenceStore
            window         : Q1-rate window (default 20)
            q1_star        : oracle target (default 1.0)
        Outputs: float in [0, q1_star]
        """
        q1 = self.compute_q1_rate(evidence_store, tick, window)
        if q1 is None:
            return 0.0
        return max(0.0, q1_star - q1)

    def compute_cum_regret(
        self,
        ticks: Iterable[int],
        evidence_store,
        window: int = 20,
        q1_star: float = DEFAULT_Q1_STAR,
    ) -> float:
        """
        Definition: Cumulative mechanism regret over a tick range.
                        R^mech_T = sum_t inst_regret_t
        Usage:      Theoretical-bound tracking; dashboards.
        Inputs:
            ticks          : iterable of ticks to sum across
            evidence_store : EvidenceStore
            window         : Q1-rate window
            q1_star        : oracle target
        Outputs: float >= 0
        """
        return float(
            sum(self.compute_inst_regret(t, evidence_store, window, q1_star) for t in ticks)
        )

    def compute_regret_slope(
        self,
        tick: int,
        evidence_store,
        window: int = 20,
        q1_star: float = DEFAULT_Q1_STAR,
    ) -> float:
        """
        Definition: Operational signal driving beta_t, B_t, rho*_swit.
                        rho_hat_slope^(t) = mean_{s in [t-W+1, t]} inst_regret_s
                    Equivalent to "average gap to Q1* over recent ticks".
        Usage:      SlowLoop.compute_eig_incentive_t, compute_swap_budget_t.
        Inputs:
            tick           : current tick
            evidence_store : EvidenceStore
            window         : ticks to average over (default 20)
            q1_star        : oracle target (default 1.0)
        Outputs: float >= 0   (small ~= converged; large ~= still learning)
        Notes:
            This is an average of recent instantaneous regrets, not a time-derivative. "Slope" reflects "how steeply
            we're still accruing regret per tick".
        """
        gaps = [
            self.compute_inst_regret(t, evidence_store, window, q1_star)
            for t in range(tick - window + 1, tick + 1)
        ]
        if not gaps:
            return 0.0
        return float(np.mean(gaps))

    def compute_outcome_regret(
        self,
        evidence_store,
        ticks: Iterable[int],
    ) -> float:
        """
        Definition: Cumulative outcome regret over `ticks`. Uses the best
                    observed cluster J as the J* proxy.
                        R^out = sum_t (J* - J^cluster_t)
        Usage:      Dashboards / reporting only. NOT used to drive
                    slow-loop knobs (too noisy).
        Inputs:
            evidence_store : EvidenceStore exposing cluster_J_at(tick)
            ticks          : iterable of ticks
        Outputs: float >= 0
        """
        history = [evidence_store.cluster_J_at(t) for t in ticks]
        history = [j for j in history if j is not None]
        if not history:
            return 0.0
        j_star = max(history)
        return float(sum(j_star - j for j in history))
