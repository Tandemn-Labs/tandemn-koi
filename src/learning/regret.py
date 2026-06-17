"""Mechanism regret based on Q1 rate over (row, mechanism) pairs.

Mechanism regret tracks progress via Q1 rate rather than noisy outcome
regret. Outcome regret is high-variance because workload jitter, neighbor
noise, and allocator drift all affect J. Mechanism regret is more stable
because Q1 is measured as a frequency over many (row, mechanism) pairs.

    R_mech = sum_t (Q1_star - Q1_rate_t)

The regret slope drives slow-loop control: a higher slope raises the EIG
exploration weight, the swap budget, and the target swap rate. Default
Q1_star is 1.0.

EvidenceRow carries q_label_per_mechanism: dict[mid, Optional[Quadrant]]
because a single rank is interpretable through multiple applicable
mechanisms. Cluster-level Q1 rate is denominated in (row, mechanism)
pairs, not rows. Every applicable mechanism contributes one count
regardless of which mechanism the agent committed to.

The flattening is done by evidence_store.iter_decided_per_mechanism,
which yields (row, mid, q) triples skipping None Q-labels. A None Q
means the row is undecided for that mechanism because at least one edge
in the mechanism's bundle had ICP=UNDECIDED.
"""

from collections.abc import Iterable

import numpy as np
from src.validation.quadrants import Quadrant

DEFAULT_Q1_STAR: float = 1.0


class RegretCalculator:
    """Q1-rate-based mechanism regret calculator."""

    def compute_q1_rate(
        self,
        evidence_store,
        tick: int,
        window: int = 20,
    ) -> float | None:
        """Compute rolling cluster Q1 fraction over a tick window.

        Counts Q1 events across all decided (row, mechanism) pairs in the
        last `window` ticks. A row with five applicable mechanisms
        contributes five events to the denominator.

        Args:
            evidence_store: EvidenceStore exposing
                iter_decided_per_mechanism(window, tick).
            tick: Current tick (inclusive upper bound).
            window: Look-back window in ticks.

        Returns:
            Q1 rate in [0, 1], or None if no decided pairs are in window.
        """
        triples = list(evidence_store.iter_decided_per_mechanism(window, tick))
        if not triples:
            return None
        q1_count = sum(1 for _, _, q in triples if q == Quadrant.Q1)
        return q1_count / len(triples)

    def compute_q1_rate_per_mechanism(
        self,
        evidence_store,
        mechanism_id: str,
        window: int,
        tick: int | None = None,
    ) -> float | None:
        """Compute Q1 rate restricted to one mechanism over a window.

        Diagnostic helper to flag mechanisms whose confidence is high but
        whose recent Q1 rate is degrading (scope drift detection).

        Args:
            evidence_store: EvidenceStore.
            mechanism_id: The mechanism to restrict to.
            window: Look-back window in ticks.
            tick: Current tick. Defaults to evidence_store.current_tick().

        Returns:
            Q1 rate in [0, 1], or None if no decided rows for this mechanism.
        """
        current = evidence_store.current_tick() if tick is None else int(tick)
        triples = [
            (row, mid, q)
            for row, mid, q in evidence_store.iter_decided_per_mechanism(window, current)
            if mid == mechanism_id
        ]
        if not triples:
            return None
        q1_count = sum(1 for _, _, q in triples if q == Quadrant.Q1)
        return q1_count / len(triples)

    def compute_q1_rate_per_env(
        self,
        evidence_store,
        env: tuple,
        window: int,
        tick: int | None = None,
    ) -> float | None:
        """Compute Q1 rate restricted to one ICP environment.

        Diagnostic helper to surface envs where mechanisms degrade
        (engine regression, driver issue, hardware quirk).

        Args:
            evidence_store: EvidenceStore.
            env: env_label tuple (market, cloud, region, zone, gpu_type).
            window: Look-back window in ticks.
            tick: Current tick. Defaults to evidence_store.current_tick().

        Returns:
            Q1 rate in [0, 1], or None if no decided pairs in this env.
        """
        current = evidence_store.current_tick() if tick is None else int(tick)
        triples = [
            (row, mid, q)
            for row, mid, q in evidence_store.iter_decided_per_mechanism(window, current)
            if row.env_label == env
        ]
        if not triples:
            return None
        q1_count = sum(1 for _, _, q in triples if q == Quadrant.Q1)
        return q1_count / len(triples)

    def compute_inst_regret(
        self,
        tick: int,
        evidence_store,
        window: int = 20,
        q1_star: float = DEFAULT_Q1_STAR,
    ) -> float:
        """Compute instantaneous mechanism regret at one tick.

        inst_regret_t = max(0, q1_star - q1_rate_t). Returns 0 when Q1
        rate is None (no decided pairs in window) so we do not claim a
        gap without evidence.

        Args:
            tick: Current tick.
            evidence_store: EvidenceStore.
            window: Q1-rate window.
            q1_star: Oracle target.

        Returns:
            Regret in [0, q1_star].
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
        """Compute cumulative mechanism regret over a tick range.

        R_mech_T = sum_t inst_regret_t.

        Args:
            ticks: Iterable of ticks to sum across.
            evidence_store: EvidenceStore.
            window: Q1-rate window.
            q1_star: Oracle target.

        Returns:
            Non-negative cumulative regret.
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
        """Compute the slow-loop operational signal.

        Average of recent instantaneous regrets over the last `window`
        ticks. Small means converged; large means still learning. Despite
        the name, this is an average of recent regrets, not a
        time-derivative.

        Args:
            tick: Current tick.
            evidence_store: EvidenceStore.
            window: Ticks to average over.
            q1_star: Oracle target.

        Returns:
            Non-negative slope.
        """
        start = max(0, tick - window + 1)
        gaps = [
            self.compute_inst_regret(t, evidence_store, window, q1_star)
            for t in range(start, tick + 1)
        ]
        if not gaps:
            return 0.0
        return float(np.mean(gaps))

    def compute_outcome_regret(
        self,
        evidence_store,
        ticks: Iterable[int],
    ) -> float:
        """Compute cumulative outcome regret over a tick range.

        Uses the best observed cluster J as the J_star proxy:
        R_out = sum_t (J_star - J_cluster_t). Used for dashboards and
        reporting only; not used to drive slow-loop knobs because it is
        too noisy.

        Args:
            evidence_store: EvidenceStore exposing cluster_J_at(tick).
            ticks: Iterable of ticks.

        Returns:
            Non-negative cumulative outcome regret.
        """
        history = [evidence_store.cluster_J_at(t) for t in ticks]
        history = [j for j in history if j is not None]
        if not history:
            return 0.0
        j_star = max(history)
        return float(sum(j_star - j for j in history))
