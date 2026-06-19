"""
Four-Quadrant classifier: V-CUSUM x Y-CUSUM -> Q1 / Q2 / Q3 / Q4.

Both axes are now CUSUM verdicts over sub-tick trajectories - V (mediators)
and Y (outcomes) treated symmetrically. The end-of-tick |y_hat - y| point check
was the degenerate n=1 case of Y-CUSUM; using the full trajectory tightens
the lucky-arm trap (Q3) because a transient outcome drift during the tick
no longer averages itself away into a passing point-check.

Why two axes (not just "did outcome match prediction?"):
    A naive outcome-only optimizer cannot distinguish
        "good outcome because mechanism is sound"        (Q1)
    from
        "good outcome because of luck"                   (Q3 - lucky-arm)
    Trapping Q3 is the main reason V-CUSUM exists. Outcome-only learning
    would reward lucky-arm mechanisms and entrench them.

The four quadrants:
    Q1  V matched   and  Y matched     -> replicable success
    Q2  V matched   and  Y diverged    -> sound mechanism, bad luck on outcome
    Q3  V diverged  and  Y matched     -> lucky-arm (PUNISH HARD)
    Q4  V diverged  and  Y diverged    -> falsified (PUNISH HARDER)

Quadrant labels drive per-edge and per-mechanism beta evidence updates
via the ConfidenceService beta update tables.
"""

from collections import Counter
from enum import Enum


class Quadrant(Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class QuadrantValidator:
    def classify_quadrant(self, v_cusum_result, y_cusum_result) -> Quadrant:
        """
        Definition: Combine the V-CUSUM (mediator) and Y-CUSUM (outcome)
                    verdicts into the four-quadrant label.
                        V_matched   and Y_matched    -> Q1
                        V_matched   and Y_diverged   -> Q2
                        V_diverged  and Y_matched    -> Q3 (lucky-arm)
                        V_diverged  and Y_diverged   -> Q4 (falsified)
        Usage:      Per (job, rank) deployed in [t-1, t], inside S2 of the
                    FSM tick. Returned label is stored in EvidenceStore and
                    consumed by ConfidenceService for beta evidence updates.
        Inputs:
            v_cusum_result : CusumResult enum or "matched"/"diverged" string
            y_cusum_result : CusumResult enum or "matched"/"diverged" string
        Outputs:
            Quadrant
        """
        v_matched = self._is_matched(v_cusum_result)
        y_matched = self._is_matched(y_cusum_result)

        if v_matched and y_matched:
            return Quadrant.Q1
        if v_matched and not y_matched:
            return Quadrant.Q2
        if not v_matched and y_matched:
            return Quadrant.Q3
        return Quadrant.Q4

    # def aggregate_quadrant_histogram(
    #     self,
    #     evidence_store,
    #     window: int,
    # ) -> dict[Quadrant, int]:
    #     """
    #     Definition: Count Q1/Q2/Q3/Q4 occurrences in recent DECIDED rows.
    #                 Excludes ICP-undecided rows so the denominator reflects
    #                 statistically-supported labels only.
    #     Usage:      Agent ingest; dashboards; building block for Q1-rate /
    #                 regret computations.
    #     Inputs:
    #         evidence_store : EvidenceStore exposing get_recently_decided(window)
    #         window         : tick count to look back
    #     Outputs:
    #         Dict[Quadrant -> int]  (zero-filled for absent quadrants)
    #     """
    #     rows = evidence_store.get_recently_decided(window)
    #     counts = Counter(r.q_label for r in rows)
    #     return {q: counts.get(q, 0) for q in Quadrant}

    def aggregate_quadrant_histogram(
        self,
        evidence_store,
        window: int,
        mechanism_id: str | None = None,
        tick: int | None = None,
    ) -> dict[Quadrant, int]:
        """Count Q1/Q2/Q3/Q4 occurrences across (row, mechanism) pairs.

        Excludes pairs whose Q label is None (those rows are undecided
        for that mechanism because at least one edge in the mechanism's
        bundle had ICP=UNDECIDED), so the denominator reflects
        statistically-supported labels only.

        Args:
            evidence_store: EvidenceStore exposing
                iter_decided_per_mechanism(window, tick).
            window: Tick count to look back.
            mechanism_id: If given, restrict to that mechanism's labels.
                If None, sum across all applicable mechanisms.
            tick: Inclusive upper-bound tick. If None, the store uses its
                own current_tick() so callers can omit it for "as of now"
                histograms.

        Returns:
            Dict mapping each Quadrant to its count, zero-filled for any
            absent quadrant.
        """
        counts: Counter[Quadrant] = Counter()
        for _row, mid, q in evidence_store.iter_decided_per_mechanism(window, tick):
            if mechanism_id is not None and mid != mechanism_id:
                continue
            counts[q] += 1
        return {q: counts.get(q, 0) for q in Quadrant}

    @staticmethod
    def _is_matched(cusum_result) -> bool:
        """Normalize Enum or string to a matched-bool."""
        if hasattr(cusum_result, "value"):
            return cusum_result.value == "matched"
        return cusum_result == "matched"
