"""Tests for the outcomes unique index in koi/tools/memory.py.

Phase 2d of contract-hardening. The unique index
`ux_outcomes_chain (decision_id, job_id, status)` is the SQL-level
second line of defense beneath inbox event dedup. Even if something
slips through the inbox (payload mutation, schema change, direct
programmatic call), duplicate per-chain outcome rows are impossible.

Important: the constraint is on the tuple (decision_id, job_id, status),
not on job_id alone. A replica failure followed by recovery into a
different status IS allowed — that's real, distinct ground truth.
"""

import pytest

from koi.tools.memory import AgenticMemory


@pytest.fixture
def memory():
    return AgenticMemory(db_path=":memory:")


def _record_decision(memory: AgenticMemory, decision_id: str = "d-1") -> str:
    return memory.record_decision(
        job_id="mo-abc",
        model_name="Qwen/Qwen3-32B",
        instance_type="g6e.12xlarge",
        gpu_type="L40S",
        tp=4,
        pp=2,
        dp=1,
        num_gpus=8,
        predicted_tps=1200.0,
        predicted_cost_per_hour=4.10,
        slo_deadline_hours=8.0,
        objective="cheapest",
        avg_input_tokens=1000,
        avg_output_tokens=1024,
        num_requests=5000,
    )


class TestDuplicateBlocking:
    def test_two_identical_calls_produce_one_row(self, memory):
        did = _record_decision(memory)
        first = memory.record_outcome(
            decision_id=did, job_id="mo-abc", status="succeeded", actual_tps=1234
        )
        second = memory.record_outcome(
            decision_id=did, job_id="mo-abc", status="succeeded", actual_tps=1234
        )
        assert memory.outcome_count() == 1
        assert first == second  # same outcome_id returned

    def test_different_actual_tps_still_dedups(self, memory):
        """If the same (decision_id, job_id, status) arrives with different tps,
        the unique index still blocks — first write wins."""
        did = _record_decision(memory)
        memory.record_outcome(
            decision_id=did, job_id="mo-abc", status="succeeded", actual_tps=1000
        )
        memory.record_outcome(
            decision_id=did, job_id="mo-abc", status="succeeded", actual_tps=2000
        )
        assert memory.outcome_count() == 1
        rows = memory.query_outcomes(limit=10)
        assert rows[0]["actual_tps"] == 1000  # first-write wins


class TestLegitimatelyDistinctOutcomes:
    def test_failure_then_success_both_recorded(self, memory):
        """A decision that first failed and later succeeded gets two outcome rows."""
        did = _record_decision(memory)
        memory.record_outcome(
            decision_id=did, job_id="mo-abc", status="replica_failed", actual_tps=300
        )
        memory.record_outcome(
            decision_id=did, job_id="mo-abc", status="succeeded", actual_tps=1200
        )
        assert memory.outcome_count() == 2

    def test_different_decision_id_recorded_separately(self, memory):
        d1 = _record_decision(memory, decision_id="d-1")
        d2 = memory.record_decision(
            job_id="mo-xyz",
            model_name="Qwen/Qwen3-32B",
            instance_type="g6e.48xlarge",
            gpu_type="L40S",
            tp=8,
            pp=1,
            dp=1,
            num_gpus=8,
            predicted_tps=2400.0,
            predicted_cost_per_hour=8.20,
            slo_deadline_hours=4.0,
            objective="cheapest",
            avg_input_tokens=1000,
            avg_output_tokens=1024,
            num_requests=5000,
        )
        memory.record_outcome(decision_id=d1, job_id="mo-abc", status="succeeded")
        memory.record_outcome(decision_id=d2, job_id="mo-abc", status="succeeded")
        assert memory.outcome_count() == 2

    def test_different_job_id_recorded_separately(self, memory):
        """Per-chain outcomes across a single group are distinct by job_id."""
        did = _record_decision(memory)
        memory.record_outcome(decision_id=did, job_id="mo-abc-r0", status="succeeded")
        memory.record_outcome(decision_id=did, job_id="mo-abc-r1", status="succeeded")
        memory.record_outcome(decision_id=did, job_id="mo-abc-r2", status="succeeded")
        assert memory.outcome_count() == 3


class TestPerChainGroupReplayScenario:
    """The real blocker #4 scenario: duplicate /job/complete webhook replays N
    per-chain outcomes, not one aggregate. The unique index makes the replay
    a SQL-level no-op."""

    def test_group_complete_replay_is_noop(self, memory):
        did = _record_decision(memory)
        chain_ids = [f"mo-abc-r{i}" for i in range(4)]
        # First delivery: 4 per-chain outcomes
        for cid in chain_ids:
            memory.record_outcome(
                decision_id=did, job_id=cid, status="succeeded", actual_tps=1200
            )
        assert memory.outcome_count() == 4
        # Replayed delivery (e.g., Orca retried /job/complete before inbox dedup)
        for cid in chain_ids:
            memory.record_outcome(
                decision_id=did, job_id=cid, status="succeeded", actual_tps=1200
            )
        assert memory.outcome_count() == 4  # still 4
