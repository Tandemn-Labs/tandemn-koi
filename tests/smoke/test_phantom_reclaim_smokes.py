"""Layer 2 phantom-replica reconciliation smokes: dead replicas are reaped,
live-idle headroom and cold-starting ranks are not, and a rank with no
telemetry is trusted (sim / monitoring-gap fail-safe)."""

import unittest
from datetime import UTC, datetime, timedelta

from src.infra.resource_map import ResourceMapManager, chain_id_for_rank


class _StubManager(ResourceMapManager):
    """ResourceMapManager with the telemetry read stubbed, so we test the
    pure reap decision without a database."""

    def __init__(self, live_map):
        super().__init__(user_id="usr_test", postgres_client=object())
        self._live_map = live_map

    def observed_replica_liveness(self, user_id, job_ids, window_seconds=None):
        return self._live_map


def _chain(rank_id, index, *, age_seconds=10_000, job_id="job1"):
    created = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
    return {
        "chain_id": chain_id_for_rank(rank_id, index),
        "chain_index": index,
        "job_id": job_id,
        "rank_created_at": created,
        "shape_json": {"rank_id": rank_id, "count": 8, "instance_type": "a3-highgpu-8g"},
    }


class PhantomChainSmokeTests(unittest.TestCase):
    def test_dead_replica_of_a_live_rank_is_reaped(self):
        # run-17 shape: rank committed 2 replicas, only replica 0 reporting.
        chains = [_chain("r1", 0), _chain("r1", 1)]
        mgr = _StubManager({"r1": {"live_indexes": {0}, "avg_util": 40.0}})
        reap = mgr._phantom_chain_ids(chains, "usr_test")
        self.assertEqual(reap, {chain_id_for_rank("r1", 1)})

    def test_live_idle_headroom_is_never_reaped(self):
        # Both replicas report (0% util is still alive) -> keep both.
        chains = [_chain("r1", 0), _chain("r1", 1)]
        mgr = _StubManager({"r1": {"live_indexes": {0, 1}, "avg_util": 0.1}})
        self.assertEqual(mgr._phantom_chain_ids(chains, "usr_test"), set())

    def test_cold_starting_rank_within_grace_is_kept(self):
        chains = [_chain("r1", 0, age_seconds=60), _chain("r1", 1, age_seconds=60)]
        mgr = _StubManager({"r1": {"live_indexes": {0}, "avg_util": 5.0}})
        self.assertEqual(mgr._phantom_chain_ids(chains, "usr_test"), set())

    def test_rank_with_no_telemetry_is_trusted(self):
        # Fail-safe: no live replicas at all -> trust the commit (sim, gaps,
        # brand-new rank). Total death is handled by the store status filter.
        chains = [_chain("r1", 0), _chain("r1", 1)]
        mgr = _StubManager({})
        self.assertEqual(mgr._phantom_chain_ids(chains, "usr_test"), set())

    def test_single_replica_live_rank_untouched(self):
        chains = [_chain("r1", 0)]
        mgr = _StubManager({"r1": {"live_indexes": {0}, "avg_util": 50.0}})
        self.assertEqual(mgr._phantom_chain_ids(chains, "usr_test"), set())

    def test_multiple_dead_replicas_all_reaped(self):
        chains = [_chain("r1", 0), _chain("r1", 1), _chain("r1", 2)]
        mgr = _StubManager({"r1": {"live_indexes": {0}, "avg_util": 30.0}})
        reap = mgr._phantom_chain_ids(chains, "usr_test")
        self.assertEqual(reap, {chain_id_for_rank("r1", 1), chain_id_for_rank("r1", 2)})


class AsUtcSmokeTests(unittest.TestCase):
    def test_parses_iso_string_and_naive_datetime(self):
        iso = "2026-09-09T08:00:00+00:00"
        self.assertIsNotNone(ResourceMapManager._as_utc(iso))
        naive = datetime(2026, 9, 9, 8, 0, 0)
        self.assertEqual(ResourceMapManager._as_utc(naive).tzinfo, UTC)
        self.assertIsNone(ResourceMapManager._as_utc(None))
        self.assertIsNone(ResourceMapManager._as_utc("not-a-date"))


if __name__ == "__main__":
    unittest.main()
