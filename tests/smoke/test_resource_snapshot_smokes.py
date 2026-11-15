from unittest.mock import Mock

import pytest
from src.infra.resource_map import ClusterResourceSnapshot, ResourceMapManager


@pytest.mark.parametrize("with_pool", [False, True])
def test_resource_and_inventory_reads_require_s0_without_store_fallback(with_pool):
    pool = Mock() if with_pool else None
    manager = ResourceMapManager(pool=pool)
    manager._client = Mock(side_effect=AssertionError("Store must not be read"))
    for read in (
        manager.get_resource_map,
        manager.refresh_resource_map,
        manager.resources_summary,
        manager.dynamic_view,
        manager.get_running_jobs,
        manager.get_running_chains,
    ):
        with pytest.raises(ValueError, match="capture it in S0"):
            read()
    manager._client.assert_not_called()
    if pool is not None:
        pool.connection.assert_not_called()


def test_resource_map_construction_waits_for_s0():
    pool = Mock()
    manager = ResourceMapManager(pool=pool)
    assert manager._pool is pool
    with pytest.raises(ValueError, match="capture it in S0"):
        manager.snapshot()
    pool.connection.assert_not_called()


def test_in_memory_snapshot_access_does_not_need_a_store_client():
    manager = ResourceMapManager()
    stored = {"tick": 5, "snapshot_id": "map-5", "stored_at": None, "resource_map": {}}
    manager._tick_snapshot = ClusterResourceSnapshot(5, {}, [], [], stored)
    assert manager.get_resource_map() == {}
    assert manager.resources_summary() == {}
    assert manager.get_waiting_jobs() == []
    assert manager.dynamic_view()["snapshot_id"] == "map-5"
