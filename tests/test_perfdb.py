"""Tests for koi/tools/perfdb.py"""

import pytest
from koi.tools.perfdb import PerfDB, query_perfdb

CSV_PATH = "perfdb/perfdb_all.csv"


@pytest.fixture
def perfdb():
    return PerfDB(CSV_PATH)


class TestPerfDBLoad:
    def test_loads(self, perfdb):
        assert perfdb.record_count > 200

    def test_models(self, perfdb):
        assert "Qwen/Qwen2.5-72B-Instruct" in perfdb.models

    def test_gpu_types(self, perfdb):
        gpu_types = perfdb.gpu_types
        assert any("L40S" in g for g in gpu_types)
        assert any("A100" in g for g in gpu_types)

    def test_has_throughput(self, perfdb):
        assert "throughput_tps" in perfdb.df.columns
        assert perfdb.df["throughput_tps"].min() > 0

    def test_io_ratio_computed(self, perfdb):
        assert "io_ratio" in perfdb.df.columns


class TestPerfDBQuery:
    def test_query_by_model(self, perfdb):
        results = perfdb.query(model_name="Qwen/Qwen2.5-72B-Instruct")
        assert len(results) > 0
        assert all("Qwen" in r["model_name"] for r in results)

    def test_query_by_gpu(self, perfdb):
        results = perfdb.query(gpu_type="L40S")
        assert len(results) > 0
        assert all("L40S" in str(r.get("gpu_type", "")) for r in results)

    def test_query_by_tp(self, perfdb):
        results = perfdb.query(tp=4)
        assert len(results) > 0
        assert all(r.get("tp") == 4 for r in results)

    def test_query_io_ratio_range(self, perfdb):
        results = perfdb.query(io_ratio_min=2.0, io_ratio_max=6.0)
        # All returned should have io_ratio in range
        for r in results:
            if "io_ratio" in r:
                assert 2.0 <= r["io_ratio"] <= 6.0

    def test_query_limit(self, perfdb):
        results = perfdb.query(limit=5)
        assert len(results) <= 5

    def test_query_sort(self, perfdb):
        results = perfdb.query(sort_by="throughput_tps", limit=10)
        tps_values = [r["throughput_tps"] for r in results]
        assert tps_values == sorted(tps_values, reverse=True)

    def test_query_no_results(self, perfdb):
        results = perfdb.query(model_name="nonexistent-model-xyz")
        assert len(results) == 0


class TestDistinctModels:
    def test_distinct(self, perfdb):
        models = perfdb.get_distinct_models()
        assert len(models) > 0
        names = [m["model_name"] for m in models]
        assert len(names) == len(set(names))  # unique

    def test_has_record_count(self, perfdb):
        models = perfdb.get_distinct_models()
        for m in models:
            assert "records_count" in m
            assert m["records_count"] > 0


class TestToolFunction:
    def test_query_perfdb_formatted(self, perfdb):
        result = query_perfdb(perfdb, model_name="Qwen/Qwen2.5-72B-Instruct", limit=5)
        assert "PerfDB" in result
        assert "TPS=" in result

    def test_query_perfdb_no_results(self, perfdb):
        result = query_perfdb(perfdb, model_name="nonexistent")
        assert "No PerfDB records" in result
