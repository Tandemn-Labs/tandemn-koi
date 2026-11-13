"""Check explicit resource-map inputs without database access or price calculation."""

import copy
import unittest
from types import SimpleNamespace

from src.infra.metrics.base import (
    NodePrice,
    RankContext,
    adapters_for_rank,
    load_plugins,
    resolve_rank_contexts,
)


def _rank():
    return {
        "rank_id": "rank_1",
        "cluster_id": "cluster_a",
        "engine_name": "vllm",
        "engine_version": "test-version",
        "device_source": "dcgm",
        "expected_replicas": 2,
        "expected_devices": 4,
        "config": {"block_size": 16},
        "committed_mechanism_id": "mechanism_1",
        "deploy_timestamp_utc": 1000.0,
        "nodes": [
            {"cluster_id": "cluster_a", "node_id": "node_1", "price_per_node_hour": 12.0},
            {"cluster_id": "cluster_a", "node_id": "node_2", "price_per_node_hour": 24.0},
        ],
    }


def _snapshot(ranks):
    jobs = [
        {
            "job_id": "job_1",
            "job_features": {"target_p99_ttft_ms": 100.0},
            "ranks": ranks,
        }
    ]
    return SimpleNamespace(active_jobs_summary=lambda: jobs)


class _Adapter:
    def __init__(self, kind, name):
        self.kind = kind
        self.name = name


class MetricsBaseSmokeTests(unittest.TestCase):
    def test_plugins_load_once_and_select_sources_per_rank(self):
        vllm = _Adapter("engine", "vllm")
        sglang = _Adapter("engine", "sglang")
        dcgm = _Adapter("device", "dcgm")
        amd = _Adapter("device", "amd_exporter")
        contexts = resolve_rank_contexts(
            _snapshot(
                [
                    _rank(),
                    {
                        **_rank(),
                        "rank_id": "rank_2",
                        "engine_name": "sglang",
                        "device_source": "amd_exporter",
                    },
                ]
            )
        )

        plugins = load_plugins(adapter for adapter in (vllm, sglang, dcgm, amd))

        self.assertEqual(adapters_for_rank(contexts[0], plugins), (vllm, dcgm))
        self.assertEqual(adapters_for_rank(contexts[1], plugins), (sglang, amd))

    def test_unknown_sources_do_not_fall_back(self):
        context = resolve_rank_contexts(_snapshot([_rank()]))[0]

        with self.assertRaisesRegex(ValueError, "no device adapter"):
            adapters_for_rank(context, load_plugins([_Adapter("engine", "vllm")]))

    def test_plugin_loading_rejects_bad_and_duplicate_registrations(self):
        for adapter in (
            _Adapter("runtime", "vllm"),
            _Adapter("engine", ""),
        ):
            with self.subTest(adapter=adapter), self.assertRaises(ValueError):
                load_plugins([adapter])
        plugin = SimpleNamespace(kind="engine", name="vllm")
        self.assertIs(load_plugins([plugin])[("engine", "vllm")], plugin)
        with self.assertRaisesRegex(ValueError, "duplicate metric adapter"):
            load_plugins([_Adapter("engine", "vllm"), _Adapter("engine", "vllm")])

    def test_resolves_rank_and_node_facts_without_chain_identifiers(self):
        context = resolve_rank_contexts(_snapshot([_rank()]))[0]

        self.assertEqual((context.job_id, context.rank_id), ("job_1", "rank_1"))
        self.assertEqual(context.cluster_id, "cluster_a")
        self.assertEqual(context.engine_name, "vllm")
        self.assertEqual(context.engine_version, "test-version")
        self.assertEqual(context.device_source, "dcgm")
        self.assertEqual(context.expected_replicas, 2)
        self.assertEqual(context.expected_devices, 4)
        self.assertEqual(context.config, {"block_size": 16})
        self.assertEqual(context.job_features, {"target_p99_ttft_ms": 100.0})
        self.assertEqual(context.committed_mechanism_id, "mechanism_1")
        self.assertEqual(context.deploy_timestamp_utc, 1000.0)
        self.assertEqual(
            context.nodes,
            (NodePrice("cluster_a", "node_1", 12.0), NodePrice("cluster_a", "node_2", 24.0)),
        )
        self.assertFalse(hasattr(context, "chain_id"))
        self.assertFalse(hasattr(context, "chain_index"))
        self.assertFalse(hasattr(context, "price_per_rank_hour"))

    def test_different_engines_in_one_job_remain_separate(self):
        second = _rank()
        second.update(rank_id="rank_2", engine_name="sglang", device_source="amd_exporter")

        contexts = resolve_rank_contexts(_snapshot([_rank(), second]))

        self.assertEqual([context.rank_id for context in contexts], ["rank_1", "rank_2"])
        self.assertEqual([context.engine_name for context in contexts], ["vllm", "sglang"])
        self.assertEqual([context.device_source for context in contexts], ["dcgm", "amd_exporter"])

    def test_missing_optional_facts_are_not_guessed(self):
        context = resolve_rank_contexts(
            _snapshot(
                [
                    {
                        "rank_id": "rank_1",
                        "cluster_id": "cluster_a",
                        "engine_name": "vllm",
                        "device_source": "dcgm",
                        "price_per_unit_hour": 99.0,
                    }
                ]
            )
        )[0]

        self.assertEqual(context.nodes, ())
        self.assertIsNone(context.engine_version)
        self.assertIsNone(context.expected_replicas)
        self.assertIsNone(context.expected_devices)
        self.assertIsNone(context.committed_mechanism_id)
        self.assertIsNone(context.deploy_timestamp_utc)

    def test_job_facts_override_rank_record_copies(self):
        rank = _rank()
        rank.update(
            job_id="other_job",
            job_features={"target_p99_ttft_ms": 999.0},
            unknown_field="ignored",
        )

        context = resolve_rank_contexts(_snapshot([rank]))[0]

        self.assertEqual(context.job_id, "job_1")
        self.assertEqual(context.job_features, {"target_p99_ttft_ms": 100.0})

    def test_node_prices_keep_zero_and_do_not_use_unit_price_aliases(self):
        rank = _rank()
        rank["nodes"][0]["price_per_node_hour"] = 0.0
        rank["nodes"][1].pop("price_per_node_hour")
        rank["nodes"][1]["price_per_unit_hour"] = 99.0

        context = resolve_rank_contexts(_snapshot([rank]))[0]

        self.assertEqual(context.nodes[0].price_per_node_hour, 0.0)
        self.assertIsNone(context.nodes[1].price_per_node_hour)

    def test_cluster_identity_distinguishes_same_node_name(self):
        self.assertNotEqual(
            NodePrice("cluster_a", "node_1", 12.0),
            NodePrice("cluster_b", "node_1", 12.0),
        )

    def test_missing_or_empty_source_facts_are_rejected(self):
        for field in ("job_id", "rank_id", "cluster_id", "engine_name", "device_source"):
            for value in (None, "", "   "):
                with self.subTest(field=field, value=value):
                    kwargs = {
                        "job_id": "job_1",
                        "rank_id": "rank_1",
                        "cluster_id": "cluster_a",
                        "engine_name": "vllm",
                        "device_source": "dcgm",
                        field: value,
                    }
                    with self.assertRaisesRegex(ValueError, field):
                        RankContext(**kwargs)

    def test_missing_required_record_fields_are_rejected(self):
        for field in ("rank_id", "cluster_id", "engine_name", "device_source"):
            with self.subTest(field=field):
                rank = _rank()
                rank.pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    resolve_rank_contexts(_snapshot([rank]))
        for field in ("cluster_id", "node_id"):
            with self.subTest(node_field=field):
                rank = _rank()
                rank["nodes"][0].pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    resolve_rank_contexts(_snapshot([rank]))

    def test_invalid_prices_are_rejected(self):
        for price in (-1.0, True, "12", float("nan"), float("inf")):
            with (
                self.subTest(price=price),
                self.assertRaisesRegex(ValueError, "price_per_node_hour"),
            ):
                NodePrice("cluster_a", "node_1", price)

    def test_live_counts_are_nonnegative_integers_or_unknown(self):
        for field in ("expected_replicas", "expected_devices"):
            for value in (-1, True, 1.5, "2"):
                with self.subTest(field=field, value=value):
                    rank = _rank()
                    rank[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        resolve_rank_contexts(_snapshot([rank]))
        rank = _rank()
        rank.update(expected_replicas=0, expected_devices=0)
        context = resolve_rank_contexts(_snapshot([rank]))[0]
        self.assertEqual(context.expected_replicas, 0)
        self.assertEqual(context.expected_devices, 0)

    def test_conflicting_duplicate_rank_records_are_rejected(self):
        second = _rank()
        second["engine_name"] = "sglang"

        with self.assertRaisesRegex(ValueError, "duplicate rank context"):
            resolve_rank_contexts(_snapshot([_rank(), second]))

    def test_rank_identity_includes_the_job(self):
        snapshot = _snapshot([_rank()])
        snapshot.active_jobs_summary().append({"job_id": "job_2", "ranks": [_rank()]})

        contexts = resolve_rank_contexts(snapshot)

        self.assertEqual(
            [(context.job_id, context.rank_id) for context in contexts],
            [("job_1", "rank_1"), ("job_2", "rank_1")],
        )

    def test_malformed_map_records_are_rejected(self):
        for ranks, field in ((None, "ranks"), (["not-a-record"], "rank")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                resolve_rank_contexts(_snapshot(ranks))
        for field, value in (("nodes", {}), ("nodes", [None]), ("config", [])):
            with self.subTest(field=field, value=value):
                rank = _rank()
                rank[field] = value
                with self.assertRaisesRegex(ValueError, field.rstrip("s")):
                    resolve_rank_contexts(_snapshot([rank]))

    def test_legacy_chains_are_not_used_to_infer_rank_context(self):
        snapshot = SimpleNamespace(
            active_jobs_summary=lambda: [{"job_id": "job_1", "active_chains": []}]
        )

        with self.assertRaisesRegex(ValueError, "requires explicit ranks"):
            resolve_rank_contexts(snapshot)

    def test_empty_snapshot_returns_no_contexts(self):
        snapshot = SimpleNamespace(active_jobs_summary=list)

        self.assertEqual(resolve_rank_contexts(snapshot), [])
        self.assertEqual(resolve_rank_contexts(_snapshot([])), [])

    def test_resolving_contexts_does_not_change_map_records(self):
        snapshot = _snapshot([_rank()])
        jobs = snapshot.active_jobs_summary()
        before = copy.deepcopy(jobs)

        contexts = resolve_rank_contexts(snapshot)

        self.assertEqual(jobs, before)
        self.assertIsNot(contexts[0].config, jobs[0]["ranks"][0]["config"])
        self.assertIsNot(contexts[0].job_features, jobs[0]["job_features"])


if __name__ == "__main__":
    unittest.main()
