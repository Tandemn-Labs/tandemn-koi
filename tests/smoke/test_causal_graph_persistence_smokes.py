"""Causal graph SQL contracts and round trips on disposable local databases."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from db.postgres import postgres_pool
from psycopg import sql
from src.bootstrap.initialization import init_causal_graph
from src.core.causal_graph_store_adapter import (
    PostgresConfidenceService,
    PostgresMechanismRegistry,
    parse_seed_tables,
)
from src.core.models import Mechanism
from src.validation.icp import ICPResult
from src.validation.quadrants import Quadrant


class CausalGraphPersistenceTests(unittest.TestCase):
    def test_seed_tables_parse_to_koi_runtime_objects(self):
        graph, registry = parse_seed_tables()

        self.assertEqual(len(graph.x), 96)
        self.assertEqual(len(graph.v), 22)
        self.assertEqual(len(graph.y), 5)
        self.assertEqual(len(graph.edge_table), 2222)
        self.assertEqual(len(registry.mechanism_table), 89)
        self.assertIn("queueing_under_burst", {m.name for m in registry.mechanism_table.values()})
        self.assertTrue(all(mid.startswith("M_") for mid in registry.mechanism_table))

    def test_registry_persists_only_new_mechanisms(self):
        pool = MagicMock()
        registry = PostgresMechanismRegistry(pool)
        mechanism = Mechanism(
            edge_ids=["batch_size->live_batch_size"],
            scope={"x": ["batch_size"], "v": ["live_batch_size"]},
            narrative="Batch size changes live batch size.",
        )
        mechanism_id = registry.add_mechanism(mechanism)
        duplicate_id = registry.add_mechanism(
            Mechanism(
                edge_ids=list(mechanism.edge_ids),
                scope=dict(mechanism.scope),
                narrative="Duplicate story.",
            )
        )

        self.assertEqual(duplicate_id, mechanism_id)
        conn = pool.connection.return_value.__enter__.return_value
        conn.execute.assert_called_once()
        self.assertEqual(conn.execute.call_args.args[1][0], mechanism_id)
        conn.transaction.assert_called_once()

    def test_confidence_flushes_metadata_in_one_transaction(self):
        pool = MagicMock()
        graph, seeds = parse_seed_tables()
        registry = PostgresMechanismRegistry(
            pool, seeds.mechanism_table, seeds.mechanism_metadata_table
        )
        confidence = PostgresConfidenceService(graph, registry, pool)
        edge_id = next(iter(graph.edge_table))
        mechanism_id = next(iter(registry.mechanism_table))
        confidence.apply_delta_c_edge(edge_id, Quadrant.Q1, ICPResult.ACCEPT, tick=7)
        confidence.apply_delta_c_mechanism(mechanism_id, Quadrant.Q4, tick=7)
        confidence.flush()

        conn = pool.connection.return_value.__enter__.return_value
        conn.transaction.assert_called_once()
        cursor = conn.cursor.return_value.__enter__.return_value
        edge_params = {params[-1]: params for params in cursor.executemany.call_args.args[1]}
        self.assertEqual(edge_params[edge_id][0], graph.edge_metadata_table[edge_id].alpha)
        self.assertEqual(edge_params[edge_id][2:4], (1, 7))
        self.assertEqual(edge_params[edge_id][4].obj["Q1"], 1)
        mechanism_params = {call.args[1][0]: call.args[1] for call in conn.execute.call_args_list}
        self.assertEqual(mechanism_params[mechanism_id][8:10], (1, 7))
        self.assertEqual(mechanism_params[mechanism_id][10].obj["Q4"], 1)


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL") and shutil.which("dbmate"),
    "Requires dbmate and TEST_DATABASE_URL pointing to local PostgreSQL",
)
class CausalGraphPostgresTests(unittest.TestCase):
    def setUp(self):
        url = os.environ["TEST_DATABASE_URL"]
        parsed = urlsplit(url)
        self.assertIn(parsed.hostname, {"localhost", "127.0.0.1", "::1"})
        name = f"koi_graph_test_{uuid4().hex}"
        admin = psycopg.connect(url, autocommit=True)
        self.addCleanup(admin.close)
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        self.addCleanup(admin.execute, sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        database_url = urlunsplit(parsed._replace(path=f"/{name}"))
        subprocess.run(
            [
                "dbmate",
                "--migration-lock",
                "--no-dump-schema",
                "--migrations-dir",
                str(Path(__file__).resolve().parents[2] / "db" / "migrations"),
                "migrate",
                "--strict",
            ],
            env={**os.environ, "DATABASE_URL": database_url},
            check=True,
            timeout=60,
        )
        self.pool = self.enterContext(postgres_pool(database_url))

    def test_seed_and_confidence_round_trip(self):
        graph, registry, confidence = init_causal_graph(self.pool)
        self.assertIsInstance(registry, PostgresMechanismRegistry)
        self.assertIsInstance(confidence, PostgresConfidenceService)
        self.assertEqual(len(graph.edge_table), 2222)
        self.assertEqual(len(registry.mechanism_table), 89)
        edge_id = next(iter(graph.edge_table))
        mechanism_id = next(iter(registry.mechanism_table))
        env = ("reserved", "aws", "us-east-1", "use1-az1", "H100")
        confidence.apply_delta_c_edge(edge_id, Quadrant.Q1, ICPResult.ACCEPT, env, tick=12)
        confidence.apply_delta_c_mechanism(mechanism_id, Quadrant.Q4, env, tick=12)
        graph.edge_metadata_table[edge_id].q3_frequency = 0.25
        registry.mechanism_metadata_table[mechanism_id].inspection_count = 4
        confidence.flush()

        reloaded_graph, reloaded_registry, _ = init_causal_graph(self.pool)
        self.assertEqual(reloaded_graph.node_table, graph.node_table)
        self.assertEqual(reloaded_graph.edge_table, graph.edge_table)
        self.assertEqual(reloaded_graph.edge_metadata_table, graph.edge_metadata_table)
        self.assertEqual(reloaded_registry.mechanism_table, registry.mechanism_table)
        self.assertEqual(
            reloaded_registry.mechanism_metadata_table, registry.mechanism_metadata_table
        )
        with self.pool.connection() as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT envs_last_seen FROM causal_mechanisms WHERE mechanism_id = %s",
                    (mechanism_id,),
                ).fetchone()["envs_last_seen"],
                [list(env)],
            )

    def test_admission_seeding_duplicate_and_archive_round_trip(self):
        graph, registry, confidence = init_causal_graph(self.pool)
        edge = next(iter(graph.edge_table.values()))
        mechanism = Mechanism(
            edge_ids=[edge.edge_id],
            scope={"x": [edge.src], "v": [edge.dst], "marker": "SQL ' quotes"},
            narrative="An agent's story; not SQL.",
        )
        mechanism_id = registry.add_mechanism(mechanism)
        self.assertEqual(registry.add_mechanism(mechanism), mechanism_id)
        confidence.seed_new_mechanism_confidence(mechanism_id, alpha=2.0, beta=3.0)
        registry.archive_mechanism(mechanism_id, "Smoke test archive")

        _, restored, restored_confidence = init_causal_graph(self.pool)
        self.assertEqual(len(restored.mechanism_table), 90)
        self.assertEqual(restored.get_mechanism(mechanism_id).narrative, mechanism.narrative)
        self.assertEqual(restored.get_mechanism(mechanism_id).scope, mechanism.scope)
        self.assertIn(mechanism_id, restored.mechanisms_by_status["archived"])
        self.assertIsNone(restored.get_mechanism(mechanism_id).archived_reason)
        self.assertEqual(restored_confidence.get_mechanism_alpha_beta(mechanism_id), (2.0, 3.0))

    def test_failed_seed_rolls_back_all_three_tables(self):
        with self.pool.connection() as conn:
            conn.execute("ALTER TABLE causal_mechanisms ADD CONSTRAINT reject_seed CHECK (false)")
        with self.assertRaises(psycopg.errors.CheckViolation):
            init_causal_graph(self.pool)
        with self.pool.connection() as conn:
            counts = conn.execute(
                """
                SELECT (SELECT count(*) FROM causal_nodes) AS nodes,
                       (SELECT count(*) FROM causal_edges) AS edges,
                       (SELECT count(*) FROM causal_mechanisms) AS mechanisms
                """
            ).fetchone()
        self.assertEqual(counts, {"nodes": 0, "edges": 0, "mechanisms": 0})

    def test_failed_confidence_flush_rolls_back_edges_and_mechanisms(self):
        graph, registry, confidence = init_causal_graph(self.pool)
        edge_id = next(iter(graph.edge_table))
        mechanism_id = next(iter(registry.mechanism_table))
        confidence.apply_delta_c_edge(edge_id, Quadrant.Q1, ICPResult.ACCEPT)
        confidence.apply_delta_c_mechanism(mechanism_id, Quadrant.Q4)
        with self.pool.connection() as conn:
            conn.execute(
                "ALTER TABLE causal_mechanisms ADD CONSTRAINT no_visits CHECK (visit_count = 0)"
            )
        with self.assertRaises(psycopg.errors.CheckViolation):
            confidence.flush()
        _, _, restored_confidence = init_causal_graph(self.pool)
        self.assertEqual(restored_confidence.get_edge_visit_count(edge_id), 0)
        self.assertEqual(restored_confidence.get_mechanism_visit_count(mechanism_id), 0)

    def test_partial_graph_is_rejected_without_reseeding(self):
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO causal_nodes (node_id, node_type) VALUES (%s, %s)", ("x", "X")
            )
        with self.assertRaisesRegex(ValueError, "Incomplete causal graph"):
            init_causal_graph(self.pool)
        with self.pool.connection() as conn:
            self.assertEqual(
                conn.execute("SELECT count(*) AS n FROM causal_nodes").fetchone()["n"], 1
            )

    def test_invalid_edge_types_and_mechanism_references_are_rejected(self):
        graph, registry, _ = init_causal_graph(self.pool)
        edge = next(iter(graph.edge_table.values()))
        mechanism_id = next(iter(registry.mechanism_table))
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE causal_edges SET src_type = %s WHERE edge_id = %s", ("bad", edge.edge_id)
            )
        with self.assertRaisesRegex(ValueError, "Invalid causal edges"):
            init_causal_graph(self.pool)
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE causal_edges SET src_type = %s WHERE edge_id = %s",
                (edge.src_type, edge.edge_id),
            )
            conn.execute(
                "UPDATE causal_mechanisms SET edge_ids = %s WHERE mechanism_id = %s",
                (["missing"], mechanism_id),
            )
        with self.assertRaisesRegex(ValueError, "Mechanisms reference unknown edges"):
            init_causal_graph(self.pool)


if __name__ == "__main__":
    unittest.main()
