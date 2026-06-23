import unittest
from datetime import UTC, datetime

from sqlalchemy import text
from src.bootstrap.initialization import init_causal_graph
from src.core.candidate_graph import CandidateGraph
from src.core.causal_graph_store_adapter import (
    StoreBackedConfidenceService,
    StoreBackedMechanismRegistry,
    parse_seed_tables,
)
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node
from src.validation.icp import ICPResult
from src.validation.quadrants import Quadrant
from tandemn_system_data.clients import PostgresClient
from tandemn_system_data.db import UserRow
from tandemn_system_data.ids import new_user_id


class FakeCausalGraphStore:
    def __init__(self):
        self.put_mechanisms = []
        self.synced_edge_metadata = None
        self.synced_mechanisms = None
        self.synced_mechanism_metadata = None

    def put_mechanism(self, mechanism, metadata):
        self.put_mechanisms.append((mechanism, metadata))

    def sync_edge_metadata(self, metadata):
        self.synced_edge_metadata = metadata

    def sync_mechanisms(self, mechanisms, metadata):
        self.synced_mechanisms = mechanisms
        self.synced_mechanism_metadata = metadata


class CausalGraphAdapterSmokeTests(unittest.TestCase):
    def test_seed_tables_parse_to_koi_runtime_objects(self):
        graph, registry = parse_seed_tables()

        self.assertEqual(len(graph.x), 103)
        self.assertEqual(len(graph.v), 22)
        self.assertEqual(len(graph.y), 5)
        self.assertEqual(len(graph.edge_table), 2376)
        self.assertEqual(len(registry.mechanism_table), 100)

        names = {mechanism.name for mechanism in registry.mechanism_table.values()}
        self.assertIn("queueing_under_burst", names)
        self.assertTrue(all(mid.startswith("M_") for mid in registry.mechanism_table))

    def test_store_backed_registry_persists_only_new_mechanisms(self):
        store = FakeCausalGraphStore()
        registry = StoreBackedMechanismRegistry(store)
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
        self.assertEqual(len(store.put_mechanisms), 1)
        self.assertEqual(store.put_mechanisms[0][0].mechanism_id, mechanism_id)

    def test_store_backed_confidence_flushes_metadata(self):
        edge_id = "batch_size->live_batch_size"
        mechanism_id = "M_demo"
        graph = CandidateGraph(
            node_table={
                "batch_size": Node("batch_size", "X"),
                "live_batch_size": Node("live_batch_size", "V"),
            },
            edge_table={
                edge_id: Edge(edge_id, "batch_size", "live_batch_size", "X", "V"),
            },
            edge_metadata_table={edge_id: EdgeMetadata(edge_id=edge_id)},
        )
        store = FakeCausalGraphStore()
        registry = StoreBackedMechanismRegistry(
            store,
            mechanism_table={
                mechanism_id: Mechanism(
                    mechanism_id=mechanism_id,
                    edge_ids=[edge_id],
                    scope={"x": ["batch_size"], "v": ["live_batch_size"]},
                    narrative="Batch size changes live batch size.",
                )
            },
            mechanism_metadata_table={mechanism_id: MechanismMetadata(mechanism_id)},
        )
        confidence = StoreBackedConfidenceService(graph, registry, store)

        confidence.apply_delta_c_edge(edge_id, Quadrant.Q1, ICPResult.ACCEPT)
        confidence.apply_delta_c_mechanism(mechanism_id, Quadrant.Q4)
        confidence.flush()

        self.assertGreater(store.synced_edge_metadata[edge_id].alpha, 1.0)
        self.assertEqual(store.synced_edge_metadata[edge_id].visit_count, 1)
        self.assertEqual(store.synced_mechanism_metadata[mechanism_id].visit_count, 1)
        self.assertEqual(store.synced_mechanisms[mechanism_id].mechanism_id, mechanism_id)

    def test_init_causal_graph_imports_and_persists_through_postgres(self):
        client = PostgresClient()
        user_id = new_user_id()
        with client.begin() as session:
            session.add(
                UserRow(user_id=user_id, name="koi causal smoke", created_at=datetime.now(UTC))
            )

        try:
            graph, registry, confidence = init_causal_graph(user_id, postgres_client=client)
            self.assertIsInstance(registry, StoreBackedMechanismRegistry)
            self.assertIsInstance(confidence, StoreBackedConfidenceService)
            self.assertEqual(len(graph.edge_table), 2376)
            self.assertEqual(len(registry.mechanism_table), 100)

            edge = next(
                edge
                for edge in graph.edge_table.values()
                if edge.src_type == "X" and edge.dst_type == "V"
            )
            mechanism_id = next(iter(registry.mechanism_table))

            confidence.apply_delta_c_edge(edge.edge_id, Quadrant.Q1, ICPResult.ACCEPT)
            confidence.apply_delta_c_mechanism(mechanism_id, Quadrant.Q4)
            confidence.flush()

            _, reloaded_registry, reloaded_confidence = init_causal_graph(
                user_id,
                postgres_client=client,
            )
            self.assertEqual(reloaded_confidence.get_edge_visit_count(edge.edge_id), 1)
            self.assertEqual(reloaded_confidence.get_mechanism_visit_count(mechanism_id), 1)

            new_mechanism = Mechanism(
                edge_ids=[edge.edge_id],
                scope={"x": [edge.src], "v": [edge.dst], "test_marker": user_id},
                narrative="DB smoke mechanism should persist after admission.",
            )
            new_mechanism_id = reloaded_registry.add_mechanism(new_mechanism)
            reloaded_confidence.seed_new_mechanism_confidence(new_mechanism_id)

            _, final_registry, _ = init_causal_graph(user_id, postgres_client=client)
            self.assertIn(new_mechanism_id, final_registry.mechanism_table)
        finally:
            with client.begin() as session:
                session.execute(
                    text("delete from users where user_id = :user_id"),
                    {"user_id": user_id},
                )


if __name__ == "__main__":
    unittest.main()
