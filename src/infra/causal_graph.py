"""Postgres-backed causal graph for Koi.

Loads topology and Beta confidence from ``CausalGraphStore``, builds
in-memory ``CandidateGraph`` and ``MechanismRegistry`` (with indexes),
and syncs metadata back after each tick's S3 confidence updates.
"""

from __future__ import annotations

from src.core.candidate_graph import CandidateGraph
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node
from tandemn_system_data.clients import CausalGraphStore, PostgresClient
from tandemn_system_data.models.causal_graph import (
    CausalEdge,
    CausalMechanism,
    CausalNode,
)
from tandemn_system_data.models.causal_graph import (
    EdgeMetadata as StoreEdgeMetadata,
)
from tandemn_system_data.models.causal_graph import (
    MechanismMetadata as StoreMechanismMetadata,
)


def _node_to_intel(node: CausalNode) -> Node:
    return Node(
        node_id=node.node_id,
        node_type=node.node_type,
        description=node.description,
        unit=node.unit,
    )


def _edge_to_intel(edge: CausalEdge) -> Edge:
    return Edge(
        edge_id=edge.edge_id,
        src=edge.src,
        dst=edge.dst,
        src_type=edge.src_type,
        dst_type=edge.dst_type,
        status=edge.status,
    )


def _edge_metadata_to_intel(metadata: StoreEdgeMetadata) -> EdgeMetadata:
    return EdgeMetadata(
        edge_id=metadata.edge_id,
        alpha=metadata.alpha,
        beta=metadata.beta,
        visit_count=metadata.visit_count,
        last_touched_tick=metadata.last_touched_tick,
        q_histogram=dict(metadata.q_histogram),
        envs_seen=set(metadata.envs_seen),
        q3_frequency=metadata.q3_frequency,
    )


def _mechanism_to_intel(mechanism: CausalMechanism) -> Mechanism:
    return Mechanism(
        mechanism_id=mechanism.mechanism_id,
        edge_ids=list(mechanism.edge_ids),
        scope=dict(mechanism.scope),
        narrative=mechanism.narrative,
        status=mechanism.status,
        archived_reason=mechanism.archived_reason,
    )


def _mechanism_metadata_to_intel(metadata: StoreMechanismMetadata) -> MechanismMetadata:
    return MechanismMetadata(
        mechanism_id=metadata.mechanism_id,
        alpha=metadata.alpha,
        beta=metadata.beta,
        visit_count=metadata.visit_count,
        envs_seen=set(metadata.envs_seen),
        last_touched_tick=metadata.last_touched_tick,
        q_histogram=dict(metadata.q_histogram),
        inspection_count=metadata.inspection_count,
    )


def _node_to_store(node: Node) -> CausalNode:
    return CausalNode(
        node_id=node.node_id,
        node_type=node.node_type,
        description=node.description,
        unit=node.unit,
    )


def _edge_to_store(edge: Edge) -> CausalEdge:
    return CausalEdge(
        edge_id=edge.edge_id,
        src=edge.src,
        dst=edge.dst,
        src_type=edge.src_type,
        dst_type=edge.dst_type,
        status=edge.status,
    )


def _edge_metadata_to_store(metadata: EdgeMetadata) -> StoreEdgeMetadata:
    return StoreEdgeMetadata(
        edge_id=metadata.edge_id,
        alpha=metadata.alpha,
        beta=metadata.beta,
        visit_count=metadata.visit_count,
        last_touched_tick=metadata.last_touched_tick,
        q_histogram=dict(metadata.q_histogram),
        envs_seen=set(metadata.envs_seen),
        q3_frequency=metadata.q3_frequency,
    )


def _mechanism_to_store(mechanism: Mechanism) -> CausalMechanism:
    return CausalMechanism(
        mechanism_id=mechanism.mechanism_id,
        edge_ids=list(mechanism.edge_ids),
        scope=dict(mechanism.scope),
        narrative=mechanism.narrative,
        status=mechanism.status,
        archived_reason=mechanism.archived_reason,
    )


def _mechanism_metadata_to_store(metadata: MechanismMetadata) -> StoreMechanismMetadata:
    return StoreMechanismMetadata(
        mechanism_id=metadata.mechanism_id,
        alpha=metadata.alpha,
        beta=metadata.beta,
        visit_count=metadata.visit_count,
        envs_seen=set(metadata.envs_seen),
        last_touched_tick=metadata.last_touched_tick,
        q_histogram=dict(metadata.q_histogram),
        inspection_count=metadata.inspection_count,
    )


class CausalGraphManager:
    """Wraps ``CausalGraphStore`` with intelligence graph types."""

    def __init__(self, user_id: str, postgres_client: PostgresClient) -> None:
        self._user_id = user_id
        self._store = CausalGraphStore(postgres_client, user_id=user_id)
        self._graph: CandidateGraph | None = None
        self._registry: MechanismRegistry | None = None

    @property
    def store(self) -> CausalGraphStore:
        return self._store

    @property
    def graph(self) -> CandidateGraph:
        if self._graph is None:
            raise RuntimeError("call load() before accessing graph")
        return self._graph

    @property
    def registry(self) -> MechanismRegistry:
        if self._registry is None:
            raise RuntimeError("call load() before accessing registry")
        return self._registry

    def load(self) -> tuple[CandidateGraph, MechanismRegistry]:
        node_rows = self._store.load_nodes()
        edge_rows, edge_meta_rows = self._store.load_edges()
        mech_rows, mech_meta_rows = self._store.load_mechanisms()

        node_table = {node_id: _node_to_intel(node) for node_id, node in node_rows.items()}
        edge_table = {edge_id: _edge_to_intel(edge) for edge_id, edge in edge_rows.items()}
        edge_metadata_table = {
            edge_id: _edge_metadata_to_intel(meta) for edge_id, meta in edge_meta_rows.items()
        }
        mechanism_table = {mid: _mechanism_to_intel(mech) for mid, mech in mech_rows.items()}
        mechanism_metadata_table = {
            mid: _mechanism_metadata_to_intel(meta) for mid, meta in mech_meta_rows.items()
        }

        self._graph = CandidateGraph(
            node_table=node_table,
            edge_table=edge_table,
            edge_metadata_table=edge_metadata_table,
        )
        self._registry = MechanismRegistry(
            mechanism_table=mechanism_table,
            mechanism_metadata_table=mechanism_metadata_table,
        )
        return self._graph, self._registry

    def seed(
        self,
        graph: CandidateGraph,
        registry: MechanismRegistry | None = None,
    ) -> None:
        """Persist an initial graph (boot). Replaces existing topology rows."""
        self._store.replace_nodes(_node_to_store(node) for node in graph.node_table.values())
        self._store.replace_edges(
            (_edge_to_store(edge) for edge in graph.edge_table.values()),
            {
                edge_id: _edge_metadata_to_store(meta)
                for edge_id, meta in graph.edge_metadata_table.items()
            },
        )
        if registry is not None:
            mech_store = {
                mid: _mechanism_to_store(mech) for mid, mech in registry.mechanism_table.items()
            }
            meta_store = {
                mid: _mechanism_metadata_to_store(meta)
                for mid, meta in registry.mechanism_metadata_table.items()
            }
            self._store.sync_mechanisms(mech_store, meta_store)
        self._graph = graph
        self._registry = registry or MechanismRegistry()

    def sync(self) -> None:
        """Write in-memory confidence and mechanism state back to Postgres."""
        if self._graph is None or self._registry is None:
            return
        self._store.sync_edge_metadata(
            {
                edge_id: _edge_metadata_to_store(meta)
                for edge_id, meta in self._graph.edge_metadata_table.items()
            }
        )
        self._store.sync_mechanisms(
            {
                mid: _mechanism_to_store(mech)
                for mid, mech in self._registry.mechanism_table.items()
            },
            {
                mid: _mechanism_metadata_to_store(meta)
                for mid, meta in self._registry.mechanism_metadata_table.items()
            },
        )

    def persist_new_mechanism(self, mechanism_id: str) -> None:
        """Upsert one mechanism after ``MechanismRegistry.add_mechanism``."""
        if self._registry is None:
            return
        mechanism = self._registry.mechanism_table[mechanism_id]
        metadata = self._registry.mechanism_metadata_table[mechanism_id]
        self._store.put_mechanism(
            _mechanism_to_store(mechanism), _mechanism_metadata_to_store(metadata)
        )
