"""Koi causal-graph adapters for Tandemn Store and seeded JSON tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.candidate_graph import CandidateGraph
from src.core.confidence_service import ConfidenceService
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node
from tandemn_system_data.models import (  # type: ignore[import-untyped]
    CausalEdge,
    CausalMechanism,
    CausalNode,
)
from tandemn_system_data.models.causal_graph import (  # type: ignore[import-untyped]
    EdgeMetadata as StoreEdgeMetadata,
)
from tandemn_system_data.models.causal_graph import (
    MechanismMetadata as StoreMechanismMetadata,
)

DEFAULT_EDGE_TABLE_PATH = (
    Path(__file__).resolve().parents[1] / "bootstrap" / "edge_confidence_table.json"
)
DEFAULT_MECHANISM_TABLE_PATH = (
    Path(__file__).resolve().parents[1] / "bootstrap" / "mechanism_seed_table.json"
)
DEFAULT_Q_HISTOGRAM = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}


class StoreBackedMechanismRegistry(MechanismRegistry):
    """MechanismRegistry that persists accepted mechanism mutations."""

    def __init__(
        self,
        store,
        mechanism_table: dict[str, Mechanism] | None = None,
        mechanism_metadata_table: dict[str, MechanismMetadata] | None = None,
    ):
        super().__init__(mechanism_table, mechanism_metadata_table)
        self._store = store

    def add_mechanism(self, mechanism: Mechanism) -> str:
        existing_ids = set(self.mechanism_table)
        mechanism_id = super().add_mechanism(mechanism)
        if mechanism_id not in existing_ids:
            self.persist_mechanism(mechanism_id)
        return mechanism_id

    def archive_mechanism(self, mechanism_id: str, reason: str) -> bool:
        archived = super().archive_mechanism(mechanism_id, reason)
        self.persist_mechanism(mechanism_id)
        return archived

    def persist_mechanism(self, mechanism_id: str) -> None:
        self._store.put_mechanism(
            koi_mechanism_to_store(self.mechanism_table[mechanism_id]),
            koi_mechanism_metadata_to_store(self.mechanism_metadata_table[mechanism_id]),
        )


class StoreBackedConfidenceService(ConfidenceService):
    """ConfidenceService that can flush Koi's in-memory Beta state to Store."""

    def __init__(self, candidate_graph, mechanism_registry, store):
        super().__init__(candidate_graph, mechanism_registry)
        self._store = store

    def seed_new_mechanism_confidence(
        self,
        mechanism_id: str,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> float:
        confidence = super().seed_new_mechanism_confidence(mechanism_id, alpha, beta)
        self._persist_mechanism(mechanism_id)
        return confidence

    def flush(self) -> None:
        self._store.sync_edge_metadata(
            {
                edge_id: koi_edge_metadata_to_store(metadata)
                for edge_id, metadata in self.candidate_graph.edge_metadata_table.items()
            }
        )
        self._store.sync_mechanisms(
            {
                mechanism_id: koi_mechanism_to_store(mechanism)
                for mechanism_id, mechanism in self.mechanism_registry.mechanism_table.items()
            },
            {
                mechanism_id: koi_mechanism_metadata_to_store(metadata)
                for mechanism_id, metadata in self.mechanism_registry.mechanism_metadata_table.items()
            },
        )

    def _persist_mechanism(self, mechanism_id: str) -> None:
        persist = getattr(self.mechanism_registry, "persist_mechanism", None)
        if callable(persist):
            persist(mechanism_id)
            return
        self._store.put_mechanism(
            koi_mechanism_to_store(self.mechanism_registry.mechanism_table[mechanism_id]),
            koi_mechanism_metadata_to_store(
                self.mechanism_registry.mechanism_metadata_table[mechanism_id]
            ),
        )


def store_node_to_koi(node: CausalNode) -> Node:
    return Node(
        node_id=node.node_id,
        node_type=node.node_type,
        description=node.description,
        unit=node.unit,
    )


def koi_node_to_store(node: Node) -> CausalNode:
    return CausalNode(
        node_id=node.node_id,
        node_type=node.node_type,
        description=node.description,
        unit=node.unit,
    )


def store_edge_to_koi(edge: CausalEdge) -> Edge:
    return Edge(
        edge_id=edge.edge_id,
        src=edge.src,
        dst=edge.dst,
        src_type=edge.src_type,
        dst_type=edge.dst_type,
        status=edge.status,
    )


def koi_edge_to_store(edge: Edge) -> CausalEdge:
    return CausalEdge(
        edge_id=edge.edge_id,
        src=edge.src,
        dst=edge.dst,
        src_type=edge.src_type,
        dst_type=edge.dst_type,
        status=edge.status,
    )


def store_edge_metadata_to_koi(metadata: StoreEdgeMetadata) -> EdgeMetadata:
    return EdgeMetadata(
        edge_id=metadata.edge_id,
        alpha=metadata.alpha,
        beta=metadata.beta,
        visit_count=metadata.visit_count,
        last_touched_tick=metadata.last_touched_tick,
        q_histogram=_q_histogram(metadata.q_histogram),
        envs_seen=set(metadata.envs_seen),
        q3_frequency=metadata.q3_frequency,
    )


def koi_edge_metadata_to_store(metadata: EdgeMetadata) -> StoreEdgeMetadata:
    return StoreEdgeMetadata(
        edge_id=metadata.edge_id,
        alpha=metadata.alpha,
        beta=metadata.beta,
        visit_count=metadata.visit_count,
        last_touched_tick=metadata.last_touched_tick,
        q_histogram=_q_histogram(metadata.q_histogram),
        envs_seen=set(metadata.envs_seen),
        q3_frequency=metadata.q3_frequency,
    )


def store_mechanism_to_koi(mechanism: CausalMechanism) -> Mechanism:
    return Mechanism(
        mechanism_id=mechanism.mechanism_id,
        name=mechanism.name or None,
        edge_ids=list(mechanism.edge_ids),
        scope=dict(mechanism.scope),
        narrative=mechanism.narrative,
        status=mechanism.status,
        archived_reason=mechanism.archived_reason,
    )


def koi_mechanism_to_store(mechanism: Mechanism) -> CausalMechanism:
    return CausalMechanism(
        mechanism_id=mechanism.mechanism_id,
        name=mechanism.name or "",
        edge_ids=list(mechanism.edge_ids),
        scope=dict(mechanism.scope),
        narrative=mechanism.narrative,
        status=mechanism.status,
        archived_reason=mechanism.archived_reason,
    )


def store_mechanism_metadata_to_koi(metadata: StoreMechanismMetadata) -> MechanismMetadata:
    return MechanismMetadata(
        mechanism_id=metadata.mechanism_id,
        alpha=metadata.alpha,
        beta=metadata.beta,
        visit_count=metadata.visit_count,
        envs_seen=set(metadata.envs_seen),
        last_touched_tick=metadata.last_touched_tick,
        q_histogram=_q_histogram(metadata.q_histogram),
        inspection_count=metadata.inspection_count,
    )


def koi_mechanism_metadata_to_store(metadata: MechanismMetadata) -> StoreMechanismMetadata:
    return StoreMechanismMetadata(
        mechanism_id=metadata.mechanism_id,
        alpha=metadata.alpha,
        beta=metadata.beta,
        visit_count=metadata.visit_count,
        envs_seen=set(metadata.envs_seen),
        last_touched_tick=metadata.last_touched_tick,
        q_histogram=_q_histogram(metadata.q_histogram),
        inspection_count=metadata.inspection_count,
    )


def parse_seed_tables(
    edge_table_path: Path | None = None,
    mechanism_table_path: Path | None = None,
) -> tuple[CandidateGraph, MechanismRegistry]:
    graph = parse_seed_candidate_graph(edge_table_path or DEFAULT_EDGE_TABLE_PATH)
    registry = parse_seed_mechanism_registry(
        mechanism_table_path or DEFAULT_MECHANISM_TABLE_PATH,
        graph,
    )
    return graph, registry


def parse_seed_candidate_graph(path: Path) -> CandidateGraph:
    raw = json.loads(path.read_text())
    node_table: dict[str, Node] = {}
    edge_table: dict[str, Edge] = {}
    edge_metadata_table: dict[str, EdgeMetadata] = {}

    for v_name, by_x in raw.get("x_to_v", {}).items():
        node_table.setdefault(v_name, Node(v_name, "V"))
        for x_name, prior in by_x.items():
            node_table.setdefault(x_name, Node(x_name, "X"))
            edge_id = f"{x_name}->{v_name}"
            edge_table[edge_id] = Edge(edge_id, x_name, v_name, "X", "V")
            edge_metadata_table[edge_id] = _seed_edge_metadata(edge_id, prior)

    for y_name, by_v in raw.get("v_to_y", {}).items():
        node_table.setdefault(y_name, Node(y_name, "Y"))
        for v_name, prior in by_v.items():
            node_table.setdefault(v_name, Node(v_name, "V"))
            edge_id = f"{v_name}->{y_name}"
            edge_table[edge_id] = Edge(edge_id, v_name, y_name, "V", "Y")
            edge_metadata_table[edge_id] = _seed_edge_metadata(edge_id, prior)

    return CandidateGraph(node_table, edge_table, edge_metadata_table)


def parse_seed_mechanism_registry(path: Path, graph: CandidateGraph) -> MechanismRegistry:
    raw = json.loads(path.read_text())
    bins = raw.get("confidence_bins", {})
    mechanism_bins = raw.get("mechanism_confidence_bins", {})
    id_maker = MechanismRegistry()

    mechanism_table: dict[str, Mechanism] = {}
    metadata_table: dict[str, MechanismMetadata] = {}
    for entry in raw.get("mechanisms", []):
        mechanism = _seed_mechanism(entry, graph)
        mechanism.mechanism_id = id_maker.make_mechanism_id(mechanism)
        if mechanism.mechanism_id in mechanism_table:
            raise ValueError(f"Duplicate seeded mechanism id: {mechanism.mechanism_id}")

        alpha, beta = _mechanism_prior(entry.get("name"), bins, mechanism_bins)
        mechanism_table[mechanism.mechanism_id] = mechanism
        metadata_table[mechanism.mechanism_id] = MechanismMetadata(
            mechanism_id=mechanism.mechanism_id,
            alpha=alpha,
            beta=beta,
        )

    return MechanismRegistry(mechanism_table, metadata_table)


def _seed_edge_metadata(edge_id: str, prior: dict[str, Any]) -> EdgeMetadata:
    return EdgeMetadata(
        edge_id=edge_id,
        alpha=float(prior.get("alpha", 1.0)),
        beta=float(prior.get("beta", 1.0)),
    )


def _seed_mechanism(entry: dict[str, Any], graph: CandidateGraph) -> Mechanism:
    name = str(entry.get("name", ""))
    edge_ids = list(entry.get("edge_ids", []))
    missing = [edge_id for edge_id in edge_ids if edge_id not in graph.edge_table]
    if missing:
        raise ValueError(f"Seeded mechanism {name!r} references missing edges: {missing}")

    return Mechanism(
        name=name or None,
        edge_ids=edge_ids,
        scope=dict(entry.get("scope", {})),
        narrative=str(entry.get("narrative", "")),
        status=str(entry.get("status", "active")),
    )


def _mechanism_prior(
    name: str | None,
    bins: dict[str, list[float]],
    mechanism_bins: dict[str, str],
) -> tuple[float, float]:
    alpha, beta = bins.get(mechanism_bins.get(name or "", "EVEN"), [1.0, 1.0])
    return float(alpha), float(beta)


def _q_histogram(raw: dict[str, int] | None) -> dict[str, int]:
    histogram = dict(DEFAULT_Q_HISTOGRAM)
    histogram.update(raw or {})
    return histogram
