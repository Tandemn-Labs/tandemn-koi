"""PostgreSQL persistence and seed parsing for Koi's causal graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb
from src.core.candidate_graph import CandidateGraph
from src.core.confidence_service import ConfidenceService
from src.core.mechanism_registry import MechanismRegistry
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node

DEFAULT_EDGE_TABLE_PATH = (
    Path(__file__).resolve().parents[1] / "bootstrap" / "edge_confidence_table.json"
)
DEFAULT_MECHANISM_TABLE_PATH = (
    Path(__file__).resolve().parents[1] / "bootstrap" / "mechanism_seed_table.json"
)
DEFAULT_Q_HISTOGRAM = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}


class PostgresMechanismRegistry(MechanismRegistry):
    """MechanismRegistry that persists accepted mechanism mutations."""

    def __init__(
        self,
        pool,
        mechanism_table: dict[str, Mechanism] | None = None,
        mechanism_metadata_table: dict[str, MechanismMetadata] | None = None,
    ):
        super().__init__(mechanism_table, mechanism_metadata_table)
        self._pool = pool

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
        with self._pool.connection() as conn, conn.transaction():
            save_mechanism(
                conn,
                self.mechanism_table[mechanism_id],
                self.mechanism_metadata_table[mechanism_id],
            )


class PostgresConfidenceService(ConfidenceService):
    """Flush Koi's in-memory confidence state in one transaction."""

    def __init__(self, candidate_graph, mechanism_registry, pool):
        super().__init__(candidate_graph, mechanism_registry)
        self._pool = pool

    def seed_new_mechanism_confidence(
        self,
        mechanism_id: str,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> float:
        confidence = super().seed_new_mechanism_confidence(mechanism_id, alpha, beta)
        self.mechanism_registry.persist_mechanism(mechanism_id)
        return confidence

    def flush(self) -> None:
        with self._pool.connection() as conn, conn.transaction():
            save_edge_metadata(conn, self.candidate_graph.edge_metadata_table)
            for mechanism_id, mechanism in self.mechanism_registry.mechanism_table.items():
                save_mechanism(
                    conn,
                    mechanism,
                    self.mechanism_registry.mechanism_metadata_table[mechanism_id],
                )


def save_edge_metadata(conn, metadata_table: dict[str, EdgeMetadata]) -> None:
    with conn.cursor() as cursor:
        cursor.executemany(
            """
            UPDATE causal_edges SET
                alpha = %s, beta = %s, visit_count = %s, last_touched_tick = %s,
                q_histogram = %s, envs_seen = %s, q3_frequency = %s
            WHERE edge_id = %s
            """,
            [
                (
                    metadata.alpha,
                    metadata.beta,
                    metadata.visit_count,
                    metadata.last_touched_tick,
                    Jsonb(_q_histogram(metadata.q_histogram)),
                    Jsonb(sorted(metadata.envs_seen)),
                    metadata.q3_frequency,
                    metadata.edge_id,
                )
                for metadata in metadata_table.values()
            ],
        )


def save_mechanism(conn, mechanism: Mechanism, metadata: MechanismMetadata) -> None:
    assert mechanism.mechanism_id is not None
    conn.execute(
        """
        INSERT INTO causal_mechanisms (
            mechanism_id, name, edge_ids, scope, narrative, status,
            alpha, beta, visit_count, last_touched_tick, q_histogram,
            envs_last_seen, inspection_count
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (mechanism_id) DO UPDATE SET
            name = EXCLUDED.name, edge_ids = EXCLUDED.edge_ids,
            scope = EXCLUDED.scope, narrative = EXCLUDED.narrative,
            status = EXCLUDED.status, alpha = EXCLUDED.alpha, beta = EXCLUDED.beta,
            visit_count = EXCLUDED.visit_count, last_touched_tick = EXCLUDED.last_touched_tick,
            q_histogram = EXCLUDED.q_histogram, envs_last_seen = EXCLUDED.envs_last_seen,
            inspection_count = EXCLUDED.inspection_count
        """,
        (
            mechanism.mechanism_id,
            mechanism.name,
            mechanism.edge_ids,
            Jsonb(mechanism.scope),
            mechanism.narrative,
            mechanism.status,
            metadata.alpha,
            metadata.beta,
            metadata.visit_count,
            metadata.last_touched_tick,
            Jsonb(_q_histogram(metadata.q_histogram)),
            Jsonb(sorted(metadata.envs_seen)),
            metadata.inspection_count,
        ),
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
