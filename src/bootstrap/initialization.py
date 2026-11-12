from __future__ import annotations

import os
from pathlib import Path

from src.core.candidate_graph import CandidateGraph
from src.core.causal_graph_store_adapter import (
    PostgresConfidenceService,
    PostgresMechanismRegistry,
    _q_histogram,
    parse_seed_tables,
    save_edge_metadata,
    save_mechanism,
)
from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node


def init_causal_graph(
    pool,
    edge_table_path: str | Path | None = None,
    mechanism_table_path: str | Path | None = None,
) -> tuple[CandidateGraph, PostgresMechanismRegistry, PostgresConfidenceService]:
    """Load Koi's causal graph, atomically importing seeds if all tables are empty."""
    with pool.connection() as conn, conn.transaction():
        counts = conn.execute(
            """
            SELECT
                (SELECT count(*) FROM causal_nodes) AS nodes,
                (SELECT count(*) FROM causal_edges) AS edges,
                (SELECT count(*) FROM causal_mechanisms) AS mechanisms
            """
        ).fetchone()
        if not any(counts.values()):
            _import_seed_causal_graph(
                conn,
                _optional_path(edge_table_path),
                _optional_path(mechanism_table_path),
            )
        elif not all(counts.values()):
            raise ValueError(
                f"Incomplete causal graph: nodes={counts['nodes']}, "
                f"edges={counts['edges']}, mechanisms={counts['mechanisms']}"
            )

        return _load_causal_graph_runtime(conn, pool)


def _import_seed_causal_graph(
    conn,
    edge_table_path: Path | None,
    mechanism_table_path: Path | None,
) -> None:
    graph, registry = parse_seed_tables(edge_table_path, mechanism_table_path)
    _validate_graph_edges(graph)
    _validate_mechanism_tables(graph, registry.mechanism_table, registry.mechanism_metadata_table)
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO causal_nodes (node_id, node_type, description, unit) "
            "VALUES (%s, %s, %s, %s)",
            [
                (node.node_id, node.node_type, node.description, node.unit)
                for node in graph.node_table.values()
            ],
        )
        cursor.executemany(
            "INSERT INTO causal_edges (edge_id, src, dst, src_type, dst_type, status) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [
                (edge.edge_id, edge.src, edge.dst, edge.src_type, edge.dst_type, edge.status)
                for edge in graph.edge_table.values()
            ],
        )
    save_edge_metadata(conn, graph.edge_metadata_table)
    for mechanism_id, mechanism in registry.mechanism_table.items():
        save_mechanism(conn, mechanism, registry.mechanism_metadata_table[mechanism_id])


def _load_causal_graph_runtime(
    conn, pool
) -> tuple[CandidateGraph, PostgresMechanismRegistry, PostgresConfidenceService]:
    nodes = {
        row["node_id"]: Node(**row)
        for row in conn.execute("SELECT * FROM causal_nodes ORDER BY node_id")
    }
    edge_table = {}
    edge_metadata_table = {}
    for row in conn.execute("SELECT * FROM causal_edges ORDER BY edge_id"):
        edge_id = row["edge_id"]
        edge_table[edge_id] = Edge(
            edge_id=edge_id,
            src=row.pop("src"),
            dst=row.pop("dst"),
            src_type=row.pop("src_type"),
            dst_type=row.pop("dst_type"),
            status=row.pop("status"),
        )
        row["envs_seen"] = {tuple(env) for env in row["envs_seen"] or []}
        row["q_histogram"] = _q_histogram(row["q_histogram"])
        edge_metadata_table[edge_id] = EdgeMetadata(**row)
    graph = CandidateGraph(nodes, edge_table, edge_metadata_table)
    _validate_graph_edges(graph)

    mechanism_table = {}
    mechanism_metadata_table = {}
    for row in conn.execute("SELECT * FROM causal_mechanisms ORDER BY mechanism_id"):
        mechanism_id = row["mechanism_id"]
        mechanism_table[mechanism_id] = Mechanism(
            mechanism_id=mechanism_id,
            name=row.pop("name") or None,
            edge_ids=row.pop("edge_ids") or [],
            scope=row.pop("scope") or {},
            narrative=row.pop("narrative"),
            status=row.pop("status"),
        )
        row["envs_seen"] = {tuple(env) for env in row.pop("envs_last_seen") or []}
        row["q_histogram"] = _q_histogram(row["q_histogram"])
        mechanism_metadata_table[mechanism_id] = MechanismMetadata(**row)
    _validate_mechanism_tables(graph, mechanism_table, mechanism_metadata_table)

    registry = PostgresMechanismRegistry(
        pool,
        mechanism_table,
        mechanism_metadata_table,
    )
    return graph, registry, PostgresConfidenceService(graph, registry, pool)


def _validate_graph_edges(graph: CandidateGraph) -> None:
    invalid = sorted(
        edge.edge_id for edge in graph.edge_table.values() if not graph.val_edges(edge)
    )
    if invalid:
        raise ValueError(f"Invalid causal edges: {invalid}")


def _validate_mechanism_tables(
    graph: CandidateGraph,
    mechanism_table: dict[str, Mechanism],
    metadata_table: dict[str, MechanismMetadata],
) -> None:
    missing_metadata = sorted(set(mechanism_table) - set(metadata_table))
    if missing_metadata:
        raise ValueError(f"Missing MechanismMetadata for mechanisms: {missing_metadata}")

    missing_edges = {
        mechanism_id: [edge_id for edge_id in mechanism.edge_ids if edge_id not in graph.edge_table]
        for mechanism_id, mechanism in mechanism_table.items()
    }
    missing_edges = {mid: edges for mid, edges in missing_edges.items() if edges}
    if missing_edges:
        raise ValueError(f"Mechanisms reference unknown edges: {missing_edges}")


def _optional_path(path: str | Path | None) -> Path | None:
    return Path(path) if path is not None else None


def init_candidate_graph(XVY_CSV):
    # Placeholder: create and return a CandidateGraph from the definition CSV
    pass


def init_edge_priors(LLM, CandidateGraph, EdgeDescription):
    # Placeholder: seed edges with priors
    pass


def init_slow_state(config):
    # Placeholder: sets the Betas and all slow-state hyperparameters
    pass


def init_resource_map(user_id: str, postgres_client=None):
    from src.infra.resource_map import ResourceMapManager

    manager = ResourceMapManager(user_id=user_id, postgres_client=postgres_client)
    manager.get_resource_map()
    return manager


def init_evidence_store(user_id: str, postgres_client=None):
    from src.core.evidence_service import EvidenceService

    return EvidenceService(
        user_id=user_id,
        postgres_client=postgres_client,
    )


def init_surrogate_stack(
    evidence_store=None,
    peer_mode: str = "shadow",
    *,
    perfdb_path: str | Path | None = None,
    perfdb_mode: str = "enabled",
    lower_quantile: float | None = None,
):
    """Build the Koi-native composer around the authoritative direct AIC path."""
    from src.prediction.backends.perfdb import PerfDBBackend
    from src.prediction.calibration import DEFAULT_LOWER_QUANTILE
    from src.prediction.composer import SurrogateComposer

    resolved_perfdb_path = perfdb_path or os.environ.get("KOI_PERFDB_PATH")
    resolved_lower_quantile = (
        float(os.environ.get("KOI_SURROGATE_LOWER_QUANTILE", DEFAULT_LOWER_QUANTILE))
        if lower_quantile is None
        else float(lower_quantile)
    )
    perfdb = PerfDBBackend(resolved_perfdb_path) if resolved_perfdb_path is not None else None
    return SurrogateComposer(
        evidence_store=evidence_store,
        peer_mode=peer_mode,
        perfdb_backend=perfdb,
        perfdb_mode=perfdb_mode,
        lower_quantile=resolved_lower_quantile,
    )


def init_seed_mechanisms_priors(LLM, CandidateGraph, NodeDescription):
    # Placeholder: seed Mechanisms
    pass


def init_ranges(config):
    # Placeholder: returns a dictionary Objective -> Range
    pass
