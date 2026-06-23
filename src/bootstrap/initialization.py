from __future__ import annotations

from pathlib import Path

from src.core.candidate_graph import CandidateGraph
from src.core.causal_graph_store_adapter import (
    StoreBackedConfidenceService,
    StoreBackedMechanismRegistry,
    koi_edge_metadata_to_store,
    koi_edge_to_store,
    koi_mechanism_metadata_to_store,
    koi_mechanism_to_store,
    koi_node_to_store,
    parse_seed_tables,
    store_edge_metadata_to_koi,
    store_edge_to_koi,
    store_mechanism_metadata_to_koi,
    store_mechanism_to_koi,
    store_node_to_koi,
)
from src.core.models import Mechanism, MechanismMetadata
from tandemn_system_data.clients import (  # type: ignore[import-untyped]
    CausalGraphStore,
    PostgresClient,
)
from tandemn_system_data.clients.causal_graph_store import (  # type: ignore[import-untyped]
    edge_to_row,
    node_to_row,
)


def init_causal_graph(
    user_id: str,
    postgres_client=None,
    edge_table_path: str | Path | None = None,
    mechanism_table_path: str | Path | None = None,
) -> tuple[CandidateGraph, StoreBackedMechanismRegistry, StoreBackedConfidenceService]:
    """Load Koi causal graph state from Tandemn Store, importing seeds if empty."""
    client = postgres_client or PostgresClient()
    store = CausalGraphStore(client, user_id=user_id)

    nodes = store.load_nodes()
    edges, _ = store.load_edges()
    mechanisms, _ = store.load_mechanisms()
    if not nodes and not edges and not mechanisms:
        _import_seed_causal_graph(
            client,
            store,
            user_id,
            _optional_path(edge_table_path),
            _optional_path(mechanism_table_path),
        )
    elif not (nodes and edges and mechanisms):
        raise ValueError(
            "Incomplete Tandemn Store causal graph for "
            f"user {user_id!r}: nodes={len(nodes)}, edges={len(edges)}, "
            f"mechanisms={len(mechanisms)}"
        )

    return _load_causal_graph_runtime(store)


def _import_seed_causal_graph(
    client,
    store,
    user_id: str,
    edge_table_path: Path | None,
    mechanism_table_path: Path | None,
) -> None:
    graph, registry = parse_seed_tables(edge_table_path, mechanism_table_path)
    with client.begin() as session:
        for node in graph.node_table.values():
            session.add(node_to_row(user_id, koi_node_to_store(node)))
        for edge_id, edge in graph.edge_table.items():
            metadata = graph.edge_metadata_table[edge_id]
            session.add(
                edge_to_row(
                    user_id,
                    koi_edge_to_store(edge),
                    koi_edge_metadata_to_store(metadata),
                )
            )

    store.sync_mechanisms(
        {
            mechanism_id: koi_mechanism_to_store(mechanism)
            for mechanism_id, mechanism in registry.mechanism_table.items()
        },
        {
            mechanism_id: koi_mechanism_metadata_to_store(metadata)
            for mechanism_id, metadata in registry.mechanism_metadata_table.items()
        },
    )


def _load_causal_graph_runtime(
    store,
) -> tuple[CandidateGraph, StoreBackedMechanismRegistry, StoreBackedConfidenceService]:
    nodes = {node_id: store_node_to_koi(node) for node_id, node in store.load_nodes().items()}
    store_edges, store_edge_metadata = store.load_edges()
    store_mechanisms, store_mechanism_metadata = store.load_mechanisms()

    edge_table = {edge_id: store_edge_to_koi(edge) for edge_id, edge in store_edges.items()}
    edge_metadata_table = {
        edge_id: store_edge_metadata_to_koi(metadata)
        for edge_id, metadata in store_edge_metadata.items()
    }
    graph = CandidateGraph(nodes, edge_table, edge_metadata_table)
    _validate_graph_edges(graph)

    mechanism_table = {
        mechanism_id: store_mechanism_to_koi(mechanism)
        for mechanism_id, mechanism in store_mechanisms.items()
    }
    mechanism_metadata_table = {
        mechanism_id: store_mechanism_metadata_to_koi(metadata)
        for mechanism_id, metadata in store_mechanism_metadata.items()
    }
    _validate_mechanism_tables(graph, mechanism_table, mechanism_metadata_table)

    registry = StoreBackedMechanismRegistry(
        store,
        mechanism_table,
        mechanism_metadata_table,
    )
    return graph, registry, StoreBackedConfidenceService(graph, registry, store)


def _validate_graph_edges(graph: CandidateGraph) -> None:
    invalid = sorted(
        edge.edge_id for edge in graph.edge_table.values() if not graph.val_edges(edge)
    )
    if invalid:
        raise ValueError(f"Invalid Tandemn Store causal edges: {invalid}")


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


def init_seed_mechanisms_priors(LLM, CandidateGraph, NodeDescription):
    # Placeholder: seed Mechanisms
    pass


def init_ranges(config):
    # Placeholder: returns a dictionary Objective -> Range
    pass
