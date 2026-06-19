"""Candidate causal graph over X, V, and Y variables.

The graph is the shared resolver for mechanism bundles, CUSUM, EIG, and
surrogate prediction. Edges are keyed by stable edge_id strings; every edge
must have an EdgeMetadata row so ConfidenceService can update Beta state
without a late KeyError during S3.
"""

from collections.abc import Sequence

from src.core.models import Edge, EdgeMetadata, Node


class CandidateGraph:
    """Indexed DAG-like view of candidate X->V and V->Y edges."""

    def __init__(
        self,
        node_table: dict[str, Node],
        edge_table: dict[str, Edge],
        edge_metadata_table: dict[str, EdgeMetadata] | None = None,
    ):
        """Initialize graph tables and lookup indexes.

        Args:
            node_table: Node id -> Node.
            edge_table: Edge id -> Edge.
            edge_metadata_table: Edge id -> EdgeMetadata. Must cover every
                edge exactly; confidence updates assume this invariant.

        Raises:
            ValueError: If edge metadata is missing, extra, or keyed under
                an id that disagrees with EdgeMetadata.edge_id.
        """
        self.node_table = node_table
        self.edge_table = edge_table
        self.edge_metadata_table = edge_metadata_table or {}
        self._validate_edge_metadata_table()

        self.edges_by_src: dict[str, set[str]] = {}
        self.edges_by_dst: dict[str, set[str]] = {}
        self.edge_by_pair: dict[tuple[str, str], str] = {}
        self.build_indexes()

    def _validate_edge_metadata_table(self) -> None:
        """Enforce one metadata record per edge before services use graph."""
        missing = sorted(set(self.edge_table) - set(self.edge_metadata_table))
        extra = sorted(set(self.edge_metadata_table) - set(self.edge_table))
        if missing:
            raise ValueError(f"Missing EdgeMetadata for edges: {missing}")
        if extra:
            raise ValueError(f"EdgeMetadata references unknown edges: {extra}")

        mismatched = sorted(
            edge_id
            for edge_id, metadata in self.edge_metadata_table.items()
            if metadata.edge_id != edge_id
        )
        if mismatched:
            raise ValueError(f"EdgeMetadata edge_id mismatch for edges: {mismatched}")

    def build_indexes(self) -> None:
        """Build source, destination, and pair indexes for edge lookup."""
        for edge_id, edge in self.edge_table.items():
            src = edge.src
            dst = edge.dst
            pair = (src, dst)

            if pair in self.edge_by_pair:
                raise ValueError(f"Duplicate edge: {src}->{dst}")

            self.edges_by_src.setdefault(src, set()).add(edge_id)
            self.edges_by_dst.setdefault(dst, set()).add(edge_id)
            self.edge_by_pair[pair] = edge_id

    def get_all_edges(self) -> list[Edge]:
        """Return all Edge objects in table order."""
        return list(self.edge_table.values())

    def get_edges_from(self, node_id: str) -> list[Edge]:
        """Return outgoing edges from one node id."""
        edge_ids = self.edges_by_src.get(node_id, set())
        return [self.edge_table[edge_id] for edge_id in edge_ids]

    def get_edges_to(self, node_id: str) -> list[Edge]:
        """Return incoming edges to one node id."""
        edge_ids = self.edges_by_dst.get(node_id, set())
        return [self.edge_table[edge_id] for edge_id in edge_ids]

    @property
    def x(self) -> list[str]:
        """Node ids classified as X decision/workload variables."""
        return self._nodes_by_type("X")

    @property
    def v(self) -> list[str]:
        """Node ids classified as V mediator variables."""
        return self._nodes_by_type("V")

    @property
    def y(self) -> list[str]:
        """Node ids classified as Y outcome variables."""
        return self._nodes_by_type("Y")

    def _nodes_by_type(self, node_type: str) -> list[str]:
        """Return sorted node ids for one node type label."""
        return sorted(
            node_id for node_id, node in self.node_table.items() if node.node_type == node_type
        )

    def val_edges(self, edge: Edge) -> bool:
        """Return True iff an edge references known nodes with matching types."""
        if edge.src not in self.node_table:
            return False

        if edge.dst not in self.node_table:
            return False

        if edge.src_type != self.get_node_type(edge.src):
            return False

        return edge.dst_type == self.get_node_type(edge.dst)

    def val_topology(self, edges: Sequence[Edge]) -> bool:
        """Return True iff edges are unique and only X->V or V->Y."""
        seen_pairs = set()

        for edge in edges:
            src_type = self.get_node_type(edge.src)
            dst_type = self.get_node_type(edge.dst)
            pair = (edge.src, edge.dst)

            if pair in seen_pairs:
                return False

            is_x_to_v = src_type == "X" and dst_type == "V"
            is_v_to_y = src_type == "V" and dst_type == "Y"

            if not (is_x_to_v or is_v_to_y):
                return False

            seen_pairs.add(pair)

        return True

    def check_connected(self, edges: Sequence[Edge]) -> bool:
        """Validate node references and allowed topology for an edge bundle."""
        for edge in edges:
            if not self.val_edges(edge):
                return False

        return self.val_topology(edges)

    def get_node_type(self, node_id: str) -> str:
        """Return a node's X/V/Y type, raising on unknown node ids."""
        if node_id not in self.node_table:
            raise ValueError(f"Unknown node: {node_id}")

        return self.node_table[node_id].node_type
