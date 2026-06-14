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


# if __name__ == "__main__":
#     x_nodes = [
#         "model_params_b",
#         "model_size_gb",
#         "num_hidden_layers",
#         "hidden_size",
#         "num_attn_heads",
#         "num_kv_heads",
#         "attn_heads_per_kv_head",
#         "intermediate_size",
#         "max_pos_embeddings",
#         "vocab_size",
#         "is_moe",
#         "num_routed_experts",
#         "num_active_experts",
#         "gpu_bandwidth_gbps",
#         "gpu_tflops_fp16",
#         "gpu_mem_gb",
#         "cuda_compute_capability",
#         "gpu_generation",
#         "gpu_per_node",
#         "nvlink_bandwidth_gbps",
#         "internode_bandwidth_gbps",
#         "pcie_bandwidth_gbps",
#         "bandwidth_per_param",
#         "flops_per_param",
#         "gpu_watts",
#         "isl_token_avg",
#         "isl_token_min",
#         "isl_token_max",
#         "isl_distribution_type",
#         "osl_token_avg",
#         "osl_token_min",
#         "osl_token_max",
#         "osl_distribution_type",
#         "pd_ratio",
#         "request_arrival_rate",
#         "request_arrival_pattern",
#         "peak_to_mean_ratio",
#         "workload_prefix_concentration",
#         "multi_turn_ratio",
#         "shared_prefix_length_avg",
#         "is_session_affinity",
#         "total_token_budget",
#         "deadline_hrs",
#         "target_p99_ttft_ms",
#         "target_p99_tpot_ms",
#         "priority_class",
#         "cloud",
#         "region",
#         "market",
#         "gpu_type",
#         "instance_type",
#         "num_nodes_per_chain",
#         "interconnect_type",
#         "tp",
#         "pp",
#         "sp",
#         "dp",
#         "ep",
#         "cp",
#         "engine_name",
#         "engine_version",
#         "attn_backend",
#         "runtime_image",
#         "max_num_seq",
#         "max_num_batched_tokens",
#         "gpu_mem_util",
#         "max_model_len",
#         "swap_space_gb",
#         "block_size",
#         "kvcache_dtype",
#         "kvcache_quantization",
#         "weight_dtype",
#         "weight_quantization_method",
#         "weight_quantization_bits",
#         "activation_quantization_method",
#         "activation_dtype",
#         "prefix_cache_enabled",
#         "chunked_prefill_enable",
#         "chunk_size",
#         "sliding_window_size",
#         "lmcache_enabled",
#         "sparse_attn_pattern",
#         "spec_decoding_enabled",
#         "draft_model_id",
#         "spec_decoding_method",
#         "num_speculative_tokens",
#         "spec_acceptance_threshold",
#         "pd_enabled",
#         "prefill_worker_count",
#         "decode_worker_count",
#         "kv_transfer_method",
#         "cuda_graph_enabled",
#         "torch_compile_enabled",
#         "compile_mode",
#         "num_jit_warmup_steps",
#         "scheduling_policy",
#         "preemption_policy",
#         "max_chunked_steps_per_request",
#         "router_policy",
#         "expert_offload_enabled",
#         "gpu_shared_fraction",
#         "max_concurrent_streaming",
#         "min_chain_warmup_time",
#     ]

#     v_nodes = [
#         "gpu_mem_used_fraction",
#         "kv_cache_util",
#         "activation_mem_pressure",
#         "vram_headroom_gb",
#         "live_batch_size",
#         "depth_req_q",
#         "input_length_observed",
#         "output_length_observed",
#         "sm_utilization",
#         "mem_bandwidth_utilization",
#         "nvlink_tput_observed",
#         "pcie_tput_observed",
#         "kvcache_hit_rate",
#         "prefill_iteration_counts_per_second",
#         "decode_itr_counts_per_second",
#         "pd_inbalance",
#         "expert_inbalance",
#         "comm_overhead_pct",
#         "pipeline_bubble_fraction",
#         "per_tok_comm_bytes",
#         "kv_pressure_score",
#         "dispatch_overhead_ms",
#     ]

#     y_nodes = [
#         "cost_per_token",
#         "p99_ttft_ms",
#         "p99_tpot_ms",
#         "throughput_token_per_sec",
#         "slo_margin",
#     ]

#     node_table = {}
#     for node_id in x_nodes:
#         node_table[node_id] = Node(node_id=node_id, node_type="X")
#     for node_id in v_nodes:
#         node_table[node_id] = Node(node_id=node_id, node_type="V")
#     for node_id in y_nodes:
#         node_table[node_id] = Node(node_id=node_id, node_type="Y")

#     edge_table = {}
#     edge_metadata_table = {}

#     for src in x_nodes:
#         for dst in v_nodes:
#             edge_id = f"{src}->{dst}"
#             edge_table[edge_id] = Edge(
#                 edge_id=edge_id,
#                 src=src,
#                 dst=dst,
#                 src_type="X",
#                 dst_type="V",
#             )
#             edge_metadata_table[edge_id] = EdgeMetadata(
#                 edge_id=edge_id,
#                 alpha=1.0,
#                 beta=1.0,
#             )

#     for src in v_nodes:
#         for dst in y_nodes:
#             edge_id = f"{src}->{dst}"
#             edge_table[edge_id] = Edge(
#                 edge_id=edge_id,
#                 src=src,
#                 dst=dst,
#                 src_type="V",
#                 dst_type="Y",
#             )
#             edge_metadata_table[edge_id] = EdgeMetadata(
#                 edge_id=edge_id,
#                 alpha=1.0,
#                 beta=1.0,
#             )

#     graph = CandidateGraph(
#         node_table=node_table,
#         edge_table=edge_table,
#         edge_metadata_table=edge_metadata_table,
#     )

#     all_edges = graph.get_all_edges()

#     print("node_count:", len(graph.node_table))
#     print("x_count:", len(x_nodes))
#     print("v_count:", len(v_nodes))
#     print("y_count:", len(y_nodes))
#     print("edge_count:", len(all_edges))
#     print("edge_metadata_count:", len(graph.edge_metadata_table))
#     print("topology_valid:", graph.val_topology(all_edges))
#     print("connected_valid:", graph.check_connected(all_edges))
#     print("node_type(tp):", graph.get_node_type("tp"))
#     print("edges_from(tp):", len(graph.get_edges_from("tp")))
#     print("edges_to(p99_ttft_ms):", len(graph.get_edges_to("p99_ttft_ms")))
#     metadata = graph.edge_metadata_table["tp->gpu_mem_used_fraction"]
#     print("confidence(tp->gpu_mem_used_fraction):", metadata.alpha / (metadata.alpha + metadata.beta))
#     import time

#     # Measure time to search edges from "tp"
#     start_time = time.time()
#     edges_from_tp = graph.get_edges_from("tp")
#     duration_from = time.time() - start_time
#     print(f"Time to search edges FROM 'tp': {duration_from:.6f} seconds, found {len(edges_from_tp)} edges")

#     # Measure time to search edges to "p99_ttft_ms"
#     start_time = time.time()
#     edges_to_p99 = graph.get_edges_to("p99_ttft_ms")
#     duration_to = time.time() - start_time
#     print(f"Time to search edges TO 'p99_ttft_ms': {duration_to:.6f} seconds, found {len(edges_to_p99)} edges")

#     import sys

#     def get_size_in_bytes(obj):
#         seen = set()
#         def inner(o):
#             if id(o) in seen:
#                 return 0
#             seen.add(id(o))
#             size = sys.getsizeof(o)
#             if isinstance(o, dict):
#                 size += sum(inner(k) + inner(v) for k, v in o.items())
#             elif isinstance(o, (list, set, tuple)):
#                 size += sum(inner(i) for i in o)
#             return size
#         return inner(obj)

#     def bytes_to_kb(num_bytes):
#         return num_bytes / 1024

#     print("Working memory size in KB:")
#     print("node_table:", f"{bytes_to_kb(get_size_in_bytes(graph.node_table)):.2f} KB")
#     print("edge_table:", f"{bytes_to_kb(get_size_in_bytes(graph.edge_table)):.2f} KB")
#     print("edge_metadata_table:", f"{bytes_to_kb(get_size_in_bytes(graph.edge_metadata_table)):.2f} KB")
#     print("x_nodes:", f"{bytes_to_kb(get_size_in_bytes(x_nodes)):.2f} KB")
#     print("v_nodes:", f"{bytes_to_kb(get_size_in_bytes(v_nodes)):.2f} KB")
#     print("y_nodes:", f"{bytes_to_kb(get_size_in_bytes(y_nodes)):.2f} KB")
#     print("all_edges:", f"{bytes_to_kb(get_size_in_bytes(all_edges)):.2f} KB")
