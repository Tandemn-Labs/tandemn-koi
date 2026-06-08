from dataclasses import dataclass, field
  
@dataclass
class Node:
    node_id: str
    node_type: str
    description: str | None = None
    unit: str | None = None


@dataclass
class Edge:
    edge_id: str
    src: str
    dst: str
    src_type: str
    dst_type: str
    status: str = "active"


@dataclass
class EdgeMetadata:
    edge_id: str
    confidence: float
    visit_count: int = 0
    last_touched_tick: int | None = None
    q_histogram: dict[str, int] = field(
        default_factory=lambda: {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    )
    q3_frequency: float = 0.0


class CandidateGraph:
    def __init__(self, node_table, edge_table, edge_metadata_table=None):
        self.node_table = node_table
        self.edge_table = edge_table
        self.edge_metadata_table = edge_metadata_table or {}

        self.edges_by_src = {}
        self.edges_by_dst = {}
        self.edge_by_pair = {}
        self.build_indexes()
        
    def build_indexes(self):
        for edge_id, edge in self.edge_table.items():
            src = edge.src
            dst = edge.dst
            pair = (src, dst)

            if pair in self.edge_by_pair:
                raise ValueError(f"Duplicate edge: {src}->{dst}")

            self.edges_by_src.setdefault(src, set()).add(edge_id) #For this source node, remember this outgoing edge.
            self.edges_by_dst.setdefault(dst, set()).add(edge_id) #For this destination node, remember this incoming edge.
            self.edge_by_pair[pair] = edge_id

    def get_all_edges(self):
        return list(self.edge_table.values())

    def get_edges_from(self, node_id):
        edge_ids = self.edges_by_src.get(node_id, set())
        return [self.edge_table[edge_id] for edge_id in edge_ids]

    def get_edges_to(self, node_id):
        edge_ids = self.edges_by_dst.get(node_id, set())
        return [self.edge_table[edge_id] for edge_id in edge_ids]

    def val_edges(self, edge):
        if edge.src not in self.node_table:
            return False

        if edge.dst not in self.node_table:
            return False

        if edge.src_type != self.get_node_type(edge.src):
            return False

        if edge.dst_type != self.get_node_type(edge.dst):
            return False

        return True


    def val_topology(self, edges):
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


    def check_connected(self, edges):
        for edge in edges:
            if not self.val_edges(edge):
                return False

        return self.val_topology(edges)


    def get_node_type(self, node_id):
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
#                 confidence=0.5,
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
#                 confidence=0.5,
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
#     print("confidence(tp->gpu_mem_used_fraction):", graph.edge_metadata_table["tp->gpu_mem_used_fraction"].confidence)
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
