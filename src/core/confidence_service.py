from src.config.hyperparameters import EDGE_BETA_UPDATE, MECHANISM_BETA_UPDATE
from src.core.models import EdgeConfidenceRecord, MechanismConfidenceRecord


def _key(value):
    return value.value if hasattr(value, "value") else str(value)


class ConfidenceService:
    def __init__(self, candidate_graph, mechanism_registry):
        self.candidate_graph = candidate_graph
        self.mechanism_registry = mechanism_registry

    def get_edge_confidence(self, edge_id):
        alpha = self.candidate_graph.edge_metadata_table[edge_id].alpha
        beta = self.candidate_graph.edge_metadata_table[edge_id].beta
        return alpha / (alpha + beta)

    def get_edge_visit_count(self, edge_id):
        return self.candidate_graph.edge_metadata_table[edge_id].visit_count

    def get_edge_environment_seen(self, edge_id):
        return self.candidate_graph.edge_metadata_table[edge_id].envs_seen

    def get_edge_last_touched(self, edge_id):
        return self.candidate_graph.edge_metadata_table[edge_id].last_touched_tick

    def get_edge_q_histogram(self, edge_id):
        return self.candidate_graph.edge_metadata_table[edge_id].q_histogram

    def get_all_edge_records(self):
        records = []
        for edge_id, edge in self.candidate_graph.edge_table.items():
            edge_metadata = self.candidate_graph.edge_metadata_table[edge_id]
            records.append(EdgeConfidenceRecord(edge=edge, metadata=edge_metadata))
        return records

    def get_mechanism_confidence(self, mechanism_id):
        alpha = self.mechanism_registry.mechanism_metadata_table[mechanism_id].alpha
        beta = self.mechanism_registry.mechanism_metadata_table[mechanism_id].beta
        return alpha / (alpha + beta)

    def get_mechanism_visit_count(self, mechanism_id):
        return self.mechanism_registry.mechanism_metadata_table[mechanism_id].visit_count

    def get_mechanism_environment_seen(self, mechanism_id):
        return self.mechanism_registry.mechanism_metadata_table[mechanism_id].envs_seen

    def get_mechanism_last_touched(self, mechanism_id):
        return self.mechanism_registry.mechanism_metadata_table[mechanism_id].last_touched_tick

    def get_mechanism_q_histogram(self, mechanism_id):
        return self.mechanism_registry.mechanism_metadata_table[mechanism_id].q_histogram

    def get_mechanism_record(self, mechanism_ids):
        records = []
        for mechanism_id in mechanism_ids:
            mechanism = self.mechanism_registry.mechanism_table[mechanism_id]
            mechanism_metadata = self.mechanism_registry.mechanism_metadata_table[mechanism_id]
            records.append(
                MechanismConfidenceRecord(
                    mechanism=mechanism,
                    metadata=mechanism_metadata,
                )
            )
        return records

    def apply_delta_c_edge(self, edge_id, q_label, icp_result):
        edge_metadata = self.candidate_graph.edge_metadata_table[edge_id]
        delta_alpha, delta_beta = self.get_delta_c_edge(q_label, icp_result)

        edge_metadata.alpha += delta_alpha
        edge_metadata.beta += delta_beta
        edge_metadata.visit_count += 1

        q_key = _key(q_label)
        edge_metadata.q_histogram[q_key] = edge_metadata.q_histogram.get(q_key, 0) + 1
        return self.get_edge_confidence(edge_id), True

    def apply_delta_c_mechanism(self, mechanism_id, q_label):
        mechanism_metadata = self.mechanism_registry.mechanism_metadata_table[mechanism_id]
        delta_alpha, delta_beta = self.get_delta_c_mechanism(q_label)

        mechanism_metadata.alpha += delta_alpha
        mechanism_metadata.beta += delta_beta
        mechanism_metadata.visit_count += 1

        q_key = _key(q_label)
        mechanism_metadata.q_histogram[q_key] = mechanism_metadata.q_histogram.get(q_key, 0) + 1
        return self.get_mechanism_confidence(mechanism_id), True

    def seed_new_mechanism_confidence(self, edges):
        pass  # TODO - an LLM should do it for now.

    def get_delta_c_edge(self, q_label, icp_result):
        return EDGE_BETA_UPDATE[_key(icp_result)][_key(q_label)]

    def get_delta_c_mechanism(self, q_label):
        return MECHANISM_BETA_UPDATE[_key(q_label)]


# if __name__ == "__main__":
#     from src.core.candidate_graph import CandidateGraph
#     from src.core.mechanism_registry import MechanismRegistry
#     from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node
#     from src.validation.icp import ICPResult
#     from src.validation.quadrants import Quadrant

#     edge_id = "shared_prefix_length_avg->kvcache_hit_rate"
#     mechanism_id = "M_demo"

#     node_table = {
#         "shared_prefix_length_avg": Node("shared_prefix_length_avg", "X"),
#         "kvcache_hit_rate": Node("kvcache_hit_rate", "V"),
#     }
#     edge_table = {
#         edge_id: Edge(
#             edge_id=edge_id,
#             src="shared_prefix_length_avg",
#             dst="kvcache_hit_rate",
#             src_type="X",
#             dst_type="V",
#         )
#     }
#     edge_metadata_table = {
#         edge_id: EdgeMetadata(
#             edge_id=edge_id,
#             alpha=1.4,
#             beta=0.6,
#             envs_seen={"h200_sxm"},
#         )
#     }
#     graph = CandidateGraph(
#         node_table=node_table,
#         edge_table=edge_table,
#         edge_metadata_table=edge_metadata_table,
#     )

#     mechanism = Mechanism(
#         mechanism_id=mechanism_id,
#         edge_ids=[edge_id],
#         scope={"x": ["shared_prefix_length_avg"], "v": ["kvcache_hit_rate"]},
#         narrative="Shared prefixes should improve KV cache hit rate.",
#     )
#     registry = MechanismRegistry(
#         mechanism_table={mechanism_id: mechanism},
#         mechanism_metadata_table={
#             mechanism_id: MechanismMetadata(
#                 mechanism_id=mechanism_id,
#                 alpha=1.0,
#                 beta=1.0,
#                 envs_seen={"h200_sxm"},
#             )
#         },
#     )
#     service = ConfidenceService(graph, registry)

#     print("initial_edge_confidence:", service.get_edge_confidence(edge_id))
#     print("initial_mechanism_confidence:", service.get_mechanism_confidence(mechanism_id))
#     print("edge_environment_seen:", service.get_edge_environment_seen(edge_id))
#     print("edge_records:", service.get_all_edge_records())
#     print("mechanism_records:", service.get_mechanism_record([mechanism_id]))
#     print("edge_delta(Q1, accept):", service.get_delta_c_edge(Quadrant.Q1, ICPResult.ACCEPT))
#     print("mechanism_delta(Q4):", service.get_delta_c_mechanism(Quadrant.Q4))

#     print(
#         "apply_delta_c_edge(Q1, accept):",
#         service.apply_delta_c_edge(edge_id, Quadrant.Q1, ICPResult.ACCEPT),
#     )
#     print(
#         "edge_alpha_beta:",
#         graph.edge_metadata_table[edge_id].alpha,
#         graph.edge_metadata_table[edge_id].beta,
#     )
#     print("edge_visit_count:", service.get_edge_visit_count(edge_id))
#     print("edge_q_histogram:", service.get_edge_q_histogram(edge_id))

#     print(
#         "apply_delta_c_mechanism(Q4):",
#         service.apply_delta_c_mechanism(mechanism_id, Quadrant.Q4),
#     )
#     print(
#         "mechanism_alpha_beta:",
#         registry.mechanism_metadata_table[mechanism_id].alpha,
#         registry.mechanism_metadata_table[mechanism_id].beta,
#     )
#     print("mechanism_visit_count:", service.get_mechanism_visit_count(mechanism_id))
#     print("mechanism_q_histogram:", service.get_mechanism_q_histogram(mechanism_id))
