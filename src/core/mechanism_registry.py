"""Mechanism registry and lookup indexes.

Mechanisms are reusable causal stories over CandidateGraph edge bundles. The
registry owns identity, duplicate detection, active/archive state, and fast
edge-to-mechanism lookups used by S2, EIG, and agent tooling.
"""

import hashlib
import json

from src.core.models import Mechanism, MechanismMetadata


class MechanismRegistry:
    """In-memory mechanism catalog plus secondary indexes."""

    def __init__(
        self,
        mechanism_table: dict[str, Mechanism] | None = None,
        mechanism_metadata_table: dict[str, MechanismMetadata] | None = None,
    ):
        """Initialize registry tables and rebuild lookup indexes."""
        self.mechanism_table = mechanism_table or {}
        self.mechanism_metadata_table = mechanism_metadata_table or {}

        self.mechanisms_by_edge: dict[str, set[str]] = {}
        self.mechanisms_by_status: dict[str, set[str]] = {
            "active": set(),
            "archived": set(),
        }
        self.build_indexes()

    def build_indexes(self) -> None:
        """Rebuild edge and status indexes from the mechanism table."""
        self.mechanisms_by_edge = {}
        self.mechanisms_by_status = {"active": set(), "archived": set()}
        for mechanism_id, mechanism in self.mechanism_table.items():
            if mechanism.status not in self.mechanisms_by_status:
                raise ValueError(f"Unknown mechanism status: {mechanism.status}")

            self.mechanisms_by_status[mechanism.status].add(mechanism_id)
            for edge_id in mechanism.edge_ids:
                self.mechanisms_by_edge.setdefault(edge_id, set()).add(mechanism_id)

    def get_mechanism(self, mechanism_id: str) -> Mechanism:
        """Return a mechanism by id, raising KeyError when absent."""
        return self.mechanism_table[mechanism_id]

    def make_mechanism_id(self, mechanism: Mechanism) -> str:
        """Return stable id from edge bundle and scope.

        The scope is part of identity: two mechanisms over the same edges but
        different applicable populations must remain distinct.
        """
        key = {
            "edge_ids": sorted(mechanism.edge_ids),
            "scope": mechanism.scope,
        }

        key_text = json.dumps(key, sort_keys=True)
        digest = hashlib.sha1(key_text.encode()).hexdigest()[:8]  # TODO - remove magic number

        return f"M_{digest}"

    def add_mechanism(self, mechanism: Mechanism) -> str:
        """Admit a mechanism unless an identical id already exists.

        Metadata is seeded with neutral Beta(1, 1) when no prior metadata was
        supplied by boot/initialization.
        """
        is_duplicate, existing_mechanism_id = self.is_duplicate_mechanism(mechanism)
        if is_duplicate:
            assert existing_mechanism_id is not None
            return existing_mechanism_id

        mechanism_id = mechanism.mechanism_id
        assert mechanism_id is not None
        self.mechanism_table[mechanism_id] = mechanism

        if mechanism_id not in self.mechanism_metadata_table:
            self.mechanism_metadata_table[mechanism_id] = MechanismMetadata(
                mechanism_id=mechanism_id,
                alpha=1.0,
                beta=1.0,  # TODO - an LLM should assign and seed these
            )
        self.mechanisms_by_status.setdefault(mechanism.status, set()).add(mechanism_id)

        for edge_id in mechanism.edge_ids:
            self.mechanisms_by_edge.setdefault(edge_id, set()).add(mechanism_id)

        return mechanism_id

    def get_usable_mechanism(self, mechanisms: list[Mechanism]) -> list[Mechanism]:
        """Return mechanisms whose status is active."""
        return [mechanism for mechanism in mechanisms if mechanism.status == "active"]

    def get_archived_mechanisms(self, mechanisms: list[Mechanism]) -> list[Mechanism]:
        """Return mechanisms whose status is archived."""
        return [mechanism for mechanism in mechanisms if mechanism.status == "archived"]

    def is_duplicate_mechanism(self, mechanism: Mechanism) -> tuple[bool, str | None]:
        """Return whether a mechanism id already exists, assigning one if needed."""
        if mechanism.mechanism_id is None:
            mechanism.mechanism_id = self.make_mechanism_id(mechanism)

        if mechanism.mechanism_id in self.mechanism_table:
            return True, mechanism.mechanism_id
        return False, None

    def get_mechanism_ids_containing_edge(self, edge_id: str) -> list[str]:
        """Return ids of mechanisms whose bundle includes edge_id."""
        return list(self.mechanisms_by_edge.get(edge_id, set()))

    def get_mechanisms_containing_edge(self, edge_id: str) -> list[Mechanism]:
        """Return mechanisms whose bundle includes edge_id."""
        mechanism_ids = self.get_mechanism_ids_containing_edge(edge_id)
        return [self.mechanism_table[mid] for mid in mechanism_ids]

    def get_edges_from_mechanism(self, mechanism_id: str) -> list[str]:
        """Return edge ids stored on one mechanism."""
        return self.mechanism_table[mechanism_id].edge_ids

    def archive_mechanism(self, mechanism_id: str, reason: str) -> bool:
        """Mark a mechanism archived and keep status indexes consistent."""
        self.mechanism_table[mechanism_id].status = "archived"
        self.mechanism_table[mechanism_id].archived_reason = reason
        self.mechanisms_by_status["archived"].add(mechanism_id)
        self.mechanisms_by_status["active"].discard(mechanism_id)
        return True

    def filter_by_scope(self, subset_x: list[str], subset_v: list[str]) -> list[Mechanism]:
        """Return mechanisms whose scope text matches enough requested terms.

        v0 uses a simple text-overlap heuristic over mechanism.scope. It is
        intentionally broad so S2 can fan out evidence to plausible
        mechanisms before stricter scopeability validation exists.
        """
        matching_mechanisms = []
        for mechanism in self.mechanism_table.values():
            if self.percentage_scope_match(subset_x, subset_v, mechanism) > 25:
                matching_mechanisms.append(mechanism)
        return matching_mechanisms

    def percentage_scope_match(
        self,
        subset_x: list[str],
        subset_v: list[str],
        mechanism: Mechanism,
    ) -> float:
        """Return percent of requested X/V terms found in mechanism.scope text."""
        requested_terms = set(subset_x) | set(subset_v)
        if not requested_terms:
            return 0.0

        scope_text = json.dumps(mechanism.scope, sort_keys=True)
        matched_terms = sum(term in scope_text for term in requested_terms)
        return 100.0 * matched_terms / len(requested_terms)


# if __name__ == "__main__":
#     registry = MechanismRegistry(
#         mechanism_table={},
#         mechanism_metadata_table={},
#     )

#     prefix_mechanism = Mechanism(
#         edge_ids=[
#             "shared_prefix_length_avg->kvcache_hit_rate",
#             "kvcache_hit_rate->p99_ttft_ms",
#         ],
#         scope={
#             "x": ["shared_prefix_length_avg", "workload_prefix_concentration", "gpu_type"],
#             "v": ["kvcache_hit_rate"],
#             "workload_type": "online",
#         },
#         narrative="Shared prefixes should improve KV cache hits and TTFT.",
#     )

#     pd_mechanism = Mechanism(
#         edge_ids=[
#             "pd_enabled->pd_inbalance",
#             "pd_inbalance->p99_tpot_ms",
#         ],
#         scope={
#             "x": ["pd_enabled", "prefill_worker_count", "decode_worker_count"],
#             "v": ["pd_inbalance"],
#             "workload_type": "online",
#         },
#         narrative="Prefill/decode imbalance can affect online TPOT.",
#     )

#     kv_pressure_mechanism = Mechanism(
#         edge_ids=[
#             "gpu_mem_util->kv_cache_util",
#             "kv_cache_util->p99_tpot_ms",
#         ],
#         scope={
#             "x": ["gpu_mem_util", "max_num_batched_tokens"],
#             "v": ["kv_cache_util"],
#             "workload_type": "batched",
#         },
#         narrative="KV pressure can affect batched decode latency.",
#         status="archived",
#     )

#     print("make_mechanism_id(prefix):", registry.make_mechanism_id(prefix_mechanism))

#     prefix_id = registry.add_mechanism(prefix_mechanism)
#     pd_id = registry.add_mechanism(pd_mechanism)
#     kv_pressure_id = registry.add_mechanism(kv_pressure_mechanism)

#     duplicate_id = registry.add_mechanism(
#         Mechanism(
#             edge_ids=list(prefix_mechanism.edge_ids),
#             scope=dict(prefix_mechanism.scope),
#             narrative="Duplicate narrative should not change the mechanism id.",
#         )
#     )

#     subset_x = ["shared_prefix_length_avg", "gpu_type", "pd_enabled"]
#     subset_v = ["kvcache_hit_rate", "pd_inbalance"]

#     print("mechanism_ids:", list(registry.mechanism_table))
#     print("prefix_id:", prefix_id)
#     print("pd_id:", pd_id)
#     print("kv_pressure_id:", kv_pressure_id)
#     print("duplicate_id:", duplicate_id)
#     print("status_index:", registry.mechanisms_by_status)
#     print("edge_index:", registry.mechanisms_by_edge)
#     print("get_mechanism(prefix_id):", registry.get_mechanism(prefix_id))
#     print("get_edges_from_mechanism(prefix_id):", registry.get_edges_from_mechanism(prefix_id))
#     print("is_duplicate_mechanism(prefix):", registry.is_duplicate_mechanism(prefix_mechanism))
#     print(
#         "get_mechanism_ids_containing_edge(shared_prefix...):",
#         registry.get_mechanism_ids_containing_edge("shared_prefix_length_avg->kvcache_hit_rate"),
#     )
#     print(
#         "get_mechanisms_containing_edge(shared_prefix...):",
#         registry.get_mechanisms_containing_edge("shared_prefix_length_avg->kvcache_hit_rate"),
#     )
#     print(
#         "get_usable_mechanism:",
#         [mechanism.mechanism_id for mechanism in registry.get_usable_mechanism(registry.mechanism_table.values())],
#     )
#     print(
#         "get_archived_mechanisms:",
#         [
#             mechanism.mechanism_id
#             for mechanism in registry.get_archived_mechanisms(registry.mechanism_table.values())
#         ],
#     )
#     print(
#         "percentage_scope_match(prefix):",
#         registry.percentage_scope_match(subset_x, subset_v, prefix_mechanism),
#     )
#     print(
#         "percentage_scope_match(pd):",
#         registry.percentage_scope_match(subset_x, subset_v, pd_mechanism),
#     )
#     print(
#         "filter_by_scope:",
#         [mechanism.mechanism_id for mechanism in registry.filter_by_scope(subset_x, subset_v)],
#     )

#     print("archive_mechanism(pd):", registry.archive_mechanism(pd_id, "demo archive"))
#     print("status_index_after_archive:", registry.mechanisms_by_status)
#     print("pd_archived_reason:", registry.get_mechanism(pd_id).archived_reason)
