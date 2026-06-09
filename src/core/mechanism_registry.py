import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class Mechanism:
    edge_ids: list[str]
    scope: dict
    narrative: str
    status: str = "active"
    mechanism_id: str | None = None
    archived_reason: str | None = None


@dataclass
class MechanismMetadata:
    mechanism_id: str
    confidence: float = 0.5
    visit_count: int = 0
    envs_seen: set[str] = field(default_factory=set)
    last_touched_tick: int | None = None
    q_histogram: dict[str, int] = field(
        default_factory=lambda: {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    )
    inspection_count: int = 0


class MechanismRegistry:
    def __init__(self, mechanism_table=None, mechanism_metadata_table=None):
        self.mechanism_table = mechanism_table
        self.mechanism_metadata_table = mechanism_metadata_table

        self.mechanisms_by_edge = {}
        self.mechanisms_by_status = {
            "active": set(),
            "archived": set(),
        }
        self.build_indexes()

    def build_indexes(self):
        # reset the index in case we have to rebuild it again.
        self.mechanisms_by_edge = {}
        self.mechanisms_by_status = {"active": set(), "archived": set()}
        for mechanism_id, mechanism in self.mechanism_table.items():
            if mechanism.status not in self.mechanisms_by_status:
                raise ValueError(f"Unknown mechanism status: {mechanism.status}")

            self.mechanisms_by_status[mechanism.status].add(mechanism_id)
            for edge_id in mechanism.edge_ids:
                self.mechanisms_by_edge.setdefault(edge_id, set()).add(mechanism_id)

    def get_mechanism(self, mechanism_id):
        return self.mechanism_table[mechanism_id]

    def make_mechanism_id(self, mechanism):
        """
        We take care of the cases when the edges are the same and the scope
        is different by hashing them and setting them as the ID
        """
        key = {
            "edge_ids": sorted(mechanism.edge_ids),
            "scope": mechanism.scope,
        }

        key_text = json.dumps(key, sort_keys=True)
        digest = hashlib.sha1(key_text.encode()).hexdigest()[:8]  # TODO - remove magic number

        return f"M_{digest}"

    def add_mechanism(self, mechanism):
        is_duplicate, existing_mechanism_id = self.is_duplicate_mechanism(mechanism)
        if is_duplicate:
            return existing_mechanism_id

        self.mechanism_table[mechanism.mechanism_id] = mechanism

        if mechanism.mechanism_id not in self.mechanism_metadata_table:
            self.mechanism_metadata_table[mechanism.mechanism_id] = MechanismMetadata(
                mechanism_id=mechanism.mechanism_id,
                confidence=0.5,  # TODO - an LLM Call should seed this, and not us
            )
        self.mechanisms_by_status.setdefault(mechanism.status, set()).add(mechanism.mechanism_id)

        for edge_id in mechanism.edge_ids:
            self.mechanisms_by_edge.setdefault(edge_id, set()).add(mechanism.mechanism_id)

        return mechanism.mechanism_id

    def get_usable_mechanism(self, mechanisms):
        return [mechanism for mechanism in mechanisms if mechanism.status == "active"]

    def get_archived_mechanisms(self, mechanisms):
        return [mechanism for mechanism in mechanisms if mechanism.status == "archived"]

    def is_duplicate_mechanism(self, mechanism):
        if mechanism.mechanism_id is None:
            mechanism.mechanism_id = self.make_mechanism_id(mechanism)

        if mechanism.mechanism_id in self.mechanism_table:
            return True, mechanism.mechanism_id
        return False, None

    def get_mechanism_ids_containing_edge(self, edge_id):
        return list(self.mechanisms_by_edge.get(edge_id, set()))

    def get_mechanisms_containing_edge(self, edge_id):
        mechanism_ids = self.get_mechanism_ids_containing_edge(edge_id)
        return [self.mechanism_table[mid] for mid in mechanism_ids]

    def get_edges_from_mechanism(self, mechanism_id):
        # TODO - clarify if this just returns the edge ids or the objects themselves.
        return self.mechanism_table[mechanism_id].edge_ids

    def archive_mechanism(self, mechanism_id, reason):
        self.mechanism_table[mechanism_id].status = "archived"
        self.mechanism_table[mechanism_id].archived_reason = reason
        self.mechanisms_by_status["archived"].add(mechanism_id)
        self.mechanisms_by_status["active"].discard(mechanism_id)
        return True

    def filter_by_scope(self, subset_x, subset_v):
        # Placeholder: filter mechanisms by adding subset_x and subset_v to the scope
        # the idea is that we just go through all the mechanisms and search the subset_x and subset_v
        # in the scope and the % match and return if >25%
        matching_mechanisms = []
        for mechanism in self.mechanism_table.values():
            if self.percentage_scope_match(subset_x, subset_v, mechanism) > 25:
                matching_mechanisms.append(mechanism)
        return matching_mechanisms

    def percentage_scope_match(self, subset_x, subset_v, mechanism):
        requested_terms = set(subset_x) | set(subset_v)
        if not requested_terms:
            return 0.0

        scope_text = json.dumps(mechanism.scope, sort_keys=True)
        matched_terms = sum(term in scope_text for term in requested_terms)
        return 100.0 * matched_terms / len(requested_terms)
