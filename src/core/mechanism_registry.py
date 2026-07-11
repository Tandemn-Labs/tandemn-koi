"""Mechanism registry and lookup indexes.

Mechanisms are reusable causal stories over CandidateGraph edge bundles. The
registry owns identity, duplicate detection, active/archive state, and fast
edge-to-mechanism lookups used by S2, EIG, and agent tooling.
"""

import hashlib
import json
from typing import Any

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

    def match_scope(
        self,
        mechanism: Mechanism,
        context: dict,
        *,
        require_x_overlap: bool = True,
    ) -> dict[str, Any]:
        """Match a mechanism against workload, model, and candidate values."""
        matched_x = sorted(set(mechanism.scope.get("x", ())) & context.keys())
        missing_x = sorted(set(mechanism.scope.get("x", ())) - context.keys())
        result: dict[str, Any] = {
            "quality": "exact",
            "matched_x": matched_x,
            "missing_x": missing_x,
            "condition_results": [],
            "reasons": [],
        }

        def reject(reason: str) -> dict[str, Any]:
            result["quality"] = "reject"
            result["reasons"].append(reason)
            return result

        if mechanism.status != "active":
            return reject("mechanism is not active")

        expected_workload = str(mechanism.scope.get("workload_type") or "any").lower()
        workload = context.get("workload_type", context.get("type", context.get("kind")))
        workload = str(workload).lower() if workload is not None else None
        if expected_workload != "any":
            if workload is None:
                result["quality"] = "partial"
                result["reasons"].append("workload type is missing")
            elif workload != expected_workload:
                return reject(f"workload {workload!r} does not match {expected_workload!r}")

        model_type = str(mechanism.scope.get("model_type") or "any").lower()
        if model_type in {"dense_small", "dense_large"}:
            return reject(f"model type {model_type!r} is not defined by Store")
        if model_type == "moe" and context.get("is_moe") is not True:
            return reject("model is not known to be MoE")
        if model_type not in {"any", "moe"}:
            return reject(f"unknown model type {model_type!r}")

        if require_x_overlap and not matched_x:
            return reject("candidate has no scoped X variable")
        if missing_x:
            result["quality"] = "partial"

        operators = {
            ">": lambda actual, expected: actual > expected,
            "<": lambda actual, expected: actual < expected,
            ">=": lambda actual, expected: actual >= expected,
            "<=": lambda actual, expected: actual <= expected,
            "==": lambda actual, expected: actual == expected,
        }
        for condition in mechanism.scope.get("conditions", ()):
            if not isinstance(condition, dict):
                return reject("condition must be a dict")
            feature = condition.get("feature")
            op = condition.get("op")
            expected = condition.get("value")
            check = {"feature": feature, "op": op, "expected": expected}
            result["condition_results"].append(check)
            if op not in operators:
                check["result"] = False
                return reject(f"unknown condition operator {op!r}")
            if feature not in context:
                check["result"] = None
                result["quality"] = "partial"
                result["reasons"].append(f"condition feature {feature!r} is missing")
                continue

            actual = context[feature]
            check["actual"] = actual
            if op != "==" and (
                isinstance(actual, bool)
                or isinstance(expected, bool)
                or not isinstance(actual, (int, float))
                or not isinstance(expected, (int, float))
            ):
                check["result"] = False
                return reject(f"condition {feature!r} requires numeric operands")
            if op == "==" and (isinstance(actual, bool) != isinstance(expected, bool)):
                check["result"] = False
                return reject(f"condition {feature!r} compares incompatible values")
            try:
                passed = operators[op](actual, expected)
            except TypeError:
                passed = False
            check["result"] = passed
            if not passed:
                return reject(f"condition {feature!r} {op} {expected!r} is false")

        return result

    def find_applicable(
        self,
        context: dict,
        *,
        require_x_overlap: bool = True,
    ) -> list[tuple[Mechanism, dict[str, Any]]]:
        """Return exact and partial matches, with exact matches first."""
        matches = []
        for mechanism in self.mechanism_table.values():
            match = self.match_scope(mechanism, context, require_x_overlap=require_x_overlap)
            if match["quality"] != "reject":
                matches.append((mechanism, match))
        return sorted(matches, key=lambda item: item[1]["quality"] != "exact")
