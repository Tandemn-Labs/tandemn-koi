"""Deterministic plan and mechanism-proposal validation.

The Validator is the authoritative C0-C6 feasibility gate. The FSM runs
``val_plan`` in S5 (one repair iteration back to the agent on failure, then a
keep-all fallback); the agent pre-screens K_P candidates with the same call;
agent tools expose it as ``check_feasibility``. ``val_mechanism_proposal`` and
``val_scopeability`` gate new mechanisms before the registry admits them.

Validation runs against the FROZEN cluster snapshot S0 produced, so it sees the
same state the plan was built against. Dependencies (candidate_graph,
mechanism_registry, ...) are injected once and any check whose dependency is
absent is skipped rather than failing - so the gate is usable in tests with only
a snapshot, and progressively stricter as more of the system is wired.

Constraint hierarchy (tenant policy ahead of resource feasibility):

    C0 structure      plan parses; one action per job (no duplicates)
    C1 state          job exists in the snapshot; action legal for its state
    C2 coverage       every job in the snapshot has an action
    C3 tenant/budget  per-tenant policy (skipped when no tenant_registry)
    C4 swap budget    active-job churn does not exceed B_t
    C5 capacity       allocation-unit footprint fits snapshot free capacity
    C6 chain physics  each rank is launchable (5-tuple env, >=1 replica, fits)

MVP note: SLO/DRO risk is score-only via compute_sigma, not validation-gated.
TODO(v1): revisit a hard SLO gate once the cutoff policy is well understood.
"""

from dataclasses import dataclass, field
from typing import Any

from src.core.models import (
    LADDER_ACTIONS,
    REQUIRED_JOB_STATE,
    SWAP_BUDGET_ACTIONS,
    ActionType,
    Mechanism,
    Plan,
)

# Actions whose target job must already exist in the cluster snapshot. PLACE and
# DEFER admit waiting jobs; DIAGNOSE/TERMINATE may reference jobs outside the
# running/waiting inventory (post-mortems, cleanup), so existence is not enforced
# for them. TODO(v0): add preempt/resume/retry with lifecycle snapshot support.
_EXISTENCE_REQUIRED = frozenset({ActionType.KEEP, ActionType.SWAP})

_KNOWN_WORKLOAD_TYPES = frozenset({"any", "online", "batch"})
_KNOWN_MODEL_TYPES = frozenset({"any", "moe", "dense_small", "dense_large"})
_CONDITION_OPERATORS = frozenset({">", "<", ">=", "<=", "=="})
_ALLOWED_SCOPE_KEYS = frozenset({"x", "v", "workload_type", "model_type", "conditions"})


@dataclass
class ValidationResult:
    """Outcome of a validation pass.

    Attributes:
        feasible: True iff no violations were found.
        violations: Flat, tier-ordered list of human-readable violation
            strings (these are fed back to the agent for one repair pass).
        by_constraint: Violations grouped by constraint code (C0..C7) for
            diagnostics and tracing.
    """

    feasible: bool
    violations: list[str] = field(default_factory=list)
    by_constraint: dict[str, list[str]] = field(default_factory=dict)


class Validator:
    """C0-C7 plan validator plus mechanism-proposal validation.

    Args:
        candidate_graph: CandidateGraph; required for mechanism/scope checks.
        mechanism_registry: MechanismRegistry; used for duplicate detection.
        tenant_registry: Optional tenant policy service. When present and it
            exposes ``check_plan(plan, snapshot) -> (ok, violations)``, C3 runs.
        slo_predictor: Reserved for a future hard SLO gate; MVP does not call it.
        slo_breach_threshold: Reserved threshold for the future SLO gate.
        resource_map: Optional resource service; when present it owns allocation
            footprint semantics for C5/C6.
    """

    def __init__(
        self,
        candidate_graph=None,
        mechanism_registry=None,
        tenant_registry=None,
        slo_predictor=None,
        slo_breach_threshold: float = 0.5,
        resource_map=None,
    ):
        self.candidate_graph = candidate_graph
        self.mechanism_registry = mechanism_registry
        self.tenant_registry = tenant_registry
        self.resource_map = resource_map
        self.slo_predictor = slo_predictor
        self.slo_breach_threshold = float(slo_breach_threshold)

    # ------------------------------------------------------------------
    # Plan validation
    # ------------------------------------------------------------------

    def val_plan(self, plan, cluster_snapshot=None, slow_state=None) -> ValidationResult:
        """Validate a cluster plan against the snapshot and slow state.

        Runs the C0-C6 hierarchy. C0 (structure) short-circuits: a plan that
        does not parse or repeats a job cannot be checked further. The
        remaining tiers all run and their violations are collected together
        (tier-ordered), so one repair pass can fix everything at once.

        Args:
            plan: A typed Plan or any raw form Plan.from_raw accepts.
            cluster_snapshot: The frozen S0 snapshot. Checks that need it
                (existence, coverage, capacity) are skipped when it is None.
            slow_state: The tick's SlowState; C4 reads its swap budget B_t.

        Returns:
            A ValidationResult with .feasible and .violations.
        """
        by: dict[str, list[str]] = {}

        # C0 - structure. Must hold before anything else can be trusted.
        typed, c0 = self._check_structure(plan)
        if c0:
            by["C0"] = c0
            return self._result(by)
        assert typed is not None

        states = self._job_states(cluster_snapshot)

        self._record(by, "C1", self._check_state(typed, states))
        self._record(by, "C2", self._check_coverage(typed, states))
        self._record(by, "C3", self._check_tenant(typed, cluster_snapshot))
        self._record(by, "C4", self._check_swap_budget(typed, cluster_snapshot, slow_state))
        self._record(by, "C5", self._check_capacity(typed, cluster_snapshot))
        self._record(by, "C6", self._check_chain_physics(typed, cluster_snapshot))
        # TODO(v1): Re-enable when SLO/DRO has a clear hard-gate policy.
        # MVP keeps SLO/DRO score-only in agent_tools.compute_sigma.
        # self._record(by, "C7", self._check_slo(typed, cluster_snapshot))

        return self._result(by)

    # ----- C0 structure -----

    @staticmethod
    def _check_structure(plan) -> tuple[Plan | None, list[str]]:
        """Coerce to a typed Plan and reject duplicate job actions."""
        try:
            typed = plan if isinstance(plan, Plan) else Plan.from_raw(plan, tick=0)
        except (ValueError, TypeError) as exc:
            return None, [f"C0 structure: plan does not parse ({exc})"]

        violations: list[str] = []
        seen: set[str] = set()
        for action in typed.actions:
            if action.job_id in seen:
                violations.append(f"C0 structure: duplicate action for job {action.job_id}")
            seen.add(action.job_id)
        return typed, violations

    # ----- C1 state legality -----

    @staticmethod
    def _check_state(typed: Plan, states: dict[str, str] | None) -> list[str]:
        """Job exists in the snapshot and the action is legal for its state.

        Existence is only enforced for actions that must target a live job
        (see _EXISTENCE_REQUIRED). State legality is enforced only when the
        job's state is known and a required state is defined.
        """
        if states is None:
            return []
        violations: list[str] = []
        for action in typed.actions:
            jid = action.job_id
            known = jid in states
            if not known and action.type in _EXISTENCE_REQUIRED:
                violations.append(f"C1 state: job {jid} ({action.type.value}) not in snapshot")
                continue
            required = REQUIRED_JOB_STATE.get(action.type)
            actual = states.get(jid)
            if required is not None and actual is not None and actual != required:
                violations.append(
                    f"C1 state: job {jid} {action.type.value} needs state "
                    f"{required!r}, job is {actual!r}"
                )
        return violations

    # ----- C2 coverage -----

    @staticmethod
    def _check_coverage(typed: Plan, states: dict[str, str] | None) -> list[str]:
        """Every job in the snapshot inventory must carry exactly one action."""
        if states is None:
            return []
        covered = typed.job_ids()
        return [
            f"C2 coverage: job {jid} ({state}) has no action in the plan"
            for jid, state in states.items()
            if jid not in covered
        ]

    # ----- C3 tenant / budget -----

    def _check_tenant(self, typed: Plan, snapshot) -> list[str]:
        """Delegate to the tenant registry's policy check when one is wired."""
        registry = self.tenant_registry
        if registry is None or not hasattr(registry, "check_plan"):
            return []
        try:
            ok, violations = registry.check_plan(typed, snapshot)
        except Exception as exc:  # a policy backend error must not crash S5
            return [f"C3 tenant: policy check failed ({exc})"]
        return [] if ok else [f"C3 tenant: {v}" for v in (violations or [])]

    # ----- C4 swap budget -----

    def _check_swap_budget(self, typed: Plan, snapshot, slow_state) -> list[str]:
        """Active-job churn must not exceed the slow loop's swap budget B_t.

        Counts actions in SWAP_BUDGET_ACTIONS (swap) on jobs
        that are currently active - matching the C4 definition that only
        running-workload churn is budgeted.
        """
        budget = getattr(slow_state, "B_t", None)
        if budget is None:
            return []
        active = self._active_job_ids(snapshot)
        churn = [
            a
            for a in typed.actions
            if a.type in SWAP_BUDGET_ACTIONS and (not active or a.job_id in active)
        ]
        if len(churn) > int(budget):
            ids = ", ".join(a.job_id for a in churn)
            return [
                f"C4 swap budget: {len(churn)} churning actions ({ids}) exceed B_t={int(budget)}"
            ]
        return []

    # ----- C5 resource feasibility -----

    def _check_capacity(self, typed: Plan, snapshot) -> list[str]:
        """Requested GPUs per env must fit the snapshot's free capacity.

        Conservative: a job's new ladder is counted in full against free
        capacity (a swap is not credited for the GPUs its old ladder releases).
        This matches ResourceMap.check_resource_feasibility and biases toward
        rejecting-and-repairing over deploying an over-subscription.
        """
        resources = self._resources(snapshot)
        if resources is None:
            return []
        try:
            requested, requested_by_pool = self._requested_capacity(typed, resources)
        except (TypeError, ValueError) as exc:
            return [f"C5 capacity: {exc}"]
        violations: list[str] = []
        resource_map = self.resource_map
        pool_capacity = (
            resource_map.pool_capacity(resources)
            if resource_map is not None and hasattr(resource_map, "pool_capacity")
            else {}
        )
        pool_failed_envs = set()
        for (env_key, instance_type), demand in sorted(requested_by_pool.items()):
            limit = pool_capacity.get((env_key, instance_type))
            if limit is None:
                pool_failed_envs.add(env_key)
                violations.append(
                    f"C5 capacity: pool {instance_type} is not available in env {env_key}"
                )
                continue
            units = int(demand["units"])
            available = int(limit["available_units"])
            if units > available:
                pool_failed_envs.add(env_key)
                unit = "GPUs" if limit["allocation_kind"] == "gpu" else "instances"
                violations.append(
                    f"C5 capacity: env {env_key} pool {instance_type} requested "
                    f"{units} {unit}, only {available} free"
                )
        for env_key, gpus in sorted(requested.items()):
            info = resources.get(env_key)
            if info is None:
                violations.append(
                    f"C5 capacity: env {env_key} requested {gpus} GPUs but is not in the resource map"
                )
                continue
            free = int(info.get("free", 0))
            if gpus > free and env_key not in pool_failed_envs:
                violations.append(
                    f"C5 capacity: env {env_key} requested {gpus} GPUs, only {free} free"
                )
        return violations

    # ----- C6 chain physics -----

    def _check_chain_physics(self, typed: Plan, snapshot) -> list[str]:
        """Each deployed rank must be physically launchable.

        env must be a 5-tuple (market, cloud, region, zone, gpu_type); replicas >= 1;
        a single chain's GPU footprint must not exceed the env's total GPUs (a
        chain that cannot fit on the env can never be placed, no matter the
        free count).
        """
        resources = self._resources(snapshot)
        violations: list[str] = []
        for action in typed.actions:
            if action.type not in LADDER_ACTIONS:
                continue
            if not action.ladder:
                violations.append(
                    f"C6 physics: job {action.job_id} {action.type.value} has no ladder"
                )
                continue
            for i, rank in enumerate(action.ladder):
                if rank.env is None or len(rank.env) != 5:
                    violations.append(
                        f"C6 physics: job {action.job_id} rank {i} env must be a 5-tuple "
                        "(market, cloud, region, zone, gpu_type) to be launchable"
                    )
                    continue
                if rank.n_replicas < 1:
                    violations.append(
                        f"C6 physics: job {action.job_id} rank {i} n_replicas must be >= 1"
                    )
                cfg = rank.config or {}
                rank_errors = []
                if not cfg.get("instance_type"):
                    rank_errors.append("instance_type is required")
                gpu_count = cfg.get("gpu_count", cfg.get("count"))
                if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count <= 0:
                    rank_errors.append("gpu_count/count must be a positive integer")
                for key in ("tp", "pp"):
                    value = cfg.get(key)
                    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                        rank_errors.append(f"{key} must be a positive integer")
                if rank_errors:
                    violations.extend(
                        f"C6 physics: job {action.job_id} rank {i} {error}" for error in rank_errors
                    )
                    continue
                per_chain, gpu_error = self._rank_engine_gpus(rank)
                if gpu_error:
                    violations.append(f"C6 physics: job {action.job_id} rank {i} {gpu_error}")
                    continue
                assert per_chain is not None
                if per_chain < 1:
                    violations.append(
                        f"C6 physics: job {action.job_id} rank {i} resolves to < 1 GPU per chain"
                    )
                if resources is not None:
                    info = resources.get(self._env_key(rank.env))
                    if info is not None:
                        allocation, allocation_error = self._rank_allocation_summary(
                            rank, resources
                        )
                        if allocation_error:
                            violations.append(
                                f"C6 physics: job {action.job_id} rank {i}: {allocation_error}"
                            )
                            continue
                        unit_gpus = int(allocation.get("gpus_per_unit", per_chain))
                        if allocation.get("allocation_kind") != "gpu" and per_chain > unit_gpus:
                            inst = allocation.get("instance_type")
                            violations.append(
                                f"C6 physics: job {action.job_id} rank {i} needs {per_chain} "
                                f"engine GPUs but {inst} has {unit_gpus}"
                            )
                        total = int(info.get("total", 0))
                        if total and unit_gpus > total:
                            violations.append(
                                f"C6 physics: job {action.job_id} rank {i} reserves {unit_gpus} "
                                f"GPUs/replica but env total is {total}"
                            )
        return violations

    # ----- Future C7 SLO chance (disabled in MVP) -----

    def _check_slo(self, typed: Plan, snapshot) -> list[str]:
        """Future hard SLO gate; intentionally not called in MVP.

        SLO/DRO risk is currently score-only via agent_tools.compute_sigma.
        """
        predictor = self.slo_predictor
        if predictor is None:
            return []
        violations: list[str] = []
        for action in typed.actions:
            if action.type not in LADDER_ACTIONS or not action.ladder:
                continue
            try:
                pr = float(predictor(action, snapshot))
            except Exception:
                continue
            if pr > self.slo_breach_threshold:
                violations.append(
                    f"C7 SLO: job {action.job_id} predicted breach probability "
                    f"{pr:.2f} exceeds {self.slo_breach_threshold:.2f}"
                )
        return violations

    # ------------------------------------------------------------------
    # Mechanism-proposal validation
    # ------------------------------------------------------------------

    def val_mechanism_proposal(self, proposal) -> tuple[bool, list[str]]:
        """Validate a new-mechanism proposal before the registry admits it.

        Checks (mirroring the admission path): every edge exists in the graph;
        the bundle topology only uses X->V and V->Y edges; the proposal is not
        a duplicate of an existing mechanism; and the scope is well-formed
        (delegated to val_scopeability).

        Args:
            proposal: a Mechanism, or a dict with edge_ids / scope / narrative.

        Returns:
            (ok, violations).
        """
        try:
            mechanism = self._as_mechanism(proposal)
        except (TypeError, ValueError) as exc:
            return False, [f"mechanism: proposal does not parse ({exc})"]
        violations: list[str] = []

        if not mechanism.edge_ids:
            violations.append("mechanism: proposal has no edges")

        cg = self.candidate_graph
        if cg is None:
            violations.append("mechanism: validator has no candidate_graph bound")
        else:
            edge_objs = []
            for eid in mechanism.edge_ids:
                if eid not in cg.edge_table:
                    violations.append(f"mechanism: edge {eid!r} not in CandidateGraph")
                else:
                    edge_objs.append(cg.edge_table[eid])
            if edge_objs:
                if not cg.val_topology(edge_objs):
                    violations.append(
                        "mechanism: topology violation - only X->V and V->Y edges allowed"
                    )
                else:
                    if not cg.check_connected(edge_objs):
                        violations.append("mechanism: edge bundle is disconnected")
                    if not self._has_complete_path(edge_objs):
                        violations.append("mechanism: edge bundle has no complete X->V->Y path")

        if self.mechanism_registry is not None:
            is_dup, existing = self.mechanism_registry.is_duplicate_mechanism(mechanism)
            if is_dup:
                violations.append(f"mechanism: duplicate of existing mechanism {existing}")

        ok_scope, scope_violations = self.val_scopeability(mechanism.scope)
        if not ok_scope:
            violations.extend(scope_violations)

        return len(violations) == 0, violations

    def val_scopeability(self, scope) -> tuple[bool, list[str]]:
        """Validate a mechanism scope.

        Structural checks always run: scope is a dict, names at least one X or
        V variable, uses known workload/model qualifiers, and has well-formed
        conditions. Graph checks ensure scoped and conditioned variables have
        the expected node types.

        Args:
            scope: The proposal's canonical scope dict.

        Returns:
            (ok, violations).
        """
        try:
            scope = self._canonical_scope(scope)
        except ValueError as exc:
            return False, [f"scope: {exc}"]

        x_vars = scope["x"]
        v_vars = scope["v"]
        violations: list[str] = []
        if not x_vars:
            violations.append("scope: must name at least one X variable")

        workload_type = scope.get("workload_type")
        if workload_type is not None and str(workload_type).lower() not in _KNOWN_WORKLOAD_TYPES:
            violations.append(
                f"scope: unknown workload_type {workload_type!r} "
                f"(expected one of {sorted(_KNOWN_WORKLOAD_TYPES)})"
            )

        model_type = scope.get("model_type")
        if model_type is not None and str(model_type).lower() not in _KNOWN_MODEL_TYPES:
            violations.append(
                f"scope: unknown model_type {model_type!r} "
                f"(expected one of {sorted(_KNOWN_MODEL_TYPES)})"
            )

        conditions = scope["conditions"]

        condition_features = []
        for index, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                violations.append(f"scope: conditions[{index}] must be a dict")
                continue
            feature = condition.get("feature")
            operator = condition.get("op")
            value = condition.get("value")
            if not isinstance(feature, str) or not feature:
                violations.append(f"scope: conditions[{index}].feature must be a non-empty string")
            else:
                condition_features.append((index, feature))
            if operator not in _CONDITION_OPERATORS:
                violations.append(
                    f"scope: conditions[{index}].op {operator!r} is unknown "
                    f"(expected one of {sorted(_CONDITION_OPERATORS)})"
                )
            if operator in {">", "<", ">=", "<="} and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                violations.append(
                    f"scope: conditions[{index}].value must be numeric for {operator}"
                )
            elif operator == "==" and not isinstance(value, (bool, int, float)):
                violations.append(
                    f"scope: conditions[{index}].value must be boolean or numeric for =="
                )

        cg = self.candidate_graph
        if cg is not None:
            node_table = cg.node_table
            for var in x_vars:
                node = node_table.get(var)
                if node is None:
                    violations.append(f"scope: X variable {var!r} is not a graph node")
                elif node.node_type != "X":
                    violations.append(f"scope: {var!r} is a {node.node_type} node, not X")
            for var in v_vars:
                node = node_table.get(var)
                if node is None:
                    violations.append(f"scope: V variable {var!r} is not a graph node")
                elif node.node_type != "V":
                    violations.append(f"scope: {var!r} is a {node.node_type} node, not V")
            for index, feature in condition_features:
                node = node_table.get(feature)
                if node is None:
                    violations.append(
                        f"scope: conditions[{index}].feature {feature!r} is not a graph node"
                    )
                elif node.node_type != "X":
                    violations.append(
                        f"scope: conditions[{index}].feature {feature!r} is a "
                        f"{node.node_type} node, not X"
                    )

        return len(violations) == 0, violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result(by_constraint: dict[str, list[str]]) -> ValidationResult:
        """Flatten per-constraint violations into a tier-ordered result."""
        ordered = [v for code in sorted(by_constraint) for v in by_constraint[code]]
        return ValidationResult(
            feasible=len(ordered) == 0,
            violations=ordered,
            by_constraint=by_constraint,
        )

    @staticmethod
    def _record(by: dict[str, list[str]], code: str, violations: list[str]) -> None:
        if violations:
            by[code] = violations

    @classmethod
    def _as_mechanism(cls, proposal) -> Mechanism:
        """Coerce a proposal dict into a Mechanism (pass objects through)."""
        if isinstance(proposal, Mechanism):
            proposal.scope = cls._canonical_scope(proposal.scope)
            return proposal
        if isinstance(proposal, dict):
            scope = proposal.get("scope") or proposal.get("applicable_to") or {}
            return Mechanism(
                edge_ids=list(proposal.get("edge_ids", [])),
                scope=cls._canonical_scope(scope),
                narrative=str(proposal.get("narrative", proposal.get("llm_blurb", ""))),
            )
        raise ValueError(
            f"mechanism proposal must be a Mechanism or dict, got {type(proposal).__name__}"
        )

    @staticmethod
    def _canonical_scope(scope) -> dict[str, Any]:
        if not isinstance(scope, dict):
            raise ValueError("scope must be a dict")
        unknown = sorted(set(scope) - _ALLOWED_SCOPE_KEYS)
        if unknown:
            raise ValueError(f"unknown scope keys: {unknown}")

        def variables(key: str) -> list[str]:
            value = scope.get(key, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"{key} must be a list of non-empty strings")
            return sorted(set(value))

        conditions = scope.get("conditions", [])
        if not isinstance(conditions, list):
            raise ValueError("conditions must be a list")
        condition_keys = {"feature", "op", "value"}
        if any(
            not isinstance(condition, dict) or set(condition) != condition_keys
            for condition in conditions
        ):
            raise ValueError("conditions require feature, op, and value")

        return {
            "x": variables("x"),
            "v": variables("v"),
            "workload_type": str(scope.get("workload_type", "any")).lower(),
            "model_type": str(scope.get("model_type", "any")).lower(),
            "conditions": [dict(condition) for condition in conditions],
        }

    @staticmethod
    def _has_complete_path(edges) -> bool:
        mediators = {edge.dst for edge in edges if edge.src_type == "X" and edge.dst_type == "V"}
        return any(
            edge.src in mediators and edge.src_type == "V" and edge.dst_type == "Y"
            for edge in edges
        )

    @staticmethod
    def _job_states(snapshot) -> dict[str, str] | None:
        """Map job_id -> semantic state from the snapshot, or None.

        Prefers an explicit job_states() accessor; otherwise infers active
        jobs as 'running' and pending jobs as 'waiting'. None means the
        snapshot exposes no inventory, so existence/state/coverage are skipped.
        """
        if snapshot is None:
            return None
        if hasattr(snapshot, "job_states"):
            return dict(snapshot.job_states())
        states: dict[str, str] = {}
        has_any = False
        if hasattr(snapshot, "active_jobs_summary"):
            has_any = True
            for j in snapshot.active_jobs_summary():
                states[j.get("job_id", j.get("id"))] = j.get("state", "running")
        if hasattr(snapshot, "pending_jobs_summary"):
            has_any = True
            for j in snapshot.pending_jobs_summary():
                states[j.get("job_id", j.get("id"))] = j.get("state", "waiting")
        return states if has_any else None

    @staticmethod
    def _active_job_ids(snapshot) -> set[str]:
        if snapshot is None or not hasattr(snapshot, "active_jobs_summary"):
            return set()
        return {j.get("job_id", j.get("id")) for j in snapshot.active_jobs_summary()}

    @staticmethod
    def _resources(snapshot) -> dict[str, Any] | None:
        """The snapshot's per-env capacity table (env_key -> {free, total, ...})."""
        if snapshot is None:
            return None
        if hasattr(snapshot, "resources_summary"):
            return snapshot.resources_summary()
        return getattr(snapshot, "resources", None)

    def _requested_gpus_by_env(
        self,
        typed: Plan,
        resources: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Reserved GPU footprint per env for ladder-bearing actions."""
        requested: dict[str, int] = {}
        for action in typed.actions:
            if action.type not in LADDER_ACTIONS:
                continue
            for rank in action.ladder or []:
                env_key = self._env_key(rank.env)
                requested[env_key] = requested.get(env_key, 0) + self._rank_footprint(
                    rank, resources
                )
        return requested

    def _requested_capacity(
        self,
        typed: Plan,
        resources: dict[str, Any],
    ) -> tuple[dict[str, int], dict[tuple[str, str], dict[str, int]]]:
        resource_map = self.resource_map
        if resource_map is not None and hasattr(resource_map, "requested_capacity"):
            return resource_map.requested_capacity(typed, resources)
        return self._requested_gpus_by_env(typed, resources), {}

    def _rank_footprint(self, rank, resources: dict[str, Any] | None) -> int:
        engine_gpus, _ = self._rank_engine_gpus(rank)
        if engine_gpus is None:
            return 0
        resource_map = self.resource_map
        if resource_map is not None and hasattr(resource_map, "rank_capacity_footprint"):
            try:
                return int(resource_map.rank_capacity_footprint(rank, resources))
            except (TypeError, ValueError):
                pass
        return int(rank.n_replicas) * engine_gpus

    def _rank_allocation_summary(
        self,
        rank,
        resources: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str | None]:
        resource_map = self.resource_map
        if resource_map is not None and hasattr(resource_map, "rank_allocation_summary"):
            try:
                return dict(resource_map.rank_allocation_summary(rank, resources)), None
            except (TypeError, ValueError) as exc:
                return {}, str(exc)
        engine_gpus, error = self._rank_engine_gpus(rank)
        if error or engine_gpus is None:
            return {}, error
        return {
            "allocation_kind": "gpu",
            "instance_type": None,
            "gpus_per_unit": engine_gpus,
            "capacity_per_replica": engine_gpus,
            "engine_gpus": engine_gpus,
        }, None

    @staticmethod
    def _rank_engine_gpus(rank) -> tuple[int | None, str | None]:
        try:
            return rank.gpus_per_chain(), None
        except (TypeError, ValueError):
            cfg = rank.config or {}
            if cfg.get("gpu_count") is not None:
                return None, "gpu_count must be a positive integer"
            if cfg.get("count") is not None:
                return None, "count must be a positive integer"
            return None, "tp and pp must be positive integers"

    @staticmethod
    def _env_key(env) -> str:
        if isinstance(env, (tuple, list)):
            return "|".join(str(part) for part in env)
        return str(env)
