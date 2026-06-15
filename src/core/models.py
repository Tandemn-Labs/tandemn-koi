from dataclasses import dataclass, field
from enum import Enum

import numpy as np

EnvLabel = tuple[str, str, str, str]


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
    alpha: float = 1.0
    beta: float = 1.0
    visit_count: int = 0
    last_touched_tick: int | None = None
    q_histogram: dict[str, int] = field(
        default_factory=lambda: {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    )
    envs_seen: set[EnvLabel] = field(default_factory=set)
    q3_frequency: float = 0.0


@dataclass
class Mechanism:
    edge_ids: list[str]
    scope: dict
    narrative: str
    status: str = "active"
    name: str | None = None
    mechanism_id: str | None = None
    archived_reason: str | None = None


@dataclass
class MechanismMetadata:
    mechanism_id: str
    alpha: float = 1.0
    beta: float = 1.0
    visit_count: int = 0
    envs_seen: set[EnvLabel] = field(default_factory=set)
    last_touched_tick: int | None = None
    q_histogram: dict[str, int] = field(
        default_factory=lambda: {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    )
    inspection_count: int = 0


@dataclass(frozen=True)  # we add frozen to make class immutable and hashable to use in dicts
class EdgeConfidenceRecord:
    edge: Edge
    metadata: EdgeMetadata


@dataclass(frozen=True)
class MechanismConfidenceRecord:
    mechanism: Mechanism
    metadata: MechanismMetadata


@dataclass
class EvidenceRow:
    row_id: str  # f"{tick}_{job_id}_{rank_id}"
    tick: int  # integer FSM tick id
    deploy_timestamp_utc: float  # forensics; replay anchoring
    job_id: str
    rank_id: str
    env_label: EnvLabel  # (cloud, region, market, gpu_type)
    X: dict[str, object]  # ~60 decision variables
    W_observed: dict[str, float]  # 22 workload features
    V_observed_trajectory: dict[str, np.ndarray]  # sub-tick V samples (all measured V's)
    V_predicted_trajectory: dict[str, np.ndarray]  # surrogate's V_hat(t)
    y_observed_trajectory: dict[str, np.ndarray]  # sub-tick Y samples — Y-CUSUM input
    y_predicted: dict[str, float]  # surrogate's y_hat (scalar; CUSUM broadcasts)
    y_observed_mean: dict[str, float]  # mean of y_observed_trajectory per obj
    residuals_per_v: dict[str, np.ndarray]  # V_obs - V_pred — ICP + CUSUM recalibration
    residuals_per_y: dict[str, np.ndarray]  # y_obs - y_hat — ICP + DRO coverage tracking
    mechanism_ids: list[str]  # all whose scope matched (includes committed)
    cusum_per_mechanism: dict[str, tuple[object, object]]  # mid -> (v_verdict, y_verdict)
    q_label_per_mechanism: dict[str, object | None]  # None = bundle not observable this rank.
    # Q comes from the two CUSUM axes only; ICP modulates EDGE update magnitude
    # via EDGE_BETA_UPDATE's "undecided" row and never nulls the Q (nulling on
    # undecided ICP would freeze all learning until n_env_min envs exist).
    icp_result_per_edge: dict[str, object]
    w_t_snapshot: dict[str, float]  # Tchebycheff weights
    z_star_snapshot: dict[str, float]  #
    J_realized: float  # achieved Tchebycheff scalar
    sigma_realized: float  #
    theory_blob: str | None = None


# ======================================================================
# Plan schema — the typed, validated output the S4 agent must emit.
#
# The root LLM assembles a loose dict/list in its REPL (easy for any
# model, including small open ones), commits it with FINAL_VAR(plan), and
# the harness materializes it into these typed objects. FINAL_VAR does NOT
# let arbitrary Python escape validation: Plan.from_raw parses, and
# agent.materialize_plan adds contextual checks (job existence, state
# legality, env presence) before S5's C0-C7 validation runs.
# ======================================================================


class ActionType(Enum):
    """What a PlanAction does to a job, named by the state transition.

    PLACE     waiting  -> running   launch a new job on a ladder
    KEEP      running  -> running   no change
    SWAP      running  -> running   relaunch on a new/modified ladder
                                     (scale up/down, migrate, retune, replace
                                      a dead chain — see swap_reason)
    DEFER     waiting  -> waiting    stay queued
    PREEMPT   running  -> paused     free resources, resumable later
    RESUME    paused   -> running    relaunch a preempted job
    RETRY     launch_failed -> running  retry after a failed launch
    TERMINATE any      -> stopped    give up (budget/policy exhausted)
    DIAGNOSE  no state change        record a theory only
    """

    PLACE = "place"
    KEEP = "keep"
    SWAP = "swap"
    DEFER = "defer"
    PREEMPT = "preempt"
    RESUME = "resume"
    RETRY = "retry"
    TERMINATE = "terminate"
    DIAGNOSE = "diagnose"


# Actions that deploy or relaunch a ladder (require one).
LADDER_ACTIONS = frozenset(
    {
        ActionType.PLACE,
        ActionType.SWAP,
        ActionType.RETRY,
        ActionType.RESUME,
    }
)

# Active-job churn counted against the swap budget B_t (C4). PLACE/DEFER are
# admission, not churn; KEEP/DIAGNOSE/TERMINATE move no running workload.
SWAP_BUDGET_ACTIONS = frozenset(
    {
        ActionType.SWAP,
        ActionType.PREEMPT,
        ActionType.RETRY,
    }
)

# Required job state per action for the consistency check. None = any state.
REQUIRED_JOB_STATE = {
    ActionType.PLACE: "waiting",
    ActionType.DEFER: "waiting",
    ActionType.KEEP: "running",
    ActionType.SWAP: "running",
    ActionType.PREEMPT: "running",
    ActionType.RESUME: "paused",
    ActionType.RETRY: "launch_failed",
    ActionType.TERMINATE: None,
    ActionType.DIAGNOSE: None,
}

# v0 is AGGREGATE-ONLY: one engine serves prefill+decode for a job. Prefill/
# decode disaggregation is disabled this version - the surrogate exposes no
# per-role throughput (it returns one system-level output tok/s) and rejects
# online PD. To re-enable, restore the full set on the commented line below.
# KNOWN_ROLES = frozenset({"prefill", "decode", "aggregate"})  # full PD set
KNOWN_ROLES = frozenset({"aggregate"})
_V0_DISABLED_ROLES = frozenset({"prefill", "decode"})  # rejected until PD lands


def _as_env_tuple(env) -> tuple | None:
    """Normalize an env list/tuple or pipe-delimited key to a tuple."""
    if env is None:
        return None
    if isinstance(env, (list, tuple)):
        return tuple(str(part) for part in env)
    text = str(env)
    if "|" in text:
        return tuple(text.split("|"))
    return (text,)


@dataclass
class RankSpec:
    """One rank in a ladder: a role-tagged chain config in one environment.

    v0 is AGGREGATE-ONLY: role is always "aggregate" (one engine serves
    prefill+decode). The role-keyed shorthand {"aggregate": {"gpu": ...,
    "chains": ...}} is accepted by from_dict, but the canonical stored form
    is explicit. env and mechanism_id are NOT in the illustrative shorthand
    yet both are required by the system: env is the ICP environment AND the
    launch target (gpu_type alone cannot be launched — which cloud/region?),
    and mechanism_id is the committed mechanism the evidence loop attributes
    each rank's CUSUM/Q to. (Prefill/decode disaggregation is disabled this
    version; see KNOWN_ROLES.)
    """

    role: str  # v0: "aggregate" only ("prefill"/"decode" disabled, see KNOWN_ROLES)
    env: tuple | None  # (cloud, region, market, gpu_type)
    config: dict  # X decision variables for this chain
    n_replicas: int = 1  # "chains" in the shorthand
    mechanism_id: str | None = None
    chain_id: str | None = None  # stable fingerprint for switch-cost ΔL matching

    @classmethod
    def from_dict(cls, raw) -> "RankSpec":
        """Parse a rank from the explicit form or the role-keyed shorthand.

        Explicit: {"role", "env", "config", "n_replicas", "mechanism_id"}.
        Shorthand: {"aggregate": {"gpu":..., "count":..., "tp":..., "pp":...,
                    "chains":..., "env":..., "mechanism_id":...}}.

        v0 is aggregate-only; a prefill/decode rank (either form) is
        rejected with a clear error until disaggregation lands.

        Raises:
            ValueError: If raw is not a dict, the role is unknown, or the
                role is a v0-disabled (prefill/decode) role.
        """
        if not isinstance(raw, dict):
            raise ValueError(f"rank must be a dict, got {type(raw).__name__}")

        requested = next(iter(raw)) if len(raw) == 1 else raw.get("role")
        if requested in _V0_DISABLED_ROLES:
            raise ValueError(
                f"role {requested!r} is disabled in v0 (aggregate-only); emit a "
                "single 'aggregate' rank - prefill/decode disaggregation is not "
                "supported this version"
            )

        if len(raw) == 1:
            only_key = next(iter(raw))
            if only_key in KNOWN_ROLES and isinstance(raw[only_key], dict):
                inner = dict(raw[only_key])
                env = inner.pop("env", None)
                mech = inner.pop("mechanism_id", None)
                chain_id = inner.pop("chain_id", None)
                n_rep_raw = inner.pop("chains", inner.pop("n_replicas", 1)) or 1
                n_rep = int(n_rep_raw)
                return cls(
                    role=only_key,
                    env=_as_env_tuple(env),
                    config=inner,
                    n_replicas=n_rep,
                    mechanism_id=mech,
                    chain_id=chain_id,
                )

        role = raw.get("role")
        if role not in KNOWN_ROLES:
            raise ValueError(f"rank role must be one of {sorted(KNOWN_ROLES)}, got {role!r}")
        n_rep_raw = (
            raw.get("n_replicas") if raw.get("n_replicas") is not None else raw.get("chains", 1)
        )
        return cls(
            role=role,
            env=_as_env_tuple(raw.get("env")),
            config=dict(raw.get("config", {})),
            n_replicas=int(n_rep_raw or 1),
            mechanism_id=raw.get("mechanism_id"),
            chain_id=raw.get("chain_id"),
        )

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "env": list(self.env) if self.env else None,
            "config": dict(self.config),
            "n_replicas": self.n_replicas,
            "mechanism_id": self.mechanism_id,
            "chain_id": self.chain_id,
        }

    def gpus_per_chain(self) -> int:
        """GPUs one replica of this rank occupies.

        Prefers an explicit gpu_count/count in the config (the user's
        ladder shorthand uses 'count'); otherwise tp * pp - one model
        replica spans the tensor x pipeline grid. Expert parallelism (ep)
        is framework-specific and not folded in here; set gpu_count
        explicitly for MoE layouts that need it.
        """
        cfg = self.config
        explicit = cfg.get("gpu_count", cfg.get("count"))
        if explicit is not None:
            return max(1, int(explicit))
        return max(1, int(cfg.get("tp", 1)) * int(cfg.get("pp", 1)))

    def total_gpus(self) -> int:
        """Total GPUs this rank consumes: n_replicas * gpus_per_chain."""
        return self.n_replicas * self.gpus_per_chain()


@dataclass
class PlanAction:
    """One job's decision this tick."""

    job_id: str
    type: ActionType
    tenant_id: str | None = None
    ladder: list[RankSpec] | None = None
    target_tps: float | None = None
    mechanism_id: str | None = None  # job-level committed mechanism; ranks may override
    swap_reason: str | None = None  # scale_up|scale_down|migrate|replace|retune (SWAP trace)
    budget_ref: str | None = None  # BudgetSlice reference for resource-consuming actions
    rationale: str | None = None
    predicted_y: dict | None = None  # advisory: LLM's predict_outcome result; scoring re-derives

    @classmethod
    def from_dict(cls, raw, job_id: str | None = None) -> "PlanAction":
        """Parse a PlanAction from a dict.

        Requires the canonical "type" key, case-insensitively. job_id may
        come from the dict or be supplied by the caller (when the plan is
        keyed by job_id).

        Raises:
            ValueError: If the dict is malformed or the action is unknown.
        """
        if not isinstance(raw, dict):
            raise ValueError(f"action must be a dict, got {type(raw).__name__}")
        jid = raw.get("job_id", job_id)
        if not jid:
            raise ValueError("action missing job_id")
        type_name = raw.get("type")
        if type_name is None:
            raise ValueError(f"job {jid}: action missing canonical 'type'")
        try:
            action_type = (
                type_name
                if isinstance(type_name, ActionType)
                else ActionType(str(type_name).lower())
            )
        except ValueError as exc:
            valid = [a.value for a in ActionType]
            raise ValueError(
                f"job {jid}: unknown action type {type_name!r}; valid: {valid}"
            ) from exc

        ladder_raw = raw.get("ladder")
        ladder = (
            [RankSpec.from_dict(r) for r in ladder_raw] if isinstance(ladder_raw, list) else None
        )
        return cls(
            job_id=str(jid),
            type=action_type,
            tenant_id=raw.get("tenant_id"),
            ladder=ladder,
            target_tps=raw.get("target_tps"),
            mechanism_id=raw.get("mechanism_id"),
            swap_reason=raw.get("swap_reason"),
            budget_ref=raw.get("budget_ref"),
            rationale=raw.get("rationale"),
            predicted_y=raw.get("predicted_y") or raw.get("y_hat"),
        )

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "type": self.type.value,
            "tenant_id": self.tenant_id,
            "ladder": [r.to_dict() for r in self.ladder] if self.ladder else None,
            "target_tps": self.target_tps,
            "mechanism_id": self.mechanism_id,
            "swap_reason": self.swap_reason,
            "budget_ref": self.budget_ref,
            "rationale": self.rationale,
        }


@dataclass
class Plan:
    """The cluster-wide decision for one tick: one action per job.

    A Plan spans every tenant's jobs, so tenant identity lives on each
    PlanAction (not on the Plan). operator_id is provenance only — who/what
    produced the plan — mapping the illustrative "user_id" field.
    """

    tick: int
    actions: list[PlanAction] = field(default_factory=list)
    koi_version: str | None = None
    operator_id: str | None = None
    tick_rationale: str | None = None

    def action_for(self, job_id: str) -> PlanAction | None:
        """Return the action for one job, or None."""
        for action in self.actions:
            if action.job_id == job_id:
                return action
        return None

    def job_ids(self) -> set:
        return {a.job_id for a in self.actions}

    @classmethod
    def from_raw(cls, raw, tick: int) -> "Plan":
        """Normalize whatever the LLM committed into a typed Plan.

        Accepts: an already-typed Plan; a dict with an "actions" list
        (Plan-shaped); a plain list of action dicts; or a dict keyed by
        job_id -> action dict (the internal convenience form). Parsing
        raises ValueError on malformed input; contextual validation
        (state legality, env presence, coverage) is the caller's job.
        """
        if isinstance(raw, Plan):
            return raw

        meta = {}
        if isinstance(raw, dict) and "actions" in raw:
            meta = raw
            raw_actions = raw["actions"]
        else:
            raw_actions = raw

        actions: list[PlanAction] = []
        if isinstance(raw_actions, list):
            for entry in raw_actions:
                actions.append(PlanAction.from_dict(entry))
        elif isinstance(raw_actions, dict):
            for job_id, entry in raw_actions.items():
                actions.append(PlanAction.from_dict(entry, job_id=job_id))
        else:
            raise ValueError("plan must be a Plan, an actions list, or a job_id->action dict")

        return cls(
            tick=int(meta.get("tick", tick)),
            actions=actions,
            koi_version=meta.get("koi_version"),
            operator_id=meta.get("operator_id", meta.get("user_id")),
            tick_rationale=meta.get("tick_rationale"),
        )
