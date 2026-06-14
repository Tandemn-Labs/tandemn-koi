from dataclasses import dataclass, field

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
