from dataclasses import dataclass, field

import numpy as np


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
    envs_seen: set[str] = field(default_factory=set)
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
    envs_seen: set[str] = field(default_factory=set)
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


# @dataclass
# class EvidenceRow:
#     row_id: str  # f"{tick}_{job_id}_{rank_id}"
#     tick: int  # integer FSM tick id
#     job_id: str
#     rank_id: str
#     env_label: tuple[str, str, str, str]  # (cloud, region, market, gpu_type) - ICP env
#     mechanism_id: tuple[str, ...] # List of all applicable ids
#     X: dict[str, object]  # ~60 decision variables (deploy snapshot)
#     W_observed: dict[str, float]  # 22 workload features at deploy time
#     V_observed_trajectory: dict[str, np.ndarray]  # sub-tick samples per V in bundle
#     V_predicted_trajectory: dict[str, np.ndarray]  # surrogate-predicted V (or broadcast scalar)
#     y_observed_trajectory: dict[str, np.ndarray]  # sub-tick Y samples - needed for Y-CUSUM
#     y_predicted: dict[str, float]  # surrogate y_hat per Y, constant during tick
#     residuals_per_v: dict[str, np.ndarray]  # precomputed observed - predicted for ICP
#     residuals_per_y: dict[str, np.ndarray]  # ditto, per-objective
#     v_cusum_result: object  # MATCHED / DIVERGED on V bundle
#     y_cusum_result: object  # MATCHED / DIVERGED on Y bundle
#     icp_result_per_edge: dict[str, object]  # ACCEPT / REJECT / UNDECIDED per edge_id
#     q_label: object | None  # None if any ICP UNDECIDED -> excluded from Q1 rate
#     w_t_snapshot: dict[str, float]  # Tchebycheff weights at deploy time
#     z_star_snapshot: dict[str, float]  # Pareto reference at deploy time
#     J_realized: float  # Tchebycheff scalar actually achieved
#     sigma_realized: float  # per-candidate sigma achieved
#     cusum_params_v: dict[str, tuple[float, float]]  # (delta, h) per V used this tick
#     cusum_params_y: dict[str, tuple[float, float]]  # (delta, h) per Y used this tick
#     theory_blob: str | None = None  # NL retrospective from agent


@dataclass
class EvidenceRow:
    row_id: str  # f"{tick}_{job_id}_{rank_id}"
    tick: int  # integer FSM tick id
    deploy_timestamp_utc: float  # forensics; replay anchoring
    job_id: str
    rank_id: str
    env_label: tuple[str, str, str, str]  # (cloud, region, market, gpu_type)
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
    cusum_per_mechanism: dict[str, tuple[object, object]]
    q_label_per_mechanism: dict[str, object | None]  # None where any ICP=UNDECIDED
    icp_result_per_edge: dict[str, object]
    w_t_snapshot: dict[str, float]  # Tchebycheff weights
    z_star_snapshot: dict[str, float]  #
    J_realized: float  # achieved Tchebycheff scalar
    sigma_realized: float  #
    theory_blob: str | None = None
