"""
koi — Automatic LLM inference placement system.

Main entry point:
    from koi import KoiPlacement
    koi = KoiPlacement(api_key="sk-ant-...")
    decision = koi.decide(request, resource_map)

Full pipeline:
    Oracle (feasibility + prediction) → LLM Ensemble (3 thinkers + judge) → PlacementDecision

Monitoring + refinement (Phase 2):
    KoiMonitor        — fetches live metrics, applies Kalman filter + deadband
    KoiRefinement     — delta store, policy memory, PES tracker, RAG correction
    SwapArbiter       — multi-job resource rebalancing
    ExplorationManager— UCB-based active exploration loop
"""

from koi.placement import KoiPlacement
from koi.intake import koi_deploy, parse_user_request, fetch_resource_map
from koi.oracle import Oracle
from koi.ensemble import KoiEnsemble
from koi.schemas import (
    JobRequest,
    ResourceMap,
    GPUResource,
    PlacementDecision,
    PlacementConfig,
    EngineConfig,
    PredictedMetrics,
    OracleCandidate,
    ThinkerProposal,
    RuntimeMetrics,
    DeltaRecord,
    PESComponents,
    TaskType,
    Objective,
    DataSource,
)
from koi.monitor import KoiMonitor, KalmanFilter1D, DeadbandController, SLOState
from koi.refinement import KoiRefinement, compute_pes
from koi.arbiter import SwapArbiter, RunningJob
from koi.exploration import ExplorationManager
from koi.metrics_api import TandemMetricsAPISource, VLLMPrometheusSource

__all__ = [
    "KoiPlacement", "Oracle", "KoiEnsemble",
    "JobRequest", "ResourceMap", "GPUResource",
    "PlacementDecision", "PlacementConfig", "EngineConfig",
    "PredictedMetrics", "OracleCandidate", "ThinkerProposal",
    "RuntimeMetrics", "DeltaRecord", "PESComponents",
    "TaskType", "Objective", "DataSource",
    "KoiMonitor", "KalmanFilter1D", "DeadbandController", "SLOState",
    "KoiRefinement", "compute_pes",
    "SwapArbiter", "RunningJob",
    "ExplorationManager",
    "TandemMetricsAPISource", "VLLMPrometheusSource",
    "koi_deploy", "parse_user_request", "fetch_resource_map",
]
