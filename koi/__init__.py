"""
koi — Evolutionary agentic cluster management for batched LLM inference.

v2: Single Claude agent with domain tools, SQLite memory, background monitoring.

Usage:
    from koi import KoiAgent, PerfDB, AgenticMemory
    agent = KoiAgent(perfdb=PerfDB("perfdb/perfdb_all.csv"), memory=AgenticMemory())
    decision = await agent.decide(job_request, resource_map)
"""

from koi.schemas import (
    JobRequest,
    ResourceMap,
    GPUResource,
    PlacementConfig,
    EngineConfig,
    PredictedMetrics,
    RuntimeMetrics,
    AgentDecision,
    JobTracker,
    MonitoringStatus,
    MonitoringTrigger,
    TaskType,
    Objective,
    DataSource,
)
from koi.agent import KoiAgent
from koi.monitor import MonitoringLoop
from koi.tools.perfdb import PerfDB
from koi.tools.memory import AgenticMemory
from koi.tools.orca_api import OrcaClient
from koi.tools.physics import GPU_SPECS, ModelFeatures, get_model_features
from koi.tools.resources import parse_orca_resources

__all__ = [
    "KoiAgent", "MonitoringLoop",
    "PerfDB", "AgenticMemory", "OrcaClient",
    "GPU_SPECS", "ModelFeatures", "get_model_features",
    "parse_orca_resources",
    "JobRequest", "ResourceMap", "GPUResource",
    "PlacementConfig", "EngineConfig", "PredictedMetrics",
    "RuntimeMetrics", "AgentDecision", "JobTracker",
    "MonitoringStatus", "MonitoringTrigger",
    "TaskType", "Objective", "DataSource",
]
