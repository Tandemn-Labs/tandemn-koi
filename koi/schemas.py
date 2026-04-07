"""
koi/schemas.py — Pydantic data models for Koi v2.

Data flow:
  JobRequest + ResourceMap
      → KoiAgent (tool calls: PerfDB, Memory, Physics, Resources)
      → AgentDecision (config + reasoning + confidence)

Monitoring:
  JobTracker (in-memory per job) → MonitoringTrigger → Agent wakes

Learning:
  Job completes → record_outcome() → Memory (decisions, outcomes, rules)
"""

import uuid
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    BATCH = "batch"
    ONLINE = "online"


class Objective(str, Enum):
    CHEAPEST = "cheapest"    # minimize total cost subject to SLO
    FASTEST = "fastest"      # minimize runtime regardless of cost
    BALANCED = "balanced"    # Pareto-optimal cost/performance


class DataSource(str, Enum):
    EXACT_MATCH = "exact_match"          # found in perf DB exactly
    INTERPOLATED = "interpolated"        # nearby I/O lengths, same GPU/TP/PP
    CROSS_GPU = "cross_gpu"             # scaled from different GPU type
    ANALYTICAL = "analytical"           # pure roofline, no nearby data
    MEMORY = "memory"                   # from past production outcomes


class MonitoringStatus(str, Enum):
    WARMING_UP = "warming_up"           # first 5 min, metrics unreliable
    ON_TRACK = "on_track"               # SLO headroom > 30%
    AT_RISK = "at_risk"                 # SLO headroom 10-30%
    FALLING_BEHIND = "falling_behind"   # SLO headroom < 10%
    OVER_PROVISIONED = "over_provisioned"  # headroom > 70% AND elapsed > 20%
    CHAIN_END = "chain_end"             # chain terminated (swap/scale/kill)
    LAUNCH_FAILED = "launch_failed"     # instance failed to start
    COMPLETED = "completed"             # job completed (all chunks done)
    FAILED = "failed"                   # job failed


# ---------------------------------------------------------------------------
# Input: Job Request (from Orca CLI / user)
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    """
    Incoming job request.

    Examples:
        orca deploy Qwen/Qwen2.5-72B-Instruct dataset.jsonl --slo 8 --cheapest
        orca deploy meta-llama/Llama-3-70B dataset.jsonl --slo 1 --fastest
    """
    job_id: str = Field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")
    model_name: str                                   # HuggingFace model ID
    task_type: TaskType = TaskType.BATCH

    # Workload characterization
    avg_input_tokens: int = Field(gt=0)
    avg_output_tokens: int = Field(gt=0)
    num_requests: Optional[int] = None               # batch: row count
    expected_concurrency: Optional[int] = None       # online: expected concurrent users

    # SLOs
    slo_deadline_hours: Optional[float] = None       # batch: total time budget
    slo_tpot_ms: Optional[float] = None              # online: time per output token
    slo_ttft_ms: Optional[float] = None              # online: time to first token

    # Objective
    objective: Objective = Objective.CHEAPEST

    # Optional user constraints
    preferred_gpu_types: Optional[List[str]] = None
    max_total_gpus: Optional[int] = None
    region: Optional[str] = None
    quantization: Optional[str] = None               # "fp8", "int8", or None (fp16 default)

    @property
    def total_tokens(self) -> Optional[int]:
        if self.num_requests is None:
            return None
        return self.num_requests * (self.avg_input_tokens + self.avg_output_tokens)

    @property
    def prefill_decode_ratio(self) -> float:
        return self.avg_input_tokens / max(self.avg_output_tokens, 1)

    @property
    def required_tps(self) -> Optional[float]:
        """Minimum throughput (tok/s) to meet batch SLO."""
        if self.total_tokens is None or self.slo_deadline_hours is None:
            return None
        return self.total_tokens / (self.slo_deadline_hours * 3600)


# ---------------------------------------------------------------------------
# Input: Resource Map (VPC GPU inventory)
# ---------------------------------------------------------------------------

class GPUResource(BaseModel):
    """Single GPU type available in the VPC."""
    gpu_type: str                        # "L40S", "A100-80GB", "A100-40GB", "H100"
    instance_type: str                   # "g6e.12xlarge"
    gpus_per_instance: int
    total_gpus: int
    allocated_gpus: int = 0
    cost_per_instance_hour_usd: float
    gpu_memory_gb: float
    region: str
    interconnect: str                    # "NVLink" or "PCIe"

    @property
    def available_gpus(self) -> int:
        return max(0, self.total_gpus - self.allocated_gpus)

    @property
    def cost_per_gpu_hour_usd(self) -> float:
        return self.cost_per_instance_hour_usd / self.gpus_per_instance


class ResourceMap(BaseModel):
    """Snapshot of all GPU resources in the VPC."""
    vpc_id: str
    region: str
    resources: List[GPUResource]
    snapshot_time: datetime = Field(default_factory=datetime.utcnow)

    def get_resource(self, gpu_type: str) -> Optional[GPUResource]:
        for r in self.resources:
            if r.gpu_type == gpu_type:
                return r
        return None

    def available_gpu_types(self) -> List[str]:
        return [r.gpu_type for r in self.resources if r.available_gpus > 0]

    def total_available_gpus(self) -> int:
        return sum(r.available_gpus for r in self.resources)


# ---------------------------------------------------------------------------
# Engine + Placement Config
# ---------------------------------------------------------------------------

class EngineConfig(BaseModel):
    """vLLM launch configuration."""
    tensor_parallel_size: int
    pipeline_parallel_size: int
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    enable_chunked_prefill: bool = False
    quantization: Optional[str] = None

    def to_vllm_args(self) -> str:
        args = [
            f"--tensor-parallel-size {self.tensor_parallel_size}",
            f"--pipeline-parallel-size {self.pipeline_parallel_size}",
            f"--max-num-seqs {self.max_num_seqs}",
            f"--gpu-memory-utilization {self.gpu_memory_utilization}",
            f"--dtype {self.dtype}",
        ]
        if self.max_model_len:
            args.append(f"--max-model-len {self.max_model_len}")
        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
        if self.quantization:
            args.append(f"--quantization {self.quantization}")
        return " \\\n  ".join(args)


class PlacementConfig(BaseModel):
    """Complete hardware + parallelism specification."""
    gpu_type: str
    instance_type: str
    num_gpus: int
    num_instances: int
    tp: int
    pp: int
    dp: int
    region: str
    engine_config: EngineConfig

    @property
    def gpus_per_replica(self) -> int:
        return self.tp * self.pp

    @property
    def summary(self) -> str:
        return (
            f"{self.num_instances}x {self.instance_type} ({self.gpu_type}) | "
            f"TP={self.tp} PP={self.pp} DP={self.dp} | "
            f"{self.num_gpus} GPUs total"
        )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class PredictedMetrics(BaseModel):
    """Performance predictions for a specific PlacementConfig."""
    throughput_tokens_per_sec: float
    throughput_per_gpu_tokens_per_sec: float

    estimated_runtime_hours: Optional[float] = None
    total_cost_usd: Optional[float] = None

    tpot_ms: Optional[float] = None
    ttft_ms: Optional[float] = None

    cost_per_hour_usd: float
    cost_per_1m_tokens_usd: Optional[float] = None

    confidence: float = Field(ge=0.0, le=1.0)
    data_source: DataSource


# ---------------------------------------------------------------------------
# Runtime metrics (from Orca telemetry)
# ---------------------------------------------------------------------------

class RuntimeMetrics(BaseModel):
    """Live metrics snapshot from a running job."""
    job_id: str
    timestamp: datetime
    throughput_tokens_per_sec: float
    tpot_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    gpu_utilization_pct: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_bw_pct: Optional[float] = None
    gpu_cache_usage_pct: Optional[float] = None
    concurrent_requests: Optional[int] = None
    queue_depth: Optional[int] = None


# ---------------------------------------------------------------------------
# Agent Decision (v2 output — replaces v1 PlacementDecision)
# ---------------------------------------------------------------------------

class AgentDecision(BaseModel):
    """Output of the Koi agent's decide() call."""
    job_id: str
    model_name: str

    # What the agent chose
    config: PlacementConfig
    predicted_tps: float
    predicted_cost_per_hour: float
    predicted_total_cost: Optional[float] = None
    predicted_runtime_hours: Optional[float] = None

    # Agent's reasoning
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    data_source: DataSource = DataSource.ANALYTICAL

    # What informed the decision
    memory_hits: int = 0               # past outcomes found for this model
    perfdb_records_used: int = 0       # PerfDB records consulted
    similar_models_used: List[str] = Field(default_factory=list)

    # Alternatives considered
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    decision_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_model: str = ""
    tool_calls_made: int = 0
    latency_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

class JobTracker(BaseModel):
    """In-memory state for a tracked running job. Links to memory via decision_id."""
    job_id: str
    decision_id: Optional[str] = None   # links to memory.decisions table (current chain)
    group_id: Optional[str] = None      # parent job ID — chains in a group share this
    config: PlacementConfig
    slo_deadline_hours: float
    total_tokens: int
    predicted_tps: float
    started_at: datetime = Field(default_factory=datetime.utcnow)

    # Live state (updated every poll)
    tokens_completed: int = 0
    tokens_remaining: int = 0
    elapsed_hours: float = 0.0
    smoothed_tps: float = 0.0
    projected_eta_hours: float = 0.0
    slo_headroom_pct: float = 100.0
    status: MonitoringStatus = MonitoringStatus.WARMING_UP
    warmup_complete: bool = False

    # GPU health
    gpu_cache_usage: float = 0.0
    gpu_sm_util: float = 0.0
    gpu_mem_bw_util: float = 0.0

    # Replicas
    replica_ids: List[str] = Field(default_factory=list)
    dead_replicas: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class MonitoringTrigger(BaseModel):
    """What gets sent to the agent when the monitor wakes it."""
    trigger_type: MonitoringStatus       # FALLING_BEHIND, OVER_PROVISIONED, COMPLETED, FAILED
    job_id: str
    job_tracker: Dict[str, Any]          # serialized JobTracker state
    recent_metrics: List[Dict[str, Any]] = Field(default_factory=list)  # last 60s of samples
    diagnosis_hint: str = ""             # monitor's arithmetic diagnosis
