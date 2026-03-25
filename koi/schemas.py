"""
koi/schemas.py — All Pydantic data models for the Koi placement system.

Data flow:
  JobRequest + ResourceMap
      → Oracle → List[OracleCandidate]
      → Ensemble → List[ThinkerProposal]
      → Judge → PlacementDecision

Refinement (later):
  RuntimeMetrics → DeltaRecord → PESComponents → policy memory
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    VPC_CORRECTED = "vpc_corrected"     # base + per-VPC learned delta


# ---------------------------------------------------------------------------
# Input: Job Request (from Tandem CLI)
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    """
    Incoming job request from `tandem launch ...`

    Examples:
        tandem launch Qwen/Qwen2.5-72B-Instruct dataset.jsonl --hours 8 --cheapest
        tandem launch meta-llama/Llama-3-70B --online --users 50 --tpot-slo 35
    """
    job_id: str = Field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")
    model_name: str                                   # HuggingFace model ID
    task_type: TaskType = TaskType.BATCH

    # Workload characterization
    avg_input_tokens: int = Field(gt=0)
    avg_output_tokens: int = Field(gt=0)
    num_requests: Optional[int] = None               # batch: row count
    expected_concurrency: Optional[int] = None       # online: expected concurrent users

    # SLOs (at least one must be set)
    slo_deadline_hours: Optional[float] = None       # batch: total time budget
    slo_tpot_ms: Optional[float] = None              # online: time per output token
    slo_ttft_ms: Optional[float] = None              # online: time to first token

    # Objective
    objective: Objective = Objective.CHEAPEST

    # Optional user constraints
    preferred_gpu_types: Optional[List[str]] = None  # e.g. ["L40S", "A100"]
    max_total_gpus: Optional[int] = None
    region: Optional[str] = None                     # override VPC default

    @property
    def total_tokens(self) -> Optional[int]:
        if self.num_requests is None:
            return None
        return self.num_requests * (self.avg_input_tokens + self.avg_output_tokens)

    @property
    def prefill_decode_ratio(self) -> float:
        return self.avg_input_tokens / max(self.avg_output_tokens, 1)


# ---------------------------------------------------------------------------
# Input: Resource Map (VPC GPU inventory)
# ---------------------------------------------------------------------------

class GPUResource(BaseModel):
    """Single GPU type available in the VPC."""
    gpu_type: str                        # canonical: "L40S", "A100", "H100", "A10G"
    instance_type: str                   # AWS: "g6e.12xlarge"
    gpus_per_instance: int
    total_gpus: int                      # total in quota (not just available)
    allocated_gpus: int = 0
    cost_per_instance_hour_usd: float
    gpu_memory_gb: float                 # VRAM per GPU
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
# Oracle outputs: Candidates with predictions
# ---------------------------------------------------------------------------

class EngineConfig(BaseModel):
    """vLLM / engine-specific launch configuration."""
    tensor_parallel_size: int
    pipeline_parallel_size: int
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    enable_chunked_prefill: bool = False
    enable_lmcache: bool = False
    spec_decode: bool = False
    quantization: Optional[str] = None  # "fp8", "awq", etc.

    def to_vllm_args(self) -> str:
        """Render as vLLM CLI args string for display."""
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
    """Complete hardware + parallelism + engine placement specification."""
    gpu_type: str                   # "L40S"
    instance_type: str              # "g6e.12xlarge"
    num_gpus: int                   # total GPUs for this job
    num_instances: int              # number of cloud instances
    tp: int                         # tensor parallelism
    pp: int                         # pipeline parallelism
    dp: int                         # data parallelism (replicas)
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


class PredictedMetrics(BaseModel):
    """Performance predictions from the Oracle for a specific PlacementConfig."""
    throughput_tokens_per_sec: float        # total tokens/sec (all replicas)
    throughput_per_gpu_tokens_per_sec: float

    # Batch-specific
    estimated_runtime_hours: Optional[float] = None
    total_cost_usd: Optional[float] = None

    # Per-replica serving latency estimates
    tpot_ms: Optional[float] = None
    ttft_ms: Optional[float] = None

    # Cost
    cost_per_hour_usd: float
    cost_per_1m_tokens_usd: Optional[float] = None

    # Hardware utilization
    estimated_gpu_mem_used_gb: Optional[float] = None
    estimated_gpu_mem_pct: Optional[float] = None

    # Prediction metadata
    confidence: float = Field(ge=0.0, le=1.0)
    data_source: DataSource
    nearest_db_entry: Optional[str] = None   # human-readable description of what we matched


class OracleCandidate(BaseModel):
    """A feasible config + its predicted metrics, output of the Oracle."""
    config: PlacementConfig
    metrics: PredictedMetrics
    meets_slo: bool = True
    slo_margin_pct: Optional[float] = None   # how much headroom vs SLO (positive = good)
    feasibility_notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Ensemble: Thinker proposals + Judge output
# ---------------------------------------------------------------------------

class ThinkerProposal(BaseModel):
    """
    One LLM's freely-proposed config + causal hypothesis.
    The LLM proposes a config from scratch (not picked from a list).
    The Oracle validates and estimates metrics AFTER the proposal.
    """
    thinker_id: str                        # "LLM1", "LLM2", "LLM3"
    directive: str                         # which exploration directive was given
    proposed_config: PlacementConfig
    oracle_estimate: Optional[PredictedMetrics] = None  # filled in post-proposal
    hypothesis: str                        # why this config should beat the frontier
    mechanism: str                         # physical/architectural principle supporting it
    evidence: str                          # past runs or domain knowledge cited
    falsification_condition: str           # what outcome would prove this hypothesis wrong
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str                         # full chain-of-thought
    guardrail_rejections: List[str] = Field(default_factory=list)  # any failed attempts


class ExplorationQueueEntry(BaseModel):
    """A proposal queued for future exploration (not deployed now)."""
    proposal: ThinkerProposal
    priority: str                 # "high", "medium", "low"
    reason: str                   # why it's worth testing later
    suggested_job_constraints: str = ""  # e.g. "low-priority, SLO headroom >= 40%"


class JudgeDecision(BaseModel):
    """
    Judge's synthesis of all thinker proposals.
    Can pick one proposal OR synthesize a novel config from combined reasoning.
    Also produces an exploration queue for future probing.
    """
    # Deployment config: either one of the proposals or a novel synthesis
    decision_source: str          # "proposal_0", "proposal_1", "proposal_2", or "synthesis"
    deployment_config: PlacementConfig
    deployment_oracle_estimate: Optional[PredictedMetrics] = None

    # If synthesis: what did the judge combine and why
    synthesis_reasoning: str = ""

    # Other proposals queued for exploration (not deployed now)
    exploration_queue: List[ExplorationQueueEntry] = Field(default_factory=list)

    # Which proposal had the most novel/interesting hypothesis worth tracking
    most_novel_hypothesis_thinker: str = ""
    most_novel_hypothesis_summary: str = ""

    # Overall
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    agreement: str = "partial"    # "full", "partial", "split"


class DiagnosisProposal(BaseModel):
    """
    Re-placement proposal generated from monitoring data.
    Includes a causal diagnosis of why the current config failed.
    """
    thinker_id: str
    failure_mode: str             # what the monitoring trace shows is failing
    causal_rule: str              # generalized rule extracted (for causal library)
    proposed_config: PlacementConfig
    oracle_estimate: Optional[PredictedMetrics] = None
    repair_hypothesis: str        # why new config fixes the diagnosed failure
    expected_improvement: str     # quantified expectation
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class PlacementDecision(BaseModel):
    """
    Final output of the Koi placement system.
    This is what gets returned to the Tandem CLI.
    """
    job_id: str
    model_name: str

    # The chosen placement
    recommendation: PlacementConfig
    predicted_metrics: PredictedMetrics

    # Explanation
    reasoning: str
    confidence: float

    # All thinker proposals + judge decision (for transparency / logging)
    thinker_proposals: List[ThinkerProposal] = Field(default_factory=list)
    judge_decision: Optional[JudgeDecision] = None

    # Proposals queued for future exploration
    exploration_queue: List[ExplorationQueueEntry] = Field(default_factory=list)

    # Metadata
    decision_timestamp: datetime = Field(default_factory=datetime.utcnow)
    oracle_candidates_evaluated: int = 0
    total_llm_calls: int = 0
    is_reconfig: bool = False          # True if triggered by monitoring, not fresh placement
    triggered_by: str = "initial"      # "initial", "monitoring_soft", "monitoring_hard"

    def display_summary(self) -> str:
        """Human-readable summary for CLI output."""
        rec = self.recommendation
        met = self.predicted_metrics
        lines = [
            f"\n{'='*60}",
            f"  KOI PLACEMENT DECISION — {self.job_id}",
            f"{'='*60}",
            f"  Model    : {self.model_name}",
            f"  Placement: {rec.summary}",
            f"  Region   : {rec.region}",
            f"",
            f"  Parallelism  : TP={rec.tp}  PP={rec.pp}  DP={rec.dp}",
            f"  Throughput   : {met.throughput_tokens_per_sec:.0f} tok/s",
        ]
        if met.estimated_runtime_hours is not None:
            lines.append(f"  Est. Runtime : {met.estimated_runtime_hours:.2f} hours")
        if met.total_cost_usd is not None:
            lines.append(f"  Est. Cost    : ${met.total_cost_usd:.2f}")
        if met.tpot_ms is not None:
            lines.append(f"  Est. TPOT    : {met.tpot_ms:.1f} ms")
        lines += [
            f"  Confidence   : {self.confidence:.0%}  (source: {met.data_source.value})",
            f"",
            f"  vLLM Args:",
            f"  {rec.engine_config.to_vllm_args()}",
            f"",
            f"  Reasoning: {self.reasoning[:300]}{'...' if len(self.reasoning) > 300 else ''}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Refinement schemas (Phase 2 — monitor + learning)
# ---------------------------------------------------------------------------

class RuntimeMetrics(BaseModel):
    """Live metrics snapshot from a running job (from vLLM /metrics or CloudWatch)."""
    job_id: str
    timestamp: datetime
    throughput_tokens_per_sec: float
    tpot_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    gpu_utilization_pct: float
    gpu_memory_used_gb: float
    gpu_memory_bw_pct: Optional[float] = None
    concurrent_requests: Optional[int] = None
    queue_depth: Optional[int] = None


class DeltaRecord(BaseModel):
    """
    Prediction error record stored in the per-VPC delta store.
    This is the ground truth dataset that the RAG correction layer learns from.
    """
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vpc_id: str
    job_id: str

    # Config context
    model_name: str
    gpu_type: str
    tp: int
    pp: int
    dp: int
    avg_input_tokens: int
    avg_output_tokens: int
    task_type: str

    # Predictions vs actuals
    predicted_throughput_tps: float
    actual_throughput_tps: float
    predicted_tpot_ms: Optional[float] = None
    actual_tpot_ms: Optional[float] = None

    # Deltas
    delta_throughput_pct: float   # (actual - predicted) / predicted × 100
    delta_tpot_ms: Optional[float] = None

    # Cluster state at time of run (for learning noise-neighbor patterns)
    cluster_gpu_utilization_pct: Optional[float] = None

    # Data source of the original prediction
    prediction_data_source: str

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PESComponents(BaseModel):
    """
    Placement Efficiency Score for a completed job.
    CER: cheapest known SLO-meeting config / what we paid
    PER: actual throughput / roofline peak throughput
    SS:  time_in_final_config / total_job_time
    """
    job_id: str
    cer: float = Field(ge=0.0, le=1.0, description="Cost Efficiency Ratio")
    per: float = Field(ge=0.0, le=1.0, description="Physical Efficiency Ratio")
    ss: float = Field(ge=0.0, le=1.0, description="Stability Score")
    composite: float = Field(ge=0.0, le=1.0, description="Weighted composite PES")
    task_type: str

    # Weights used (differ by task_type)
    alpha: float  # CER weight
    beta: float   # PER weight
    gamma: float  # SS weight

    # Context
    num_reconfigurations: int = 0
    slo_violations_count: int = 0
    total_job_hours: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
