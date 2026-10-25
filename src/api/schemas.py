from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobFeatures(RequestModel):
    # Required for online jobs; not used to select batch duration.
    target_p99_ttft_ms: float | None = None
    target_p99_tpot_ms: float | None = None

    # Required for batch jobs. Online jobs default this to 0 during normalization.
    deadline_hrs: float | None = None

    isl_token_avg: float = 4000
    isl_token_min: float = 1
    isl_token_max: float = 8000
    isl_distribution_type: Literal["Constant", "Uniform", "Normal", "LogNormal", "Exponential"] = (
        "LogNormal"
    )

    osl_token_avg: float = 1000
    osl_token_min: float = 1
    osl_token_max: float = 4000
    osl_distribution_type: Literal["Constant", "Uniform", "Normal", "LogNormal", "Exponential"] = (
        "LogNormal"
    )

    # Leave unset to let the server apply the existing request-rate default.
    # If one is supplied, the server derives the other.
    request_arrival_rate: float | None = None
    max_concurrent_streaming: float | None = None

    request_arrival_pattern: Literal["Poisson", "Constant", "Bursty", "Diurnal"] = "Poisson"
    peak_to_mean_ratio: float = 2
    workload_prefix_concentration: float = 0
    multi_turn_ratio: float = 0.5
    shared_prefix_length_avg: float = 500
    # sliding_window_attention: float = 0
    # TDOO - put this in model catalog
    is_session_affinity: bool = True
    total_token_budget: int = 1_000_000  # only for batched
    priority_class: Literal["HIGH", "STANDARD", "LOW"] = "STANDARD"
    preemption_policy: Literal["lifo", "fifo"] = "lifo"
    router_policy: Literal["kv_router"] = "kv_router"


class JobSpec(RequestModel):
    model_id: str
    job_features: JobFeatures = Field(default_factory=JobFeatures)


class JobSubmission(RequestModel):
    kind: Literal["online", "batch"]
    spec: JobSpec
