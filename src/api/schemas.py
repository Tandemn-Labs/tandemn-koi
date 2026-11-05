from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


PositiveInt = Annotated[int, Field(gt=0)]
GpuMemoryUtilization = Annotated[float, Field(gt=0, le=1)]
Probability = Annotated[float, Field(ge=0, le=1)]


class PerGpuIntOverride(RequestModel):
    gpu_type: str = Field(min_length=1)
    value: PositiveInt


class PerGpuStringOverride(RequestModel):
    gpu_type: str = Field(min_length=1)
    value: str = Field(min_length=1)


class JobEngineArguments(RequestModel):
    max_model_len: PositiveInt | None = None
    gpu_mem_util: GpuMemoryUtilization | None = None

    # Scalar applies to any GPU; a list overrides only the listed GPU types.
    max_num_seq: PositiveInt | list[PerGpuIntOverride] | None = None
    max_num_batched_tokens: PositiveInt | list[PerGpuIntOverride] | None = None
    block_size: PositiveInt | list[PerGpuIntOverride] | None = None
    kvcache_dtype: str | list[PerGpuStringOverride] | None = None

    attn_backend: str | None = None
    activation_quantization_method: str | None = None
    activation_dtype: str | None = None

    prefix_cache_enabled: bool | None = None
    chunked_prefill_enable: bool | None = None
    lmcache_enabled: bool | None = None

    spec_decoding_enabled: bool | None = None
    draft_model_id: str | None = None
    spec_decoding_method: str | None = None
    num_speculative_tokens: PositiveInt | None = None
    spec_acceptance_threshold: Probability | None = None

    cuda_graph_enabled: bool | None = None
    torch_compile_enabled: bool | None = None

    chunk_size: PositiveInt | None = None
    max_chunked_steps_per_request: PositiveInt | None = None

    # Exact argv tokens; do not join into or execute as a shell command.
    extra_args: list[str] = Field(default_factory=list)


class JobIntent(RequestModel):
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
    scheduling_policy: Literal["fcfs", "lcfs", "wspt"] | None = None
    router_policy: Literal["kv_aware", "round_robin"] | None = None


class JobSpec(RequestModel):
    model_id: str
    job_intent: JobIntent = Field(default_factory=JobIntent)
    engine_config: JobEngineArguments | None = None


class JobSubmission(RequestModel):
    kind: Literal["online", "batch"]
    spec: JobSpec
