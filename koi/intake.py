"""
koi/intake.py — Front-door: natural language → JobRequest + live cluster → ResourceMap.

This is the missing piece between "user says something" and the Koi placement pipeline.

Two responsibilities:
  1. parse_user_request(text)      NL string → JobRequest via Claude
  2. fetch_resource_map(url, ...)  GET getmeresourcemap endpoint → ResourceMap
  3. koi_deploy(text, ...)         combines both and calls KoiPlacement.decide()

Usage:
    from koi.intake import koi_deploy

    decision = koi_deploy(
        "deploy meta-llama/Llama-3-70B for online serving, 50 users, TPOT under 40ms",
        resource_map_url="http://your-cluster/getmeresourcemap",
        api_key="sk-ant-...",
    )
    print(decision.display_summary())

ResourceMap endpoint contract:
    GET /getmeresourcemap  →  JSON with one of two shapes:

    Shape A — list of GPU resources (simplest):
    [
      {
        "gpu_type": "H100",
        "instance_type": "p5.48xlarge",
        "gpus_per_instance": 8,
        "total_gpus": 64,
        "allocated_gpus": 16,
        "cost_per_instance_hour_usd": 98.32,
        "gpu_memory_gb": 80,
        "region": "us-east-1",
        "interconnect": "NVLink"
      },
      ...
    ]

    Shape B — wrapper object:
    {
      "vpc_id": "vpc-abc123",
      "region": "us-east-1",
      "resources": [ ...same as Shape A... ]
    }

    Any missing field falls back to a sensible default so a partial response
    still produces a usable ResourceMap.
"""

import json
import os
from typing import Any, Dict, List, Optional

import anthropic
import requests

from koi.placement import KoiPlacement
from koi.schemas import (
    GPUResource,
    JobRequest,
    Objective,
    PlacementDecision,
    ResourceMap,
    TaskType,
)

# ---------------------------------------------------------------------------
# GPU memory lookup — used when the endpoint omits gpu_memory_gb
# ---------------------------------------------------------------------------

_GPU_MEMORY_GB: Dict[str, float] = {
    "H100": 80.0, "H100_SXM": 80.0,
    "H200": 141.0,
    "A100": 80.0, "A100-80GB": 80.0, "A100-40GB": 40.0,
    "L40S": 48.0,
    "A10G": 24.0,
    "L4": 24.0,
    "B200": 192.0, "GB200": 192.0,
}

_GPU_INTERCONNECT: Dict[str, str] = {
    "H100": "NVLink", "H100_SXM": "NVLink",
    "H200": "NVLink",
    "A100": "NVLink", "A100-80GB": "NVLink", "A100-40GB": "NVLink",
    "L40S": "PCIe",
    "A10G": "PCIe",
    "L4": "PCIe",
    "B200": "NVLink", "GB200": "NVLink",
}


# ---------------------------------------------------------------------------
# 1. Live cluster → ResourceMap
# ---------------------------------------------------------------------------

def fetch_resource_map(
    url: str,
    vpc_id: Optional[str] = None,
    region: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
) -> ResourceMap:
    """
    Call your getmeresourcemap endpoint and convert the response to a ResourceMap.

    Tolerant of missing fields — uses GPU-type defaults for memory/interconnect
    and zeros for allocated_gpus if not provided.

    Args:
        url:     Full URL to your getmeresourcemap endpoint.
        vpc_id:  Override VPC ID (taken from response if present).
        region:  Override region (taken from response if present).
        headers: Optional auth / custom headers.
        timeout: HTTP timeout in seconds.
    """
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return _parse_resource_map_response(data, vpc_id=vpc_id, region=region)


def _parse_resource_map_response(
    data: Any,
    vpc_id: Optional[str] = None,
    region: Optional[str] = None,
) -> ResourceMap:
    """
    Convert raw endpoint JSON → ResourceMap.
    Handles Shape A (plain list), Shape B (wrapper object),
    and Shape C (Orca's {instances[], quotas[]} format).
    """
    # Shape C — Orca's {instances[], quotas[]} format
    if isinstance(data, dict) and "instances" in data and "quotas" in data:
        return _parse_orca_resource_format(data, vpc_id=vpc_id, region=region)

    if isinstance(data, list):
        raw_resources = data
        resolved_vpc = vpc_id or "vpc-unknown"
        resolved_region = region or "us-east-1"
    elif isinstance(data, dict):
        raw_resources = data.get("resources", [data])  # fallback: treat whole object as one resource
        resolved_vpc = vpc_id or data.get("vpc_id") or data.get("id") or "vpc-unknown"
        resolved_region = region or data.get("region") or "us-east-1"
    else:
        raise ValueError(f"Unexpected resource map response type: {type(data)}")

    resources: List[GPUResource] = []
    for r in raw_resources:
        gpu_type = (
            r.get("gpu_type")
            or r.get("gpu_model")
            or r.get("gpu")
            or r.get("accelerator")
            or "UNKNOWN"
        )
        gpu_type_upper = gpu_type.upper()

        mem_gb = (
            _coerce_float(r.get("gpu_memory_gb"))
            or _coerce_float(r.get("vram_gb"))
            or _coerce_float(r.get("memory_gb"))
            or _GPU_MEMORY_GB.get(gpu_type_upper)
            or 40.0  # conservative fallback
        )

        interconnect = (
            r.get("interconnect")
            or r.get("network")
            or r.get("nvlink")
            or _GPU_INTERCONNECT.get(gpu_type_upper)
            or "PCIe"
        )

        gpus_per_instance = int(
            r.get("gpus_per_instance")
            or r.get("gpus_per_node")
            or r.get("gpu_per_instance")
            or 8
        )

        total_gpus = int(
            r.get("total_gpus")
            or r.get("gpu_count")
            or r.get("available_gpus")  # if endpoint only sends available
            or r.get("count")
            or gpus_per_instance
        )

        allocated_gpus = int(
            r.get("allocated_gpus")
            or r.get("used_gpus")
            or r.get("in_use_gpus")
            or 0
        )

        cost = (
            _coerce_float(r.get("cost_per_instance_hour_usd"))
            or _coerce_float(r.get("cost_per_hour"))
            or _coerce_float(r.get("price_per_hour"))
            or _coerce_float(r.get("hourly_cost"))
            or 0.0
        )

        instance_type = (
            r.get("instance_type")
            or r.get("instance")
            or r.get("machine_type")
            or f"unknown-{gpu_type.lower()}"
        )

        resource_region = r.get("region") or resolved_region

        resources.append(GPUResource(
            gpu_type=gpu_type,
            instance_type=instance_type,
            gpus_per_instance=gpus_per_instance,
            total_gpus=total_gpus,
            allocated_gpus=allocated_gpus,
            cost_per_instance_hour_usd=cost,
            gpu_memory_gb=mem_gb,
            region=resource_region,
            interconnect=interconnect,
        ))

    if not resources:
        raise ValueError("Resource map endpoint returned no GPU resources.")

    return ResourceMap(
        vpc_id=resolved_vpc,
        region=resolved_region,
        resources=resources,
    )


def _parse_orca_resource_format(
    data: Dict[str, Any],
    vpc_id: Optional[str] = None,
    region: Optional[str] = None,
) -> ResourceMap:
    """
    Convert Orca's {instances[], quotas[]} → ResourceMap (Shape C).

    Orca returns raw AWS instance catalog + per-region quota info.
    This function joins them: for each instance type, find the best region
    (most available vCPU for that quota family), compute max launchable
    instances from quota headroom, and emit GPUResource entries.

    This is the only place that understands AWS quota semantics.
    Koi is stateless w.r.t. quota — every /decide call gets fresh data.
    """
    instances = data.get("instances", [])
    quotas = data.get("quotas", [])
    resolved_vpc = vpc_id or "vpc-unknown"

    resources: List[GPUResource] = []
    for inst in instances:
        family = inst.get("quota_family", "")
        vcpus = int(inst.get("vcpus", 0))
        if vcpus <= 0:
            continue

        # Find best region for this family (most available on-demand vCPU)
        family_quotas = [
            q for q in quotas
            if q.get("family") == family and q.get("market") == "on_demand"
        ]
        best = max(
            family_quotas,
            key=lambda q: q.get("baseline_vcpus", 0) - q.get("used_vcpus", 0),
            default=None,
        )
        if not best or best.get("baseline_vcpus", 0) <= 0:
            continue

        available_vcpu = best["baseline_vcpus"] - best.get("used_vcpus", 0)
        max_instances = available_vcpu // vcpus
        gpus_per_instance = int(inst.get("gpus_per_instance", 1))
        total_gpus = max_instances * gpus_per_instance
        if total_gpus <= 0:
            continue

        gpu_type = inst.get("gpu_type", "UNKNOWN")
        gpu_type_upper = gpu_type.upper()

        gpu_memory_gb = (
            _coerce_float(inst.get("gpu_memory_gb"))
            or _GPU_MEMORY_GB.get(gpu_type_upper)
            or 40.0
        )
        interconnect = (
            inst.get("interconnect")
            or _GPU_INTERCONNECT.get(gpu_type_upper)
            or "PCIe"
        )
        cost = _coerce_float(inst.get("cost_per_instance_hour_usd")) or 0.0
        instance_type = inst.get("instance_type", f"unknown-{gpu_type.lower()}")
        best_region = best.get("region") or region or "us-east-1"

        resources.append(GPUResource(
            gpu_type=gpu_type,
            instance_type=instance_type,
            gpus_per_instance=gpus_per_instance,
            total_gpus=total_gpus,
            allocated_gpus=0,  # Orca already accounts for used vCPUs
            cost_per_instance_hour_usd=cost,
            gpu_memory_gb=gpu_memory_gb,
            region=best_region,
            interconnect=interconnect,
        ))

    if not resources:
        raise ValueError("Orca resource map yielded no GPU resources (all quota families exhausted).")

    # Use the first resource's region as the map-level region
    resolved_region = region or (resources[0].region if resources else "us-east-1")

    return ResourceMap(
        vpc_id=resolved_vpc,
        region=resolved_region,
        resources=resources,
    )


# ---------------------------------------------------------------------------
# 2. Natural language → JobRequest
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """\
You are a placement request parser for an LLM inference scheduler.
Extract deployment parameters from the user's message and return ONLY valid JSON.

Rules:
- model_name: HuggingFace model ID if given, else infer from common names
  (e.g. "llama 70b" → "meta-llama/Llama-3-70B-Instruct",
        "qwen 72b"  → "Qwen/Qwen2.5-72B-Instruct",
        "mistral 7b"→ "mistralai/Mistral-7B-Instruct-v0.3",
        "deepseek r1" → "deepseek-ai/DeepSeek-R1")
- task_type: "batch" if the user mentions dataset / file / rows / offline;
             "online" if they mention serving / endpoint / users / API / latency / TPOT / TTFT
- avg_input_tokens: default 512 if not specified
- avg_output_tokens: default 256 if not specified (use 2048+ for reasoning/chain-of-thought)
- objective: "cheapest" | "fastest" | "balanced" (default "balanced")
- num_requests: only for batch (integer, null otherwise)
- slo_deadline_hours: only for batch (float, null otherwise)
- expected_concurrency: only for online (integer, null otherwise)
- slo_tpot_ms: time-per-output-token SLO in ms, only for online (float, null otherwise)
- slo_ttft_ms: time-to-first-token SLO in ms, only for online (float, null otherwise)
- preferred_gpu_types: list of GPU types if user mentioned specific hardware, else null
- max_total_gpus: integer if user set a GPU cap, else null

Return JSON only, no markdown, no explanation.
"""

_PARSE_SCHEMA = {
    "model_name": str,
    "task_type": str,
    "avg_input_tokens": int,
    "avg_output_tokens": int,
    "objective": str,
    "num_requests": "int|null",
    "slo_deadline_hours": "float|null",
    "expected_concurrency": "int|null",
    "slo_tpot_ms": "float|null",
    "slo_ttft_ms": "float|null",
    "preferred_gpu_types": "list|null",
    "max_total_gpus": "int|null",
}


def parse_user_request(
    text: str,
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
) -> JobRequest:
    """
    Parse a natural language deployment request into a JobRequest.

    Uses Claude Haiku (cheap, fast) for the extraction — the heavy reasoning
    happens later in the ensemble with Opus.

    Args:
        text:    User's natural language request.
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        model:   Claude model to use for parsing (default: Haiku for speed).
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)

    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_PARSE_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = message.content[0].text.strip()

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse Claude's response as JSON.\n"
            f"Response was: {raw}\n"
            f"Error: {e}"
        )

    return _dict_to_job_request(parsed)


def _dict_to_job_request(d: Dict[str, Any]) -> JobRequest:
    """Convert the parsed JSON dict → JobRequest, filling defaults."""
    task_type_str = str(d.get("task_type", "batch")).lower()
    task_type = TaskType.ONLINE if "online" in task_type_str else TaskType.BATCH

    objective_str = str(d.get("objective", "balanced")).lower()
    if objective_str == "cheapest":
        objective = Objective.CHEAPEST
    elif objective_str == "fastest":
        objective = Objective.FASTEST
    else:
        objective = Objective.BALANCED

    kwargs: Dict[str, Any] = dict(
        model_name=str(d.get("model_name", "unknown")),
        task_type=task_type,
        avg_input_tokens=int(d.get("avg_input_tokens") or 512),
        avg_output_tokens=int(d.get("avg_output_tokens") or 256),
        objective=objective,
    )

    if d.get("num_requests"):
        kwargs["num_requests"] = int(d["num_requests"])
    if d.get("slo_deadline_hours"):
        kwargs["slo_deadline_hours"] = float(d["slo_deadline_hours"])
    if d.get("expected_concurrency"):
        kwargs["expected_concurrency"] = int(d["expected_concurrency"])
    if d.get("slo_tpot_ms"):
        kwargs["slo_tpot_ms"] = float(d["slo_tpot_ms"])
    if d.get("slo_ttft_ms"):
        kwargs["slo_ttft_ms"] = float(d["slo_ttft_ms"])
    if d.get("preferred_gpu_types"):
        kwargs["preferred_gpu_types"] = list(d["preferred_gpu_types"])
    if d.get("max_total_gpus"):
        kwargs["max_total_gpus"] = int(d["max_total_gpus"])

    return JobRequest(**kwargs)


# ---------------------------------------------------------------------------
# 3. Unified entry point: text + live cluster → PlacementDecision
# ---------------------------------------------------------------------------

def koi_deploy(
    user_text: str,
    resource_map_url: str,
    api_key: Optional[str] = None,
    perfdb_path: str = "./perfdb",
    data_dir: str = "./data",
    llm_model: str = "claude-opus-4-6",
    resource_map_headers: Optional[Dict[str, str]] = None,
    resource_map_timeout: int = 10,
    parse_model: str = "claude-haiku-4-5-20251001",
) -> PlacementDecision:
    """
    Full front-door: user text + live cluster endpoint → PlacementDecision.

    Steps:
      1. Call getmeresourcemap → ResourceMap (live GPU inventory)
      2. Parse user text → JobRequest (via Claude Haiku)
      3. KoiPlacement.decide(request, resource_map) → PlacementDecision

    Args:
        user_text:             Natural language request, e.g.
                               "deploy llama 70b online, 50 users, TPOT < 40ms"
        resource_map_url:      Your getmeresourcemap endpoint URL.
        api_key:               Anthropic API key (or set ANTHROPIC_API_KEY).
        perfdb_path:           Path to directory containing benchmark CSVs.
        data_dir:              Path to evolutionary DB (delta store, policy memory).
        llm_model:             Claude model for the ensemble (default: opus).
        resource_map_headers:  Optional headers for the resource map HTTP call.
        resource_map_timeout:  Timeout for the resource map HTTP call.
        parse_model:           Claude model for NL parsing (default: haiku).

    Returns:
        PlacementDecision — call .display_summary() for a human-readable report.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    print("[Koi] Fetching live resource map...")
    resource_map = fetch_resource_map(
        url=resource_map_url,
        headers=resource_map_headers,
        timeout=resource_map_timeout,
    )
    print(
        f"[Koi] Resource map: {resource_map.vpc_id} | "
        f"{resource_map.total_available_gpus()} GPUs available across "
        f"{len(resource_map.resources)} GPU type(s): "
        f"{resource_map.available_gpu_types()}"
    )

    print(f"[Koi] Parsing request: {user_text!r}")
    request = parse_user_request(user_text, api_key=key, model=parse_model)
    print(
        f"[Koi] Parsed → model={request.model_name} "
        f"task={request.task_type.value} "
        f"in={request.avg_input_tokens} out={request.avg_output_tokens} "
        f"objective={request.objective.value}"
    )

    koi = KoiPlacement(
        api_key=key,
        perfdb_path=perfdb_path,
        data_dir=data_dir,
        llm_model=llm_model,
    )
    return koi.decide(request, resource_map)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
