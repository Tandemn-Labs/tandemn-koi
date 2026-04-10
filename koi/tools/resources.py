"""
koi/tools/resources.py — Fetch live GPU resources from Orca.

Parses Orca's GET /resources response (Shape C: {instances[], quotas[]})
into Koi's ResourceMap. Handles A100-40GB/80GB normalization.
"""

from typing import Any, Dict, List, Optional

from koi.schemas import GPUResource, ResourceMap


# ---------------------------------------------------------------------------
# GPU defaults (used when Orca response is incomplete)
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


def _coerce_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Shape C parser (Orca's {instances[], quotas[]})
# ---------------------------------------------------------------------------

def parse_orca_resources(
    data: Dict[str, Any],
    vpc_id: Optional[str] = None,
    region: Optional[str] = None,
) -> ResourceMap:
    """
    Parse Orca's GET /resources response into a ResourceMap.

    Handles Shape C: {instances[], quotas[]}
    Also handles Shape A (list) and Shape B (wrapper with resources key).
    """
    # Shape C detection
    if isinstance(data, dict) and "instances" in data and "quotas" in data:
        return _parse_shape_c(data, vpc_id, region)

    # Shape A: plain list of GPU resources
    if isinstance(data, list):
        return _parse_shape_a(data, vpc_id, region)

    # Shape B: wrapper with resources key
    if isinstance(data, dict) and "resources" in data:
        return _parse_shape_a(
            data["resources"],
            vpc_id or data.get("vpc_id", "vpc-unknown"),
            region or data.get("region", "us-east-1"),
        )

    raise ValueError(f"Unrecognized resource map format: {type(data)}")


def _parse_shape_c(
    data: Dict[str, Any],
    vpc_id: Optional[str] = None,
    region: Optional[str] = None,
) -> ResourceMap:
    """Parse Orca's {instances[], quotas[]} format."""
    instances = data.get("instances", [])
    quotas = data.get("quotas", [])
    resolved_vpc = vpc_id or data.get("vpc_id", "vpc-unknown")

    # Orca may include live allocation counts from ClusterManager
    orca_allocated = data.get("allocated_gpus", {})  # {"H100": 16, "L40S": 4}

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
        # Normalize generic "A100" to A100-40GB/A100-80GB based on VRAM
        gpu_mem_raw = _coerce_float(inst.get("gpu_memory_gb"))
        if gpu_type.upper() == "A100" and gpu_mem_raw:
            gpu_type = "A100-80GB" if gpu_mem_raw >= 70 else "A100-40GB"

        gpu_type_upper = gpu_type.upper()
        gpu_memory_gb = gpu_mem_raw or _GPU_MEMORY_GB.get(gpu_type_upper, 40.0)
        interconnect = inst.get("interconnect") or _GPU_INTERCONNECT.get(gpu_type_upper, "PCIe")
        cost = _coerce_float(inst.get("cost_per_instance_hour_usd")) or 0.0
        instance_type = inst.get("instance_type", f"unknown-{gpu_type.lower()}")
        best_region = best.get("region") or region or "us-east-1"

        resources.append(GPUResource(
            gpu_type=gpu_type,
            instance_type=instance_type,
            gpus_per_instance=gpus_per_instance,
            total_gpus=total_gpus,
            allocated_gpus=orca_allocated.get(gpu_type, 0),
            cost_per_instance_hour_usd=cost,
            gpu_memory_gb=gpu_memory_gb,
            region=best_region,
            interconnect=interconnect,
        ))

    if not resources:
        raise ValueError("Orca resource map yielded no GPU resources.")

    resolved_region = region or (resources[0].region if resources else "us-east-1")
    return ResourceMap(vpc_id=resolved_vpc, region=resolved_region, resources=resources)


def _parse_shape_a(
    data: List[Dict],
    vpc_id: Optional[str] = None,
    region: Optional[str] = None,
) -> ResourceMap:
    """Parse plain list of GPU resources (Shape A)."""
    resolved_vpc = vpc_id or "vpc-unknown"
    resolved_region = region or "us-east-1"
    resources = []

    for r in data:
        gpu_type = r.get("gpu_type", r.get("gpu_model", "UNKNOWN"))
        gpu_type_upper = gpu_type.upper()
        resources.append(GPUResource(
            gpu_type=gpu_type,
            instance_type=r.get("instance_type", f"unknown-{gpu_type.lower()}"),
            gpus_per_instance=int(r.get("gpus_per_instance", 8)),
            total_gpus=int(r.get("total_gpus", r.get("available_gpus", 8))),
            allocated_gpus=int(r.get("allocated_gpus", 0)),
            cost_per_instance_hour_usd=_coerce_float(r.get("cost_per_instance_hour_usd")) or 0.0,
            gpu_memory_gb=_coerce_float(r.get("gpu_memory_gb")) or _GPU_MEMORY_GB.get(gpu_type_upper, 40.0),
            region=r.get("region", resolved_region),
            interconnect=r.get("interconnect") or _GPU_INTERCONNECT.get(gpu_type_upper, "PCIe"),
        ))

    if not resources:
        raise ValueError("Resource map returned no GPU resources.")

    return ResourceMap(vpc_id=resolved_vpc, region=resolved_region, resources=resources)


# ---------------------------------------------------------------------------
# Agent tool function
# ---------------------------------------------------------------------------

def get_resources(resource_map: ResourceMap) -> str:
    """Format ResourceMap as readable summary for the agent."""
    lines = [f"AVAILABLE RESOURCES ({resource_map.total_available_gpus()} GPUs total):\n"]

    # Group by gpu_type
    by_gpu: Dict[str, List[GPUResource]] = {}
    for r in resource_map.resources:
        by_gpu.setdefault(r.gpu_type, []).append(r)

    for gpu_type, rs in sorted(by_gpu.items(), key=lambda x: -sum(r.total_gpus for r in x[1])):
        total = sum(r.total_gpus for r in rs)
        lines.append(f"  {gpu_type}: {total} GPUs")
        for r in sorted(rs, key=lambda x: -x.total_gpus):
            n_inst = r.total_gpus // r.gpus_per_instance
            lines.append(
                f"    {r.instance_type:20s} {n_inst}x {r.gpus_per_instance}GPU = {r.total_gpus} "
                f"| ${r.cost_per_instance_hour_usd:.2f}/inst/hr "
                f"| {r.gpu_memory_gb}GB VRAM | {r.interconnect} | {r.region}"
            )

    return "\n".join(lines)
