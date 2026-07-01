"""Build rank-level deployment X from Store snapshots and hardware catalog.

Telemetry owns runtime V/Y. This module owns deployment-time X and is strict:
missing rank identity, mixed replica shapes, missing resources, or missing
hardware facts are contract errors rather than values to guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.core.models import EnvLabel

RankKey = tuple[str, str]

_X_SKIP = {"env", "rank_id", "replica_index", "mechanism_id"}
_LOAD_FIELDS = ("request_arrival_rate", "total_token_budget")
_GPU_FIELDS = (
    "gpu_bandwidth_gbps",
    "gpu_tflops_fp16",
    "cuda_compute_capability",
    "gpu_generation",
    "nvlink_bandwidth_gbps",
    "pcie_bandwidth_gbps",
    "gpu_watts",
)
_ALIASES = {
    "deadline_hrs": ("deadline_hrs", "deadline_hours"),
    "isl_token_avg": ("isl_token_avg", "input_len_tokens_avg"),
    "isl_token_min": ("isl_token_min", "input_len_tokens_min"),
    "isl_token_max": ("isl_token_max", "input_len_tokens_max"),
    "osl_token_avg": ("osl_token_avg", "output_len_tokens_avg"),
    "osl_token_min": ("osl_token_min", "output_len_tokens_min"),
    "osl_token_max": ("osl_token_max", "output_len_tokens_max"),
}


@dataclass
class RankDeployment:
    """Deployment X for one homogeneous rank of chain replicas."""

    job_id: str
    rank_id: str
    env_label: EnvLabel
    x: dict[str, object]


@dataclass
class DeploymentXIndex:
    """Lookup table from telemetry rank identity to deployment X."""

    by_rank: dict[RankKey, RankDeployment]

    def resolve(self, job_id: str, rank_id: str | None = None) -> RankDeployment:
        """Resolve deployment X by required ``(job_id, rank_id)``."""
        if rank_id is None:
            raise ValueError("rank_id is required to resolve deployment X")
        key = (str(job_id), str(rank_id))
        if key not in self.by_rank:
            raise KeyError(f"deployment X missing for job_id={key[0]!r}, rank_id={key[1]!r}")
        return self.by_rank[key]


def build_deployment_x_index(
    snapshot: Any,
    *,
    hardware_catalog: dict[str, Any],
    x_fields: list[str] | tuple[str, ...],
) -> DeploymentXIndex:
    """Return the rank-level X index consumed by S2 evidence creation."""
    if not hardware_catalog:
        raise ValueError("hardware catalog is required to build deployment X")
    if not x_fields:
        raise ValueError("candidate graph X fields are required")

    resources = dict(snapshot.resources_summary())
    catalog = _catalog_by_instance(hardware_catalog)
    by_rank: dict[RankKey, RankDeployment] = {}

    for job in snapshot.active_jobs_summary():
        job_id = str(job["job_id"])
        groups = _groups_by_rank(job["active_chains"])
        total_replicas = sum(len(chains) for chains in groups.values())
        for rank_id, chains in groups.items():
            by_rank[(job_id, rank_id)] = _rank_deployment(
                job=job,
                job_id=job_id,
                rank_id=rank_id,
                chains=chains,
                total_replicas=total_replicas,
                resources=resources,
                catalog=catalog,
                x_fields=x_fields,
            )
    return DeploymentXIndex(by_rank)


def _groups_by_rank(chains: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group chain replicas by explicit rank id."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for chain in chains:
        shape = dict(chain["shape_json"])
        rank_id = shape.get("rank_id")
        if not rank_id:
            raise ValueError(f"chain {chain.get('chain_id')!r} missing rank_id")
        groups.setdefault(str(rank_id), []).append(chain)
    return groups


def _rank_deployment(
    *,
    job: dict[str, Any],
    job_id: str,
    rank_id: str,
    chains: list[dict[str, Any]],
    total_replicas: int,
    resources: dict[str, Any],
    catalog: dict[tuple[str, str, str], dict[str, Any]],
    x_fields: list[str] | tuple[str, ...],
) -> RankDeployment:
    """Assemble, enrich, derive, and filter X for one rank."""
    shape = dict(chains[0]["shape_json"])
    env = _env(chains[0])
    replica_count = len(chains)
    for chain in chains[1:]:
        if _env(chain) != env:
            raise ValueError(f"rank {rank_id!r} has mixed env labels")

    job_values = _job_x(job)
    x: dict[str, Any] = {
        **job_values,
        "market": env[0],
        "cloud": env[1],
        "region": env[2],
        "gpu_type": env[4],
        **{key: value for key, value in shape.items() if key not in _X_SKIP},
    }

    pool = _resource_pool(resources, env, str(x["instance_type"]))
    x["interconnect_type"] = pool["fabric_type"]

    hardware = catalog[(env[1], env[2], str(x["instance_type"]))]
    x.update(_hardware_x(hardware, env[4]))
    _derive_x(x)
    _allocate_load_x(x, job_values, shape, replica_count, total_replicas)

    return RankDeployment(
        job_id=job_id,
        rank_id=rank_id,
        env_label=env,
        x=_project_x(x, x_fields),
    )


def _job_x(job: dict[str, Any]) -> dict[str, Any]:
    """Flatten job spec/profile fields that can contribute to X."""
    spec = dict(job.get("spec_json") or {})
    values: dict[str, Any] = {}
    for source in (spec, spec.get("features"), spec.get("job_features"), job.get("job_features")):
        if isinstance(source, dict):
            values.update(source)
            for nested in ("model_profile", "model_config", "workload_profile", "slo"):
                if isinstance(source.get(nested), dict):
                    values.update(source[nested])
    for canonical, aliases in _ALIASES.items():  # this is just aliasing the names
        for alias in aliases:
            if alias in values:
                values[canonical] = values[alias]
                break
    return values


def _env(chain: dict[str, Any]) -> EnvLabel:
    """Resolve the five-part Koi environment label for a deployed chain."""
    shape = dict(chain["shape_json"])
    target_env = _parse_env(chain["target_node"]) if chain.get("target_node") else None
    shape_env = _parse_env(shape["env"])
    if target_env is not None and target_env != shape_env:
        raise ValueError(f"chain {chain.get('chain_id')!r} has conflicting env labels")
    return shape_env


def _parse_env(raw: Any) -> EnvLabel:
    """Parse a strict five-part env label."""
    parts = (
        raw.split("|")
        if isinstance(raw, str)
        else list(raw)
        if isinstance(raw, (list, tuple))
        else []
    )
    if len(parts) != 5 or any(part in (None, "") for part in parts):
        raise ValueError(f"env label must have 5 non-empty parts, got {raw!r}")
    return (str(parts[0]), str(parts[1]), str(parts[2]), str(parts[3]), str(parts[4]))


def _resource_pool(resources: dict[str, Any], env: EnvLabel, instance_type: str) -> dict[str, Any]:
    """Return the resource-map pool backing this rank."""
    for pool in resources["|".join(env)]["pools"]:
        if pool.get("instance_type") == instance_type:
            return dict(pool)
    raise ValueError(f"resource env {'|'.join(env)!r} missing instance_type {instance_type!r}")


def _catalog_by_instance(catalog: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index hardware catalog entries by cloud, region, and instance type."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for region_catalog in catalog["regions"]:
        cloud = region_catalog["cloud"]
        region = region_catalog["region"]
        for instance in region_catalog["instance_types"]:
            instance_type = instance["instance_type"]
            out[(str(cloud), str(region), str(instance_type))] = dict(instance)
    return out


def _hardware_x(hardware: dict[str, Any], gpu_type: str) -> dict[str, object]:
    """Return required hardware X from one catalog instance entry."""
    gpu = _gpu(hardware, gpu_type)
    out = {key: gpu[key] for key in _GPU_FIELDS}
    out["gpu_mem_gb"] = float(gpu["memory_mib_each"]) / 1024.0
    out["gpu_per_node"] = gpu["count"]
    out["internode_bandwidth_gbps"] = _network_bandwidth(hardware)
    return out


def _gpu(hardware: dict[str, Any], gpu_type: str) -> dict[str, Any]:
    """Return the catalog GPU accelerator matching deployment GPU type."""
    for accelerator in hardware["accelerators"]:
        names = {accelerator.get("name"), accelerator.get("canonical_gpu_name")}
        if accelerator.get("kind") == "gpu" and gpu_type in names:
            return dict(accelerator)
    raise ValueError(f"hardware catalog missing GPU accelerator {gpu_type!r}")


def _network_bandwidth(hardware: dict[str, Any]) -> float:
    """Return required peak network bandwidth for inter-node X."""
    return max(float(card["peak_bandwidth_gbps"]) for card in hardware["network"]["network_cards"])


def _derive_x(x: dict[str, Any]) -> None:
    """Compute derived X values from already assembled source fields."""
    _set_ratio(x, "attn_heads_per_kv_head", "num_attn_heads", "num_kv_heads")
    _set_ratio(x, "bandwidth_per_param", "gpu_bandwidth_gbps", "model_params_b")
    _set_ratio(x, "flops_per_param", "gpu_tflops_fp16", "model_params_b")
    gpu_count = x["count"] if "count" in x else x["gpu_count"]
    x["num_nodes_per_chain"] = math.ceil(float(gpu_count) / float(x["gpu_per_node"]))


def _set_ratio(x: dict[str, Any], out: str, numerator: str, denominator: str) -> None:
    """Set a derived ratio only when both source fields are present."""
    if numerator in x and denominator in x:
        bottom = float(x[denominator])
        if bottom == 0.0:
            raise ValueError(f"cannot derive {out}: {denominator} is zero")
        x[out] = float(x[numerator]) / bottom


def _allocate_load_x(
    x: dict[str, Any],
    job_values: dict[str, Any],
    shape: dict[str, Any],
    replica_count: int,
    total_replicas: int,
) -> None:
    """Allocate job-level load fields to a per-replica rank average."""
    share = _rank_traffic_share(shape, replica_count, total_replicas)
    for field in _LOAD_FIELDS:
        if field in job_values:
            x[field] = float(job_values[field]) * share / replica_count


def _rank_traffic_share(shape: dict[str, Any], replica_count: int, total_replicas: int) -> float:
    """Return rank traffic share; multi-rank jobs must declare it."""
    if total_replicas == replica_count:
        return 1.0
    for key in ("rank_traffic_share", "traffic_share"):
        if key in shape:
            return float(shape[key])
    raise ValueError("multi-rank jobs require rank_traffic_share per rank")


def _project_x(x: dict[str, Any], x_fields: list[str] | tuple[str, ...]) -> dict[str, object]:
    """Keep only candidate-graph X fields with non-missing source values."""
    missing = [key for key in x_fields if key not in x or x[key] in (None, "NA")]
    if missing:
        raise ValueError(f"missing deployment X fields: {missing}")
    return {key: x[key] for key in x_fields}
