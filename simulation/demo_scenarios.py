"""Demo presets and scenario scheduling for the browser-based simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from koi.model_features import _KNOWN_MODELS

from simulation.model_registry import resolve_model_spec


@dataclass(frozen=True)
class InstancePreset:
    instance_type: str
    gpu_type: str
    gpus_per_instance: int
    gpu_memory_gb: float
    vcpus: int
    quota_family: str
    cost_per_instance_hour_usd: float


@dataclass(frozen=True)
class QuotaPreset:
    slug: str
    title: str
    cloud: str
    notes: str
    instances: tuple[InstancePreset, ...]
    quotas: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ScenarioEvent:
    event_id: str
    at_seconds: float
    action: str
    label: str
    description: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioPreset:
    slug: str
    title: str
    description: str
    initial_replicas: int
    launch_timing_multiplier: float = 1.0
    events: tuple[ScenarioEvent, ...] = ()


@dataclass(frozen=True)
class DemoModelChoice:
    model_name: str
    label: str
    params_billions: float
    architecture_family: str
    is_moe: bool


_QUOTA_PRESETS: dict[str, QuotaPreset] = {
    "aws_l40s_roomy": QuotaPreset(
        slug="aws_l40s_roomy",
        title="AWS L40S Roomy",
        cloud="aws",
        notes="Fast, forgiving demo preset with L40, L4, A10G, and A100 headroom.",
        instances=(
            InstancePreset("g6e.12xlarge", "L40S", 4, 48.0, 48, "G6E", 7.35),
            InstancePreset("g6e.48xlarge", "L40S", 8, 48.0, 192, "G6E", 14.69),
            InstancePreset("g6.12xlarge", "L4", 4, 24.0, 48, "G6", 4.10),
            InstancePreset("g5.12xlarge", "A10G", 4, 24.0, 48, "G5", 5.67),
            InstancePreset("p4de.24xlarge", "A100-80GB", 8, 80.0, 96, "P4DE", 40.96),
        ),
        quotas=(
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 384,
                "used_vcpus": 0,
            },
            {
                "family": "G6",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 192,
                "used_vcpus": 0,
            },
            {
                "family": "G5",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "P4DE",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
        ),
    ),
    "aws_mixed_demo": QuotaPreset(
        slug="aws_mixed_demo",
        title="AWS Mixed Demo",
        cloud="aws",
        notes="Mixed fleet with L40, A10G, A100, and H100 so Koi can visibly choose.",
        instances=(
            InstancePreset("g6e.12xlarge", "L40S", 4, 48.0, 48, "G6E", 7.35),
            InstancePreset("g5.12xlarge", "A10G", 4, 24.0, 48, "G5", 5.67),
            InstancePreset("p4de.24xlarge", "A100-80GB", 8, 80.0, 96, "P4DE", 40.96),
            InstancePreset("p5.48xlarge", "H100", 8, 80.0, 192, "P5", 98.32),
        ),
        quotas=(
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 192,
                "used_vcpus": 0,
            },
            {
                "family": "G5",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "P4DE",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "P5",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 192,
                "used_vcpus": 0,
            },
        ),
    ),
    "aws_no_a100": QuotaPreset(
        slug="aws_no_a100",
        title="AWS No A100",
        cloud="aws",
        notes="No A100 quota at all. Forces the demo to choose between L40S, L4, and A10G capacity.",
        instances=(
            InstancePreset("g6e.12xlarge", "L40S", 4, 48.0, 48, "G6E", 7.35),
            InstancePreset("g6e.48xlarge", "L40S", 8, 48.0, 192, "G6E", 14.69),
            InstancePreset("g6.12xlarge", "L4", 4, 24.0, 48, "G6", 4.10),
            InstancePreset("g5.12xlarge", "A10G", 4, 24.0, 48, "G5", 5.67),
        ),
        quotas=(
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 288,
                "used_vcpus": 0,
            },
            {
                "family": "G6",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 144,
                "used_vcpus": 0,
            },
            {
                "family": "G5",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
        ),
    ),
    "aws_a100_tight": QuotaPreset(
        slug="aws_a100_tight",
        title="AWS A100 Tight",
        cloud="aws",
        notes="Constrained quota preset with A100, L40, and A10G to show tradeoffs.",
        instances=(
            InstancePreset("p4de.24xlarge", "A100-80GB", 8, 80.0, 96, "P4DE", 40.96),
            InstancePreset("g6e.12xlarge", "L40S", 4, 48.0, 48, "G6E", 7.35),
            InstancePreset("g5.12xlarge", "A10G", 4, 24.0, 48, "G5", 5.67),
        ),
        quotas=(
            {
                "family": "P4DE",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 48,
            },
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "G5",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 48,
                "used_vcpus": 0,
            },
        ),
    ),
    "aws_spot_heavy": QuotaPreset(
        slug="aws_spot_heavy",
        title="AWS Spot Heavy",
        cloud="aws",
        notes="Spot-biased demo preset to show volatility and fallback behavior.",
        instances=(
            InstancePreset("g6e.12xlarge", "L40S", 4, 48.0, 48, "G6E", 7.35),
            InstancePreset("g5.12xlarge", "A10G", 4, 24.0, 48, "G5", 5.67),
            InstancePreset("p4de.24xlarge", "A100-80GB", 8, 80.0, 96, "P4DE", 40.96),
            InstancePreset("g6.12xlarge", "L4", 4, 24.0, 48, "G6", 4.10),
        ),
        quotas=(
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "spot",
                "baseline_vcpus": 192,
                "used_vcpus": 0,
            },
            {
                "family": "G5",
                "region": "us-east-1",
                "market": "spot",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "P4DE",
                "region": "us-east-1",
                "market": "spot",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "G6",
                "region": "us-east-1",
                "market": "spot",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
        ),
    ),
    "aws_on_demand_safe": QuotaPreset(
        slug="aws_on_demand_safe",
        title="AWS On-Demand Safe",
        cloud="aws",
        notes="Stable on-demand fleet for predictable demo throughput and recovery.",
        instances=(
            InstancePreset("g6e.12xlarge", "L40S", 4, 48.0, 48, "G6E", 7.35),
            InstancePreset("g5.12xlarge", "A10G", 4, 24.0, 48, "G5", 5.67),
            InstancePreset("p4de.24xlarge", "A100-80GB", 8, 80.0, 96, "P4DE", 40.96),
            InstancePreset("g6.12xlarge", "L4", 4, 24.0, 48, "G6", 4.10),
        ),
        quotas=(
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 288,
                "used_vcpus": 0,
            },
            {
                "family": "G5",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 144,
                "used_vcpus": 0,
            },
            {
                "family": "P4DE",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 192,
                "used_vcpus": 0,
            },
            {
                "family": "G6",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 144,
                "used_vcpus": 0,
            },
        ),
    ),
    "aws_multi_region_balanced": QuotaPreset(
        slug="aws_multi_region_balanced",
        title="AWS Multi-Region Balanced",
        cloud="aws",
        notes="Balanced east/west inventory to visualize region-aware placement paths.",
        instances=(
            InstancePreset("g6e.12xlarge", "L40S", 4, 48.0, 48, "G6E", 7.35),
            InstancePreset("g5.12xlarge", "A10G", 4, 24.0, 48, "G5", 5.67),
            InstancePreset("p4de.24xlarge", "A100-80GB", 8, 80.0, 96, "P4DE", 40.96),
            InstancePreset("g6.12xlarge", "L4", 4, 24.0, 48, "G6", 4.10),
        ),
        quotas=(
            {
                "family": "G6E",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 144,
                "used_vcpus": 0,
            },
            {
                "family": "G6E",
                "region": "us-west-2",
                "market": "on_demand",
                "baseline_vcpus": 144,
                "used_vcpus": 0,
            },
            {
                "family": "G5",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "G5",
                "region": "us-west-2",
                "market": "spot",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "P4DE",
                "region": "us-east-1",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
            {
                "family": "G6",
                "region": "us-west-2",
                "market": "on_demand",
                "baseline_vcpus": 96,
                "used_vcpus": 0,
            },
        ),
    ),
}


_SCENARIOS: dict[str, ScenarioPreset] = {
    "hero_elastic": ScenarioPreset(
        slug="hero_elastic",
        title="Hero Elastic",
        description="Start slightly underprovisioned, push it behind, let Koi scale up, then relieve pressure so scale-down becomes sensible.",
        initial_replicas=1,
        events=(
            ScenarioEvent(
                event_id="hero-pressure-rise",
                at_seconds=18.0,
                action="degrade_replica",
                label="Input spike",
                description="Simulate a harder batch and rising decode pressure.",
                params={"target_tps": 320, "over_seconds": 20},
            ),
            ScenarioEvent(
                event_id="hero-pressure-relief",
                at_seconds=95.0,
                action="restore_cluster_tps",
                label="Pressure relief",
                description="Ease workload pressure so Koi can consider trimming excess capacity.",
                params={"target_tps": 1250},
            ),
        ),
    ),
    "kill_and_recover": ScenarioPreset(
        slug="kill_and_recover",
        title="Kill And Recover",
        description="Kill a running replica in the middle of the batch and let Koi react in real time.",
        initial_replicas=2,
        events=(
            ScenarioEvent(
                event_id="kill-primary",
                at_seconds=30.0,
                action="kill_oldest_running",
                label="Replica loss",
                description="Simulate a real EC2 termination on one live worker.",
                params={"reason": "Simulated EC2 termination"},
            ),
        ),
    ),
    "overprovisioned": ScenarioPreset(
        slug="overprovisioned",
        title="Overprovisioned",
        description="Start with excess replicas and a loose deadline so Koi should scale down.",
        initial_replicas=4,
    ),
    "slow_launch": ScenarioPreset(
        slug="slow_launch",
        title="Slow Launch",
        description="Emphasize searching/provisioning/model-ready phases before the job starts serving.",
        initial_replicas=1,
        launch_timing_multiplier=2.5,
        events=(
            ScenarioEvent(
                event_id="slow-launch-pressure",
                at_seconds=0.0,
                action="capacity_pressure",
                label="Capacity search",
                description="Bias launch timing upward to showcase heartbeats and pending leases.",
                params={"pressure": 0.8},
            ),
        ),
    ),
}


def list_quota_presets() -> list[QuotaPreset]:
    return list(_QUOTA_PRESETS.values())


def get_quota_preset(
    slug: str,
    *,
    overrides: Optional[Mapping[str, float]] = None,
) -> QuotaPreset:
    """Return the quota preset, optionally overlaying per-row baseline_vcpus overrides.

    ``overrides`` maps quota row keys (``"FAMILY|region|market"``) to their new
    ``baseline_vcpus`` value. Unknown keys are ignored. None means "use defaults".
    """

    preset = _QUOTA_PRESETS[slug]
    if not overrides:
        return preset
    normalized = {str(k): float(v) for k, v in overrides.items()}
    if not normalized:
        return preset
    patched_quotas: list[dict[str, Any]] = []
    changed = False
    for quota in preset.quotas:
        entry = dict(quota)
        key = quota_row_key(entry)
        if key in normalized:
            new_value = max(0, int(round(normalized[key])))
            if new_value != int(entry.get("baseline_vcpus", 0) or 0):
                changed = True
            entry["baseline_vcpus"] = new_value
        patched_quotas.append(entry)
    if not changed:
        return preset
    return QuotaPreset(
        slug=preset.slug,
        title=preset.title,
        cloud=preset.cloud,
        notes=preset.notes,
        instances=preset.instances,
        quotas=tuple(patched_quotas),
    )


def quota_row_key(row: Mapping[str, Any]) -> str:
    """Stable key for a quota row (family / region / market)."""

    family = str(row.get("family") or "").upper()
    region = str(row.get("region") or "")
    market = str(row.get("market") or "")
    return f"{family}|{region}|{market}"


def quota_preset_to_resource_map(
    slug: str,
    *,
    overrides: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    preset = get_quota_preset(slug, overrides=overrides)
    return {
        "instances": [asdict(instance) for instance in preset.instances],
        "quotas": list(preset.quotas),
    }


def default_quota_overrides(slug: str) -> dict[str, int]:
    """Baseline vCPU values keyed by quota_row_key for the given preset."""

    preset = _QUOTA_PRESETS[slug]
    return {
        quota_row_key(row): int(row.get("baseline_vcpus", 0) or 0)
        for row in preset.quotas
    }


def quota_preset_editable_rows(slug: str) -> list[dict[str, Any]]:
    """Serialize the preset's quota rows with slider metadata for the UI."""

    preset = _QUOTA_PRESETS[slug]
    rows: list[dict[str, Any]] = []
    for row in preset.quotas:
        baseline = int(row.get("baseline_vcpus", 0) or 0)
        # Upper bound: give headroom to 4× default or 1024, whichever is larger,
        # rounded to the nearest 16 vCPUs for tidy slider snapping.
        upper = max(baseline * 4, 1024)
        upper = int(round(upper / 16.0)) * 16
        rows.append(
            {
                "key": quota_row_key(row),
                "family": str(row.get("family") or "").upper(),
                "region": str(row.get("region") or ""),
                "market": str(row.get("market") or ""),
                "default_vcpus": baseline,
                "min_vcpus": 0,
                "max_vcpus": max(upper, 16),
                "step_vcpus": 16,
            }
        )
    return rows


def list_scenarios() -> list[ScenarioPreset]:
    return list(_SCENARIOS.values())


def get_scenario(slug: str) -> ScenarioPreset:
    return _SCENARIOS[slug]


def due_scenario_events(
    scenario_slug: str,
    *,
    elapsed_seconds: float,
    completed_event_ids: Optional[Iterable[str]] = None,
) -> list[ScenarioEvent]:
    completed = set(completed_event_ids or ())
    scenario = get_scenario(scenario_slug)
    return [
        event
        for event in scenario.events
        if event.event_id not in completed and event.at_seconds <= elapsed_seconds
    ]


def list_demo_models() -> list[DemoModelChoice]:
    choices = []
    for model_name in sorted(_KNOWN_MODELS):
        spec = resolve_model_spec(model_name)
        choices.append(
            DemoModelChoice(
                model_name=model_name,
                label=model_name.split("/")[-1],
                params_billions=spec.num_params_billions,
                architecture_family=spec.architecture_family,
                is_moe=spec.is_moe,
            )
        )
    return choices


def serialize_catalog() -> dict[str, Any]:
    return {
        "models": [asdict(model) for model in list_demo_models()],
        "quota_presets": [
            {
                "slug": preset.slug,
                "title": preset.title,
                "cloud": preset.cloud,
                "notes": preset.notes,
                "instances": [asdict(instance) for instance in preset.instances],
                "quotas": list(preset.quotas),
            }
            for preset in list_quota_presets()
        ],
        "scenarios": [
            {
                "slug": scenario.slug,
                "title": scenario.title,
                "description": scenario.description,
                "initial_replicas": scenario.initial_replicas,
                "launch_timing_multiplier": scenario.launch_timing_multiplier,
                "events": [
                    {
                        "event_id": event.event_id,
                        "at_seconds": event.at_seconds,
                        "action": event.action,
                        "label": event.label,
                        "description": event.description,
                        "params": dict(event.params),
                    }
                    for event in scenario.events
                ],
            }
            for scenario in list_scenarios()
        ],
    }
