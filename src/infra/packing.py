"""Co-hosted instance packing for cloud pools (KOI_COHOST_PACKING).

The store cannot express an allocation unit smaller than an instance, so by
default Koi reserves one whole instance per replica. Orca, however, requests
only ``gpu_count`` GPUs per worker pod and Kubernetes packs several such pods
onto one node. With ``KOI_COHOST_PACKING`` on, Koi models an instance as
``gpus_per_instance`` slots and packs replica footprints into instances with
first-fit-decreasing. Only an EMPTY instance can take a full-width replica, so
fragmentation stays visible to the planner.

This module has no Koi imports so every consumer (resource map, agent tools,
validator, switch cost, planner prompt) can share it without new import edges.
The plan / ladder schema Koi hands to Orca is untouched.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

COHOST_PACKING_ENV = "KOI_COHOST_PACKING"
PACKED = "packed"


def cohost_packing_enabled() -> bool:
    return os.environ.get(COHOST_PACKING_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def effective_allocation_kind(kind: object) -> str:
    """Map a pool's declared allocation kind to the kind Koi accounts with.

    ``instance`` pools become ``packed`` when co-hosting is on; ``gpu`` pools
    and explicit ``packed`` pools are returned unchanged.
    """
    value = str(kind or "instance").lower()
    if value == "instance" and cohost_packing_enabled():
        return PACKED
    return value


def is_slot_kind(kind: object) -> bool:
    """True when one replica reserves its engine GPUs, not a whole instance."""
    return str(kind or "").lower() in {"gpu", PACKED}


def capacity_unit_label(kind: object) -> str:
    value = str(kind or "").lower()
    if value == "gpu":
        return "GPUs"
    if value == PACKED:
        return "GPU slots"
    return "instances"


def pack_footprints(
    footprints: Iterable[int],
    gpus_per_instance: int,
    bins: Sequence[int] | None = None,
) -> list[int]:
    """First-fit-decreasing pack of replica GPU footprints into instances.

    Returns GPUs used per instance. ``bins`` seeds already-occupied instances
    (GPUs used each) and is not mutated. A footprint at or above the instance
    size takes a whole new instance.
    """
    gpi = max(1, int(gpus_per_instance))
    packed = [min(gpi, max(0, int(used))) for used in (bins or [])]
    for size in sorted((int(size) for size in footprints), reverse=True):
        if size <= 0:
            continue
        if size >= gpi:
            packed.append(gpi)
            continue
        for index, used in enumerate(packed):
            if used + size <= gpi:
                packed[index] = used + size
                break
        else:
            packed.append(size)
    return packed


def partial_free_slots(bins: Sequence[int], gpus_per_instance: int) -> list[int]:
    """Free GPUs on each partially used instance, largest first."""
    gpi = max(1, int(gpus_per_instance))
    return sorted((gpi - int(used) for used in bins if 0 < int(used) < gpi), reverse=True)


def packing_shortfall(
    footprints: Iterable[int],
    gpus_per_instance: int,
    partial_slots: Sequence[int],
    empty_instances: int,
) -> int:
    """Whole instances missing to place ``footprints``; 0 means they fit.

    Existing partial instances are described by their free slots. New
    instances opened by the pack are charged against ``empty_instances``.
    """
    gpi = max(1, int(gpus_per_instance))
    seeded = [gpi - int(slot) for slot in partial_slots]
    after = pack_footprints(footprints, gpi, bins=seeded)
    opened = len(after) - len(seeded)
    return max(0, opened - max(0, int(empty_instances)))


def packable_replicas(
    engine_gpus: int,
    gpus_per_instance: int,
    partial_slots: Sequence[int],
    empty_instances: int,
) -> int:
    """How many replicas of ``engine_gpus`` GPUs fit without fragmenting."""
    gpi = max(1, int(gpus_per_instance))
    size = int(engine_gpus)
    if size <= 0 or size > gpi:
        return 0
    from_partial = sum(int(slot) // size for slot in partial_slots)
    return from_partial + max(0, int(empty_instances)) * (gpi // size)


def apply_footprints_to_pool(
    footprints: Iterable[int],
    gpus_per_instance: int,
    partial_slots: Sequence[int],
    empty_instances: int,
) -> tuple[list[int], int]:
    """Return ``(partial_slots, empty_instances)`` after placing ``footprints``.

    Used to fold not-yet-materialized (pending) deployments into a pool's
    packing state before candidates are generated or jointly selected.
    """
    gpi = max(1, int(gpus_per_instance))
    seeded = [gpi - int(slot) for slot in partial_slots]
    after = pack_footprints(footprints, gpi, bins=seeded)
    opened = len(after) - len(seeded)
    return partial_free_slots(after, gpi), max(0, int(empty_instances) - opened)
