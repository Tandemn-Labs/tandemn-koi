"""
koi/resource_ledger.py — Tracks pending GPU reservations between /decide and /job/started.

Active allocations are Orca's responsibility (ground truth via GET /resources).
This ledger only covers the decision-to-launch window to prevent over-allocation
when multiple concurrent /decide calls arrive.

Keyed by (cloud, region, gpu_type) for multi-cloud readiness.
Carries tenant_id for multi-tenant readiness.
"""

import threading
import time
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from koi.logging_config import get_logger
from koi.runtime_state import RuntimeStateStore

logger = get_logger("koi.resource_ledger")

PENDING_TTL_SECONDS = 600.0  # 10 min — auto-expire if Orca never confirms


@dataclass
class GPUAllocation:
    gpu_type: str
    num_gpus: int
    cloud: str = "aws"
    region: str = "unknown"
    tenant_id: str = "default"
    instance_type: str = "unknown"
    decision_id: str = ""
    created_at: float = field(default_factory=time.time)


class ResourceLedger:
    """Thread-safe tracking of pending GPU reservations.

    Usage:
        ledger = ResourceLedger()
        ledger.reserve("dec-123", "H100", 8, region="us-east-1")
        adjusted_map = ledger.apply_to_resource_map(base_map)
        ledger.release("dec-123")  # on /job/started or /job/launch-failed
    """

    def __init__(
        self,
        pending_ttl: float = PENDING_TTL_SECONDS,
        runtime_state: Optional[RuntimeStateStore] = None,
    ):
        self._pending: Dict[str, GPUAllocation] = {}
        self._lock = threading.Lock()
        self._ttl = pending_ttl
        self._runtime_state = runtime_state

    def reserve(self, decision_id: str, gpu_type: str, num_gpus: int,
                cloud: str = "aws", region: str = "unknown",
                instance_type: str = "unknown",
                tenant_id: str = "default") -> None:
        """Reserve GPUs after /decide or scale_chain_tool."""
        with self._lock:
            self._expire_stale()
            alloc = GPUAllocation(
                gpu_type=gpu_type, num_gpus=num_gpus,
                cloud=cloud, region=region,
                instance_type=instance_type,
                tenant_id=tenant_id,
                decision_id=decision_id,
            )
            self._pending[decision_id] = alloc
            if self._runtime_state:
                self._runtime_state.upsert_ledger_reservation(
                    decision_id=decision_id,
                    reservation=asdict(alloc),
                    expires_at=alloc.created_at + self._ttl,
                )
        logger.info("gpu_reserved", decision_id=decision_id, gpu_type=gpu_type,
                     num_gpus=num_gpus, region=region)

    def release(self, decision_id: str) -> Optional[GPUAllocation]:
        """Release a pending reservation (on /job/started or /job/launch-failed)."""
        with self._lock:
            alloc = self._pending.pop(decision_id, None)
            if alloc and self._runtime_state:
                self._runtime_state.delete_ledger_reservation(decision_id)
        if alloc:
            logger.info("gpu_released", decision_id=decision_id, gpu_type=alloc.gpu_type,
                         num_gpus=alloc.num_gpus)
        return alloc

    def restore(self) -> int:
        """Rebuild pending reservations from persistent runtime state."""
        if not self._runtime_state:
            return 0

        restored = 0
        now = time.time()
        persisted = self._runtime_state.load_ledger_reservations()
        with self._lock:
            self._expire_stale()
            for decision_id, entry in persisted.items():
                expires_at = entry.get("expires_at")
                if expires_at is not None and expires_at <= now:
                    self._runtime_state.delete_ledger_reservation(decision_id)
                    continue
                reservation = entry["reservation"]
                self._pending[decision_id] = GPUAllocation(**reservation)
                restored += 1
        if restored:
            logger.info("ledger_restored", count=restored)
        return restored

    def get_pending_by_type(self, cloud: str = None, region: str = None,
                            gpu_type: str = None,
                            tenant_id: str = None) -> Dict[str, int]:
        """Aggregated pending GPUs per gpu_type. Filterable by any dimension."""
        with self._lock:
            self._expire_stale()
            result: Dict[str, int] = {}
            for alloc in self._pending.values():
                if cloud and alloc.cloud != cloud:
                    continue
                if region and alloc.region != region:
                    continue
                if gpu_type and alloc.gpu_type != gpu_type:
                    continue
                if tenant_id and alloc.tenant_id != tenant_id:
                    continue
                result[alloc.gpu_type] = result.get(alloc.gpu_type, 0) + alloc.num_gpus
            return result

    def apply_to_resource_map(self, base_map):
        """Return a new ResourceMap with pending reservations subtracted.

        Accepts and returns a koi.schemas.ResourceMap. Import deferred to avoid
        circular imports.
        """
        adjusted = []
        for res in base_map.resources:
            extra = self.get_pending_by_type(
                cloud=getattr(res, "cloud", None),
                region=getattr(res, "region", None),
                gpu_type=res.gpu_type,
            ).get(res.gpu_type, 0)
            if extra > 0:
                new_res = res.model_copy(update={
                    "allocated_gpus": res.allocated_gpus + extra,
                })
                adjusted.append(new_res)
            else:
                adjusted.append(res)

        return base_map.model_copy(update={"resources": adjusted})

    def summary(self) -> List[dict]:
        """For GET /resources endpoint."""
        with self._lock:
            self._expire_stale()
            return [
                {
                    "decision_id": alloc.decision_id,
                    "gpu_type": alloc.gpu_type,
                    "num_gpus": alloc.num_gpus,
                    "cloud": alloc.cloud,
                    "region": alloc.region,
                    "tenant_id": alloc.tenant_id,
                    "age_seconds": round(time.time() - alloc.created_at),
                }
                for alloc in self._pending.values()
            ]

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def _expire_stale(self):
        """Remove pending reservations older than TTL. Must hold lock."""
        cutoff = time.time() - self._ttl
        expired = [k for k, v in self._pending.items() if v.created_at < cutoff]
        for k in expired:
            alloc = self._pending.pop(k)
            if self._runtime_state:
                self._runtime_state.delete_ledger_reservation(k)
            logger.warning("pending_expired", decision_id=k, gpu_type=alloc.gpu_type,
                           num_gpus=alloc.num_gpus, age_seconds=round(time.time() - alloc.created_at))
