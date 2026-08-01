"""Typed contracts for Koi surrogate backends."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Candidate:
    """Normalized inputs for one Koi rank candidate."""

    job_config: dict[str, Any]
    job_features: dict[str, Any]
    env: tuple[str, ...] | None = None


@dataclass
class SurrogateEstimate:
    """Status-rich backend output in canonical Koi V/Y vocabulary."""

    v_hat: dict[str, float] = field(default_factory=dict)
    y_hat: dict[str, float] = field(default_factory=dict)
    status: str = "success"
    version: str | None = None
    coverage: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""


@runtime_checkable
class SurrogateBackend(Protocol):
    name: str

    def provides(self) -> set[str]: ...

    def estimate(
        self,
        candidate: Candidate,
        *,
        candidate_graph=None,
        method=("AIC_DynoSim",),
        scenario: str = "mean",
    ) -> SurrogateEstimate: ...
