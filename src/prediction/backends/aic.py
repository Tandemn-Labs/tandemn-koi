"""Thin backend adapter for Koi's authoritative direct AIC surrogate."""

import hashlib
import importlib.metadata
import json
from threading import RLock
from typing import TYPE_CHECKING

from src.prediction.backends.base import Candidate, SurrogateEstimate

if TYPE_CHECKING:
    from src.prediction.surrogate import SurrogatePrediction


_AIC_DIRECT_METHOD = ("AIC_Direct",)


class AICBackend:
    """Delegate inputs to SurrogatePrediction with the production Direct method."""

    name = "primary"

    def __init__(self, surrogate: "SurrogatePrediction | None" = None):
        self.surrogate = surrogate
        implementation = (
            "aic:src.prediction.surrogate.SurrogatePrediction"
            if surrogate is None
            else f"aic:{type(surrogate).__module__}.{type(surrogate).__qualname__}"
        )
        self.version = (
            f"{implementation}:aic-{_package_version('aiconfigurator')}:"
            f"dynamo-{_package_version('ai-dynamo')}:joint-knn-v1"
        )
        self._lock = RLock()

    def provides(self) -> set[str]:
        return {
            "cost_per_token",
            "p99_ttft_ms",
            "p99_tpot_ms",
            "slo_margin",
            "throughput_token_per_sec",
        }

    def estimate(
        self,
        candidate: Candidate,
        *,
        candidate_graph=None,
        method=("AIC_Direct",),
        scenario: str = "mean",
    ) -> SurrogateEstimate:
        normalized_method = self._normalize_method(method)
        with self._lock:
            return self._estimate_locked(
                candidate,
                candidate_graph=candidate_graph,
                method=normalized_method,
                scenario=scenario,
            )

    @staticmethod
    def _normalize_method(method) -> tuple:
        """Retain the method input contract while enforcing Direct AIC."""
        del method
        return _AIC_DIRECT_METHOD

    def _estimate_locked(
        self,
        candidate: Candidate,
        *,
        candidate_graph=None,
        method=("AIC_Direct",),
        scenario: str = "mean",
    ) -> SurrogateEstimate:
        method = self._normalize_method(method)
        surrogate = self.surrogate
        if surrogate is None:
            from src.prediction.surrogate import SurrogatePrediction

            surrogate = SurrogatePrediction()
            self.surrogate = surrogate
        try:
            y_hat, v_hat = surrogate.compose_prediction(
                job_config=candidate.job_config,
                job_features=candidate.job_features,
                candidate_graph=candidate_graph,
                method=method,
                scenario=scenario,
            )
        except Exception as exc:
            from src.prediction.surrogate import (
                SurrogateExecutionError,
                SurrogateUnsupportedConfig,
            )

            if not isinstance(exc, (SurrogateUnsupportedConfig, SurrogateExecutionError)):
                raise
            return SurrogateEstimate(
                status=("unsupported" if isinstance(exc, SurrogateUnsupportedConfig) else "failed"),
                version=self.version,
                source=self.name,
                metadata={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **dict(getattr(surrogate, "last_metadata", {}) or {}),
                },
            )
        nodes = set(y_hat or {}) | set(v_hat or {})
        metadata = {
            "method": list(method) if isinstance(method, list | tuple) else method,
            **dict(getattr(surrogate, "last_metadata", {}) or {}),
        }
        version = self.version
        if metadata.get("aic_database_mode"):
            version = f"{version}:{metadata['aic_database_mode'].lower()}"
        compatibility = metadata.get("compatibility") or {}
        approximate = {
            name: {
                key: resolution.get(key)
                for key in ("canonical", "resolved", "backend_value", "kind")
            }
            for name, resolution in compatibility.items()
            if resolution.get("kind") == "nearest" or name.endswith("_dtype")
        }
        for resolution in approximate.values():
            if resolution["kind"] != "nearest":
                resolution["kind"] = "compatible"
        if approximate:
            payload = json.dumps(approximate, sort_keys=True, separators=(",", ":"))
            version = f"{version}:compat-{hashlib.sha256(payload.encode()).hexdigest()[:8]}"
        profile_match = metadata.get("aic_profile_match") or {}
        if profile_match:
            identity = {key: profile_match.get(key) for key in ("profile_id", "search_version")}
            payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            version = f"{version}:profile-{hashlib.sha256(payload.encode()).hexdigest()[:8]}"
        confidence = min(
            (float(entry.get("confidence", 1.0)) for entry in compatibility.values()),
            default=1.0,
        )
        confidence = min(confidence, float(profile_match.get("confidence", 1.0)))
        return SurrogateEstimate(
            y_hat=dict(y_hat or {}),
            v_hat=dict(v_hat or {}),
            status="success",
            version=version,
            coverage=dict.fromkeys(nodes, confidence),
            source=self.name,
            metadata=metadata,
        )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
