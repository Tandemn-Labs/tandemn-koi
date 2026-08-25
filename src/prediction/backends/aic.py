"""Thin backend adapter for Koi's authoritative direct AIC surrogate."""

import hashlib
import importlib.metadata
import json
import math
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from typing import TYPE_CHECKING

from src.prediction.backends.base import Candidate, SurrogateEstimate

if TYPE_CHECKING:
    from src.prediction.surrogate import SurrogatePrediction


_CACHE_KEY_VERSION = "aic-primary-cache-v1"
_CACHE_MAX_ENTRIES = 512


class AICBackend:
    """Delegate unchanged inputs and structured failures to SurrogatePrediction."""

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
        self._cache: OrderedDict[str, SurrogateEstimate] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_bypasses = 0
        self._cache_evictions = 0

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
            cache_key = self._cache_key(
                candidate,
                candidate_graph=candidate_graph,
                method=normalized_method,
                scenario=scenario,
            )
            if cache_key is None:
                self._cache_bypasses += 1
                estimate = self._estimate_locked(
                    candidate,
                    candidate_graph=candidate_graph,
                    method=normalized_method,
                    scenario=scenario,
                )
                return self._returned_estimate(estimate, hit=False)

            if cache_key in self._cache:
                self._cache_hits += 1
                self._cache.move_to_end(cache_key)
                return self._returned_estimate(self._cache[cache_key], hit=True)

            self._cache_misses += 1
            estimate = self._estimate_locked(
                candidate,
                candidate_graph=candidate_graph,
                method=normalized_method,
                scenario=scenario,
            )
            if self._is_cacheable(estimate):
                self._cache[cache_key] = deepcopy(estimate)
                if len(self._cache) > _CACHE_MAX_ENTRIES:
                    self._cache.popitem(last=False)
                    self._cache_evictions += 1
            return self._returned_estimate(estimate, hit=False)

    def has_cached(
        self,
        candidate: Candidate,
        *,
        candidate_graph=None,
        method=("AIC_Direct",),
        scenario: str = "mean",
    ) -> bool:
        """Return whether this exact raw prediction is in the process-local cache."""
        normalized_method = self._normalize_method(method)
        with self._lock:
            cache_key = self._cache_key(
                candidate,
                candidate_graph=candidate_graph,
                method=normalized_method,
                scenario=scenario,
            )
            return cache_key is not None and cache_key in self._cache

    def clear_cache(self) -> None:
        """Clear cached estimates and cache statistics."""
        with self._lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_bypasses = 0
            self._cache_evictions = 0

    def cache_info(self) -> dict[str, int]:
        """Return bounded process-local cache statistics."""
        with self._lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "bypasses": self._cache_bypasses,
                "evictions": self._cache_evictions,
                "entries": len(self._cache),
                "max_entries": _CACHE_MAX_ENTRIES,
            }

    def _cache_key(
        self,
        candidate: Candidate,
        *,
        candidate_graph,
        method,
        scenario: str,
    ) -> str | None:
        try:
            payload = {
                "key_version": _CACHE_KEY_VERSION,
                "backend_version": self.version,
                "job_config": candidate.job_config,
                "job_features": candidate.job_features,
                "env": getattr(candidate, "env", None),
                "method": method,
                "scenario": scenario,
                "candidate_graph": {
                    scope: sorted(getattr(candidate_graph, scope, ()) or ())
                    for scope in ("x", "v", "y")
                },
            }
            canonical = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (RecursionError, TypeError, UnicodeEncodeError, ValueError):
            return None
        digest = hashlib.sha256(canonical).hexdigest()
        return f"{_CACHE_KEY_VERSION}:{digest}"

    @staticmethod
    def _normalize_method(method) -> tuple:
        return tuple(method) if isinstance(method, list | tuple) else (method,)

    @staticmethod
    def _is_cacheable(estimate: SurrogateEstimate) -> bool:
        throughput = estimate.y_hat.get("throughput_token_per_sec")
        if isinstance(throughput, bool) or not isinstance(throughput, int | float):
            return False
        return (
            estimate.status == "success"
            and math.isfinite(float(throughput))
            and float(throughput) > 0.0
            and not _metadata_indicates_degraded(estimate.metadata)
        )

    def _returned_estimate(self, estimate: SurrogateEstimate, *, hit: bool) -> SurrogateEstimate:
        result = deepcopy(estimate)
        result.metadata = dict(result.metadata or {})
        result.metadata["aic_raw_cache"] = {
            "hit": hit,
            "key_version": _CACHE_KEY_VERSION,
            "entries": len(self._cache),
            "max_entries": _CACHE_MAX_ENTRIES,
        }
        return result

    def _estimate_locked(
        self,
        candidate: Candidate,
        *,
        candidate_graph=None,
        method=("AIC_Direct",),
        scenario: str = "mean",
    ) -> SurrogateEstimate:
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


def _metadata_indicates_degraded(value) -> bool:
    if isinstance(value, dict):
        for raw_name, item in value.items():
            name = str(raw_name).lower()
            if name == "aic_fallback_omitted_nodes":
                continue
            if (
                name in {"degraded", "unavailable", "execution_error", "memory_no_fit"}
                or "fallback" in name
            ) and bool(item):
                return True
            if name in {"status", "state", "availability"} and str(item).lower() in {
                "degraded",
                "unavailable",
                "fallback",
                "failed",
            }:
                return True
            if _metadata_indicates_degraded(item):
                return True
        return False
    if isinstance(value, list | tuple | set):
        return any(_metadata_indicates_degraded(item) for item in value)
    return False
