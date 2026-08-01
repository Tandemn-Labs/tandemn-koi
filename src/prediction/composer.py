"""Koi-native composition around the authoritative direct AIC prediction."""

from __future__ import annotations

import hashlib
import logging
import time
import traceback
from collections.abc import Callable, Mapping
from copy import deepcopy
from threading import RLock
from typing import Any

from src.observability.events import EventSink
from src.prediction.analytic_parallelism import compute_parallelism_v
from src.prediction.analytic_v import compute_memory_v
from src.prediction.backends.aic import AICBackend
from src.prediction.backends.base import Candidate, SurrogateEstimate
from src.prediction.calibration import (
    build_prediction_context,
    calibrate_prediction,
    learn_throughput_fusion,
)
from src.prediction.normalization import normalize_candidate_inputs
from src.prediction.peer_client import PeerPredictorClient

log = logging.getLogger("koi.surrogate.composer")

_MODES = frozenset({"off", "shadow", "enabled"})
_MEASURED_ONLY_MIN_COVERAGE = 0.8


class LoggingEventSink:
    """Emit structured prediction events through stdlib logging."""

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO):
        self.logger = logger or log
        self.level = level

    def emit(self, event: Mapping[str, Any]) -> None:
        self.logger.log(self.level, "%s", dict(event))


class SurrogateComposer:
    """Compose analytic V, measured shadows, peers, fusion, and one calibration."""

    def __init__(
        self,
        primary=None,
        *,
        perfdb_backend=None,
        perfdb_mode: str = "enabled",
        peer_client=None,
        peer_mode: str = "shadow",
        evidence_store=None,
        event_sink: EventSink | Callable[[Mapping[str, Any]], None] | None = None,
    ):
        if perfdb_mode not in _MODES:
            raise ValueError("perfdb_mode must be off, shadow, or enabled")
        if peer_mode not in _MODES:
            raise ValueError("peer_mode must be off, shadow, or enabled")
        if primary is None:
            self.primary = AICBackend()
        elif hasattr(primary, "estimate") and hasattr(primary, "provides"):
            self.primary = primary
        else:
            self.primary = AICBackend(primary)
        self.perfdb_backend = perfdb_backend
        self.perfdb_mode = perfdb_mode
        self.peer_client = peer_client or PeerPredictorClient()
        self.peer_mode = peer_mode
        self.evidence_store = evidence_store
        self.event_sink = event_sink
        self.last_trace: dict[str, Any] = {}
        self.version = "koi-surrogate-v3"
        self._lock = RLock()

    def bind_evidence_store(self, evidence_store) -> None:
        self.evidence_store = evidence_store

    def compose_prediction(
        self,
        job_config,
        job_features,
        candidate_graph,
        method=("AIC_DynoSim",),
        scenario="mean",
        as_of_timestamp_utc: float | None = None,
        purpose: str | None = None,
        correlation_context: Mapping[str, Any] | None = None,
    ):
        with self._lock:
            y_hat, v_hat, _ = self._compose(
                job_config,
                job_features,
                candidate_graph,
                method=method,
                scenario=scenario,
                as_of_timestamp_utc=as_of_timestamp_utc,
                purpose=purpose,
                correlation_context=correlation_context,
            )
            return y_hat, v_hat

    def compose_prediction_with_trace(
        self,
        job_config,
        job_features,
        candidate_graph,
        method=("AIC_DynoSim",),
        scenario="mean",
        as_of_timestamp_utc: float | None = None,
        purpose: str | None = None,
        correlation_context: Mapping[str, Any] | None = None,
    ):
        with self._lock:
            y_hat, v_hat, trace = self._compose(
                job_config,
                job_features,
                candidate_graph,
                method=method,
                scenario=scenario,
                as_of_timestamp_utc=as_of_timestamp_utc,
                purpose=purpose,
                correlation_context=correlation_context,
            )
            return y_hat, v_hat, deepcopy(trace)

    def _compose(
        self,
        job_config,
        job_features,
        candidate_graph,
        *,
        method,
        scenario,
        as_of_timestamp_utc,
        purpose,
        correlation_context,
    ):
        started = time.perf_counter()
        prediction_timestamp = time.time()
        timings: dict[str, float] = {}
        stage = "normalization"
        config: dict[str, Any] = {}
        features: dict[str, Any] = {}
        components: dict[str, dict[str, Any]] = {}
        backends: dict[str, dict[str, Any]] = {}
        started_emitted = False
        raw_inputs = {
            "job_config": job_config,
            "job_features": job_features,
        }
        graph_request = _candidate_graph_request(candidate_graph)
        event_context = dict(correlation_context or {})
        event_purpose = purpose or "predict candidate outcomes for Koi planning"

        try:
            normalization_started = time.perf_counter()
            config, features = normalize_candidate_inputs(job_config, job_features)
            normalized_candidate = Candidate(config, features)
            primary_candidate = Candidate(job_config, job_features)
            context = build_prediction_context(config, features, scenario=scenario)
            timings[stage] = _elapsed_ms(normalization_started)
            self._emit(
                {
                    "event": "surrogate.prediction.started",
                    "component": "prediction.surrogate_composer",
                    "purpose": event_purpose,
                    **event_context,
                    "inputs": {
                        "raw": raw_inputs,
                        "normalized": {
                            "job_config": config,
                            "job_features": features,
                        },
                    },
                    "candidate_graph": graph_request,
                    "method": list(method) if isinstance(method, list | tuple) else method,
                    "scenario": scenario,
                    "as_of_timestamp_utc": as_of_timestamp_utc,
                    "configuration": {
                        "composer_version": self.version,
                        "primary": type(self.primary).__name__,
                        "perfdb_mode": self.perfdb_mode,
                        "peer_mode": self.peer_mode,
                        "learning_enabled": as_of_timestamp_utc is not None,
                    },
                    "prediction_timestamp_utc": prediction_timestamp,
                }
            )
            started_emitted = True

            stage = "primary"
            primary_started = time.perf_counter()
            primary = self.primary.estimate(
                primary_candidate,
                candidate_graph=candidate_graph,
                method=method,
                scenario=scenario,
            )
            timings[stage] = _elapsed_ms(primary_started)
            components["primary"] = _estimate_trace(primary, timings[stage])
            backends["primary"] = _backend_trace(primary)
            raw_y = dict(primary.y_hat)
            raw_v = dict(primary.v_hat)
            y_hat = dict(raw_y)
            v_hat = dict(raw_v)

            stage = "analytic_memory"
            analytic_started = time.perf_counter()
            memory_v = compute_memory_v(config, features)
            v_hat.update(memory_v)
            timings[stage] = _elapsed_ms(analytic_started)
            components[stage] = _analytic_trace(memory_v, timings[stage], "analytic-memory-v1")

            stage = "analytic_parallelism"
            analytic_started = time.perf_counter()
            parallel_v = compute_parallelism_v(normalized_candidate, y_hat)
            v_hat.update(parallel_v)
            timings[stage] = _elapsed_ms(analytic_started)
            components[stage] = _analytic_trace(
                parallel_v, timings[stage], "analytic-parallelism-v1"
            )

            stage = "perfdb"
            perfdb = self._run_perfdb(normalized_candidate, candidate_graph, method, scenario)
            timings[stage] = perfdb[1]
            perfdb_estimate = perfdb[0]
            components[stage] = _estimate_trace(perfdb_estimate, timings[stage])
            if self.perfdb_mode != "off":
                backends["perfdb"] = _backend_trace(perfdb_estimate)
            if self.perfdb_mode == "enabled" and perfdb_estimate.status == "success":
                y_hat = _coverage_blend(y_hat, perfdb_estimate.y_hat, perfdb_estimate.coverage)
                v_hat = _coverage_blend(v_hat, perfdb_estimate.v_hat, perfdb_estimate.coverage)

            stage = "peers"
            peer_entries, peer_timing = self._run_peers(job_config, job_features, scenario)
            timings[stage] = peer_timing
            components[stage] = {
                "status": "off"
                if self.peer_mode == "off"
                else "skipped"
                if scenario == "peak_all_multiturn_stress"
                else _peer_component_status(peer_entries),
                "version": None,
                "coverage": {},
                "spread": {},
                "metadata": {"mode": self.peer_mode, "peers": list(peer_entries)},
                "timing_ms": peer_timing,
            }
            backends.update(peer_entries)

            stage = "fusion"
            fusion_started = time.perf_counter()
            predictions = {}
            versions = {}
            if raw_y.get("throughput_token_per_sec") is not None:
                predictions["primary"] = float(raw_y["throughput_token_per_sec"])
                versions["primary"] = primary.version
            if (
                self.perfdb_mode == "enabled"
                and perfdb_estimate.status == "success"
                and perfdb_estimate.y_hat.get("throughput_token_per_sec") is not None
            ):
                predictions["perfdb"] = perfdb_estimate.y_hat["throughput_token_per_sec"]
                versions["perfdb"] = perfdb_estimate.version
            for name, entry in peer_entries.items():
                value = (entry.get("y_hat") or {}).get("throughput_token_per_sec")
                if entry.get("status") == "success" and value is not None:
                    predictions[name] = float(value)
                    versions[name] = entry.get("version")
            fusion = learn_throughput_fusion(
                predictions,
                versions,
                config,
                features,
                self.evidence_store,
                scenario=scenario,
                as_of_timestamp_utc=as_of_timestamp_utc,
            )
            peer_fusion_enabled = self.peer_mode == "enabled" and any(
                name not in {"primary", "perfdb"} for name in predictions
            )
            fusion_applied = fusion.status == "learned" and peer_fusion_enabled
            if fusion_applied:
                y_hat["throughput_token_per_sec"] = fusion.throughput
            timings[stage] = _elapsed_ms(fusion_started)
            fusion_trace = {
                "status": fusion.status,
                "applied": fusion_applied,
                "weights": fusion.weights,
                "factors": fusion.factors,
                "intercept": fusion.intercept,
                "sample_count": fusion.sample_count,
                "disagreement": fusion.disagreement,
                "lower_throughput": fusion.lower_throughput if fusion_applied else None,
                "reason": fusion.reason,
            }

            pre_calibration_y = dict(y_hat)
            pre_calibration_v = dict(v_hat)
            composite_version = self._composite_version(backends)

            stage = "calibration"
            calibration_started = time.perf_counter()
            skip_y_nodes = {"throughput_token_per_sec"} if fusion_applied else set()
            calibration = calibrate_prediction(
                y_hat,
                v_hat,
                config,
                features,
                self.evidence_store,
                composite_version,
                scenario=scenario,
                as_of_timestamp_utc=as_of_timestamp_utc,
                skip_y_nodes=skip_y_nodes,
            )
            y_hat = calibration.y_hat
            v_hat = calibration.v_hat
            timings[stage] = _elapsed_ms(calibration_started)
            calibration_trace = {
                "status": calibration.status,
                "reason": calibration.reason,
                "offsets_y": calibration.offsets_y,
                "offsets_v": calibration.offsets_v,
                "coverage_y": calibration.coverage_y,
                "coverage_v": calibration.coverage_v,
                "skipped_y_nodes": sorted(skip_y_nodes),
            }

            stage = "derived_outcomes"
            derived_started = time.perf_counter()
            y_hat = _rederive_cost_and_slo(raw_y, y_hat, config, features, candidate_graph)
            timings[stage] = _elapsed_ms(derived_started)
            timings["total"] = _elapsed_ms(started)

            trace = {
                "schema_version": 3,
                "prediction_timestamp_utc": prediction_timestamp,
                "as_of_timestamp_utc": as_of_timestamp_utc,
                "normalized_candidate": {
                    "job_config": deepcopy(config),
                    "job_features": deepcopy(features),
                },
                "context": context,
                "scenario": scenario,
                "method": list(method) if isinstance(method, list | tuple) else method,
                "components": components,
                "backends": backends,
                "raw": {"y_hat": raw_y, "v_hat": raw_v},
                "pre_calibration": {
                    "y_hat": pre_calibration_y,
                    "v_hat": pre_calibration_v,
                },
                "final": {"y_hat": dict(y_hat), "v_hat": dict(v_hat)},
                "composite": {"y_hat": dict(y_hat), "v_hat": dict(v_hat)},
                "fusion": fusion_trace,
                "calibration": calibration_trace,
                "timings_ms": timings,
                "composite_version": composite_version,
                "surrogate_version": composite_version,
                "metadata": {
                    "learning_enabled": as_of_timestamp_utc is not None,
                    "learning_requires_persisted_schema_v3_lineage": True,
                    "perfdb_mode": self.perfdb_mode,
                    "peer_mode": self.peer_mode,
                },
            }
            self.last_trace = trace
            self.version = composite_version
            event = {
                "event": "surrogate.prediction.completed",
                "component": "prediction.surrogate_composer",
                "purpose": event_purpose,
                **event_context,
                "trace_schema_version": 3,
                "scenario": scenario,
                "composite_version": composite_version,
                "inputs": {"raw": raw_inputs, "normalized": trace["normalized_candidate"]},
                "outputs": {"y_hat": dict(y_hat), "v_hat": dict(v_hat)},
                "component_statuses": {
                    name: component.get("status") for name, component in components.items()
                },
                "fusion_status": fusion.status,
                "calibration_status": calibration.status,
                "timings_ms": dict(timings),
                "trace": trace,
            }
            log.info(
                "surrogate prediction completed: scenario=%s version=%s total_ms=%s",
                scenario,
                composite_version,
                timings.get("total"),
            )
            self._emit(event)
            return dict(y_hat), dict(v_hat), trace
        except Exception as exc:
            timings[stage] = timings.get(stage, _elapsed_ms(started))
            timings["total"] = _elapsed_ms(started)
            failure = {
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "exception": exc,
                "traceback": "".join(traceback.format_exception(exc)),
            }
            if not started_emitted:
                self._emit(
                    {
                        "event": "surrogate.prediction.started",
                        "component": "prediction.surrogate_composer",
                        "purpose": event_purpose,
                        **event_context,
                        "inputs": {
                            "raw": raw_inputs,
                            "normalized": {
                                "job_config": config,
                                "job_features": features,
                            },
                        },
                        "candidate_graph": graph_request,
                        "method": list(method) if isinstance(method, list | tuple) else method,
                        "scenario": scenario,
                        "as_of_timestamp_utc": as_of_timestamp_utc,
                        "configuration": {
                            "composer_version": self.version,
                            "primary": type(self.primary).__name__,
                            "perfdb_mode": self.perfdb_mode,
                            "peer_mode": self.peer_mode,
                        },
                        "prediction_timestamp_utc": prediction_timestamp,
                    }
                )
            self.last_trace = {
                "schema_version": 3,
                "prediction_timestamp_utc": prediction_timestamp,
                "scenario": scenario,
                "method": list(method) if isinstance(method, list | tuple) else method,
                "as_of_timestamp_utc": as_of_timestamp_utc,
                "raw_candidate": raw_inputs,
                "normalized_candidate": {
                    "job_config": deepcopy(config),
                    "job_features": deepcopy(features),
                },
                "components": components,
                "backends": backends,
                "failure": failure,
                "timings_ms": timings,
            }
            event = {
                "event": "surrogate.prediction.failed",
                "component": "prediction.surrogate_composer",
                "purpose": event_purpose,
                **event_context,
                "trace_schema_version": 3,
                "scenario": scenario,
                "inputs": {
                    "raw": raw_inputs,
                    "normalized": self.last_trace["normalized_candidate"],
                },
                "candidate_graph": graph_request,
                "failure": failure,
                "timings_ms": dict(timings),
                "components": components,
                "backends": backends,
                "available_trace": self.last_trace,
            }
            log.error(
                "surrogate prediction failed: stage=%s error=%s",
                stage,
                failure["message"],
            )
            self._emit(event)
            raise

    def _run_perfdb(self, candidate, candidate_graph, method, scenario):
        started = time.perf_counter()
        if self.perfdb_mode == "off":
            return (
                SurrogateEstimate(
                    status="off",
                    source="perfdb",
                    metadata={"mode": self.perfdb_mode},
                ),
                _elapsed_ms(started),
            )
        if self.perfdb_backend is None:
            return (
                SurrogateEstimate(
                    status="unconfigured",
                    source="perfdb",
                    metadata={"mode": self.perfdb_mode, "reason": "no_csv_backend"},
                ),
                _elapsed_ms(started),
            )
        try:
            estimate = self.perfdb_backend.estimate(
                candidate,
                candidate_graph=candidate_graph,
                method=method,
                scenario=scenario,
            )
        except Exception as exc:
            estimate = SurrogateEstimate(
                status="failed",
                version=getattr(self.perfdb_backend, "version", None),
                source="perfdb",
                metadata={"error_type": type(exc).__name__, "error": str(exc)},
            )
        estimate.metadata = {**estimate.metadata, "mode": self.perfdb_mode}
        return estimate, _elapsed_ms(started)

    def _run_peers(self, job_config, job_features, scenario):
        started = time.perf_counter()
        if self.peer_mode == "off" or scenario == "peak_all_multiturn_stress":
            return {}, _elapsed_ms(started)
        try:
            results = self.peer_client.predict(
                job_config,
                job_features,
                scenario=scenario,
            )
        except Exception as exc:
            results = {
                "peer_client": {
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            }
        entries = {}
        for name, result in results.items():
            entries[name] = {
                "status": result.get("status", "failed"),
                "version": result.get("backend_version"),
                "coverage": {},
                "spread": {},
                "metadata": {
                    "request_hash": result.get("request_hash"),
                    "task": result.get("task"),
                    "ignored_fields": list(result.get("ignored_fields") or []),
                    "error": result.get("error"),
                    "error_type": result.get("error_type"),
                    "mode": self.peer_mode,
                },
                "y_hat": _peer_y(result),
                "v_hat": {},
            }
        return entries, _elapsed_ms(started)

    def _composite_version(self, backends: dict[str, dict[str, Any]]) -> str:
        versions = sorted((name, entry.get("version")) for name, entry in backends.items())
        identity = (
            "koi-surrogate-v3",
            self.perfdb_mode,
            self.peer_mode,
            versions,
        )
        digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:12]
        return f"koi-surrogate-v3:{digest}"

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            if hasattr(self.event_sink, "emit"):
                self.event_sink.emit(event)
            else:
                self.event_sink(event)
        except Exception:
            log.exception("surrogate event sink failed")


def _estimate_trace(estimate: SurrogateEstimate, timing_ms: float) -> dict[str, Any]:
    return {
        "status": estimate.status,
        "version": estimate.version,
        "coverage": dict(estimate.coverage),
        "spread": dict(estimate.spread),
        "metadata": deepcopy(estimate.metadata),
        "y_hat": dict(estimate.y_hat),
        "v_hat": dict(estimate.v_hat),
        "timing_ms": timing_ms,
    }


def _backend_trace(estimate: SurrogateEstimate) -> dict[str, Any]:
    trace = _estimate_trace(estimate, 0.0)
    trace.pop("timing_ms")
    return trace


def _analytic_trace(values: dict[str, float], timing_ms: float, version: str):
    return {
        "status": "success" if values else "omitted",
        "version": version,
        "coverage": dict.fromkeys(values, 1.0),
        "spread": {},
        "metadata": {"omits_missing_inputs": True, "dp_invariant": True},
        "v_hat": dict(values),
        "y_hat": {},
        "timing_ms": timing_ms,
    }


def _coverage_blend(base: dict, measured: dict, coverage: dict) -> dict[str, float]:
    output = dict(base)
    for node, value in measured.items():
        weight = max(0.0, min(1.0, float(coverage.get(node, 0.0))))
        if output.get(node) is None:
            if weight >= _MEASURED_ONLY_MIN_COVERAGE:
                output[node] = float(value)
        elif weight > 0:
            output[node] = (1.0 - weight) * float(output[node]) + weight * float(value)
    return output


def _peer_y(result: dict) -> dict[str, float]:
    mapping = {
        "output_tps": "throughput_token_per_sec",
        "ttft_ms_p99": "p99_ttft_ms",
        "tpot_ms_p99": "p99_tpot_ms",
    }
    return {
        target: float(result[source])
        for source, target in mapping.items()
        if result.get(source) is not None
    }


def _peer_component_status(entries: dict[str, dict[str, Any]]) -> str:
    statuses = {entry.get("status") for entry in entries.values()}
    if not statuses:
        return "omitted"
    if statuses == {"success"}:
        return "success"
    if "success" in statuses:
        return "partial"
    return "failed"


def _rederive_cost_and_slo(raw_y, final_y, config, features, candidate_graph):
    output = dict(final_y)
    raw_throughput = raw_y.get("throughput_token_per_sec")
    throughput = output.get("throughput_token_per_sec")
    if (
        raw_y.get("cost_per_token") is not None
        and raw_throughput is not None
        and throughput is not None
        and float(raw_throughput) > 0
        and float(throughput) > 0
        and float(raw_throughput) != float(throughput)
    ):
        output["cost_per_token"] = (
            float(raw_y["cost_per_token"]) * float(raw_throughput) / float(throughput)
        )

    requested_y = set(getattr(candidate_graph, "y", ()) or ())
    wants_slo = "slo_margin" in raw_y or "slo_margin" in requested_y
    latency_changed = any(
        output.get(node) != raw_y.get(node) for node in ("p99_ttft_ms", "p99_tpot_ms")
    )
    if wants_slo and latency_changed:
        values = {**features, **config}
        margins = []
        if values.get("target_p99_ttft_ms") is not None and output.get("p99_ttft_ms") is not None:
            margins.append(float(values["target_p99_ttft_ms"]) - float(output["p99_ttft_ms"]))
        if values.get("target_p99_tpot_ms") is not None and output.get("p99_tpot_ms") is not None:
            margins.append(float(values["target_p99_tpot_ms"]) - float(output["p99_tpot_ms"]))
        if margins:
            output["slo_margin"] = min(margins)
    return output


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _candidate_graph_request(candidate_graph: Any) -> dict[str, Any]:
    try:
        return {
            "x_nodes": list(getattr(candidate_graph, "x", ()) or ()),
            "v_nodes": list(getattr(candidate_graph, "v", ()) or ()),
            "y_nodes": list(getattr(candidate_graph, "y", ()) or ()),
            "node_ids": list((getattr(candidate_graph, "node_table", {}) or {}).keys()),
            "edge_ids": list((getattr(candidate_graph, "edge_table", {}) or {}).keys()),
        }
    except Exception as exc:
        return {"summary_error": exc, "candidate_graph": candidate_graph}
