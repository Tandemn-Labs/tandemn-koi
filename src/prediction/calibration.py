"""Leakage-safe deployment-grouped throughput fusion and residual calibration."""

import math
from dataclasses import dataclass

import numpy as np

from src.prediction.normalization import (
    effective_batch_size,
    normalize_candidate_inputs,
    normalize_precision,
    normalize_workload_type,
)

MIN_DEPLOYMENTS = 5
MAX_DEPLOYMENTS = 32
CALIBRATION_K = 16
EVIDENCE_SCAN_LIMIT = 1000
DISTANCE_TAU = 1.0
HALF_LIFE_SEC = 7 * 24 * 60 * 60
DEFAULT_LOWER_QUANTILE = 0.05
WEIGHT_RIDGE = 0.25
INTERCEPT_RIDGE = 2.0

_Y_CALIBRATABLE = frozenset({"throughput_token_per_sec", "p99_ttft_ms", "p99_tpot_ms"})
_NONNEGATIVE_V = frozenset(
    {
        "comm_overhead_pct",
        "gpu_mem_used_fraction",
        "kv_cache_util",
        "kv_pressure_score",
        "mem_bandwidth_utilization",
        "per_tok_comm_bytes",
        "pipeline_bubble_fraction",
        "sm_utilization",
    }
)
_NUMERIC_SCALES = {
    "tp": 1.0,
    "pp": 1.0,
    "isl_token_avg": 1024.0,
    "osl_token_avg": 1024.0,
    "max_num_seq": 128.0,
    "effective_batch_size": 32.0,
    "request_arrival_rate": 10.0,
    "max_num_batched_tokens": 8192.0,
    "gpu_mem_util": 0.1,
    "num_nodes_per_chain": 1.0,
}


@dataclass(frozen=True)
class CalibrationResult:
    y_hat: dict[str, float]
    v_hat: dict[str, float]
    offsets_y: dict[str, float]
    offsets_v: dict[str, float]
    coverage_y: dict[str, float]
    coverage_v: dict[str, float]
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class FusionResult:
    throughput: float
    lower_throughput: float | None
    weights: dict[str, float]
    factors: dict[str, float]
    intercept: float
    sample_count: int
    disagreement: float
    status: str
    reason: str | None = None


def build_prediction_context(
    job_config: dict,
    job_features: dict,
    *,
    scenario: str,
) -> dict | None:
    """Build exact deployment scope plus numeric distance context."""
    config, features = normalize_candidate_inputs(job_config, job_features)
    values = {**features, **config}
    try:
        dp = int(values.get("dp") or 1)
        hard = {
            "model_id": _text(values.get("model_id")),
            "gpu_type": _text(values.get("gpu_type")),
            "precision": normalize_precision(values.get("precision") or values.get("weight_dtype")),
            "workload_type": normalize_workload_type(
                values.get("workload_type") or values.get("type")
            ),
            "engine_name": _text(values.get("engine_name")),
            "engine_version": _text(values.get("engine_version")),
            "scenario": _text(scenario),
            "dp": dp,
        }
    except (TypeError, ValueError, OverflowError):
        return None
    if any(value is None for value in hard.values()) or dp < 1:
        return None

    numeric = {}
    candidates = {
        "tp": values.get("tp", 1),
        "pp": values.get("pp", 1),
        "isl_token_avg": values.get("isl_token_avg"),
        "osl_token_avg": values.get("osl_token_avg"),
        "max_num_seq": values.get("max_num_seq") or values.get("max_num_seqs"),
        "effective_batch_size": effective_batch_size(values),
        "request_arrival_rate": values.get("request_arrival_rate"),
        "max_num_batched_tokens": values.get("max_num_batched_tokens"),
        "gpu_mem_util": values.get("gpu_mem_util"),
        "num_nodes_per_chain": values.get("num_nodes_per_chain"),
    }
    for name, raw in candidates.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            numeric[name] = value
    return {"hard": hard, "numeric": numeric}


def calibrate_prediction(
    y_hat: dict[str, float],
    v_hat: dict[str, float],
    job_config: dict,
    job_features: dict,
    evidence_store,
    surrogate_version: str | None,
    *,
    scenario: str,
    as_of_timestamp_utc: float | None,
    skip_y_nodes: set[str] | frozenset[str] = frozenset(),
    evidence_rows: list | None = None,
) -> CalibrationResult:
    """Apply one residual correction from prior, independently deployed evidence."""
    raw_y, raw_v = dict(y_hat), dict(v_hat)
    if evidence_store is None:
        return _calibration_noop(raw_y, raw_v, "no_evidence_store")
    if as_of_timestamp_utc is None:
        return _calibration_noop(raw_y, raw_v, "missing_as_of_timestamp_utc")
    if not surrogate_version:
        return _calibration_noop(raw_y, raw_v, "missing_surrogate_version")
    context = build_prediction_context(job_config, job_features, scenario=scenario)
    if context is None:
        return _calibration_noop(raw_y, raw_v, "incomplete_hard_context")

    ranked = []
    rows = (
        evidence_rows
        if evidence_rows is not None
        else load_evidence_snapshot(evidence_store, as_of_timestamp_utc)
    )
    for row in rows:
        lineage = getattr(row, "prediction_lineage", None) or {}
        row_version = lineage.get("composite_version") or lineage.get("surrogate_version")
        if int(lineage.get("schema_version", 0)) < 3 or row_version != surrogate_version:
            continue
        row_context = lineage.get("context")
        if not isinstance(row_context, dict) or not _same_hard_context(context, row_context):
            continue
        distance = _context_distance(context, row_context)
        if distance is not None:
            ranked.append((distance, row))
    ranked.sort(key=lambda item: (item[0], getattr(item[1], "row_id", "")))

    calibrated_y, offsets_y, coverage_y = _calibrate_nodes(
        raw_y,
        ranked,
        is_y=True,
        as_of_timestamp_utc=as_of_timestamp_utc,
        skip_nodes=set(skip_y_nodes) | (set(raw_y) - set(_Y_CALIBRATABLE)),
    )
    calibrated_v, offsets_v, coverage_v = _calibrate_nodes(
        raw_v,
        ranked,
        is_y=False,
        as_of_timestamp_utc=as_of_timestamp_utc,
        skip_nodes=set(),
    )
    learned = bool(offsets_y or offsets_v)
    return CalibrationResult(
        calibrated_y,
        calibrated_v,
        offsets_y,
        offsets_v,
        coverage_y,
        coverage_v,
        "learned" if learned else "insufficient_evidence",
        None if learned else "fewer_than_five_compatible_deployments",
    )


def learn_throughput_fusion(
    predictions: dict[str, float],
    versions: dict[str, str | None],
    job_config: dict,
    job_features: dict,
    evidence_store,
    *,
    scenario: str,
    as_of_timestamp_utc: float | None,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    evidence_rows: list | None = None,
) -> FusionResult:
    """Fit a constrained log-space ensemble from prior deployment groups."""
    lower_quantile = float(lower_quantile)
    if not 0.0 <= lower_quantile <= 0.5:
        raise ValueError("lower_quantile must be between 0.0 and 0.5")
    values = {
        name: float(value)
        for name, value in predictions.items()
        if value is not None and math.isfinite(float(value)) and float(value) > 0
    }
    primary = values.get("primary", next(iter(values.values()), 0.0))
    fallback = _fusion_fallback(values, primary)
    if len(values) < 2:
        return FusionResult(**{**fallback.__dict__, "reason": "fewer_than_two_predictions"})
    if evidence_store is None:
        return FusionResult(**{**fallback.__dict__, "reason": "no_evidence_store"})
    if as_of_timestamp_utc is None:
        return FusionResult(**{**fallback.__dict__, "reason": "missing_as_of_timestamp_utc"})
    if any(not versions.get(name) for name in values):
        return FusionResult(**{**fallback.__dict__, "reason": "missing_backend_version"})
    context = build_prediction_context(job_config, job_features, scenario=scenario)
    if context is None:
        return FusionResult(**{**fallback.__dict__, "reason": "incomplete_hard_context"})

    grouped: dict[str, list] = {}
    disagreement = _disagreement(values)
    rows = (
        evidence_rows
        if evidence_rows is not None
        else load_evidence_snapshot(evidence_store, as_of_timestamp_utc)
    )
    for row in rows:
        lineage = getattr(row, "prediction_lineage", None) or {}
        if int(lineage.get("schema_version", 0)) < 3:
            continue
        row_context = lineage.get("context")
        if not isinstance(row_context, dict) or not _same_hard_context(context, row_context):
            continue
        distance = _context_distance(context, row_context)
        observed = _observed(row, "throughput_token_per_sec", is_y=True)
        if distance is None or observed is None or observed <= 0:
            continue
        backends = lineage.get("backends") or {}
        historical: dict[str, float] = {}
        for name in values:
            entry = backends.get(name) or {}
            value = (entry.get("y_hat") or {}).get("throughput_token_per_sec")
            if entry.get("version") != versions[name] or value is None or float(value) <= 0:
                historical = {}
                break
            historical[name] = float(value)
        if historical:
            grouped.setdefault(_deployment_id(row), []).append(
                (historical, float(observed), distance, row)
            )

    samples = []
    for rows in grouped.values():
        historical = {name: float(np.mean([item[0][name] for item in rows])) for name in values}
        observed = float(np.mean([item[1] for item in rows]))
        distance = min(item[2] for item in rows)
        available = max(float(item[3].evidence_available_timestamp_utc) for item in rows)
        sample_weight = _sample_weight(distance, available, as_of_timestamp_utc)
        sample_weight *= math.exp(-(((_disagreement(historical) - disagreement) / 0.5) ** 2))
        samples.append(
            (
                {name: math.log(value) for name, value in historical.items()},
                math.log(observed),
                sample_weight,
                distance,
            )
        )
    samples.sort(key=lambda item: item[3])
    samples = samples[:MAX_DEPLOYMENTS]
    if len(samples) < MIN_DEPLOYMENTS:
        return FusionResult(
            **{
                **fallback.__dict__,
                "sample_count": len(samples),
                "reason": "fewer_than_five_compatible_deployments",
            }
        )

    names = tuple(values)
    weights, intercept, biases = _fit_fusion(samples, names)
    prediction_log = intercept + sum(
        weights[name] * (math.log(values[name]) + biases[name]) for name in names
    )
    residuals = _cross_fitted_residuals(samples, names)
    lower_shift = _weighted_quantile(residuals, lower_quantile)
    throughput = math.exp(prediction_log)
    return FusionResult(
        throughput=throughput,
        lower_throughput=min(throughput, math.exp(prediction_log + lower_shift)),
        weights=weights,
        factors={name: math.exp(bias) for name, bias in biases.items()},
        intercept=intercept,
        sample_count=len(samples),
        disagreement=disagreement,
        status="learned",
    )


def _calibrate_nodes(raw, ranked, *, is_y, as_of_timestamp_utc, skip_nodes):
    calibrated = dict(raw)
    offsets: dict[str, float] = {}
    coverage: dict[str, float] = {}
    for node, value in raw.items():
        if node in skip_nodes or value is None:
            continue
        grouped: dict[str, list[tuple[float, float, float]]] = {}
        for distance, row in ranked:
            lineage = getattr(row, "prediction_lineage", None) or {}
            predicted = (lineage.get("pre_calibration") or {}).get("y_hat" if is_y else "v_hat", {})
            observed = _observed(row, node, is_y=is_y)
            if predicted.get(node) is None or observed is None:
                continue
            predicted_value = float(predicted[node])
            use_log = is_y and predicted_value > 0 and observed > 0
            residual = (
                math.log(observed / predicted_value) if use_log else observed - predicted_value
            )
            available = float(row.evidence_available_timestamp_utc)
            grouped.setdefault(_deployment_id(row), []).append((distance, residual, available))
        samples = []
        for group in grouped.values():
            samples.append(
                (
                    min(item[0] for item in group),
                    float(np.mean([item[1] for item in group])),
                    max(item[2] for item in group),
                )
            )
        samples.sort(key=lambda item: item[0])
        samples = samples[:CALIBRATION_K]
        if len(samples) < MIN_DEPLOYMENTS:
            continue
        weights = [
            _sample_weight(distance, available, as_of_timestamp_utc)
            for distance, _, available in samples
        ]
        total_weight = sum(weights)
        if total_weight <= 0:
            continue
        residual = (
            sum(weight * sample[1] for weight, sample in zip(weights, samples, strict=True))
            / total_weight
        )
        node_coverage = math.exp(-((samples[0][0] / DISTANCE_TAU) ** 2))
        corrected = (
            float(value) * math.exp(node_coverage * residual)
            if is_y and float(value) > 0
            else float(value) + node_coverage * residual
        )
        if is_y or node in _NONNEGATIVE_V:
            corrected = max(0.0, corrected)
        calibrated[node] = corrected
        offsets[node] = corrected - float(value)
        coverage[node] = node_coverage
    return calibrated, offsets, coverage


def _calibration_noop(y_hat, v_hat, reason):
    return CalibrationResult(y_hat, v_hat, {}, {}, {}, {}, "disabled", reason)


def _fusion_fallback(values, primary):
    equal = 1.0 / max(1, len(values))
    return FusionResult(
        throughput=primary,
        lower_throughput=None,
        weights=dict.fromkeys(values, equal),
        factors={},
        intercept=0.0,
        sample_count=0,
        disagreement=_disagreement(values),
        status="insufficient_evidence",
    )


def load_evidence_snapshot(
    evidence_store, as_of: float | None, *, limit: int = EVIDENCE_SCAN_LIMIT
) -> list:
    """Load one bounded, point-in-time evidence snapshot for fusion and calibration."""
    if as_of is None:
        return []
    if hasattr(evidence_store, "get_rows_available_before"):
        return list(evidence_store.get_rows_available_before(as_of, limit=limit))
    if not hasattr(evidence_store, "get_all_rows"):
        return []
    return [
        row
        for row in evidence_store.get_all_rows(limit=limit)
        if getattr(row, "evidence_available_timestamp_utc", None) is not None
        and float(row.evidence_available_timestamp_utc) < as_of
    ]


def _same_hard_context(query: dict, historical) -> bool:
    return isinstance(historical, dict) and historical.get("hard") == query.get("hard")


def _context_distance(query: dict, historical: dict) -> float | None:
    query_numeric = query.get("numeric") or {}
    historical_numeric = historical.get("numeric") or {}
    names = set(query_numeric) & set(historical_numeric) & set(_NUMERIC_SCALES)
    try:
        return math.sqrt(
            sum(
                (
                    (float(query_numeric[name]) - float(historical_numeric[name]))
                    / _NUMERIC_SCALES[name]
                )
                ** 2
                for name in names
            )
        )
    except (TypeError, ValueError):
        return None


def _deployment_id(row) -> str:
    lineage = getattr(row, "prediction_lineage", None) or {}
    return str(
        getattr(row, "deployment_id", None)
        or lineage.get("deployment_id")
        or getattr(row, "row_id", id(row))
    )


def _observed(row, node: str, *, is_y: bool) -> float | None:
    if is_y:
        mean = getattr(row, "y_observed_mean", None) or {}
        if mean.get(node) is not None:
            return float(mean[node])
        trajectory = (getattr(row, "y_observed_trajectory", None) or {}).get(node)
    else:
        trajectory = (getattr(row, "V_observed_trajectory", None) or {}).get(node)
    if trajectory is None:
        return None
    array = np.asarray(trajectory, dtype=float)
    return float(np.mean(array)) if array.size else None


def _sample_weight(distance: float, available: float, as_of: float) -> float:
    distance_weight = math.exp(-((distance / DISTANCE_TAU) ** 2))
    age = max(0.0, as_of - available)
    return distance_weight * math.exp(-math.log(2.0) * age / HALF_LIFE_SEC)


def _disagreement(values: dict[str, float]) -> float:
    logs = [math.log(value) for value in values.values() if value > 0]
    return max(logs) - min(logs) if len(logs) > 1 else 0.0


def _weighted_mean(samples: list[tuple[float, float]]) -> float:
    total = sum(weight for _, weight in samples)
    return sum(value * weight for value, weight in samples) / max(total, 1e-12)


def _weighted_quantile(samples: list[tuple[float, float]], quantile: float) -> float:
    ordered = sorted(samples, key=lambda item: item[0])
    threshold = quantile * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0] if ordered else 0.0


def _simplex(names: tuple[str, ...], units: int = 20):
    def build(index: int, remaining: int, current: list[int]):
        if index == len(names) - 1:
            yield dict(zip(names, [*current, remaining], strict=True))
            return
        for value in range(remaining + 1):
            yield from build(index + 1, remaining - value, [*current, value])

    for integers in build(0, units, []):
        yield {name: value / units for name, value in integers.items()}


def _fit_fusion(samples, names: tuple[str, ...]):
    total_weight = sum(sample[2] for sample in samples)
    shrinkage = total_weight / (total_weight + INTERCEPT_RIDGE)
    biases = {
        name: _weighted_mean(
            [(target - predictors[name], weight) for predictors, target, weight, _ in samples]
        )
        * shrinkage
        for name in names
    }
    prior = 1.0 / len(names)
    best = (math.inf, dict.fromkeys(names, prior), 0.0)
    for weights in _simplex(names):
        residuals = []
        for predictors, target, sample_weight, _ in samples:
            predicted = sum(weights[name] * (predictors[name] + biases[name]) for name in names)
            residuals.append((target - predicted, sample_weight))
        raw_intercept = _weighted_mean(residuals)
        intercept = raw_intercept * total_weight / (total_weight + INTERCEPT_RIDGE)
        loss = sum(weight * (residual - intercept) ** 2 for residual, weight in residuals)
        loss += WEIGHT_RIDGE * sum((weights[name] - prior) ** 2 for name in names)
        if loss < best[0]:
            best = (loss, weights, intercept)
    return best[1], best[2], biases


def _cross_fitted_residuals(samples, names: tuple[str, ...]):
    output = []
    for index, held_out in enumerate(samples):
        training = [sample for sample_index, sample in enumerate(samples) if sample_index != index]
        weights, intercept, biases = _fit_fusion(training, names)
        predictors, target, sample_weight, _ = held_out
        predicted = intercept + sum(
            weights[name] * (predictors[name] + biases[name]) for name in names
        )
        output.append((target - predicted, sample_weight))
    return output


def _text(value) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
