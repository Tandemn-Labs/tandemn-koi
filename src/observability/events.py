"""Fail-open, chronological JSONL event logging for Koi."""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("koi.observability.events")


@runtime_checkable
class EventSink(Protocol):
    """Destination for structured Koi events."""

    def emit(self, event: Mapping[str, Any]) -> None: ...


class CallbackEventSink:
    """Adapt a mapping callback to the EventSink interface."""

    def __init__(self, callback: Callable[[Mapping[str, Any]], None]) -> None:
        self.callback = callback

    def emit(self, event: Mapping[str, Any]) -> None:
        self.callback(event)


class CompositeEventSink:
    """Fan events out to independent sinks without propagating sink failures."""

    def __init__(self, *sinks: EventSink) -> None:
        self.sinks = tuple(sinks)

    def emit(self, event: Mapping[str, Any]) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception:
                log.exception("Koi composite event sink failed")


class ChronologicalEventLogger:
    """Append a globally ordered event stream as compact, durable JSONL."""

    schema_name = "koi-event"
    schema_version = 1

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        flush: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex
        self.flush = bool(flush)
        self._lock = RLock()
        self._sequence = _last_sequence(self.path)
        self._handle = self.path.open("a", encoding="utf-8")

    @property
    def event_count(self) -> int:
        with self._lock:
            return self._sequence

    def emit(self, event: Mapping[str, Any]) -> None:
        """Serialize and append one event; logging failures never escape."""
        try:
            with self._lock:
                self._sequence += 1
                sequence = self._sequence
                epoch = time.time()
                monotonic_ns = time.monotonic_ns()
                payload = _jsonable(dict(event))
                if not isinstance(payload, dict):
                    payload = {"payload": payload}
                event_name = str(payload.get("event") or "logger.unknown")
                component = str(
                    payload.get("component") or event_name.partition(".")[0] or "unknown"
                )
                record = {
                    **payload,
                    "schema_name": self.schema_name,
                    "schema_version": self.schema_version,
                    "sequence": sequence,
                    "timestamp_utc": datetime.fromtimestamp(epoch, UTC).isoformat(),
                    "timestamp_epoch_s": epoch,
                    "monotonic_ns": monotonic_ns,
                    "run_id": self.run_id,
                    "event": event_name,
                    "component": component,
                }
                encoded = json.dumps(
                    record,
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                self._handle.write(encoded + "\n")
                if self.flush:
                    self._handle.flush()
                    os.fsync(self._handle.fileno())
        except Exception:
            log.exception("Koi event serialization/write failed")
            self._emit_fallback(event)

    def _emit_fallback(self, event: Mapping[str, Any]) -> None:
        """Best-effort minimal record when normal conversion or writing fails."""
        try:
            with self._lock:
                epoch = time.time()
                fallback = {
                    "schema_name": self.schema_name,
                    "schema_version": self.schema_version,
                    "sequence": self._sequence,
                    "timestamp_utc": datetime.fromtimestamp(epoch, UTC).isoformat(),
                    "timestamp_epoch_s": epoch,
                    "monotonic_ns": time.monotonic_ns(),
                    "run_id": self.run_id,
                    "event": "logger.error",
                    "component": "observability.event_logger",
                    "purpose": "record an event logging failure without interrupting Koi",
                    "original_event": _safe_repr(event),
                }
                self._handle.write(
                    json.dumps(fallback, ensure_ascii=True, separators=(",", ":")) + "\n"
                )
                if self.flush:
                    self._handle.flush()
                    os.fsync(self._handle.fileno())
        except Exception:
            log.exception("Koi logger.error fallback failed")

    def close(self) -> None:
        """Flush and close the event stream."""
        try:
            with self._lock:
                if self._handle.closed:
                    return
                self._handle.flush()
                if self.flush:
                    os.fsync(self._handle.fileno())
                self._handle.close()
        except Exception:
            log.exception("Koi event logger close failed")

    def __enter__(self) -> ChronologicalEventLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _jsonable(value: Any, seen: set[int] | None = None) -> Any:
    """Convert common Koi values to JSON without importing optional libraries."""
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Enum):
        return _jsonable(value.value, seen)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "module": type(value).__module__,
            "message": str(value),
            "args": _jsonable(value.args, seen),
        }

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return {"serialization": "cycle", "repr": _safe_repr(value)}

    if is_dataclass(value) and not isinstance(value, type):
        seen.add(identity)
        try:
            return {field.name: _jsonable(getattr(value, field.name), seen) for field in fields(value)}
        finally:
            seen.remove(identity)
    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            return {str(key): _jsonable(item, seen) for key, item in value.items()}
        finally:
            seen.remove(identity)
    if isinstance(value, list | tuple | set | frozenset):
        seen.add(identity)
        try:
            items = list(value)
            if isinstance(value, set | frozenset):
                items.sort(key=_safe_repr)
            return [_jsonable(item, seen) for item in items]
        finally:
            seen.remove(identity)

    module = type(value).__module__.partition(".")[0]
    if module == "numpy":
        try:
            converted = value.tolist() if hasattr(value, "tolist") else value.item()
            return _jsonable(converted, seen)
        except Exception:
            return _safe_repr(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict(), seen)
        except Exception:
            pass
    try:
        attributes = vars(value)
    except TypeError:
        return _safe_repr(value)
    seen.add(identity)
    try:
        return {
            "type": type(value).__name__,
            "attributes": {
                str(key): _jsonable(item, seen)
                for key, item in attributes.items()
                if not str(key).startswith("_")
            },
            "repr": _safe_repr(value),
        }
    finally:
        seen.remove(identity)


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}: repr failed>"


def _last_sequence(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        last_sequence = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                last_sequence = max(last_sequence, int(record.get("sequence", 0)))
        return last_sequence
    except Exception:
        log.exception("Could not recover event sequence from %s; starting at zero", path)
        return 0
