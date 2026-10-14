"""Load query-language profiles without coupling them to source adapters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROFILE_NAME = re.compile(r"^[a-z0-9_-]+$")
_BUILTIN_PROFILES = Path(__file__).with_name("profiles")


@dataclass(frozen=True)
class MetricQueryProfile:
    """Queries for one metric source expressed in one query language."""

    source: str
    language: str
    queries: dict[str, str]
    version: int = 1

    def __post_init__(self) -> None:
        if not self.source or not self.language:
            raise ValueError("metric query profile requires source and language")
        if self.version < 1:
            raise ValueError("metric query profile version must be positive")
        if not self.queries:
            raise ValueError("metric query profile requires at least one query")
        if any(
            not name or not isinstance(query, str) or not query.strip()
            for name, query in self.queries.items()
        ):
            raise ValueError("metric query profile names and queries must be non-empty strings")


def load_builtin_profile(source: str, language: str) -> MetricQueryProfile:
    """Load the checked-in profile for one source and query language."""

    if not _PROFILE_NAME.fullmatch(source) or not _PROFILE_NAME.fullmatch(language):
        raise ValueError("profile source and language must use lowercase identifier characters")
    return load_profile(_BUILTIN_PROFILES / f"{source}_{language}.json")


def load_profile(path: str | Path) -> MetricQueryProfile:
    """Load and validate a metric query profile from JSON."""

    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("metric query profile must be a JSON object")
    return _profile_from_dict(raw)


def _profile_from_dict(raw: dict[str, Any]) -> MetricQueryProfile:
    queries = raw.get("queries")
    if not isinstance(queries, dict):
        raise ValueError("metric query profile queries must be an object")
    return MetricQueryProfile(
        source=str(raw.get("source") or ""),
        language=str(raw.get("language") or ""),
        version=int(raw.get("version", 1)),
        queries={str(name): query for name, query in queries.items()},
    )
