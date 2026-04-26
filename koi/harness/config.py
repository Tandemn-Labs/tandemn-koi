"""Feature flag helpers for incremental harness rollout."""

from __future__ import annotations

import os


def harness_enabled() -> bool:
    return os.environ.get("KOI_HARNESS", "0").lower() in {"1", "true", "yes", "on"}


def prompt_enabled(prompt: str) -> bool:
    if not harness_enabled():
        return False
    raw = os.environ.get("KOI_HARNESS_PROMPTS", "").strip()
    if not raw:
        return True
    enabled = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return prompt.lower() in enabled


def fail_open_enabled() -> bool:
    return os.environ.get("KOI_HARNESS_FAIL_OPEN", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
