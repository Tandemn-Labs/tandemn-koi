"""Small shared ID helpers for harness menus."""

from __future__ import annotations

import string


def action_id(index: int) -> str:
    if index < len(string.ascii_lowercase):
        return string.ascii_lowercase[index]
    return f"a{index + 1}"
