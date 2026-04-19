"""Tests for the conditional ANTHROPIC_API_KEY startup validation.

Phase 2f of contract-hardening. The rule: the real-agent path REQUIRES
a valid API key at startup; the KOI_TEST_FAKE_DECIDE=1 path must keep
working WITHOUT a key (sim / CI / local dev).

This is a narrow, targeted fix — not a live API probe.
"""

import os
from contextlib import contextmanager

import pytest

from koi.server import app, lifespan


@contextmanager
def _env(tmp_path, **overrides):
    """Temporarily set env vars, clearing any that are set to None.

    Always redirects DB paths to a tmp dir so tests never touch the real
    ./data/koi_memory.db or koi_runtime.db.
    """
    forced = {
        "KOI_MEMORY_PATH": str(tmp_path / "memory.db"),
        "KOI_RUNTIME_STATE_PATH": str(tmp_path / "runtime.db"),
        "KOI_PERFDB_PATH": "./perfdb/perfdb_all.csv",  # read-only, safe
        "ORCA_URL": "",  # avoid network
    }
    overrides = {**forced, **overrides}
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def _start_and_stop():
    """Run the lifespan context: raises if startup validation fails."""
    async with lifespan(app):
        pass


class TestFakeModeBypassesKeyCheck:
    @pytest.mark.asyncio
    async def test_fake_mode_starts_without_key(self, tmp_path):
        """KOI_TEST_FAKE_DECIDE=1 must work with no API key."""
        with _env(tmp_path, KOI_TEST_FAKE_DECIDE="1", ANTHROPIC_API_KEY=None):
            await _start_and_stop()  # should NOT raise

    @pytest.mark.asyncio
    async def test_fake_mode_starts_with_empty_key(self, tmp_path):
        with _env(tmp_path, KOI_TEST_FAKE_DECIDE="1", ANTHROPIC_API_KEY=""):
            await _start_and_stop()


class TestRealModeRequiresKey:
    @pytest.mark.asyncio
    async def test_missing_key_raises(self, tmp_path):
        with _env(tmp_path, KOI_TEST_FAKE_DECIDE=None, ANTHROPIC_API_KEY=None):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is required"):
                await _start_and_stop()

    @pytest.mark.asyncio
    async def test_empty_key_raises(self, tmp_path):
        with _env(tmp_path, KOI_TEST_FAKE_DECIDE=None, ANTHROPIC_API_KEY="   "):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is required"):
                await _start_and_stop()

    @pytest.mark.asyncio
    async def test_malformed_key_raises(self, tmp_path):
        with _env(tmp_path, KOI_TEST_FAKE_DECIDE=None, ANTHROPIC_API_KEY="not-an-anthropic-key"):
            with pytest.raises(RuntimeError, match="malformed"):
                await _start_and_stop()

    @pytest.mark.asyncio
    async def test_valid_prefix_accepted(self, tmp_path):
        """Format-only check — no live API probe. 'sk-ant-' prefix is enough."""
        # We can't fully boot the real agent in a unit test (it wants an
        # Anthropic client), but we can verify the key check itself doesn't
        # reject a correctly-prefixed placeholder. KOI_TEST_FAKE_DECIDE=1 lets
        # us skip the actual KoiAgent construction while still triggering the
        # key-validation branch by clearing it afterwards — but the cleanest
        # assertion is: with an sk-ant- key in real mode, the key check passes
        # and any later failure is NOT about the key format. We approximate by
        # checking that the RuntimeError message does not mention the key.
        with _env(
            tmp_path,
            KOI_TEST_FAKE_DECIDE=None,
            ANTHROPIC_API_KEY="sk-ant-placeholder-for-format-check",
        ):
            try:
                await _start_and_stop()
            except RuntimeError as e:
                assert "ANTHROPIC_API_KEY" not in str(e), (
                    f"Valid-prefix key should not be rejected by format check: {e}"
                )
            except Exception:
                # Any other failure (network, Anthropic client init) is
                # acceptable — it's past the key-format gate.
                pass
