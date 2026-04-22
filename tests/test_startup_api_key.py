"""Tests for startup API-key validation.

Real-agent startup requires a provider-specific key:
- KOI_API_KEY (or OPENROUTER_API_KEY) when KOI_LLM_PROVIDER=openrouter (default)
- ANTHROPIC_API_KEY when KOI_LLM_PROVIDER=anthropic

KOI_TEST_FAKE_DECIDE=1 bypasses the agent and must keep working without any key.
"""

import os
from contextlib import contextmanager

import pytest

from koi.server import app, lifespan

_LLM_ENV_VARS = (
    "KOI_LLM_PROVIDER",
    "KOI_BASE_URL",
    "KOI_AGENT_MODEL",
    "KOI_LLM_MODEL",
    "KOI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
)


@contextmanager
def _env(tmp_path, **overrides):
    """Temporarily set env vars, clearing any set to None.

    Always redirects DB paths to a tmp dir so tests never touch real data.
    Also clears any ambient LLM env vars not in overrides, so the dev shell
    can't leak keys into the test.
    """
    forced = {
        "KOI_MEMORY_PATH": str(tmp_path / "memory.db"),
        "KOI_RUNTIME_STATE_PATH": str(tmp_path / "runtime.db"),
        "KOI_PERFDB_PATH": "./perfdb/perfdb_all.csv",  # read-only, safe
        "ORCA_URL": "",  # avoid network
    }
    overrides = {**forced, **overrides}
    saved = {k: os.environ.get(k) for k in overrides}
    for ambient in _LLM_ENV_VARS:
        if ambient not in overrides:
            saved[ambient] = os.environ.get(ambient)
            os.environ.pop(ambient, None)
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
        with _env(tmp_path, KOI_TEST_FAKE_DECIDE="1"):
            await _start_and_stop()

    @pytest.mark.asyncio
    async def test_fake_mode_starts_with_empty_key(self, tmp_path):
        with _env(tmp_path, KOI_TEST_FAKE_DECIDE="1", KOI_API_KEY=""):
            await _start_and_stop()


class TestOpenRouterProviderRequiresKey:
    """Default provider 'openrouter' requires KOI_API_KEY or OPENROUTER_API_KEY."""

    @pytest.mark.asyncio
    async def test_missing_key_raises(self, tmp_path):
        with _env(tmp_path):
            with pytest.raises(RuntimeError, match="KOI_API_KEY"):
                await _start_and_stop()

    @pytest.mark.asyncio
    async def test_empty_key_raises(self, tmp_path):
        with _env(tmp_path, KOI_API_KEY="   "):
            with pytest.raises(RuntimeError, match="KOI_API_KEY"):
                await _start_and_stop()

    @pytest.mark.asyncio
    async def test_openrouter_api_key_alias_accepted(self, tmp_path):
        """OPENROUTER_API_KEY works as an alias for KOI_API_KEY."""
        with _env(tmp_path, OPENROUTER_API_KEY="sk-or-placeholder"):
            try:
                await _start_and_stop()
            except RuntimeError as e:
                assert "KOI_API_KEY" not in str(e)
                assert "OPENROUTER_API_KEY" not in str(e)
            except Exception:
                # Any later failure (network, etc.) is fine — past the key gate.
                pass


class TestAnthropicProviderRequiresKey:
    """KOI_LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY."""

    @pytest.mark.asyncio
    async def test_missing_key_raises(self, tmp_path):
        with _env(tmp_path, KOI_LLM_PROVIDER="anthropic"):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                await _start_and_stop()

    @pytest.mark.asyncio
    async def test_empty_key_raises(self, tmp_path):
        with _env(tmp_path, KOI_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY=""):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                await _start_and_stop()

    @pytest.mark.asyncio
    async def test_key_accepted(self, tmp_path):
        with _env(
            tmp_path,
            KOI_LLM_PROVIDER="anthropic",
            ANTHROPIC_API_KEY="sk-ant-placeholder",
        ):
            try:
                await _start_and_stop()
            except RuntimeError as e:
                assert "ANTHROPIC_API_KEY" not in str(e)
            except Exception:
                pass
