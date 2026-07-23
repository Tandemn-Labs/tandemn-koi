"""LLM client adapters for the Koi agent harness.

The harness contract is one method: complete(messages) -> str, where
messages are [{"role": "system" | "user" | "assistant", "content": str}].
Anything that satisfies it can drive the root planner or the specialists:
frontier APIs, or any open model (Gemma, Qwen, Llama, DeepSeek, ...)
served behind an OpenAI-compatible endpoint by vLLM, Ollama, llama.cpp,
SGLang, or TGI.

Adapters here:
    OpenAICompatClient  Chat-completions client for any OpenAI-compatible
                        endpoint. fold_system=True merges the system
                        prompt into the first user turn for chat
                        templates without a system role (Gemma).
    MockLLMClient       Scripted responses for harness tests. No network.

Example, Gemma 3 27B served locally by vLLM:

    vllm serve google/gemma-3-27b-it --port 8000

    root_llm = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="google/gemma-3-27b-it",
        fold_system=True,
        temperature=0.4,
    )
    specialist_llm = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        model="google/gemma-3-27b-it",
        fold_system=True,
        temperature=0.2,
    )
    harness = KoiAgentHarness(
        llm_client=root_llm,
        specialist_llm_client=specialist_llm,
        resource_map=resource_map,
        config={"k_p": 3, "max_history_messages": 0},
    )

For an 8K-context model (Gemma 2), set config={"k_max": 24,
"max_history_messages": 24, "stdout_limit": 1200}.
"""

import logging
import time
from typing import Any, cast

log = logging.getLogger("koi.llm_clients")


class OpenAICompatClient:
    """Chat-completions client for any OpenAI-compatible endpoint.

    Covers vLLM, Ollama (/v1), llama.cpp server, SGLang, TGI, and the
    hosted APIs that speak the same protocol. The openai package is
    imported lazily so this module loads without it.

    Args:
        base_url: Endpoint base, e.g. "http://localhost:8000/v1".
        model: Served model name as the endpoint knows it.
        api_key: Key if the endpoint enforces one. Local servers
            usually accept any non-empty string.
        fold_system: Merge the system message into the first user turn.
            Required for chat templates with no system role (Gemma);
            harmless elsewhere.
        temperature: Sampling temperature.
        max_tokens: Completion cap per call.
        timeout_sec: Per-request timeout.
        extra: Additional chat.completions.create kwargs, e.g.
            {"seed": 7} for engines that support seeding.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        fold_system: bool = False,
        temperature: float | None = 0.4,
        max_tokens: int = 8192,
        timeout_sec: float = 120.0,
        extra: dict | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatClient needs the openai package: pip install openai"
            ) from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)
        self.model = model
        self.fold_system = bool(fold_system)
        self.temperature = None if temperature is None else float(temperature)
        self.max_tokens = int(max_tokens)
        self.extra = dict(extra or {})

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Run one chat completion and return the assistant text."""
        payload = self._fold(messages) if self.fold_system else list(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, payload),
            self._token_limit_param(): self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        kwargs.update(self.extra)
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _token_limit_param(self) -> str:
        """Return the token-limit parameter supported by the target model."""
        model = self.model.lower()
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            return "max_completion_tokens"
        return "max_tokens"

    @staticmethod
    def _fold(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Merge system messages into the first user turn.

        Gemma's chat template rejects the system role; folding preserves
        the instructions while keeping strict user/assistant alternation.
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        if not system_parts:
            return rest
        prefix = "\n\n".join(system_parts)
        if rest and rest[0]["role"] == "user":
            first = dict(rest[0])
            first["content"] = f"{prefix}\n\n{first['content']}"
            return [first, *rest[1:]]
        return [{"role": "user", "content": prefix}, *rest]


class RecordingLLMClient:
    """Wrap an LLM client and retain request/response transcripts."""

    def __init__(
        self,
        inner,
        *,
        live: bool = False,
        print_messages: bool = False,
        log_string_limit: int = 1200,
    ):
        """Create a recording wrapper around an object exposing ``complete``."""
        self.inner = inner
        self.live = bool(live)
        self.print_messages = bool(print_messages)
        self.log_string_limit = int(log_string_limit)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Run one completion, record it, and return the assistant text."""
        call_index = len(self.calls)
        copied_messages = [dict(message) for message in messages]
        if self.live:
            log.info("LLM call %d started", call_index)
            if self.print_messages:
                for message in copied_messages:
                    log.info(
                        "LLM call %d [%s]\n%s",
                        call_index,
                        message.get("role", "unknown"),
                        self._compact(message.get("content", "")),
                    )

        started = time.time()
        try:
            response = self.inner.complete(messages)
        except Exception:
            raise

        elapsed = round(time.time() - started, 3)
        call = {"elapsed_sec": elapsed, "messages": copied_messages, "response": response}
        self.calls.append(call)
        if self.live:
            log.info("LLM response %d (%.3fs)\n%s", call_index, elapsed, self._compact(response))
        return response

    def _compact(self, text: str) -> str:
        """Return a bounded text preview for live logs."""
        if self.log_string_limit <= 0 or len(text) <= self.log_string_limit:
            return text
        return text[: self.log_string_limit] + f"\n... [truncated at {self.log_string_limit} chars]"


class MockLLMClient:
    """Scripted client for harness tests. Returns canned responses in order.

    Args:
        responses: Assistant responses, popped front to back. The last
            response repeats once the script is exhausted so bounded
            loops terminate deterministically.

    Attributes:
        calls: Every messages list received, for assertions.
    """

    def __init__(self, responses: list[str]):
        if not responses:
            raise ValueError("MockLLMClient needs at least one response")
        self._responses = list(responses)
        self._index = 0
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the next scripted response."""
        self.calls.append([dict(m) for m in messages])
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response
