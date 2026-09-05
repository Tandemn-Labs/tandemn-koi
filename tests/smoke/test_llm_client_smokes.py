"""Smoke tests for LLM client wrappers."""

import sys
import types
import unittest
from unittest.mock import patch

from src.agent.llm_clients import MockLLMClient, OpenAICompatClient, RecordingLLMClient


class RecordingLLMClientSmokeTests(unittest.TestCase):
    """Verify the recording wrapper preserves client behavior."""

    def test_records_transcript(self):
        """A successful call stores request, response, and timing."""
        client = RecordingLLMClient(MockLLMClient(["answer"]))
        messages = [{"role": "user", "content": "question"}]

        response = client.complete(messages)

        self.assertEqual(response, "answer")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["messages"], messages)
        self.assertEqual(client.calls[0]["response"], "answer")
        self.assertIn("elapsed_sec", client.calls[0])

    def test_reraises_inner_client_errors(self):
        """A failing inner client re-raises and records error metadata only."""
        client = RecordingLLMClient(_FailingLLMClient())

        with self.assertRaises(RuntimeError):
            client.complete([{"role": "user", "content": "question"}])

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["call_index"], 0)
        self.assertEqual(client.calls[0]["error"], "RuntimeError('boom')")
        self.assertNotIn("messages", client.calls[0])


class OpenAICompatClientSmokeTests(unittest.TestCase):
    """Verify provider-specific request settings reach the compatible client."""

    def test_foundry_omits_temperature_and_uses_completion_token_limit(self):
        fake_openai = _FakeOpenAI()
        with patch.dict(
            sys.modules, {"openai": types.SimpleNamespace(OpenAI=lambda **_kwargs: fake_openai)}
        ):
            client = OpenAICompatClient(
                base_url="https://deepseek-models.openai.azure.com/openai/v1",
                model="DeepSeek-V4-Pro",
                api_key="test-key",
                temperature=None,
                max_tokens=20_000,
                token_limit_param="max_completion_tokens",
            )
            response = client.complete([{"role": "user", "content": "plan"}])

        self.assertEqual(response, "answer")
        self.assertEqual(
            fake_openai.calls,
            [
                {
                    "model": "DeepSeek-V4-Pro",
                    "messages": [{"role": "user", "content": "plan"}],
                    "max_completion_tokens": 20_000,
                }
            ],
        )


class _FailingLLMClient:
    """LLM test double that always fails."""

    def complete(self, messages):
        """Raise a deterministic completion error."""
        raise RuntimeError("boom")


class _FakeOpenAI:
    """Minimal OpenAI SDK substitute retaining create payloads."""

    def __init__(self):
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="answer"))]
        )


if __name__ == "__main__":
    unittest.main()
