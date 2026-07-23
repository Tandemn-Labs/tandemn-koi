"""Smoke tests for LLM client wrappers."""

import unittest

from src.agent.llm_clients import MockLLMClient, RecordingLLMClient


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
        """A failing inner client re-raises and records no transcript."""
        client = RecordingLLMClient(_FailingLLMClient())

        with self.assertRaises(RuntimeError):
            client.complete([{"role": "user", "content": "question"}])

        self.assertEqual(client.calls, [])


class _FailingLLMClient:
    """LLM test double that always fails."""

    def complete(self, messages):
        """Raise a deterministic completion error."""
        raise RuntimeError("boom")


if __name__ == "__main__":
    unittest.main()
