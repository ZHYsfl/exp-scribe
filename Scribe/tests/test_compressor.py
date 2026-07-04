"""Unit tests for compressor — LLMSummarizerCompressor wrapping a fake client,
and the Compressor Protocol (duck typing)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Scribe.scribe_gym import Compressor, LLMSummarizerCompressor


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeClient:
    """Fake AsyncOpenAI-like client. Returns a queue of canned responses."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    class _Completions:
        def __init__(self, outer):
            self._outer = outer

        async def create(self, **kwargs):
            idx = min(self._outer.calls, len(self._outer._responses) - 1)
            self._outer.calls += 1
            return FakeResponse(self._outer._responses[idx])

    @property
    def chat(self):
        return type("C", (), {"completions": self._Completions(self)})()


def test_summarize_returns_bare_summary():
    client = FakeClient(['{"summary": "the folded summary text"}'])
    comp = LLMSummarizerCompressor(client, model="m")
    result = asyncio.run(comp.summarize("sys", "kick", ["fb0"], ["s0"]))
    assert "<turn_summary>" not in result
    assert "</turn_summary>" not in result
    assert result == "the folded summary text"


def test_summarize_retries_on_invalid_json():
    client = FakeClient(["not json", '{"summary": "valid after retry"}'])
    comp = LLMSummarizerCompressor(client, model="m", max_retries=1)
    result = asyncio.run(comp.summarize("sys", "kick", [], ["s0"]))
    assert result == "valid after retry"
    assert client.calls == 2


def test_compressor_protocol_duck_typing():
    class MyComp:
        async def summarize(self, system, kickoff, feedbacks, s_blocks):
            return "bare summary"

    assert isinstance(MyComp(), Compressor)  # runtime_checkable Protocol


if __name__ == "__main__":
    asyncio.run(test_summarize_returns_bare_summary())
    print("ok")
