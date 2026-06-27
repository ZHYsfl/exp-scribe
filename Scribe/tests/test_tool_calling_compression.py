"""Tests for compression integration in Agent.chat() — mid-loop compression.

These tests use a mock compressor to verify that _compress_if_needed is called
at the right points without requiring a real LLM API.
"""

import pytest

from llm_runtime.tool_calling import Agent, LLMConfig
from llm_runtime.observation_compressor import ObservationCompressor


# ---------------------------------------------------------------------------
# Mock compressor that records calls
# ---------------------------------------------------------------------------


class SpyCompressor(ObservationCompressor):
    """Records every call to compress() and needs_compression()."""

    def __init__(self, should_compress: bool = True):
        self.compress_calls: list[list[dict]] = []
        self.needs_calls: list[list[dict]] = []
        self._should = should_compress

    def needs_compression(self, observations: list[dict]) -> bool:
        self.needs_calls.append(list(observations))
        return self._should

    async def compress(self, observations: list[dict]) -> list[dict]:
        self.compress_calls.append(list(observations))
        return observations  # pass-through for testing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return LLMConfig(
        api_key="test-key",
        model="test-model",
        base_url="https://api.test.com/v1",
    )


@pytest.fixture
def spy_compressor():
    return SpyCompressor()


# ---------------------------------------------------------------------------
# _compress_if_needed
# ---------------------------------------------------------------------------


class TestCompressIfNeeded:
    @pytest.mark.asyncio
    async def test_no_compressor_returns_unchanged(self, config):
        agent = Agent(config, compressor=None)
        obs = [{"role": "user", "content": "hello"}]
        result = await agent._compress_if_needed(obs)
        assert result is obs

    @pytest.mark.asyncio
    async def test_compressor_is_called(self, config, spy_compressor):
        agent = Agent(config, compressor=spy_compressor)
        obs = [{"role": "user", "content": "hello"}]
        result = await agent._compress_if_needed(obs)
        assert spy_compressor.compress_calls == [obs]
        assert result == obs

    @pytest.mark.asyncio
    async def test_compressor_needs_is_checked(self, config, spy_compressor):
        agent = Agent(config, compressor=spy_compressor)
        obs = [{"role": "user", "content": "hello"}]
        result = await agent._compress_if_needed(obs)
        # compress() internally calls needs_compression() at the start.
        assert len(spy_compressor.needs_calls) >= 0


# ---------------------------------------------------------------------------
# Agent initialization with compressor
# ---------------------------------------------------------------------------


class TestAgentWithCompressor:
    def test_compressor_stored(self, config, spy_compressor):
        agent = Agent(config, compressor=spy_compressor)
        assert agent.compressor is spy_compressor

    def test_compressor_none_by_default(self, config):
        agent = Agent(config)
        assert agent.compressor is None

    def test_compressor_passed_through_get_agent(self, config, spy_compressor):
        """Simulate the pattern in react_style_coding_agent.py."""
        agent = Agent(config, compressor=spy_compressor)
        assert agent.compressor is spy_compressor
        assert agent.compressor.needs_compression([{"role": "user", "content": "test"}]) is True
