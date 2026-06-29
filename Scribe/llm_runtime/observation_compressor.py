"""Observation compression utilities for the llm_runtime agent.

This module is intentionally minimal; concrete compressors can be added
without changing the public interface used by the rest of the package.
"""

from typing import Any


class ObservationCompressor:
    """Base class for compressing observation message lists."""

    async def compress(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the observations unchanged by default."""
        return observations


class SummarizeCompressor(ObservationCompressor):
    """Placeholder for a summarization-based compressor."""


class LLMCompressor(ObservationCompressor):
    """Placeholder for an LLM-based compressor."""
