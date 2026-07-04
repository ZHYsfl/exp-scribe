"""Stage-2 compression for SCRIBE: sota-LLM summarization.

When Stage-1 (collect S blocks) still leaves the input over the token budget,
Stage 2 folds the kickoff + all feedbacks + all S blocks into a single
summary via a sota LLM. The system prompt stays verbatim.

The Compressor is a Protocol — inject any LLM client (DeepSeek/Claude/GPT/...).
SCRIBE depends on the Protocol, never a concrete class (open-closed + DI).
The retry/backoff for an over-length summary is the compressor's responsibility,
decoupled from HistoryManager.
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

from pydantic import BaseModel

from ..llm_runtime import StructuredGenerator


@runtime_checkable
class Compressor(Protocol):
    async def summarize(
        self,
        system: str,
        kickoff: str,
        feedbacks: List[str],
        s_blocks: List[str],
    ) -> str:
        """Return a single bare summary string (no <turn_summary> tags).

        The returned string is the inner content of what will become a
        TURN_SUMMARY block; HistoryManager owns whether/when to wrap it in
        block-level tags. The string must fit under the token budget; if the
        first attempt is too long, the implementation retries (its own backoff).
        """
        ...


class SummaryOutput(BaseModel):
    """Structured JSON schema for the sota compressor's summary."""

    summary: str


_SUMMARY_INSTRUCTION = (
    "You are compressing the history of a multi-turn agent task into one "
    "summary. Below are: the first user kickoff, the feedback "
    "messages after each turn, and each turn's own turn_summary. Produce ONE "
    "concise summary that captures the essential state: what has "
    "been done, what is known, and what remains. "
    'Respond with raw JSON matching this schema: {"summary": "string"}. '
    "Do not include anything else."
)


class LLMSummarizerCompressor:
    """Compressor backed by an OpenAI-compatible AsyncOpenAI client.

    Uses StructuredGenerator to guarantee valid JSON output matching the
    SummaryOutput schema (with Pydantic validation + retry).
    """

    def __init__(
        self,
        client: Any,
        model: str,
        max_retries: int = 3,
        max_output_tokens: int = 8192,
        temperature: float = 0.1,
    ):
        self._generator = StructuredGenerator(
            client=client,
            model=model,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

    def _build_user_content(
        self, kickoff: str, feedbacks: List[str], s_blocks: List[str]
    ) -> str:
        parts = [f"# Kickoff\n{kickoff}"]
        if feedbacks:
            parts.append("# Feedbacks\n" + "\n---\n".join(feedbacks))
        if s_blocks:
            parts.append("# Turn summaries\n" + "\n---\n".join(s_blocks))
        return "\n\n".join(parts)

    async def summarize(
        self,
        system: str,
        kickoff: str,
        feedbacks: List[str],
        s_blocks: List[str],
    ) -> str:
        messages = [
            {"role": "system", "content": _SUMMARY_INSTRUCTION},
            {"role": "user", "content": self._build_user_content(kickoff, feedbacks, s_blocks)},
        ]
        parsed = await self._generator.generate(messages, SummaryOutput)
        return parsed.summary.strip()


__all__ = ["Compressor", "LLMSummarizerCompressor", "SummaryOutput"]
