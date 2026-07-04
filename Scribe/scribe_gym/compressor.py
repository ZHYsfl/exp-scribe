"""Stage-2 compression for SCRIBE: sota-LLM summarization.

When Stage-1 (collect S blocks) still leaves the input over the token budget,
Stage 2 folds the kickoff + all feedbacks + all S blocks into a single
<turn_summary> block via a sota LLM. The system prompt stays verbatim.

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
        """Return a single <turn_summary>...</turn_summary> block string.

        The returned string should include the `<turn_summary>` tags so the
        block boundary remains explicit in the message content. If the first
        attempt is too long, the implementation retries (its own backoff).
        """
        ...


class SummaryOutput(BaseModel):
    """Structured JSON schema for the sota compressor's summary."""

    summary: str


_SUMMARY_INSTRUCTION = (
    "You are compressing the history of a multi-turn agent task into one "
    "turn_summary block. Below are: the first user kickoff, the feedback "
    "messages after each turn, and each turn's own turn_summary. Produce ONE "
    "concise <turn_summary> block that captures the essential state: what has "
    "been done, what is known, and what remains. Wrap the result in "
    "<turn_summary> and </turn_summary> tags. Do not include anything else. "
    'Respond with raw JSON matching this schema: {"summary": "string"}.'
)


class LLMSummarizerCompressor:
    """Compressor backed by an OpenAI-compatible AsyncOpenAI client.

    Uses StructuredGenerator to guarantee valid JSON output matching the
    SummaryOutput schema (with Pydantic validation + retry). The returned
    summary always includes the `<turn_summary>` tags.
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

    @staticmethod
    def _validate_summary_tags(parsed: SummaryOutput) -> str | None:
        """Extra validator: the summary must be exactly one <turn_summary> block.

        Returns None if valid, otherwise a correction message for the LLM.
        """
        text = parsed.summary.strip()
        if text.startswith("<turn_summary>") and text.endswith("</turn_summary>"):
            return None
        return (
            "The summary must start exactly with '<turn_summary>' and end "
            "exactly with '</turn_summary>'. No extra text before or after."
        )

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
        parsed = await self._generator.generate(
            messages,
            SummaryOutput,
            extra_validate=self._validate_summary_tags,
        )
        return parsed.summary.strip()


__all__ = ["Compressor", "LLMSummarizerCompressor", "SummaryOutput"]
