"""Tests for observation_compressor.py — SummarizeCompressor & LLMCompressor."""

import pytest

from llm_runtime.observation_compressor import (
    LLMCompressor,
    SummarizeCompressor,
    _SUMMARY_MARKER,
    _format_for_summary,
)
from llm_runtime.token_counter import DeepSeekTokenCounter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def counter():
    return DeepSeekTokenCounter()


@pytest.fixture
def small_compressor(counter):
    """Compressor with a tiny budget so it always triggers."""
    return SummarizeCompressor(counter, max_context_tokens=10, keep_recent_rounds=1)


@pytest.fixture
def large_compressor(counter):
    """Compressor with large budget — never triggers for small inputs."""
    return SummarizeCompressor(counter, max_context_tokens=999_999, keep_recent_rounds=5)


@pytest.fixture
def long_observations():
    """Build a conversation that exceeds a tiny token budget."""
    obs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, please help me with a task."},
        {"role": "assistant", "content": "Sure, I'd be happy to help!"},
        {"role": "user", "content": "I need to write a Python script."},
        {"role": "assistant", "content": "Let me write that for you."},
        {"role": "user", "content": "Also I need to test it."},
        {"role": "assistant", "content": "Here are the tests."},
    ]
    return obs


# ---------------------------------------------------------------------------
# SummarizeCompressor — needs_compression
# ---------------------------------------------------------------------------


class TestNeedsCompression:
    def test_under_budget_returns_false(self, large_compressor, long_observations):
        assert large_compressor.needs_compression(long_observations) is False

    def test_over_budget_returns_true(self, small_compressor, long_observations):
        assert small_compressor.needs_compression(long_observations) is True

    def test_empty_list(self, small_compressor):
        assert small_compressor.needs_compression([]) is False

    def test_short_system_only(self, small_compressor):
        """A single short system message should be under the budget."""
        obs = [{"role": "system", "content": "hi"}]
        assert small_compressor.needs_compression(obs) is False


# ---------------------------------------------------------------------------
# SummarizeCompressor — compress
# ---------------------------------------------------------------------------


class TestCompress:
    @pytest.mark.asyncio
    async def test_no_compression_needed(self, large_compressor, long_observations):
        result = await large_compressor.compress(long_observations)
        assert result == long_observations  # unchanged

    @pytest.mark.asyncio
    async def test_system_preserved(self, small_compressor, long_observations):
        result = await small_compressor.compress(long_observations)
        system_msgs = [o for o in result if o.get("role") == "system"]
        assert "You are a helpful assistant." in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_recent_messages_preserved(self, small_compressor, long_observations):
        result = await small_compressor.compress(long_observations)
        contents = [str(o.get("content", "")) for o in result]
        assert any("Also I need to test it" in c for c in contents)
        assert any("Here are the tests" in c for c in contents)

    @pytest.mark.asyncio
    async def test_summary_marker_present(self, small_compressor, long_observations):
        result = await small_compressor.compress(long_observations)
        all_content = " ".join(str(o.get("content", "")) for o in result)
        assert _SUMMARY_MARKER in all_content

    @pytest.mark.asyncio
    async def test_output_is_shorter_after_compression(self, counter):
        """Compression should reduce message count (and token count for long conversations)."""
        # Build a long conversation that will clearly benefit from compression.
        obs = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        for i in range(20):
            obs.append({"role": "user", "content": f"This is user message number {i} with some additional padding."})
            obs.append({"role": "assistant", "content": f"And this is the assistant reply number {i} with more detailed response content."})
        comp = SummarizeCompressor(counter, max_context_tokens=100, keep_recent_rounds=2)
        original_count = comp._counter.count_observations(obs)
        result = await comp.compress(obs)
        compressed_count = comp._counter.count_observations(result)
        # Message count should be drastically reduced.
        assert len(result) < len(obs)
        # Token count should also be lower.
        assert compressed_count < original_count

    @pytest.mark.asyncio
    async def test_empty_conversation(self, small_compressor):
        result = await small_compressor.compress([])
        assert result == []

    @pytest.mark.asyncio
    async def test_only_system_messages(self, small_compressor):
        obs = [{"role": "system", "content": "sys1"}, {"role": "system", "content": "sys2"}]
        result = await small_compressor.compress(obs)
        assert result == obs  # no non-system messages to summarise


# ---------------------------------------------------------------------------
# Re-compression safety
# ---------------------------------------------------------------------------


class TestReCompression:
    @pytest.mark.asyncio
    async def test_recompression_does_not_nest(self, small_compressor, long_observations):
        """Compressing twice should not create nested summary blocks."""
        once = await small_compressor.compress(long_observations)
        twice = await small_compressor.compress(once)
        count = sum(
            1
            for o in twice
            if isinstance(o.get("content"), str) and _SUMMARY_MARKER in o["content"]
        )
        assert count == 1, f"Expected 1 summary block, found {count}"

    @pytest.mark.asyncio
    async def test_recompression_not_increase_messages(self, counter):
        """Re-compression should not increase the number of summary blocks."""
        obs = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        for i in range(20):
            obs.append({"role": "user", "content": f"User msg {i} with padding."})
            obs.append({"role": "assistant", "content": f"Assistant reply {i} with more detail."})
        comp = SummarizeCompressor(counter, max_context_tokens=100, keep_recent_rounds=2)
        once = await comp.compress(obs)
        twice = await comp.compress(once)
        # Should have exactly one summary block.
        count = sum(
            1
            for o in twice
            if isinstance(o.get("content"), str) and _SUMMARY_MARKER in o["content"]
        )
        assert count == 1, f"Expected 1 summary block, found {count}"
        # Token count should not increase.
        assert comp._counter.count_observations(twice) <= comp._counter.count_observations(once)

    @pytest.mark.asyncio
    async def test_recompression_preserves_system(
        self, small_compressor, long_observations
    ):
        once = await small_compressor.compress(long_observations)
        twice = await small_compressor.compress(once)
        system_msgs = [o for o in twice if o.get("role") == "system"]
        assert any(
            "You are a helpful assistant." in str(o.get("content", ""))
            for o in system_msgs
        )


# ---------------------------------------------------------------------------
# Token counter — structural overhead
# ---------------------------------------------------------------------------


class TestTokenCounterOverhead:
    def test_overhead_reflected_in_count(self, counter):
        obs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        content_only = counter.count("hello") + counter.count("hi")
        total = counter.count_observations(obs)
        assert total > content_only, "Structural overhead not reflected in count"


# ---------------------------------------------------------------------------
# LLMCompressor — basic contract (no real LLM call)
# ---------------------------------------------------------------------------


class TestLLMCompressorContract:
    @pytest.mark.asyncio
    async def test_needs_compression(self, counter, long_observations):
        async def dummy_llm_call(msgs):
            return "dummy summary"

        comp = LLMCompressor(counter, max_context_tokens=10, async_llm_call=dummy_llm_call)
        assert comp.needs_compression(long_observations) is True

    @pytest.mark.asyncio
    async def test_no_compression_when_under_budget(self, counter, long_observations):
        async def dummy_llm_call(msgs):
            return "dummy summary"

        comp = LLMCompressor(
            counter, max_context_tokens=999_999, async_llm_call=dummy_llm_call
        )
        result = await comp.compress(long_observations)
        assert result == long_observations  # unchanged

    @pytest.mark.asyncio
    async def test_compress_calls_llm(self, counter, long_observations):
        """LLMCompressor should call the LLM when context overflows."""
        called = False

        async def dummy_llm_call(msgs):
            nonlocal called
            called = True
            return "semantic summary result"

        # Use keep_recent_rounds=1 so only 2 recent messages are kept,
        # forcing the older 4 messages to be summarised via LLM.
        comp = LLMCompressor(
            counter,
            max_context_tokens=10,
            async_llm_call=dummy_llm_call,
            keep_recent_rounds=1,
        )
        result = await comp.compress(long_observations)
        assert called, "LLM was not called"
        contents = " ".join(str(o.get("content", "")) for o in result)
        assert "semantic summary result" in contents


# ---------------------------------------------------------------------------
# _format_for_summary helper
# ---------------------------------------------------------------------------


class TestFormatForSummary:
    def test_basic_formatting(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = _format_for_summary(msgs)
        assert "[user]: hello" in result
        assert "[assistant]: world" in result

    def test_empty_list(self):
        assert _format_for_summary([]) == ""

    def test_truncation(self):
        long = "x" * 3000
        msgs = [{"role": "user", "content": long}]
        result = _format_for_summary(msgs)
        assert len(result) < 2500  # truncated
        assert "truncated" in result

# ---------------------------------------------------------------------------
# Edge cases — missing coverage lines
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_compress_with_existing_summary_uses_previous_summary_note(
        self, counter
    ):
        """When compressing content that already has a summary block,
        and there are *many* new messages beyond the keep threshold,
        the new output should mention the previous summary was preserved."""
        # Build a long conversation.
        obs = [
            {"role": "system", "content": "sys"},
        ]
        for i in range(15):
            obs.append({"role": "user", "content": f"msg {i}"})
            obs.append({"role": "assistant", "content": f"reply {i}"})

        # Use keep_recent_rounds=2 (keep 4 messages) so the first compression
        # summarises the older 26 messages into one summary block.
        comp = SummarizeCompressor(counter, max_context_tokens=50, keep_recent_rounds=2)
        once = await comp.compress(obs)
        assert any(
            _SUMMARY_MARKER in str(o.get("content", "")) for o in once
        ), "First compression should add a summary"

        # Add more messages on top so the second compression has work to do.
        once.append({"role": "user", "content": "extra user msg"})
        once.append({"role": "assistant", "content": "extra assistant reply"})

        # Now compress again. After extracting the existing summary,
        # remaining = [sys, 4 recent, 2 extra] = 5 non-system messages.
        # keep_count = 4 (keep_recent_rounds=2 * 2), so 5 > 4 → still summarises.
        twice = await comp.compress(once)
        all_content = " ".join(str(o.get("content", "")) for o in twice)
        assert "[previous-summary]: preserved" in all_content, (
            "Re-compression should mention the previous summary was preserved"
        )

    @pytest.mark.asyncio
    async def test_llm_compressor_not_enough_non_system_messages(self, counter):
        """LLMCompressor should skip LLM call when there aren't enough
        non-system messages to summarise."""
        called = False

        async def dummy_llm_call(msgs):
            nonlocal called
            called = True
            return "summary"

        comp = LLMCompressor(
            counter,
            max_context_tokens=999,  # large budget → needs_compression returns False
            async_llm_call=dummy_llm_call,
            keep_recent_rounds=5,
        )
        obs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = await comp.compress(obs)
        assert not called, "LLM should not be called when under budget"
        assert result == obs  # unchanged

    @pytest.mark.asyncio
    async def test_llm_compressor_early_return_not_enough_to_summarise(self, counter):
        """LLMCompressor should return early when over budget but too few
        non-system messages to summarise (line 196)."""
        called = False

        async def dummy_llm_call(msgs):
            nonlocal called
            called = True
            return "summary"

        # Tiny budget so needs_compression is True, but only 1 non-system msg.
        comp = LLMCompressor(
            counter,
            max_context_tokens=1,
            async_llm_call=dummy_llm_call,
            keep_recent_rounds=5,  # keep_count=10, but only 1 non-system msg
        )
        obs = [
            {"role": "system", "content": "x" * 200},
            {"role": "user", "content": "hi"},
        ]
        # needs_compression should be True (long system text).
        assert comp.needs_compression(obs) is True
        result = await comp.compress(obs)
        assert not called, "LLM should not be called with too few messages"
        # Should return original observations unchanged.
        assert result is obs

    def test_has_existing_summary_returns_true_when_present(self, counter):
        """_has_existing_summary should detect the marker."""
        comp = SummarizeCompressor(counter, max_context_tokens=10)
        obs = [
            {"role": "system", "content": f"Previous context ({_SUMMARY_MARKER}):\nsummary"},
            {"role": "user", "content": "hi"},
        ]
        assert comp._has_existing_summary(obs) is True

    def test_has_existing_summary_returns_false_when_absent(self, counter):
        """_has_existing_summary should return False when no marker."""
        comp = SummarizeCompressor(counter, max_context_tokens=10)
        obs = [
            {"role": "system", "content": "normal system message"},
            {"role": "user", "content": "hi"},
        ]
        assert comp._has_existing_summary(obs) is False

    def test_has_existing_summary_empty_list(self, counter):
        comp = SummarizeCompressor(counter, max_context_tokens=10)
        assert comp._has_existing_summary([]) is False

    def test_extract_existing_summary_found(self, counter):
        comp = SummarizeCompressor(counter, max_context_tokens=10)
        obs = [
            {"role": "system", "content": f"Previous context ({_SUMMARY_MARKER}):\nsummary"},
            {"role": "user", "content": "hi"},
        ]
        summary, remaining = comp._extract_existing_summary(obs)
        assert summary is not None
        assert _SUMMARY_MARKER in summary
        assert len(remaining) == 1
        assert remaining[0]["role"] == "user"

    def test_extract_existing_summary_not_found(self, counter):
        comp = SummarizeCompressor(counter, max_context_tokens=10)
        obs = [
            {"role": "system", "content": "normal"},
            {"role": "user", "content": "hi"},
        ]
        summary, remaining = comp._extract_existing_summary(obs)
        assert summary is None
        assert len(remaining) == 2

    def test_extract_existing_summary_mixed(self, counter):
        """Multiple system messages, one of which is a summary."""
        comp = SummarizeCompressor(counter, max_context_tokens=10)
        obs = [
            {"role": "system", "content": "normal sys"},
            {"role": "system", "content": f"Previous context ({_SUMMARY_MARKER}):\ndata"},
            {"role": "user", "content": "hi"},
        ]
        summary, remaining = comp._extract_existing_summary(obs)
        assert summary is not None
        assert len(remaining) == 2  # normal sys + user

    def test_llm_compressor_custom_prompt(self, counter):
        """LLMCompressor should accept a custom summarise prompt."""
        custom_prompt = "Custom: summarise this"
        comp = LLMCompressor(
            counter,
            max_context_tokens=10,
            async_llm_call=lambda msgs: None,  # type: ignore
            summarise_prompt=custom_prompt,
        )
        assert comp._summarise_prompt == custom_prompt

# ---------------------------------------------------------------------------
# Edge cases for _summarise_message (indirectly via compress)
# ---------------------------------------------------------------------------


def _build_conv_with_extra_pairs(
    system: str, pairs: list[tuple[str, str]], extra_pairs: int = 3
) -> list[dict]:
    """Build a conversation with system + (base pairs) + (extra padding pairs)."""
    obs = [{"role": "system", "content": system}]
    for u, a in pairs:
        obs.append({"role": "user", "content": u})
        obs.append({"role": "assistant", "content": a})
    for i in range(extra_pairs):
        obs.append({"role": "user", "content": f"padding_user_{i}"})
        obs.append({"role": "assistant", "content": f"padding_asst_{i}"})
    return obs


class TestSummariseMessage:
    """Test _summarise_message edge cases through compress()."""

    @pytest.mark.asyncio
    async def test_message_with_blank_lines(self, counter):
        """Blank lines in the middle of content should be skipped."""
        comp = SummarizeCompressor(counter, max_context_tokens=10, keep_recent_rounds=1)
        obs = _build_conv_with_extra_pairs(
            "sys",
            [("line1\n\n\nline2", "reply with\n\n\n\nblank lines")],
            extra_pairs=3,
        )
        result = await comp.compress(obs)
        summary_content = " ".join(
            str(o.get("content", "")) for o in result
            if _SUMMARY_MARKER in str(o.get("content", ""))
        )
        assert "line1" in summary_content or "line2" in summary_content

    @pytest.mark.asyncio
    async def test_message_exceeds_max_summary_chars(self, counter):
        """Very long messages should be truncated by _summarise_message."""
        comp = SummarizeCompressor(counter, max_context_tokens=10, keep_recent_rounds=1)
        long_line = "word " * 500  # far exceeds _MAX_SUMMARY_LINE_CHARS (600)
        obs = _build_conv_with_extra_pairs(
            "sys",
            [(long_line, "short reply")],
            extra_pairs=3,
        )
        result = await comp.compress(obs)
        assert any(_SUMMARY_MARKER in str(o.get("content", "")) for o in result)

    @pytest.mark.asyncio
    async def test_message_many_lines_exceeds_max_lines(self, counter):
        """Message with many lines beyond _MAX_SUMMARY_LINES should be truncated."""
        comp = SummarizeCompressor(counter, max_context_tokens=10, keep_recent_rounds=1)
        many_lines = "\n".join(f"line_{i}" for i in range(20))
        obs = _build_conv_with_extra_pairs(
            "sys",
            [(many_lines, "short reply")],
            extra_pairs=3,
        )
        result = await comp.compress(obs)
        assert any(_SUMMARY_MARKER in str(o.get("content", "")) for o in result)


# ---------------------------------------------------------------------------
# LLMCompressor re-compression safety
# ---------------------------------------------------------------------------


def _build_long_conv(system: str, num_pairs: int = 10) -> list[dict]:
    """Build a conversation with system + num_pairs user/assistant pairs."""
    obs = [{"role": "system", "content": system}]
    for i in range(num_pairs):
        obs.append({"role": "user", "content": f"user msg {i} with some padding"})
        obs.append({"role": "assistant", "content": f"asst reply {i} with more padding"})
    return obs


class TestLLMCompressorReCompression:
    @pytest.mark.asyncio
    async def test_recompression_does_not_nest(self, counter):
        """LLMCompressor should avoid nested summary blocks on re-compression."""
        call_count = 0

        async def counting_llm(msgs):
            nonlocal call_count
            call_count += 1
            return f"llm summary call #{call_count}"

        # Tight budget so BOTH compressions trigger.
        comp = LLMCompressor(
            counter,
            max_context_tokens=30,
            async_llm_call=counting_llm,
            keep_recent_rounds=1,
        )

        obs = _build_long_conv("sys", num_pairs=10)

        # First compression.
        once = await comp.compress(obs)
        count_once = sum(
            1 for o in once
            if isinstance(o.get("content"), str) and _SUMMARY_MARKER in o["content"]
        )
        assert count_once == 1, (
            f"Expected 1 summary block after first compress, got {count_once}"
        )

        # Add enough extra pairs to trigger re-compression.
        for i in range(3):
            once.append({"role": "user", "content": f"extra user {i}"})
            once.append({"role": "assistant", "content": f"extra asst {i}"})

        twice = await comp.compress(once)
        count_twice = sum(
            1 for o in twice
            if isinstance(o.get("content"), str) and _SUMMARY_MARKER in o["content"]
        )
        assert count_twice == 1, (
            f"Expected 1 summary block after re-compression, got {count_twice}"
        )
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_recompression_merges_previous_summary(self, counter):
        """LLMCompressor should pass the previous summary to the LLM on re-compression."""
        seen_previous = False

        async def checking_llm(msgs):
            nonlocal seen_previous
            combined = " ".join(str(m.get("content", "")) for m in msgs)
            if "previous summary" in combined.lower():
                seen_previous = True
            return "merged summary"

        comp = LLMCompressor(
            counter,
            max_context_tokens=30,
            async_llm_call=checking_llm,
            keep_recent_rounds=1,
        )

        obs = _build_long_conv("sys", num_pairs=10)
        once = await comp.compress(obs)

        # Add enough extra pairs to trigger re-compression.
        for i in range(3):
            once.append({"role": "user", "content": f"extra user {i}"})
            once.append({"role": "assistant", "content": f"extra asst {i}"})

        await comp.compress(once)
        assert seen_previous, (
            "Re-compression should pass previous summary to LLM"
        )


# ---------------------------------------------------------------------------
# _summarise_message: empty content edge case
# ---------------------------------------------------------------------------


class TestSummariseMessageEmpty:
    @pytest.mark.asyncio
    async def test_empty_content_not_crashing(self, counter):
        """Messages with empty content should not crash _summarise_message."""
        comp = SummarizeCompressor(counter, max_context_tokens=10, keep_recent_rounds=1)
        obs = _build_conv_with_extra_pairs(
            "sys",
            [("", "")],
            extra_pairs=3,
        )
        result = await comp.compress(obs)
        assert any(_SUMMARY_MARKER in str(o.get("content", "")) for o in result)


# ---------------------------------------------------------------------------
# Summarise message: char limit break (line 142)
# ---------------------------------------------------------------------------


class TestSummariseCharLimit:
    @pytest.mark.asyncio
    async def test_summary_char_limit_triggers_break(self, counter):
        """Many long lines should trigger the char-count break in _summarise_message."""
        comp = SummarizeCompressor(counter, max_context_tokens=10, keep_recent_rounds=1)
        # Build a message with 5 lines each ~200 chars → total ~1000 chars > 600 limit.
        long_lines = "\n".join(("x" * 200 + f"_{i}") for i in range(5))
        obs = _build_conv_with_extra_pairs(
            "sys",
            [(long_lines, "short")],
            extra_pairs=3,
        )
        result = await comp.compress(obs)
        assert any(_SUMMARY_MARKER in str(o.get("content", "")) for o in result)
