"""Tests for token_counter.py — DeepSeekTokenCounter & structural overhead."""

import pytest

from llm_runtime.token_counter import DeepSeekTokenCounter, TokenCounter


# ---------------------------------------------------------------------------
# Abstract base contract
# ---------------------------------------------------------------------------


class TestTokenCounterABC:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            TokenCounter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# DeepSeekTokenCounter
# ---------------------------------------------------------------------------


class TestDeepSeekTokenCounter:
    @pytest.fixture
    def counter(self):
        return DeepSeekTokenCounter()

    def test_count_returns_positive_int(self, counter):
        tokens = counter.count("Hello, world!")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_empty_string(self, counter):
        tokens = counter.count("")
        assert isinstance(tokens, int)
        assert tokens >= 0

    def test_count_lazy_loads_tokenizer(self, counter):
        """Tokenizer should be None before first call."""
        assert counter._tokenizer is None
        counter.count("test")
        assert counter._tokenizer is not None

    def test_count_observations_empty(self, counter):
        assert counter.count_observations([]) == 0

    def test_count_observations_single(self, counter):
        obs = [{"role": "user", "content": "hello"}]
        count = counter.count_observations(obs)
        assert count > 0

    def test_count_observations_structural_overhead(self, counter):
        """Structural overhead should add tokens beyond content."""
        single = counter.count_observations([{"role": "user", "content": "hi"}])
        double = counter.count_observations([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi"},
        ])
        # With structural overhead of ~4 tokens per message,
        # double should have at least 2x the content tokens + some overhead.
        # Content: 2 * count("hi"), overhead: 2 * 4 = 8
        # So double should be > 2 * content-only (without overhead)
        hi_tokens = counter.count("hi")
        content_only_2 = hi_tokens * 2
        assert double > content_only_2, (
            f"Structural overhead not reflected: double={double}, content_only_2={content_only_2}"
        )

    def test_count_observations_missing_content(self, counter):
        obs = [{"role": "system"}]  # no "content" key
        assert counter.count_observations(obs) >= 0  # structural overhead only

    def test_count_observations_none_content(self, counter):
        obs = [{"role": "system", "content": None}]
        assert counter.count_observations(obs) >= 0

    def test_default_model_path(self, counter):
        """Default path should point to deepseek_tokenizer directory."""
        assert "deepseek_tokenizer" in counter._model_path
        assert counter._model_path.endswith("deepseek_v3_tokenizer")

    def test_custom_model_path(self):
        counter = DeepSeekTokenCounter(model_path="/tmp/fake_path")
        assert counter._model_path == "/tmp/fake_path"


# ---------------------------------------------------------------------------
# LLMConfig model_dump
# ---------------------------------------------------------------------------


class TestLLMConfigModelDump:
    def test_model_dump_excludes_none_extra_body(self):
        from llm_runtime._models import LLMConfig

        cfg = LLMConfig(api_key="key", model="m", base_url="u")
        dumped = cfg.model_dump()
        assert "extra_body" not in dumped

    def test_model_dump_includes_extra_body_when_set(self):
        from llm_runtime._models import LLMConfig

        cfg = LLMConfig(api_key="key", model="m", base_url="u", extra_body={"temperature": 0.5})
        dumped = cfg.model_dump()
        assert dumped.get("extra_body") == {"temperature": 0.5}
