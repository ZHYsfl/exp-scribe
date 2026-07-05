"""Token counting abstraction with lazy-loaded DeepSeek tokenizer."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class TokenCounter(ABC):
    """Abstract token counter. Subclass to plug in any tokenizer."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Token count for a single text string."""

    @abstractmethod
    def count_observations(self, observations: list[dict]) -> int:
        """Sum of tokens across all observation messages."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Token ids for a single text string (for set/n-gram metrics)."""


class DeepSeekTokenCounter(TokenCounter):
    """Token counter backed by the DeepSeek V3 HuggingFace tokenizer.

    The tokenizer is loaded lazily (on first use) to avoid import overhead
    when compression never triggers.
    """

    def __init__(self, model_path: str | None = None) -> None:
        # Resolve default path relative to this file's project root
        if model_path is None:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(
                _root, "deepseek_tokenizer", "deepseek_v3_tokenizer"
            )
        self._model_path = os.path.abspath(model_path)
        self._tokenizer: Any = None  # Lazy-loaded transformers tokenizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lazy_load(self) -> None:
        if self._tokenizer is not None:
            return
        import transformers  # Heavy import – delay until needed

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self, text: str) -> int:
        self._lazy_load()
        return len(self._tokenizer.encode(text))  # type: ignore[union-attr]

    def count_observations(self, observations: list[dict]) -> int:
        total = 0
        for obs in observations:
            content = obs.get("content")
            if content:
                total += self.count(str(content))
        return total

    def encode(self, text: str) -> list[int]:
        self._lazy_load()
        return list(self._tokenizer.encode(text))  # type: ignore[union-attr]
