"""Shared Pydantic models for Agent, Tools, and configuration.

Extracted from tool_calling.py to keep files under 250 lines while
preserving backward-compatible imports via tool_calling re-exports.
"""

from typing import Any, Callable

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    api_key: str
    model: str
    base_url: str
    extra_body: dict[str, Any] | None = Field(default=None)

    def model_dump(self, **kwargs):
        """Override to exclude None values by default."""
        data = super().model_dump(**kwargs)
        if data.get("extra_body") is None:
            data.pop("extra_body", None)
        return data


class Tool(BaseModel):
    name: str
    description: str
    function: Callable
    parameters: dict
