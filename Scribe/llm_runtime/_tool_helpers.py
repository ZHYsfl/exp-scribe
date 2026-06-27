"""Shared helpers for Linux tool implementations.

Keeps basic_linux_tools.py under 250 lines and provides a single
place for path resolution, schema generation, and output truncation.
"""

from pathlib import Path
from typing import Any

MAX_TOOL_OUTPUT_CHARS = 25000
GREP_TIMEOUT_SECONDS = 30


def schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def resolve(root: Path, path: str | None) -> Path:
    target = root / (path or ".")
    target = target.resolve()
    if root not in (target, *target.parents):
        raise ValueError(f"path is outside workspace: {path}")
    return target


def truncate(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    notice = f"\nOutput was truncated to {max_chars} characters."
    return text[: max_chars - len(notice)] + notice
