import json
from typing import Any, Dict, List, Optional, Tuple

from .types import ScribeBlock, ScribeBlockType

_SCRIBE_TAG_TYPES = {
    "think": ScribeBlockType.THINK,
    "tool_call": ScribeBlockType.TOOL_CALL,
    "tool_response": ScribeBlockType.TOOL_RESPONSE,
    "reflect": ScribeBlockType.REFLECT,
    "turn_summary": ScribeBlockType.TURN_SUMMARY,
}

def render_scribe_blocks(blocks: List[ScribeBlock]) -> str:
    return "".join(block.content for block in blocks)

def _contains_known_tag(text: str) -> bool:
    for tag in _SCRIBE_TAG_TYPES:
        if f"<{tag}>" in text or f"</{tag}>" in text:
            return True
    return False

def _find_next_valid_pair(
    response: str, i: int
) -> Tuple[Optional[Tuple[str, int, int]], bool]:
    n = len(response)
    best_start = n
    best_tag: Optional[str] = None
    best_end = n
    saw_malformed = False

    for tag, block_type in _SCRIBE_TAG_TYPES.items():
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = response.find(start_tag, i)
        if start == -1:
            continue

        if start >= best_start:
            continue

        content_start = start + len(start_tag)
        end = response.find(end_tag, content_start)
        if end == -1:
            saw_malformed = True
            continue

        content = response[content_start:end]

        if block_type in (
            ScribeBlockType.THINK,
            ScribeBlockType.REFLECT,
            ScribeBlockType.TURN_SUMMARY,
            ScribeBlockType.TOOL_RESPONSE,
        ) and _contains_known_tag(content):
            saw_malformed = True
            continue

        if block_type == ScribeBlockType.TOOL_CALL:
            try:
                json.loads(content.strip())
            except Exception:
                saw_malformed = True
                continue

        best_start = start
        best_tag = tag
        best_end = end + len(end_tag)

    if best_tag is None:
        return None, saw_malformed
    return (best_tag, best_start, best_end), saw_malformed

def parse_scribe_blocks(response: str) -> Tuple[List[ScribeBlock], bool]:
    blocks: List[ScribeBlock] = []
    is_valid = True
    i = 0
    n = len(response)

    while i < n:
        pair, saw_malformed = _find_next_valid_pair(response, i)
        if saw_malformed:
            is_valid = False
        if pair is None:
            remaining = response[i:]
            if remaining:
                if _contains_known_tag(remaining):
                    is_valid = False
                blocks.append(ScribeBlock(ScribeBlockType.OUTPUT, remaining))
            break

        tag, start, end = pair
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        if start > i:
            text = response[i:start]
            if text:
                if _contains_known_tag(text):
                    is_valid = False
                blocks.append(ScribeBlock(ScribeBlockType.OUTPUT, text))

        content = response[start:end]
        block_type = _SCRIBE_TAG_TYPES[tag]
        parsed = None
        if block_type == ScribeBlockType.TOOL_CALL:
            inner = content[len(start_tag) : -len(end_tag)]
            parsed = json.loads(inner.strip())
        blocks.append(ScribeBlock(block_type, content, parsed))
        i = end

    return blocks, is_valid

__all__ = [
    "ScribeBlock",
    "ScribeBlockType",
    "parse_scribe_blocks",
    "render_scribe_blocks",
]
