from typing import Any, Dict, List, Tuple

from .chat_template import assistant_message_to_scribe, tool_calls_to_scribe
from .parsers import ScribeBlock, ScribeBlockType, parse_scribe_blocks


_T_R_TYPES = (ScribeBlockType.THINK, ScribeBlockType.REFLECT)


def tool_calls_to_action(tool_calls: List[Any]) -> str:
    return tool_calls_to_scribe(tool_calls)


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return tool_call.get("id", "")
    return getattr(tool_call, "id", "")


def build_tool_messages(
    tool_calls: List[Any],
    info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results = info.get("tool_results", [])
    ids = [_tool_call_id(tc) for tc in tool_calls]
    assert len(results) == len(ids), (
        f"tool_results length {len(results)} != tool_calls length {len(ids)}"
    )
    return [
        {"role": "tool", "tool_call_id": tid, "content": r["output"]}
        for tid, r in zip(ids, results)
    ]


def message_to_scribe_blocks(
    message: Dict[str, Any],
) -> Tuple[str, List[ScribeBlock], bool]:
    action_text = assistant_message_to_scribe(message)
    blocks, is_valid = parse_scribe_blocks(action_text)
    return action_text, blocks, is_valid


def extract_tool_calls(blocks: List[ScribeBlock]) -> List[Dict[str, Any]]:
    return [
        b.parsed
        for b in blocks
        if b.type == ScribeBlockType.TOOL_CALL and b.parsed is not None
    ]


def strip_think_reflect(message: Dict[str, Any]) -> Dict[str, Any]:
    """Remove THINK/REFLECT blocks from an assistant message's content.

    Tool messages and non-assistant messages pass through unchanged.
    tool_calls are preserved (ids intact). This is a pure function so it can
    be reused by any component that needs to strip disposable reasoning blocks
    before appending an assistant message to the context.
    """
    if message.get("role") != "assistant":
        return dict(message)
    content = message.get("content") or ""
    if not content:
        return dict(message)
    from .parsers import render_scribe_blocks

    blocks, _ = parse_scribe_blocks(content)
    kept = [b for b in blocks if b.type not in _T_R_TYPES]
    new = dict(message)
    new["content"] = render_scribe_blocks(kept)
    return new


__all__ = [
    "build_tool_messages",
    "extract_tool_calls",
    "message_to_scribe_blocks",
    "strip_think_reflect",
    "tool_calls_to_action",
]
