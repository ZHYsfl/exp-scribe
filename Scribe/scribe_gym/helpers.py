from typing import Any, Dict, List, Tuple

from .chat_template import assistant_message_to_scribe, tool_calls_to_scribe
from .parsers import ScribeBlock, ScribeBlockType, parse_scribe_blocks


def tool_calls_to_action(tool_calls: List[Any]) -> str:
    return tool_calls_to_scribe(tool_calls)


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return tool_call.get("id", "")
    return getattr(tool_call, "id", "")


def observation_to_tool_messages(
    observation: str,
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


__all__ = [
    "extract_tool_calls",
    "message_to_scribe_blocks",
    "observation_to_tool_messages",
    "tool_calls_to_action",
]
