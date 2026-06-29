import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

_chat_template: Optional[Template] = None

def _get_chat_template() -> Template:
    global _chat_template
    if _chat_template is None:
        tpl_path = Path(__file__).with_name("chat_template.jinja2")
        _chat_template = Template(tpl_path.read_text())
    return _chat_template

def _normalize_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        return json.loads(arguments)
    return arguments

def render_messages(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
) -> str:
    return _get_chat_template().render(
        messages=messages,
        tools=tools or [],
        add_generation_prompt=add_generation_prompt,
    )

def tool_calls_to_scribe(tool_calls: List[Any]) -> str:
    tool_call_dicts = []
    for tc in tool_calls:
        func = tc.function
        tool_call_dicts.append(
            {
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": func.name,
                    "arguments": _normalize_arguments(func.arguments),
                },
            }
        )

    messages = [
        {"role": "assistant", "content": None, "tool_calls": tool_call_dicts}
    ]
    rendered = render_messages(messages, tools=[], add_generation_prompt=False)
    blocks = re.findall(r"<tool_call>.*?</tool_call>", rendered, re.DOTALL)
    return "\n".join(blocks)

def assistant_message_to_scribe(message: Dict[str, Any]) -> str:
    normalized = dict(message)
    if normalized.get("tool_calls"):
        normalized["tool_calls"] = [
            {
                **tc,
                "function": {
                    **tc.get("function", {}),
                    "arguments": _normalize_arguments(
                        tc.get("function", {}).get("arguments")
                    ),
                },
            }
            for tc in normalized["tool_calls"]
        ]
    rendered = render_messages([normalized], tools=[], add_generation_prompt=False)
    start = rendered.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    end = rendered.find("<|im_end|>", start)
    return rendered[start:end]


def _sanitize_tool_output(output: str) -> str:
    return output.replace("<tool_response>", "").replace("</tool_response>", "")


def tool_results_to_scribe(tool_results: List[Dict[str, Any]]) -> str:
    messages = []
    for r in tool_results:
        content = _sanitize_tool_output(r["output"])
        if r.get("status") != "success":
            content = f"[{r['status']}]\n{content}"
        messages.append({"role": "tool", "content": content})

    rendered = render_messages(messages, tools=[], add_generation_prompt=False)
    blocks = re.findall(r"<tool_response>.*?</tool_response>", rendered, re.DOTALL)
    return "\n".join(blocks)

__all__ = [
    "assistant_message_to_scribe",
    "render_messages",
    "tool_calls_to_scribe",
    "tool_results_to_scribe",
]
