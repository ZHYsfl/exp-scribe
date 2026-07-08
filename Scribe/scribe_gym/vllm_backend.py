"""vLLM OpenAI-server rollout backend for SCRIBE.

Instead of embedding ``vllm.AsyncLLMEngine`` in-process, this backend connects
to a standalone vLLM OpenAI-compatible server via HTTP. This is the recommended
setup for online RL:

1. Start the server (in a separate shell / process):

    vllm serve /home/zane/exp-scribe/qwen2.5-0.5b-instruct \
        --enable-lora \
        --lora-modules scribe_adapter=outputs/scribe_sft/final_lora \
        --gpu-memory-utilization 0.45 \
        --max-model-len 4096 \
        --port 8000

2. Pass ``VLLMBackend(base_url="http://localhost:8000/v1")`` to
   ``GymBackedAgent`` as ``llm_backend``.

3. After each GRPO update, call ``backend.update_weights(adapter_path)`` to load
   the new LoRA adapter into the running server.

The backend inherits ``AsyncOpenAI`` and is used exactly like the OpenAI client
inside ``GymBackedAgent.chat``:

    response = await backend.chat.completions.create(
        model=backend.lora_name,
        messages=[...],
        tools=[...],
        tool_choice="auto",
    )
    message = response.choices[0].message.model_dump()

Plain-text tool mode
--------------------
By default vLLM's Hermes tool parser is not concurrency-safe and raises
``RuntimeError: Already borrowed`` under concurrent requests. To avoid this,
set ``plain_text_tools=True``. In this mode the backend does **not** send the
``tools=`` / ``tool_choice=`` parameters to vLLM. Instead it injects the tool
schemas as plain text into the system prompt and parses the model's
``<tool_call>...</tool_call>`` output blocks locally using the same extraction
logic as vLLM's ``Hermes2ProToolParser``. The returned message is still a
standard OpenAI-style dict, so callers need no changes.
"""

from __future__ import annotations

import json
import re as std_re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from openai import AsyncOpenAI


_TOOL_CALL_START_TOKEN = "<tool_call>"
_TOOL_CALL_END_TOKEN = "</tool_call>"
# vLLM uses the third-party ``regex`` module for partial streaming matches. We
# use the stdlib ``re`` here because we operate on complete responses, so the
# partial-match capture group ``<tool_call>(.*)`` is unnecessary.
_TOOL_CALL_REGEX = std_re.compile(
    r"<tool_call>(.*?)</tool_call>", std_re.DOTALL
)


def _is_complete_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _make_tool_call_id() -> str:
    return f"chatcmpl-tool-{uuid.uuid4().hex}"


def _tools_to_prompt_schema(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Render OpenAI-style tool definitions as plain text for the prompt.

    This text is byte-for-byte identical to the tool schema appended by the
    Qwen2.5-Instruct chat template (see ``tokenizer_config.json``
    ``chat_template``) when ``tools`` is provided. The caller is responsible
    for prepending the leading ``\\n\\n`` when appending it to an existing
    system prompt.
    """
    if not tools:
        return ""
    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    for tool in tools:
        lines.append(json.dumps(tool, ensure_ascii=False))
    lines.extend([
        "</tools>",
        "",
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:",
        "<tool_call>",
        '{"name": <function-name>, "arguments": <args-json-object>}',
        "</tool_call>",
    ])
    return "\n".join(lines)


class HermesToolExtractor:
    """Client-side re-implementation of vLLM's ``Hermes2ProToolParser``.

    Unlike the server-side parser, this class does not touch the model
    tokenizer, so it is safe to call from multiple concurrent rollouts.
    """

    tool_call_start_token: str = _TOOL_CALL_START_TOKEN
    tool_call_end_token: str = _TOOL_CALL_END_TOKEN

    @classmethod
    def extract_tool_calls(
        cls, model_output: str
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """Extract OpenAI-style tool_calls from a model output string.

        Returns a tuple of ``(tools_called, tool_calls, content)``.  This
        mirrors the behavior of vLLM's ``Hermes2ProToolParser.extract_tool_calls``
        (``vllm/vllm/tool_parsers/hermes_tool_parser.py:70-120``) but returns
        plain dicts instead of vLLM protocol objects.
        """
        if cls.tool_call_start_token not in model_output:
            return False, [], model_output

        try:
            function_call_tuples = _TOOL_CALL_REGEX.findall(model_output)
            raw_function_calls = [
                json.loads(match) for match in function_call_tuples
            ]
            tool_calls = [
                {
                    "id": _make_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        # function call args are JSON but as a string
                        "arguments": json.dumps(
                            function_call["arguments"], ensure_ascii=False
                        ),
                    },
                }
                for function_call in raw_function_calls
            ]
            content = model_output[
                : model_output.find(cls.tool_call_start_token)
            ]
            return bool(tool_calls), tool_calls, content if content else None
        except Exception:
            return False, [], model_output


class VLLMBackend(AsyncOpenAI):
    """AsyncOpenAI client pointing at a local vLLM OpenAI server.

    Args:
        base_url: vLLM server URL, e.g. ``http://localhost:8000/v1``.
        api_key: Optional API key (vLLM defaults ignore this).
        lora_name: Name of the LoRA module registered on the server.
        plain_text_tools: If True, do not send ``tools=`` to vLLM; inject tool
            schemas into the system prompt and parse ``<tool_call>`` blocks
            locally. This avoids the Hermes parser concurrency bug.
        tool_schema_renderer: Optional callable ``(tools) -> str`` that renders
            OpenAI-style tool definitions into plain text for the system prompt.
            If None, a default Qwen2.5-compatible renderer is used.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "vllm",
        lora_name: str = "scribe_adapter",
        plain_text_tools: bool = True,
        tool_schema_renderer: Optional[Any] = None,
        timeout: Union[float, httpx.Timeout] = httpx.Timeout(None, connect=10.0, read=120.0, write=30.0),
    ):
        super().__init__(base_url=base_url, api_key=api_key, timeout=timeout)
        self.lora_name = lora_name
        self.base_url = base_url
        self.plain_text_tools = plain_text_tools
        self.tool_schema_renderer = tool_schema_renderer or _tools_to_prompt_schema

    def _inject_tool_schema(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Append tool schemas to the system prompt for plain-text generation."""
        schema_text = self.tool_schema_renderer(tools)
        if not schema_text:
            return messages
        out: List[Dict[str, Any]] = []
        injected = False
        for msg in messages:
            if msg.get("role") == "system" and not injected:
                content = msg.get("content") or ""
                if content:
                    content = f"{content}\n\n{schema_text}"
                else:
                    # Match the chat template's behavior for an empty system
                    # message: it keeps the leading newlines before # Tools.
                    content = f"\n\n{schema_text}"
                out.append({**msg, "content": content})
                injected = True
            else:
                out.append(dict(msg))
        if not injected:
            out.insert(0, {"role": "system", "content": schema_text})
        return out

    async def chat_completions_create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Drop-in replacement for ``client.chat.completions.create``.

        When ``plain_text_tools`` is enabled, ``tools``/``tool_choice`` are
        removed from the API call and tool schema is injected into the system
        prompt. The response is then post-processed to reconstruct OpenAI-style
        ``tool_calls`` from ``<tool_call>`` blocks using vLLM's extraction
        logic.
        """
        if not self.plain_text_tools:
            return await super().chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

        messages = self._inject_tool_schema(messages, tools)
        response = await super().chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        tools_called, tool_calls, new_content = (
            HermesToolExtractor.extract_tool_calls(content)
        )

        if tools_called:
            from types import SimpleNamespace

            message_proxy = SimpleNamespace(
                role="assistant",
                content=new_content,
                tool_calls=tool_calls,
                model_dump=lambda: {
                    "role": "assistant",
                    "content": new_content,
                    "tool_calls": tool_calls,
                },
            )
            choice_proxy = SimpleNamespace(
                message=message_proxy,
                finish_reason="tool_calls",
            )
            return SimpleNamespace(choices=[choice_proxy])
        return response

    def update_weights(self, lora_path: str) -> None:
        """Load a new LoRA adapter into the running vLLM server.

        Uses vLLM's ``/v1/load_lora_adapter`` admin endpoint. If the call fails
        (e.g. endpoint disabled), the user can fall back to restarting the
        server with the new ``--lora-modules`` argument.
        """
        import httpx

        url = f"{str(self.base_url).rstrip('/')}/v1/load_lora_adapter"
        payload = {
            "lora_name": self.lora_name,
            "lora_path": lora_path,
            "load_inplace": True,
        }
        try:
            resp = httpx.post(url, json=payload, timeout=60.0)
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LoRA adapter at {lora_path} into vLLM server. "
                f"Ensure the server was started with --enable-lora and that the "
                f"load_lora_adapter admin endpoint is enabled. Original error: {exc}"
            ) from exc


__all__ = ["VLLMBackend", "HermesToolExtractor"]
