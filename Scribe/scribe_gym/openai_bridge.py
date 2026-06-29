import json
from typing import Any, Dict, List, Tuple

from ..llm_runtime import Agent, LLMConfig

from .chat_template import assistant_message_to_scribe
from .parsers import ScribeBlock, ScribeBlockType, parse_scribe_blocks
from .tool_calling_env import ToolCallingScribeEnv


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


class GymBackedAgent(Agent):
    def __init__(
        self,
        config: LLMConfig,
        env: ToolCallingScribeEnv,
        max_tool_retries: int = 3,
        debug: bool = False,
        compressor: Any | None = None,
    ):
        super().__init__(
            config, max_tool_retries=max_tool_retries, debug=debug, compressor=compressor
        )
        self.env = env
        self.trajectory: List[Dict[str, Any]] = []

    def _get_tools(self) -> List[Dict[str, Any]]:
        return self.env.get_tools()

    async def _env_step(
        self, action: Any
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        astep = getattr(self.env, "astep", None)
        if astep is not None:
            return await astep(action)
        return self.env.step(action)

    def _message_to_scribe_blocks(
        self, message: Dict[str, Any]
    ) -> Tuple[str, List[ScribeBlock], bool]:
        action_text = assistant_message_to_scribe(message)
        blocks, is_valid = parse_scribe_blocks(action_text)
        return action_text, blocks, is_valid

    def _extract_tool_calls(self, blocks: List[ScribeBlock]) -> List[Dict[str, Any]]:
        return [
            b.parsed
            for b in blocks
            if b.type == ScribeBlockType.TOOL_CALL and b.parsed is not None
        ]

    async def _step_env_for_message(
        self, message: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], bool, bool]:
        action_text, blocks, is_valid = self._message_to_scribe_blocks(message)
        parsed_tool_calls = self._extract_tool_calls(blocks)

        action = parsed_tool_calls if parsed_tool_calls else action_text

        if self.debug:
            print("[Debug] env.step action:\n", action)

        obs, reward, terminated, truncated, info = await self._env_step(action)

        if self.debug:
            print(
                f"[Debug] reward={reward} terminated={terminated} truncated={truncated}"
            )

        self.trajectory.append({
            "action_text": action_text,
            "blocks": blocks,
            "is_valid": is_valid,
            "action": action,
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        })

        done = terminated or truncated
        if not parsed_tool_calls:
            return obs, [], done, False

        message_tool_calls = message.get("tool_calls", [])
        tool_messages = observation_to_tool_messages(obs, message_tool_calls, info)
        return obs, tool_messages, done, True

    async def _get_tool_response_observations(self, response) -> List[Dict[str, Any]]:
        message = response.choices[0].message.model_dump()
        _, tool_messages, done, _ = await self._step_env_for_message(message)
        return tool_messages

    def _should_stop_after_tool_response(
        self, tool_responses: List[Dict[str, Any]]
    ) -> bool:
        if not self.trajectory:
            return False
        return self.trajectory[-1]["terminated"] or self.trajectory[-1]["truncated"]

    async def _on_final_response(self, response) -> None:
        message = response.choices[0].message.model_dump()
        if not message.get("tool_calls"):
            await self._step_env_for_message(message)

    def _has_tool_errors(self, tool_responses: List[Dict[str, Any]]) -> bool:
        # Disable the base-class retry prompt. Tool error content is already
        # present in the tool response messages; the model can adjust from
        # there without an extra "please retry" user message.
        return False


__all__ = [
    "GymBackedAgent",
    "observation_to_tool_messages",
    "tool_calls_to_action",
]
