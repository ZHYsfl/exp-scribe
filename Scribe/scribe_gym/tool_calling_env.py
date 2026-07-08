import asyncio
import json
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from ..llm_runtime import Tool

from .base_env import ScribeEnv
from .chat_template import tool_results_to_scribe


ToolFunction = Callable[..., Union[Any, Awaitable[Any]]]


class ToolCallingScribeEnv(ScribeEnv):

    def __init__(self, tools: Optional[List[Tool]] = None):
        self.tools: List[Tool] = list(tools or [])
        self._tool_map = {tool.name: tool for tool in self.tools}

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        obs = self.get_task_description()
        info = {"task_description": obs, "tools": self.get_tools()}
        return obs, info

    def step(self, action: Any) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        try:
            return asyncio.run(self.astep(action))
        except RuntimeError as exc:
            raise RuntimeError(
                "ToolCallingScribeEnv.step() cannot be called from a running "
                "event loop. Use astep() in async contexts."
            ) from exc

    async def astep(
        self, action: Any
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, list):
            tool_calls = action
            if not tool_calls:
                return self._handle_non_tool_action("")
            results = await self._execute_tool_calls(tool_calls)
            observation = self._format_results(results)
            reward, terminated, truncated, info = self._compute_transition(results)
            info["tool_results"] = results
            info["tool_messages"] = [
                {"role": "tool", "content": r["output"]}
                for r in results
            ]
            return observation, reward, terminated, truncated, info

        return self._handle_non_tool_action(str(action))

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        tasks = [self._execute_single_tool_call(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)

    async def _execute_single_tool_call(
        self, tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        # Robustness: models occasionally emit arguments as a JSON-encoded string
        # instead of an object (e.g. {"arguments": "{\"command\": \"...\"}"}).
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        tool = self._tool_map.get(name)
        if tool is None:
            return {
                "name": name,
                "output": f"[NOT_FOUND] No tool named '{name}'",
                "status": "error",
                "error_type": "not_found",
            }

        try:
            output = tool.function(**arguments)
            if asyncio.iscoroutine(output) or asyncio.isfuture(output):
                output = await output
            return {
                "name": name,
                "output": str(output),
                "status": "success",
            }
        except Exception as exc:
            import traceback
            return {
                "name": name,
                "output": f"[EXEC_ERROR] {exc}\n{traceback.format_exc()}",
                "status": "error",
                "error_type": "exec_error",
            }

    @abstractmethod
    def _handle_non_tool_action(
        self, action: str
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        ...

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        return tool_results_to_scribe(results)

    @abstractmethod
    def _compute_transition(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[float, bool, bool, Dict[str, Any]]:
        ...


__all__ = ["ToolCallingScribeEnv"]
