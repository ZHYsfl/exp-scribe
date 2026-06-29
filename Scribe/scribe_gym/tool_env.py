import asyncio
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from ..llm_runtime import Tool
from ..llm_runtime._tool_helpers import schema
from ..llm_runtime.basic_linux_tools import create_basic_linux_tools

from .tool_calling_env import ToolCallingScribeEnv

def _make_async(func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    if inspect.iscoroutinefunction(func):
        return func

    async def wrapper(**kwargs: Any) -> Any:
        return await asyncio.to_thread(func, **kwargs)

    return wrapper

class LinuxWorkspaceEnv(ToolCallingScribeEnv):

    def __init__(
        self,
        task_description: str,
        workspace_root: str | Path = ".",
        max_steps: int = 32,
        right_answer: Optional[str] = None,
    ):
        self.task_description = task_description
        self.workspace_root = Path(workspace_root).resolve()
        self.max_steps = max_steps
        self.right_answer = right_answer
        self._step_count = 0
        self._submitted = False
        self._answer: str | None = None

        tools = create_basic_linux_tools(self.workspace_root)
        async_tools = [
            Tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                function=_make_async(tool.function),
            )
            for tool in tools
        ]
        async_tools.append(self._create_submit_tool())
        super().__init__(tools=async_tools)

    def _create_submit_tool(self) -> Tool:
        return Tool(
            name="submit",
            description="Submit the final answer to complete the task.",
            function=self._submit,
            parameters=schema({
                "answer": {
                    "type": "string",
                    "description": "The final answer or summary of the completed task.",
                }
            }, ["answer"]),
        )

    def _submit(self, answer: str) -> str:
        self._submitted = True
        self._answer = answer
        return f"Submitted final answer: {answer}"

    def get_task_description(self) -> str:
        return self.task_description

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self._step_count = 0
        self._submitted = False
        self._answer = None
        obs = self.task_description
        info = {
            "task_description": obs,
            "tools": self.get_tools(),
            "workspace_root": str(self.workspace_root),
            "right_answer": self.right_answer,
        }
        return obs, info

    def reset_step_count(self) -> None:
        self._step_count = 0

    async def astep(
        self, action: Any
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, list) and action:
            self._step_count += 1
        obs, reward, terminated, truncated, info = await super().astep(action)

        if self._step_count >= self.max_steps and not terminated and not self._submitted:
            truncated = True
        return obs, reward, terminated, truncated, info

    def _compute_answer_reward(self) -> float:
        if self.right_answer is None:
            return 1.0
        return 1.0 if self._answer.strip() == self.right_answer.strip() else 0.0

    def _handle_non_tool_action(
        self,
        action: str,
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._answer is None:
            return (
                "[STOP] No executable tool calls were produced; episode terminated.",
                0.0,
                True,
                False,
                {"done_reason": "no_tool_call", "raw_action": action},
            )

        reward = self._compute_answer_reward()
        return (
            f"[STOP] Final answer submitted: {self._answer}",
            reward,
            True,
            False,
            {
                "done_reason": "no_tool_call",
                "answer": self._answer,
                "right_answer": self.right_answer,
                "raw_action": action,
            },
        )

    def _compute_transition(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[float, bool, bool, Dict[str, Any]]:
        info = {"right_answer": self.right_answer}
        if self._submitted:
            info["answer"] = self._answer
        return 0.0, False, False, info

__all__ = ["LinuxWorkspaceEnv"]
