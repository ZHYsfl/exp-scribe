"""GSM8K → submit-only ScribeEnv factory.

This is an ablation environment for validating the SCRIBE protocol without the
complexity of bash tool-calling / JSON escaping. It exposes only the ``submit``
tool, so the model must reason entirely in its own <think> blocks and submit a
final answer directly.

Everything else (turn structure, 15-metric reward, trajectory decay, GRPO credit
assignment) stays identical to the full LinuxWorkspaceEnv setup.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_SCRIBE_ROOT = Path("/root/autodl-tmp/Scribe").resolve()
if str(_SCRIBE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIBE_ROOT.parent))

from Scribe.scribe_gym import LinuxWorkspaceEnv  # noqa: E402


class SubmitOnlyWorkspaceEnv(LinuxWorkspaceEnv):
    """GSM8K environment with only the submit tool.

    Inherits all GSM8K-specific logic (right-answer checking, submit penalty,
    max-steps truncation) from ``LinuxWorkspaceEnv``, but does not expose
    ``bash`` or any other linux tools. This lets us test whether the SCRIBE
    turn/step/format protocol trains cleanly when JSON tool-call syntax is
    removed from the problem.
    """

    def __init__(
        self,
        task_description: str,
        workspace_root: str | Path = ".",
        max_steps: int = 32,
        right_answer: str | None = None,
        submit_penalty: float = 0.5,
    ):
        # Initialize the base ScribeEnv state directly, skipping the
        # basic_linux_tools creation in LinuxWorkspaceEnv.__init__.
        self.task_description = task_description
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        if not self.workspace_root.is_dir():
            raise NotADirectoryError(
                f"SubmitOnlyWorkspaceEnv workspace_root is not a directory: {self.workspace_root}"
            )
        self.max_steps = max_steps
        self.right_answer = right_answer
        self.submit_penalty = submit_penalty
        self._step_count = 0
        self._submitted = False
        self._submit_count = 0
        self._answer: str | None = None

        # Only the submit tool.
        from Scribe.scribe_gym.tool_calling_env import ToolCallingScribeEnv
        from Scribe.llm_runtime import Tool
        from Scribe.llm_runtime._tool_helpers import schema

        tools = [
            Tool(
                name="submit",
                description="Submit the final answer to complete the task.",
                function=self._submit,
                parameters=schema(
                    {
                        "answer": {
                            "type": "string",
                            "description": "The final answer or summary of the completed task.",
                        }
                    },
                    ["answer"],
                ),
            )
        ]
        ToolCallingScribeEnv.__init__(self, tools=tools)


def make_gsm8k_submit_only_env(
    item: Dict[str, Any],
    workspace_root: str | Path = "/tmp/gsm8k_ws_submit_only",
    max_steps: int = 3,
) -> SubmitOnlyWorkspaceEnv:
    """Create a submit-only GSM8K environment for one sample."""
    task_description = f"Solve this math problem:\n\n{item['task_description']}"
    return SubmitOnlyWorkspaceEnv(
        task_description=task_description,
        workspace_root=Path(workspace_root) / item["task_id"],
        max_steps=max_steps,
        right_answer=item["right_answer"],
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gsm8k_loader import load_gsm8k

    items = load_gsm8k("train", "main", limit=1)
    env = make_gsm8k_submit_only_env(items[0])
    obs, info = env.reset()
    print("task:", obs)
    print("right_answer:", info.get("right_answer"))
    print("tools:", [t["function"]["name"] for t in env.get_tools()])
