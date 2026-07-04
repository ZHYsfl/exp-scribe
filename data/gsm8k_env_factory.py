"""GSM8K → ScribeEnv factory.

Builds `LinuxWorkspaceEnv` instances from the GSM8K loader output.
This file lives under `/root/autodl-tmp/data/` so data logic stays out of the
Scribe source tree; it only imports `Scribe.scribe_gym` to construct the env.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Scribe lives at /root/autodl-tmp/Scribe; add its parent to the path so we can
# import `Scribe.scribe_gym` regardless of where this script is invoked from.
_SCRIBE_ROOT = Path("/root/autodl-tmp/Scribe").resolve()
if str(_SCRIBE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIBE_ROOT.parent))

from Scribe.scribe_gym import LinuxWorkspaceEnv  # noqa: E402


def make_gsm8k_env(
    item: Dict[str, Any],
    workspace_root: str | Path = "/tmp/gsm8k_ws",
    max_steps: int = 3,
) -> LinuxWorkspaceEnv:
    """Create a LinuxWorkspaceEnv for one GSM8K sample.

    The task description contains only the math problem itself. Tool usage
    instructions and the submit rule live in the runner's system prompt.
    """
    task_description = f"Solve this math problem:\n\n{item['task_description']}"
    return LinuxWorkspaceEnv(
        task_description=task_description,
        workspace_root=Path(workspace_root) / item["task_id"],
        max_steps=max_steps,
        right_answer=item["right_answer"],
    )


if __name__ == "__main__":
    from gsm8k_loader import load_gsm8k

    items = load_gsm8k("train", "main", limit=1)
    env = make_gsm8k_env(items[0])
    obs, info = env.reset()
    print("task:", obs)
    print("right_answer:", info.get("right_answer"))
    print("tools:", [t["function"]["name"] for t in env.get_tools()])
