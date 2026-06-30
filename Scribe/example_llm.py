"""scribe_gym example with a real LLM.

Drives GymBackedAgent with DeepSeek (OpenAI-compatible API) through ScribeRunner
for a full tool-calling episode: the agent autonomously calls linux tools to
solve the task and submits the final answer.

Env vars are read from the sibling .env file:
    LLM_API_KEY=...
    LLM_MODEL=deepseek-chat
    LLM_BASE_URL=https://api.deepseek.com

Run:
    python Scribe/example_llm.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make the Scribe package importable no matter where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Explicitly load the sibling .env file.
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from Scribe.llm_runtime import LLMConfig  # noqa: E402
from Scribe.scribe_gym import GymBackedAgent, LinuxWorkspaceEnv, ScribeRunner  # noqa: E402
from Scribe.scribe_gym.chat_template import render_messages  # noqa: E402

# Base system prompt. The task description is appended to it by ScribeRunner.
SYSTEM_PROMPT = (
    "You are a tool-using assistant. Work as follows:\n"
    "1. Solve the problem step by step using the provided linux tools "
    "(bash/read_file/write_file/list_dir, etc.).\n"
    "2. Think about what to do at each step, then call a tool.\n"
    "3. Once you have the result, call the submit tool with the final answer; "
    "after submitting, finish with a short natural-language summary.\n"
    "Note: always provide valid JSON arguments for tool calls."
)


def build_config() -> LLMConfig:
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise SystemExit("Missing LLM_API_KEY. Check Scribe/.env.")
    print(f"[config] model={model}  base_url={base_url}")
    return LLMConfig(api_key=api_key, model=model, base_url=base_url)


async def main() -> None:
    config = build_config()

    # Use a fixed directory as the workspace so you can directly observe the
    # files the agent creates or modifies.
    workspace = Path("/root/autodl-tmp/exp")
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"[workspace] {workspace}  (agent artifacts land here)")

    # 1. Env. The task_description is appended to the system prompt by ScribeRunner.
    env = LinuxWorkspaceEnv(
        task_description=(
            "Compute 456 raised to the power of 3, then call submit with the "
            "result. The answer should be the numeric result only."
        ),
        workspace_root=workspace,
        max_steps=12,
        right_answer="94818816",  # 456 ** 3
    )

    # 2. Agent: GymBackedAgent bridges the LLM's tool_calls to env.step.
    agent = GymBackedAgent(config=config, env=env, debug=True)

    # 3. Runner: controls the episode (reward threshold / max turns / feedback).
    runner = ScribeRunner(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        done_mode="threshold",
        reward_threshold=1.0,
        max_turns=5,
    )

    print("\n" + "#" * 70)
    print("# episode start")
    print("#" * 70)
    observations = await runner.run()
    print("#" * 70)
    print("# episode end")
    print("#" * 70)

    # 1) observations: the return value of runner.run()
    print("\n" + "=" * 70)
    print("(1) observations")
    print("=" * 70)
    for i, m in enumerate(observations):
        print(f"--- [{i}] ---")
        print(str(m))

    # 2) Rendered through the chat template — this reflects what the model
    # actually receives: tools are injected from the `tools=` API field by the
    # chat template (the same way the server does it), NOT baked into the
    # system content by ScribeRunner.
    print("\n" + "=" * 70)
    print("(2) chat template rendering (what the model actually receives)")
    print("=" * 70)
    rendered = render_messages(
        observations, tools=agent._get_tools(), add_generation_prompt=False
    )
    print(rendered)

    # 3) All fields of agent.trajectory.
    print("\n" + "=" * 70)
    print(f"(3) agent.trajectory  ({len(agent.trajectory)} steps)")
    print("=" * 70)
    for i, step in enumerate(agent.trajectory, 1):
        print(f"\n--- step [{i}] ---")
        for k, v in step.items():
            print(f"  {k}: {v}")

    env.close()


if __name__ == "__main__":
    asyncio.run(main())
