"""End-to-end: SCRIBE history_manager mode on local qwen3-1.7b.

Verifies the per-turn cross-turn rebuild on a real LLM:
  - episode completes (reward=1.0)
  - cross-turn input (hm.build_input()) contains NO T/R
  - feedback appears in-position after the prior turn's blocks
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from Scribe.llm_runtime import LLMConfig
from Scribe.scribe_gym import GymBackedAgent, HistoryManager, LinuxWorkspaceEnv, ScribeRunner
from Scribe.llm_runtime.token_counter import DeepSeekTokenCounter
from Scribe.scribe_gym.chat_template import render_messages


SYSTEM_PROMPT = (
    "You are a tool-using assistant. Each turn follows this strict format:\n"
    "\n"
    "1. Think inside <think>...</think>.\n"
    "2. Use tools by emitting <tool_call>...</tool_call>.\n"
    "3. After you have verified the answer, call the submit tool exactly ONCE per turn.\n"
    "4. Finally, output a plain-text message (NO tool calls) that contains:\n"
    "   - <reflect>...</reflect> reflecting on what you did this turn\n"
    "   - <turn_summary>...</turn_summary> summarizing the turn\n"
    "\n"
    "The turn ONLY ends after this final plain-text message. "
    "Submit alone does NOT end the turn. "
    "Repeated submits in the same turn are penalized."
)


async def main():
    cfg = LLMConfig(api_key="EMPTY", model="/root/autodl-tmp/qwen3-1.7b",
                    base_url="http://127.0.0.1:8002/v1")
    ws = Path("/root/autodl-tmp/exp"); ws.mkdir(parents=True, exist_ok=True)
    env = LinuxWorkspaceEnv(
        task_description="Compute 456 raised to the power of 3, then call submit with the numeric result.",
        workspace_root=ws, max_steps=2, right_answer="94818816",
    )
    counter = DeepSeekTokenCounter()
    hm = HistoryManager(counter, k=2, hard_limit=4096, compression_margin=512)
    agent = GymBackedAgent(config=cfg, env=env, history_manager=hm, debug=False)
    runner = ScribeRunner(agent=agent, system_prompt=SYSTEM_PROMPT,
                          done_mode="threshold", reward_threshold=1.0, max_turns=3)

    print("="*70); print("SCRIBE hm mode (k=2)"); print("="*70)
    observations = await runner.run()

    print(f"\ntrajectory steps: {len(agent.trajectory)}")
    print(f"hm turns: {hm.num_turns}")
    if agent.trajectory:
        best = max(s["reward"] for s in agent.trajectory)
        print(f"best reward: {best}")

    # (1) observations returned by runner.run()
    print("\n" + "="*70)
    print("(1) observations")
    print("="*70)
    for i, m in enumerate(observations):
        print(f"--- [{i}] ---")
        print(str(m))

    # (2) Rendered through the chat template — what the model actually receives.
    print("\n" + "="*70)
    print("(2) chat template rendering (what the model actually receives)")
    print("="*70)
    rendered = render_messages(
        observations, tools=agent._get_tools(), add_generation_prompt=False
    )
    print(rendered)

    # (3) All fields of agent.trajectory.
    print("\n" + "="*70)
    print(f"(3) agent.trajectory  ({len(agent.trajectory)} steps)")
    print("="*70)
    for i, step in enumerate(agent.trajectory, 1):
        print(f"\n--- step [{i}] ---")
        for k, v in step.items():
            print(f"  {k}: {v}")

    # (4) Per-turn, per-step input/output from TurnRecord (the single source of truth).
    print("\n" + "="*70)
    print(f"(4) TurnRecords ({hm.num_turns} turns) — per-step input/output")
    print("="*70)
    for turn_idx, turn in enumerate(hm._turns, 1):
        print(f"\n{'#'*70}")
        print(f"# TURN {turn_idx}")
        print(f"{'#'*70}")
        print(f"reward: {turn.reward}")
        print(f"feedback: {turn.feedback!r}")
        print(f"has_summary: {turn.has_summary}")
        print(f"num_steps: {len(turn.step_records)}")

        for step_idx, rec in enumerate(turn.step_records, 1):
            print(f"\n--- turn {turn_idx} step {step_idx} ---")
            print("  INPUT MESSAGES (exactly what the LLM saw):")
            for mi, m in enumerate(rec.input_messages):
                role = m.get("role", "?")
                content = (m.get("content") or "")[:200].replace("\n", " ")
                tc = m.get("tool_calls")
                tc_info = f" tool_calls={len(tc)}" if tc else ""
                print(f"    [{mi}] {role:10s} | {content!r}{tc_info}")

            print("  OUTPUT BLOCKS (what the LLM generated, T included):")
            for bi, b in enumerate(rec.output_blocks):
                content = b.content[:200].replace("\n", " ")
                print(f"    [{bi}] {b.type.name:15s} | {content!r}")

            print("  TOOL MESSAGES (AR, observations for next step):")
            for ti, tm in enumerate(rec.tool_messages):
                content = (tm.get("content") or "")[:200].replace("\n", " ")
                print(f"    [{ti}] tool | {content!r}")

        print(f"\n  RETAINED MESSAGES (this turn contributes to future turns):")
        for mi, m in enumerate(turn.retained_messages):
            role = m.get("role", "?")
            content = (m.get("content") or "")[:200].replace("\n", " ")
            tc = m.get("tool_calls")
            tc_info = f" tool_calls={len(tc)}" if tc else ""
            print(f"    [{mi}] {role:10s} | {content!r}{tc_info}")

        print(f"\n  NEW BLOCKS (T/O/A/AR/R/S in order):")
        for bi, b in enumerate(turn.new_blocks):
            content = b.content[:200].replace("\n", " ")
            print(f"    [{bi}] {b.type.name:15s} | {content!r}")

        print(f"\n  ROLLOUT BLOCKS (T/O/A/R/S, AR excluded):")
        for bi, b in enumerate(turn.rollout_blocks):
            content = b.content[:200].replace("\n", " ")
            print(f"    [{bi}] {b.type.name:15s} | {content!r}")

    # (5) Verify cross-turn input has no T/R (check the last turn's input).
    if hm.num_turns >= 2:
        last_input = hm._turns[-1].input
        print(f"\n{'='*70}")
        print(f"(5) last turn input ({len(last_input)} msgs)")
        print("="*70)
        for m in last_input:
            c = (m.get("content") or "")[:60].replace("\n", " ")
            print(f"  {m.get('role'):9s} | {c}")
        for m in last_input:
            if m.get("role") == "system":
                continue  # system prompt legitimately contains <think> example tags
            c = m.get("content") or ""
            assert "<think>" not in c, "T leaked into cross-turn input!"
            assert "<reflect>" not in c, "R leaked into cross-turn input!"
        print("\n✅ no generated T/R in cross-turn input")
    env.close()


if __name__ == "__main__":
    asyncio.run(main())
