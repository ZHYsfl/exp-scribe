"""Collect SFT rollouts for SCRIBE using a strong teacher model.

Loads GSM8K samples, runs them with DeepSeek (or any strong model configured
via Scribe/.env), and saves the resulting Step-level training data as JSONL.

Each line in the output JSONL corresponds to one LLM call (one Step):
  - input_messages: the exact messages fed to the model
  - output_text: the model's raw output with SCRIBE tags
  - loss_mask: per-block mask (True for model-generated blocks)
  - reward: the turn-level reward propagated to this step
  - metadata: task_id, turn index, step index, etc.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "Scribe" / ".env", override=True)

from Scribe.llm_runtime import LLMConfig
from Scribe.scribe_gym import (
    GymBackedAgent,
    HistoryManager,
    ScribeRunner,
    expand_turn,
    render_scribe_blocks,
)
from Scribe.llm_runtime.token_counter import DeepSeekTokenCounter

from gsm8k_loader import load_gsm8k
from gsm8k_env_factory import make_gsm8k_env


SYSTEM_PROMPT = (
    "You are a tool-using assistant following the SCRIBE protocol.\n"
    "\n"
    "Each turn follows this strict format:\n"
    "1. Think inside <think>...</think> (optional).\n"
    "2. Use tools by emitting <tool_call>...</tool_call>.\n"
    "3. When you are ready to answer, call the submit tool exactly ONCE per turn.\n"
    "4. Finally, output a plain-text message (NO tool calls) that contains:\n"
    "   - <reflect>...</reflect> reflecting on what you did this turn\n"
    "   - <turn_summary>...</turn_summary> summarizing the turn\n"
    "\n"
    "The turn ONLY ends after this final plain-text message. "
    "Submit alone does NOT end the turn. "
    "Repeated submits in the same turn are penalized."
)


def step_to_training_example(
    step,
    task_id: str,
    turn_idx: int,
    step_idx: int,
    turn_reward: float,
) -> Dict[str, Any]:
    """Convert one Step into a training example dict."""
    output_text = render_scribe_blocks(step.output)
    return {
        "task_id": task_id,
        "turn_idx": turn_idx,
        "step_idx": step_idx,
        "input_messages": step.input,
        "output_text": output_text,
        "output_blocks": [
            {"type": b.type.name, "content": b.content} for b in step.output
        ],
        "loss_mask": [b.type.name != "TOOL_RESPONSE" for b in step.output],
        "turn_reward": turn_reward,
    }


async def collect_one(
    item: Dict[str, Any],
    max_steps: int = 5,
    max_turns: int = 3,
    enable_judge: bool = False,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "deepseek-v4-pro")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

    cfg = LLMConfig(api_key=api_key, model=model, base_url=base_url)

    ws_root = Path("/tmp/gsm8k_sft") / item["task_id"]
    ws_root.mkdir(parents=True, exist_ok=True)

    env = make_gsm8k_env(item, workspace_root=ws_root, max_steps=max_steps)
    counter = DeepSeekTokenCounter()
    hm = HistoryManager(counter, k=2, hard_limit=4096, compression_margin=512)
    agent = GymBackedAgent(
        config=cfg, env=env, history_manager=hm, debug=False,
        enable_judge=enable_judge,
    )
    runner = ScribeRunner(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        done_mode="threshold",
        reward_threshold=1.0,
        max_turns=max_turns,
    )

    await runner.run()

    examples: List[Dict[str, Any]] = []
    for turn_idx, turn in enumerate(hm._turns):
        steps = expand_turn(turn)
        for step_idx, step in enumerate(steps):
            ex = step_to_training_example(
                step, item["task_id"], turn_idx, step_idx, turn.reward
            )
            ex["right_answer"] = item["right_answer"]
            ex["submitted_answer"] = turn.step_records[-1].raw_assistant_message.get(
                "content", ""
            ) if turn.step_records else ""
            examples.append(ex)

    env.close()
    return examples


async def main(
    num_samples: int = 20,
    max_steps: int = 5,
    max_turns: int = 3,
    output_path: Optional[str] = None,
    enable_judge: bool = False,
):
    items = load_gsm8k("train", "main", limit=num_samples)
    if output_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        output_path = str(repo_root / "data" / "gsm8k_sft_steps.jsonl")
    print(f"Collecting SFT rollouts for {len(items)} GSM8K samples...")
    print(f"Teacher model: {os.getenv('LLM_MODEL', 'deepseek-v4-pro')}")
    print(f"LLM-as-judge enabled: {enable_judge}")
    print(f"Output: {output_path}")
    print("=" * 70)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    total_turns = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(items, 1):
            print(f"\n[{i}/{len(items)}] {item['task_id']}")
            try:
                examples = await collect_one(
                    item, max_steps=max_steps, max_turns=max_turns,
                    enable_judge=enable_judge,
                )
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                total_examples += len(examples)
                total_turns += max(ex["turn_idx"] for ex in examples) + 1 if examples else 0
                print(f"  wrote {len(examples)} steps")
            except Exception as exc:
                print(f"  ERROR: {exc}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Samples attempted: {num_samples}")
    print(f"Total step examples written: {total_examples}")
    print(f"Output file: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: <repo-root>/data/gsm8k_sft_steps.jsonl).")
    parser.add_argument("--enable_judge", action="store_true",
                        help="Enable LLM-as-judge for metrics 7-10 (extra API calls).")
    args = parser.parse_args()

    asyncio.run(main(
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        output_path=args.output,
        enable_judge=args.enable_judge,
    ))
