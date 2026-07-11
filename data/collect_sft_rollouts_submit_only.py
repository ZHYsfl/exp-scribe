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
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "Scribe" / ".env", override=True)

from Scribe.llm_runtime import LLMConfig
from Scribe.scribe_gym import (
    GymBackedAgent,
    HistoryManager,
    ScribeRunner,
    VLLMBackend,
    render_scribe_blocks,
)
from Scribe.scribe_gym.system_prompts import SUBMIT_ONLY_SYSTEM_PROMPT
from Scribe.scribe_gym.turn_record import StepRecord
from Scribe.llm_runtime.token_counter import DeepSeekTokenCounter

from gsm8k_loader import load_gsm8k
from gsm8k_submit_only_env_factory import make_gsm8k_submit_only_env


SYSTEM_PROMPT = SUBMIT_ONLY_SYSTEM_PROMPT


def step_to_training_example(
    rec: StepRecord,
    task_id: str,
    turn_idx: int,
    step_idx: int,
    turn_reward: float,
) -> Dict[str, Any]:
    """Convert one captured StepRecord into a training example dict.

    The shared system prompt already embeds the tool schema, so the captured
    ``input_messages`` are exactly the runtime prompt the teacher saw. We keep
    them verbatim.
    """
    output_text = render_scribe_blocks(rec.output_blocks)
    return {
        "task_id": task_id,
        "turn_idx": turn_idx,
        "step_idx": step_idx,
        "input_messages": [dict(m) for m in rec.input_messages],
        "output_text": output_text,
        "output_blocks": [
            {"type": b.type.name, "content": b.content} for b in rec.output_blocks
        ],
        "loss_mask": [b.type.name != "TOOL_RESPONSE" for b in rec.output_blocks],
        "turn_reward": turn_reward,
    }


async def collect_one(
    item: Dict[str, Any],
    max_steps: int = 5,
    max_turns: int = 3,
    enable_judge: bool = False,
    min_turn_reward: float = 0.7,
    min_metric_4: float = 0.8,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "deepseek-v4-pro")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

    cfg = LLMConfig(api_key=api_key, model=model, base_url=base_url)
    # Use the same plain-text tool backend as GRPO rollout so the teacher sees
    # the exact Qwen-style tool schema that the student will see at inference.
    backend = VLLMBackend(
        base_url=base_url,
        api_key=api_key,
        lora_name=model,
        plain_text_tools=True,
    )

    ws_root = Path("/tmp/gsm8k_sft") / item["task_id"]
    ws_root.mkdir(parents=True, exist_ok=True)

    env = make_gsm8k_submit_only_env(item, workspace_root=ws_root, max_steps=max_steps)
    counter = DeepSeekTokenCounter()
    hm = HistoryManager(counter, k=2, hard_limit=4096, compression_margin=512)
    agent = GymBackedAgent(
        config=cfg, env=env, history_manager=hm, debug=False,
        enable_judge=enable_judge,
        llm_backend=backend,
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
        # Data quality gate: only keep turns that are correct and well-formatted.
        bd = turn.reward_breakdown
        m4 = getattr(bd, "metric_4", 1.0) if bd is not None else 1.0
        if turn.reward < min_turn_reward or m4 < min_metric_4:
            print(
                f"  skip turn {turn_idx}: reward={turn.reward:.3f}, m4={m4:.3f} "
                f"(below gate reward>={min_turn_reward}, m4>={min_metric_4})"
            )
            continue
        for step_idx, rec in enumerate(turn.step_records):
            ex = step_to_training_example(
                rec, item["task_id"], turn_idx, step_idx, turn.reward
            )
            ex["right_answer"] = item["right_answer"]
            ex["final_assistant_content"] = turn.step_records[-1].raw_assistant_message.get(
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
    min_turn_reward: float = 0.7,
    min_metric_4: float = 0.9,
):
    items = load_gsm8k("train", "main", limit=num_samples)
    if output_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        output_path = str(repo_root / "data" / "gsm8k_sft_steps_submit_only.jsonl")
    print(f"Collecting SFT rollouts for {len(items)} GSM8K samples...")
    print(f"Teacher model: {os.getenv('LLM_MODEL', 'deepseek-v4-pro')}")
    print(f"LLM-as-judge enabled: {enable_judge}")
    print(f"Quality gate: reward>={min_turn_reward}, m4>={min_metric_4}")
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
                    min_turn_reward=min_turn_reward,
                    min_metric_4=min_metric_4,
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
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: <repo-root>/data/gsm8k_sft_steps_submit_only.jsonl).")
    parser.add_argument("--enable_judge", action="store_true",
                        help="Enable LLM-as-judge for metrics 7-10 (extra API calls).")
    parser.add_argument("--min_turn_reward", type=float, default=0.7,
                        help="Minimum turn reward for a turn to be kept in SFT data.")
    parser.add_argument("--min_metric_4", type=float, default=0.9,
                        help="Minimum metric_4 (format) score for a turn to be kept.")
    args = parser.parse_args()

    asyncio.run(main(
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        output_path=args.output,
        enable_judge=args.enable_judge,
        min_turn_reward=args.min_turn_reward,
        min_metric_4=args.min_metric_4,
    ))
