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
    expand_turn,
    render_scribe_blocks,
)
from Scribe.llm_runtime.token_counter import DeepSeekTokenCounter

from gsm8k_loader import load_gsm8k
from gsm8k_env_factory import make_gsm8k_env


SYSTEM_PROMPT = (
    "You are a SCRIBE agent. You solve tasks through explicit turns. "
    "Each turn is one self-contained solving attempt from kickoff to final answer.\n"
    "\n"
    "== SCRIBE BLOCKS (use exactly these tags) ==\n"
    "- <think>...</think> (T): optional step-level reasoning. Stripped from future context.\n"
    "- <tool_call>...</tool_call> (A): one tool call as JSON {\"name\": ..., \"arguments\": {...}}.\n"
    "- <tool_response>...</tool_response> (AR): produced by the environment ONLY. Never generate this yourself.\n"
    "- OUTPUT (O): plain text without any tags.\n"
    "- <reflect>...</reflect> (R): reflection on this turn. ONLY in the final step.\n"
    "- <turn_summary>...</turn_summary> (S): concise summary of this turn. ONLY in the final step.\n"
    "\n"
    "== TURN STRUCTURE ==\n"
    "A turn has multiple steps (LLM calls). Each non-final step produces T/O/A blocks. "
    "After a tool_call you get a tool_response and continue. "
    "The FINAL step must be a plain-text message with O+R+S and NO tool_calls.\n"
    "\n"
    "Final step required order:\n"
    "1. OUTPUT: final answer / concluding plain text\n"
    "2. <reflect>...</reflect>\n"
    "3. <turn_summary>...</turn_summary>\n"
    "\n"
    "== SUBMIT RULE (CRITICAL) ==\n"
    "You MUST call the submit tool BEFORE writing the final O+R+S message. "
    "Writing the answer in plain text does NOT count as a submission. "
    "Calling submit does NOT end the turn; after submit you still produce O+R+S.\n"
    "\n"
    "The submit answer must be ONLY the final answer, with no extra words, units, or symbols:\n"
    "- CORRECT: {\"name\": \"submit\", \"arguments\": {\"answer\": \"10\"}}\n"
    "- WRONG:   {\"name\": \"submit\", \"arguments\": {\"answer\": \"$10.00\"}}\n"
    "- WRONG:   {\"name\": \"submit\", \"arguments\": {\"answer\": \"Betty needs $5 more.\"}}\n"
    "If the expected answer is a number, submit just the number. If it is a word, submit just the word.\n"
    "\n"
    "== HARD RULES ==\n"
    "1. Call submit exactly ONCE per turn.\n"
    "2. Do NOT call multiple tools with the same name AND same arguments in one step.\n"
    "3. Do NOT repeat the exact same tool call across consecutive steps (e.g. bash ls -> bash ls).\n"
    "4. Every tool call must include all required arguments (e.g. bash needs 'command').\n"
    "5. <reflect> and <turn_summary> appear ONLY in the final step.\n"
    "6. The final step must contain NO tool_calls.\n"
    "7. Do not invent tags such as <submit>, <bash>, <action>, <plan>, <final_answer>.\n"
    "8. Keep outputs concise; avoid repeating the same phrase.\n"
    "\n"
    "== HOW TO MAXIMIZE YOUR REWARD (15 metrics) ==\n"
    "1. Answer correctly and end naturally: the last step must be non-tool, not truncated.\n"
    "2. Submit exactly once; no parallel duplicate tool calls.\n"
    "3. Use as few steps as possible.\n"
    "4. Format: only allowed tags, R+S last, no R/S in non-final steps.\n"
    "5. Be concise: fewer rollout tokens is better.\n"
    "6. Turn_summary should reuse key tokens/concepts from output and tool_call blocks.\n"
    "7. Faithfulness: summary must truthfully report what you did.\n"
    "8. Direction neutrality: summary looks back only, no future planning.\n"
    "9. Turn focus: summary describes ONLY this turn.\n"
    "10. Fluency: summary reads like natural language.\n"
    "11. Compression: summary is shorter than the output+tool_call blocks combined.\n"
    "12. No verbatim copy-paste: rephrase, don't lift 4+ word runs from earlier blocks.\n"
    "13. No internal repetition: avoid looping phrases within any block.\n"
    "14. No malformed tool calls: every tool call must have valid, complete arguments.\n"
    "15. No repeated tool calls: never emit the same (name, args) call twice in one turn.\n"
    "\n"
    "== REFLECT BLOCK GUIDANCE ==\n"
    "<reflect> is a chain-of-thought for THIS specific problem. Do not use a generic checklist. "
    "Analyze what you actually computed, whether the arithmetic is correct, which tool calls you made, "
    "and what specific facts/numbers the <turn_summary> must retain. "
    "Then write a concise <turn_summary> that is faithful, retrospective, and focused only on this turn."
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
    seed: Optional[int] = None,
    concurrency: int = 5,
):
    items = load_gsm8k("train", "main", limit=num_samples, seed=seed)
    if output_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        output_path = str(repo_root / "data" / "gsm8k_sft_steps.jsonl")
    print(f"Collecting SFT rollouts for {len(items)} GSM8K samples...")
    print(f"Teacher model: {os.getenv('LLM_MODEL', 'deepseek-v4-pro')}")
    print(f"LLM-as-judge enabled: {enable_judge}")
    print(f"Seed: {seed}")
    print(f"Concurrency: {concurrency}")
    print(f"Output: {output_path}")
    print("=" * 70)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    total_examples = 0
    total_turns = 0

    with open(out_file, "w", encoding="utf-8") as f:
        async def run_one(idx: int, item: Dict[str, Any]) -> List[Dict[str, Any]]:
            async with sem:
                print(f"\n[{idx}/{len(items)}] {item['task_id']}")
                try:
                    examples = await collect_one(
                        item, max_steps=max_steps, max_turns=max_turns,
                        enable_judge=enable_judge,
                    )
                except Exception as exc:
                    print(f"  [{idx}/{len(items)}] {item['task_id']}: ERROR: {exc}")
                    return []
                # Synchronous writes (no await) -> safe under single-thread asyncio.
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{idx}/{len(items)}] {item['task_id']}: wrote {len(examples)} steps")
                return examples

        results = await asyncio.gather(
            *(run_one(i, item) for i, item in enumerate(items, 1))
        )

    for examples in results:
        total_examples += len(examples)
        total_turns += max(ex["turn_idx"] for ex in examples) + 1 if examples else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Samples attempted: {num_samples}")
    print(f"Total step examples written: {total_examples}")
    print(f"Output file: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: <repo-root>/data/gsm8k_sft_steps.jsonl).")
    parser.add_argument("--enable_judge", action="store_true",
                        help="Enable LLM-as-judge for metrics 7-10 (extra API calls).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling a reproducible subset of the train split.")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of GSM8K samples to collect in parallel (async).")
    args = parser.parse_args()

    asyncio.run(main(
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        output_path=args.output,
        enable_judge=args.enable_judge,
        seed=args.seed,
        concurrency=args.concurrency,
    ))
