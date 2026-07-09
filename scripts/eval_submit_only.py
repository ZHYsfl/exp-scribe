"""Greedy eval for the submit-only SFT checkpoint on GSM8K test.

Loads the latest SFT LoRA into a local vLLM server (started separately with
scripts/start_vllm_submit_only.sh) and reports answer accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scribe.llm_runtime import LLMConfig
from Scribe.llm_runtime.token_counter import DeepSeekTokenCounter
from Scribe.scribe_gym import GymBackedAgent, HistoryManager, ScribeRunner, VLLMBackend
from Scribe.scribe_gym.system_prompts import SUBMIT_ONLY_SYSTEM_PROMPT

sys.path.insert(0, str(REPO_ROOT / "data"))
from gsm8k_loader import load_gsm8k  # type: ignore
from gsm8k_submit_only_env_factory import make_gsm8k_submit_only_env  # type: ignore

SYSTEM_PROMPT = SUBMIT_ONLY_SYSTEM_PROMPT


class GreedyGymBackedAgent(GymBackedAgent):
    """Agent that always calls the backend with greedy decoding."""

    async def _call_llm(self, **kwargs: Any) -> Any:
        kwargs["temperature"] = 0.0
        kwargs["top_p"] = 1.0
        return await super()._call_llm(**kwargs)


async def eval_one(
    item: Dict[str, Any],
    base_url: str = "http://localhost:8000/v1",
    lora_name: str = "scribe_adapter",
    max_steps: int = 5,
    max_turns: int = 3,
) -> Dict[str, Any]:
    cfg = LLMConfig(api_key="vllm", model=lora_name, base_url=base_url)
    backend = VLLMBackend(
        base_url=base_url,
        api_key="vllm",
        lora_name=lora_name,
        plain_text_tools=True,
    )
    env = make_gsm8k_submit_only_env(
        item, workspace_root=Path("/tmp/gsm8k_eval") / item["task_id"], max_steps=max_steps
    )
    counter = DeepSeekTokenCounter()
    hm = HistoryManager(counter, k=2, hard_limit=16384, compression_margin=512)
    agent = GreedyGymBackedAgent(
        config=cfg,
        env=env,
        history_manager=hm,
        llm_backend=backend,
        max_tokens=768,
    )
    runner = ScribeRunner(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        done_mode="threshold",
        reward_threshold=1.0,
        max_turns=max_turns,
    )
    try:
        await runner.run()
    except Exception as exc:
        return {"task_id": item["task_id"], "error": str(exc)}

    # Determine correctness from the last env step.
    last = agent.trajectory[-1] if agent.trajectory else {}
    info = last.get("info") or {}
    answer = info.get("answer")
    right_answer = info.get("right_answer")
    correct = False
    if answer is not None and right_answer is not None:
        correct = answer.strip() == right_answer.strip()
    return {
        "task_id": item["task_id"],
        "correct": correct,
        "answer": answer,
        "right_answer": right_answer,
        "reward": last.get("reward"),
        "terminated": last.get("terminated"),
        "truncated": last.get("truncated"),
    }


async def main(
    num_samples: int = 50,
    base_url: str = "http://localhost:8000/v1",
    lora_name: str = "scribe_adapter",
    max_steps: int = 5,
    max_turns: int = 3,
):
    items = load_gsm8k("test", "main", limit=num_samples)
    print(f"Evaluating {len(items)} GSM8K test samples...")
    results: List[Dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {item['task_id']}")
        res = await eval_one(item, base_url, lora_name, max_steps, max_turns)
        results.append(res)
        status = "✓" if res.get("correct") else "✗"
        ans = res.get("answer")
        ra = res.get("right_answer")
        print(f"  {status} answer={ans!r} right={ra!r}")

    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    print("\n" + "=" * 60)
    print(f"Accuracy: {correct}/{n} = {correct / n * 100:.1f}%")
    out_path = REPO_ROOT / "outputs" / "eval_submit_only_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved per-sample results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--lora_name", default="scribe_adapter")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(
        num_samples=args.num_samples,
        base_url=args.base_url,
        lora_name=args.lora_name,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
    ))
