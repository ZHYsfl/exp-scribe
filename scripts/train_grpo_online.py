"""Online GRPO training for SCRIBE using a local vLLM rollout engine.

This script implements the full online loop described in the README:

    for iteration:
        1. rollout: sample G trajectories per task with vLLM
        2. reward:  compute per-trajectory reward and group advantages
        3. update:  run GRPO update on rollout tokens
        4. sync:    push the updated LoRA weights back to vLLM

The rollout logic reuses the existing SCRIBE runner/env/reward infra; only the
LLM generation backend is swapped for a local vLLM engine.

Example (3090, 0.5B model):
    python scripts/train_grpo_online.py \
        --base_model /path/to/qwen2.5-0.5b-instruct \
        --sft_lora_path outputs/scribe_sft/final_lora \
        --output_dir outputs/scribe_grpo \
        --num_iterations 10 \
        --batch_size 4 \
        --group_size 4 \
        --num_inner_epochs 1
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


# Make project root importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scribe.llm_runtime import LLMConfig
from Scribe.llm_runtime.token_counter import DeepSeekTokenCounter
from Scribe.scribe_gym import (
    DEFAULT_TURN_REWARD_CONFIG,
    GymBackedAgent,
    HistoryManager,
    ScribeRunner,
    VLLMBackend,
)
from Scribe.scribe_gym.grpo_loss import compute_grpo_loss, gather_logprobs
from Scribe.scribe_gym.rl_utils import (
    build_training_samples,
    compute_group_advantages,
    compute_trajectory_reward,
)
from Scribe.scribe_gym.turn_record import TurnRecord

sys.path.insert(0, str(REPO_ROOT / "data"))
from gsm8k_env_factory import make_gsm8k_env  # type: ignore
from gsm8k_loader import load_gsm8k  # type: ignore

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online GRPO for SCRIBE")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/home/zane/exp-scribe/qwen2.5-0.5b-instruct",
        help="Base model path or HF id",
    )
    parser.add_argument(
        "--sft_lora_path",
        type=str,
        default="outputs/scribe_sft/final_lora",
        help="Initial LoRA adapter from SFT",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/scribe_grpo",
        help="Directory for checkpoints and final adapter",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/gsm8k",
        help="Directory containing GSM8K parquet files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="GSM8K split to train on",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Number of online rollout-train-sync iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of distinct tasks per iteration",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=6,
        help="Number of trajectories sampled per task",
    )
    parser.add_argument(
        "--num_inner_epochs",
        type=int,
        default=1,
        help="Gradient epochs over the rollout data per iteration",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Mini-batch size for the GRPO update",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="AdamW learning rate for GRPO",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.06,
        help="KL penalty coefficient beta",
    )
    parser.add_argument(
        "--clip_epsilon",
        type=float,
        default=0.2,
        help="GRPO clipping epsilon",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.8,
        help="Trajectory reward decay factor",
    )
    parser.add_argument(
        "--max_steps_per_turn",
        type=int,
        default=5,
        help="Max steps (LLM calls) per turn",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=3,
        help="Max turns per trajectory",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=1.0,
        help="Threshold for ScribeRunner threshold mode",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI server URL",
    )
    parser.add_argument(
        "--vllm_lora_name",
        type=str,
        default="scribe_adapter",
        help="LoRA module name registered on the vLLM server",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1,
        help="Save a checkpoint every N iterations",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 for training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 for training",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=1,
        help="Max concurrent vLLM rollouts (lower if you see Hermes 'Already borrowed' errors)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1536,
        help="Max tokens per LLM call during rollout",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Max tokenized sequence length for GRPO training samples; longer samples are skipped",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(base_model_path: str, sft_lora_path: str, bf16: bool = False, fp16: bool = False):
    """Load base model + SFT LoRA and a frozen reference model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use explicit dtype flags if provided; otherwise auto-detect.
    dtype = torch.float32
    if fp16:
        dtype = torch.float16
    elif bf16:
        dtype = torch.bfloat16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    policy = PeftModel.from_pretrained(base, sft_lora_path, is_trainable=True)
    policy.print_trainable_parameters()

    # Reference model: same weights, frozen.
    ref_model = copy.deepcopy(policy)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    return policy, ref_model, tokenizer


def save_adapter(model: Any, path: Path, iteration: int) -> None:
    """Save current LoRA weights."""
    out = path / f"iter_{iteration}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    print(f"Saved adapter to {out}")


async def rollout_one(
    backend: VLLMBackend,
    item: Dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
) -> List[TurnRecord]:
    """Run one SCRIBE trajectory for a single task using the vLLM backend."""
    env = make_gsm8k_env(
        item,
        workspace_root=Path(args.output_dir) / "rollout_ws" / item["task_id"],
        max_steps=args.max_steps_per_turn,
    )
    counter = DeepSeekTokenCounter()
    # Compression disabled by default for small-model experiments; vLLM handles
    # its own context budget and the trajectory is short.
    hm = HistoryManager(counter, k=2, hard_limit=None)
    # Dummy LLMConfig: the actual generation happens through the vLLM backend.
    cfg = LLMConfig(api_key="", model=args.vllm_lora_name, base_url=args.vllm_base_url)
    agent = GymBackedAgent(
        config=cfg,
        env=env,
        history_manager=hm,
        llm_backend=backend,
        reward_config=DEFAULT_TURN_REWARD_CONFIG,
        token_counter=counter,
        max_tokens=args.max_tokens,
    )
    runner = ScribeRunner(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        done_mode="threshold",
        reward_threshold=args.reward_threshold,
        max_turns=args.max_turns,
        tokenizer=tokenizer,
    )
    await runner.run()
    env.close()
    return list(hm._turns)


async def collect_group_rollouts(
    backend: VLLMBackend,
    items: List[Dict[str, Any]],
    group_size: int,
    tokenizer: Any,
    args: argparse.Namespace,
    max_concurrent: int = 8,
) -> List[List[TurnRecord]]:
    """Sample ``group_size`` trajectories for each task in ``items``.

    Rollouts are parallelized with a semaphore to bound concurrent load on the
    vLLM server. When the backend uses plain-text tool parsing it avoids the
    Hermes parser concurrency bug; in that case ``max_concurrent`` can be
    raised. If you still see ``Already borrowed`` errors, lower it.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(item: Dict[str, Any]) -> List[TurnRecord]:
        async with semaphore:
            return await rollout_one(backend, item, tokenizer, args)

    tasks = [
        _one(item)
        for _ in range(group_size)
        for item in items
    ]
    return await asyncio.gather(*tasks)


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences to the max length in the batch."""
    max_len = max(len(x["input_ids"]) for x in batch)

    def _pad(name: str, fill_value: Any) -> torch.Tensor:
        out = []
        for x in batch:
            seq = list(x[name])
            seq.extend([fill_value] * (max_len - len(seq)))
            out.append(seq[:max_len])
        return torch.tensor(out)

    return {
        "input_ids": _pad("input_ids", pad_token_id),
        "attention_mask": _pad("attention_mask", 0),
        "token_credits": _pad("token_credits", 0.0),
        "rollout_mask": _pad("rollout_mask", 0),
        "labels_mask": _pad("rollout_mask", 0),
    }


def do_grpo_update(
    policy: Any,
    ref_model: Any,
    optimizer: torch.optim.Optimizer,
    samples: List[Dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Run one epoch of GRPO updates over the sampled trajectories."""
    policy.train()
    ref_model.eval()

    loader = DataLoader(
        samples,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, int(tokenizer.pad_token_id)),
    )

    total_metrics: Dict[str, float] = defaultdict(float)
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(policy.device)
        attention_mask = batch["attention_mask"].to(policy.device)
        token_credits = batch["token_credits"].to(policy.device)
        rollout_mask = batch["rollout_mask"].to(policy.device)

        with torch.no_grad():
            ref_logprobs = gather_logprobs(
                ref_model, input_ids, attention_mask, rollout_mask
            )

        policy_logprobs = gather_logprobs(
            policy, input_ids, attention_mask, rollout_mask
        )

        loss, metrics = compute_grpo_loss(
            policy_logprobs,
            ref_logprobs,
            token_credits,
            rollout_mask,
            epsilon=args.clip_epsilon,
            beta=args.kl_coef,
        )

        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if (n_batches + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in total_metrics.items()}


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args for reproducibility.
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading models...")
    policy, ref_model, tokenizer = load_models(
        args.base_model, args.sft_lora_path, bf16=args.bf16, fp16=args.fp16
    )

    # Enable gradient checkpointing on the policy to fit long-context GRPO
    # updates on a single GPU alongside the vLLM rollout server.
    policy.enable_input_require_grads()
    policy.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Starting vLLM backend...")
    backend = VLLMBackend(
        base_url=args.vllm_base_url,
        lora_name=args.vllm_lora_name,
    )

    train_items = load_gsm8k(
        split=args.split,
        config="main",
        data_dir=args.data_dir,
    )
    print(f"Loaded {len(train_items)} training items")

    global_step = 0
    for iteration in range(args.num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.num_iterations}")
        print(f"{'='*60}")

        # 1. Sample task batch.
        items = random.sample(
            train_items, min(args.batch_size, len(train_items))
        )

        # 2. Rollout group.
        print("Collecting rollouts...")
        trajectories = asyncio.run(
            collect_group_rollouts(
                backend, items, args.group_size, tokenizer, args,
                max_concurrent=args.max_concurrent,
            )
        )

        # 3. Compute rewards and advantages per task group.
        groups: Dict[str, List[List[TurnRecord]]] = defaultdict(list)
        for item, turns in zip(
            [it for it in items for _ in range(args.group_size)], trajectories
        ):
            groups[item["task_id"]].append(turns)

        all_samples: List[Dict[str, Any]] = []
        all_rewards: List[float] = []
        for task_id, task_turns in groups.items():
            rewards = [
                compute_trajectory_reward(turns, decay=args.decay)
                for turns in task_turns
            ]
            all_rewards.extend(rewards)
            advantages = compute_group_advantages(rewards)
            for turns, adv in zip(task_turns, advantages):
                samples = build_training_samples(
                    turns, adv, tokenizer, max_seq_length=args.max_seq_length
                )
                all_samples.extend(samples)

        print(
            f"Collected {len(all_samples)} samples; "
            f"mean reward = {np.mean(all_rewards):.3f}, "
            f"std = {np.std(all_rewards):.3f}"
        )

        if not all_samples:
            print("No valid samples, skipping update.")
            continue

        # 4. GRPO update.
        print("Running GRPO update...")
        # Free rollout-side CUDA cache before the backward pass; vLLM and the
        # rollout graph can leave fragmented allocations that compete with the
        # training activation memory.
        torch.cuda.empty_cache()
        for epoch in range(args.num_inner_epochs):
            metrics = do_grpo_update(
                policy,
                ref_model,
                optimizer,
                all_samples,
                tokenizer,
                args,
            )
            print(f"  inner epoch {epoch}: {metrics}")
            global_step += 1

        # 5. Sync updated LoRA to vLLM.
        if (iteration + 1) % args.save_steps == 0 or iteration == args.num_iterations - 1:
            save_adapter(policy, output_dir, iteration)
            latest = output_dir / f"iter_{iteration}"
            backend.update_weights(str(latest))
            print(f"Synced vLLM to {latest}")

    # Final save.
    final_path = output_dir / "final_lora"
    final_path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final LoRA to {final_path}")


if __name__ == "__main__":
    main()
