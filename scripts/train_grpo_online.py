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
        default=4,
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
        default=0.04,
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
                samples = build_training_samples(turns, adv, tokenizer)
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
