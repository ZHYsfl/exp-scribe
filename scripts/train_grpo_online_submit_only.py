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
        --group_size 6 \
        --num_inner_epochs 1
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
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
from Scribe.scribe_gym.system_prompts import SUBMIT_ONLY_SYSTEM_PROMPT

sys.path.insert(0, str(REPO_ROOT / "data"))
from gsm8k_submit_only_env_factory import make_gsm8k_submit_only_env  # type: ignore
from gsm8k_loader import load_gsm8k  # type: ignore

SYSTEM_PROMPT = SUBMIT_ONLY_SYSTEM_PROMPT




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
        default="outputs/scribe_sft_submit_only/final_lora",
        help="Initial LoRA adapter from SFT",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/scribe_grpo_submit_only",
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
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (bounds per-step policy drift; "
             "0 disables). Standard RL safety - without it a single high-"
             "advantage trajectory can shove the policy far from the ref and "
             "detonate the k3 KL estimator (cf. iter-15 spike).",
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
        default=8,
        help="Max concurrent vLLM rollouts. Plain-text tool parsing (submit-only, "
             "no --tool-call-parser) avoids the Hermes parser concurrency bug, so "
             "this can be >1; lower if you still see 'Already borrowed' errors.",
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
    # ---- LR schedule: linear warmup then cosine anneal ----
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of --num_iterations used for linear LR warmup (0 disables warmup).",
    )
    parser.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.1,
        help="Minimum LR as a fraction of --learning_rate (cosine anneal floor).",
    )
    # ---- LLM-as-judge for metrics 7-10 ----
    parser.add_argument(
        "--enable_judge",
        action="store_true",
        help="Enable the LLM-as-judge for metrics 7-10. The judge endpoint is "
             "resolved from Scribe/.env (LLM_MODEL/LLM_BASE_URL/LLM_API_KEY, "
             "e.g. deepseek-chat); the --judge_* flags below are optional "
             "overrides.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Override the judge model (else LLM_MODEL from Scribe/.env).",
    )
    parser.add_argument(
        "--judge_base_url",
        type=str,
        default=None,
        help="Override the judge base URL (else LLM_BASE_URL from Scribe/.env).",
    )
    parser.add_argument(
        "--judge_api_key",
        type=str,
        default=None,
        help="Override the judge API key (else LLM_API_KEY from Scribe/.env).",
    )
    parser.add_argument(
        "--judge_max_concurrent",
        type=int,
        default=5,
        help="Max concurrent judge calls (per process) when --enable_judge is set.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lr_lambda(num_iterations: int, warmup_ratio: float, lr_min_ratio: float):
    """Linear warmup then cosine anneal -> LR multiplier (relative to peak LR).

    warmup: linear from 1/warmup_steps up to 1.0 across the first warmup_steps
            iterations (keeps the very first update non-zero).
    anneal: cosine from 1.0 down to lr_min_ratio across the remaining iters.
    Returns (lr_lambda, warmup_steps); lr_lambda is a function of the scheduler
    step (the iteration index).
    """
    warmup_steps = max(0, round(warmup_ratio * num_iterations))
    min_ratio = max(0.0, min(1.0, lr_min_ratio))
    # Anneal spans iterations [warmup_steps, num_iterations-1]; the denominator
    # is (last_index - warmup) so the final iteration maps to progress=1.0
    # (i.e. LR reaches min_ratio exactly on the last step).
    anneal_steps = max(1, num_iterations - warmup_steps - 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / anneal_steps
        progress = min(1.0, max(0.0, progress))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return lr_lambda, warmup_steps


def aggregate_metric_vectors(
    trajectories: List[List[TurnRecord]],
) -> Dict[str, Any]:
    """Mean of each of the 16 turn metrics across all turns this iteration.

    Each TurnRecord carries a reward_breakdown (TurnRewardBreakdown). Averaging
    metric_1..metric_16 over all turns makes per-metric behavior visible
    alongside the scalar mean reward - e.g. metric_16 rising = less cross-turn
    resubmitting, metric_4 rising = cleaner format. metric_1 > 0 is a proxy for
    "answer correct" (correctness is a binary gate inside metric_1).
    """
    n_metrics = 16
    sums = [0.0] * n_metrics
    n_turns = 0
    n_answer_correct = 0
    n_judge_used = 0
    for turns in trajectories:
        for t in turns:
            bd = t.reward_breakdown
            if bd is None:
                continue
            n_turns += 1
            vals = bd.metrics
            for i in range(min(n_metrics, len(vals))):
                sums[i] += vals[i]
            if bd.metric_1 > 0.0:
                n_answer_correct += 1
            if getattr(bd, "judge_used", False):
                n_judge_used += 1
    means = [s / n_turns for s in sums] if n_turns else [0.0] * n_metrics
    return {
        "metric_means": {f"m{i + 1}": means[i] for i in range(n_metrics)},
        "n_turns": n_turns,
        "answer_correct_turn_frac": (n_answer_correct / n_turns) if n_turns else 0.0,
        "judge_used_turn_frac": (n_judge_used / n_turns) if n_turns else 0.0,
    }


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
    env = make_gsm8k_submit_only_env(
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
        enable_judge=args.enable_judge,
    )
    runner = ScribeRunner(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        done_mode="threshold",
        reward_threshold=args.reward_threshold,
        max_turns=args.max_turns,
        tokenizer=tokenizer,
    )
    try:
        await runner.run()
    finally:
        # Close the per-agent judge HTTP client so its httpx connection pool
        # doesn't leak "Event loop is closed" cleanup tasks across iterations.
        await agent.aclose()
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
    # Drift peaks are global maxima, not means - track the worst token across
    # all batches so a detonation isn't diluted by averaging.
    max_log_ratio_peak = 0.0
    kl_peak = 0.0

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
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad()

        for k, v in metrics.items():
            total_metrics[k] += v
        max_log_ratio_peak = max(
            max_log_ratio_peak, metrics.get("grpo/max_log_ratio", 0.0)
        )
        kl_peak = max(kl_peak, metrics.get("grpo/kl_max", 0.0))
        n_batches += 1

    if n_batches == 0:
        return {}
    result = {k: v / n_batches for k, v in total_metrics.items()}
    # Overwrite the two drift peaks with global maxima (not batch-averaged).
    result["grpo/max_log_ratio"] = max_log_ratio_peak
    result["grpo/kl_max"] = kl_peak
    return result


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # LLM-as-judge (metrics 7-10): the judge is just another LLM call, so it
    # uses the shared Scribe/.env config (LLM_MODEL/LLM_BASE_URL/LLM_API_KEY,
    # e.g. deepseek-chat). GymBackedAgent resolves JUDGE_* -> LLM_* -> actor
    # config; we load .env and only set JUDGE_* for explicit --judge_* overrides.
    if args.enable_judge:
        from dotenv import load_dotenv
        env_path = REPO_ROOT / "Scribe" / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=True)
        # Optional CLI overrides take precedence over .env.
        if args.judge_model:
            os.environ["JUDGE_MODEL"] = args.judge_model
        if args.judge_base_url:
            os.environ["JUDGE_BASE_URL"] = args.judge_base_url
        if args.judge_api_key:
            os.environ["JUDGE_API_KEY"] = args.judge_api_key
        os.environ["JUDGE_MAX_CONCURRENT"] = str(args.judge_max_concurrent)
        resolved_model = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL")
        resolved_base = os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL")
        # Resolve the key with the same JUDGE_* -> LLM_* fallback the agent uses
        # (openai_bridge.py). A missing key would otherwise silently fall back to
        # the dummy actor config ("") -> API 401 -> metrics 7-10's try/except
        # degrades to 0.5 defaults, so the judge runs but produces no signal.
        resolved_key = os.getenv("JUDGE_API_KEY") or os.getenv("LLM_API_KEY")
        if not resolved_model or not resolved_base or not resolved_key:
            raise SystemExit(
                "--enable_judge set but judge endpoint incomplete: set "
                "LLM_MODEL/LLM_BASE_URL/LLM_API_KEY in Scribe/.env (or pass "
                "--judge_model/--judge_base_url/--judge_api_key)."
            )
        print(
            f"Judge enabled: model={resolved_model} @ {resolved_base} "
            f"key=({'set' if resolved_key else 'MISSING'}) "
            f"(max_concurrent={args.judge_max_concurrent})"
        )

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

    # LR schedule: linear warmup then cosine anneal, driven by the iteration
    # index. LambdaLR sets the initial lr to base_lr * lr_lambda(0), so the
    # first iteration already uses the warmup starting value.
    lr_lambda, warmup_steps = build_lr_lambda(
        args.num_iterations, args.warmup_ratio, args.lr_min_ratio
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(
        f"LR schedule: peak={args.learning_rate} warmup_steps={warmup_steps} "
        f"min_ratio={args.lr_min_ratio} (cosine anneal over {args.num_iterations} iters)"
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

    # Per-iteration metrics log: each line is one iteration's record (lr, mean
    # reward, the full 16-metric mean vector, GRPO loss metrics, ...). Truncate
    # at start so a re-run starts clean.
    metrics_log_path = output_dir / "metrics_log.jsonl"
    metrics_log_path.write_text("")

    global_step = 0
    # One persistent event loop for the whole run. Using asyncio.run() per
    # iteration creates+closes a fresh loop each iter, but the shared VLLMBackend
    # (an AsyncOpenAI/httpx client) keeps connection-pool tasks bound to the old
    # loop; when a later iteration's loop schedules them they raise
    # "RuntimeError: Event loop is closed" (and so do the per-trajectory judge
    # clients). A single loop for all iters avoids the cross-loop leak.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
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
        trajectories = loop.run_until_complete(
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

        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
        std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
        print(
            f"Collected {len(all_samples)} samples; "
            f"mean reward = {mean_reward:.3f}, std = {std_reward:.3f}"
        )

        # Per-metric behavior vector (16 metrics) averaged over all turns this
        # iteration - surfaces what the model is actually doing, not just the
        # scalar reward.
        agg = aggregate_metric_vectors(trajectories)
        mm = agg["metric_means"]
        print(
            "  metrics mean: "
            + " ".join(f"m{i}={mm[f'm{i}']:.3f}" for i in range(1, 17))
            + f" | correct_turn_frac={agg['answer_correct_turn_frac']:.3f}"
            + f" | judge_used={agg['judge_used_turn_frac']:.3f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        log_record: Dict[str, Any] = {
            "iter": iteration,
            "lr": current_lr,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_samples": len(all_samples),
            "n_trajectories": len(trajectories),
            **agg,
            "grpo": {},
        }

        if not all_samples:
            print("No valid samples, skipping update.")
            with open(metrics_log_path, "a") as f:
                f.write(json.dumps(log_record) + "\n")
            scheduler.step()
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
        log_record["grpo"] = metrics

        # 5. Sync updated LoRA to vLLM.
        if (iteration + 1) % args.save_steps == 0 or iteration == args.num_iterations - 1:
            save_adapter(policy, output_dir, iteration)
            latest = output_dir / f"iter_{iteration}"
            backend.update_weights(str(latest))
            print(f"Synced vLLM to {latest}")

        with open(metrics_log_path, "a") as f:
            f.write(json.dumps(log_record) + "\n")
        # Advance the LR schedule once per iteration (warmup -> cosine anneal).
        scheduler.step()

    # Final save.
    final_path = output_dir / "final_lora"
    final_path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final LoRA to {final_path}")
    loop.close()


if __name__ == "__main__":
    main()
