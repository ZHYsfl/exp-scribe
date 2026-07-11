"""SFT training script for SCRIBE using Unsloth.

Reads step-level JSONL produced by `data/collect_sft_rollouts.py` and runs
supervised fine-tuning on (input_messages, output_text) conversation pairs.

Usage:
    python scripts/train_sft.py \
        --data_path data/gsm8k_sft_steps.jsonl \
        --model_name unsloth/Llama-3.2-3B-Instruct \
        --output_dir outputs/scribe_sft \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --lora_r 64

Requirements (add to Scribe/requirements.txt or install manually):
    unsloth
    unsloth-zoo
    trl>=0.12.0
    datasets
    torch

On WSL/Windows Unsloth needs a working CUDA toolchain. If you see build
errors, install the pre-built wheels first:
    pip install unsloth unsloth-zoo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT SCRIBE rollouts with Unsloth")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/gsm8k_sft_steps.jsonl",
        help="Path to the JSONL produced by collect_sft_rollouts.py",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen2.5-0.5b-instruct",
        help="Unsloth-compatible model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/scribe_sft",
        help="Where to save the trained LoRA adapter + checkpoints",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Max sequence length for model + training",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA r",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0, overrides num_train_epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--use_rewards_as_weights",
        action="store_true",
        help="If set, repeat higher-reward examples more often via reward-based sampling",
    )
    return parser.parse_args()


def load_step_examples(data_path: str) -> List[Dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT data not found: {path}")
    examples: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def build_conversation(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert one step example into a chat conversation.

    The input_messages already contain the system/user/assistant/tool context
    seen by the model at inference time. We keep all of them as the prompt and
    append the model's own output_text as the assistant target.
    """
    messages = [dict(m) for m in example.get("input_messages", [])]
    output_text = example.get("output_text", "")
    if output_text:
        messages.append({"role": "assistant", "content": output_text})
    return messages


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    # Lazy import Unsloth so that the script can still be imported/parsed
    # in environments where Unsloth is not installed.
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as exc:
        raise ImportError(
            "Unsloth is required for training. Install with:\n"
            "  pip install unsloth unsloth-zoo trl datasets"
        ) from exc

    # Import TRL after Unsloth so that Unsloth's patches are applied first.
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect float16/bfloat16
        load_in_4bit=True,
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    examples = load_step_examples(args.data_path)
    print(f"Loaded {len(examples)} step examples from {args.data_path}")

    if args.use_rewards_as_weights:
        # Higher reward -> repeat more often. Simple heuristic: weight ∝ reward.
        weights = [max(0.0, ex.get("turn_reward", 0.0)) for ex in examples]
        total = sum(weights) or 1.0
        probs = [w / total for w in weights]
        # Sample with replacement to a fixed-size dataset ~2x original.
        import random
        random.seed(args.seed)
        n_out = len(examples) * 2
        indices = random.choices(range(len(examples)), weights=probs, k=n_out)
        examples = [examples[i] for i in indices]
        print(f"Reward-weighted sampling: expanded to {len(examples)} examples")

    conversations = [build_conversation(ex) for ex in examples]
    dataset = Dataset.from_dict({"messages": conversations})

    # Use the model's chat template to format conversations. Unsloth/TRL SFT
    # expects a formatting function that returns a list of processed strings.
    # We accept both single examples (as used by TRL's internal smoke test) and
    # batched inputs, but always return a list.
    def formatting_func(sample) -> List[str]:
        messages = sample["messages"]
        if isinstance(messages, list) and messages and isinstance(messages[0], list):
            return [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                for msgs in messages
            ]
        return [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        ]

    train_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=train_args,
    )

    trainer.train()

    # Save final adapter + tokenizer
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path / "final_lora")
    tokenizer.save_pretrained(output_path / "final_lora")
    print(f"Saved LoRA adapter to {output_path / 'final_lora'}")


if __name__ == "__main__":
    main()
