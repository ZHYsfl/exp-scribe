"""RL utilities for SCRIBE: trajectory reward, group advantages, and step-level
training samples for GRPO.

We build training samples at **step granularity**, not at trajectory granularity.
During rollout the LLM is called once per step; each call sees the preceding
messages as prompt and produces the current step's output blocks as completion.
Training on individual steps preserves the exact chat-template format the model
sees at inference time, including all ``<|im_start|>`` / ``<|im_end|>`` boundaries.

Credit assignment follows the README chain:

    trajectory_advantage
        → turn_credit = adv / num_turns
        → step_credit = turn_credit / num_steps_in_turn
        → token_credit = step_credit / num_rollout_tokens_in_step

Only the model-generated rollout tokens (T/O/A/R/S) inside the current step get
non-zero credit. AR/prompt/system/feedback/tool tokens are masked out.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .parsers import render_scribe_blocks
from .step_expander import expand_turn
from .turn_record import Step, TurnRecord


def compute_trajectory_reward(
    turns: List[TurnRecord], decay: float = 0.8
) -> float:
    """README Third: final_turn_reward * decay ** len(turns_used)."""
    if not turns:
        return 0.0
    final_turn_reward = turns[-1].reward
    return float(final_turn_reward * (decay ** len(turns)))


def compute_group_advantages(
    rewards: List[float], eps: float = 1e-8
) -> List[float]:
    """README Fourth: (reward - group_mean) / group_std."""
    arr = np.array(rewards, dtype=np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < eps or len(arr) < 2:
        return [0.0 for _ in rewards]
    return ((arr - mean) / (std + eps)).tolist()


def _step_rollout_text(step: Step) -> str:
    """Completion text for one step = its rollout blocks only (AR excluded)."""
    rollout_blocks = [b for b in step.output if b.type.value != "TOOL_RESPONSE"]
    return render_scribe_blocks(rollout_blocks)


def _build_step_training_sample(
    step: Step,
    step_credit: float,
    tokenizer,
) -> Dict[str, Any]:
    """Build one training sample from a single LLM step.

    ``step.input`` is the exact message list the model saw at inference time.
    ``step.output`` is the raw blocks it generated. We render these as a
    two-message conversation (history + assistant message) so the chat template
    produces the same tokens as during rollout.
    """
    # ``step.input`` is the exact message list the model saw before this step.
    prompt_messages = [dict(m) for m in step.input]
    completion_text = _step_rollout_text(step)

    if not completion_text:
        return {}

    # Render full conversation with the model's chat template.
    messages = prompt_messages + [
        {"role": "assistant", "content": completion_text}
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    encoded = tokenizer(
        full_text,
        return_tensors=None,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Compute rollout token mask inside the completion.
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False
    )["input_ids"]
    n_prompt = len(prompt_ids)

    completion_encoded = tokenizer(
        completion_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = completion_encoded.get("offset_mapping", [])
    n_completion_tokens = len(offsets)

    credits = [0.0] * n_completion_tokens
    mask = [0] * n_completion_tokens

    # Find each rollout block's character span and mark corresponding tokens.
    cursor = 0
    for block in step.output:
        if block.type.value == "TOOL_RESPONSE":
            continue
        block_text = block.content
        start = completion_text.find(block_text, cursor)
        if start == -1:
            continue
        end = start + len(block_text)
        cursor = end

        token_ids = [
            i
            for i, (tok_start, tok_end) in enumerate(offsets)
            if tok_start >= start and tok_end <= end
        ]
        n_tokens = max(1, len(token_ids))
        token_credit = step_credit / n_tokens
        for tid in token_ids:
            credits[tid] = token_credit
            mask[tid] = 1

    # Align to full sequence length (prompt tokens have zero credit/mask).
    full_credits = [0.0] * n_prompt + credits
    full_mask = [0] * n_prompt + mask
    seq_len = len(input_ids)
    if len(full_credits) < seq_len:
        full_credits.extend([0.0] * (seq_len - len(full_credits)))
        full_mask.extend([0] * (seq_len - len(full_mask)))
    full_credits = full_credits[:seq_len]
    full_mask = full_mask[:seq_len]

    return {
        "prompt": prompt_text,
        "completion": completion_text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_credits": full_credits,
        "rollout_mask": full_mask,
    }


def build_training_samples(
    turns: List[TurnRecord],
    trajectory_advantage: float,
    tokenizer,
) -> List[Dict[str, Any]]:
    """Expand a trajectory into step-level GRPO training samples.

    The trajectory advantage is divided evenly across turns, then steps, then
    rollout tokens. Each sample corresponds to one LLM call and uses the exact
    prompt messages seen during rollout, so the chat-template boundaries match.
    """
    if not turns:
        return []

    n_turns = len(turns)
    turn_credit = trajectory_advantage / n_turns

    samples: List[Dict[str, Any]] = []
    for turn in turns:
        steps = expand_turn(turn)
        n_steps = max(1, len(steps))
        step_credit = turn_credit / n_steps

        for step in steps:
            sample = _build_step_training_sample(step, step_credit, tokenizer)
            if sample:
                samples.append(sample)

    return samples


__all__ = [
    "compute_trajectory_reward",
    "compute_group_advantages",
    "build_training_samples",
]
