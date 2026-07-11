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

from .parsers import ScribeBlockType, render_scribe_blocks
from .step_expander import expand_turn
from .turn_record import Step, TurnRecord


def compute_trajectory_reward(
    turns: List[TurnRecord], decay: float = 0.8
) -> float:
    """README Third: mean turn reward multiplied by a length decay.

    Using the mean turn reward (instead of only the final turn reward) ensures
    that poor behavior in earlier turns is directly penalized. A trajectory
    that wastes early turns with malformed/repeated tool calls receives a lower
    reward even if the final turn eventually succeeds. The length decay still
    rewards solving in fewer turns.
    """
    if not turns:
        return 0.0
    mean_turn_reward = sum(turn.reward for turn in turns) / len(turns)
    return float(mean_turn_reward * (decay ** len(turns)))


def compute_group_advantages(
    rewards: List[float], eps: float = 1e-8
) -> List[float]:
    """README Fourth: (reward - group_mean) / group_std."""
    arr = np.array(rewards, dtype=np.float32)
    mean = arr.mean()
    # Use sample std (ddof=1) so small groups don't get artificially small
    # variance and explode the advantage magnitudes.
    std = arr.std(ddof=1)
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

    ``step_credit`` is the PER-TOKEN advantage for this step (uniform across
    the step's rollout tokens, DAPO-pure). It is assigned directly to each
    rollout token; the loss's masked_mean (1/Σ|o_i|) does the normalization.
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
        # DAPO-pure: step_credit IS the per-token advantage for this step
        # (uniform across the step's rollout tokens). Do NOT divide by n_tokens:
        # the loss's masked_mean (1/Σ|o_i|) is the sole normalization. Dividing
        # here would double-normalize -- shrinking the signal ~1/n_tokens AND
        # cancelling the step weighting in the node's total gradient (the
        # step_weight would affect only direction, not magnitude).
        for tid in token_ids:
            credits[tid] = step_credit
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


def _left_truncate_sample(
    sample: Dict[str, Any],
    max_seq_length: int,
) -> Dict[str, Any]:
    """Trim a sample from the left so it fits ``max_seq_length``.

    We keep the rightmost tokens (which include the model-generated completion)
    so the rollout tokens and their credits are preserved. Prompt-only tokens
    on the left are dropped first. This lets long/loopy trajectories still be
    trained on instead of skipped.
    """
    seq_len = len(sample["input_ids"])
    if seq_len <= max_seq_length:
        return sample
    keep = max_seq_length
    start = seq_len - keep
    return {
        "prompt": sample["prompt"],
        "completion": sample["completion"],
        "input_ids": sample["input_ids"][start:],
        "attention_mask": sample["attention_mask"][start:],
        "token_credits": sample["token_credits"][start:],
        "rollout_mask": sample["rollout_mask"][start:],
    }


_SUBMIT_STEP_WEIGHT = 0.35
_SUMMARY_STEP_WEIGHT = 0.35
_ORDINARY_STEP_POOL = 0.30


def _has_reflect_then_summary(blocks) -> bool:
    """True if a REFLECT block is immediately followed by a TURN_SUMMARY block.

    The SCRIBE final non-tool step must end with R then S (README metric 4b:
    reflect is second-to-last, turn_summary is last). "Summary step" credit
    requires BOTH blocks present AND consecutive - a step with only R, only S,
    or non-adjacent R/S is not a valid summary step (its 0.35 is dropped, see
    _step_weights).
    """
    for j in range(len(blocks) - 1):
        if (
            blocks[j].type == ScribeBlockType.REFLECT
            and blocks[j + 1].type == ScribeBlockType.TURN_SUMMARY
        ):
            return True
    return False


def _step_weights(steps: List[Step]) -> List[float]:
    """Per-step credit weights for one turn (one tree node).

    Mapping (README "key steps"): the turn's signal concentrates on the
    submit step (carries metric 1, the w1=0.60 dominant term) and the summary
    step (carries metrics 6-13). Other tool-call steps carry only the light
    anti-hacking metrics 14-15 and are shared infrastructure across sibling
    branches, so they get a small shared pool.

      - submit step   = last step whose output has a submit tool call  -> 0.35
      - summary step  = last non-tool step with R/S                    -> 0.35
      - ordinary steps (the rest) share 0.30 (each 0.30 / k)

    A MISSING special step's 0.35 is DROPPED, never redistributed. A degenerate
    turn with no submit/summary gets only the 0.30 ordinary pool, so its
    (usually negative) advantage spreads thinly across reasoning tokens. This
    avoids hammering reasoning tokens for what is typically a format failure
    (format is SFT's job; metric 4 already flags it in turn_reward) rather than
    a reasoning failure - the same spirit as LLD's "don't penalize correct
    actions for the wrong reason". Total weight is in (0, 1.0].
    """
    n = len(steps)
    submit_idx = None
    summary_idx = None
    for i, s in enumerate(steps):
        is_tool = any(b.type == ScribeBlockType.TOOL_CALL for b in s.output)
        if is_tool:
            for b in s.output:
                if b.type != ScribeBlockType.TOOL_CALL:
                    continue
                p = getattr(b, "parsed", None)
                if isinstance(p, dict) and p.get("name") == "submit":
                    submit_idx = i  # last submit step (it sets the final answer)
                    break
        else:
            # Non-tool step: it is the summary step only if it carries a proper
            # R-then-S tail (both present AND consecutive). R alone / S alone /
            # non-adjacent R+S does NOT count -> that 0.35 is dropped.
            if _has_reflect_then_summary(s.output):
                summary_idx = i  # last non-tool step with R immediately then S
    weights = [0.0] * n
    ordinary = [i for i in range(n) if i != submit_idx and i != summary_idx]
    if submit_idx is not None:
        weights[submit_idx] = _SUBMIT_STEP_WEIGHT
    if summary_idx is not None:
        weights[summary_idx] = _SUMMARY_STEP_WEIGHT
    if ordinary:
        ow = _ORDINARY_STEP_POOL / len(ordinary)
        for i in ordinary:
            weights[i] = ow
    return weights


def build_tree_node_samples(
    turn: TurnRecord,
    advantage: float,
    tokenizer,
    max_seq_length: int = 4096,
) -> List[Dict[str, Any]]:
    """Expand ONE tree node (a single turn) into step-level GRPO samples with
    step-role-weighted credit.

    Replaces the trajectory-level equal-split ``build_training_samples`` for the
    n-ary tree rollout. ``advantage`` is the node's SIBLING-group advantage
    (turn_reward vs the n siblings sampled from the same parent state) - it is
    NOT divided by num_turns: there is no trajectory, each node is an
    independent short-horizon GRPO problem. Within the node, the per-token
    advantage for each step = advantage * step_role_weight (submit/summary/
    ordinary), uniform across the step's tokens (DAPO-pure: no ÷n_tokens; the
    loss's masked_mean is the sole normalization).
    """
    steps = expand_turn(turn)
    if not steps:
        return []
    weights = _step_weights(steps)
    samples: List[Dict[str, Any]] = []
    for step, w in zip(steps, weights):
        step_credit = advantage * w
        sample = _build_step_training_sample(step, step_credit, tokenizer)
        if sample:
            samples.append(_left_truncate_sample(sample, max_seq_length))
    return samples


def build_training_samples(
    turns: List[TurnRecord],
    trajectory_advantage: float,
    tokenizer,
    max_seq_length: int = 4096,
) -> List[Dict[str, Any]]:
    """Expand a trajectory into step-level GRPO training samples.

    The trajectory advantage is divided evenly across turns, then steps, then
    rollout tokens. Each sample corresponds to one LLM call and uses the exact
    prompt messages seen during rollout, so the chat-template boundaries match.

    Samples longer than ``max_seq_length`` are left-truncated (oldest prompt
    tokens dropped) rather than skipped, so low-reward long/loopy trajectories
    still contribute to the GRPO update.
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
                samples.append(_left_truncate_sample(sample, max_seq_length))

    return samples


__all__ = [
    "compute_trajectory_reward",
    "compute_group_advantages",
    "build_training_samples",
    "build_tree_node_samples",
]
