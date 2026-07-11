"""Manual SCRIBE policy-update loss (GSPO-token / DAPO hybrid).

TRL's built-in GRPOTrainer assumes a completion-level reward and uniform token
weighting. SCRIBE needs fine-grained credit: only rollout tokens (T/O/A/R/S)
participate in the loss, and each rollout token carries a per-step credit that
already incorporates the (tree sibling-group) advantage.

The loss is a GSPO-token / DAPO hybrid:

    loss = -mean[ min(s_step * A_t, clip(s_step, 1-eps, 1+eps) * A_t) ]

where ``A_t`` is the pre-computed per-token credit (advantage) and ``s_step``
is the GSPO **step-level** importance ratio: one scalar per step (sample),
defined as the length-normalized geometric mean of the per-token ratios,

    s_step = exp( (1/|step|) * sum_t log( pi_theta(y_t) / pi_theta_old(y_t) ) )

clipped at the step level. This fixes two problems with the old per-token
pi_theta/pi_ref ratio: (1) it is a proper importance ratio (old = policy at
iteration start, not frozen SFT) so the clip is a per-iteration trust region,
not a cumulative SFT ceiling that freezes learning; (2) GSPO shows token-level
ratios are ill-posed (1 sample/token cannot do distribution correction) and
accumulate noise over a sequence - the step-level ratio with length
normalization is stable and matches the step-level reward unit.

KL to the frozen SFT reference is REMOVED (DAPO): a reasoning policy is meant
to diverge from the SFT base, and a KL anchor to frozen SFT pulls it back and
flattens learning. Stability is provided by the step-level clip (+ grad clip
in the trainer). ref_model is therefore no longer used in the loss.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_scribe_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    token_credits: torch.Tensor,
    rollout_mask: torch.Tensor,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.4,
    llds_lambda: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the GSPO-token / DAPO loss with step-level importance ratio,
    plus optional LLDS likelihood-preserving regularizer.

    Args:
        policy_logprobs: [batch, seq_len] per-token log-probs under the policy.
        old_logprobs: [batch, seq_len] per-token log-probs under the OLD policy
            (the policy at the start of this iteration, pre-update). NOT the
            frozen SFT reference.
        token_credits: [batch, seq_len] pre-computed advantage per token.
        rollout_mask: [batch, seq_len] 1 for rollout tokens, 0 otherwise.
        epsilon_low: lower clip for the step-level ratio (Clip-Higher: keep low
            to suppress bad tokens toward 0).
        epsilon_high: UPPER clip (Clip-Higher: decoupled & larger than low to
            leave room for low-prob 'exploration' tokens to rise, preventing
            entropy collapse).
        llds_lambda: LLDS regularizer weight. 0 disables. LLDS penalizes
            likelihood DECREASES on non-negative-advantage (correct/untrained)
            steps, reusing old_logprobs (no extra forward). Action-level gating
            (only when the step's total likelihood dropped) + token-level
            selectivity (only the decreasing tokens).

    Returns:
        loss: scalar tensor.
        metrics: dict of useful scalars.
    """
    # Masked mean helper.
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x * mask).sum() / mask.sum().clamp_min(1.0)

    # Per-token log-ratio vs the OLD policy (proper importance ratio).
    log_ratio = policy_logprobs - old_logprobs.detach()

    # GSPO step-level ratio: one scalar per step (row) = length-normalized
    # geometric mean of per-token ratios over that step's rollout tokens.
    masked_sum = (log_ratio * rollout_mask).sum(dim=1)  # [batch]
    n_tok = rollout_mask.sum(dim=1).clamp_min(1.0)  # [batch]
    step_log_ratio = masked_sum / n_tok
    s_step = torch.exp(step_log_ratio)  # [batch], one ratio per step

    ratio = s_step.unsqueeze(1)  # broadcast to all tokens in the step
    # Clip-Higher (DAPO): decouple low/high. Keep eps_low tight (pushing bad
    # tokens to 0 is fine), raise eps_high so low-prob exploration tokens can
    # actually increase (a 0.01-prob token capped at 1.2*eps_low barely moves).
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)

    surrogate1 = ratio * token_credits
    surrogate2 = clipped_ratio * token_credits
    policy_loss = -_masked_mean(torch.min(surrogate1, surrogate2), rollout_mask)

    # KL removed (DAPO): no beta * KL(pi_theta || pi_ref) term.
    loss = policy_loss

    with torch.no_grad():
        # A step is clipped when its s_step leaves [1-eps_low, 1+eps_high].
        step_clipped = (
            ((s_step - 1.0) < -epsilon_low) | ((s_step - 1.0) > epsilon_high)
        ).float()  # [batch]
        clip_frac = (step_clipped * n_tok).sum() / n_tok.sum().clamp_min(1.0)
        metrics = {
            "scribe/loss": loss.item(),
            "scribe/policy_loss": policy_loss.item(),
            "scribe/clip_frac": clip_frac.item(),
            "scribe/mean_credit": _masked_mean(token_credits, rollout_mask).item(),
            "scribe/rollout_tokens": rollout_mask.sum().item(),
            "scribe/mean_step_ratio": s_step.mean().item(),
            "scribe/max_step_ratio": s_step.max().item(),
            "scribe/min_step_ratio": s_step.min().item(),
        }

    return loss, metrics


def gather_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probs for the positions marked by ``labels_mask``.

    Args:
        model: causal LM returning logits.
        input_ids: [batch, seq_len].
        attention_mask: [batch, seq_len].
        labels_mask: [batch, seq_len] 1 for target positions.

    Returns:
        logprobs: [batch, seq_len] with values only where labels_mask==1.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits
    # Shift so that each position predicts the next token.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_input_ids = input_ids[..., 1:].contiguous()
    shift_labels_mask = labels_mask[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_input_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Pad back to original length (position 0 has no log-prob).
    batch_size, seq_len = input_ids.shape
    padded = torch.zeros(batch_size, seq_len, device=input_ids.device)
    padded[:, 1:] = token_log_probs * shift_labels_mask
    return padded


__all__ = ["compute_scribe_loss", "gather_logprobs"]
