"""Manual GRPO loss for SCRIBE with token-level rollout mask.

TRL's built-in GRPOTrainer assumes a completion-level reward and uniform token
weighting. SCRIBE needs fine-grained credit: only rollout tokens (T/O/A/R/S)
participate in the loss, and each rollout token carries a per-step credit that
already incorporates the trajectory advantage.

The loss implemented here is the standard GRPO objective:

    loss = -mean[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ] + β * KL

where ``A`` is the pre-computed token credit (advantage).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_grpo_loss(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    token_credits: torch.Tensor,
    rollout_mask: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute GRPO loss with token-level credit mask.

    Args:
        policy_logprobs: [batch, seq_len] per-token log-probs under the policy.
        reference_logprobs: [batch, seq_len] per-token log-probs under ref.
        token_credits: [batch, seq_len] pre-computed advantage per token.
        rollout_mask: [batch, seq_len] 1 for rollout tokens, 0 otherwise.
        epsilon: PPO/GRPO clipping parameter.
        beta: KL penalty coefficient.

    Returns:
        loss: scalar tensor.
        metrics: dict of useful scalars (loss, policy_loss, kl, clip_frac, ...).
    """
    # Masked mean helper.
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x * mask).sum() / mask.sum().clamp_min(1.0)

    ratio = torch.exp(policy_logprobs - reference_logprobs.detach())
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    surrogate1 = ratio * token_credits
    surrogate2 = clipped_ratio * token_credits
    policy_loss = -_masked_mean(torch.min(surrogate1, surrogate2), rollout_mask)

    # Per-token KL using the standard log-ratio estimator.
    # KL(π_θ || π_ref) ≈ exp(log_ratio) - log_ratio - 1
    log_ratio = policy_logprobs - reference_logprobs.detach()
    kl_per_token = torch.exp(log_ratio) - log_ratio - 1.0
    kl_loss = _masked_mean(kl_per_token, rollout_mask)

    loss = policy_loss + beta * kl_loss

    with torch.no_grad():
        clip_frac = _masked_mean(
            ((ratio - 1.0).abs() > epsilon).float(), rollout_mask
        )
        # Per-token drift peaks: a few extreme-ratio tokens can detonate k3's
        # exp() while the mean KL looks calm. Surface the worst offender so the
        # trainer can spot impending detonation (max_log_ratio / kl_max).
        mask_bool = rollout_mask.bool()
        masked_lr = log_ratio[mask_bool]
        masked_kl = kl_per_token[mask_bool]
        metrics = {
            "grpo/loss": loss.item(),
            "grpo/policy_loss": policy_loss.item(),
            "grpo/kl": kl_loss.item(),
            "grpo/clip_frac": clip_frac.item(),
            "grpo/mean_credit": _masked_mean(token_credits, rollout_mask).item(),
            "grpo/rollout_tokens": rollout_mask.sum().item(),
            "grpo/max_log_ratio": masked_lr.abs().max().item() if masked_lr.numel() else 0.0,
            "grpo/kl_max": masked_kl.max().item() if masked_kl.numel() else 0.0,
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


__all__ = ["compute_grpo_loss", "gather_logprobs"]
