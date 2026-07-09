"""Diagnose reference-model initialization for GRPO.

Loads the SFT LoRA checkpoint the same way train_grpo_online_submit_only.py does,
then creates the reference model via copy.deepcopy() and checks:

1. Whether policy and reference share any LoRA weight tensors (identity check).
2. Whether their LoRA weight values are exactly equal.
3. Whether forward-pass log-probs differ on a sample prompt.

If (1) or (2) is true, the reference model is not an independent frozen snapshot,
so the KL penalty is effectively disabled.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_MODEL = REPO_ROOT / "qwen2.5-1.5b-instruct"
SFT_LORA = REPO_ROOT / "outputs" / "qwen2.5-1.5b-sft-gsm8k-submit-only-100" / "final_lora"

DEVICE = "cpu"  # stay off the GPU that vLLM is using
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16

SAMPLE_PROMPT = "Solve this math problem step by step: 2 + 3 * 4 ="


def find_lora_layers(model: PeftModel):
    """Return list of (name, module) for all LoRA layers."""
    layers = []
    for name, module in model.named_modules():
        if "lora_" in name.lower() and hasattr(module, "weight"):
            layers.append((name, module))
    return layers


def main():
    if not SFT_LORA.exists():
        print(f"ERROR: SFT LoRA not found at {SFT_LORA}")
        sys.exit(1)

    print(f"Loading base model from {BASE_MODEL} ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    print(f"Loading SFT LoRA from {SFT_LORA} ...")
    policy = PeftModel.from_pretrained(base, SFT_LORA, is_trainable=True)
    policy.print_trainable_parameters()

    print("\n--- 1. copy.deepcopy() as current training code does ---")
    ref_model = copy.deepcopy(policy)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    lora_layers = find_lora_layers(policy)
    print(f"Found {len(lora_layers)} LoRA parameter modules.")

    shared_tensors = 0
    equal_values = 0
    different_values = 0
    for name, p_mod in lora_layers:
        r_mod = dict(find_lora_layers(ref_model)).get(name)
        if r_mod is None:
            print(f"  MISSING in ref: {name}")
            continue
        pw = p_mod.weight
        rw = r_mod.weight
        if pw is rw:
            shared_tensors += 1
            print(f"  SHARED tensor: {name}")
        elif torch.equal(pw, rw):
            equal_values += 1
        else:
            different_values += 1
            diff = (pw - rw).abs().max().item()
            print(f"  DIFFERENT values: {name}, max abs diff = {diff:.6f}")

    print(f"\nSummary: shared_tensors={shared_tensors}, equal_values={equal_values}, different_values={different_values}")
    if shared_tensors > 0:
        print("!!! CRITICAL: policy and reference share LoRA weight tensors. KL penalty is broken.")
    elif different_values == 0:
        print("!!! WARNING: policy and reference have identical LoRA weights. If policy updates in-place, ref will drift with it.")

    print("\n--- 2. Forward-pass log-prob comparison ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        policy_logits = policy(**inputs).logits
        ref_logits = ref_model(**inputs).logits

    # Compare logits on the last position
    p_logprob = torch.log_softmax(policy_logits[0, -1], dim=-1)
    r_logprob = torch.log_softmax(ref_logits[0, -1], dim=-1)

    mean_abs_diff = (p_logprob - r_logprob).abs().mean().item()
    max_abs_diff = (p_logprob - r_logprob).abs().max().item()
    kl_token = torch.exp(p_logprob - r_logprob) - (p_logprob - r_logprob) - 1.0
    mean_kl = kl_token.mean().item()

    print(f"On sample prompt: '{SAMPLE_PROMPT}'")
    print(f"  mean |policy_logprob - ref_logprob| = {mean_abs_diff:.6f}")
    print(f"  max  |policy_logprob - ref_logprob| = {max_abs_diff:.6f}")
    print(f"  mean per-token KL estimator         = {mean_kl:.6f}")

    if mean_abs_diff < 0.001 and max_abs_diff < 0.01:
        print("!!! CRITICAL: policy and reference produce nearly identical log-probs. KL cannot penalize deviations.")
    else:
        print("OK: policy and reference log-probs differ; KL penalty can in principle work.")

    print("\n--- 3. Proper independent reference model (fresh load) ---")
    base2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    ref_proper = PeftModel.from_pretrained(base2, SFT_LORA, is_trainable=False)
    ref_proper.eval()
    with torch.no_grad():
        ref_proper_logits = ref_proper(**inputs).logits
    rp_logprob = torch.log_softmax(ref_proper_logits[0, -1], dim=-1)
    mean_abs_diff_proper = (p_logprob - rp_logprob).abs().mean().item()
    max_abs_diff_proper = (p_logprob - rp_logprob).abs().max().item()
    print(f"  mean |policy_logprob - proper_ref_logprob| = {mean_abs_diff_proper:.6f}")
    print(f"  max  |policy_logprob - proper_ref_logprob| = {max_abs_diff_proper:.6f}")

    if mean_abs_diff_proper > mean_abs_diff * 10:
        print("!!! copy.deepcopy ref behaves very differently from a freshly-loaded ref. This confirms initialization bug.")


if __name__ == "__main__":
    main()
