"""Evaluate the SFT LoRA checkpoint alone (no RL) on a small GSM8K slice.

Uses HuggingFace directly so it does not disturb the running vLLM server.
Default is 10 samples on CPU; increase --num_samples at the cost of time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))
from gsm8k_loader import load_gsm8k  # type: ignore

BASE_MODEL = REPO_ROOT / "qwen2.5-1.5b-instruct"
SFT_LORA = REPO_ROOT / "outputs" / "qwen2.5-1.5b-sft-gsm8k-submit-only-100" / "final_lora"

ANSWER_RE = re.compile(r"(\-?\d+(?:\.\d+)?)")
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem and put your final numeric answer inside \\boxed{}."
)


def extract_boxed(text: str) -> str | None:
    """Extract last \\boxed{...} content, fallback to last number."""
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        return boxes[-1].strip()
    nums = ANSWER_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def eval_one(model, tokenizer, item, device):
    question = item["task_description"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    pred = extract_boxed(generated)
    gt = item["right_answer"] or ""
    correct = pred is not None and pred.strip() == gt.strip()
    return {
        "task_id": item["task_id"],
        "correct": correct,
        "predicted": pred,
        "ground_truth": gt,
        "generation": generated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not SFT_LORA.exists():
        print(f"ERROR: SFT LoRA not found at {SFT_LORA}")
        sys.exit(1)

    print(f"Loading base model from {BASE_MODEL} ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
        device_map=args.device,
        trust_remote_code=True,
    )
    print(f"Loading SFT LoRA from {SFT_LORA} ...")
    model = PeftModel.from_pretrained(base, SFT_LORA, is_trainable=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    items = load_gsm8k("test", "main", limit=args.num_samples)
    print(f"Evaluating {len(items)} GSM8K test samples on SFT model...\n")

    results = []
    for i, item in enumerate(items, 1):
        res = eval_one(model, tokenizer, item, args.device)
        results.append(res)
        status = "✓" if res["correct"] else "✗"
        print(f"[{i}/{len(items)}] {status} pred={res['predicted']!r} gt={res['ground_truth']!r} | {res['task_id']}")

    correct = sum(r["correct"] for r in results)
    n = len(results)
    print(f"\nSFT-only accuracy: {correct}/{n} = {correct/n*100:.1f}%")

    out = REPO_ROOT / "outputs" / "diagnostic_sft_only.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
