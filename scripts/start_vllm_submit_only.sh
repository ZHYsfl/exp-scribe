#!/bin/bash
# Start vLLM for the submit-only online GRPO ablation.
#
# Context-length budget reasoning (submit-only, no bash/python):
#   system prompt ~741 tok + kickoff ~41 tok
#   each additional step adds ~output_len + tool_response ~29 + feedback ~62
#   With --max_tokens 768 that's ~859 tok/step.
#   Worst case 3 turns * 5 steps = 15 steps => input ~13k + output 768.
#   We set --max-model-len 16384 to stay well above that.
#
# Key points:
# - VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 is required for /v1/load_lora_adapter.
# - --max-num-seqs 2 keeps the KV-cache small because GRPO uses max_concurrent=1.
# - --gpu-memory-utilization 0.4 leaves room for the GRPO training process.
# - The initial LoRA module is the latest SFT adapter; GRPO will call
#   /v1/load_lora_adapter at runtime to swap in new checkpoints.

set -e

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

/root/.venv/bin/vllm serve /root/autodl-tmp/qwen2.5-1.5b-instruct \
  --enable-lora \
  --lora-modules scribe_adapter=/root/autodl-tmp/outputs/qwen2.5-1.5b-sft-gsm8k-submit-only-100/final_lora \
  --gpu-memory-utilization 0.4 \
  --max-model-len 16384 \
  --max-num-seqs 2 \
  --port 8000
