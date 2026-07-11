#!/bin/bash
# Start vLLM for EVALUATION ONLY (no GRPO training running).
#
# Differences vs scripts/start_vllm_submit_only.sh (which is tuned for training):
#   - --gpu-memory-utilization 0.85  (training script uses 0.4 to leave room for
#     the GRPO training process; eval has nothing else on the GPU, so take more
#     for a bigger KV cache -> higher throughput).
#   - --max-num-seqs 32  (training uses 2 because GRPO uses max_concurrent=1;
#     eval runs many samples concurrently, so allow a real batch).
#   - --max-loras / --max-cpu-loras 8  (cache every iter adapter so switching
#     iters during the sweep never reloads from disk).
#   - Pre-registers ALL GRPO iter LoRA adapters (0,10,20,30,40,47) plus the SFT
#     adapter at startup, so eval can pick any iter by --lora_name without
#     hitting /v1/load_lora_adapter.
#
# Run:  screen -dmS vllm_eval bash scripts/start_vllm_eval.sh
# (or foreground: bash scripts/start_vllm_eval.sh)

set -e

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

ADAPTER_DIR=/root/autodl-tmp/outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100

/root/.venv/bin/vllm serve /root/autodl-tmp/qwen2.5-1.5b-instruct \
  --enable-lora \
  --lora-modules \
    scribe_adapter=/root/autodl-tmp/outputs/qwen2.5-1.5b-sft-gsm8k-submit-only-100/final_lora \
    iter_0=$ADAPTER_DIR/iter_0 \
    iter_10=$ADAPTER_DIR/iter_10 \
    iter_20=$ADAPTER_DIR/iter_20 \
    iter_30=$ADAPTER_DIR/iter_30 \
    iter_40=$ADAPTER_DIR/iter_40 \
    iter_47=$ADAPTER_DIR/iter_47 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-loras 8 \
  --max-cpu-loras 8 \
  --port 8000
