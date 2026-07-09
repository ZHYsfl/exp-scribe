#!/bin/bash
# Auto-pipeline for submit-only ablation while user is away.
# 1. Wait for current SFT collection to finish.
# 2. Train SFT.
# 3. Start vLLM, run greedy eval, keep vLLM running for GRPO.
# 4. Start online GRPO.
set -e

cd /root/autodl-tmp

COLLECT_OUTPUT="/tmp/claude-0/-root-autodl-tmp/2d14f88e-e82e-4861-bbc6-d01e3b22563d/tasks/b33igccfh.output"
DATA="data/gsm8k_sft_steps_submit_only.jsonl"
SFT_DIR="outputs/qwen2.5-1.5b-sft-gsm8k-submit-only-100"
GRPO_DIR="outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100"

echo "[pipeline] waiting for SFT collection to finish..."
for i in $(seq 1 360); do
    if [ -f "$COLLECT_OUTPUT" ] && grep -q "SUMMARY" "$COLLECT_OUTPUT"; then
        echo "[pipeline] collection finished."
        break
    fi
    sleep 10
done
if ! grep -q "SUMMARY" "$COLLECT_OUTPUT" 2>/dev/null; then
    echo "[pipeline] ERROR: collection did not finish in time."
    exit 1
fi

if [ ! -s "$DATA" ]; then
    echo "[pipeline] ERROR: $DATA is missing or empty."
    exit 1
fi

wc -l "$DATA"

echo "[pipeline] training SFT..."
rm -rf "$SFT_DIR"
/root/.venv/bin/python scripts/train_sft.py \
  --model_name /root/autodl-tmp/qwen2.5-1.5b-instruct \
  --data_path "$DATA" \
  --output_dir "$SFT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 16384 \
  --lora_r 16 --lora_alpha 16 \
  --learning_rate 2e-4 \
  --warmup_steps 5 \
  --logging_steps 5 \
  --save_steps 50 \
  --seed 42

echo "[pipeline] SFT done. Starting vLLM for eval..."
scripts/start_vllm_submit_only.sh > /tmp/vllm_eval_grpo.log 2>&1 &
VLLM_PID=$!

echo "[pipeline] waiting for vLLM health..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "[pipeline] vLLM ready."
        break
    fi
    sleep 2
done
if ! curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "[pipeline] ERROR: vLLM did not start."
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

echo "[pipeline] running greedy eval..."
/root/.venv/bin/python scripts/eval_submit_only.py --num_samples 50

echo "[pipeline] eval done. Cleaning old GRPO dir and starting GRPO..."
rm -rf "$GRPO_DIR"
scripts/run_grpo_submit_only.sh

echo "[pipeline] GRPO finished."
