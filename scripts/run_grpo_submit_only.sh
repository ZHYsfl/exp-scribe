#!/bin/bash
# Run submit-only online GRPO.
#
# Context-budget match with scripts/start_vllm_submit_only.sh:
# vLLM is started with --max-model-len 16384, so we keep --max_tokens at 768
# to leave headroom for 3 turns * 5 steps of accumulated history.

set -e

cd "$(dirname "$0")/.."

python scripts/train_grpo_online_submit_only.py \
  --base_model /root/autodl-tmp/qwen2.5-1.5b-instruct \
  --sft_lora_path /root/autodl-tmp/outputs/qwen2.5-1.5b-sft-gsm8k-submit-only-100/final_lora \
  --output_dir outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100 \
  --data_dir data/gsm8k --split train --num_iterations 50 --batch_size 4 --group_size 4 \
  --num_inner_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 --kl_coef 0.04 --clip_epsilon 0.2 --decay 0.8 \
  --max_steps_per_turn 5 --max_turns 3 --reward_threshold 1.0 \
  --max_tokens 768 --max_seq_length 8192 --max_concurrent 1 --save_steps 1
