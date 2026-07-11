#!/bin/bash
# Run submit-only online RL (tree rollout + GSPO step-ratio).
#
# Context-budget match with scripts/start_vllm_submit_only.sh:
# vLLM is started with --max-model-len 16384, so we keep --max_tokens at 768
# to leave headroom for 3 turns * 5 steps of accumulated history.
#
# LR schedule: linear warmup (10% of iters) then cosine anneal down to 10% of
# peak. Tree rollout: branch_n 6 (GRPO group size), branch_dropout 0.5.
#
# LLM-as-judge (metrics 7-10): --enable_judge resolves the judge endpoint from
# Scribe/.env (LLM_MODEL/LLM_BASE_URL/LLM_API_KEY, e.g. deepseek-chat). No
# separate judge server is needed. Drop --enable_judge if you want metrics 7-10
# to fall back to neutral 0.5 defaults.

set -e

cd "$(dirname "$0")/.."

/root/.venv/bin/python scripts/train_grpo_online_submit_only.py \
  --base_model /root/autodl-tmp/qwen2.5-1.5b-instruct \
  --sft_lora_path /root/autodl-tmp/outputs/qwen2.5-1.5b-sft-gsm8k-submit-only-100/final_lora \
  --output_dir outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100 \
  --data_dir data/gsm8k --split train --num_iterations 50 --batch_size 4 \
  --branch_n 6 --branch_dropout 0.5 \
  --num_inner_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 --warmup_ratio 0.1 --lr_min_ratio 0.1 \
  --clip_epsilon 0.2 --max_grad_norm 1.0 \
  --max_steps_per_turn 5 --max_turns 3 --reward_threshold 1.0 \
  --max_tokens 768 --max_seq_length 16384 --max_concurrent 8 --save_steps 1 \
  --enable_judge --judge_max_concurrent 5
