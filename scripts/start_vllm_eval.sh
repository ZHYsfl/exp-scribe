set -e
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
ADAPTER_DIR=/root/autodl-tmp/outputs/qwen2.5-1.5b-tree-gspo-gsm8k-submit-only
/root/.venv/bin/vllm serve /root/autodl-tmp/qwen2.5-1.5b-instruct \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules \
    scribe_adapter=/root/autodl-tmp/outputs/qwen2.5-1.5b-sft-gsm8k-submit-only-r64/final_lora \
    iter_0=$ADAPTER_DIR/iter_0 \
    iter_25=$ADAPTER_DIR/iter_25 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-loras 8 \
  --max-cpu-loras 8 \
  --port 8000
