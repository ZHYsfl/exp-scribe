#!/usr/bin/env bash
# Evaluate multiple GRPO checkpoints + SFT baseline on the same 200 GSM8K test samples.
# Assumes vLLM is running with VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 and the
# /adapters endpoint available.

set -e

REPO_ROOT=/root/autodl-tmp
VLLM_URL=http://localhost:8000
NUM_SAMPLES=200

CHECKPOINTS=(
    "sft_adapter:$REPO_ROOT/outputs/qwen2.5-1.5b-sft-gsm8k-submit-only-100/final_lora"
    "iter_0:$REPO_ROOT/outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100/iter_0"
    "iter_10:$REPO_ROOT/outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100/iter_10"
    "iter_20:$REPO_ROOT/outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100/iter_20"
    "iter_30:$REPO_ROOT/outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100/iter_30"
    "iter_40:$REPO_ROOT/outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100/iter_40"
    "iter_49:$REPO_ROOT/outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100/iter_49"
)

OUT_DIR=$REPO_ROOT/outputs/checkpoint_evals
mkdir -p "$OUT_DIR"

load_adapter() {
    local name=$1
    local path=$2
    echo "Registering adapter $name from $path ..."
    curl -s -X POST "$VLLM_URL/adapters" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$name\",\"src\":\"$path\"}" > /dev/null
}

run_eval() {
    local name=$1
    echo ""
    echo "========================================"
    echo "Evaluating $name on $NUM_SAMPLES samples"
    echo "========================================"
    /root/.venv/bin/python "$REPO_ROOT/scripts/eval_submit_only.py" \
        --num_samples "$NUM_SAMPLES" \
        --lora_name "$name" \
        | tee "$OUT_DIR/${name}_n${NUM_SAMPLES}.log"
}

# Load all adapters first
for cp in "${CHECKPOINTS[@]}"; do
    name="${cp%%:*}"
    path="${cp#*:}"
    load_adapter "$name" "$path"
done

# Run evals sequentially so they share the same vLLM server & same sample order
for cp in "${CHECKPOINTS[@]}"; do
    name="${cp%%:*}"
    run_eval "$name"
done

echo ""
echo "All evaluations done. Logs in $OUT_DIR"
# Print summary
grep -h "Accuracy:" "$OUT_DIR"/*.log | sed "s#^#$OUT_DIR/#"
