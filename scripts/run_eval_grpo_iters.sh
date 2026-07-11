#!/bin/bash
# Evaluate GRPO iter checkpoints (0,10,20,30,40,47) on 200 GSM8K test samples
# each, using the already-running eval vLLM server on :8000 (all iter adapters
# pre-registered via scripts/start_vllm_eval.sh). Greedy decoding -> reproducible;
# --concurrency parallelizes samples against vLLM's batch (max-num-seqs 32).
#
# Run persistently:  nohup bash scripts/run_eval_grpo_iters.sh > outputs/eval_grpo_iters.log 2>&1 &

set -u
cd /root/autodl-tmp
PY=/root/.venv/bin/python
ITERS="0 10 20 30 40 47"
CONC=32

echo "################################################"
echo "# GRPO iter eval start: $(date)"
echo "# iters=$ITERS  num_samples=200  greedy  concurrency=$CONC"
echo "################################################"

for N in $ITERS; do
  echo ""
  echo "==================== iter_$N ===================="
  echo "start: $(date)"
  $PY scripts/eval_submit_only.py \
    --lora_name "iter_$N" \
    --num_samples 200 \
    --concurrency "$CONC" \
    --out_path "outputs/eval_grpo_iter_$N.json" 2>&1
  rc=$?
  echo "exit_code=$rc end: $(date)"
  if [ $rc -eq 0 ] && [ -f "outputs/eval_grpo_iter_$N.json" ]; then
    $PY - "$N" <<'PYEOF'
import json, sys
n = sys.argv[1]
d = json.load(open(f"outputs/eval_grpo_iter_{n}.json"))
total = len(d); correct = sum(1 for r in d if r.get("correct"))
errs = sum(1 for r in d if "error" in r)
print(f"  >>> iter_{n}: {correct}/{total} = {correct/total*100:.1f}%  (errors={errs})")
PYEOF
  else
    echo "  >>> iter_$N: FAILED (rc=$rc)"
  fi
done

echo ""
echo "==================== AGGREGATE ===================="
$PY scripts/aggregate_eval.py
echo "==================== DONE: $(date) ===================="
