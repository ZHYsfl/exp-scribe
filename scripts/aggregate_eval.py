"""Aggregate per-iter GRPO eval results into a single summary."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ITERS = [0, 10, 20, 30, 40, 47]

summary: dict = {}
for n in ITERS:
    p = REPO_ROOT / "outputs" / f"eval_grpo_iter_{n}.json"
    if not p.exists():
        summary[f"iter_{n}"] = {"error": "missing"}
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    total = len(data)
    correct = sum(1 for r in data if r.get("correct"))
    errors = sum(1 for r in data if "error" in r)
    summary[f"iter_{n}"] = {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "accuracy_pct": round(correct / total * 100, 1) if total else 0.0,
        "errors": errors,
    }

out = REPO_ROOT / "outputs" / "eval_grpo_iters_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved summary to {out}")
