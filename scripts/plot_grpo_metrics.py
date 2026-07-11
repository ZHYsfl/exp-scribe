"""Plot GRPO training metrics from metrics_log.jsonl into pics/.

Reads the live metrics_log.jsonl (one JSON record per iteration) and draws 19
line charts: m1..m16 plus KL, std_reward, mean_reward. Safe to run while
training is in progress - it only reads the file and skips any partially
written trailing line.

Usage:
    python scripts/plot_grpo_metrics.py \
        [--log outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-100/metrics_log.jsonl] \
        [--out pics]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Short descriptions for plot titles (from rewards.py docstring).
METRIC_DESC = {
    1: "answer correctness + natural termination",
    2: "tool usage (submit count + dup)",
    3: "step length (fewer steps = higher)",
    4: "format correctness",
    5: "total rollout tokens",
    6: "token reuse ratio",
    7: "faithfulness (judge)",
    8: "direction neutrality (judge)",
    9: "turn focus (judge)",
    10: "fluency (judge)",
    11: "compression ratio",
    12: "cross-block n-gram overlap",
    13: "intra-block n-gram overlap",
    14: "malformed tool-call penalty",
    15: "repeated tool-call penalty",
    16: "cross-turn duplicate submit penalty",
}


def load_records(log_path: Path) -> list[dict]:
    """Read all complete JSONL records, sorted by iter; skip partial lines."""
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # Trailing partial write while training appends - skip it.
            continue
        records.append(rec)
    records.sort(key=lambda r: r.get("iter", 0))
    return records


def _plot(xs, ys, title, ylabel, out_path: Path, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "-o", linewidth=1.6, markersize=4, color="#1f77b4")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=str,
        default="outputs/qwen2.5-1.5b-grpo-gsm8k-submit-only-r64-w060-b040-conc8/metrics_log.jsonl",
    )
    ap.add_argument("--out", type=str, default="pics")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(log_path)
    if not records:
        raise SystemExit(f"No complete records found in {log_path}")
    iters = [r["iter"] for r in records]
    n = len(records)
    print(f"Loaded {n} records (iter {iters[0]}..{iters[-1]}) from {log_path}")

    # 1-16: per-metric line charts.
    for i in range(1, 17):
        key = f"m{i}"
        ys = [r["metric_means"][key] for r in records]
        _plot(
            iters, ys,
            title=f"metric_{i} - {METRIC_DESC[i]}",
            ylabel=f"m{i} (mean over turns)",
            out_path=out_dir / f"m{i}.png",
            ylim=(0.0, 1.05),
        )

    # 17: GSPO step-ratio drift (scribe/max_step_ratio). With the old-anchored
    # step-level ratio this stays near 1 each iter; a spike means some step
    # moved far in one update. (Replaces the old grpo/kl plot - KL to frozen
    # SFT was removed; the step-ratio peak is now the drift gauge.)
    ratios = [r.get("scribe", {}).get("scribe/max_step_ratio", 0.0) for r in records]
    _plot(iters, ratios, title="GSPO step-ratio peak (π_θ/π_θold per step)",
          ylabel="scribe/max_step_ratio", out_path=out_dir / "step_ratio.png")

    # 18: reward std (group spread -> advantage signal strength).
    stds = [r.get("std_reward", 0.0) for r in records]
    _plot(iters, stds, title="Reward std (within group)",
          ylabel="std_reward", out_path=out_dir / "std.png", ylim=(0.0, None))

    # 19: mean reward (the headline trajectory reward).
    means = [r.get("mean_reward", 0.0) for r in records]
    _plot(iters, means, title="Mean trajectory reward",
          ylabel="mean_reward", out_path=out_dir / "mean_reward.png",
          ylim=(0.0, 1.05))

    print(f"Done: {len(list(out_dir.glob('*.png')))} plots in {out_dir}/")


if __name__ == "__main__":
    main()
