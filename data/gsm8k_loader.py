"""GSM8K parquet loader.

Reads the locally cached GSM8K parquet files and extracts each sample's
question and final numeric answer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "gsm8k"


def extract_right_answer(answer_text: str) -> Optional[str]:
    """Extract the final numeric answer from GSM8K's `#### 1234` marker."""
    if not isinstance(answer_text, str):
        return None
    m = re.search(r"####\s*([\-\d\.\,\$]+)", answer_text)
    if not m:
        return None
    return m.group(1).replace(",", "").replace("$", "").strip()


def load_gsm8k(
    split: str = "train",
    config: str = "main",
    data_dir: str | Path | None = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load GSM8K samples from local parquet.

    Args:
        split: "train" or "test".
        config: "main" or "socratic". We recommend "main" for SCRIBE training.
        data_dir: Directory containing `{config}_{split}.parquet`. Defaults to
            the `data/gsm8k` folder next to this script.
        limit: If set, only return the first N samples.

    Returns:
        List of dicts with keys: task_id, task_description, right_answer, raw_answer.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    path = data_dir / f"{config}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"GSM8K parquet not found: {path}")

    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)

    items: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        answer_raw = str(row.get("answer", ""))
        items.append(
            {
                "task_id": f"gsm8k_{config}_{split}_{idx}",
                "task_description": question,
                "right_answer": extract_right_answer(answer_raw),
                "raw_answer": answer_raw,
            }
        )
    return items


if __name__ == "__main__":
    items = load_gsm8k("train", "main", limit=3)
    for it in items:
        print(it["task_id"])
        print("Q:", it["task_description"])
        print("A:", it["right_answer"])
        print("-" * 40)
