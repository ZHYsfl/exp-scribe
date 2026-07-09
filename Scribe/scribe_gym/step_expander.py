"""Expand a TurnRecord into list[Step] for credit assignment + loss mask.

NO REBUILD: step.input comes directly from turn.step_records — the faithful
推理实况 captured at inference time (the exact messages list sent to the LLM
for that call, T/R already stripped). This guarantees ZERO training/inference
distribution drift. We do NOT reconstruct step.input from blocks (that would
risk divergence from what the model actually saw).

  - step.input  = StepRecord.input_messages   (T/R-stripped, what the model saw)
  - step.output = StepRecord.output_blocks     (raw, INCLUDES T — training target)

AR is excluded from step.output (observation, masked in loss) — but note the
captured output_blocks are parsed from the assistant message, which by
construction contains no AR (AR comes from the env as tool-role messages, not
in the assistant content). So output_blocks is already AR-free.

Grounded in the README turn7 example (4 steps within one turn).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .parsers import ScribeBlock, ScribeBlockType
from .turn_record import Step, TurnRecord


def expand_turn(turn: TurnRecord) -> List[Step]:
    """Turn -> list[Step], using the captured 推理实况 (turn.step_records).

    The shared system prompt already embeds the tool schema, so the captured
    input_messages are exactly what the model saw during rollout. No rebuild,
    no extra injection — zero training/inference drift.
    """
    return [Step(
        input=[dict(m) for m in rec.input_messages],
        output=list(rec.output_blocks),
        tools=rec.tools,
    ) for rec in turn.step_records]


def compute_loss_mask(step: Step) -> List[bool]:
    """Block-level loss mask for step.output.

    True for model-generated blocks (T/O/A/R/S — training targets, loss=on).
    False for any AR that slips into output (defensive; AR shouldn't be there
    since output is parsed from assistant content). Input blocks are all
    loss=off (context), but this returns only the output-side mask — the writer
    lays out [input(off...) output(this mask...)].

    Token-level expansion is deferred to the future training-data writer.
    """
    return [
        b.type != ScribeBlockType.TOOL_RESPONSE for b in step.output
    ]


__all__ = ["expand_turn", "compute_loss_mask"]
