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
from .vllm_backend import _tools_to_prompt_schema


def _inject_tool_schema_into_messages(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Reconstruct the exact system prompt a plain-text-tools backend produced.

    VLLMBackend injects the tool schema into the system prompt before sending it
    to the model. step_record.input_messages is captured BEFORE that injection,
    so training would otherwise see a different prompt than inference. This
    helper appends the same schema text so step.input matches the model's actual
    rollout prompt.
    """
    schema_text = _tools_to_prompt_schema(tools)
    if not schema_text:
        return [dict(m) for m in messages]

    out: List[Dict[str, Any]] = []
    injected = False
    for msg in messages:
        if msg.get("role") == "system" and not injected:
            content = msg.get("content") or ""
            if content:
                content = f"{content}\n\n{schema_text}"
            else:
                content = f"\n\n{schema_text}"
            out.append({**msg, "content": content})
            injected = True
        else:
            out.append(dict(msg))
    if not injected:
        out.insert(0, {"role": "system", "content": schema_text})
    return out


def expand_turn(turn: TurnRecord) -> List[Step]:
    """Turn -> list[Step], using the captured 推理实况 (turn.step_records).

    step.input  = StepRecord.input_messages with the tool schema the model saw
                  during rollout (re-injected here for plain-text backends).
    step.output = StepRecord.output_blocks    (raw, INCLUDES T — training target)
    step.tools  = tools active during the call.

    AR is excluded from step.output by construction (output_blocks is parsed
    from the assistant message content, which contains no AR). No rebuild —
    zero training/inference drift. Empty step_records -> empty list.
    """
    return [Step(
        input=_inject_tool_schema_into_messages(rec.input_messages, rec.tools),
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
