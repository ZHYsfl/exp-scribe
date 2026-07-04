"""Per-turn data structures for the SCRIBE protocol.

A TurnRecord captures one turn (one agent.chat call). step_records is the
SINGLE SOURCE OF TRUTH (推理实况, captured at inference); all other fields
(input / new_blocks / rollout_blocks / retained_messages / has_summary) are
DERIVED from it via properties — no redundant storage, no drift.

A Step is the training-side view of one LLM call (input + output), built by
step_expander from the captured StepRecords.

Block notation: T=think, A=tool_call, O=output, R=reflect, S=turn_summary,
AR=tool_response. T/R are NEVER in any inference input (disposable); they live
only in output blocks (training, loss=on). AR is excluded from rollout
(it's an observation, masked in loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .parsers import ScribeBlock, ScribeBlockType

_AR_OPEN = chr(60) + "tool_response" + chr(62)
_AR_CLOSE = chr(60) + "/tool_response" + chr(62)
_T = ScribeBlockType.THINK
_R = ScribeBlockType.REFLECT


@dataclass
class Step:
    """One LLM call within a turn (training-side view).

    input: messages fed to this call (cross-turn history + within-turn prior
           steps' O/A/AR accumulation). NEVER contains T/R.
    output: blocks this call generated. INCLUDES T (model-generated, loss=on).
            EXCLUDES AR (observation). The last step's output has R and S.
    """

    input: List[Dict[str, Any]]
    output: List[ScribeBlock]


@dataclass
class StepRecord:
    """The faithful record of one LLM call (推理实况), captured at inference
    time so training data has ZERO distribution drift — step.input is exactly
    what the model saw, step.output is exactly what it generated.

    input_messages: the messages list ACTUALLY sent to the LLM for this call
           (T/R stripped, since T/R are disposable next step — matches spec).
           This is the ground truth for training step.input, NOT a rebuild.
    output_blocks: the blocks the model generated this call, parsed from the
           RAW assistant message (INCLUDES T — T is the training target).
    raw_assistant_message: the original assistant message dict (with tool_calls
           + tool_call_id preserved) for downstream reconstruction.
    tool_messages: the tool-role response messages for this step's tool calls
           (with tool_call_id), in call order. Empty for the final non-tool step.
    """

    input_messages: List[Dict[str, Any]]
    output_blocks: List[ScribeBlock]
    raw_assistant_message: Dict[str, Any]
    tool_messages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnRecord:
    """One turn's complete record.

    step_records is the SINGLE SOURCE OF TRUTH (推理实况). Everything else is
    derived: input/retained_messages/new_blocks/rollout_blocks/has_summary are
    properties computed from step_records. This avoids redundant storage and
    the drift that redundancy invites.

    reward: this turn's 13-metric reward.
    feedback: feedback string emitted at this turn's end (attached to THIS
              turn; HistoryManager emits it after this turn's blocks in the
              next turn's input — in-position, not piled at end).
    """

    step_records: List[StepRecord]
    reward: float = 0.0
    feedback: Optional[str] = None

    # ---- derived views (single source of truth = step_records) -------------

    @property
    def input(self) -> List[Dict[str, Any]]:
        """Messages fed to the FIRST LLM call of this turn (cross-turn input)."""
        if not self.step_records:
            return []
        return self.step_records[0].input_messages

    @property
    def retained_messages(self) -> List[Dict[str, Any]]:
        """The (T/R-stripped) assistant + tool messages this turn contributed
        to FUTURE turns' inputs. Built by concatenating each step's raw
        assistant message (T/R stripped) + its tool messages, in order."""
        from .parsers import parse_scribe_blocks, render_scribe_blocks
        out: List[Dict[str, Any]] = []
        for rec in self.step_records:
            msg = dict(rec.raw_assistant_message)
            if msg.get("role") == "assistant":
                content = msg.get("content") or ""
                if content:
                    blocks, _ = parse_scribe_blocks(content)
                    kept = [b for b in blocks if b.type not in (_T, _R)]
                    msg["content"] = render_scribe_blocks(kept)
            out.append(msg)
            out.extend(rec.tool_messages)
        return out

    @property
    def new_blocks(self) -> List[ScribeBlock]:
        """ALL blocks this turn generated, in order: T O [A AR] R S. AR blocks
        are reconstructed from each step's tool_messages (wrapped in
        <tool_response> tags)."""
        out: List[ScribeBlock] = []
        for rec in self.step_records:
            out.extend(rec.output_blocks)
            for tm in rec.tool_messages:
                content = tm.get("content") or ""
                out.append(ScribeBlock(
                    type=ScribeBlockType.TOOL_RESPONSE,
                    content=f"{_AR_OPEN}{content}{_AR_CLOSE}",
                    parsed=None,
                ))
        return out

    @property
    def rollout_blocks(self) -> List[ScribeBlock]:
        """Training sequence T O A R S (AR EXCLUDED — observation, masked)."""
        return [b for b in self.new_blocks if b.type != ScribeBlockType.TOOL_RESPONSE]

    @property
    def has_summary(self) -> bool:
        """True if any block this turn generated is a TURN_SUMMARY."""
        return any(
            b.type == ScribeBlockType.TURN_SUMMARY
            for rec in self.step_records
            for b in rec.output_blocks
        )


def has_summary(new_blocks: List[ScribeBlock]) -> bool:
    """True if the block list contains a TURN_SUMMARY block."""
    return any(b.type == ScribeBlockType.TURN_SUMMARY for b in new_blocks)


__all__ = ["Step", "StepRecord", "TurnRecord", "has_summary"]

