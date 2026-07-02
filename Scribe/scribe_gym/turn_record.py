"""Per-turn data structures for the SCRIBE protocol.

A TurnRecord captures one turn (one agent.chat call): the cross-turn input it
received, the blocks it generated (new/rollout), the raw messages it contributed
to future inputs (with tool_call_id preserved), its reward, and the feedback
emitted at its end.

A Step is one LLM call within a turn — only materialized when expanding a turn
for credit assignment / loss-mask construction (see step_expander.py).

Block notation: T=think, A=tool_call, O=output, R=reflect, S=turn_summary,
AR=tool_response. T/R are NEVER in any inference input (disposable); they live
only in new/rollout blocks (training, loss=on). AR is excluded from rollout
(it's an observation, masked in loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .parsers import ScribeBlock, ScribeBlockType


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
class TurnRecord:
    """One turn's complete record.

    input: the (already cross-turn-compressed) messages fed to the FIRST LLM
           call of this turn.
    new_blocks: ALL blocks this turn generated, in order: T O [A AR] R S.
    rollout_blocks: training sequence T O A R S (AR EXCLUDED — observation).
    retained_messages: the raw assistant+tool messages this turn contributed to
           FUTURE turns' inputs (tool_call_id preserved). Used by HistoryManager
           to rebuild cross-turn input without re-deriving ids from blocks.
    reward: this turn's 13-metric reward.
    feedback: feedback string emitted at this turn's end (attached to THIS
              turn; HistoryManager emits it after this turn's blocks in the
              next turn's input — in-position, not piled at end).
    """

    input: List[Dict[str, Any]]
    new_blocks: List[ScribeBlock]
    rollout_blocks: List[ScribeBlock]
    retained_messages: List[Dict[str, Any]]
    reward: float = 0.0
    feedback: Optional[str] = None
    has_summary: bool = field(init=False)

    def __post_init__(self) -> None:
        self.has_summary = any(
            b.type == ScribeBlockType.TURN_SUMMARY for b in self.new_blocks
        )


def has_summary(new_blocks: List[ScribeBlock]) -> bool:
    """True if the block list contains a TURN_SUMMARY block."""
    return any(b.type == ScribeBlockType.TURN_SUMMARY for b in new_blocks)


def rollout_from_new(new_blocks: List[ScribeBlock]) -> List[ScribeBlock]:
    """rollout = new_blocks with TOOL_RESPONSE (AR) removed.

    AR is an observation (env-produced), not model-generated, so it's masked
    out of the training sequence. The rollout keeps T O A R S.
    """
    return [b for b in new_blocks if b.type != ScribeBlockType.TOOL_RESPONSE]


__all__ = ["Step", "TurnRecord", "has_summary", "rollout_from_new"]
