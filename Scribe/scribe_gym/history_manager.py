"""Per-turn history management for the SCRIBE protocol (inference side).

HistoryManager owns CROSS-turn input construction: before each turn's first
LLM call it rebuilds the input messages from prior turns per the spec:

  - T/R NEVER in input (disposable the moment generated).
  - recent turn (<k): keep its retained_messages (O/A/AR/S, ids intact).
  - old turn (>=k): keep ONLY its S message; if no S, keep O+A raw (fallback).
  - feedback sits AFTER its turn's blocks (in-position, not piled at end).
  - compression is turn-level + atomic (not mid-turn), checked before rollout:
      Stage 1 = collapse old turns to S (kickoff+feedbacks verbatim).
      Stage 2 = sota-LLM folds kickoff+feedbacks+S into one <turn_summary>.
      Hard boundary = if system+kickoff+feedbacks alone overflow -> error.

Hot-swappable: pass a HistoryManager to GymBackedAgent for SCRIBE; pass None
for plain ReAct (the ablation baseline).

Within-turn accumulation (step k+1 sees step k's O/A/AR) is NOT here — that's
Agent.chat's while-loop. This module is called ONCE per turn at chat entry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import re

from .parsers import ScribeBlockType, parse_scribe_blocks
from .turn_record import TurnRecord


# All scribe block tags (open and close) that must be stripped when collapsing
# O+A into a synthetic S inner text — no nested tags may survive inside an S.
_SCRIBE_TAGS = [
    "think", "tool_call", "tool_response", "reflect", "turn_summary",
]
_SCRIBE_TAG_RE = re.compile(
    "|".join(rf"</?{t}>" for t in _SCRIBE_TAGS)
)


def _strip_all_scribe_tags(text: str) -> str:
    """Remove every scribe open/close tag from text, leaving bare inner content.
    Used when synthesizing an S from O+A: the S inner text must contain no
    nested tags (else parsing breaks and tool-call structure leaks back in)."""
    return _SCRIBE_TAG_RE.sub("", text).strip()


class InfeasibleContextError(RuntimeError):
    """Raised when the incompressible prefix (system+kickoff+all feedbacks)
    alone exceeds the hard limit — the trajectory config is infeasible."""


@runtime_checkable
class _TokenCounterLike(Protocol):
    def count_observations(self, observations: list[dict]) -> int: ...


class HistoryManager:
    def __init__(
        self,
        token_counter: _TokenCounterLike,
        compressor: Any | None = None,
        *,
        k: int = 3,
        hard_limit: Optional[int] = None,
        compression_margin: int = 0,
    ):
        # hard_limit: the served model's context window (in tokens). REQUIRED
        # for compression to engage; pass None to DISABLE compression entirely
        # (pure recent/old block retention — fine for early experiments / small
        # tasks that never overflow). No magic 1M default: a wrong default
        # silently never triggers on small models or is wrong for big ones.
        # compression_margin: budget headroom; compression fires when token
        # count exceeds (hard_limit - margin). Ignored when hard_limit is None.
        if hard_limit is not None:
            if hard_limit <= 0:
                raise ValueError("hard_limit must be positive (match your model's max context)")
            if not (0 <= compression_margin < hard_limit):
                raise ValueError("compression_margin must be in [0, hard_limit)")
        self._counter = token_counter
        self._compressor = compressor
        self._k = max(0, k)
        self._margin = compression_margin
        self._hard_limit = hard_limit
        self._turns: List[TurnRecord] = []
        self._system_msg: Optional[Dict[str, Any]] = None
        self._kickoff_msg: Optional[Dict[str, Any]] = None

    # -- lifecycle ---------------------------------------------------------

    def set_prefix(self, system_msg: Dict[str, Any], kickoff_msg: Dict[str, Any]) -> None:
        self._system_msg = dict(system_msg)
        self._kickoff_msg = dict(kickoff_msg)

    def reset(self) -> None:
        self._turns.clear()
        self._system_msg = None
        self._kickoff_msg = None

    def add_turn(self, turn: TurnRecord) -> None:
        self._turns.append(turn)

    def set_feedback(self, turn_index: int, feedback: Optional[str]) -> None:
        if 0 <= turn_index < len(self._turns):
            self._turns[turn_index].feedback = feedback

    @property
    def num_turns(self) -> int:
        return len(self._turns)

    # -- cross-turn input build -------------------------------------------

    def build_input(self) -> List[Dict[str, Any]]:
        """Assemble the LLM input for the upcoming turn: prefix + each prior
        turn's retained blocks (recent full / old S-or-O+A) + in-position
        feedback, then compress if over budget."""
        msgs: List[Dict[str, Any]] = []
        if self._system_msg is not None:
            msgs.append(self._system_msg)
        if self._kickoff_msg is not None:
            msgs.append(self._kickoff_msg)

        n = len(self._turns)
        for i, turn in enumerate(self._turns):
            is_recent = i >= n - self._k
            msgs.extend(self._turn_messages(turn, is_recent))
            if turn.feedback is not None:
                msgs.append({"role": "user", "content": turn.feedback})

        return self._compress_if_needed(msgs)

    def _turn_messages(self, turn: TurnRecord, is_recent: bool) -> List[Dict[str, Any]]:
        """Return the messages a turn contributes to the next-turn input.

        recent -> all retained_messages (O/A/AR/S, ids intact).
        old + has_summary -> only the S block (O+R stripped from the final msg).
        old + no S (special case) -> O+A rendered as a single plain-text
            assistant message (O+A "as S" — no tool_calls, no AR). See
            _output_action_only_messages.
        retained_messages never contain T/R (they were captured post-strip).
        """
        if is_recent:
            return [dict(m) for m in turn.retained_messages]
        if turn.has_summary:
            return self._summary_only_messages(turn)
        return self._output_action_only_messages(turn)

    def _summary_only_messages(self, turn: TurnRecord) -> List[Dict[str, Any]]:
        """Old turn with S: keep ONLY the S block's BARE inner text as the
        assistant message content (no <turn_summary> tags in the message —
        message content is what the LLM sees; S tags are a block-protocol
        concern). The final assistant message of a turn carries O+R+S
        concatenated; we parse it, take the TURN_SUMMARY block, strip its tags,
        and emit one plain-text assistant message."""
        out: List[Dict[str, Any]] = []
        for m in turn.retained_messages:
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                content = m.get("content") or ""
                blocks, _ = parse_scribe_blocks(content)
                summary_blocks = [
                    b for b in blocks if b.type == ScribeBlockType.TURN_SUMMARY
                ]
                if summary_blocks:
                    new_msg = dict(m)
                    new_msg["content"] = "\n".join(
                        _strip_all_scribe_tags(b.content) for b in summary_blocks
                    )
                    out.append(new_msg)
        return out

    def _output_action_only_messages(self, turn: TurnRecord) -> List[Dict[str, Any]]:
        """Special-case fallback: old turn with NO S block -> synthesize an S
        block from O+A (spec: "preserve O&A blocks instead when this turn is
        old enough to just need to preserve S").

        Two layers, kept distinct:
          - BLOCK layer: this turn is represented by ONE TURN_SUMMARY block
            whose content carries the <turn_summary>...</turn_summary> tags.
          - MESSAGE layer: the assistant message content is the BARE inner
            text (no <turn_summary> tags) — message content is what the LLM
            sees, and S tags are a block-protocol concern, not a message one.

        The inner text is built by stripping ALL scribe tags from each O and A
        block's content (so no nested <tool_call>/tool_call> tags survive — they
        would break parsing and re-introduce tool-call structure), then
        joining O and A pieces with '\n' separators."""
        inner = "\n".join(
            _strip_all_scribe_tags(b.content)
            for b in turn.new_blocks
            if b.type in (ScribeBlockType.OUTPUT, ScribeBlockType.TOOL_CALL)
        )
        if not inner:
            return []
        # Message content: bare inner text (no <turn_summary> wrapper).
        # (The block-layer TURN_SUMMARY with tags is constructed where blocks
        # are needed for training data; here we only build the LLM message.)
        return [{"role": "assistant", "content": inner}]

    # -- compression (turn-level, atomic) ---------------------------------

    def _under_budget(self, msgs: List[Dict[str, Any]]) -> bool:
        if self._hard_limit is None:
            return True  # compression disabled: always "under budget"
        return self._counter.count_observations(msgs) <= self._hard_limit - self._margin

    def _compress_if_needed(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._hard_limit is None:
            return msgs  # compression disabled
        if self._under_budget(msgs):
            return msgs
        # Stage 1: collapse every turn (incl. recent) to S / O+A fallback.
        # Kickoff+feedbacks verbatim, system verbatim.
        stage1 = self._compress_stage1()
        if self._under_budget(stage1):
            return stage1
        # Stage 2: sota-LLM fold kickoff+feedbacks+S into one <turn_summary>.
        if self._compressor is None:
            # No compressor configured: return stage1 as best effort (may still
            # be over budget; the caller's LLM call may fail, but we don't
            # silently lose data). Hard boundary check still applies.
            self._check_hard_boundary(stage1)
            return stage1
        stage2 = self._compress_stage2_sync(stage1)
        self._check_hard_boundary(stage2)
        return stage2

    def _compress_stage1(self) -> List[Dict[str, Any]]:
        """Stage 1: collapse EVERY turn (including recent ones) to S-only (or
        the O+A fallback). Kickoff + all feedbacks verbatim, system verbatim.

        Rather than re-derive retention from the flat message list (which
        duplicated _turn_messages logic and drifted out of sync — the old
        _retain_s_or_oa left O+R in the S branch and kept tool_calls+AR in the
        fallback), we rebuild directly from self._turns treating every turn as
        OLD (is_recent=False). This reuses the single source of truth:
        _turn_messages -> _summary_only_messages / _output_action_only_messages.
        """
        out: List[Dict[str, Any]] = []
        if self._system_msg is not None:
            out.append(self._system_msg)
        if self._kickoff_msg is not None:
            out.append(self._kickoff_msg)
        for turn in self._turns:
            out.extend(self._turn_messages(turn, is_recent=False))
            if turn.feedback is not None:
                out.append({"role": "user", "content": turn.feedback})
        return out

    def _compress_stage2_sync(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 2: fold kickoff + all feedbacks + all S blocks into one
        <turn_summary> via the sota LLM. System stays verbatim.

        NOTE: the compressor is async; this method is sync and runs the event
        loop via asyncio.run. HistoryManager.build_input is called from a sync
        context (chat entry). If an event loop is already running, callers
        must wrap appropriately. For the SCRIBE rollout path the runner drives
        an async chat, so build_input runs outside the loop — safe.
        """
        import asyncio

        system = ""
        kickoff = ""
        feedbacks: List[str] = []
        s_blocks: List[str] = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                system = content
            elif role == "user" and not kickoff:
                kickoff = content
            elif role == "user":
                feedbacks.append(content)
            elif role == "assistant":
                s_blocks.append(content)

        summary = asyncio.run(
            self._compressor.summarize(system, kickoff, feedbacks, s_blocks)
        )
        out: List[Dict[str, Any]] = []
        if self._system_msg is not None:
            out.append(self._system_msg)
        out.append({"role": "assistant", "content": summary})
        return out

    def _check_hard_boundary(self, msgs: List[Dict[str, Any]]) -> None:
        """If the incompressible prefix (system+kickoff+all feedbacks) alone
        overflows the hard limit, raise — config is infeasible. No-op when
        compression is disabled (hard_limit is None)."""
        if self._hard_limit is None:
            return
        prefix: List[Dict[str, Any]] = []
        if self._system_msg is not None:
            prefix.append(self._system_msg)
        if self._kickoff_msg is not None:
            prefix.append(self._kickoff_msg)
        for m in msgs:
            if m.get("role") == "user":
                prefix.append(m)
        if self._counter.count_observations(prefix) > self._hard_limit:
            raise InfeasibleContextError(
                "Incompressible prefix (system + kickoff + all feedbacks) "
                f"exceeds hard limit {self._hard_limit}; trajectory config is "
                "infeasible."
            )


__all__ = ["HistoryManager", "InfeasibleContextError"]
