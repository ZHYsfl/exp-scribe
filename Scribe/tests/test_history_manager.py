"""Unit tests for HistoryManager (SCRIBE cross-turn input rebuild).

Covers: block retention (recent/old), special case (no S -> O+A), T/R never
in input, feedback in-position, Stage-1 compression, Stage-2 (mock compressor),
hard boundary, reset.

No LLM, no tokenizer — uses a fake token counter (counts messages)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest

from Scribe.scribe_gym import (
    HistoryManager,
    InfeasibleContextError,
    TurnRecord,
)
from Scribe.scribe_gym.parsers import parse_scribe_blocks, ScribeBlockType
from Scribe.scribe_gym.turn_record import rollout_from_new


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------

class FakeCounter:
    """Counts 10 tokens per message — deterministic, no tokenizer."""
    def count_observations(self, observations):
        return len(observations) * 10


def make_blocks(scribe_text):
    blocks, _ = parse_scribe_blocks(scribe_text)
    return blocks


def mk_turn(scribe_text, retained_messages, feedback=None, reward=0.0):
    blocks = make_blocks(scribe_text)
    return TurnRecord(
        input=[],
        new_blocks=blocks,
        rollout_blocks=rollout_from_new(blocks),
        retained_messages=retained_messages,
        reward=reward,
        feedback=feedback,
    )


SYS = {"role": "system", "content": "SYS"}
KICK = {"role": "user", "content": "KICK"}


# ---------------------------------------------------------------------------
# Block retention
# ---------------------------------------------------------------------------

def test_recent_turn_retains_all_retained_messages():
    hm = HistoryManager(FakeCounter(), k=3)
    hm.set_prefix(SYS, KICK)
    retained = [
        {"role": "assistant", "content": "<turn_summary>sum0</turn_summary>"},
    ]
    hm.add_turn(mk_turn("民主党t0 Канаoutput0<turn_summary>sum0</turn_summary>", retained))
    inp = hm.build_input()
    # system + kickoff + the retained summary message
    assert inp[0] == SYS
    assert inp[1] == KICK
    assert inp[2]["content"] == "<turn_summary>sum0</turn_summary>"


def test_old_turn_collapses_to_summary_only():
    hm = HistoryManager(FakeCounter(), k=2)
    hm.set_prefix(SYS, KICK)
    # add 4 turns; turn0 becomes old (>=k=2 from the end)
    for i in range(4):
        retained = [
            {"role": "assistant", "content": f"output{i}", "tool_calls": [{"id": f"c{i}"}]},
            {"role": "tool", "content": f"resp{i}"},
            {"role": "assistant", "content": f"<turn_summary>sum{i}</turn_summary>"},
        ]
        hm.add_turn(mk_turn(
            f"民主党t{i} Канаoutput{i}<turn_summary>sum{i}</turn_summary>", retained
        ))
    inp = hm.build_input()
    # turn0 is old -> only its summary's BARE inner text (tags stripped), NOT output0/resp0
    contents = [m.get("content", "") for m in inp]
    assert "sum0" in contents
    assert "<turn_summary>sum0</turn_summary>" not in contents  # tags stripped in message
    assert "output0" not in contents
    assert "resp0" not in contents
    # turn3 (recent) keeps output3/resp3 (recent = raw retained, not stripped)
    assert "output3" in contents
    assert "resp3" in contents


def test_old_turn_strips_O_and_R_keeps_only_S():
    """The final assistant message of a turn carries O+R+S concatenated in
    one content string. An old turn must keep ONLY S — O and R must be
    stripped, not the whole O+R+S message kept verbatim."""
    hm = HistoryManager(FakeCounter(), k=2)
    hm.set_prefix(SYS, KICK)
    # turn0's final message has O + R + S in one content (realistic shape).
    retained0 = [
        {"role": "assistant", "content": "o0", "tool_calls": [{"id": "c0"}]},
        {"role": "tool", "content": "r0"},
        {"role": "assistant", "content": "final_output<reflect>refl0</reflect><turn_summary>sum0</turn_summary>"},
    ]
    # turn1, turn2 to push turn0 into "old" (k=2).
    retained1 = [{"role": "assistant", "content": "<turn_summary>sum1</turn_summary>"}]
    retained2 = [{"role": "assistant", "content": "<turn_summary>sum2</turn_summary>"}]
    hm.add_turn(mk_turn("民主党t0 Канаo0<reflect>refl0</reflect><turn_summary>sum0</turn_summary>", retained0))
    hm.add_turn(mk_turn("民主党t1 Канаo1<turn_summary>sum1</turn_summary>", retained1))
    hm.add_turn(mk_turn("民主党t2 Канаo2<turn_summary>sum2</turn_summary>", retained2))
    inp = hm.build_input()
    # turn0 is old -> ONLY its S block's BARE inner text survives (tags stripped);
    # O ('final_output') and R must be gone.
    contents = [m.get("content", "") for m in inp]
    assert "sum0" in contents
    assert "<turn_summary>sum0</turn_summary>" not in contents  # tags stripped in old turn
    assert "final_output" not in contents   # O stripped
    assert "refl0" not in contents           # R stripped
    assert "<reflect>refl0</reflect>" not in contents


def test_old_turn_no_S_keeps_O_and_A_as_plain_text():
    """Old turn with no S: O+A are rendered as ONE plain-text assistant
    message (O+A 'as S'). No tool_calls, no AR (resp) — those are dropped."""
    TC = chr(60) + "tool_call" + chr(62)
    TC_END = chr(60) + "/tool_call" + chr(62)
    AR = chr(60) + "tool_response" + chr(62)
    AR_END = chr(60) + "/tool_response" + chr(62)
    hm = HistoryManager(FakeCounter(), k=2)
    hm.set_prefix(SYS, KICK)
    # turn0: O+A+AR, no S. new_blocks has O, A, AR.
    nb0 = f"output0{TC}\n" + '{"name":"bash","arguments":{}}' + f"\n{TC_END}{AR}resp0{AR_END}"
    retained0 = [
        {"role": "assistant", "content": "output0", "tool_calls": [{"id": "c0"}]},
        {"role": "tool", "content": "resp0"},
    ]
    hm.add_turn(mk_turn(nb0, retained0))
    # turn1, turn2 with S, to push turn0 into "old".
    for i in (1, 2):
        hm.add_turn(mk_turn(
            f"o{i}<turn_summary>sum{i}</turn_summary>",
            [{"role": "assistant", "content": f"<turn_summary>sum{i}</turn_summary>"}],
        ))
    inp = hm.build_input()
    # turn0 old + no S -> one plain-text assistant message whose content is the
    # BARE inner text of O+A (all scribe tags stripped), joined by newline.
    turn0_msgs = [m for m in inp if "output0" in (m.get("content") or "")]
    assert len(turn0_msgs) == 1, f"expected 1 O+A msg for turn0, got {turn0_msgs}"
    msg = turn0_msgs[0]
    assert msg["role"] == "assistant"
    assert not msg.get("tool_calls"), "O+A fallback must not carry tool_calls"
    # Message content has NO scribe tags (bare inner text only).
    TC = chr(60) + "tool_call" + chr(62)
    TS = chr(60) + "turn_summary" + chr(62)
    assert TS not in msg["content"]
    assert TC not in msg["content"]
    assert "output0" in msg["content"]
    # The A block's JSON arguments survive as bare text (tags stripped).
    assert '"name":"bash"' in msg["content"]
    # AR (resp0) must NOT appear anywhere in the input.
    all_contents = [m.get("content", "") for m in inp]
    assert not any("resp0" in c for c in all_contents), "AR must be dropped in O+A fallback"


def test_recent_turn_no_S_keeps_O_A_AR():
    hm = HistoryManager(FakeCounter(), k=3)
    hm.set_prefix(SYS, KICK)
    retained = [
        {"role": "assistant", "content": "output0", "tool_calls": [{"id": "c0"}]},
        {"role": "tool", "content": "resp0"},
    ]
    hm.add_turn(mk_turn("民主党t0 Канаoutput0", retained))  # no S
    inp = hm.build_input()
    contents = [m.get("content", "") for m in inp]
    assert "output0" in contents
    assert "resp0" in contents


# ---------------------------------------------------------------------------
# T/R never in input
# ---------------------------------------------------------------------------

def test_T_R_never_in_input():
    hm = HistoryManager(FakeCounter(), k=3)
    hm.set_prefix(SYS, KICK)
    # retained messages must already be T/R-stripped (the agent strips them);
    # but verify build_input doesn't introduce any T/R-bearing content.
    retained = [{"role": "assistant", "content": "<turn_summary>sum0</turn_summary>"}]
    hm.add_turn(mk_turn("民主党t0 Канаoutput0<turn_summary>sum0</turn_summary>", retained))
    inp = hm.build_input()
    for m in inp:
        c = m.get("content", "") or ""
        assert "民主党" not in c
        assert "<reflect>" not in c


# ---------------------------------------------------------------------------
# Feedback position
# ---------------------------------------------------------------------------

def test_feedback_in_position_after_turn_blocks():
    hm = HistoryManager(FakeCounter(), k=5)
    hm.set_prefix(SYS, KICK)
    r0 = [{"role": "assistant", "content": "<turn_summary>sum0</turn_summary>"}]
    r1 = [{"role": "assistant", "content": "<turn_summary>sum1</turn_summary>"}]
    hm.add_turn(mk_turn("民主党t0 Канаoutput0<turn_summary>sum0</turn_summary>", r0, feedback="FB0"))
    hm.add_turn(mk_turn("民主党t1 Канаoutput1<turn_summary>sum1</turn_summary>", r1, feedback="FB1"))
    inp = hm.build_input()
    # Expected order: SYS, KICK, sum0, FB0, sum1, FB1
    assert inp[2]["content"] == "<turn_summary>sum0</turn_summary>"
    assert inp[3]["content"] == "FB0"
    assert inp[4]["content"] == "<turn_summary>sum1</turn_summary>"
    assert inp[5]["content"] == "FB1"


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def test_stage1_compresses_when_over_budget():
    # FakeCounter = 10 tokens/msg. Full input = 8 msgs = 80 tokens. Prefix
    # (SYS+KICK+FB0+FB1) = 4 msgs = 40 tokens. Set hard_limit=70 so full input
    # (80) is over budget -> stage1 triggers; stage1 output (6 msgs=60) < 70;
    # prefix (40) < 70 so no hard boundary. No compressor -> returns stage1.
    hm = HistoryManager(FakeCounter(), k=1, hard_limit=70, compression_margin=0)
    hm.set_prefix(SYS, KICK)
    r0 = [{"role": "assistant", "content": "<turn_summary>sum0</turn_summary>"}]
    r1 = [
        {"role": "assistant", "content": "output1", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "resp1"},
        {"role": "assistant", "content": "<turn_summary>sum1</turn_summary>"},
    ]
    hm.add_turn(mk_turn("民主党t0 Канаoutput0<turn_summary>sum0</turn_summary>", r0, feedback="FB0"))
    hm.add_turn(mk_turn("民主党t1 Канаoutput1<turn_summary>sum1</turn_summary>", r1, feedback="FB1"))
    inp = hm.build_input()
    # stage1 collapses recent turn1's O/A/AR away, keeping only S + feedbacks.
    contents = [m.get("content", "") for m in inp]
    assert "output1" not in contents  # recent turn also collapsed under stage1
    assert "resp1" not in contents
    # S kept as BARE inner text (tags stripped), not the raw <turn_summary> msg.
    assert "sum1" in contents
    assert "<turn_summary>sum1</turn_summary>" not in contents
    # feedbacks preserved
    assert "FB0" in contents
    assert "FB1" in contents


class FakeCompressor:
    def __init__(self):
        self.called = False
    async def summarize(self, system, kickoff, feedbacks, s_blocks):
        self.called = True
        return f"<turn_summary>folded {len(s_blocks)} summaries</turn_summary>"


def test_stage2_invokes_compressor():
    comp = FakeCompressor()
    hm = HistoryManager(FakeCounter(), compressor=comp, k=1, hard_limit=30, compression_margin=0)
    hm.set_prefix(SYS, KICK)
    r0 = [{"role": "assistant", "content": "<turn_summary>sum0</turn_summary>"}]
    hm.add_turn(mk_turn("民主党t0 Канаoutput0<turn_summary>sum0</turn_summary>", r0, feedback="FB0"))
    inp = hm.build_input()
    assert comp.called
    # Result is system + single folded summary
    assert inp[0] == SYS
    assert inp[1]["content"].startswith("<turn_summary>folded")


def test_hard_boundary_raises():
    hm = HistoryManager(FakeCounter(), compressor=FakeCompressor(), k=1, hard_limit=15, compression_margin=0)
    hm.set_prefix(SYS, KICK)
    r0 = [{"role": "assistant", "content": "<turn_summary>sum0</turn_summary>"}]
    hm.add_turn(mk_turn("民主党t0 Канаoutput0<turn_summary>sum0</turn_summary>", r0, feedback="FB0"))
    # system+kickoff+FB0 = 3 messages = 30 tokens > 15 hard_limit
    with pytest.raises(InfeasibleContextError):
        hm.build_input()


def test_reset_clears():
    hm = HistoryManager(FakeCounter(), k=3)
    hm.set_prefix(SYS, KICK)
    hm.add_turn(mk_turn("民主党t0 Канаoutput0<turn_summary>sum0</turn_summary>",
                        [{"role": "assistant", "content": "<turn_summary>sum0</turn_summary>"}]))
    hm.reset()
    assert hm.num_turns == 0
    inp = hm.build_input()
    assert inp == []  # no prefix after reset
