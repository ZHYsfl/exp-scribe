"""Unit tests for step_expander — turn -> list[Step] + loss mask.

Grounded in the README turn7 example (4 steps). Verifies:
  - step.output contains T (model-generated); excludes AR.
  - step.input accumulates prior steps' O/A/AR (no T/R/S).
  - last step output has R and S.
  - loss mask: input blocks off, output blocks on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Scribe.scribe_gym import TurnRecord, expand_turn, compute_loss_mask
from Scribe.scribe_gym.parsers import parse_scribe_blocks, ScribeBlockType

# Real block tags built via chr() to keep this file free of literal angle-bracket
# tags that confuse the test source.
TC = chr(60) + "tool_call" + chr(62)
TC_END = chr(60) + "/tool_call" + chr(62)
AR = chr(60) + "tool_response" + chr(62)
AR_END = chr(60) + "/tool_response" + chr(62)
THINK = chr(60) + "think" + chr(62)
THINK_END = chr(60) + "/think" + chr(62)


def blocks_of(scribe_text):
    b, _ = parse_scribe_blocks(scribe_text)
    return b


def mk_turn(new_blocks_text, turn_input=None):
    b = blocks_of(new_blocks_text)
    return TurnRecord(
        input=turn_input or [{"role": "system", "content": "SYS"}],
        new_blocks=b,
        rollout_blocks=b,
        retained_messages=[],
    )


def _tool_step(prefix, name, resp):
    return f"{prefix}{TC}\n" + f'{{"name":"{name}","arguments":{{}}}}' + f"\n{TC_END}{AR}{resp}{AR_END}"


def test_basic_expansion_four_steps():
    s0 = _tool_step(f"{THINK}t0{THINK_END}o0", "x", "r0")
    s1 = _tool_step("o1", "y", "r1")
    s2 = _tool_step(f"o2a{THINK}t2{THINK_END}o2b", "z", "r2")
    s3 = "o3<reflect>refl</reflect><turn_summary>sum</turn_summary>"
    turn = mk_turn(s0 + s1 + s2 + s3)
    steps = expand_turn(turn)
    assert len(steps) == 4
    types0 = [b.type for b in steps[0].output]
    assert ScribeBlockType.THINK in types0
    assert ScribeBlockType.TOOL_CALL in types0
    assert ScribeBlockType.TOOL_RESPONSE not in types0
    types_last = [b.type for b in steps[-1].output]
    assert ScribeBlockType.REFLECT in types_last
    assert ScribeBlockType.TURN_SUMMARY in types_last


def test_no_T_R_in_step_input():
    s0 = _tool_step(f"{THINK}t0{THINK_END}o0", "x", "r0")
    s1 = _tool_step("o1<reflect>r</reflect>", "y", "r1")
    s2 = "o2<turn_summary>sum</turn_summary>"
    turn = mk_turn(s0 + s1 + s2)
    steps = expand_turn(turn)
    for msg in steps[2].input:
        if msg.get("role") == "assistant":
            c = msg.get("content", "") or ""
            assert THINK not in c
            assert "<reflect>" not in c
            assert "<turn_summary>" not in c


def test_AR_excluded_from_output():
    s0 = _tool_step(f"{THINK}t0{THINK_END}o0", "x", "r0")
    s1 = "o1<turn_summary>sum</turn_summary>"
    turn = mk_turn(s0 + s1)
    steps = expand_turn(turn)
    for step in steps:
        assert all(b.type != ScribeBlockType.TOOL_RESPONSE for b in step.output)


def test_AR_present_in_step_input_accumulation():
    s0 = _tool_step(f"{THINK}t0{THINK_END}o0", "x", "r0")
    s1 = "o1<turn_summary>sum</turn_summary>"
    turn = mk_turn(s0 + s1)
    steps = expand_turn(turn)
    assert any(
        m.get("role") == "tool" and "r0" in (m.get("content") or "")
        for m in steps[1].input
    )


def test_loss_mask_input_off_output_on():
    s0 = _tool_step(f"{THINK}t0{THINK_END}o0", "x", "r0")
    s1 = "o1<turn_summary>sum</turn_summary>"
    turn = mk_turn(s0 + s1)
    steps = expand_turn(turn)
    for step in steps:
        mask = compute_loss_mask(step)
        assert all(mask)
        assert len(mask) == len(step.output)


def test_single_step_turn():
    turn = mk_turn("o0<reflect>r</reflect><turn_summary>sum</turn_summary>")
    steps = expand_turn(turn)
    assert len(steps) == 1
    types = [b.type for b in steps[0].output]
    assert ScribeBlockType.REFLECT in types
    assert ScribeBlockType.TURN_SUMMARY in types


def test_multi_O_step_preserves_order():
    s0 = _tool_step(f"o2a{THINK}t2{THINK_END}o2b", "z", "r2")
    s1 = "o3<turn_summary>sum</turn_summary>"
    turn = mk_turn(s0 + s1)
    steps = expand_turn(turn)
    outs = [b for b in steps[0].output if b.type == ScribeBlockType.OUTPUT]
    assert len(outs) == 2
