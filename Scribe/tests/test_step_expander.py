"""Unit tests for step_expander — turn -> list[Step] + loss mask.

step.input now comes from turn.step_records (推理实况, captured at inference),
NOT rebuilt. These tests build StepRecords the way the agent does: each step's
input_messages is the T/R-stripped accumulation, output_blocks is the raw
model-generated blocks (includes T, excludes AR).

Verifies:
  - step.output contains T (model-generated); excludes AR.
  - step.input (from 推理实况) has no T/R (stripped at capture).
  - last step output has R and S.
  - loss mask: output blocks on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Scribe.scribe_gym import TurnRecord, expand_turn, compute_loss_mask
from Scribe.scribe_gym.parsers import parse_scribe_blocks, render_scribe_blocks, ScribeBlockType
from Scribe.scribe_gym.turn_record import StepRecord

TC = chr(60) + "tool_call" + chr(62)
TC_END = chr(60) + "/tool_call" + chr(62)
AR = chr(60) + "tool_response" + chr(62)
AR_END = chr(60) + "/tool_response" + chr(62)
THINK = chr(60) + "think" + chr(62)
THINK_END = chr(60) + "/think" + chr(62)

SYS = [{"role": "system", "content": "SYS"}]


def _blocks(text):
    b, _ = parse_scribe_blocks(text)
    return b


def _strip_t_r(blocks):
    return [b for b in blocks if b.type not in (ScribeBlockType.THINK, ScribeBlockType.REFLECT)]


def _tool_msg(text, tc_id):
    """Build an assistant message dict (content + tool_calls) for a tool step.
    text must already include the  block in the content."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": tc_id, "type": "function",
                        "function": {"name": "x", "arguments": "{}"}}],
    }


def _final_msg(text):
    return {"role": "assistant", "content": text, "tool_calls": None}


def _tool_resp(tc_id, content):
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def build_turn(steps_script):
    """steps_script: list of either ("tool", asst_content, tc_id, resp_content)
    or ("final", asst_content). Builds new_blocks + step_records + retained the
    way the agent captures them (input T/R-stripped, output raw)."""
    step_records = []
    retained = []
    new_blocks = []
    msgs = list(SYS)
    for entry in steps_script:
        if entry[0] == "tool":
            _, content, tc_id, resp = entry
            # input for this step = current msgs (T/R stripped accumulation)
            input_msgs = [dict(m) for m in msgs]
            out_blocks = _blocks(content)
            step_records.append(StepRecord(
                input_messages=input_msgs,
                output_blocks=out_blocks,
                raw_assistant_message=_tool_msg(content, tc_id),
                tool_messages=[_tool_resp(tc_id, resp)],
            ))
            new_blocks.extend(out_blocks)
            new_blocks.extend(_blocks(AR + resp + AR_END))
            # accumulate: stripped assistant + tool resp
            stripped = _tool_msg(render_scribe_blocks(_strip_t_r(out_blocks)), tc_id)
            msgs.append(stripped)
            retained.append(stripped)
            msgs.append(_tool_resp(tc_id, resp))
            retained.append(_tool_resp(tc_id, resp))
        else:  # final
            _, content = entry
            input_msgs = [dict(m) for m in msgs]
            out_blocks = _blocks(content)
            step_records.append(StepRecord(
                input_messages=input_msgs,
                output_blocks=out_blocks,
                raw_assistant_message=_final_msg(content),
            ))
            new_blocks.extend(out_blocks)
            stripped = _final_msg(render_scribe_blocks(_strip_t_r(out_blocks)))
            msgs.append(stripped)
            retained.append(stripped)
    return TurnRecord(step_records=step_records)


def _tc_block(name="x"):
    return f'{TC}\n' + f'{{"name":"{name}","arguments":{{}}}}' + f'\n{TC_END}'


def test_basic_expansion_four_steps():
    turn = build_turn([
        ("tool", f"{THINK}t0{THINK_END}o0{_tc_block()}", "c0", "r0"),
        ("tool", f"o1{_tc_block()}", "c1", "r1"),
        ("tool", f"o2a{THINK}t2{THINK_END}o2b{_tc_block()}", "c2", "r2"),
        ("final", "o3<reflect>refl</reflect><turn_summary>sum</turn_summary>"),
    ])
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
    turn = build_turn([
        ("tool", f"{THINK}t0{THINK_END}o0{_tc_block()}", "c0", "r0"),
        ("tool", f"o1<reflect>r</reflect>{_tc_block()}", "c1", "r1"),
        ("final", "o2<turn_summary>sum</turn_summary>"),
    ])
    steps = expand_turn(turn)
    for msg in steps[2].input:
        if msg.get("role") == "assistant":
            c = msg.get("content", "") or ""
            assert THINK not in c
            assert "<reflect>" not in c


def test_AR_excluded_from_output():
    turn = build_turn([
        ("tool", f"{THINK}t0{THINK_END}o0{_tc_block()}", "c0", "r0"),
        ("final", "o1<turn_summary>sum</turn_summary>"),
    ])
    steps = expand_turn(turn)
    for step in steps:
        assert all(b.type != ScribeBlockType.TOOL_RESPONSE for b in step.output)


def test_AR_present_in_step_input():
    turn = build_turn([
        ("tool", f"{THINK}t0{THINK_END}o0{_tc_block()}", "c0", "r0"),
        ("final", "o1<turn_summary>sum</turn_summary>"),
    ])
    steps = expand_turn(turn)
    assert any(
        m.get("role") == "tool" and "r0" in (m.get("content") or "")
        for m in steps[1].input
    )


def test_loss_mask_output_on():
    turn = build_turn([
        ("tool", f"{THINK}t0{THINK_END}o0{_tc_block()}", "c0", "r0"),
        ("final", "o1<turn_summary>sum</turn_summary>"),
    ])
    steps = expand_turn(turn)
    for step in steps:
        mask = compute_loss_mask(step)
        assert all(mask)
        assert len(mask) == len(step.output)


def test_single_step_turn():
    turn = build_turn([
        ("final", "o0<reflect>r</reflect><turn_summary>sum</turn_summary>"),
    ])
    steps = expand_turn(turn)
    assert len(steps) == 1
    types = [b.type for b in steps[0].output]
    assert ScribeBlockType.REFLECT in types
    assert ScribeBlockType.TURN_SUMMARY in types


def test_multi_O_step_preserves_order():
    turn = build_turn([
        ("tool", f"o2a{THINK}t2{THINK_END}o2b{_tc_block()}", "c2", "r2"),
        ("final", "o3<turn_summary>sum</turn_summary>"),
    ])
    steps = expand_turn(turn)
    outs = [b for b in steps[0].output if b.type == ScribeBlockType.OUTPUT]
    assert len(outs) == 2


def test_empty_turn_expands_to_zero_steps():
    """A TurnRecord with no step_records expands to zero steps (single source
    of truth = step_records; no legacy rebuild path)."""
    turn = TurnRecord(step_records=[])
    assert expand_turn(turn) == []
