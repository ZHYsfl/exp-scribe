"""Unit tests for rewards.py — the 16 turn-level reward metrics.

Builds TurnRecords the same way test_step_expander does (parse_scribe_blocks
over scripted content). Injects a FakeCounter (chars as pseudo-tokens) so no
real tokenizer is needed, and a FakeJudge for metrics 7-10. Each test focuses
on one metric for easy attribution.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Scribe.scribe_gym import (
    DEFAULT_TURN_REWARD_CONFIG,
    ScribeBlockType,
    SummaryJudgeScores,
    TurnOutcome,
    TurnRecord,
    TurnRewardConfig,
    build_judge_prompt,
    compute_turn_reward,
)
from Scribe.scribe_gym.parsers import parse_scribe_blocks
from Scribe.scribe_gym.rewards import (
    metric_1, metric_2, metric_3, metric_4, metric_5, metric_6,
    metric_11, metric_12, metric_13, metric_14, metric_15, metric_16, metrics_7_10,
)
from Scribe.scribe_gym.turn_record import StepRecord

TC = chr(60) + "tool_call" + chr(62)
TC_END = chr(60) + "/tool_call" + chr(62)
THINK = chr(60) + "think" + chr(62)
THINK_END = chr(60) + "/think" + chr(62)
SYS = [{"role": "system", "content": "SYS"}]


class FakeCounter:
    def count(self, text):
        return len(text)

    def count_observations(self, obs):
        return sum(self.count(m.get("content") or "") for m in obs)

    def encode(self, text):
        return [ord(ch) for ch in text]


class FakeJudge:
    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def __call__(self, ground_truth, summary):
        self.calls += 1
        return self._scores


def _tc(name="submit", arguments=None):
    args = arguments if arguments is not None else {}
    import json
    return f'{TC}\n' + json.dumps({"name": name, "arguments": args}) + f'\n{TC_END}'


def _blocks(text):
    b, _ = parse_scribe_blocks(text)
    return b


def _msg(content, tc_id=None):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": tc_id, "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}] if tc_id else None}


def build_turn(steps, rights=None):
    """steps: list of ("tool", content, tc_id, resp) or ("final", content).
    rights: optional list of tool_messages contents, one per tool step."""
    step_records = []
    msgs = list(SYS)
    for entry in steps:
        if entry[0] == "tool":
            _, content, tc_id, resp = entry
            out = _blocks(content)
            step_records.append(StepRecord(
                input_messages=[dict(m) for m in msgs],
                output_blocks=out,
                raw_assistant_message=_msg(content, tc_id),
                tool_messages=[{"role": "tool", "tool_call_id": tc_id, "content": resp}],
            ))
            msgs.append(_msg(content))
            msgs.append({"role": "tool", "tool_call_id": tc_id, "content": resp})
        else:
            _, content = entry
            out = _blocks(content)
            step_records.append(StepRecord(
                input_messages=[dict(m) for m in msgs],
                output_blocks=out,
                raw_assistant_message=_msg(content),
            ))
    return TurnRecord(step_records=step_records)


def _oc(**kw):
    base = dict(truncated=False, terminated=True, done_reason="no_tool_call",
                answer="42", right_answer="42", submit_count=1)
    base.update(kw)
    return TurnOutcome(**base)


# ---- metric 1: answer + termination ----
def test_m1_correct_natural():
    # metric_1 no longer checks block content; only termination + correctness matter
    turn = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    assert metric_1(turn, _oc(), DEFAULT_TURN_REWARD_CONFIG) == 1.0


def test_m1_wrong_answer():
    turn = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    oc = _oc(answer="99")
    assert metric_1(turn, oc, DEFAULT_TURN_REWARD_CONFIG) == 0.0


def test_m1_truncated_penalty():
    turn = build_turn([("tool", "o" + _tc(), "c0", "r"), ("tool", "o2", "c1", "r2")])
    oc = _oc(truncated=True, terminated=False, done_reason=None, answer="42")
    truncated_score = metric_1(turn, oc, DEFAULT_TURN_REWARD_CONFIG)
    # correctness=1 but no natural termination -> raw=0; truncated flag makes raw=-0.2
    # affine-rescaled to [0,1] -> 0.0
    assert 0.0 <= truncated_score < 0.1
    # non-truncated without natural termination also gets 0
    non_trunc = metric_1(turn, _oc(truncated=False, terminated=False, done_reason=None, answer="42"),
                         DEFAULT_TURN_REWARD_CONFIG)
    assert truncated_score == non_trunc == 0.0


def test_m1_terminated_by_tool_call_no_bonus():
    # terminated but done_reason != no_tool_call -> no termination bonus
    turn = build_turn([("tool", "o" + _tc(), "c0", "r"), ("final", "o<turn_summary>s</turn_summary>")])
    oc = _oc(terminated=True, done_reason="other")
    assert metric_1(turn, oc, DEFAULT_TURN_REWARD_CONFIG) == 0.0


# ---- metric 2: tool usage ----
def test_m2_submit_once_no_dup():
    turn = build_turn([
        ("tool", "o" + _tc("bash", {"command": "echo 1"}), "c0", "r"),
        ("tool", "o" + _tc("submit", {"answer": "42"}), "c1", "rs"),
        ("final", "o<reflect>rf</reflect><turn_summary>s</turn_summary>"),
    ])
    oc = _oc(submit_count=1)
    assert metric_2(turn, oc, DEFAULT_TURN_REWARD_CONFIG) == 1.0


def test_m2_double_submit_penalized():
    turn = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    oc = _oc(submit_count=2)
    assert metric_2(turn, oc, DEFAULT_TURN_REWARD_CONFIG) < 1.0


def test_m2_parallel_identical_penalized():
    # one assistant message with two identical bash calls
    content = "o" + _tc("bash", {"command": "ls"}) + _tc("bash", {"command": "ls"})
    turn = build_turn([
        ("tool", content, "c0", "r"),
        ("final", "o<turn_summary>s</turn_summary>"),
    ])
    oc = _oc(submit_count=1)
    assert metric_2(turn, oc, DEFAULT_TURN_REWARD_CONFIG) < 1.0
    # a clean turn with the same number of steps/tool-calls should score higher
    clean = build_turn([
        ("tool", "o" + _tc("bash", {"command": "ls"}), "c1", "r"),
        ("final", "o<turn_summary>s</turn_summary>"),
    ])
    assert metric_2(clean, oc, DEFAULT_TURN_REWARD_CONFIG) > metric_2(turn, oc, DEFAULT_TURN_REWARD_CONFIG)


# ---- metric 3: step length ----
def test_m3_fewer_steps_higher():
    turn1 = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    turn3 = build_turn([
        ("tool", "o" + _tc(), "c0", "r"),
        ("tool", "o" + _tc(), "c1", "r2"),
        ("final", "o<turn_summary>s</turn_summary>"),
    ])
    assert metric_3(turn1, DEFAULT_TURN_REWARD_CONFIG) == 1.0
    assert metric_3(turn3, DEFAULT_TURN_REWARD_CONFIG) == 1.0 / 3.0


# ---- metric 4: format ----
def test_m4_perfect_format():
    turn = build_turn([
        ("tool", "<think>t</think>o" + _tc("bash", {"command": "ls"}), "c0", "r"),
        ("tool", "o" + _tc("submit", {"answer": "42"}), "c1", "ok"),
        ("final", "<think>t2</think>o<reflect>rf</reflect><turn_summary>s</turn_summary>"),
    ])
    assert metric_4(turn, _oc(submit_count=1), DEFAULT_TURN_REWARD_CONFIG) == 1.0


def test_m4_all_expected_block_types():
    # perfect: T/O/A/AR/R/S all present, with submit before R+S
    perfect = build_turn([
        ("tool", "<think>t</think>o" + _tc("bash", {"command": "ls"}), "c0", "r"),
        ("tool", "o" + _tc("submit", {"answer": "42"}), "c1", "ok"),
        ("final", "<think>t2</think>o<reflect>rf</reflect><turn_summary>s</turn_summary>"),
    ])
    assert metric_4(perfect, _oc(submit_count=1), DEFAULT_TURN_REWARD_CONFIG) == 1.0

    # missing think only: one type missing -> one small penalty
    no_think = build_turn([
        ("tool", "o" + _tc("bash", {"command": "ls"}), "c0", "r"),
        ("tool", "o" + _tc("submit", {"answer": "42"}), "c1", "ok"),
        ("final", "o<reflect>rf</reflect><turn_summary>s</turn_summary>"),
    ])
    assert metric_4(no_think, _oc(submit_count=1), DEFAULT_TURN_REWARD_CONFIG) == 0.92

    # missing output in a tool-only turn (no final step) -> multiple missing types
    tool_only = build_turn([
        ("tool", "<think>t</think>" + _tc("bash", {"command": "ls"}), "c0", "r"),
    ])
    assert metric_4(tool_only, _oc(submit_count=0), DEFAULT_TURN_REWARD_CONFIG) < 1.0


def test_m4_reflect_in_nonfinal_step():
    content = "o<reflect>early</reflect>" + _tc("bash", {"command": "ls"})
    turn = build_turn([
        ("tool", content, "c0", "r"),
        ("final", "o<turn_summary>s</turn_summary>"),
    ])
    assert metric_4(turn, _oc(), DEFAULT_TURN_REWARD_CONFIG) < 1.0


def test_m4_summary_without_submit():
    turn = build_turn([("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>")])
    oc = _oc(submit_count=0)
    assert metric_4(turn, oc, DEFAULT_TURN_REWARD_CONFIG) < 1.0


def test_m4_disallowed_tag_penalized():
    turn = build_turn([("final", "o<submit>x</submit><reflect>r</reflect><turn_summary>s</turn_summary>")])
    assert metric_4(turn, _oc(submit_count=1), DEFAULT_TURN_REWARD_CONFIG) < 1.0


def test_m4_closing_disallowed_tag_penalized():
    turn = build_turn([("final", "o</bash><reflect>r</reflect><turn_summary>s</turn_summary>")])
    assert metric_4(turn, _oc(submit_count=1), DEFAULT_TURN_REWARD_CONFIG) < 1.0


def test_m4_rs_before_submit_penalized():
    # R+S appears before submit call: invalid ordering
    turn = build_turn([
        ("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>"),
        ("tool", "o" + _tc("submit", {"answer": "42"}), "c1", "ok"),
    ])
    assert metric_4(turn, _oc(submit_count=1), DEFAULT_TURN_REWARD_CONFIG) < 1.0


# ---- metric 5: rollout tokens ----
def test_m5_shorter_higher():
    short = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    long_t = build_turn([("final", "o" * 2000 + "<turn_summary>s</turn_summary>")])
    c = FakeCounter()
    cfg = DEFAULT_TURN_REWARD_CONFIG
    assert metric_5(short, c, cfg) > metric_5(long_t, c, cfg)
    # soft decay: even a 2000-token rollout still has non-zero signal
    assert metric_5(long_t, c, cfg) > 0.0


# ---- metric 6: token reuse ----
def test_m6_reuse_positive():
    # summary reuses chars from the output block
    turn = build_turn([("final", "alpha beta<turn_summary>alpha</turn_summary>")])
    c = FakeCounter()
    assert metric_6(turn, c, DEFAULT_TURN_REWARD_CONFIG) > 0.0


def test_m6_no_reuse():
    turn = build_turn([("final", "zzzzz<turn_summary>qqqqq</turn_summary>")])
    c = FakeCounter()
    assert metric_6(turn, c, DEFAULT_TURN_REWARD_CONFIG) == 0.0


# ---- metrics 7-10: judge ----
def test_judge_injected():
    turn = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    j = FakeJudge(SummaryJudgeScores(0.9, 0.8, 0.7, 0.6))
    m7, m8, m9, m10, used = asyncio.run(
        metrics_7_10(turn, j, DEFAULT_TURN_REWARD_CONFIG)
    )
    assert (m7, m8, m9, m10) == (0.9, 0.8, 0.7, 0.6)
    assert used and j.calls == 1


def test_judge_async_injected():
    class AsyncFakeJudge:
        def __init__(self, scores):
            self._scores = scores
            self.calls = 0
        async def __call__(self, ground_truth, summary):
            self.calls += 1
            return self._scores

    turn = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    j = AsyncFakeJudge(SummaryJudgeScores(0.9, 0.8, 0.7, 0.6))
    m7, m8, m9, m10, used = asyncio.run(
        metrics_7_10(turn, j, DEFAULT_TURN_REWARD_CONFIG)
    )
    assert (m7, m8, m9, m10) == (0.9, 0.8, 0.7, 0.6)
    assert used and j.calls == 1


def test_judge_none_defaults():
    turn = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    m7, m8, m9, m10, used = asyncio.run(
        metrics_7_10(turn, None, DEFAULT_TURN_REWARD_CONFIG)
    )
    assert not used
    assert (m7, m8, m9, m10) == (0.5, 0.5, 0.5, 0.5)


def test_judge_prompt_builds_two_messages():
    msgs = build_judge_prompt("GT", "SUM")
    assert len(msgs) == 2
    assert "GT" in msgs[1]["content"] and "SUM" in msgs[1]["content"]


# ---- metric 11: compression ratio ----
def test_m11_short_summary_high():
    turn = build_turn([("final", "abcdefghij<turn_summary>ab</turn_summary>")])
    c = FakeCounter()
    assert metric_11(turn, c, DEFAULT_TURN_REWARD_CONFIG) > 0.5


def test_m11_verbose_summary_low():
    turn = build_turn([("final", "ab<turn_summary>abcdefghij</turn_summary>")])
    c = FakeCounter()
    assert metric_11(turn, c, DEFAULT_TURN_REWARD_CONFIG) < 0.5


# ---- metric 12: cross-block n-gram overlap ----
def test_m12_verbatim_copy_penalized():
    # summary copies the whole output verbatim -> max overlap -> low score
    gt = "the quick brown fox jumps over the lazy dog here"
    turn = build_turn([("final", f"{gt}<turn_summary>{gt}</turn_summary>")])
    c = FakeCounter()
    assert metric_12(turn, c, DEFAULT_TURN_REWARD_CONFIG) < 0.1


def test_m12_rephrased_high():
    turn = build_turn([("final", "zebra stripes<turn_summary>totally different</turn_summary>")])
    c = FakeCounter()
    assert metric_12(turn, c, DEFAULT_TURN_REWARD_CONFIG) > 0.5


# ---- metric 13: intra-block overlap ----
def test_m13_no_repetition_high():
    text = "The quick brown fox jumps over the lazy dog and then runs away"
    turn = build_turn([("final", f"{text}<turn_summary>{text[::-1]}</turn_summary>")])
    c = FakeCounter()
    assert metric_13(turn, c, DEFAULT_TURN_REWARD_CONFIG) > 0.9


def test_m13_repetition_low():
    rep = "again " * 30
    turn = build_turn([("final", f"o<reflect>{rep}</reflect><turn_summary>s</turn_summary>")])
    c = FakeCounter()
    assert metric_13(turn, c, DEFAULT_TURN_REWARD_CONFIG) < 0.2


# ---- metrics 14-15 ----
def test_m14_no_malformed():
    turn = build_turn([("tool", "o" + _tc("bash", {"command": "ls"}), "c0", "ok"),
                       ("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>")])
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="42", right_answer="42", submit_count=1,
                     malformed_tool_calls=0, repeated_tool_calls=0)
    assert metric_14(turn, oc, DEFAULT_TURN_REWARD_CONFIG) == 1.0


def test_m14_malformed_penalty():
    turn = build_turn([("tool", "o" + _tc("bash", {"command": "ls"}), "c0", "ok"),
                       ("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>")])
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="42", right_answer="42", submit_count=1,
                     malformed_tool_calls=2, repeated_tool_calls=0)
    cfg = DEFAULT_TURN_REWARD_CONFIG
    assert metric_14(turn, oc, cfg) == 1.0 - 2 * cfg.malformed_call_penalty


def test_m15_no_repeat():
    turn = build_turn([
        ("tool", "o" + _tc("bash", {"command": "ls"}), "c0", "ok"),
        ("tool", "o" + _tc("bash", {"command": "pwd"}), "c1", "ok"),
        ("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>")])
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="42", right_answer="42", submit_count=1,
                     malformed_tool_calls=0, repeated_tool_calls=0)
    assert metric_15(turn, oc, DEFAULT_TURN_REWARD_CONFIG) == 1.0


def test_m15_repeated_penalty():
    turn = build_turn([
        ("tool", "o" + _tc("bash", {"command": "ls"}), "c0", "ok"),
        ("tool", "o" + _tc("bash", {"command": "ls"}), "c1", "ok"),
        ("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>")])
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="42", right_answer="42", submit_count=1,
                     malformed_tool_calls=0, repeated_tool_calls=1)
    cfg = DEFAULT_TURN_REWARD_CONFIG
    assert metric_15(turn, oc, cfg) == 1.0 - cfg.repeated_call_penalty


def _submit_turn(answer: str = "42") -> TurnRecord:
    return build_turn([
        ("tool", "o" + _tc("submit", {"answer": answer}), "c0", "ok"),
        ("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>")])


def test_m16_no_prev_turn():
    turn = _submit_turn("42")
    turn.prev_turn_answer = None
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="42", right_answer="42", submit_count=1)
    assert metric_16(turn, oc, DEFAULT_TURN_REWARD_CONFIG) == 1.0


def test_m16_same_answer_penalty():
    turn = _submit_turn("42")
    turn.prev_turn_answer = "42"
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="42", right_answer="42", submit_count=1)
    cfg = DEFAULT_TURN_REWARD_CONFIG
    assert metric_16(turn, oc, cfg) == 1.0 - cfg.cross_turn_dup_penalty


def test_m16_different_answer_no_penalty():
    turn = _submit_turn("42")
    turn.prev_turn_answer = "41"
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="42", right_answer="42", submit_count=1)
    assert metric_16(turn, oc, DEFAULT_TURN_REWARD_CONFIG) == 1.0


def test_m16_no_submit_no_penalty():
    turn = _submit_turn("42")
    turn.prev_turn_answer = "42"
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="42", right_answer="42", submit_count=0)
    assert metric_16(turn, oc, DEFAULT_TURN_REWARD_CONFIG) == 1.0


def test_m16_whitespace_ignored():
    turn = _submit_turn("  42 ")
    turn.prev_turn_answer = "42"
    oc = TurnOutcome(truncated=False, terminated=True, done_reason="no_tool_call",
                     answer="  42 ", right_answer="42", submit_count=1)
    cfg = DEFAULT_TURN_REWARD_CONFIG
    assert metric_16(turn, oc, cfg) == 1.0 - cfg.cross_turn_dup_penalty


def test_final_submit_answer_returns_last_submit():
    turn = build_turn([
        ("tool", "o" + _tc("submit", {"answer": "first"}), "c0", "ok"),
        ("tool", "o" + _tc("submit", {"answer": "second"}), "c1", "ok"),
        ("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>")])
    assert turn.final_submit_answer == "second"


def test_final_submit_answer_none_when_no_submit():
    turn = build_turn([
        ("final", "o<reflect>r</reflect><turn_summary>s</turn_summary>")])
    assert turn.final_submit_answer is None


# ---- aggregation ----
def test_compute_turn_reward_clamps_and_weights():
    turn = build_turn([
        ("tool", "<think>t</think>o" + _tc("bash", {"command": "ls"}), "c0", "r"),
        ("tool", "o" + _tc("submit", {"answer": "42"}), "c1", "ok"),
        ("final", "<think>t2</think>o<reflect>rf</reflect><turn_summary>s</turn_summary>"),
    ])
    bd = asyncio.run(compute_turn_reward(
        turn, DEFAULT_TURN_REWARD_CONFIG,
        outcome=_oc(submit_count=1), judge=None, token_counter=FakeCounter(),
    ))
    assert 0.0 <= bd.total <= 1.0
    assert not bd.judge_used
    ws = [DEFAULT_TURN_REWARD_CONFIG.w1, DEFAULT_TURN_REWARD_CONFIG.w2,
          DEFAULT_TURN_REWARD_CONFIG.w3, DEFAULT_TURN_REWARD_CONFIG.w4,
          DEFAULT_TURN_REWARD_CONFIG.w5, DEFAULT_TURN_REWARD_CONFIG.w6,
          DEFAULT_TURN_REWARD_CONFIG.w7, DEFAULT_TURN_REWARD_CONFIG.w8,
          DEFAULT_TURN_REWARD_CONFIG.w9, DEFAULT_TURN_REWARD_CONFIG.w10,
          DEFAULT_TURN_REWARD_CONFIG.w11, DEFAULT_TURN_REWARD_CONFIG.w12,
          DEFAULT_TURN_REWARD_CONFIG.w13, DEFAULT_TURN_REWARD_CONFIG.w14,
          DEFAULT_TURN_REWARD_CONFIG.w15, DEFAULT_TURN_REWARD_CONFIG.w16]
    assert abs(bd.total - sum(w * v for w, v in zip(ws, bd.metrics))) < 1e-9


def test_compute_turn_reward_judge_marks_used():
    turn = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    j = FakeJudge(SummaryJudgeScores(1.0, 1.0, 1.0, 1.0))
    bd = asyncio.run(compute_turn_reward(
        turn, DEFAULT_TURN_REWARD_CONFIG,
        outcome=_oc(submit_count=1), judge=j, token_counter=FakeCounter(),
    ))
    assert bd.judge_used
    assert bd.metric_7 == 1.0


def test_compute_turn_reward_empty_turn():
    turn = TurnRecord(step_records=[])
    bd = asyncio.run(compute_turn_reward(
        turn, DEFAULT_TURN_REWARD_CONFIG,
        outcome=_oc(submit_count=0, terminated=False, done_reason=None, answer=None),
        token_counter=FakeCounter(),
    ))
    assert bd.total == 0.0


# ---- wiring sanity: TurnRecord.reward_breakdown settable ----
def test_record_reward_breakdown_field():
    turn = build_turn([("final", "o<turn_summary>s</turn_summary>")])
    bd = asyncio.run(compute_turn_reward(
        turn, DEFAULT_TURN_REWARD_CONFIG,
        outcome=_oc(submit_count=1), token_counter=FakeCounter(),
    ))
    turn.reward = bd.total
    turn.reward_breakdown = bd
    assert turn.reward == bd.total
    assert turn.reward_breakdown is bd