"""SCRIBE 13 turn-level reward metrics (pure functions).

Given a TurnRecord (the turn's parsed blocks, single source of truth) and a
TurnOutcome (termination/answer info pulled from the env trajectory at
finalization time), compute the 13 README metrics, weighted-sum them into one
turn reward, and return a full breakdown for debugging / ablation.

Design (my_project_philosophy.md): pure functions, no agent/env dependency,
deterministic. The LLM-judge (metrics 7-10) is an injected callable returning a
SummaryJudgeScores; judge=None -> neutral defaults (keeps RL batching and unit
tests unblocked). Token counting goes through a TokenCounter abstraction
(DeepSeekTokenCounter by default).

Metric scores are normalized to [0,1] (higher=better) or carry README's signed
penalties; the weighted sum is clamped to [0,1]. See README "Second" section.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .parsers import ScribeBlock, ScribeBlockType, parse_scribe_blocks
from .turn_record import TurnRecord

_T = ScribeBlockType.THINK
_O = ScribeBlockType.OUTPUT
_A = ScribeBlockType.TOOL_CALL
_AR = ScribeBlockType.TOOL_RESPONSE
_R = ScribeBlockType.REFLECT
_S = ScribeBlockType.TURN_SUMMARY

# Blocks eligible for intra-block n-gram (metric 13): the model-generated
# free-text blocks. TOOL_CALL is excluded because its structured JSON is
# naturally repetitive (not a degenerate loop); TOOL_RESPONSE is env-produced,
# not the model's responsibility. NOTE: TOOL_CALL IS model-generated — the
# exclusion is about repetition structure, not authorship.
_INTRA_ELG_BLOCKS = (_T, _O, _R, _S)

# Allowed SCRIBE tags for metric 4e.
_ALLOWED_TAGS = {"think", "tool_call", "tool_response", "reflect", "turn_summary"}
_TAG_RE = re.compile(r"<\s*/?\s*([A-Za-z_][A-Za-z0-9_]*)\s*>")


def _contains_disallowed_tag(text: str) -> bool:
    """True if text contains any tag that is not one of the allowed SCRIBE tags."""
    for m in _TAG_RE.finditer(text):
        if m.group(1).lower() not in _ALLOWED_TAGS:
            return True
    return False


@dataclass
class TurnOutcome:
    """Termination/answer facts for this turn, lifted from the env trajectory
    at finalization. Kept out of TurnRecord so the reward module stays a pure
    function of (turn, outcome, config) with no agent/env import."""

    truncated: bool
    terminated: bool
    done_reason: Optional[str]
    answer: Optional[str]
    right_answer: Optional[str]
    submit_count: int


@dataclass
class SummaryJudgeScores:
    """The 4 LLM-judge sub-scores (metrics 7-10), each in [0,1]."""

    faithfulness: float
    direction_neutrality: float
    turn_focus: float
    fluency: float


# A judge takes (ground_truth_text, summary_text) -> SummaryJudgeScores.
Judge = Callable[[str, str], SummaryJudgeScores]


@dataclass
class TurnRewardConfig:
    # weights
    w1: float = 0.25   # answer + termination
    w2: float = 0.10   # tool usage (submit + parallel dup)
    w3: float = 0.05   # step length
    w4: float = 0.15   # format
    w5: float = 0.02   # rollout tokens
    w6: float = 0.15   # token reuse ratio
    w7: float = 0.05   # faithfulness (judge)
    w8: float = 0.05   # direction neutrality (judge)
    w9: float = 0.05    # turn focus (judge)
    w10: float = 0.05  # fluency (judge)
    w11: float = 0.03  # compression ratio
    w12: float = 0.03  # cross-block n-gram overlap
    w13: float = 0.02  # intra-block n-gram overlap
    # penalty magnitudes
    truncation_penalty: float = 0.20
    submit_penalty: float = 0.20
    parallel_dup_penalty: float = 0.20
    # format violation penalties
    fmt_missing_type: float = 0.20
    fmt_bad_order: float = 0.20
    fmt_rs_in_nonfinal: float = 0.30
    fmt_tool_in_final: float = 0.30
    fmt_rs_without_submit: float = 0.20
    fmt_disallowed_tag: float = 0.30
    fmt_invalid_share_weight: float = 0.30
    # m5 rollout-token reference length (linear bridge -> [0,1])
    rollout_token_ref: int = 1024
    # n-gram size
    n_ngram: int = 4
    # judge default when judge is None
    judge_default: float = 0.5
    # final clamp — full reward is normalized to [0,1] per spec.
    total_low: float = 0.0
    total_high: float = 1.0


DEFAULT_TURN_REWARD_CONFIG = TurnRewardConfig()


@dataclass
class TurnRewardBreakdown:
    metric_1: float = 0.0
    metric_2: float = 0.0
    metric_3: float = 0.0
    metric_4: float = 0.0
    metric_5: float = 0.0
    metric_6: float = 0.0
    metric_7: float = 0.0
    metric_8: float = 0.0
    metric_9: float = 0.0
    metric_10: float = 0.0
    metric_11: float = 0.0
    metric_12: float = 0.0
    metric_13: float = 0.0
    total: float = 0.0
    judge_used: bool = False

    @property
    def metrics(self) -> List[float]:
        return [
            self.metric_1, self.metric_2, self.metric_3, self.metric_4,
            self.metric_5, self.metric_6, self.metric_7, self.metric_8,
            self.metric_9, self.metric_10, self.metric_11, self.metric_12,
            self.metric_13,
        ]

    @property
    def weights(self) -> List[float]:
        c = DEFAULT_TURN_REWARD_CONFIG
        return [c.w1, c.w2, c.w3, c.w4, c.w5, c.w6,
                c.w7, c.w8, c.w9, c.w10, c.w11, c.w12, c.w13]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _blocks_of(turn: TurnRecord, btype: ScribeBlockType) -> List[ScribeBlock]:
    return [b for b in turn.new_blocks if b.type == btype]


def _ngrams(tokens, n: int) -> List[Tuple[Any, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _last_summary(turn: TurnRecord) -> str:
    s_blocks = _blocks_of(turn, _S)
    if not s_blocks:
        return ""
    # inner text of the <turn_summary> tag
    c = s_blocks[-1].content
    lo, hi = "<turn_summary>", "</turn_summary>"
    if c.startswith(lo) and c.endswith(hi):
        return c[len(lo):-len(hi)]
    return c


def _ground_truth_text(turn: TurnRecord) -> str:
    parts = [b.content for b in turn.new_blocks if b.type in (_O, _A)]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def metric_1(turn: TurnRecord, oc: TurnOutcome, cfg: TurnRewardConfig) -> float:
    """Metric 1 — final-step reward (answer × natural termination × truncation).

    README 1: = correctness(1a) * natural-termination-bonus(1b) + truncation-penalty(1c).
      1a answer correctness: 1.0 if answer==right_answer (strip), else 0.0. Spec
         says this factor is binary — left as 0/1, NOT staircase.
      1b natural termination bonus: README says the turn "must end with a final
         non-tool LLM call". This is signalled by the environment via
         terminated && done_reason=="no_tool_call". We intentionally do NOT
         inspect block content (O/R/S presence) here — that is metric_4's job;
         mixing format checks into metric_1 would break ablations.
      1c truncated penalty: -cfg.truncation_penalty (e.g. -0.2) when truncated.
         The MECHANISM is unchanged — correctness*termination produces its
         natural product, and truncation is a real overshoot below 0. Then the
         per-metric value is affine-rescaled to [0,1] so the RL-facing metric
         (and the weighted total) stay in spec range without needing the outer
         clamp to wipe the negative signal away. lo = -truncation_penalty.
    Per-metric output rescaled to [0,1] in-metric. A correct answer from a
    naturally terminated turn -> 1.0; truncation maps toward (but not to) 0,
    strictly below any non-truncated score.
    Note: env's repeated-submit penalty is NOT re-counted here (see metric 2a).
    """
    if oc.answer is None:
        answer_correct = False
    elif oc.right_answer is None:
        answer_correct = True  # no ground truth -> grant correctness
    else:
        answer_correct = oc.answer.strip() == oc.right_answer.strip()
    correctness = 1.0 if answer_correct else 0.0
    if correctness == 0.0 or not turn.step_records:
        # 1(a) is binary: wrong answer -> 0 reward regardless of format.
        # Empty turn also has no meaningful final-step signal.
        return 0.0
    # 1(b): natural termination bonus. README says the turn "must end with a
    # final non-tool LLM call". The env signals this via terminated &&
    # done_reason == "no_tool_call". We deliberately do NOT re-check block
    # content here (O/R/S presence) because that belongs to metric_4; keeping
    # metric_1 free of format details makes ablations clean.
    natural = (
        oc.terminated
        and not oc.truncated
        and oc.done_reason == "no_tool_call"
    )
    if not natural:
        return 0.0
    termination = 1.0
    # 1(c): MECHANISM unchanged — correctness*termination gives its natural
    # product, truncation is a real overshoot below 0. raw spans
    # [-truncation_penalty, 1.0]. We then AFFINE-RESCALE to [0,1] IN-METRIC so
    # the RL-facing value lives in spec range by construction.
    raw = correctness * termination
    if oc.truncated:
        raw -= cfg.truncation_penalty
    lo = -cfg.truncation_penalty
    hi = 1.0
    return _clamp((raw - lo) / (hi - lo), 0.0, 1.0)


def _tool_call_norm(b: ScribeBlock) -> Optional[Tuple[str, str]]:
    p = b.parsed
    if not isinstance(p, dict):
        return None
    name = p.get("name", "")
    args = p.get("arguments", {})
    return name, json.dumps(args, sort_keys=True)


def metric_2(turn: TurnRecord, oc: TurnOutcome, cfg: TurnRewardConfig) -> float:
    """Metric 2 — tool-usage quality in this turn's rollout blocks.

    README 2:
      2a submit count: submit must be called exactly once per turn.
         ==1 -> +1.0; otherwise -cfg.submit_penalty per unit of deviation.
      2b parallel identical tool calls: within ONE assistant message, two
         tool_calls with the same name AND normalized arguments (json.dumps,
         sort_keys=True) are duplicates. Each extra copy incurs
         -cfg.parallel_dup_penalty. Cross-step repetitions are NOT penalized
         (env state may have changed: run -> edit -> run).
    Mechanism uses signed penalties, then affine-rescaled to [0,1] so the
    RL-facing metric stays in spec range while preserving ordering.
    """
    submit_count = oc.submit_count

    # 2a: submit exactly once
    if submit_count == 1:
        m2a = 1.0
    else:
        m2a = 1.0 - cfg.submit_penalty * max(1, abs(submit_count - 1))

    # 2b: parallel identical tool calls within ONE assistant message (cross-step NOT penalized)
    dup_count = 0
    for rec in turn.step_records:
        groups: Dict[Tuple[str, str], int] = {}
        for b in rec.output_blocks:
            if b.type != _A:
                continue
            key = _tool_call_norm(b)
            if key is None:
                continue
            groups[key] = groups.get(key, 0) + 1
        dup_count += sum(c - 1 for c in groups.values() if c > 1)
    m2b = -cfg.parallel_dup_penalty * dup_count

    raw = m2a + m2b
    # worst case: no submit and many dups; raw can go arbitrarily negative.
    # rescale against the natural range [1 - submit_penalty - many_dups, 1].
    # For a single-dup or single-submit-miss, the reference low is symmetric.
    lo = min(1.0 - cfg.submit_penalty, 1.0 - cfg.parallel_dup_penalty, raw)
    hi = 1.0
    return _clamp((raw - lo) / (hi - lo), 0.0, 1.0)


def metric_3(turn: TurnRecord, cfg: TurnRewardConfig) -> float:
    """Metric 3 — length of the steps used (count of TOOL_CALL/A blocks).

    README 3: keep metric 1's level while making metric 3 as low as possible.
    Fewer tool-call steps -> higher score. Score = 1/(1+n_a), range [0,1].
    """
    n_a = sum(1 for b in turn.new_blocks if b.type == _A)
    return _clamp(1.0 / (1.0 + n_a), 0.0, 1.0)


def metric_4(turn: TurnRecord, oc: TurnOutcome, cfg: TurnRewardConfig) -> float:
    """Metric 4 — format correctness of this turn.

    README 4 sub-rules (each violation subtracts a cfg penalty, clamped [0,1]):
      (a) expected block types present (think, output, tool_call, tool_response,
          reflect, turn_summary). Each missing type in the turn's new_blocks is
          a separate violation.
      (b) final non-tool step ends with exactly <reflect> then <turn_summary>
          (last two blocks in order R, S).
      (c) REFLECT/TURN_SUMMARY appear only in the final step.
      (d) final non-tool step contains NO tool_calls.
      (e) only the agreed-upon SCRIBE tags are allowed — any tag other than
          <think>, <tool_call>, <tool_response>, <reflect>, <turn_summary> is
          invalid. Penalized once per turn if any step's raw content contains a
          disallowed tag (open or close).
      (f) R+S in final step implies submit was called this turn.
      (g) invalid share: per-step parse_scribe_blocks is_valid flag (covers
          malformed/nested tags, bad tool_call JSON, stray known tags in wrong
          places). Penalty scales with the share of steps that failed to parse
          cleanly.
    Perfect format -> 1.0.
    """
    steps = turn.step_records
    if not steps:
        return 0.0
    pen = 0.0
    present = {b.type for b in turn.new_blocks}
    for expected in (_T, _O, _A, _AR, _R, _S):
        if expected not in present:
            pen += cfg.fmt_missing_type

    final = steps[-1]
    final_out = final.output_blocks
    last_two = [b.type for b in final_out[-2:]] if len(final_out) >= 2 else []
    if last_two != [_R, _S]:
        pen += cfg.fmt_bad_order

    # R/S only in final step
    for rec in steps[:-1]:
        if any(b.type in (_R, _S) for b in rec.output_blocks):
            pen += cfg.fmt_rs_in_nonfinal
            break

    # final step must have no tool calls
    if any(b.type == _A for b in final_out):
        pen += cfg.fmt_tool_in_final

    # 4f: R+S may only appear after submit has been called this turn.
    # Find the step index where R/S first appears (normally the final step) and
    # the step index where submit is first called; R/S must come strictly after.
    rs_step_idx: Optional[int] = None
    submit_step_idx: Optional[int] = None
    for idx, rec in enumerate(steps):
        if rs_step_idx is None and any(b.type in (_R, _S) for b in rec.output_blocks):
            rs_step_idx = idx
        if submit_step_idx is None:
            for b in rec.output_blocks:
                if b.type == _A:
                    parsed = b.parsed
                    if isinstance(parsed, dict) and parsed.get("name") == "submit":
                        submit_step_idx = idx
                        break
        if rs_step_idx is not None and submit_step_idx is not None:
            break
    if rs_step_idx is not None and (submit_step_idx is None or rs_step_idx <= submit_step_idx):
        pen += cfg.fmt_rs_without_submit

    # 4e: any disallowed tag anywhere in the turn's assistant content
    for rec in steps:
        raw = (rec.raw_assistant_message or {}).get("content") or ""
        if _contains_disallowed_tag(raw):
            pen += cfg.fmt_disallowed_tag
            break

    # 4g: invalid share per step (parser is_valid covers malformed/nested/bad-json/stray tags)
    invalid = 0
    for rec in steps:
        raw = (rec.raw_assistant_message or {}).get("content") or ""
        if raw.strip():
            _, valid = parse_scribe_blocks(raw)
            if not valid:
                invalid += 1
    share = invalid / len(steps)
    pen += cfg.fmt_invalid_share_weight * share

    return _clamp(1.0 - pen, 0.0, 1.0)


def metric_5(turn: TurnRecord, counter, cfg: TurnRewardConfig) -> float:
    """Metric 5 — total tokens rolled out this turn.

    README 5: rollout blocks = T/O/A/R/S (exclude TOOL_RESPONSE). More tokens ->
    lower reward, but should not dominate. Score = 1 - n/ref (ref=
    cfg.rollout_token_ref), range [0,1]. Given a small weight by design.

    Note: rollout_token_ref (default 1024) is just a conservative placeholder, NOT
    an empirically tuned optimum — it's where the linear score hits 0. Real RL
    rollouts are usually multiple-thousand tokens, so 1024 will clamp most
    samples to 0 and wipe the signal. Tune ref to the batch's rollout-token
    distribution (e.g. 4096) or switch to a softer 1 - n/(n+ref) decay once you
    see real data. Shaping lives in cfg (rollout_token_ref = shape, w5 =
    weight); both are open for tweaking without touching source.
    """
    n = sum(counter.count(b.content) for b in turn.rollout_blocks)
    ref = cfg.rollout_token_ref or 1
    return _clamp(1.0 - n / ref, 0.0, 1.0)


def metric_6(turn: TurnRecord, counter, cfg: TurnRewardConfig) -> float:
    """Metric 6 — token reuse ratio (no LLM judge, pure stats).

    README 6: reuse = |S ∩ (O∪A)| / |O∪A|
      S = token set of turn_summary; O = output; A = tool_call. Denominator is
      O∪A only (S excluded on purpose so the ratio measures "what fraction of
      the ground-truth key tokens the summary reuses", not inflated by S length).
    If no summary or no O∪A tokens -> 0.0. Range [0,1].
    """
    s_text = _last_summary(turn)
    oa = [b.content for b in turn.new_blocks if b.type in (_O, _A)]
    if not s_text or not oa:
        return 0.0
    s_set = set(counter.encode(s_text))
    oa_set = set()
    for t in oa:
        oa_set.update(counter.encode(t))
    if not oa_set:
        return 0.0
    return _clamp(len(s_set & oa_set) / len(oa_set), 0.0, 1.0)


def metrics_7_10(
    turn: TurnRecord, judge: Optional[Judge], cfg: TurnRewardConfig
) -> Tuple[float, float, float, float, bool]:
    """Metrics 7-10 — LLM-judge summary qualities (returned from ONE judge call).

    README "Note: metrics 7,8,9,10 will be given completely in one llm call."
      metric 7  faithfulness     — does the summary truthfully report O/A?
      metric 8  direction_neutrality — only retrospective, never plan next turn.
      metric 9  turn_focus       — describes only THIS turn, not prior turns.
      metric 10 fluency          — natural & fluent (guards against keyword dumps).
    judge=None OR no summary -> four cfg.judge_default (0.5) and judge_used=False
    (keeps RL batching / unit tests unblocked). Returns (m7, m8, m9, m10, used).
    """
    s_text = _last_summary(turn)
    if judge is None or not s_text:
        d = cfg.judge_default
        return d, d, d, d, False
    gt = _ground_truth_text(turn)
    try:
        sc = judge(gt, s_text)
    except Exception:
        d = cfg.judge_default
        return d, d, d, d, False
    return (
        _clamp(sc.faithfulness), _clamp(sc.direction_neutrality),
        _clamp(sc.turn_focus), _clamp(sc.fluency), True,
    )


def metric_11(turn: TurnRecord, counter, cfg: TurnRewardConfig) -> float:
    """Metric 11 — compression ratio of the summary (pure stats, no judge).

    README 11: ratio = len(S tokens) / len(O∪A tokens). Guards against the model
    dumping the whole ground-truth blocks as the summary (which would tank #11).
    Lower ratio -> higher score. Score = 1 - ratio, range [0,1].
    No summary / no O∪A tokens -> 0.0.

    Note: metric 6 uses *sets* (|S ∩ (O∪A)| / |O∪A|) to measure token reuse,
    but metric 11's spec uses len(...) — total token length, not unique tokens.
    """
    s_text = _last_summary(turn)
    oa = [b.content for b in turn.new_blocks if b.type in (_O, _A)]
    if not s_text or not oa:
        return 0.0
    s_len = len(counter.encode(s_text))
    oa_len = sum(len(counter.encode(t)) for t in oa)
    if oa_len == 0:
        return 0.0
    ratio = s_len / oa_len
    return _clamp(1.0 - ratio, 0.0, 1.0)


def metric_12(turn: TurnRecord, counter, cfg: TurnRewardConfig) -> float:
    """Metric 12 — cross-block n-gram overlap penalty (pure stats, no judge).

    README 12: penalize copy-paste from ground-truth (O∪A) into the summary.
      shared = set(ngrams(summary, n)) ∩ set(ngrams(ground_truth, n))
      overlap = |shared| / |set(ngrams(summary, n))|   (n = cfg.n_ngram, default 4)
      Slips past #11 (not long enough to tank compression) and may pass #10
      (fluent, real text). Higher overlap -> higher penalty -> lower score.
    Score = 1 - overlap, range [0,1].
    """
    s_text = _last_summary(turn)
    gt_text = _ground_truth_text(turn)
    if not s_text or not gt_text:
        return 0.0
    s_ng = set(_ngrams(counter.encode(s_text), cfg.n_ngram))
    gt_ng = set(_ngrams(counter.encode(gt_text), cfg.n_ngram))
    if not s_ng:
        return 0.0
    overlap = len(s_ng & gt_ng) / len(s_ng)
    return _clamp(1.0 - overlap, 0.0, 1.0)


def metric_13(turn: TurnRecord, counter, cfg: TurnRewardConfig) -> float:
    """Metric 13 — intra-block n-gram overlap penalty (pure stats, no judge).

    README 13: penalize repetition WITHIN a block (#12 only catches cross-block
    copy; it can't see a block looping itself internally). For each
    model-generated free-text block (T/O/R/S — TOOL_CALL excluded: structured
    JSON is naturally repetitive; TOOL_RESPONSE excluded: env-produced):
      intra = (total_ngrams - unique_ngrams) / total_ngrams
    Block ratios aggregated by token-weighted mean into one penalty. Pure
    statistics, no judge. Score = 1 - penalty, range [0,1]. No eligible blocks
    -> 1.0 (nothing degenerate).
    """
    pairs = []
    for b in turn.new_blocks:
        if b.type not in _INTRA_ELG_BLOCKS:
            continue
        toks = counter.encode(b.content)
        if not toks:
            continue
        ng = _ngrams(toks, cfg.n_ngram)
        if not ng:
            continue
        intra = (len(ng) - len(set(ng))) / len(ng)
        pairs.append((intra, len(toks)))
    if not pairs:
        return 1.0
    wsum = sum(intra * w for intra, w in pairs)
    tot = sum(w for _, w in pairs) or 1
    penalty = wsum / tot
    return _clamp(1.0 - penalty, 0.0, 1.0)


def build_judge_prompt(ground_truth: str, summary: str) -> List[Dict[str, str]]:
    """Messages for the ONE structured-output judge call backing metrics 7-10.

    Asks the judge to return the 4 floats {faithfulness(7), direction_neutrality
    (8), turn_focus (9), fluency (10)} in [0,1]. Feed its JSON to a
    SummaryJudgeScores; pair this with StructuredGenerator in the RL wiring
    (reward module itself stays sync and judge-agnostic).
    """
    sys = (
        "You are a strict judge of a SCRIBE agent's <turn_summary>. Given the "
        "ground-truth blocks (the agent's own OUTPUT + TOOL_CALL content this "
        "turn) and the agent's turn_summary, score four qualities each in "
        "[0,1]. faithfulness: does the summary truthfully report what the "
        "output/tool_call blocks did (high=accurate, low=fabricates/omits/"
        "contradicts)? direction_neutrality: does it only look back, never "
        "plan/predict the next turn (high=retrospective)? turn_focus: does it "
        "describe only THIS turn, not bleed in prior turns (high=scoped)? "
        "fluency: is the summary natural and fluent (high=fluent)? Respond with "
        "raw JSON only: {\"faithfulness\": float, \"direction_neutrality\": "
        "float, \"turn_focus\": float, \"fluency\": float}."
    )
    user = (
        f"GROUND-TRUTH BLOCKS:\n{ground_truth}\n\n"
        f"TURN SUMMARY:\n{summary}\n\nReturn the JSON scores."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def compute_turn_reward(
    turn: TurnRecord,
    config: Optional[TurnRewardConfig] = None,
    *,
    outcome: TurnOutcome,
    judge: Optional[Judge] = None,
    token_counter: Any = None,
) -> TurnRewardBreakdown:
    """Compute the 13 README turn metrics and weight them into one reward.

    metric_1  answer + natural termination + truncation penalty
    metric_2  tool usage (submit count + parallel identical dup)
    metric_3  step length (fewer tool-call steps = higher)
    metric_4  format correctness (README 4a-g structural rules + parser is_valid)
    metric_5  total rollout tokens (T/O/A/R/S, AR excluded)
    metric_6  token reuse ratio  |S∩(O∪A)| / |O∪A|
    metric_7  faithfulness (judge)            ┐
    metric_8  direction neutrality (judge)    ├ from ONE judge call (metrics 7-10)
    metric_9  turn focus (judge)             ┤
    metric_10 fluency (judge)                 ┘
    metric_11 compression ratio |S|/|O∪A|
    metric_12 cross-block n-gram overlap (S vs ground-truth O∪A)
    metric_13 intra-block n-gram overlap (within T/O/R/S blocks)

    total = clamp(Σ wi·metrici, 0..1). token_counter defaults to
    DeepSeekTokenCounter; judge=None -> judge sub-scores use cfg.judge_default.
    """
    cfg = config or DEFAULT_TURN_REWARD_CONFIG
    if not turn.step_records:
        # Empty turn has no meaningful reward signal.
        return TurnRewardBreakdown(total=0.0)
    if token_counter is None:
        from ..llm_runtime.token_counter import DeepSeekTokenCounter
        token_counter = DeepSeekTokenCounter()

    m7, m8, m9, m10, judge_used = metrics_7_10(turn, judge, cfg)
    bd = TurnRewardBreakdown(
        metric_1=metric_1(turn, outcome, cfg),
        metric_2=metric_2(turn, outcome, cfg),
        metric_3=metric_3(turn, cfg),
        metric_4=metric_4(turn, outcome, cfg),
        metric_5=metric_5(turn, token_counter, cfg),
        metric_6=metric_6(turn, token_counter, cfg),
        metric_7=m7, metric_8=m8, metric_9=m9, metric_10=m10,
        metric_11=metric_11(turn, token_counter, cfg),
        metric_12=metric_12(turn, token_counter, cfg),
        metric_13=metric_13(turn, token_counter, cfg),
        judge_used=judge_used,
    )
    ws = [cfg.w1, cfg.w2, cfg.w3, cfg.w4, cfg.w5, cfg.w6,
          cfg.w7, cfg.w8, cfg.w9, cfg.w10, cfg.w11, cfg.w12, cfg.w13]
    total = sum(w * v for w, v in zip(ws, bd.metrics))
    bd.total = _clamp(total, cfg.total_low, cfg.total_high)
    return bd


__all__ = [
    "TurnRewardConfig", "TurnRewardBreakdown", "TurnOutcome",
    "SummaryJudgeScores", "Judge", "DEFAULT_TURN_REWARD_CONFIG",
    "compute_turn_reward", "build_judge_prompt",
    "metric_1", "metric_2", "metric_3", "metric_4", "metric_5", "metric_6",
    "metric_7", "metric_8", "metric_9", "metric_10", "metric_11",
    "metric_12", "metric_13",
]