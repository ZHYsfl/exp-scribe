# SCRIBE : Agents Should Write What They Learn

A new design paradigm and infra to LLM-agent training for Loop Engineering.

SCRIBE extends the classic ReAct loop with explicit `Reflect` and `Summarize` phases.

One trajectory is a loop,with many turns,and each turn with many steps.

Our infra now can collect the rollout data,in threshold mode we can stop once the result of a turn exceeds the pass threshold.Each turn we have a reward,and a trajectory reward is associated with the reward of all turns.

## First ：the context engineering of each turn:

Every LLM response follows this strict tag order. For convenience,we use T to stand for the think block,A to stand for the tool_call block,O to stand for the output block,R to stand for the reflect block,S to stand for the turn_summary block and AR to stand for the tool_response block.A step's rollout usually is a
list[T/O/list[A]+list[AR]]+R+S  structure.
each step stores its rollout blocks and its reward(most of the step is 0,only the step after the last submit step gets the real reward).

History Compression rule:

1.Disposable T&R blocks: The T and R blocks are both disposable,next rollout(next step) they will disappear in input blocks.
2.The steps before the recent k(default k=3) steps only preserve the S block.
3.the whole input blocks' tokens nears limit:Compression is triggered automatically **before rollout**. Two sets must be kept distinct:

- **Token-count set (decides WHETHER to compress):** the **whole input context** — system prompt + first user prompt + every turn-feedback + all input blocks, counted together. The system prompt IS included in this count.
- **Compression set (decides WHAT to compress):** grows by stage. The system prompt is NEVER compressed — it is always preserved verbatim (it carries the task/protocol and must stay identical across the trajectory). But the first user prompt and turn-feedbacks are NOT always preserved:
  - **Stage 1 (collect S blocks):** only the input blocks are touched; first user prompt + turn-feedbacks are preserved verbatim.
  - **Stage 2 (sota-LLM compression, triggered when the S-block sequence still nears the limit):** the LLM input INCLUDES the first user prompt and turn-feedbacks (see the explicit "the first user prompt and turn-feedbacks are included!" below) — they get folded into the single `<turn_summary>` output. Only the system prompt stays verbatim through both stages.

If the whole-input-context token count nears the limit, we compress **until it no longer nears the limit**, then proceed to rollout. This step finally stores the compressed input and its rollout blocks. The compression is firstly collecting all the S block in the input blocks as the compression result(new input blocks),the first user prompt and turn-feedback and the system prompt are ignored,then judge the length of the new input blocks(S block sequence),if the sequence's tokens nears limit again,use sota llm to do a compression(the first user prompt and turn-feedbacks are included!) and use <turn_summary>\nThe Output Of llm\n</turn_summary> as the new input block and compression results. This one block's tokens should be guaranteed that they don't near the limit; if the sota llm output still exceeds the limit, retry the llm compression (the retry/backoff mechanism of the sota llm is decoupled from SCRIBE) until the single block fits.

**Hard boundary:** if the incompressible prefix — system prompt + first user prompt + all turn-feedbacks, with every input block already compressed away to nothing — still exceeds the limit on its own, compression cannot help. In that case SCRIBE raises an error (the trajectory config is infeasible: the preserved-verbatim parts alone overflow the context window).

for instance:

```text
if k=3,max_steps=5,max_turns=3

turn0:

step0:
system_prompt
first_user_prompt
list[T/O/list[A]+list[AR]]+R+S,REWARD0

step1:
system_prompt
first_user_prompt
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD1

step2:
system_prompt
first_user_prompt
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD2

step3:
system_prompt
first_user_prompt
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD3

step4:
system_prompt
first_user_prompt
S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD4

REWARD4 does not exceed the threshold.add feedback,
next turn:

turn1:

step0:
system_prompt
first_user_prompt
S
S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
feedback_from_turn_0
list[T/O/list[A]+list[AR]]+R+S,
REWARD5

step1：
len(
system_prompt
first_user_prompt
S
S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
feedback_from_turn_0
list[T/O/list[A]+list[AR]]+R+S
) > limit-small_number,compression triggered before rollout (check-before-rollout rule),other steps are all <=.

-> 
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S

len(
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S
) <= limit-small_number

system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S
list[T/O/list[A]+list[AR]]+R+S,
REWARD6

step2：
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD7

step3:
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD8

step4: 
len(
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S
)
> limit-small_number,compression triggered,the second to last and the last steps(turn1-step2,turn1-step3) are both <=.

->
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S S S S

len(
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S S S S
)<= limit-small_number

system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S S S S
list[T/O/list[A]+list[AR]]+R+S,
REWARD9

REWARD9 does not exceed the threshold.add feedback,
next turn:

turn2:

step0：
len(
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S S S S
list[T/O/list[A]+list[AR]]+R+S
feedback_from_turn_1
)
>limit-small_number,compression triggered.

->
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S S S S S
feedback_from_turn_1

len(
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S S S S S
feedback_from_turn_1
) still
> limit-small_number,compression triggered.

-> system_prompt S(call sota llm to summarize once,the its tokens does not near the limit)

system_prompt S,REWARD10

step1:

system_prompt S list[T/O/list[A]+list[AR]]+R+S,REWARD11

REWARD11 exceeds threshold!!!In threshold mode,the loop stops.
```

```text
The trajectory finally is:

step0:
system_prompt
first_user_prompt
list[T/O/list[A]+list[AR]]+R+S,REWARD0

step1:
system_prompt
first_user_prompt
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD1

step2:
system_prompt
first_user_prompt
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD2

step3:
system_prompt
first_user_prompt
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD3

step4:
system_prompt
first_user_prompt
S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD4

step5:
system_prompt
first_user_prompt
S
S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
feedback_from_turn_0
list[T/O/list[A]+list[AR]]+R+S,
REWARD5

step6:
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S
list[T/O/list[A]+list[AR]]+R+S,
REWARD6

step7：
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD7

step8:
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S
list[O/list[A]+list[AR]]+S
list[O/list[A]+list[AR]]+S
list[T/O/list[A]+list[AR]]+R+S,
REWARD8

step9:
system_prompt
first_user_prompt
S S S S S
feedback_from_turn_0
S S S S
list[T/O/list[A]+list[AR]]+R+S,
REWARD9

step10：
system_prompt S,REWARD10

step11:
system_prompt S list[T/O/list[A]+list[AR]]+R+S,REWARD11
```

## Second : the reward rule of a turn is:

the metrics(MULTI-DIMENSIONAL REWARD PREVENTS HACKING)
（ALL THE METRICS ARE JUST FOCUED ON THIS TURN'S NEW BLOCKS,NOT THE INPUT BLOCKS）:

1.the reward of the final step.

2.the submit tool call time.

3.the length of the steps used.(max is max_steps)

4.the format correctness(if have think,output,tool_call,tool_response,reflect,turn_summary block,if reflect are the second to last block,if the turn_summary are the last block,if the data of the turn is valid,etc.)

5.the total tokens rollouted out in this turn(include think,output,tool_call,reflect and turn_summary block,exclude the tool_response block.This metric is larger,the reward is lower,but should not effect reward so much)

6.token reuse ratio,
reuse = |S ∩ (O∪A)| / |O∪A|
where S = token set of the turn_summary block, O = token set of the output block, A = token set of the tool_call block. The denominator is O∪A only (S is excluded from the denominator on purpose, so the ratio measures "what fraction of the ground-truth key tokens the summary reuses" and is not inflated by the summary's own length).

7.the faithfulness of the turn summary— did the summary tell the truth?(use llm as a judge to give a score of the faithfulness of the turn summary,the turn summary should be a statement of what we do in the output and tool_call block.The judge compares the summary against 
the ground-truth blocks(output and tool_call blocks) and scores how consistent they are.) High score is the summary 
accurately reports what was thought and done,while the low score is the summary fabricates, omits, or contradicts the real behavior.

8.the direction neutrality - did the summary stay retrospective?The turn summary must only **look back** at the current turn. It must not predict, plan, or prescribe what the *next* turn should do.High score is the summary purely recaps,while the low score is the summary leaks into future planning.use llm as a judge.

9.the turn focus of the turn summary - does the summary describe only this turn?
Then turn summary must be scoped strictly to the **current** turn.It must not mix in content from **earlier** turns - each turn's summary stands alone as the record of that turn alone.
(Note: later turns' content cannot appear by construction — the summary is generated at the end of the current turn, before any subsequent turn exists. So this dimension only guards against bleeding **prior** turns in.)
High score is the summary describes only what happened in this turn,while the low score is the summary folds prior turns' actions into the current one.
use llm as a judge.

10.the fluency of the turn summary,use llm as a judge.we need to improve the token reuse ratio metric,but if we pile up the key tokens awkwardly,that sucks.so we need to judge the fluency of the turn summary to make the turn summary fluent and natural.

11.the compression ratio of the summary,to improve the the token reuse ratio metric and the fluency of the turn summary,the model maybe use the whole ground-truth blocks(output and tool_call blocks) as a turn summary,that sucks.so we need to count the compression ratio of the summary:len(the token set in turn_summary block) / 
len(the token set in output,tool_call block),this metric lower,the reward higher.This is a pure statistical metric (no LLM judge needed).

12.Cross block N-gram overlap penalty,penalize copy-paste from ground-truth blocks.Even with #11 (compression ratio) blocking wholesale copying, the model may **partially** copy-paste: lifting consecutive phrases verbatim from the ground-truth blocks into the summary. This slips past #11 (not long enough to tank compression) and may even pass #10 (fluent, since it's real text),yet it is not genuine summarization.
This metric computes the overlap of **contiguous n-grams** (n=4 recommended) between the turn_summary block and the ground-truth (output + tool_call) blocks:
shared = set(ngrams(summary,4)) ∩ set(ngrams(ground_truth, 4))
overlap_ratio = |shared| / |set(ngrams(summary, 4))|
Higher overlap → higher penalty → lower reward. It is a pure statistical metric (no LLM judge needed), cheap to compute.low overlap is the summary rephrases in its own words,the high overlap is the summary lifts verbatim runs from ground-truth.

13.Intra-block N-gram Overlap Penalty — penalize repetition within a block.#12 only catches the summary copying from ground-truth (cross-block). It cannot see a block repeating *itself internally* — a common small-model /early-RL failure where a block loops the same phrase to pad length.For each model-generated block(think / output / reflect / turn_summary),compute the share of its n-grams 
(n=4) that are redundant due to internal repetition:
```python
def intra_overlap(block_tokens,n=4):
  ngrams = [tuple(block_tokens[i:i+n]) 
  for i in range(len(block_tokens) - n +1)]
  total = len(ngrams)
  unique = len(set(ngrams))
  return (total - unique) / total if total else 0.0
```
Block-level ratios are aggregated (e.g. weighted mean) into one penalty. Pure statistics, no judge. Scope: only
model-generated blocks — tool_call (structured JSON, naturally repetitive) and tool_response(env-produced, not
the model's responsibility) are excluded.
the metric is low,that means the block does not repeat itself,otherwise the block loops.

**Note：the 7,8,9,10,11 metric will be given completely in one llm call.**

**Note：the reflect block is a cot process for model to let the latter turn_summary get higher reward.**

## Third : the reward rule of a trajectory is:

trajectory_reward = final_turn_reward * decay ** len(turns_used)

where `final_turn_reward` is the turn-level reward of the last turn (0~1, from the 13 metrics), `len(turns_used)` is the number of turns actually used in this trajectory, and `decay` is a discount factor in (0,1) (e.g. 0.8). This rewards solving in fewer turns without diluting the main signal as a plain division would: a 1-turn success gives `final_turn_reward * decay`, a 3-turn success gives `final_turn_reward * decay^3` — the gap scales geometrically but the base signal (`final_turn_reward`) is preserved.

## Fourth : Credit Assignment:

we use grpo for different trajectory.
-not reinforce: not stable,variance so large
-not ppo: critic model is expensive
-but grpo: stable enough,more effective

The reward→credit pipeline (written out explicitly to avoid confusing "reward" with "credit"):

1. **trajectory_reward** (from Section Third) is the quantity GRPO compares across the group of trajectories sampled on the same task. The GRPO group baseline = mean of trajectory_reward over the group; each trajectory's **advantage** = (its trajectory_reward − group mean) / group std.
2. **turn-level credit**: the trajectory advantage is shared equally across all turns in that trajectory → each turn's credit = trajectory_advantage / len(turns_used).
3. **token-level credit**: the turn's credit is shared equally across all rollout tokens in that turn → each rollout token's credit = turn_credit / num_rollout_tokens_in_turn. (Rollout tokens = model-generated tokens only: think / output / tool_call / reflect / turn_summary blocks. Input blocks, tool_response, system/kickoff/feedback are masked — not model-generated.)

So the chain is fixed: **trajectory_reward → (GRPO group) → trajectory_advantage → (÷turns) → turn credit → (÷rollout tokens) → token credit**, and the token credit is what multiplies ∇log P(token) in the policy gradient.

we will assign the same credit for each turn in the same trajectory,and the same credit for each rollout token in the same turn.

we have TOKEN-LEVEL CREDIT ASSIGNMENT,because the turn_summary has the token reuse ratio.