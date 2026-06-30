# SCRIBE : Agents Should Write What They Learn

A new design paradiam and infra to LLM-agent training for Loop Engineering.

SCRIBE extends the classic ReAct loop with explicit `Reflect` and `Summarize` phases.

One trajectory is a loop,with many turns,and each turn with many steps.

Our infra now can collect the rollout data,in threshold mode we can stop once the result of a turn exceeds the pass threshold.Each turn we have a reward,and a trajectory reward is associated with the reward of all turns.

## First ：the context engineering of each turn:

Every LLM response follows this strict tag order. For convenience,we use T to stand for the think block,A to stand for the tool_call block,O to stand for the output block,R to stand for the reflect block,S to stand for the turn_summary block and AR to stand for the tool_response block.A step's rollout usually is a
list[T/O/list[A]+list[AR]]+R+S  structure.
each step stores its rollout blocks and its reward(most of the step is 0,only the step after the last submit step gets the real reward).

History Compression rule:

1.Disposible T&R blocks: The T and R blocks are both disposible,next rollout(next step) they will disappear in input blocks.
2.The steps before the recent k(default k=3) steps only preserve the S block.
3.the whole input blocks' tokens nears limit:Compression triggered automatically before rollout.this step finally stores the compressed input and its rollout blocks.the compression is firstly collecting all the S block in the input blocks as the compression result(new input blocks),the first user prompt and turn-feedback and the system prompt are ignored,then judge the length of the new input blocks(S block sequence),if the sequence's tokens nears limit again,use sota llm to do a compression(the first user prompt and turn-feedbacks are included!) and use <turn_summary>\nThe Output Of llm\n</turn_summary> as the new input block and compression results. This one block's tokens should be guaranteed that they don't near the limit.

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
) > limit-small_number,compression triggered,before this step,other steps are all <=.

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
len(the token set in turn_summary block ∩ the token set in output,tool_call,turn_summary block) / 
len(the token set in output,tool_call,turn_summary block)

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
len(the token set in output,tool_call block),this metric lower,the reward higher.use llm as a judge.

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
the metric is low,that means the block does not repear itself,oterwise the block loops.

**Note：the 7,8,9,10,11 metric will be given completely in one llm call.**

**Note：the reflect block is a cot process for model to let the latter turn_summary get higher reward.**

## Third : the reward rule of a trajectory is:

the reward of the final turn / len(turns_used) 

## Fourth : Credit Assignment:

we use grpo for different trajectory.
-not reinforce: not stable,variance so large
-not ppo: critic model is expansives
-but grpo: stable enough,more effective

we will assign the same credit for each turn in the same trajectory,and the same credit for each rollout token in the same turn.

we have TOKEN-LEVEL CREDIT ASSIGNMENT,because the turn_summary has the token reuse ratio.