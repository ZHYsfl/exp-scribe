# SCRIBE: State Compression in Recursive Iterative Batched Execution

A new approach to LLM-agent training for Loop Engineering, built on verl-agent.

SCRIBE extends the classic ReAct loop (`Reason → Act → Observe`) with explicit `Reflect` and `Summarize` phases, producing a token-grounded, append-only step summary at every iteration.

---

## Core Design Philosophy

### 1. **NO CRITIC MODEL NEEDED**
Every step has an **objective environment reward**. No need for a separate value network.

### 2. **1 Trajectory = max_steps Training Samples**
A `max_steps=10` loop produces 10 training samples. Each step's reward = **discounted cumulative return from that step onward** plus SCRIBE quality signals.

### 3. **MULTI-DIMENSIONAL REWARD PREVENTS HACKING**
8 orthogonal reward signals make it extremely hard to game the system:

| Reward Signal | Range | Weight | Purpose |
|---------------|-------|--------|---------|
| Environment reward | [0, 1] | 1.0 | Task objective progress |
| Format compliance | [0, 1] | 0.3 | All tags present and ordered correctly |
| Token reuse ratio | [0, 1] | 0.5 | Key tokens appear in summary |
| Faithfulness | [0, 1] | 0.4 | Summary matches facts |
| Direction neutrality | [0, 1] | 0.3 | Summary doesn't predict next step |
| Fluency | [0, 1] | 0.3 | Summary reads naturally |
| Compression penalty | [0, ∞) | 0.3 | Penalize summary being too long |
| N-gram overlap penalty | [0, ∞) | 0.3 | Penalize copy-paste from results |

### 4. **TOKEN-LEVEL CREDIT ASSIGNMENT**
Reward is not just applied to the last token:
- **Base reward**: Spread evenly across all response tokens
- **Bonus reward**: Applied ONLY to tokens shared between **(action + observation/result)** and **summary**
  - These are the state-carrying, fact-grounded tokens
  - Natural-language tokens from `think`/`reflect` are not bonused unless they also appear in the factual source

This aligns with the token-reuse metric: the summary should reuse key tokens from what actually happened this step.

---

## Architecture

### 5 Core Modules (each < 250 LOC)

| Module | File | Purpose |
|--------|------|---------|
| ScribeMemory | `scribe_memory.py` | Append-only T/E/R/S trajectory history with hybrid compression |
| ScribePrompts | `scribe_prompts.py` | SCRIBE-format prompt templates |
| ScribeUtils | `scribe_utils.py` | Response parsing + quality metric computation |
| ScribeRewardManager | `scribe_reward.py` | Multi-dimensional token-level reward computation |
| ScribeEnvManager | `scribe_env_manager.py` | Environment wrapper with SCRIBE memory injection |

---

## SCRIBE Step Format

Every LLM response follows this strict tag order. For convenience,we use T to stand for <think>\nxxx\n</think>\n block,A to stand for <action>\nxxx\n</action>\n<action_result>\nxxx\n</action_result>\n block,O to stand for the natural language output of model except for T and A,R to stand for the <reflect>\nxxxx\n</reflect>\n block,S to stand for the <step_summary>\nxxxx\n</step_summary>\n block.**WE CALL list[O/A](once A has emerged,list[O/A] object finished) AS E(execute) block.**

A standard unit of SCRIBE is:

SCRIBE_UNIT = list[Optional[T]+E] + R + S 

```text
<think>
Natural, fluent, task-specific reasoning before the first output.
</think>
O
<action>
action here
</action>
<action_result>
environment result after action
</action_result>
<think>
After seeing the result, reason again before the next output. The previous think is no longer the active reasoning.
</think>
O
<action>
action here
</action>
<action_result>
environment result after action
</action_result>
......
<reflect>
Now I need to write the step_summary for this entire step. I will explicitly check every SCRIBE reward dimension:
1. Token reuse: the key tokens from my actions/results are ... and must appear in the summary.
2. Fluency: I will phrase the summary as a natural sentence; I must not sacrifice readability to stuff tokens.
3. Faithfulness: the summary must only state facts present in the observations/results.
4. Direction neutrality: I must not write "next I should..." or "the next step is...".
5. Compression: the summary should be much shorter than the raw results.
6. N-gram overlap: I must not copy long phrases verbatim; I should paraphrase while keeping key tokens.
7. Format compliance: I have included think, action/result  or natural-language outputs, optional answer, reflect, and step_summary in the right order.
8. Step scope: the summary covers ONLY this step's actions and results, not previous or future steps.
</reflect>
<step_summary>
Short, fluent, factual summary of ONLY THIS STEP. Must reuse key tokens from actions and results. No predictions of next steps. No subjective opinions. No long copy-paste.
</step_summary>
```

---

## Memory Design

### Stored Format

Each step stores (the trajectory of Optional[T]/E(O/A)/R/S,REWARD)

### History Compression

- **Disposible T&R blocks**: The T and R blocks are both disposible,next rollout time they will disappear.
- **Recent k steps (default k=3)**: Full list[E]+S block context preserved
- **Older steps**: Only S block context retained (compressed form)
- **Trajectory-level compression**: Triggered automatically when context nears limit

for instance:

```text
if k=3,max_steps=11,

iter0: list[Optional[T]+E] R S,REWARD1

iter1: list[E]+S list[Optional[T]+E] R S,REWARD2(computed by [iter0+discounted cumulative* iter1]/2)

iter2: list[E]+S list[E]+S list[Optional[T]+E] R S,REWARD3

iter3: list[E]+S list[E]+S list[E]+S list[Optional[T]+E] R S,REWARD4

iter4: S list[E]+S list[E]+S list[E]+S list[Optional[T]+E] R S,REWARD5

iter5: S S list[E]+S list[E]+S list[E]+S list[Optional[T]+E] R S,REWARD6

len(S S list[E]+S list[E]+S list[E]+S list[Optional[T]+E] R S)>limit-small_num,compress all

iter6：S S S S S S list[Optional[T]+E] R S,REWARD6

iter7：S S S S S S list[E]+S list[Optional[T]+E] R S,REWARD7

iter8: S S S S S S list[E]+S list[E]+S list[Optional[T]+E] R S,REWARD8

len(S S S S S S list[E]+S list[E]+S list[Optional[T]+E] R S)
>limit-small_num,compress all

iter9: S S S S S S S S S list[Optional[T]+E] R S,REWARD9

len(S S S S S S S S S list[Optional[T]+E] R S)
>limit-small_num,compress all
len(S S S S S S S S S S)
>limit-small_num,compress all
-> S(call llm to summarize once)

iter10：S list[Optional[T]+E] R S,REWARD10
```


---

## Quick Start

### 1. Run Smoke Tests
```bash
cd /root/autodl-tmp/verl_agent_repo
source /root/.venv/bin/activate
python -m recipe.scribe.test_scribe
```

### 2. Generate SFT Data
```bash
python -m recipe.scribe.generate_sft_data
```

### 3. Train on AlfWorld
```bash
python -m recipe.scribe.main_scribe \
    env.env_name=alfworld \
    env.alfworld.eval_dataset=eval_in_distribution
```

### 4. Use as a Library
```python
from recipe.scribe import (
    ScribeMemory,
    ScribeRewardManager,
    parse_scribe_response,
    compute_scribe_metrics,
)
```

---

## Comparison with Baselines

| Method | Has Memory | Has Token Credit | Needs Critic | # Signals Against Hacking |
|--------|-----------|------------------|--------------|---------------------------|
| Standard PPO | ❌ | ❌ | ✅ | 1 |
| GRPO | ❌ | ❌ | ❌ | 1 |
| GiGPO | ✅ | ❌ | ✅ | 2 |
| **SCRIBE** | ✅ | ✅ | ❌ | 8 |

---

## YAML Configuration

Key parameters in `config/scribe_trainer.yaml`:

```yaml
algorithm:
  gamma: 0.95                           # Discount factor
  scribe_reward:
    w_format: 0.3                       # Format compliance
    w_token_reuse: 0.5                  # Summary reuses key tokens
    w_faithfulness: 0.4                 # Summary matches facts
    w_direction_neutrality: 0.3         # No forward-looking statements
    w_fluency: 0.3                      # Summary reads naturally
    w_compression_penalty: 0.3          # Penalize long summaries
    w_overlap_penalty: 0.3              # Penalize copy-paste
    max_compression_ratio: 0.7          # Threshold for compression penalty
    max_trigram_overlap: 0.5            # Threshold for copy-paste penalty

  scribe_memory:
    k_recent_steps: 3                    # Recent steps get full list[E]+S
    compression_enabled: True            # Older steps compressed to S-only
```

---

## TODO / Future Work

- [ ] WebShop environment support
- [ ] Sokoban environment support
- [ ] Ablation studies for each reward dimension
- [ ] Trajectory-level compression trigger
- [ ] Multi-environment batch training

---

## Citation

```
@software{scribe2026,
  title = {SCRIBE: State Compression in Recursive Iterative Batched Execution},
  year   = 2026,
  note   = {Built on verl-agent framework}
}
```
