# SCRIBE: Externalising Agent State as Token-Grounded, Append-Only Summaries

> Agent 不是这篇工作的主题。Agent 只是最先把问题暴露出来的 use case。
>
> 我们想说的是更基础的一件事：**transformer 的"状态"不应该藏在 hidden state 里隐式归纳，而应该被外化成 token stream 上的、append-only 的、token-grounded 的显式表征**——而且这个表征不应该是事后压缩的产物，应该是模型每一步本来就在产生的东西。
>
> 当这一点被设计进数据格式时，三个看起来不相关的问题同时塌缩成一个：长程任务的上下文管理、KV cache 复用、RL on LLM 的 token-level credit assignment。

---

## 1. Thesis

当前 LLM agent 的状态管理范式是**隐式 + 重读历史**的：模型把"我现在处于任务的哪一步、可用事实是什么"藏在 hidden state 里靠归纳保持，需要时通过重读整条 trajectory 唤起。

这个范式在 ReAct 框架下被定型：

```
Reason → Act → Observe → Reason → Act → Observe → ...
```

它有三个互相纠缠的痛点，每一个单独看都已经催生了独立 sub-field：

| 痛点 | 经典缓解方向 | 局限 |
|---|---|---|
| 长程任务超过 context window | trajectory-level 压缩、memory module | 压缩昂贵、一次性损失大、改写前缀 → KV cache 失效 |
| KV cache 难复用 | StreamingLLM / H2O / attention sink eviction | "猜"哪些 token 重要，无信号告诉系统哪些 token 是状态 |
| RL on LLM 的 credit assignment | GRPO / DPO / PPO 的算法层优化 | 整轨 reward 平摊到所有 token，无 token 级 grounding |

**SCRIBE 主张这三件事是同一件事的三个面**——只要 agent 在每一步结束时被强制产出一段满足下列三个性质的小段文本，这三个痛点同时消失：

1. **Append-only**：本段文本只追加在历史末尾，永不改写前缀 → KV cache 完全复用。
2. **Token-grounded**：本段文本必须复用本步 tool output / final answer 中的载重 token，禁止脱离 observation 的自由发挥 → token 级 credit assignment 有锚点。
3. **Direction-neutral**：本段文本只陈述已发生事实，不引导下一步、不带主观倾向 → 它是状态快照，不是规划稿，可以被下一步当作 grounding 输入而不引入污染。

我们把这段文本叫 `<step_summary>`。

---

## 2. 从 ReAct 到 SCRIBE：一个看起来很小的改动

ReAct 一个 step 的形状：

```
<think>R</think>  →  tool_call(A)  →  tool_response(O)
```

SCRIBE 一个 step 的形状：

```
<think>R</think>  →  tool_call(A)  →  tool_response(O)  →  final_answer
→  <reflect>CoT for writing the step_summary</reflect>
→  <step_summary>append-only, token-grounded, direction-neutral state snapshot</step_summary>
```

差别看起来只是"多写了两段"。但下面这一节会说明：这两段的设计是支撑前述三个性质的全部。

### 2.1 `<step_summary>`：状态被外化

`<step_summary>` 永远满足：

- 只覆盖本 step（从开头的 `<think>` 到刚写完的 final answer），**不**回顾以往 step，**不**预测未来 step。
- 必须重用本 step tool output 和本 step final answer 中的关键 token（文件名、命令、错误码、数字、专有名词）——但不得为塞 token 而破坏句子通顺性。
- 不写主观判断、不写"接下来应该…"、不写"看起来…"。

下一 step 的模型不依赖 hidden state 也不依赖重读整条 trajectory 来知道"现在到哪了"——它只需要读前一个 step 留下的 `<step_summary>`。

### 2.2 `<reflect>`：把"写出合规 summary"变成可监督的 CoT

`<reflect>` 不是开放式自评，它是**写好 `<step_summary>` 的 chain-of-thought 规划稿**。结构化为五步：

1. **事实回放** — 本 step 实际发生了什么
2. **载重 token 枚举** — 哪些 token 必须复用，为什么是 load-bearing
3. **通顺性检查** — 怎么让 token 复用不破坏句子流畅性
4. **方向性 / 主观倾向排除** — 列出"我可能会写但不该写"的措辞并显式 ban 掉
5. **范围检查** — 确认只覆盖本 step

`<reflect>` 在训练时拿 loss——模型学会**动笔前先想清楚怎么写状态**。推理时被 runtime 正则剥掉，是 throw-away 计算，不进下游 KV cache。

### 2.3 一个间接但关键的设计点：模型不知道自己是第几 step

`<step_summary>` 不写 `Step-1` / `Step-2` 这种编号——模型在写它时没有可靠的计数源（上一 step 的 `<think>` 已被 Qwen 模板剥离）。要引用上一 step 留下的产物，统一说 "the prior `<step_summary>`"。

这件事看起来是个小约束，实际上是 SCRIBE 状态表征**位置无关、可压缩、可重排**的前提。

---

## 3. 三个问题为什么同时塌缩

### 3.1 长程任务上下文管理

经典 ReAct 的压缩是**事后补的**：trajectory 长到不行了，单独花一次 LLM 调用把 N 步压成一段 S̄。代价：一次昂贵 forward；前缀被改写；信息损失是大块的而非平摊。

SCRIBE 的压缩是**每步同步产出的**：`<step_summary>` 就是本 step 的压缩态。压缩成本被平摊到每个 step 的尾巴上，且和"输出 final answer"共享同一次前向——**无额外 LLM 调用**。

```
ReAct:   step₁ step₂ ... stepₙ ─────压缩───► S̄
                                  ↑ 一次性大开销

SCRIBE:  step₁ → s₁         （压缩在 step 内完成）
         step₂ → s₂         （append-only）
         ...
         stepₙ → sₙ
         
         模型能 reason 的有效历史 = s₁ s₂ ... sₙ + 当前 step 的 fresh trace
         物理 context window 没变，但有效历史步数 ↑↑
```

只有当 `s₁ s₂ ... sₙ` 这个序列本身长到再次需要压缩时，才触发"二级压缩"（trajectory-level）——README §压缩策略 处理这部分。

### 3.2 KV cache 友好

经典压缩在 KV cache 上是灾难：前缀变了，cache 全失效。

SCRIBE 的 token stream 结构：

```
[system] + [user₁] + [final_answer₁ + <step_summary₁>] + [user₂] + [final_answer₂ + <step_summary₂>] + ...
                     └──────────永久保留，前缀不变──────────┘
```

- `<step_summary>` 是 append-only 的：上一 step 的 KV cache 永远保留不动。
- `<think>` 被 Qwen 模板自然丢（在 prompt assembly 阶段就不渲染），不影响 cache 复用。
- `<reflect>` 被我们正则剥（只剥过去 turn 的尾段），前面的 final_answer + step_summary 部分前缀完全没动。

唯一会触发前缀改写的事件，是 `s₁ ... sₙ` 序列长到要做二级压缩——但这个事件被设计成**尽可能稀有**（README §压缩策略），且发生时也能享受压缩前那段 S 序列的公共前缀（压缩到一个 token 序列 → 多次 inference 复用）。

**结果**：高频事件（每 step 的状态更新）是 cache-friendly append；低频事件（二级压缩）才是 cache 屠杀——把昂贵事件推到罕见时刻。

### 3.3 RL on LLM 的 token-level credit assignment

GRPO / PPO on LLM 的根本困境：reward 是 trajectory-level 标量，反传时所有 token 共享同一个梯度信号。没有信号告诉模型"这个 token 是关键决策点，那个 token 是噪声"。

SCRIBE 的解法不是设计新 RL 算法，而是**用数据格式预先把 token 级 credit 标好了**：

- `<step_summary>` 被强制复用本 step tool output / final answer 里的载重 token。
- 这些被复用的 token 是程序可识别的（`step_summary` 的 token 集 ∩ tool_output ∪ final_answer 的 token 集）。
- 教师模型/judge 给本 step 打 reward 时，自然落到这些被复用的 token 上——它们是本 step 决策的"可验证残留"。
- 训练时对这些 token 加权 → 反传时它们的梯度被放大 → 等价于在告诉模型"你刚才决定把这些 token 留下来是对的（或错的）"。

**关键洞察**：这不是 RL 算法上的进步，是用**输出格式约束**把"哪些 token 是该被奖励的决策载体"这件事**预先标好**了。token-level credit 从需要新算法变成需要新格式。`<reflect>` 里的"载重 token 枚举"那一步，本质上是教师模型在帮 trainer 标 attention mask。

#### 3.3.1 但格式约束自己会被 hack——SCRIBE 怎么防 Goodhart

把 `step_summary_token_reuse_ratio` 当 reward 立刻引出一个攻击：模型可以**把 final answer 整段抄进 `<step_summary>`**，reuse ratio 直接到 1.0。任何单一指标作为 reward 都会被 hack——这是 Goodhart's law。

SCRIBE 的设计**没有用单一指标**，而是引入四个**结构上互不相容**的防御指标，让作弊路径不存在 Pareto-最优：

| 指标 | 抓的作弊形态 |
|---|---|
| `step_summary_compression_ratio = len(summary)/len(final_answer)` | summary 太长 / 几乎等同 final answer（直接抄） |
| `step_summary_max_ngram_overlap_with_final_answer` | summary 大段 n-gram 来自 final answer（抄袭式作弊） |
| `step_summary_fluency_score` (LLM-as-judge) | 为塞 token 把句子写崩（token reuse 推过头） |
| `step_summary_noise_token_reuse_ratio` | 复用 "and / is / the / as expected" 等噪声 token 刷 reuse ratio |

四道闸的不相容性是**结构性**的，不是经验调参出来的：

- 直接抄 final answer → 过 `reuse_ratio` 但被 `compression_ratio` 和 `max_ngram_overlap` 同时抓。
- 复用噪声词刷分 → 过 `reuse_ratio` 但被 `noise_token_reuse_ratio` 抓。
- 硬把载重 token 塞进句子 → 过 `reuse_ratio` 但被 `fluency_score` 抓。
- 写超短无信息 summary 避免被抓 → 过 `compression_ratio` 但 `faithfulness` 和 `reuse_ratio` 双双崩盘。

合成 reward：

```
step_summary_reward = w_faith·faithfulness + w_neutral·direction_neutrality + w_reuse·token_reuse_ratio
                     - max(0, compression_ratio - 0.7)·p1
                     - max(0, max_ngram_overlap - 0.5)·p2
                     - max(0, 0.85 - fluency_score)·p3
                     - max(0, noise_token_reuse_ratio - 0.3)·p4
```

`<reflect>` 的第六步——"反 reward hacking 自检"——是这个防御的**前向对偶**：模型在产出 summary 前就被 CoT 强制问自己 "我的 summary 能不能被 diff 工具显示为 final answer 的子集"。前向 CoT 自检 + 后向多指标钳制，两边夹击。

**论点**：SCRIBE 的格式约束不只是"换个角度做 credit assignment"，它还**自带反作弊的结构性属性**。一个 reward 设计能不能 work，不在于单一指标多漂亮，在于它的指标集是否**互相钳制到无 Pareto-最优 hack**。这一性质是 SCRIBE 三机制塌缩的隐含第四面——前三面（context / KV-cache / credit）是性能优势，这一面是训练**可行性**优势。

---

## 4. 一个被低估的副产物：显式 world model

## 4. 一个被低估的副产物：显式 state 表征 + 一条 world-model follow-up 通道

### 4.1 SCRIBE 把 state 显式化了，但 transition 还是隐式的

ReAct 假设 LLM 的 hidden state 会自己从整条 trajectory 里隐式归纳出当前任务状态。这对长程任务不成立——hidden state 没有显式 anchor，越往后越漂。

SCRIBE 强制每 step 输出一个**显式的、文本化的、token-grounded 的状态表征**——`<step_summary>`。但要诚实区分两个不同的层级：

|  | 经典 LLM agent (L0) | **SCRIBE (L1)** | 真·显式 world model (L2) |
|---|---|---|---|
| 状态表征 | 隐式（hidden state） | **显式**（step_summary token 序列） | 显式 |
| 转移函数 `(s, a) → s'` | 隐式（attention） | **仍然隐式**（同一个 LLM 内部 attention） | 显式（独立网络） |
| 谁实现 | 一个 LLM 全包 | 一个 LLM，但 state 被强制 surface 到 token level | LLM + 独立 WM 两个模块 |

**SCRIBE 当前的位置是 L1，不是 L2**。说 SCRIBE "内置 world model" 是 over-claim——更准确的说法是：

> **SCRIBE 把 state representation 从 hidden state 搬到了 token stream 上；transition function 仍然在 LLM 黑盒里，但因为它的 input state 现在被 grounding 到 token level，这个隐式 transition 被训得比经典 LLM 的隐式 transition 更可靠、更可监督。**

SCRIBE 的贡献是**state externalization**，不是 world model。但下面这一节会说为什么这件事比单独的 "state representation" 更有后续可能性。

### 4.2 SCRIBE 的隐藏第二贡献：它产出了训练真·world model 的理想数据集

这一节是 follow-up paper 的种子，不在首篇 paper 的 scope 里——但值得在这里讲清楚，因为它解释了"为什么 SCRIBE 不只是又一种 prompt engineering"。

经典 ReAct trajectory 里临时推理、错误猜测、工具噪声、关键事实、状态更新全混在一起，**没法可靠地从中抽取出干净的 `(s, a, s')` 序列**——因为 R/A/O 之间没有结构性分离，"state" 这个概念本身是模糊的。

SCRIBE 不一样：

```
SCRIBE LLM 训练完后，跑大量 trajectory 产出：
  (step_summary₁, tool_calls₁) → step_summary₂
  (step_summary₂, tool_calls₂) → step_summary₃
  ...

每条 trajectory 都自动是一个干净、结构化、token-grounded、direction-neutral 的
                            (s, a, s') 序列数据集
```

这个性质是 SCRIBE 几个设计约束的**联合产物**，不是 ReAct 或 Reflexion 能复现的：

- `<step_summary>` 是 **direction-neutral** 的 → state 是 clean state，不混杂下一步的 intention bias。
- `<step_summary>` 是 **token-grounded** 的 → state 不是模型幻想，是被 observation token 锚定的客观事实。
- `<step_summary>` 是 **scoped to one step** 的 → 每条记录的 (s, a, s') 是原子的、可独立训练的。

于是 follow-up 路径是清晰的：

```
[首篇 paper / 本工作]                  [follow-up paper]

训练 SCRIBE LLM                  →    SCRIBE LLM 产出大量干净 trajectory
↓                                     ↓
state 外化、token-grounded、       从中抽取 (s, a, s') 数据集
direction-neutral                  ↓
↓                                  训练独立的小型 world model
agent 行为更可靠 + 三机制塌缩       (一个几十 M 参数的 transformer)
                                   ↓
                                   model-based planning / MCTS
                                   hallucination detection
                                   inverse planning: (s, s*) → a
```

为什么把 WM 训练留到 follow-up 而不是塞进首篇：

1. 首篇 paper 一个核心 claim（state externalization + 三机制塌缩 + token-level credit）已经足够漂亮，塞 WM 进来会让 scope 失控、审稿人觉得 claim 太多压不住。
2. WM training 自己就是一篇独立 paper 的体量——data ablation、network size scaling、planning benchmark 都要专门设计。
3. 把 WM 当 follow-up 留住，反而**抬高了 SCRIBE 首篇的高度**：SCRIBE 不再是一个孤立工作，而是一个 generative engine——它产出的不只是 agent 行为，是后续一系列工作的**训练数据源**。这是 Voyager → skill library 后续工作链的标准动作。

### 4.3 与 Generative Agents 的对比（修正版）

Generative Agents (Park et al. 2023) 的 memory stream 在结构上和 SCRIBE 的 step_summary 序列有相似性，但有三个本质区别：

| 维度 | Generative Agents | SCRIBE |
|---|---|---|
| 状态更新触发 | importance-score threshold，aperiodic | per-step deterministic |
| reflection vs summary | 合并为一个机制 | 显式分离：reflect = throw-away CoT，summary = retained state |
| 状态写入是否被监督 | 否（emergent behavior） | 是（reflect 拿 loss，summary 复用率可程序算，四指标 anti-hack 钳制） |
| state 序列是否可用作下游 WM 训练数据 | 否（reflection 与 observation 混杂，且 importance-triggered 造成时间稀疏不一致） | 是（per-step、direction-neutral、token-grounded） |

---

## 5. 为什么这件事可能影响基模训练端

如果 SCRIBE 在 agent 任务上 work，下列推论是自然但不平凡的：

1. **如果 token-grounded 显式状态比 hidden-state 隐式状态更适合长程任务**——预训练目标本身就有问题。预训练只教模型"预测下一个 token"，没教模型"在每个语义单元结束时显式 commit 一个状态快照"。`<step_summary>` 这种结构可以作为预训练阶段的辅助目标。
2. **如果"reflect-as-CoT + token-grounded summary"这套机制让 RL 不再需要新算法只需要新格式**——RLHF / RLAIF 的整个范式该被重新审视。现在所有 RL on LLM 工作默认 reward 是 sequence-level 的，是因为没有更好的 grounding 点；SCRIBE 提供了一个，且四指标 anti-hack 钳制让"格式即 reward"在实践中可行。
3. **如果 append-only step_summary 包含足够任务状态**——inference-time 的 KV cache 管理策略要重写。现在的策略（H2O、StreamingLLM 等）都在"猜"哪些 token 重要；SCRIBE 直接告诉系统哪些 token 是状态。
4. **如果"动笔前先 reflect 再 output"这种元能力可以蒸馏到 2B 模型**——它就是一个通用的 self-supervision signal，不止 agent 能用。所有需要"长程一致性"的任务（写代码、写论文、做数学证明）都受益。这一点等价于 System-2 思维在 token level 的显式化。
5. **如果 SCRIBE 的 trajectory 可以作为干净的 `(s, a, s')` 数据集训练独立 world model**（§4.2）——agent 训练范式可以进入一个新阶段：先用 SCRIBE 把"什么是 state"这件事 grounded 到 token，再用产出的 trajectory 训独立 WM，最后用 WM 做 model-based planning / MCTS / inverse planning。SCRIBE 在这条链条上是**数据生成器**，是 model-based agent learning 的入口。

Agent 只是第一个 use case，因为长程任务把 hidden-state-as-implicit-memory 这个隐患放大到了不可忽视——**但这个隐患在所有 LLM 任务里都存在**。

---

## 6. Method 速览

### 6.1 数据格式

详见 `dataset_format.md` 和 `dataset_format_example.jsonl`。要点：

- 一个 step = 一次 user→assistant 往返。
- 一个 step 物理上可能因为夹了 tool_call 被切成多条 assistant 消息，但 `<reflect>` 和 `<step_summary>` 只出现在该 step 的**最后一条** assistant 消息里，紧跟 final answer 之后。
- 跨 step 拼上下文时：`<think>` 由 Qwen 模板的 `loop.index0 > ns.last_query_index` 规则自动剥；`<reflect>` 由我们的一行正则剥：`re.sub(r"<reflect>.*?</reflect>\s*", "", content, flags=re.DOTALL)`。`<step_summary>` 和 final answer 永久保留。

### 6.2 训练

1. **数据生成**：用一个强教师模型（如 Deepseek-v4-flash）按 SCRIBE 格式跑出大量轨迹，自动算 reward 字段（token 复用率程序可算；direction_neutrality / faithfulness 用 LLM-as-judge）。
2. **SFT 阶段**：让 Qwen3.5-2B 学会三段结构（final answer → `<reflect>` → `<step_summary>`）和五步 CoT 写法。
3. **RL 阶段**：reward 加权直接落到被复用的载重 token 上——这是 SCRIBE 相对 GRPO 的核心优势点。预期：信用分配明确，收敛快。

### 6.3 推理时

- 渲染输入：`<think>` 模板自动剥；`<reflect>` 正则剥；其余原样保留。
- 触发二级压缩：当 `s₁...sₙ` 序列累计长度超过阈值，启动 trajectory-level 压缩。压缩策略见下节。

---

## 7. 压缩时机与压缩策略

### 7.1 何时压缩

`React` && `SCRIBE` 什么时候做二级压缩？和 ReAct vs SCRIBE 的区别无关，本质是上下文工程问题：**KV CACHE 复用** 和 **Token 总数** 的 trade-off。建议写一个事件驱动模拟程序，通过蒙特卡洛实验找出最佳压缩策略。

### 7.2 压缩策略

`<step_summary>` 是 **step-level** 的，比 ReAct 的 trajectory-level 压缩信息更密、更无损；但 trajectory-level 层面的规律提取会差一些。因此 SCRIBE 扬长避短：

- **step-level**：尽量简短，只陈述本 step 的 action 和 output 做了什么，复用 output 有用 token，但不破坏通顺性（这是为训练做的设计）。
- **trajectory-level**：需要触发点（快满足压缩条件了），触发点的那一 step 在 `<step_summary>` 末尾追加一行 `[traj] ...`，下一 step 退回纯 step-level。若某 step 直接被压缩跳过了触发点，那次压缩本身就是触发点。

---

## 8. 实验设计

### 实验 1：SCRIBE 触发压缩的时间晚于 ReAct
- 理论分析：事件驱动蒙特卡洛模拟，覆盖不同上下文长度、KV cache 行为、LLM rollout 行为。
- 实测：在真实 agent benchmark 上跑 SCRIBE vs ReAct，记录何时首次触发二级压缩。

### 实验 2：SCRIBE 的总 token 消耗不显著增加
- 预期：SCRIBE 每 step 多一段 `<reflect>` 和 `<step_summary>`，但它们短；累积起来与 ReAct 末尾的 trajectory-level summary tokens 量级相当甚至更少（因为 ReAct 的 summary 是事后补的、要 cover 更长的历史）。
- 实测：埋点统计 `<reflect> + <step_summary>` 总 token 数 vs ReAct 末端 summary token 数。
- 实验采用事件驱动法，`Event` 驱动 `State` 流转：

```python
Event = {Think, Act, Observe, Reflect, Summarize}
State = {ReasonPhase, ActionPhase, ObservationPhase, ReflectionPhase, SummaryPhase}

transition = {
    (ReasonPhase, Act):         ActionPhase,
    (ReasonPhase, Observe):     ObservationPhase,
    (ActionPhase, Observe):     ObservationPhase,
    (ObservationPhase, Act):    ActionPhase,
    (ObservationPhase, Reflect): ReflectionPhase,
    (ReflectionPhase, Summarize): SummaryPhase,
    (SummaryPhase, Think):      ReasonPhase,
}
```
状态-动作图见 `状态动作图.png`。`Think` 时统计上一 `ReflectionPhase` 和 `SummaryPhase` 的 token 消耗，累计即可。

### 实验 3：信息保留度，SCRIBE 的分阶段 S 优于 ReAct 的最终 S
- 预期：ReAct 的 prompt 再好，长上下文 Transformer 的性能有上限。
- 方法：LLM-as-judge 比较 SCRIBE 的 step_summary 序列 vs ReAct 的 trajectory summary，看哪个保留的可验证事实更多。
- 防 prompt 不公平的设计：10 款 AI 各自做 10 轮 loop engineering 打磨 ReAct 压缩 prompt；找 Agent/LLM 领域 10 个 PhD 各给一个 prompt；这 20 个 prompt 与我们设置的 SCRIBE 策略比较。如果 SCRIBE 都打过，实验立。
- LLM-as-judge 采用多款 AI、多次、顺序调换、统计学分析，证明 judge 可信。

### 实验 4：同 round 数下 SCRIBE 总成本 ≤ ReAct
- 理论：同实验 1 的蒙特卡洛模拟，模拟成本。
- 实测：埋点 OpenAI API 金额消耗（如果 API 提供）。

### 实验 5：token-level credit assignment 实证（**新增，强建议加**）
- 预期：SCRIBE + 简化版 RL（reward 加权到载重 token）的收敛速度 / 最终性能 ≥ GRPO（同算力 budget）。
- 方法：固定 base model（Qwen3.5-2B），固定 benchmark（ALFWorld / HotpotQA / SWE-bench Verified），SCRIBE-trained vs GRPO-trained 对比。
- 这是 §5 论点 2 的核心证据。

### 实验 6：world model 性质实证（**新增，强建议加**）
- 预期：如果 step_summary 序列真的是 world model，模型应该能从 `s₁...sₖ` 预测 `s_{k+1}` 之前的 observation 分布。
- 方法：在 SCRIBE-trained 模型上做 next-observation prediction，与 ReAct-trained 同模型对比。
- 这是 §4 主张的直接验证。

### 实验 7：SCRIBE 的 S 对后续推理无方向性影响（先搁置）
- 前面效果好自然说明影响不大；如果效果好但仍想直接验证，可设计 swap-summary 对照实验。

---

## 9. 与已有工作的关系

| 工作 | 与 SCRIBE 的核心区别 |
|---|---|
| **ReAct** (Yao et al. 2022) | ReAct 只显式建模 R→A→O 三步，状态隐式藏在 trajectory 重读里。SCRIBE 把状态外化为 token stream 上的 append-only 段。 |
| **Reflexion** (Shinn et al. 2023) | Reflexion 的 reflection 是 episode 末尾的 verbal self-critique，跨 episode 用；SCRIBE 的 reflect 是 step 内的、为写好 summary 服务的 CoT，跨 step 即剥。 |
| **Generative Agents** (Park et al. 2023) | 见 §4 的对比表：触发机制、reflect/summary 分离、是否被监督，三处本质不同。 |
| **Voyager** (Wang et al. 2023) | skill library 是技能复用，正交于状态表征。 |
| **ExpeL** (Zhao et al. 2024) | 跨 trajectory 的 insight 抽取，是 post-hoc，不是 per-step 状态更新。 |
| **CoALA** (Sumers et al. 2023) | cognitive architecture 框架，描述空间。SCRIBE 是一个具体实例，但其核心贡献（token-level credit + KV-cache friendly + world model 三者合一）不被 CoALA 框架覆盖。 |
| **Self-Notes / Scratchpad** (Nye 2021; Lanchantin et al. 2023) | scratchpad 是临时工作区，不是 append-only retained state；不被设计来满足 token-grounded / direction-neutral 约束。 |

---

## 10. 当前状态

- ✅ 数据格式规范定稿（`dataset_format.md` + `dataset_format_example.jsonl`）。
- ✅ Qwen3.5-2B 选定为蒸馏目标。
- 🚧 数据生成 pipeline（教师 = Deepseek-v4-flash）。
- 🚧 SFT + RL 训练 pipeline。
- ⬜ Benchmark 实验（ALFWorld / HotpotQA / SWE-bench Verified）。

---

## Appendix A：作者原始 GPT 分析（保留备查）

一个长程 agent 每步其实都应该完成五件事：

1. 根据当前状态决定做什么
2. 对外部世界采取行动
3. 接收世界反馈
4. 解释这个反馈意味着什么
5. 更新自己的任务状态

ReAct 只显式建模了前 3 个：

```
Reason → Act → Observe
```

它的问题是把第 4、5 步混回下一轮 Reason 里了。于是所有东西都堆在一条 append-only trace 里：临时推理、错误猜测、工具噪声、关键事实、状态更新，全混在一起。

SCRIBE 把它拆开：

```
Reason → Act → Observe → Reflect → Summarize
```

分工明确：

- Reason：决策
- Act：干预世界
- Observe：世界反馈
- Reflect：解释反馈（throw-away CoT）
- Summarize：提交状态（retained）

从长期任务角度也合理：

- R 是临时计算
- O 是外部噪声/证据
- Reflection 是解释过程
- S 是可保留状态

如果没有 S，agent 的长期状态只能靠完整历史或事后压缩；如果有 S，状态更新在每步自然发生。
