# SCRIBE 微调数据集格式说明（v3）

> v3 修订要点（相对 v2）：
> 1. `<summary>` → **`<step_summary>`**。强调这只是"本 step"（一次 user→assistant 往返）内从开头 `<think>` 到 final answer 这一段的总结，**不**总结整条 trajectory、**不**复述以往的 step。
> 2. `<reflect>` 的定位从"对本轮的开放式反思"收紧为：**写好 `<step_summary>` 的 CoT 规划稿**。它是模型"如何写出正确的 step_summary"的思维过程，运行时会丢弃。

---

## 0. 一个 step 的结构（=一次 user→assistant 往返）

```
user 发话
  └─ assistant 在这一 step 内：
       <think>R</think>                       ← 思考
       并行/串行多次  tool_call → tool_response   ← 可能多次 A↔O
       <think>R</think>  (允许再想，再调工具)
       ...
       最终给用户的答案文字 (final answer)
       <reflect>……</reflect>                  ← CoT：怎么写好下面的 step_summary
       <step_summary>……</step_summary>        ← 只总结"本 step"，永久保留
user 再发话
  └─ ……下一个 step
```

"一个 step" 在 SCRIBE 里覆盖的具体内容：**从这一 step 开头的 `<think>` → 中间所有 tool 调用与 observation → 直到刚刚写完的 final answer**。`<step_summary>` 必须严格落在这个范围内，不向前覆盖以往 step，也不向后预测未来 step。

---

## 1. `<reflect>` 的新定位：写 `<step_summary>` 的 CoT

`<reflect>` 不再是"对本轮的开放式自评"，而是**写出正确 `<step_summary>` 的链式思考**。它至少要走完这六步（前五步是写好 summary 的必要思考，第六步是反 reward hacking 的关键防线）：

1. **事实回放** — 用一两句口述这一 step 里实际发生了什么（`<think>` → tool 调用 → observation → final answer）。
2. **载重 token 枚举** — 列出从 tool 输出和自己刚写的 final answer 里**必须复用**的关键 token：文件路径、命令字符串、错误码、数字、函数名、专有名词。明确写出"为什么这些是 load-bearing 的"。
3. **通顺性检查** — 这些 token 拼到一句通顺的中/英文里有没有别扭？哪些适合直接 inline，哪些适合包在反引号 / 括号里？凡是为了塞 token 而写出"扭曲句"的诱惑，必须在 reflect 里被识别并规避。
4. **方向性 / 主观倾向检查** — 列出"我可能会写但**不该写**"的措辞（比如"看起来不错"、"接下来应该…"、"和预期一致"），并显式排除。
5. **范围检查** — 确认 step_summary 只覆盖本 step；如果有上一 step 的 step_summary 可复用结论（如累计计数），只承认"我从 prior step_summary 里读到了某结论"这一事实，不重新总结上一 step。
6. **反 reward hacking 检查** — 显式确认 `<step_summary>` **不是 final answer 的删词版本**。具体步骤：① 在脑子里把 final answer 摆开；② 计划用什么结构重组它（drop 哪些 connective、drop 哪些 user-facing 措辞、drop 哪些 hedging）；③ 估算目标长度（应明显短于 final answer，经验值约 30%–50%）；④ 自问 "如果用 diff 工具对比 final answer 和我的 summary，会不会看起来像 summary 大段被 final answer 包含？"——如果会，重写。

> **不要写 step 编号**。模型在写 `<step_summary>` 时不知道自己处于第几 step——上一 step 的 `<think>` 已经被模板剥离，模型没有可靠的计数源。同理，`<reflect>` 和 `reasoning_content` 里也不写 `Step-1` / `Step-2` 这类编号。要引用上一 step 的产物，统一说 "the prior `<step_summary>`" 或 "the previous `<step_summary>`"。step 编号只在训练侧的 `step_rewards[*].step` 里出现，那是给 trainer 看的。

写完这六步后，`<step_summary>` 的内容基本就是这份 CoT 的"结晶"。

---

## 1.5 closing `<think>` 的特殊职责

一个 step 内会出现多次 `<think>`（每次决定下一步前都会想），但**最后一次 `<think>`**——也就是模型决定不再调工具、准备写 final answer 的那一次——有它独有的、不可替代的职责：

**它是模型在本 step 内最后一次"看得见全部信息"的机会**。下一 step 开始时，这次 closing `<think>` 也会被 Qwen 模板剥离（`loop.index0 > ns.last_query_index` 规则只保留**最新**用户消息之后的 think），模型对本 step 的"私密整合"就彻底没了。所以 closing `<think>` 必须承担：

- **整合**：把本 step 所有 observation 和中间结论拼成一个完整图景。
- **三段规划**：提前在脑子里勾出 final answer、`<reflect>`、`<step_summary>` 三段的形状（关键 token 各放在哪、各段长度大概多少、anti-hack 重组思路是什么），保证三段不漂移。

实践上意味着 closing `<think>` **应该比中间几次 `<think>` 都厚**——中间的 `<think>` 是局部决策（"下一步调哪个工具"），closing `<think>` 是全局收束 + 输出预规划。

数据里如果 closing `<think>` 写得很短、敷衍，那就是失败样本——因为它意味着模型在写 final answer / reflect / summary 时没经过整合，三段会脱节、anti-hack 检查会落空。

> 这样设计的意义：把"写出合规 summary"这件事从隐式期望变成显式可监督的 CoT。训练时 `<reflect>` 也会拿到 loss，让模型学会"先想清楚要复用哪些 token、怎么不带方向，再下笔"。推理时把 `<reflect>` 当 throw-away 计算丢掉，不污染下游 KV-cache。

---

## 2. `<step_summary>` 写作守则

1. **范围**：只陈述本 step 做了什么以及由本 step 直接得到的事实；不复述以往 step，不预测未来 step。
2. **载重 token 复用**：来自 tool 输出（命令、文件名、错误码、数字、专有名词）和本 step final answer 的关键名词，按 `<reflect>` 里的枚举来用。复用的对象是**单 token**（文件名、函数名、数字、专有名词、verbatim 输出片段），不是 n-gram 短语。
3. **通顺性硬约束**：复用 token **不得**以扭曲句法为代价；如果某个 token 塞不顺，宁可包反引号、加括号、改写位置，也不要写出"为了塞而塞"的怪句。
4. **无方向性**：不写主观判断、不写"接下来应该…"、不写"看起来…"。**只陈述发生的事实**。
5. **简短**：1-3 句为主。
6. **trajectory-level 混入**：只在压缩触发点的那一 step，在 `<step_summary>` 末尾追加一行 `[traj] …`，并把该 step 的 `step_rewards[i].trajectory_mixin: true`。下一 step 的 `<step_summary>` 退回纯 step-level。若某 step 直接被压缩跳过了触发点，那次压缩本身就是触发点（README L84）。
7. **反 reward hacking（关键）**：
   - `<step_summary>` 是**压缩后的重组陈述**，**不是 final answer 的剪辑版**。如果你的 summary 可以通过对 final answer 删词得到，就算失败。
   - 长度应明显短于 final answer。和它一样长甚至更长的"summary"不是 summary。
   - 复用的是**载重 token**（receipts、proper nouns、numbers），不是 connectives / hedges / 评价性形容词。复用噪声词来刷复用率属于作弊。
   - **结构重组**：drop user-facing 措辞（"There are"、"All ... pass"）、drop 加粗、drop hedging、drop bullet 结构，把内容压成 1-3 句 declarative 陈述。同样的 token，不同的句式骨架。

---

## 3. 顶层 schema（一行 JSONL = 一条多 step 轨迹）

```jsonc
{
  "id": "scribe-example-0001",
  "task_type": "coding_qa",
  "tools": [ ...OpenAI/Qwen tool schema 数组... ],
  "messages": [ ...见 §4... ],
  "step_rewards": [ ...每 step 一条，见 §6... ],
  "trajectory_reward": 0.96,
  "meta": {
    "base_model": "qwen3.5-2B",
    "teacher": "deepseek-v4-flash",
    "scheme": "SCRIBE-v0",
    "context_assembly": {
      "strip_from_old_assistant_turns": ["<think>", "<reflect>"],
      "keep_from_old_assistant_turns":   ["final_answer_prose", "<step_summary>"]
    }
  }
}
```

---

## 4. `messages` 里各角色

### 4.1 system（仅最前一条）
完整版规范文本见 `dataset_format_example.jsonl` 第一行 `messages[0].content`。两个示例样本的 system 已逐字对齐，可直接作为训练统一的 system prompt 复用。

### 4.2 user
- 真实任务发话即可。
- **不需要**任何"占位 user"。step 边界由 user 真实发话决定。

### 4.3 assistant —— 一个 step 物理上可被切成多条 assistant 消息
受 OpenAI/Qwen 协议限制：每次 tool_call 后必须先收 tool_response 才能继续，所以一个语义上的 step 在物理上是"多条 assistant 消息 + 多条 tool 消息"交错。但**`<reflect>` 和 `<step_summary>` 只出现在该 step 的最后一条 assistant 消息里**，紧跟在 final answer 之后。

**中间 assistant 消息**（决定继续调工具）：
```jsonc
{
  "role": "assistant",
  "reasoning_content": "R: 这一步打算怎么走、为什么调这些工具",
  "content": "",
  "tool_calls": [ ... ]
}
```

**最终 assistant 消息**（决定收尾、给答案、然后 reflect+step_summary）：
```jsonc
{
  "role": "assistant",
  "reasoning_content": "R-final: 信息够了，准备出答案 + reflect + step_summary",
  "content":
    "<这一 step 给用户看的最终答案，plain prose>\n\n"
    "<reflect>\n"
    "1. 事实回放：……\n"
    "2. 载重 token：……（逐项写出为什么是 load-bearing）\n"
    "3. 通顺性：……\n"
    "4. 方向性/主观倾向排除：……\n"
    "5. 范围：本 step 覆盖 X；如有引用 prior step_summary 的结论，注明引用而不复述。\n"
    "</reflect>\n\n"
    "<step_summary>\n"
    "Step-k: …仅陈述本 step 发生的事实，复用上述 load-bearing token，通顺、无方向性。\n"
    "</step_summary>"
}
```

### 4.4 tool（Observe）
```jsonc
{ "role": "tool", "name": "read_file", "content": "...原始返回..." }
```
连续多条 tool 消息会被模板自动拼进同一个 `<tool_response>` 块。

---

## 5. 跨 step 上下文的拼接规则（runtime/trainer 侧）

下一 step 喂模型的上下文 =
- system
- 历史所有 user 原文
- 历史所有 assistant 消息，**每条 assistant 的 content 在喂回前预处理**：

```python
import re
def strip_for_history(content: str) -> str:
    # <think> 由 chat_template.jinja 的 `loop.index0 > ns.last_query_index` 规则自动剥离。
    # 我们只需手动剥 <reflect>...</reflect>，保留 final answer 文本和 <step_summary>...</step_summary>。
    return re.sub(r"<reflect>.*?</reflect>\s*", "", content, flags=re.DOTALL)
```

- 当前正在生成的这一 step 的 assistant 消息**原样不动**。

最终效果：
- `<think>` 仅当前 step 可见 → "R 保留一轮"成立
- `<reflect>` 仅当 step 可见，下一 step 起即被剥离 → reflect 是 step_summary 的 throw-away 规划稿
- `<step_summary>` 和 final answer 永久保留 → 跨 step 状态，KV-cache 公共前缀来源

---

## 6. `step_rewards` —— 每 step 一条

```jsonc
{
  "step": 1,
  "phase_rewards": {
    "reason":        0.95,  // R 是否合理推进（包括 closing <think> 是否做了整合与三段预规划）
    "actions":       1.00,  // 工具选择+参数+是否善用并行
    "final_answer":  0.97,  // 给用户的回答是否对、是否完整
    "reflect":       0.95,  // <reflect> 是否真的当作 step_summary 的 CoT 在用：六步是否齐全
    "step_summary":  0.96   // 下面所有 step_summary_* 维度的加权综合分
  },
  // === 真信号：summary 是否忠实、是否无方向性、是否复用载重 token ===
  "step_summary_faithfulness":         0.99,  // 对本 step 事实的忠实度
  "step_summary_direction_neutrality": 0.99,  // 不引入下 step 偏置
  "step_summary_token_reuse_ratio":    0.74,  // summary 中出现在本 step tool 输出/final answer 里的 token 占比

  // === 防 reward hacking 信号（关键，新增）===
  "step_summary_compression_ratio":              0.38,  // len(summary) / len(final_answer)；> 0.7 = 警告，> 1.0 = 失败
  "step_summary_max_ngram_overlap_with_final_answer": 0.22,  // summary 与 final answer 之间最长公共 n-gram 占 summary 总长比例；高 = 复制嫌疑
  "step_summary_fluency_score":                  0.97,  // LLM-as-judge 给的通顺度 [0,1]；低 = 为塞 token 破坏了句法
  "step_summary_noise_token_reuse_ratio":        0.08,  // summary 中"复用了但不属于载重 token"（connective、hedging、评价词）的 token 占比；高 = 用噪声词刷复用率

  // === 元信号 ===
  "reflect_is_cot_for_summary":        true,   // <reflect> 是否结构化地服务于 <step_summary>（六步是否齐全）
  "parallel_action_bonus":             0.05,   // 本 step 是否合理并行调工具
  "trajectory_mixin":                  false,  // 本 step <step_summary> 是否含 [traj]
  "uses_prior_step_summary":           false,  // 本 step R 是否显式利用了上一 step 的 <step_summary>
  "iterative_tool_rounds":             3,      // 本 step 内 assistant 消息条数
  "notes": "..."
}
```

### 训练侧用法（呼应 README §`Scribe Agent`训练）

**正向信号**（鼓励的行为）：
- `step_summary_token_reuse_ratio` 驱动"对 summary 中**确实出现在本 step tool 输出 / final answer 中**的 token 加权"——SCRIBE 的信用分配优势直接在这里落地。
- `reflect_is_cot_for_summary=true` 的样本是高质量蒸馏样本——示范了"先把 CoT 写清楚再下笔写 summary"。
- `uses_prior_step_summary=true` 的样本鼓励"读 prior step_summary → 直接复用结论而非重做工具调用"。

**负向信号 / 反 reward hacking**（必须过滤或惩罚的样本形态）：
- `step_summary_compression_ratio > 0.7`：summary 几乎和 final answer 一样长。这是最容易出现的作弊形态——直接抄 final answer 当 summary。该样本要么丢弃要么大幅 down-weight。
- `step_summary_max_ngram_overlap_with_final_answer > 0.5`：summary 大段 n-gram 直接来自 final answer。即使复用率高，也是抄袭式作弊。
- `step_summary_fluency_score < 0.85`：为了塞 token 把句子写崩了。这是 token reuse 推过头的失败模式。
- `step_summary_noise_token_reuse_ratio > 0.3`：summary 大量复用 "and"、"is"、"the"、"as expected" 等噪声 token 来刷 reuse_ratio。
- `step_summary_direction_neutrality` 低的样本要 down-weight 或丢弃。

**核心 reward 公式建议**（编排上四个 step_summary 维度，避免单一指标被 hack）：

```python
S = step_summary
base = w_faith * S.faithfulness + w_neutral * S.direction_neutrality + w_reuse * S.token_reuse_ratio
anti_hack_penalty = (
    max(0, S.compression_ratio - 0.7) * p1 +   # 太长惩罚
    max(0, S.max_ngram_overlap - 0.5) * p2 +   # 抄袭惩罚
    max(0, 0.85 - S.fluency_score) * p3 +      # 句法崩坏惩罚
    max(0, S.noise_token_reuse_ratio - 0.3) * p4  # 噪声 token 刷分惩罚
)
step_summary_reward = base - anti_hack_penalty
```

> 任何单一指标作为 reward 都会被 hack——这是 Goodhart's law。SCRIBE 的做法是**让多个指标互相钳制**：要拿 `token_reuse_ratio` 高分必须同时通过 compression / overlap / fluency / noise 四道闸——这四道闸**结构上互不相容**（直接抄 final answer 会过 reuse 但被 overlap 抓；塞噪声会过 reuse 但被 noise 抓；硬塞 token 会过 reuse 但被 fluency 抓），所以无 Pareto-最优 hack 路径。

---

## 7. 验证渲染

```python
from transformers import AutoTokenizer
import json
tok = AutoTokenizer.from_pretrained("/home/zane/ReactS/qwen3.5-2B")
rec = json.loads(open("/home/zane/ReactS/dataset_format_example.jsonl").readline())
print(tok.apply_chat_template(
    rec["messages"], tools=rec["tools"],
    tokenize=False, add_generation_prompt=False))
```

预期：
- 中间 assistant 消息：`<think>R</think>` + `<tool_call>...</tool_call>`，无正文。
- 最终 assistant 消息：`<think>R-final</think>` + final answer 文字 + `<reflect>...</reflect>` + `<step_summary>...</step_summary>`。
- 旧 step 的 `<think>` 已被模板自动剥离；旧的 `<reflect>` 需要 §5 的正则在喂模型前剥掉。

---

## 8. 示例文件

`/home/zane/ReactS/dataset_format_example.jsonl`：

- **`scribe-example-0001`**：两个 step。
  - Step-1：并行 read 两个测试文件 + 一次 pytest → 7 个全过。`<reflect>` 走完五步 CoT，`<step_summary>` 严格只覆盖本 step。
  - Step-2：只新读第三个文件，显式复用 Step-1 的 `<step_summary>` 拿到 prior 7，累加到 11。`<reflect>` 中明确说"我引用了 Step-1 的累计结论而不复述"。
- **`scribe-example-0002-iterative-tools`**：单 step 但内部 3 次 tool 交互（list_dir → 并行 grep×2 → read_file），结尾追加一次 `<reflect>` + `<step_summary>`。`<reflect>` 展示如何把"grep 模式串"识别为载重 token 并用反引号保持通顺。
