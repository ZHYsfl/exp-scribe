# SCRIBE: Agents Should Write What They Learn

`React` 是姚顺雨的代表作，核心思想是`Agent`思考方式的根基：`Reasoning`,`Action`,`observation`.

```TEXT
原则：R保留一轮
R A O
↓
A O | R A O 
↓
A O | A O | R A O
↓
A O | A O | A O | R A O
↓ 压缩
S 
↓
S | R A O 
↓ 
S | A O | R A O
↓
S | A O | A O | R A O
↓ 压缩
S S
↓ 
...
```

`React`缺少`Reflection`和`Summary`层,Ours `Scribe`:  
```TEXT
R A O R S
↓
A O S | R A O R S
↓
A O S | A O S | R A O R S
↓
A O S | A O S | A O S | R A O R S
↓ 压缩(无延迟)
S S S S
↓
S S S S | R A O R S
↓ 
S S S S | A O S | R A O R S
↓
S S S S | A O S | A O S | R A O R S
↓ 压缩(无延迟)
S S S S S S S
↓ 
S S S S S S S | R A O R S 
↓ 
S S S S S S S | A O S | R A O R S
↓ 压缩(无延迟)
S S S S S S S S S
↓ S序列大到满足压缩条件，压缩(享受公共前缀的KV CACHE)
S S S S S S S S S -> S
```

其中 S 只是 each step 的 summary,通过Reflection，我觉得S能比较好的陈述发生了什么，不做出方向性影响，而且有了reflection,我们也会出现observation里一些重要的token，这有助于信用分配

综上，Scribe 的each step流程是:

`thinking/reasoning` -> `actioning` -> 'observe' -> 'reflection'->'summary'

---

# 压缩时机

`React` && `Scribe` 什么时候压缩？

和`React`&&`Scribe`的区别无关，本质是一个上下文工程问题。`KV CACHE`和`Token`数的trade off.建议写一个模拟程序，通过大量蒙特卡洛实验找出最佳压缩策略。


---

# 压缩策略

如果`Scribe`的`summary`是`step-level`的，应该比`React`的`trajectory-level`信息更多更无损，但是`trajectory-level`层面的规律提取、整理会差一些，而且还有一个很有意思的因素：上下文越长，`Token`数越大，成本越高，成本这么高，就更要做一些“回本”的事，所以`Scribe`也不是每个`step`的`summary`都在`step-level`，还应该在快要满足压缩条件的时候的那轮`summary`聪明加一些`trajectory-level`的东西，下次的`summary`就还是`step-level`即可，只在某个触发点（快压缩了的某个条件满足）的那一轮做一些含着`trajectory-level`的总结。

所以我们`Scribe`的压缩策略要扬长避短：

`step-level`尽量简短，而且只是陈述这轮action和output做了什么（这样才能尽量保证对后续推理无明显方向性影响），而且尽量重复output有用的token（那也不该委屈通顺性）（这是为了训练，后面会讲）。

`trajectory-level`的东西需要一个触发点（快满足压缩条件了），满足的那一轮就不仅简单说说`step-level`,也混杂`trajectory-level`的`summary`。

如果某轮上下文太长，导致直接压缩了，触发点被跳过，那其实这次压缩就是触发点。

---

# 实验:证明 Scribe 的优势

## 实验1：比 React 压缩时间短

理论分析(可做蒙特卡洛模拟实验，模拟不同长度的上下文，kv cache行为以及LLM rollout行为).

\+

具体实测

## 实验2：Token消耗增长不大

预期：
`Scribe`每轮多一个reflection和summary,Summary的性质决定了each step Token消耗增长不大，累计起来和正常`React`的summary Tokens差别不大。
我们只需要比较`Scribe`的`summary tokens`+和`relfection tokens`之和 与`React`的`summary tokens`的大小即可。

在实验一中的具体实测里埋点即可。实验采用事件驱动法，`Event`驱动`State`流转,

`Event`包括 `Think`,`Act`,`Observe`,`Reflect`,`Summarize`

`State`包括 `ReasonPhase`,`ActionPhase`,`ObservationPhase`,`ReflectionPhase`,`SummaryPhase`

状态-动作表：

```python
dict[Tuple['State','Event'],'State'] =
{
    {('ReasonPhase','Act') : 'ActionPhase'},
    {('ReasonPhase','Observe') : 'ObservationPhase'},
    {('ActionPhase','Observe') : 'ObservationPhase'},
    {('ObservationPhase','Act') : 'ActionPhase'},
    {('ObservationPhase','Reflect') : 'ReflectionPhase'},
    {('ReflectionPhase','Summarize') : 'SummaryPhase'},
    {('SummaryPhase','Think') : 'ReasonPhase'},
}
```
/home/zane/ReactS/状态动作图.png
`Think`的时候，看一下上轮`RflectionPhase`和`SummaryPhase`的`token`消费，累计就可以统计了。    

## 实验3：React 最后S比 Scribe 分阶段S压缩总结丢失信息多

预期：`React`的提示词再好，长上下文的`Transformer`性能是有上限的，LLM AS A JUDGE 来证明`Scribe`的S就是比`React`好。

在实验一中的具体实测里埋点即可。为了防止Reviewer说`React`使用的压缩策略提示词不够好，可能是为了迎合实验结果而设置，我们使用10款AI各自进行LOOP ENGINEERNIG打磨提示词10轮，并且找Agent/LLM领域的10个PhD给出他们的提示词，拿最终20个提示词和我们自己设置的`Scribe`的压缩策略的效果比较，发现都打不过我们，那实验就成了。

LLM AS A JUDGE采用多款AI，多次，顺序调换和一些统计学分析来证明JUDGER是可信的。

## 实验4:轮数相同的情况下，Scribe 的综合成本低于 React，或者相差不多

理论分析（还是实验1的那个蒙特卡洛实验，模拟成本即可）

\+

在实验一中的具体实测里埋点即可。看看OPENAI API提不提供金额消耗。

## 实验5：Scribe 的 S 对后续推理无明显方向性影响（先搁置，前面效果好自然说明没影响或者影响不大无关紧要）

---

# `Scribe Agent`训练：

拿一个开源小参数模型（比如`Qwen3.5-2B`),

我们采取经典蒸馏策略，用Deepseek-v4-flash作为我们`Scribe Agent Harness`的基模，
造出来符合我们期望的数据集(带`reward`)。然后对`Qwen3.5-2B`进行`SFT`微调，`RL`微调。
因为每轮都有`reward`，`Scribe`里的`Summary`很多`Token`是`Output`里也有的，信用分配问题天然有所缓解。

预期：比 `GRPO` 学习效率更高（因为信用分配问题解决）

---

# 数据集格式规定：

查看`chat_template.jinja`！


---

`gpt`分析:

一个长程 agent 每步其实都应该完成五件事：

1. 根据当前状态决定做什么
2. 对外部世界采取行动
3. 接收世界反馈
4. 解释这个反馈意味着什么
5. 更新自己的任务状态

ReAct 只显式建模了前 3 个：

Reason -> Act -> Observe

它的问题是把第 4、5 步混回下一轮 Reason 里了。于是所有东西都堆在一条 append-only trace 里：临时推理、错误猜测、工具噪声、关键事实、状态更新，全混在一起。

你的结构把它拆开：

Reason -> Act -> Observe -> Reflect -> Summarize

这里的合理性在于分工明确：

Reason: 决策
Act: 干预世界
Observe: 世界反馈
Reflect: 解释反馈
Summary: 提交状态

这很像一个最小认知控制回路。不是为了多写 token，而是为了让 agent 每一步都有明确的“状态更新”动作。

从长期任务角度也合理：

R 是临时计算
O 是外部噪声/证据
Reflection 是解释过程
S 是可保留状态

如果没有 S，agent 的长期状态只能靠完整历史或事后压缩；如果有 S，状态更新在每步自然发生。

所以我会很明确地说：这个 agent 架构在思想上是站得住的。


  你记得的 k 是我之前提的 hybrid memory 超参数：

  历史 = S_1 S_2 ... S_{t-k-1} + A_{t-k}O_{t-k}S_{t-k} ...
  A_{t-1}O_{t-1}S_{t-1} + current_obs

  - k=0：历史只有 S 序列（最激进）
  - k=1：最近 1 步保留 AOS，其余是 S
  - k=2：最近 2 步保留 AOS
  - k=∞：全部 AOS（v1 当前设计）

  但注意

  v1 的设计里，压缩后的远端历史会变成纯 S。所以 v1
  本身已经有两种历史密度：

  近期：A O S
  压缩后的远期：S

  我提的 k 是更细粒度的控制：

  最近 k 步：A O S
  k 步之前但还没压缩：S
  压缩后：SSSS 或更短

  对 SCRIBE 的意义

  k 可以作为一个消融维度：

  k=0: 纯 S 历史
  k=1: 最近 1 步 AOS
  k=2: 最近 2 步 AOS

  看哪个 k 在你们的任务上效果最好。


 巧妙利用summary做了训练效率优化，上下  
  文扩展，无痛压缩，成本降低，准确率提升 
  ，这么多事                             
  
  Thought for 5s (ctrl+o to expand)

● 对，这就是 SCRIBE 的核心卖点。
   
  一句话总结：
     
  ▎ SCRIBE 让 agent 每步产出的 summary 
  ▎ 同时承担四种角色：训练信号锚点、上下
  ▎ 文压缩器、KV cache 
  ▎ 优化器、状态传递载体。

  ---
  四个价值点

  ┌──────────────┬───────────────────┐
  │     角色     │    解决的问题     │
  ├──────────────┼───────────────────┤
  │ 训练信号锚点 │ token-level       │
  │              │ credit assignment │
  ├──────────────┼───────────────────┤
  │ 上下文压缩器 │ 长 horizon 任务的 │
  │              │  context 爆炸     │
  ├──────────────┼───────────────────┤
  │ KV cache     │ append-only       │
  │ 优化器       │ 历史复用          │
  ├──────────────┼───────────────────┤
  │ 状态传递载体 │ 跨 step           │
  │              │ 的干净状态表示    │
  └──────────────┴───────────────────┘

  ---
  一个更锐利的 paper pitch

  ▎ "We show that a single design choice 
  ▎ — forcing the agent to write a 
  ▎ token-grounded, append-only step 
  ▎ summary — simultaneously improves 
  ▎ credit assignment, context 
  ▎ efficiency, KV cache reuse, and 
  ▎ long-horizon reasoning. SCRIBE turns 
  ▎ the state representation problem into
  ▎ a data format problem."

  这句话可以直接放 abstract。

每个loop设置max_steps,所以一个loop可以有10个可以训练的RL信号项，
每个信号项的reward是iter的reward的考虑折扣因子的和/len，以及summary的一些质量指标以及一些格式指标等，你注意指标越多越容易防止reward hacking。

包括设置了压缩条件，那一旦满足，下次的iter x迭代后，那其实iter0 - iter x这个信号项的轨迹就是S S S S R A O R S.你懂吧。

无论如何训练的时候一定loop就是硬性到max_step。

