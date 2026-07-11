GLM-5.2 为什么在 long-horizon Agentic RL 上从 GRPO 回到 PPO？
ThisIsSoMe
ThisIsSoMe
不学无术的数学系学生
761 人赞同了该文章
过去一年半，很多人对大模型强化学习的叙事是这样的：

PPO 是上一代 RLHF 的老工程，昂贵、复杂、难调；
GRPO 去掉 critic，更轻、更适合推理模型；
所以未来大模型 RL 会越来越 GRPO 化。
这个判断并不算错，但它漏掉了一个关键前提：任务形态。

在数学题、代码题、可验证问答这类相对短程、单轮、结果清晰的 RLVR 任务里，GRPO 的确是非常漂亮的工程折中。但到了 long-horizon agentic RL，也就是模型要连续多轮调用工具、观察环境、压缩上下文、拆分子任务、执行几十甚至上百步时，问题的结构变了。

智谱 GLM-5.2 的技术博客里提到，他们在长程任务训练中从 group-wise optimization 转向了 critic-based PPO：长轨迹经过 compaction 后会被拆成多个 sub-trace，同一个 prompt 下不同 rollout 产生的可训练轨迹数量不同、长度也高度不均匀。因此他们用 critic 估计 token-level advantage，而不是依赖组内相对比较。

这件事值得认真看。它不是简单的“PPO 复辟”，也不是“GRPO 失败”。更像是提醒我们：PPO 和 GRPO 并不是谁淘汰谁，而是适配不同 credit assignment 结构的两种工具。

一、GRPO 的成功，来自它抓住了推理 RL 的主要矛盾
GRPO 之所以火，是因为它解决了 PPO 在大模型 RL 上最痛的几个问题。

传统 PPO 需要 actor、reference model、reward model、critic/value model 等组件。尤其 critic 很麻烦：要训练一个 value model，估计每个 token 或 state 的价值。它不仅吃显存、吃算力，还会带来额外的不稳定性。（PPO的理论知识推荐人人都能看懂的RL-PPO理论知识，深入浅出）

而 GRPO 的思路很直接：

对同一个 prompt 采样多个回答，组成一个 group。然后比较这些回答的奖励：

比组内平均好，就增强；比组内平均差，就削弱。

这个设计非常适合一类任务：同一个题目可以采样多个答案，每个答案最终能被 verifier 判断好坏。

比如数学题：

prompt 相同；
采样 8 个解法；
答案对就是高 reward，答案错就是低 reward；
组内相对比较能提供很强的学习信号；
不需要 critic，也能知道哪个样本更值得强化。
DeepSeek-R1 这类推理模型的训练范式，让 GRPO 获得了极高的关注度。它的美感在于：用更多采样换掉一个复杂的 critic。

对于短程 RLVR，这很划算。

二、但 long-horizon agentic RL 不是“多采几个答案”这么简单
Agentic RL 的任务形态完全不同。

模型不再是一次性输出答案，而是在环境中连续行动：

观察 -> 思考 -> 调工具 -> 得到反馈 -> 再思考 -> 再行动 -> ...
比如一个 coding agent 修 bug：

读 issue；
搜索代码；
打开文件；
推断原因；
修改代码；
跑测试；
测试失败；
重新定位；
再修改；
最终通过。
最后成功了，奖励是正的。但中间哪一步真正关键？哪一步是绕路？哪一步虽然看起来没用，但实际上提供了必要信息？这就是 long-horizon RL 的核心难题：credit assignment。

GRPO 在这里会遇到几个不舒服的地方。

第一，group 不再整齐。

在数学题里，同一个 prompt 采样 8 个答案，每个答案都是一个完整 response，长度差异有，但结构相似。

而在 agentic RL 中，同一个 prompt 下：

有的 rollout 5 步就完成；
有的 rollout 50 步还在绕；
有的中途调用搜索；
有的进入代码执行；
有的经过上下文 compaction；
有的被拆成多个 sub-trace；
有的轨迹很短但质量高，有的轨迹很长但最终成功。
这时候“同一 prompt 的多个 rollout 做组内归一化”会变得别扭。组内样本不再是同质的回答，而是形态差异很大的交互历史。

第二，最终 reward 太粗。

如果一个长轨迹最终成功，是否意味着中间每个动作都应该被强化？显然不是。

如果一个长轨迹最终失败，是否意味着中间每个动作都应该被惩罚？也不对。它可能前 80% 都很好，只是最后一步工具调用参数错了。

GRPO 用最终结果做组内相对 advantage，在短程任务里足够有效；但在长程任务里，它容易把一个 episode 的整体成败过度传播到局部 token 和局部 action 上。

第三，compaction 改变了训练样本结构。

GLM-5.2 提到的关键点在这里：长轨迹会因为上下文限制或训练组织需要被压缩、拆分成多个 sub-trace。不同 rollout 拆出来的 sub-trace 数量不同，长度也不同。

这对 GRPO 很不友好，因为 GRPO 天然依赖“组”的结构：同一个 prompt 下采样 G 个候选，然后做相对比较。

但如果 rollout 被拆成了不等长、不等数的训练轨迹，那么训练对象已经从“完整回答”变成“轨迹片段”。这时，critic-based PPO 的优势就出来了：它可以对单条 rollout、单个 sub-trace、甚至 token-level 估计 advantage。

换句话说，PPO 对不规则轨迹更自然，利用效率更高。

三、PPO 的 critic 不是包袱，而是 long-horizon 下的价格
很多人讨厌 PPO，主要讨厌的是 critic。

critic 难训、占资源、引入 bias，还可能和 actor 发生训练冲突。GRPO 去掉 critic 后，工程体验会清爽很多。

但在 long-horizon agentic RL 里，critic 的价值重新变大了。

因为 critic 本质上是在回答一个问题：

走到当前这个状态，未来成功的期望有多大？
这正是长程任务最需要的信号。

对于一个 agent 来说，某一步动作本身未必立刻带来 reward。比如“打开某个文件”这一步没有最终奖励，但如果它显著提高了后续修复成功率，那么它就是好动作。

只有最终 outcome reward 的 GRPO，很难细粒度地区分这种中间状态的价值。而 critic 可以学习：

当前 observation 是否接近成功；
某类工具调用是否把任务推进了；
某段推理是否进入了错误分支；
哪些 token/action 对未来 reward 的贡献更高。
当然，critic 不是免费午餐。它是用复杂度换 credit assignment 能力。

所以更准确的说法不是：

PPO 比 GRPO 更强。
而是：

当任务的 horizon 变长、轨迹变不规则、局部决策价值变重要时，critic 的成本开始变得值得支付。
四、重新理解 PPO：它不是旧时代遗留物
PPO 在大模型领域的名声有点微妙。

一方面，InstructGPT、ChatGPT 时代的 RLHF 让 PPO 成为经典方案；另一方面，后来大家发现 PPO 太重，训练链路复杂，于是 DPO、IPO、KTO、ORPO、GRPO 等方法不断出现，PPO 好像成了“能不用就不用”的东西。

但这更像是工程周期里的钟摆。

PPO 的核心优点一直都在：

它支持单 rollout 学习；
它不要求同一 prompt 下有固定数量的候选；
它可以结合 value function 做 token-level / step-level credit assignment；
它对连续交互、多步决策、环境反馈这类经典 RL 结构更自然；
它和异步 rollout、trajectory replay、长轨迹切分的兼容性更好。
这些优点在单轮推理题上不明显，甚至显得笨重。但在 agentic task 上，它们重新变成优势。

因此，GLM-5.2 的选择并不是“回到过去”，而是承认了一个事实：

Agentic RL 更像真正的强化学习，而不只是带 verifier 的 rejection sampling plus policy optimization。
只要任务越来越接近“智能体在环境中行动”，PPO 这类带 value estimation 的方法就很难彻底退出舞台。

五、重新理解 GRPO：它不是万能 RL 算法
GRPO 的成功有很强的场景依赖。

它适合：

单轮或短程任务；
同一 prompt 可以采样多个候选；
reward 比较可靠；
输出之间可比性强；
不太需要复杂的中间 credit assignment；
工程资源希望尽量省；
verifier 能稳定判断最终结果。
这正好覆盖了数学、代码、逻辑推理等 RLVR 任务的一大块核心场景。

但它不天然适合：

多轮交互；
超长轨迹；
工具调用；
动态环境反馈；
上下文压缩；
不同 rollout 长度差异巨大；
中间动作价值强烈依赖未来状态的任务。
这不是 GRPO 的缺陷，而是它的设计边界。

所有“去 critic”的方法，本质上都在做一个交易：牺牲一部分状态价值建模能力，换取训练简单、显存节省、吞吐更高。

在短程可验证任务里，这笔交易很赚；在长程 agentic 任务里，这笔交易未必赚。

六、真正的问题不是 PPO vs GRPO，而是 credit assignment 怎么做
我觉得这次 GLM-5.2 最有启发的地方，不是“PPO 赢了 GRPO”，而是把大家的注意力重新拉回强化学习最古老的问题：

一个最终结果，应该如何分配给中间的每一步？
短程推理任务让我们一度产生错觉：只要最终答案可验证，RL 就很好做。采样多个答案，正确的上调，错误的下调，模型就能变强。

但 agentic RL 把问题重新变难了。

一个 agent 的成功不是单个回答的成功，而是一条行为链的成功。行为链里有探索、观察、试错、工具使用、计划调整、上下文管理。最终 reward 只是链条末端的一个信号。

未来的关键方向大概率包括：

更好的 critic；
process reward model；
step-level verifier；
trajectory-level 与 token-level 混合优化；
分层 credit assignment；
对工具调用和自然语言 token 分别建模；
对长轨迹进行 compaction 后仍保持奖励一致性；
异步 rollout 和训练稳定性的系统工程。
PPO 和 GRPO 都只是这个大问题下的两个局部解法。

七、一个更实际的判断框架
如果我们把场景拆开看，选择会更清楚。

如果你训练的是数学、代码、问答这类任务：

每个 prompt 可以采样多个答案；
最终 reward 准确；
轨迹长度相对可控；
不太需要环境多轮交互；
那么 GRPO 仍然非常有吸引力。它简单、高效、吞吐好，而且已经被大量推理模型验证过。

如果你训练的是 coding agent、browser agent、GUI agent、research agent：

轨迹很长；
中间动作重要；
工具调用频繁；
rollout 长度差异大；
有上下文压缩和子轨迹拆分；
同一个 prompt 下的样本结构不整齐；
那么 PPO、actor-critic 或其他 value-based / critic-based 方法就会重新变得重要。

一句话总结：

GRPO 适合“比较多个答案”；PPO 更适合“评估一条行为链中的每一步”。
八、结语：别把算法当宗教
大模型领域很容易把某个方法迅速神化。

DeepSeek-R1 火了，GRPO 就像变成了“新标准”；现在 GLM-5.2 在 long-horizon agentic RL 里使用 PPO，又可能有人说“PPO 才是正统”。

这两种看法都太粗糙。

GRPO 的意义，是证明了在一大类可验证推理任务上，我们可以不用昂贵的 critic，也能做出强大的 RL 训练。它把推理模型 RL 的门槛大幅降低了。

PPO 的意义，是在更复杂、更接近真实智能体的场景里，仍然提供了一套稳健的 credit assignment 框架。它贵，但它贵得有理由。

所以，GLM-5.2 这次选择 PPO，不是对 GRPO 的否定，而是对任务结构的尊重。

未来的大模型 RL 可能不会收敛到一个统一算法。更可能的格局是：

短程 RLVR：GRPO、RLOO、REINFORCE++ 等轻量方法继续占优；
长程 Agentic RL：PPO、分层 PPO、critic-based 方法重新重要；
复杂系统训练：不同任务、不同轨迹形态混合使用不同优化目标；
真正的竞争点从单个算法，转向 rollout 系统、reward 设计、轨迹组织和 credit assignment。
PPO 没有过时，GRPO 也没有失灵。

只是当模型从“答题者”变成“行动者”，强化学习又回到了它最难、也最有意思的地方：如何让一次漫长行动中的每一步，都学到它该学的东西。



参考