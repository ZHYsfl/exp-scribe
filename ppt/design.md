我们的PPT整体用中文，术语用英文。字体、排版、风格高度模仿这个PPT /root/autodl-tmp/ppt/examples。PPT不应是文字的堆砌，而是内容的有效组织排版，以方便我进行进一步的展示，但也不能追求形式主义，内容和形式要找到平衡，我觉得这个PPT /root/autodl-tmp/ppt/examples的这点也值得我们学习。

首先是封面，还是用cover.jpg当背景,/root/autodl-tmp/ppt/cover.jpg
封面的文字格式也是重点学习 /root/autodl-tmp/ppt/examples的。关于封面的主题，就是：SCRIBE : Loop Engineering in Long Horizon Tasks Needs Agent to Write What They Did Each Turn然后标注一些我的个人信息 周浩洋，JLU,SE，这个项目的RA日期是从4.17开始一直持续到现在以及之后等。

你可以先通读一下我们的工作：/root/autodl-tmp/README.md

---

然后就是我们工作的介绍了，第一部分：

01 training free 方法不可能实现loop engineering的token效率提升 这个是每页中上位置放的字，
然后起两页：
/root/autodl-tmp/ppt/pics/training-free-token效率不可能提升-1.png
/root/autodl-tmp/ppt/pics/training-free-token效率不可能提升-2.png

---

02 

03 第三个创新点：算法方面：支持self-evolving的history graph机制

04 算法验证的实验设计和目前执行情况and some future work

---

01 具体的一些想法和信息：

一个核心insight是目前auto research有两种理解：一个是karpathy的https://github.com/karpathy/autoresearch，图片/root/workspace/bot/ppt/progress.png，和翁家翌的HL（启发式学习）https://github.com/Trinkle23897/learning-beyond-gradients，图片/root/workspace/bot/ppt/hl.png，专注于用coding agent对ML调参或者强化学习经典小游戏(比如atari)等可量化评分任务进行调优迭代，整体处在做实验，得反馈，做迭代的执行层。一个是端到端自动写论文的智能体，强调全自动或者human in the loop方式完成自动搜集论文，发现/提出idea,然后设计验证实验方案，写实验代码，执行实验验证分析，最终写论文的端到端闭环。代表https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep。

我们第一个创新点，是提出了auto research系统的5层分层模型和当前的bottleneck：
1.AI/人 提出或定义问题
2.AI/人 搜集学习论文
3.AI/人 提出可能的idea集合
4.AI/人 设计验证实验并执行
5.AI/人 对可行idea撰写论文，撰写rebuttal等写作环节
前者专注于第4层AI/人设计验证实验并执行，后者专注于第1-5的端到端闭环。提升前者有助于后者整体的提升，后者是对前者研究范围的进一步延伸，两者并不矛盾。
1,2,3,5层的关键是context learning,对于CS领域的综述性论文，bottleneck目前就是这些层。
第4层的关键是code出正确的实验得到正确的结果，idea可以批量生成，但idea如何快速有效验证并没有那么容易，对于CS领域的非综述性论文，bottleneck目前是第四层。

请你花个几页去讲明白我们的这些信息。务必高效组织，虽不是文字的堆砌，但是我这些信息每一个字都很关键，保持高效组织的同时要求信息几乎无损！

---

02 
infra层面：
对于第四层来说，需要支持把任何自然语言的idea转化为可执行可验证可迭代的环境

我们的第二个创新点是对于把任何自然语言的idea转化为可执行可验证可迭代的环境，我们设计了一个支持并行探索迭代的infra:框架AgentGenesis，AgentGenesis使用适配器模式，封装很深，使得自然语言转换为环境的过程灵活而简洁，并且设计了自然语言->AgentGenesis出题语言的skill。

使用的技术栈是git worktree,docker,claude code python sdk，gRPC
代码的架构是：/root/workspace/bot/ppt/infra_framework.png
skill包的目录结构：（这里你自己探索目录结构/root/workspace/bot/AgentGenesis/skills/author-agentgenesis-problem）
系统已经能够稳定运行，迭代案例：/root/workspace/bot/ppt/iteration_example.png,/root/workspace/bot/ppt/iteration_example1.png,/root/workspace/bot/ppt/iteration_example2.png这三张图

---

03 
第三个创新点agent架构算法层面创新：


第四层执行层有了把任何自然语言的idea转化为可执行可验证可迭代的环境，但还不够，另一方面需要这个环境有self-evolving能力，或者说复用经验的能力。才能越做越聪明，快速迭代出理想的实验代码。
RL or OPD 需要10^4次迭代，我们只需要10次

为什么我们独特：

- 记忆机制：
唯一跨不同任务+跨同一任务不同分支的共享记忆，
唯一多agent之间图结构作为共享记忆
即：唯一实现跨任务历史经验图复用，其他系统的记忆要么是同任务内的搜索状态（AIDE/OpenEvolve），要么是同任务内文件级别的共享（CORAL），都不支持"新任务遇到旧问题时自动检索历史修复路径"。
- Agent协作模式：当前其他系统只支持空间上的协作，我们支持时空上都进行协作。
- 反馈信号：我们的 feedback 结构是专门为跨任务复用设计的，feedback（反馈）+ strategy（策略）+ concepts（概念标签）+ - result（结果）+ pass_criteria（通过标准）这个五元组，恰好是"检索相似问题"所需的全部信息。其他系统只服务于同任务内，没有长期知识库。


| 维度             | Ours                                       | CORAL(2026.04)                   | AI-Scientist-v2(2025.04)   | AIDE(2025.02)       | OpenEvolve                  | GEPA                        | FunSearch                   | autoresearch        |
| -------------- | ------------------------------------------ | -------------------------------- | -------------------------- | ------------------- | --------------------------- | --------------------------- | --------------------------- | ------------------- |
| **系统定位**       | 多Agent自主编码 + 跨任务经验复用                       | 多 Agent 自主编码+共享文件系统              | 端到端科研自动化                   | ML 工程树搜索 Agent      | 进化式代码优化                     | 反射式 Prompt/代码进化             | 数学发现程序搜索                    | 单 GPU 训练自治          |
| **架构家族**       | **Search-Based + Multi-Agent 融合**          | Multi-Agent + Skill-Based        | Search-Based (Tree)        | Search-Based (Tree) | Search-Based (Evolutionary) | Search-Based (Evolutionary) | Search-Based (Evolutionary) | Sequential Pipeline |
| **搜索拓扑**       | **异步并行 + 历史图检索+Agentic Search**            | 异步并行 + 心跳调度                      | Agentic Tree Search (BFTS) | 解树 (UCB/贪婪)         | MAP-Elites + 岛屿迁移           | Pareto 前沿 + 反射突变            | 遗传程序搜索 ( islands )          | 线性迭代                |
| **记忆架构**       | **共享图结构记忆 (History Graph)**                | 共享文件系统池 `.coral/public/`         | 树节点状态                      | 解树节点                | MAP-Elites 网格 + 档案库         | Pareto 候选池                  | 程序档案库                       | Git 历史              |
| **知识复用范围**     | **跨任务、跨会话、可累积**                            | 同任务内共享                           | 同任务内树回溯                    | 同任务内树回溯             | 同任务内岛屿迁移                    | 同任务内 Pareto 合并              | 同任务内遗传继承                    | 无                   |
| **进化机制**       | **Embedding 相似度 + LLM 合并决策**               | 心跳触发 (reflect/consolidate/pivot) | 树扩展 + Debug 回溯             | Patch 生成 + 剪枝       | LLM 变异 + 双选择                | LLM 反射诊断 + 定向突变             | 随机组合 + 评估筛选                 | Agent 自主编辑          |
| **Agent 协作模式** | **并行竞争 + 图记忆共享**                           | 并行自主 + 元控制器心跳                    | 单 Agent 树探索                | 单 Agent 树探索         | 岛屿隔离 + 定期迁移                 | 单线程候选进化                     | 分布式岛屿                       | 单 Agent             |
| **反馈深度**       | **结构化 (feedback/strategy/result/concept)** | 复合评分 + Agent notes               | 实验指标 + 报错信息                | 评估指标                | 标量评分 + Artifacts            | 执行轨迹全量反射                    | 标量评分                        | 标量 (val_bpb)        |
| **生命周期覆盖**     | s4 (Coding)                                | s4 (Coding)                      | s1-s5 (Creation→Writing)   | s4 (Coding)         | s4 (Coding)                 | s4 (Coding/Config)          | s4 (Coding)                 | s4 (Coding)         |
| **核心差异点**      | **唯一实现跨任务历史经验图复用**                         | 最强多 Agent 基础设施                   | 唯一产出审稿级论文                  | 4x 奖牌率的树搜索          | 质量多样性进化                     | 反射优于 RL 的优化                 | 数学定理发现                      | 极简自治                |


算法的机制讲解图：/root/workspace/bot/ppt/history_graph.png(你先放到PPT里，我看要是图东西太多看不清的话，到时候再说，可能会换/root/workspace/bot/ppt/add_logic.png和/root/workspace/bot/ppt/use_logic.png这两个子图)。

---

04 实验设计和future work

benchmark: openml-cc18-ag（72个task）

实验1：pass_criteria的确定：

每个实验先初始criteria是accurary>=0.8，然后max_iteration设置为5次，stop_on_pass设置为false.

threshold = mean(best_score of 3 times)

最终结果：

| Dataset                                    | Repeats | Best Scores               | Threshold | Min      | Max      |
| ------------------------------------------ | ------- | ------------------------- | --------- | -------- | -------- |
| 001_kr_vs_kp                               | 3       | 99.6875, 99.6875, 100.0   | 99.7917   | 99.6875  | 100.0000 |
| 002_letter                                 | 3       | 97.5, 97.8, 97.7          | 97.6667   | 97.5000  | 97.8000  |
| 003_balance_scale                          | 3       | 100.0, 100.0, 100.0       | 100.0000  | 100.0000 | 100.0000 |
| 004_mfeat_factors                          | 3       | 96.5, 95.5, 96.5          | 96.1667   | 95.5000  | 96.5000  |
| 005_mfeat_fourier                          | 3       | 87.0, 85.0, 82.0          | 84.6667   | 82.0000  | 87.0000  |
| 006_breast_w                               | 3       | 97.1429, 97.1429, 97.1429 | 97.1429   | 97.1429  | 97.1429  |
| 007_mfeat_karhunen                         | 3       | 97.0, 96.0, 96.5          | 96.5000   | 96.0000  | 97.0000  |
| 008_mfeat_morphological                    | 3       | 83.0, 81.0, 81.0          | 81.6667   | 81.0000  | 83.0000  |
| 009_mfeat_zernike                          | 3       | 83.5, 92.0, 80.5          | 85.3333   | 80.5000  | 92.0000  |
| 010_cmc                                    | 3       | 100.0, 100.0, 100.0       | 100.0000  | 100.0000 | 100.0000 |
| 011_optdigits                              | 3       | 98.7544, 98.7544, 98.7544 | 98.7544   | 98.7544  | 98.7544  |
| 012_credit_approval                        | 3       | 84.058, 86.9565, 85.5072  | 85.5072   | 84.0580  | 86.9565  |
| 013_credit_g                               | 3       | 76.0, 79.0, 73.0          | 76.0000   | 73.0000  | 79.0000  |
| 014_pendigits                              | 3       | 99.4545, 99.5455, 99.6364 | 99.5455   | 99.4545  | 99.6364  |
| 015_diabetes                               | 3       | 77.9221, 81.8182, 75.3247 | 78.3550   | 75.3247  | 81.8182  |
| 016_spambase                               | 3       | 95.4447, 95.8785, 95.2278 | 95.5170   | 95.2278  | 95.8785  |
| 017_splice                                 | 3       | 97.1787, 96.2382, 96.8652 | 96.7607   | 96.2382  | 97.1787  |
| 018_tic_tac_toe                            | 3       | 95.8333, 96.875, 100.0    | 97.5694   | 95.8333  | 100.0000 |
| 019_vehicle                                | 3       | 83.5294, 83.5294, 82.3529 | 83.1373   | 82.3529  | 83.5294  |
| 020_electricity                            | 3       | 94.594, 91.3063, 93.4466  | 93.1156   | 91.3063  | 94.5940  |
| 021_satimage                               | 3       | 92.535, 92.535, 92.535    | 92.5350   | 92.5350  | 92.5350  |
| 022_eucalyptus                             | 3       | 77.027, 63.5135, 66.2162  | 68.9189   | 63.5135  | 77.0270  |
| 023_sick                                   | 3       | 99.2063, 98.6772, 98.9418 | 98.9418   | 98.6772  | 99.2063  |
| 024_vowel                                  | 3       | 98.9899, 96.9697, 98.9899 | 98.3165   | 96.9697  | 98.9899  |
| 025_isolet                                 | 3       | 93.8462, 93.8462, 93.8462 | 93.8462   | 93.8462  | 93.8462  |
| 026_analcatdata_authorship                 | 3       | 98.8235, 100.0, 100.0     | 99.6078   | 98.8235  | 100.0000 |
| 027_analcatdata_dmft                       | 3       | 18.75, 16.25, 16.25       | 17.0833   | 16.2500  | 18.7500  |
| 028_mnist_784                              | 3       | 97.1143, 98.4571, 97.8429 | 97.8048   | 97.1143  | 98.4571  |
| 029_pc4                                    | 3       | 91.7808, 93.1507, 91.0959 | 92.0091   | 91.0959  | 93.1507  |
| 030_pc3                                    | 3       | 90.4459, 92.3567, 91.0828 | 91.2951   | 90.4459  | 92.3567  |
| 031_jm1                                    | 3       | 81.5427, 81.8182, 81.4509 | 81.6039   | 81.4509  | 81.8182  |
| 032_kc2                                    | 3       | 81.1321, 81.1321, 81.1321 | 81.1321   | 81.1321  | 81.1321  |
| 033_kc1                                    | 3       | 87.2038, 87.2038, 87.6777 | 87.3618   | 87.2038  | 87.6777  |
| 034_pc1                                    | 3       | 90.991, 91.8919, 90.991   | 91.2913   | 90.9910  | 91.8919  |
| 035_adult                                  | 3       | 87.2467, 87.1034, 87.2876 | 87.2126   | 87.1034  | 87.2876  |
| 036_Bioresponse                            | 3       | 82.1809, 82.1809, 82.4468 | 82.2695   | 82.1809  | 82.4468  |
| 037_wdbc                                   | 3       | 98.2456, 100.0, 98.2456   | 98.8304   | 98.2456  | 100.0000 |
| 038_phoneme                                | 3       | 91.1275, 90.7579, 89.8336 | 90.5730   | 89.8336  | 91.1275  |
| 039_qsar_biodeg                            | 3       | 89.6226, 89.6226, 90.566  | 89.9371   | 89.6226  | 90.5660  |
| 040_wall_robot_navigation                  | 3       | 100.0, 100.0, 100.0       | 100.0000  | 100.0000 | 100.0000 |
| 041_semeion                                | 3       | 97.5, 97.5, 97.5          | 97.5000   | 97.5000  | 97.5000  |
| 042_ilpd                                   | 3       | 76.2712, 76.2712, 77.9661 | 76.8362   | 76.2712  | 77.9661  |
| 043_madelon                                | 3       | 78.0769, 75.0, 83.8462    | 78.9744   | 75.0000  | 83.8462  |
| 044_nomao                                  | 3       | 97.8532, 97.8532, 97.7082 | 97.8049   | 97.7082  | 97.8532  |
| 045_ozone_level_8hr                        | 3       | 94.8819, 94.8819, 94.8819 | 94.8819   | 94.8819  | 94.8819  |
| 046_cnae_9                                 | 3       | 95.3704, 93.5185, 97.2222 | 95.3704   | 93.5185  | 97.2222  |
| 047_first_order_theorem_proving            | 3       | 61.9281, 62.4183, 61.6013 | 61.9826   | 61.6013  | 62.4183  |
| 048_banknote_authentication                | 3       | 100.0, 100.0, 100.0       | 100.0000  | 100.0000 | 100.0000 |
| 049_blood_transfusion_service_center       | 3       | 80.0, 78.6667, 77.3333    | 78.6667   | 77.3333  | 80.0000  |
| 050_PhishingWebsites                       | 3       | 97.1067, 97.1971, 97.2875 | 97.1971   | 97.1067  | 97.2875  |
| 051_cylinder_bands                         | 3       | 87.037, 85.1852, 85.1852  | 85.8025   | 85.1852  | 87.0370  |
| 052_bank_marketing                         | 3       | 91.3976, 90.9774, 91.2428 | 91.2060   | 90.9774  | 91.3976  |
| 053_GesturePhaseSegmentationProcessed      | 3       | 66.1943, 67.8138, 67.8138 | 67.2740   | 66.1943  | 67.8138  |
| 054_har                                    | 3       | 97.6699, 98.932, 98.6408  | 98.4142   | 97.6699  | 98.9320  |
| 055_dresses_sales                          | 3       | 68.0, 70.0, 60.0          | 66.0000   | 60.0000  | 70.0000  |
| 056_texture                                | 3       | 99.8182, 99.4545, 100.0   | 99.7576   | 99.4545  | 100.0000 |
| 057_connect_4                              | 3       | 83.9106, 84.8579, 83.0225 | 83.9303   | 83.0225  | 84.8579  |
| 058_MiceProtein                            | 3       | 100.0, 100.0, 100.0       | 100.0000  | 100.0000 | 100.0000 |
| 059_steel_plates_fault                     | 3       | 84.6154, 81.0256, 85.1282 | 83.5897   | 81.0256  | 85.1282  |
| 060_climate_model_simulation_crashes       | 3       | 98.1481, 100.0, 98.1481   | 98.7654   | 98.1481  | 100.0000 |
| 061_wilt                                   | 3       | 97.5207, 98.3471, 98.1405 | 98.0028   | 97.5207  | 98.3471  |
| 062_car                                    | 3       | 100.0, 99.422, 100.0      | 99.8073   | 99.4220  | 100.0000 |
| 063_segment                                | 3       | 93.9394, 92.6407, 93.0736 | 93.2179   | 92.6407  | 93.9394  |
| 064_mfeat_pixel                            | 3       | 97.0, 97.0, 97.0          | 97.0000   | 97.0000  | 97.0000  |
| 065_Fashion_MNIST                          | 3       | 88.0, 88.0714, 88.0       | 88.0238   | 88.0000  | 88.0714  |
| 066_jungle_chess_2pcs_raw_endgame_complete | 3       | 83.5788, 82.5078, 84.4489 | 83.5118   | 82.5078  | 84.4489  |
| 067_numerai28_6                            | 3       | 53.2911, 52.8758, 51.9934 | 52.7201   | 51.9934  | 53.2911  |
| 069_CIFAR_10                               | 3       | 53.2333, 41.5, 41.5       | 45.4111   | 41.5000  | 53.2333  |
| 070_Internet_Advertisements                | 3       | 97.8659, 97.2561, 98.4756 | 97.8659   | 97.2561  | 98.4756  |
| 071_dna                                    | 3       | 96.5517, 96.5517, 95.9248 | 96.3427   | 95.9248  | 96.5517  |
| 072_churn                                  | 3       | 95.8, 92.8, 96.4          | 95.0000   | 92.8000  | 96.4000  |


future work(接下来要做的实验):

实验2：

证明跨任务复用历史经验比不复用好。（其他系统都是同任务经验复用，无法对比）

iterations_to_first_pass：首次达到 threshold 的迭代数（越小越好）
success_rate@K：K 次迭代内达到 threshold 的比例（K=3, 5, 10）
best_score：迭代预算内的最高 accuracy
graph_compression：raw_commits / merged_nodes（merge 压缩比）

打乱数据集，然后对一个seed的顺序跑：
复用历史经验：
记录each task开始之前的压缩比graph_compression，each task在max_iteration=10的情况下的iterations_to_first_pass,success_rate@K,best_score
不复用历史经验：
记录each task在max_iteration=10的情况下的iterations_to_first_pass,success_rate@K,best_score

重复三次。

最后若
each task在max_iteration=10的情况下的iterations_to_first_pass平均值复用比不复用更低，success_rate@K平均值复用比不复用更高，best_score平均值复用比不复用更高，则说明复用经验有效。


实验3：

核心问题：当 3 个 branch 同时启动、同时探索时，它们产生的历史能否实时互相补充，从而整体比 3 个独立 branch 更快找到 pass 策略？

验证多 branch 同时并行时，共享同一个 history graph 是否比各自独立进化更快达到 threshold，验证history graph比共享文件系统的coral好。

## 实验条件

| 条件 | 描述 | 共享机制 |
|---|---|---|
| **Independent** | 3 个 branch 完全独立，各自维护空历史图，互不共享 | 无 |
| **CORAL** | CORAL 原生运行，3 个 agent 同任务共享 `.coral/public/` | 文件系统共享 |
| **Shared Graph (Ours)** | 3 个 branch 同时启动，共享同一个 merged history graph | 结构化图记忆（embedding + merge + relevance） |
## 方法

对每个数据集 × 每个条件：
1. 跑 **3 组重复**（每组都是全新的 graph）
2. 每组内启动 **3 个并行 branch**（`num_nodes=3`）
3. 3 个 branch **同时启动**（无时间差）
4. 每个 branch：`max_iter=10`
5. 记录每个 branch 的 `iterations_to_first_pass`

## 指标
- `iterations_to_first_pass_by_branch`：按 branch_0 / branch_1 / branch_2 分别统计
- `avg_iterations_to_pass`：3 个 branch 的平均
- `min_iterations_to_pass`：3 个 branch 中最快达到 pass 的迭代数
- `best_score_of_all_branch`:3个branch中最好成绩

## 预期结论

即使同时启动，由于各 branch 探索路径不同，产生的历史可以互相补充，指标说明Shared Graph (Ours) > CORAL共享文件系统 > independent


实验4：

目标：验证 LLM merge 不是简单拼接，而是真正聚合出有意义的通用经验。通过对比 merged 节点与其原始节点，评估 merge 的信息增益。

### 定量分析

1. 在实验一（Full 条件）中，收集所有发生 merge 的事件
2. 对每个 merged 节点，通过 `merged_from` 中的 commit hash 从 git log 找回原始节点
3. 计算以下指标：

| 指标 | 定义 |
|---|---|
| `retrieval_hit_rate_merged` | merged 节点在后续迭代中被检索命中的次数 / 总检索次数 |
| `retrieval_hit_rate_raw` | 假设原始节点未被 merge，它们在后续迭代中被检索命中的次数（模拟） |
| `compression_ratio` | `len(raw_nodes) / len(merged_nodes)` |
| `concept_coverage` | merged 节点的 concepts 并集大小 / 原始节点 concepts 并集大小 |

### 定性分析（抽样）

随机抽取 20 组 merge 事件，人工/LLM 评估：

1. **Feedback 质量**：merged 的 feedback_text 是否比原始节点更 general、更准确地描述问题模式？
2. **Strategy 质量**：merged 的 strategy 是否融合了两种具体做法，形成更通用的修复思路？
3. **信息损失**：原始节点中是否有独特且重要的信息在 merge 后被丢失？

评估采用 3 分制：
- 1 分：merge 质量差，信息丢失或拼接生硬
- 2 分：merge 质量一般，无明显增益也无明显损失
- 3 分：merge 质量好，真正提炼出通用模式

## 预期结论

- `retrieval_hit_rate_merged > retrieval_hit_rate_raw`：merge 后的节点更容易被后续检索命中（因为更 general）
- `compression_ratio > 2`：merge 显著压缩了历史空间
- 定性评分均值 > 2.0：LLM merge 确实提炼出了有意义的通用经验

---

最后谢谢聆听，然后Q&A