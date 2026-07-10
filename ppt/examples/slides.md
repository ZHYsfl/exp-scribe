---
theme: seriph
background: ./cover.jpg
title: "Self-Evolving Auto Research 系统"
class: text-center
transition: slide-left
mdc: true
download: true
---

<style scoped>
.slidev-layout {
  background-image: url('/cover.jpg') !important;
  background-size: cover !important;
  background-position: center !important;
  background-repeat: no-repeat !important;
  width: 100% !important;
  height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
  color: white !important;
  text-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
}
.slidev-layout h1 {
  font-size: 3.5rem !important;
  font-weight: 500 !important;
  margin-bottom: 0.5rem !important;
  color: white !important;
}
.slidev-layout h2 {
  font-size: 1.5rem !important;
  font-weight: 400 !important;
  margin-bottom: 2rem !important;
  color: rgba(255,255,255,0.9) !important;
}
.slidev-layout .info {
  font-size: 1rem !important;
  line-height: 1.8 !important;
  color: rgba(255,255,255,0.85) !important;
  margin-top: 1.5rem !important;
}
</style>

# Self-Evolving Auto Research 系统

## 从执行层优化到跨任务经验复用

<div class="info">
周浩洋 · Jilin University, School of Software<br>
Research Assistant · 2025.04 — Present
</div>

---
layout: center
class: text-center
transition: slide-left
title: "目录"
---

<h1 style="color: #5a7a8a; font-weight: 400; font-size: 2.5rem; margin-bottom: 4rem;">目录</h1>

<div style="display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap;">

<div style="text-align: left; max-width: 240px;">
<div style="font-size: 3.5rem; font-weight: 300; color: #93c5fd; line-height: 1; margin-bottom: 0.5rem;">01</div>
<div style="font-size: 1.3rem; font-weight: 500; color: #1e40af; margin-bottom: 0.5rem;">创新点1 · 洞察：5层模型与瓶颈</div>
<div style="font-size: 0.95rem; color: #6b7280;">Auto Research 分层 · 执行层 bottleneck</div>
</div>

<div style="text-align: left; max-width: 240px;">
<div style="font-size: 3.5rem; font-weight: 300; color: #86efac; line-height: 1; margin-bottom: 0.5rem;">02</div>
<div style="font-size: 1.3rem; font-weight: 500; color: #047857; margin-bottom: 0.5rem;">创新点2 · Infra：AgentGenesis 框架</div>
<div style="font-size: 0.95rem; color: #6b7280;">并行探索 · 适配器模式 · 自然语言转环境</div>
</div>

<div style="text-align: left; max-width: 240px;">
<div style="font-size: 3.5rem; font-weight: 300; color: #fdba74; line-height: 1; margin-bottom: 0.5rem;">03</div>
<div style="font-size: 1.3rem; font-weight: 500; color: #9a3412; margin-bottom: 0.5rem;">创新点3 · 算法：History Graph</div>
<div style="font-size: 0.95rem; color: #6b7280;">跨任务经验复用 · 图结构共享记忆</div>
</div>

<div style="text-align: left; max-width: 240px;">
<div style="font-size: 3.5rem; font-weight: 300; color: #c4b5fd; line-height: 1; margin-bottom: 0.5rem;">04</div>
<div style="font-size: 1.3rem; font-weight: 500; color: #5b21b6; margin-bottom: 0.5rem;">实验与 Future Work</div>
<div style="font-size: 0.95rem; color: #6b7280;">72 数据集阈值 · 对比实验设计</div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
---

<div style="font-size: 2.5rem; font-weight: 400; color: #5a7a8a;">创新点1 · 洞察</div>

<div style="font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;">Auto Research 5层分层模型与当前 bottleneck</div>

---
transition: slide-up
title: "创新点1 · Auto Research 的两种理解"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">创新点1 · Auto Research 的两种理解</div>

<div class="grid grid-cols-2 gap-6 max-w-5xl mx-auto">

<div class="p-4 bg-blue-50 rounded-lg border border-blue-100">
<div class="text-blue-800 font-semibold text-lg mb-2">范式 A：执行层优化</div>
<div class="text-gray-700 text-sm space-y-1">
<div>用 Coding Agent 对<strong>可量化评分任务</strong>进行调优迭代</div>
<div>典型：ML 超参数搜索、RL 经典小游戏（Atari）</div>
<div style="color: #6b7280; font-size: 0.8rem;">代表：Karpathy /autoresearch，翁家翌 HL（启发式学习）</div>
<div style="color: #6b7280; font-size: 0.8rem;">核心：实验 → 反馈 → 迭代（第 4 层）</div>
</div>
<div style="text-align: center; margin-top: 0.5rem;">
<img src="/progress.png" style="max-height: 140px; width: auto; border-radius: 6px; border: 1px solid #e5e7eb;">
</div>
</div>

<div class="p-4 bg-green-50 rounded-lg border border-green-100">
<div class="text-green-800 font-semibold text-lg mb-2">范式 B：端到端论文生成</div>
<div class="text-gray-700 text-sm space-y-1">
<div>全自动或 Human-in-the-Loop 完成论文全流程</div>
<div>搜集文献 → 发现 Idea → 设计验证 → 执行实验 → 撰写论文</div>
<div style="color: #6b7280; font-size: 0.8rem;">代表：AI-Scientist-v2，Auto-claude-code-research-in-sleep</div>
<div style="color: #6b7280; font-size: 0.8rem;">核心：第 1-5 层端到端闭环</div>
</div>
<div style="text-align: center; margin-top: 0.5rem;">
<img src="/hl.png" style="max-height: 140px; width: auto; border-radius: 6px; border: 1px solid #e5e7eb;">
</div>
</div>

</div>

<div style="text-align: center; margin-top: 1.5rem; color: #5a7a8a; font-size: 1.05rem; font-weight: 500;">
<b>提升前者有助于后者整体提升</b>，后者是对前者的延伸，两者不矛盾
</div>

---
transition: slide-up
title: "创新点1 · 我们提出的 5 层分层模型"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">创新点1 · 我们提出的 Auto Research 5 层分层模型</div>

<div class="max-w-4xl mx-auto">

<div style="display: flex; flex-direction: column; gap: 0.6rem;">

<div class="p-3 bg-gray-50 rounded-lg border border-gray-200" style="opacity: 0.6;">
<div style="font-size: 1rem; font-weight: 600; color: #6b7280;">Layer 1 · 提出或定义问题</div>
<div style="font-size: 0.8rem; color: #9ca3af;">瓶颈：Context Learning（综述性论文）</div>
</div>

<div class="p-3 bg-gray-50 rounded-lg border border-gray-200" style="opacity: 0.6;">
<div style="font-size: 1rem; font-weight: 600; color: #6b7280;">Layer 2 · 搜集学习论文</div>
<div style="font-size: 0.8rem; color: #9ca3af;">瓶颈：Context Learning（综述性论文）</div>
</div>

<div class="p-3 bg-gray-50 rounded-lg border border-gray-200" style="opacity: 0.6;">
<div style="font-size: 1rem; font-weight: 600; color: #6b7280;">Layer 3 · 提出可能的 Idea 集合</div>
<div style="font-size: 0.8rem; color: #9ca3af;">瓶颈：Context Learning（综述性论文）</div>
</div>

<div class="p-3 bg-blue-50 rounded-lg border border-blue-100" style="border-left: 4px solid #3b82f6;">
<div style="font-size: 1rem; font-weight: 600; color: #1e40af;">Layer 4 · 设计验证实验并执行 ⭐ 我们的聚焦点</div>
<div style="font-size: 0.8rem; color: #3b82f6;">瓶颈：Code 出正确的实验得到正确结果。Idea 可批量生成，但快速有效验证并不容易</div>
</div>

<div class="p-3 bg-gray-50 rounded-lg border border-gray-200" style="opacity: 0.6;">
<div style="font-size: 1rem; font-weight: 600; color: #6b7280;">Layer 5 · 撰写论文与 Rebuttal</div>
<div style="font-size: 0.8rem; color: #9ca3af;">瓶颈：Context Learning（综述性论文）</div>
</div>

</div>

</div>

<div style="text-align: center; margin-top: 1.2rem; color: #6b7280; font-size: 0.85rem;">
对于 CS <b>非综述性论文</b>，bottleneck 在第 4 层；对于综述性论文，bottleneck 在 1,2,3,5 层的 Context Learning
</div>

---
transition: slide-up
title: "创新点1 · 第 4 层的核心挑战"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">创新点1 · 第 4 层的核心挑战</div>

<div class="grid grid-cols-2 gap-6 max-w-5xl mx-auto">

<div class="p-4 bg-red-50 rounded-lg border border-red-100">
<div class="text-red-800 font-semibold text-lg mb-2">别人的痛点</div>
<div class="text-gray-700 text-sm space-y-2">
<div><span style="color: #991b1b; font-weight: 600;">×</span> 单次任务内搜索，不跨任务复用（AIDE / OpenEvolve）</div>
<div><span style="color: #991b1b; font-weight: 600;">×</span> 多 Agent 共享文件系统，无结构化知识抽象（CORAL）</div>
<div><span style="color: #991b1b; font-weight: 600;">×</span> 需要 10⁴ 次迭代才能收敛（RL）</div>
<div><span style="color: #991b1b; font-weight: 600;">×</span> 反馈信号简单，无长期知识库</div>
</div>
</div>

<div class="p-4 bg-green-50 rounded-lg border border-green-100">
<div class="text-green-800 font-semibold text-lg mb-2">我们的方案</div>
<div class="text-gray-700 text-sm space-y-2">
<div><span style="color: #166534; font-weight: 600;">✓</span> <b>跨任务</b>历史经验图复用（History Graph）</div>
<div><span style="color: #166534; font-weight: 600;">✓</span> 多 Agent 之间<b>图结构</b>作为共享记忆</div>
<div><span style="color: #166534; font-weight: 600;">✓</span> 只需 <b>10 次</b>迭代（LLM + 经验复用）</div>
<div><span style="color: #166534; font-weight: 600;">✓</span> 结构化五元组反馈（feedback / strategy / concept / result / pass_criteria）</div>
</div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
---

<div style="font-size: 2.5rem; font-weight: 400; color: #5a7a8a;">创新点2 · Infra</div>

<div style="font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;">AgentGenesis 框架：把任何 Idea 转化为可迭代可评测的执行层环境</div>

---
transition: slide-up
title: "创新点2 · AgentGenesis 框架架构"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">创新点2 · AgentGenesis 框架架构</div>

<div style="text-align: center;">
<img src="/infra_framework.png" style="max-height: 480px; width: auto; margin: 0 auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
</div>

---
transition: slide-up
title: "创新点2 · Skill 包目录结构"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">创新点2 · Skill 包目录结构</div>

<div class="max-w-4xl mx-auto p-4 bg-slate-50 rounded-lg border border-slate-200 font-mono text-sm">
<div class="text-slate-800">
AgentGenesis/skills/author-agentgenesis-problem/<br>
├── <span class="text-blue-700 font-semibold">SKILL.md</span>                  # Skill manifest & entry point<br>
├── <span class="text-blue-700 font-semibold">examples/</span>                 # 11 official problem patterns<br>
├── <span class="text-green-700 font-semibold">manuals/出题指南.md</span>        # Complete problem-authoring guide<br>
├── <span class="text-green-700 font-semibold">references/</span>               # Format, patterns, quality rubric<br>
│   ├── problem-format.md<br>
│   ├── runtime-patterns.md<br>
│   ├── progressive-disclosure.md<br>
│   └── quality-rubric.md<br>
├── <span class="text-purple-700 font-semibold">templates/</span>                # Design brief & skeletons<br>
│   ├── problem-design-brief.md<br>
│   ├── single-agent-problem.md<br>
│   └── isolated-multi-agent-problem.md<br>
└── <span class="text-orange-700 font-semibold">problems/</span>                 # 11 official examples<br>
    ├── maze/、structured_output/、parallel_weather/<br>
    ├── sports_shopping/、resilient_scraper/<br>
    ├── tool_creator_challenge/、log_hunter/<br>
    ├── werewolf/、microservice_avalanche/ ...
</div>
</div>

<div style="text-align: center; margin-top: 1rem; color: #6b7280; font-size: 0.85rem;">
文档驱动 + 模板驱动：从自然语言 Idea → 完整可评测 Problem Package
</div>

---
transition: slide-up
title: "创新点2 · 迭代案例：从 Fail 到 Pass"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1rem;">创新点2 · 迭代案例：从 Fail 到 Pass</div>

<div class="grid grid-cols-3 gap-4 max-w-6xl mx-auto">

<div>
<img src="/iteration_example.png" style="max-height: 240px; width: auto; margin: 0 auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
<div style="text-align: center; margin-top: 0.5rem; color: #6b7280; font-size: 0.8rem;">Commit 结构化记录</div>
</div>

<div>
<img src="/iteration_example1.png" style="max-height: 240px; width: auto; margin: 0 auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
<div style="text-align: center; margin-top: 0.5rem; color: #6b7280; font-size: 0.8rem;">代码 Diff：模型替换</div>
</div>

<div>
<img src="/iteration_example2.png" style="max-height: 240px; width: auto; margin: 0 auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
<div style="text-align: center; margin-top: 0.5rem; color: #6b7280; font-size: 0.8rem;">反馈驱动优化</div>
</div>

</div>

<div style="text-align: center; margin-top: 1rem; color: #6b7280; font-size: 0.85rem;">
以 052_bank_marketing 为例：Iteration 0 失败 → 诊断 StringType 问题 → 换 OrdinalEncoder → Iteration 4 通过（91.24%）
</div>

---
layout: center
class: text-center
transition: slide-left
---

<div style="font-size: 2.5rem; font-weight: 400; color: #5a7a8a;">创新点3 · 算法</div>

<div style="font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;">History Graph：跨任务经验复用的图结构记忆</div>

---
transition: slide-up
title: "创新点3 · Add Logic"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1rem;">创新点3 · Add Logic</div>

<div style="text-align: center;">
<img src="/add_logic.png" style="max-height: 480px; width: auto; margin: 0 auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
</div>

<div style="text-align: center; margin-top: 1rem; color: #6b7280; font-size: 0.9rem;">
Commit → Embedding → 相似度聚类 → LLM Merge → History Graph
</div>

---
transition: slide-up
title: "创新点3 · Use Logic"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1rem;">创新点3 · Use Logic</div>

<div style="text-align: center;">
<img src="/use_logic.png" style="max-height: 480px; width: auto; margin: 0 auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
</div>

<div style="text-align: center; margin-top: 1rem; color: #6b7280; font-size: 0.9rem;">
当前路径 → Retrieval(similarity+LLM) → 指导/避坑经验注入 Prompt
</div>

---
transition: slide-up
title: "创新点3 · 为什么 History Graph 是独特的？"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">创新点3 · 为什么 History Graph 是独特的？</div>

<div class="grid grid-cols-3 gap-4 max-w-5xl mx-auto text-sm">

<div class="p-4 bg-blue-50 rounded-lg border border-blue-100">
<div class="text-blue-800 font-semibold mb-2">记忆机制</div>
<div class="text-gray-700">唯一<b>跨不同任务</b>+<b>跨同一任务不同分支</b>的共享记忆</div>
<div class="text-gray-600 mt-1">唯一多 Agent 之间<b>图结构</b>作为共享记忆</div>
</div>

<div class="p-4 bg-green-50 rounded-lg border border-green-100">
<div class="text-green-800 font-semibold mb-2">协作模式</div>
<div class="text-gray-700">其他系统只支持<b>空间</b>上的协作</div>
<div class="text-gray-600 mt-1">我们支持<b>时空</b>上都进行协作（历史经验 + 当前并行）</div>
</div>

<div class="p-4 bg-purple-50 rounded-lg border border-purple-100">
<div class="text-purple-800 font-semibold mb-2">反馈信号</div>
<div class="text-gray-700">五元组：feedback + strategy + concepts + result + pass_criteria</div>
<div class="text-gray-600 mt-1">恰好是"检索相似问题"所需的全部信息</div>
</div>

</div>

<div style="text-align: center; margin-top: 1.5rem; color: #6b7280; font-size: 0.9rem;">
CORAL 共享文件系统 · AIDE/OpenEvolve 同任务内搜索 · 我们：<b>跨任务历史经验图复用</b>
</div>

---
transition: slide-up
title: "创新点3 · 与现有系统的对比"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1rem;">创新点3 · 与现有系统的对比</div>

<div style="font-size: 0.65rem; overflow-x: auto;">
<table class="max-w-6xl mx-auto" style="border-collapse: collapse; width: 100%;"><tbody>
<tr style="background:#f3f4f6;">
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left;">维度</th>
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left; color:#1e40af;">Ours</th>
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left;">CORAL</th>
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left;">AI-Scientist-v2</th>
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left;">AIDE</th>
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left;">OpenEvolve</th>
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left;">GEPA</th>
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left;">FunSearch</th>
<th style="padding: 4px 6px; border: 1px solid #d1d5db; text-align: left;">autoresearch</th>
</tr>
<tr>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">系统定位</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">多Agent自主编码 + 跨任务经验复用</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">多Agent自主编码+共享文件系统</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">端到端科研自动化</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">ML工程树搜索</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">进化式代码优化</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">反射式Prompt/代码进化</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">数学发现程序搜索</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">单GPU训练自治</td>
</tr>
<tr style="background:#f9fafb;">
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">架构家族</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">Search-Based + Multi-Agent 融合</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Multi-Agent + Skill-Based</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Search-Based (Tree)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Search-Based (Tree)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Search-Based (Evolutionary)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Search-Based (Evolutionary)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Search-Based (Evolutionary)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Sequential Pipeline</td>
</tr>
<tr>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">搜索拓扑</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">异步并行 + 历史图检索 + Agentic Search</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">异步并行 + 心跳调度</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Agentic Tree Search (BFTS)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">解树 (UCB/贪婪)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">MAP-Elites + 岛屿迁移</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Pareto前沿 + 反射突变</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">遗传程序搜索 (islands)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">线性迭代</td>
</tr>
<tr style="background:#f9fafb;">
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">记忆架构</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">共享图结构记忆 (History Graph)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">共享文件系统池 .coral/public/</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">树节点状态</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">解树节点</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">MAP-Elites网格 + 档案库</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Pareto候选池</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">程序档案库</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Git历史</td>
</tr>
<tr>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">知识复用范围</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">跨任务、跨会话、可累积</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">同任务内共享</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">同任务内树回溯</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">同任务内树回溯</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">同任务内岛屿迁移</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">同任务内 Pareto 合并</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">同任务内遗传继承</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">无</td>
</tr>
<tr style="background:#f9fafb;">
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">进化机制</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">Embedding 相似度 + LLM 合并决策</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">心跳触发 (reflect/consolidate/pivot)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">树扩展 + Debug 回溯</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Patch 生成 + 剪枝</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">LLM 变异 + 双选择</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">LLM 反射诊断 + 定向突变</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">随机组合 + 评估筛选</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">Agent 自主编辑</td>
</tr>
<tr>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">Agent 协作模式</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">并行竞争 + 图记忆共享</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">并行自主 + 元控制器心跳</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">单Agent树探索</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">单Agent树探索</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">岛屿隔离 + 定期迁移</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">单线程候选进化</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">分布式岛屿</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">单Agent</td>
</tr>
<tr style="background:#f9fafb;">
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">反馈深度</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">结构化五元组 (feedback/strategy/result/concept/criteria)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">复合评分 + Agent notes</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">实验指标 + 报错信息</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">评估指标</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">标量评分 + Artifacts</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">执行轨迹全量反射</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">标量评分</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">标量 (val_bpb)</td>
</tr>
<tr>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">生命周期覆盖</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">s4 (Coding)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">s4 (Coding)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">s1-s5 (Creation→Writing)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">s4 (Coding)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">s4 (Coding)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">s4 (Coding/Config)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">s4 (Coding)</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">s4 (Coding)</td>
</tr>
<tr style="background:#f9fafb;">
<td style="padding: 4px 6px; border: 1px solid #d1d5db; font-weight:600;">核心差异点</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db; color:#1e40af;">唯一实现跨任务历史经验图复用</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">最强多Agent基础设施</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">唯一产出审稿级论文</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">4x 奖牌率的树搜索</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">质量多样性进化</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">反射优于RL的优化</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">数学定理发现</td>
<td style="padding: 4px 6px; border: 1px solid #d1d5db;">极简自治</td>
</tr>
</tbody></table>
</div>

---
layout: center
class: text-center
transition: slide-left
---

<div style="font-size: 2.5rem; font-weight: 400; color: #5a7a8a;">Part 4 · 实验与 Future Work</div>

<div style="font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;">72 数据集 Benchmark · 阈值测定 · 对比实验设计</div>

---
transition: slide-up
title: "实验 1：阈值测定"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1rem;">实验 1：pass_criteria 阈值测定（Pilot）</div>

<div class="grid grid-cols-3 gap-4 max-w-5xl mx-auto mb-4">

<div class="p-3 bg-blue-50 rounded-lg border border-blue-100 text-center">
<div class="text-blue-800 font-semibold text-sm mb-1">Benchmark</div>
<div class="text-gray-700 text-xs">OpenML-CC18-AG<br>72 个 tabular 分类任务</div>
</div>

<div class="p-3 bg-orange-50 rounded-lg border border-orange-100 text-center">
<div class="text-orange-800 font-semibold text-sm mb-1">初始标准</div>
<div class="text-gray-700 text-xs">Accuracy ≥ 0.8<br>stop_on_pass = False</div>
</div>

<div class="p-3 bg-green-50 rounded-lg border border-green-100 text-center">
<div class="text-green-800 font-semibold text-sm mb-1">Threshold 计算</div>
<div class="text-gray-700 text-xs">跑 3 次<br>mean(best_score)</div>
</div>

</div>

<div style="font-size: 0.75rem; overflow-x: auto;">
<table class="max-w-4xl mx-auto" style="border-collapse: collapse; width: 100%;"><tbody>
<tr style="background:#f3f4f6;">
<th style="padding: 4px 10px; border: 1px solid #d1d5db;">Dataset</th>
<th style="padding: 4px 10px; border: 1px solid #d1d5db;">Threshold</th>
<th style="padding: 4px 10px; border: 1px solid #d1d5db;">Min</th>
<th style="padding: 4px 10px; border: 1px solid #d1d5db;">Max</th>
</tr>
<tr><td style="padding: 3px 10px; border: 1px solid #d1d5db;">001_kr_vs_kp</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;"><b>99.79</b></td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">99.69</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">100.0</td></tr>
<tr style="background:#f9fafb;"><td style="padding: 3px 10px; border: 1px solid #d1d5db;">003_balance_scale</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;"><b>100.0</b></td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">100.0</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">100.0</td></tr>
<tr><td style="padding: 3px 10px; border: 1px solid #d1d5db;">014_pendigits</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;"><b>99.55</b></td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">99.45</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">99.64</td></tr>
<tr style="background:#f9fafb;"><td style="padding: 3px 10px; border: 1px solid #d1d5db;">022_eucalyptus</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;"><b>68.92</b></td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">63.51</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">77.03</td></tr>
<tr><td style="padding: 3px 10px; border: 1px solid #d1d5db;">069_CIFAR_10</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;"><b>45.41</b></td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">41.5</td><td style="padding: 3px 10px; border: 1px solid #d1d5db;">53.23</td></tr>
</tbody></table>
</div>

<div style="text-align: center; margin-top: 0.8rem; color: #6b7280; font-size: 0.8rem;">
72 个数据集全部完成 · 完整数据见 <code>exp/pilot/THRESHOLDS.md</code>
</div>

---
transition: slide-up
title: "Future Work"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">Future Work · 待验证实验</div>

<div class="grid grid-cols-3 gap-4 max-w-5xl mx-auto text-sm">

<div class="p-4 bg-blue-50 rounded-lg border border-blue-100">
<div class="text-blue-800 font-semibold mb-2">实验 2</div>
<div class="text-gray-700">跨任务经验复用 vs 无复用 Baseline</div>
<div class="text-gray-600 mt-1">验证 History Graph 跨任务复用是否加速收敛</div>
</div>

<div class="p-4 bg-green-50 rounded-lg border border-green-100">
<div class="text-green-800 font-semibold mb-2">实验 3</div>
<div class="text-gray-700">多 Branch 共享 Graph vs CORAL vs Independent</div>
<div class="text-gray-600 mt-1">验证结构化图记忆是否优于文件系统共享</div>
</div>

<div class="p-4 bg-purple-50 rounded-lg border border-purple-100">
<div class="text-purple-800 font-semibold mb-2">实验 4</div>
<div class="text-gray-700">LLM Merge 质量验证</div>
<div class="text-gray-600 mt-1">定量 + 定性评估 Merge 是否提炼出通用经验</div>
</div>

</div>

---
transition: slide-up
title: "实验 2：跨任务复用"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">实验 2：跨任务经验复用有效吗？</div>

<div class="grid grid-cols-2 gap-6 max-w-5xl mx-auto">

<div class="p-4 bg-red-50 rounded-lg border border-red-100">
<div class="text-red-800 font-semibold text-lg mb-2">Baseline（无历史复用）</div>
<div class="text-gray-700 text-sm space-y-1">
<div>history_mode = disabled</div>
<div>每个任务从零开始探索</div>
<div>打乱数据集顺序，消除顺序偏差</div>
<div>记录：iterations_to_first_pass, success_rate@K, best_score</div>
</div>
</div>

<div class="p-4 bg-green-50 rounded-lg border border-green-100">
<div class="text-green-800 font-semibold text-lg mb-2">Ours_Full（跨任务历史复用）</div>
<div class="text-gray-700 text-sm space-y-1">
<div>history_mode = full, history_scope = cross-task</div>
<div>打乱数据集顺序，历史经验累积复用</div>
<div>记录：同上 + graph_compression（raw_commits / merged_nodes）</div>
<div>max_iteration = 10</div>
</div>
</div>

</div>

<div style="text-align: center; margin-top: 1.5rem; color: #5a7a8a; font-size: 1rem; font-weight: 500;">
验证指标：若 ours_full 的 iterations_to_first_pass ↓ 且 success_rate@K ↑ 且 best_score ↑<br>
→ 证明 <b>跨任务经验复用有效</b>（72 数据集 × 3 rounds）
</div>

---
transition: slide-up
title: "实验 3：多 Branch 共享 Graph"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1rem;">实验 3：多 Branch 共享 Graph vs CORAL vs Independent</div>

<div style="font-size: 0.75rem; margin-bottom: 1rem;">
<table class="max-w-5xl mx-auto" style="border-collapse: collapse; width: 100%;"><tbody>
<tr style="background:#f3f4f6;">
<th style="padding: 5px 8px; border: 1px solid #d1d5db;">条件</th>
<th style="padding: 5px 8px; border: 1px solid #d1d5db;">描述</th>
<th style="padding: 5px 8px; border: 1px solid #d1d5db;">共享机制</th>
</tr>
<tr>
<td style="padding: 5px 8px; border: 1px solid #d1d5db; font-weight:600;">Independent</td>
<td style="padding: 5px 8px; border: 1px solid #d1d5db;">3 个 branch 完全独立，各自维护空历史图，互不共享</td>
<td style="padding: 5px 8px; border: 1px solid #d1d5db;">无</td>
</tr>
<tr style="background:#f9fafb;">
<td style="padding: 5px 8px; border: 1px solid #d1d5db; font-weight:600;">CORAL</td>
<td style="padding: 5px 8px; border: 1px solid #d1d5db;">3 个 agent 同任务共享 .coral/public/ 文件系统</td>
<td style="padding: 5px 8px; border: 1px solid #d1d5db;">文件系统共享</td>
</tr>
<tr>
<td style="padding: 5px 8px; border: 1px solid #d1d5db; font-weight:600; color:#1e40af;">Shared Graph (Ours)</td>
<td style="padding: 5px 8px; border: 1px solid #d1d5db;">3 个 branch 同时启动，共享同一个 merged history graph</td>
<td style="padding: 5px 8px; border: 1px solid #d1d5db; color:#1e40af;">结构化图记忆（embedding + merge + relevance）</td>
</tr>
</tbody></table>
</div>

<div class="grid grid-cols-2 gap-4 max-w-5xl mx-auto text-sm">
<div class="p-3 bg-blue-50 rounded-lg border border-blue-100">
<div class="text-blue-800 font-semibold mb-1">方法</div>
<div class="text-gray-700 text-xs space-y-1">
<div>每组 3 个并行 branch（num_nodes=3），同时启动</div>
<div>每个 branch：max_iter=10，3 组重复</div>
</div>
</div>
<div class="p-3 bg-green-50 rounded-lg border border-green-100">
<div class="text-green-800 font-semibold mb-1">指标</div>
<div class="text-gray-700 text-xs space-y-1">
<div>avg_iterations_to_pass / min_iterations_to_pass</div>
<div>best_score_of_all_branch</div>
</div>
</div>
</div>

<div style="text-align: center; margin-top: 1rem; color: #5a7a8a; font-size: 0.9rem; font-weight: 500;">
预期：Shared Graph (Ours) > CORAL 文件系统共享 > Independent
</div>

---
transition: slide-up
title: "实验 4：LLM Merge 质量"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1rem;">实验 4：LLM Merge 质量验证</div>

<div class="grid grid-cols-2 gap-6 max-w-5xl mx-auto">

<div class="p-4 bg-blue-50 rounded-lg border border-blue-100">
<div class="text-blue-800 font-semibold text-lg mb-2">定量分析</div>
<div class="text-gray-700 text-sm space-y-1">
<div><b>retrieval_hit_rate_merged</b>：merged 节点后续被检索命中次数 / 总检索次数</div>
<div><b>retrieval_hit_rate_raw</b>：假设原始节点未被 merge，模拟被检索命中次数</div>
<div><b>compression_ratio</b>：len(raw_nodes) / len(merged_nodes)</div>
<div><b>concept_coverage</b>：merged 节点 concepts 并集 / 原始节点 concepts 并集</div>
</div>
</div>

<div class="p-4 bg-green-50 rounded-lg border border-green-100">
<div class="text-green-800 font-semibold text-lg mb-2">定性分析（抽样 20 组）</div>
<div class="text-gray-700 text-sm space-y-1">
<div><b>Feedback 质量</b>：merged 是否更 general、更准确描述问题模式？</div>
<div><b>Strategy 质量</b>：merged 是否融合两种做法，形成更通用修复思路？</div>
<div><b>信息损失</b>：是否有独特且重要的信息在 merge 后丢失？</div>
<div>评估：3 分制（1=差，2=一般，3=好），预期均值 > 2.0</div>
</div>
</div>

</div>

<div style="text-align: center; margin-top: 1.5rem; color: #5a7a8a; font-size: 0.95rem; font-weight: 500;">
预期结论
</div>

<div class="max-w-5xl mx-auto mt-3 text-sm text-gray-700 space-y-1">
<div>• <b>retrieval_hit_rate_merged > retrieval_hit_rate_raw</b>：merge 后的节点更 general，更容易被后续检索命中 → 证明 merge 提炼出了通用模式</div>
<div>• <b>compression_ratio > 2</b>：merge 显著压缩了历史空间 → 证明不是简单拼接，而是真正聚合了经验</div>
<div>• <b>定性评分均值 > 2.0</b>：LLM merge 确实提炼出有意义的通用经验，而非生硬拼接或信息丢失</div>
</div>

---
transition: slide-up
title: "近期与远期计划"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; text-align: center; margin-bottom: 1.5rem;">近期与远期计划</div>

<div class="grid grid-cols-2 gap-6 max-w-5xl mx-auto">

<div class="p-4 bg-blue-50 rounded-lg border border-blue-100">
<div class="text-blue-800 font-semibold text-lg mb-2">近期</div>
<div class="text-gray-700 text-sm space-y-2">
<div>🧪 跑通实验 2/3/4 全量对比（72 数据集 × 3 rounds）</div>
<div>📝 论文撰写（执行层优化 + History Graph）</div>
</div>
</div>

<div class="p-4 bg-purple-50 rounded-lg border border-purple-100">
<div class="text-purple-800 font-semibold text-lg mb-2">远期</div>
<div class="text-gray-700 text-sm space-y-2">
<div>🔮 向 Layer 1-3, 5 延伸（端到端论文生成）</div>
<div>📦 开源 AgentGenesis 框架与 Benchmark</div>
</div>
</div>

</div>

---
background: ./cover.jpg
class: text-center
transition: slide-left
---

<style scoped>
.slidev-layout {
  background-image: url('/cover.jpg') !important;
  background-size: cover !important;
  background-position: center !important;
  background-repeat: no-repeat !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
  color: white !important;
  text-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
}
</style>

<div style="font-size: 4rem; font-weight: 500; margin-bottom: 1rem;">谢谢聆听</div>

<div style="font-size: 1.8rem; font-weight: 400; color: rgba(255,255,255,0.9);">Q & A</div>
