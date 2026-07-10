---
theme: seriph
background: ./cover.jpg
title: "SCRIBE"
class: text-center
transition: slide-left
mdc: true
download: true
---

<style>
.page-title{font-size:2.05rem;font-weight:400;color:#5a7a8a;text-align:center;margin-bottom:1rem;line-height:1.3}
.sec-big{font-size:2.6rem;font-weight:400;color:#5a7a8a}
.sec-sub{font-size:1.15rem;color:#6b7280;margin-top:0.5rem}
.chip{display:block;padding:3px 8px;border-radius:5px;font-size:0.7rem;font-family:ui-monospace,SFMono-Regular,monospace;margin:2px 0;line-height:1.3}
.reuse{background:#dcfce7;color:#166534;border:1px solid #86efac}
.new{background:#dbeafe;color:#1e3a8a;border:1px solid #93c5fd}
.comp{background:#fef3c7;color:#92400e;border:1px solid #fcd34d}
.strike{text-decoration:line-through;opacity:0.65}
.neu{background:#f3f4f6;color:#4b5563;border:1px solid #d1d5db}
.col-h{font-size:0.85rem;font-weight:600;color:#374151;text-align:center;margin-bottom:4px}
.cap{font-size:0.76rem;color:#4b5563;text-align:center;line-height:1.4}
.leg{font-size:0.68rem}
.inl{display:inline-block;width:auto;padding:2px 7px}
</style>

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
  font-size: 4.2rem !important;
  font-weight: 600 !important;
  letter-spacing: 2px !important;
  margin-bottom: 0.5rem !important;
  color: white !important;
}
.slidev-layout h2 {
  font-size: 1.35rem !important;
  font-weight: 400 !important;
  line-height: 1.5 !important;
  max-width: 900px !important;
  margin-bottom: 2rem !important;
  color: rgba(255,255,255,0.92) !important;
}
.slidev-layout .info {
  font-size: 1rem !important;
  line-height: 1.8 !important;
  color: rgba(255,255,255,0.85) !important;
  margin-top: 1.5rem !important;
}
</style>

# SCRIBE

## Loop Engineering in Long Horizon Tasks Needs Agent to Write What They Did Each Turn

<div class="info">
周浩洋 · Jilin University (JLU), School of Software (SE)<br>
Research Assistant · 2025.04.17 – Present
</div>

---
layout: center
class: text-center
transition: slide-left
title: "目录"
---

<h1 style="color: #5a7a8a; font-weight: 400; font-size: 2.5rem; margin-bottom: 3rem;">目录</h1>

<div style="display:flex;justify-content:center;gap:1.6rem;flex-wrap:wrap;max-width:1000px;margin:0 auto;">

<div style="text-align:left;max-width:215px;">
<div style="font-size:3.2rem;font-weight:300;color:#93c5fd;line-height:1;margin-bottom:0.4rem;">01</div>
<div style="font-size:1.15rem;font-weight:500;color:#1e40af;margin-bottom:0.3rem;">training-free 方法</div>
<div style="font-size:0.85rem;color:#6b7280;">不可能实现 loop engineering 的 token 效率提升</div>
</div>

<div style="text-align:left;max-width:215px;">
<div style="font-size:3.2rem;font-weight:300;color:#86efac;line-height:1;margin-bottom:0.4rem;">02</div>
<div style="font-size:1.15rem;font-weight:500;color:#047857;margin-bottom:0.3rem;">training 方法 · ReAct</div>
<div style="font-size:0.85rem;color:#6b7280;">使用 ReAct 进行 loop engineering</div>
</div>

<div style="text-align:left;max-width:215px;">
<div style="font-size:3.2rem;font-weight:300;color:#fdba74;line-height:1;margin-bottom:0.4rem;">03</div>
<div style="font-size:1.15rem;font-weight:500;color:#9a3412;margin-bottom:0.3rem;">ReAct 的问题</div>
<div style="font-size:0.85rem;color:#6b7280;">使用 ReAct 进行 loop engineering 的问题</div>
</div>

<div style="text-align:left;max-width:215px;">
<div style="font-size:3.2rem;font-weight:300;color:#c4b5fd;line-height:1;margin-bottom:0.4rem;">04</div>
<div style="font-size:1.15rem;font-weight:500;color:#5b21b6;margin-bottom:0.3rem;">training 方法 · SCRIBE</div>
<div style="font-size:0.85rem;color:#6b7280;">使用 SCRIBE 进行 loop engineering</div>
</div>

<div style="text-align:left;max-width:215px;">
<div style="font-size:3.2rem;font-weight:300;color:#5eead4;line-height:1;margin-bottom:0.4rem;">05</div>
<div style="font-size:1.15rem;font-weight:500;color:#0f766e;margin-bottom:0.3rem;">SCRIBE 上下文管理机制</div>
<div style="font-size:0.85rem;color:#6b7280;">细述 turn 级上下文与 KV-cache 复用</div>
</div>

<div style="text-align:left;max-width:215px;">
<div style="font-size:3.2rem;font-weight:300;color:#fca5a5;line-height:1;margin-bottom:0.4rem;">06</div>
<div style="font-size:1.15rem;font-weight:500;color:#991b1b;margin-bottom:0.3rem;">reward mechanism of a turn</div>
<div style="font-size:0.85rem;color:#6b7280;">16 个 metric 与权重配比</div>
</div>

<div style="text-align:left;max-width:215px;">
<div style="font-size:3.2rem;font-weight:300;color:#fde68a;line-height:1;margin-bottom:0.4rem;">07</div>
<div style="font-size:1.15rem;font-weight:500;color:#92400e;margin-bottom:0.3rem;">abstract of our contribution</div>
<div style="font-size:0.85rem;color:#6b7280;">小改动 → 上下文机制变化 → 衍生收益</div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
---

<div class="sec-big">01 · training-free 方法</div>

<div class="sec-sub">不可能实现 loop engineering 的 token 效率提升</div>

---
transition: slide-up
title: "01 · training-free 方法 (1/2)"
---

<div class="page-title">01 · training-free 方法 — 不可能实现 loop engineering 的 token 效率提升</div>

<div style="text-align:center;">
<img src="/tf-1.png" style="max-height:50vh;max-width:86%;width:auto;height:auto;object-fit:contain;margin:0 auto;display:block;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.12);">
</div>

<div class="cap" style="margin-top:0.6rem;">training-free 路径下，loop engineering 的 token 效率提升瓶颈（一）</div>

---
transition: slide-up
title: "01 · training-free 方法 (2/2)"
---

<div class="page-title">01 · training-free 方法 — 不可能实现 loop engineering 的 token 效率提升</div>

<div style="text-align:center;">
<img src="/tf-2.png" style="max-height:50vh;max-width:86%;width:auto;height:auto;object-fit:contain;margin:0 auto;display:block;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.12);">
</div>

<div class="cap" style="margin-top:0.6rem;">training-free 路径下，loop engineering 的 token 效率提升瓶颈（二）</div>

---
layout: center
class: text-center
transition: slide-left
---

<div class="sec-big">02 · training 方法</div>

<div class="sec-sub">使用 ReAct 进行 loop engineering</div>

---
transition: slide-up
title: "02 · ReAct 训练方法"
---

<div class="page-title">02 · training 方法 — 使用 ReAct 进行 loop engineering</div>

<div style="text-align:center;">
<img src="/react-method.png" style="max-height:50vh;max-width:86%;width:auto;height:auto;object-fit:contain;margin:0 auto;display:block;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.12);">
</div>

<div class="cap" style="margin-top:0.6rem;">broad ReAct：经典 Think→Act→Observe 循环作为 loop engineering 的训练范式</div>

---
layout: center
class: text-center
transition: slide-left
---

<div class="sec-big">03 · ReAct 的问题</div>

<div class="sec-sub">使用 ReAct 进行 loop engineering 的问题</div>

---
transition: slide-up
title: "03 · ReAct 的问题"
---

<div class="page-title">03 · training 方法 — 使用 ReAct 进行 loop engineering 的问题</div>

<div style="text-align:center;">
<img src="/react-problems.png" style="max-height:50vh;max-width:86%;width:auto;height:auto;object-fit:contain;margin:0 auto;display:block;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.12);">
</div>

<div class="cap" style="margin-top:0.6rem;">broad ReAct 在长程 loop engineering 中暴露的问题</div>

---
layout: center
class: text-center
transition: slide-left
---

<div class="sec-big">04 · training 方法 · SCRIBE</div>

<div class="sec-sub">ReAct 问题的解法：使用 SCRIBE 进行 loop engineering</div>

---
transition: slide-up
title: "04 · SCRIBE 训练方法"
---

<div class="page-title">04 · training 方法 — 使用 SCRIBE 进行 loop engineering</div>

<div style="text-align:center;">
<img src="/scribe-method.png" style="max-height:50vh;max-width:86%;width:auto;height:auto;object-fit:contain;margin:0 auto;display:block;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.12);">
</div>

<div class="cap" style="margin-top:0.6rem;">SCRIBE：在 ReAct 基础上为每个 turn 显式写入 Reflect (R) 与 Turn-Summary (S)</div>

---
layout: center
class: text-center
transition: slide-left
---

<div class="sec-big">05 · SCRIBE 上下文管理机制</div>

<div class="sec-sub">细述 turn 级上下文管理：KV-cache 复用、recent-k 保留、自动压缩</div>

---
transition: slide-up
title: "05 · turn0 -> turn1"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn0 ➜ turn1</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:60rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn0 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD0 ✗ 未达阈值</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:1.8rem;">
<div style="font-size:2rem;color:#5a7a8a;line-height:1;">➜</div>
<div class="leg" style="color:#6b7280;margin-top:4px;text-align:center;">append</div>
</div>
<div>
<div class="col-h">turn1 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip new">T0 ▸ O·A·AR·S · fb0</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD1 ✗ 未达阈值</div>
</div>
</div>

<div class="cap" style="max-width:58rem;margin:0.5rem auto 0;">公共前缀 <b>SP + FUP</b> 完全复用（KV-cache 命中）；turn1 仅追加 turn0 的留存块 + fb0 + 本轮 rollout。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算</span>
</div>

---
transition: slide-up
title: "05 · turn1 -> turn2"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn1 ➜ turn2</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:60rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn1 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0 ▸ O·A·AR·S · fb0</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD1 ✗ 未达阈值</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:1.8rem;">
<div style="font-size:2rem;color:#5a7a8a;line-height:1;">➜</div>
<div class="leg" style="color:#6b7280;margin-top:4px;text-align:center;">append</div>
</div>
<div>
<div class="col-h">turn2 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0 ▸ O·A·AR·S · fb0</div>
<div class="chip new">T1 ▸ O·A·AR·S · fb1</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD2 ✗ 未达阈值</div>
</div>
</div>

<div class="cap" style="max-width:58rem;margin:0.5rem auto 0;">前缀 <b>SP + FUP + T0</b> 全部复用，仅追加 turn1 留存块 + fb1。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算</span>
</div>

---
transition: slide-up
title: "05 · turn2 -> turn3"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn2 ➜ turn3</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:60rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn2 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0 ▸ O·A·AR·S · fb0</div>
<div class="chip reuse">T1 ▸ O·A·AR·S · fb1</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD2 ✗ 未达阈值</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:1.8rem;">
<div style="font-size:2rem;color:#5a7a8a;line-height:1;">➜</div>
<div class="leg" style="color:#6b7280;margin-top:4px;text-align:center;">append</div>
</div>
<div>
<div class="col-h">turn3 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0 ▸ O·A·AR·S · fb0</div>
<div class="chip reuse">T1 ▸ O·A·AR·S · fb1</div>
<div class="chip new">T2 ▸ O·A·AR·S · fb2</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD3 ✗ 未达阈值</div>
</div>
</div>

<div class="cap" style="max-width:58rem;margin:0.5rem auto 0;">recent-k=3 窗口内逐 turn 追加，整段前缀 <b>SP+FUP+T0+T1</b> 复用，仅新增 T2。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算</span>
</div>

---
transition: slide-up
title: "05 · turn3 -> turn4"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn3 ➜ turn4（首次压缩）</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:62rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn3 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip comp strike">T0 ▸ O·A·AR·S · fb0</div>
<div class="chip comp">T1 ▸ O·A·AR·S · fb1</div>
<div class="chip comp">T2 ▸ O·A·AR·S · fb2</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD3 ✗</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:2rem;">
<div style="font-size:2rem;color:#9a3412;line-height:1;">➜</div>
<div class="leg" style="color:#9a3412;margin-top:4px;text-align:center;font-weight:600;">Stage1<br>压缩</div>
</div>
<div>
<div class="col-h">turn4 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip comp">T0 ▸ S · fb0</div>
<div class="chip comp">T1 ▸ O·A·AR·S · fb1</div>
<div class="chip comp">T2 ▸ O·A·AR·S · fb2</div>
<div class="chip new">T3 ▸ O·A·AR·S · fb3</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD4 ✗</div>
</div>
</div>

<div class="cap" style="max-width:60rem;margin:0.5rem auto 0;">turn0 超出 recent-k=3 被降级为 S（O/A/AR 丢弃，只留 turn_summary）；公共前缀在 <b>T0 处断裂</b> → T1/T2 虽内容不变也须重算。代价是换得后续 append 不再无限增长。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算（删除线=被丢弃）</span>
</div>

---
transition: slide-up
title: "05 · turn4 -> turn5"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn4 ➜ turn5（压缩推进）</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:62rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn4 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0 ▸ S · fb0</div>
<div class="chip comp strike">T1 ▸ O·A·AR·S · fb1</div>
<div class="chip comp">T2 ▸ O·A·AR·S · fb2</div>
<div class="chip comp">T3 ▸ O·A·AR·S · fb3</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD4 ✗</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:2rem;">
<div style="font-size:2rem;color:#9a3412;line-height:1;">➜</div>
<div class="leg" style="color:#9a3412;margin-top:4px;text-align:center;font-weight:600;">Stage1<br>压缩</div>
</div>
<div>
<div class="col-h">turn5 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0 ▸ S · fb0</div>
<div class="chip comp">T1 ▸ S · fb1</div>
<div class="chip comp">T2 ▸ O·A·AR·S · fb2</div>
<div class="chip comp">T3 ▸ O·A·AR·S · fb3</div>
<div class="chip new">T4 ▸ O·A·AR·S · fb4</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD5 ✗</div>
</div>
</div>

<div class="cap" style="max-width:60rem;margin:0.5rem auto 0;">turn1 降级为 S；前缀延至 <b>T0:S</b> 后断裂，T2/T3 重算。recent-k=3 窗口（T2/T3/T4）保持全量。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算（删除线=被丢弃）</span>
</div>

---
transition: slide-up
title: "05 · turn5 -> turn6"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn5 ➜ turn6（批量压缩）</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:64rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn5 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0 ▸ S · fb0</div>
<div class="chip reuse">T1 ▸ S · fb1</div>
<div class="chip comp strike">T2 ▸ O·A·AR·S · fb2</div>
<div class="chip comp strike">T3 ▸ O·A·AR·S · fb3</div>
<div class="chip comp strike">T4 ▸ O·A·AR·S · fb4</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD5 ✗</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:2.4rem;">
<div style="font-size:2rem;color:#9a3412;line-height:1;">➜</div>
<div class="leg" style="color:#9a3412;margin-top:4px;text-align:center;font-weight:600;">Stage1<br>批量压缩</div>
</div>
<div>
<div class="col-h">turn6 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0 ▸ S · fb0</div>
<div class="chip reuse">T1 ▸ S · fb1</div>
<div class="chip comp">T2 ▸ S · fb2</div>
<div class="chip comp">T3 ▸ S · fb3</div>
<div class="chip comp">T4 ▸ S · fb4</div>
<div class="chip new">T5 ▸ S · fb5</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD6 ✗</div>
</div>
</div>

<div class="cap" style="max-width:62rem;margin:0.5rem auto 0;">输入逼近上限 → <b>rollout 前</b>触发 Stage1 批量压缩：T2/T3/T4 的 O/A/AR 全折叠为 S；前缀在 T2 断裂，T3/T4 重算。原子 turn：压缩只发生在 turn 之间，不打断 turn 内 rollout。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算（删除线=被丢弃）</span>
</div>

---
transition: slide-up
title: "05 · turn6 -> turn7"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn6 ➜ turn7（高复用）</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:60rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn6 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0…T5 ▸ S · fb{0…5}（6 个）</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD6 ✗</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:1.8rem;">
<div style="font-size:2rem;color:#166534;line-height:1;">➜</div>
<div class="leg" style="color:#166534;margin-top:4px;text-align:center;font-weight:600;">append<br>高复用</div>
</div>
<div>
<div class="col-h">turn7 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0…T5 ▸ S · fb{0…5}（6 个）</div>
<div class="chip new">T6 ▸ O·A·AR·S · fb6</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD7 ✗</div>
</div>
</div>

<div class="cap" style="max-width:58rem;margin:0.5rem auto 0;">压缩后前缀短而稳定 ➜ <b>整段 KV-cache 命中</b>，仅追加 turn6。SCRIBE 的 KV-cache 红利显现：每 turn 只为新增后缀付费。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算（删除线=被丢弃）</span>
</div>

---
transition: slide-up
title: "05 · turn7 -> turn8"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn7 ➜ turn8（高复用）</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:60rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn7 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0…T5 ▸ S · fb{0…5}（6 个）</div>
<div class="chip reuse">T6 ▸ O·A·AR·S · fb6</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD7 ✗</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:1.8rem;">
<div style="font-size:2rem;color:#166534;line-height:1;">➜</div>
<div class="leg" style="color:#166534;margin-top:4px;text-align:center;font-weight:600;">append<br>高复用</div>
</div>
<div>
<div class="col-h">turn8 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0…T5 ▸ S · fb{0…5}（6 个）</div>
<div class="chip reuse">T6 ▸ O·A·AR·S · fb6</div>
<div class="chip new">T7 ▸ O·A·AR·S · fb7</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD8 ✗</div>
</div>
</div>

<div class="cap" style="max-width:58rem;margin:0.5rem auto 0;">整段复用 + 追加 turn7；前缀越长，复用比例越高，单 turn 增量成本越小。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算（删除线=被丢弃）</span>
</div>

---
transition: slide-up
title: "05 · turn8 -> turn9"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn8 ➜ turn9（再次批量压缩）</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:62rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn8 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0…T5 ▸ S · fb{0…5}（6 个）</div>
<div class="chip comp strike">T6 ▸ O·A·AR·S · fb6</div>
<div class="chip comp strike">T7 ▸ O·A·AR·S · fb7</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD8 ✗</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:2rem;">
<div style="font-size:2rem;color:#9a3412;line-height:1;">➜</div>
<div class="leg" style="color:#9a3412;margin-top:4px;text-align:center;font-weight:600;">Stage1<br>批量压缩</div>
</div>
<div>
<div class="col-h">turn9 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">FUP · first_user_prompt</div>
<div class="chip reuse">T0…T5 ▸ S · fb{0…5}（6 个）</div>
<div class="chip comp">T6 ▸ S · fb6</div>
<div class="chip comp">T7 ▸ S · fb7</div>
<div class="chip new">T8 ▸ S · fb8</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD9 ✗</div>
</div>
</div>

<div class="cap" style="max-width:60rem;margin:0.5rem auto 0;">再次逼近上限 ➜ Stage1 压缩 T6/T7 为 S；前缀在 T6 断裂。压缩是周期性的：每当 recent-k 窗口外累积过多 O/A/AR，就折叠一次。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算（删除线=被丢弃）</span>
</div>

---
transition: slide-up
title: "05 · turn9 -> turn10"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn9 ➜ turn10（Stage1 ➜ Stage2 双重压缩）</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:60rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn9 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip comp strike">FUP · first_user_prompt</div>
<div class="chip comp strike">T0…T8 ▸ S · fb{0…8}（9 个）</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD9 ✗</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:1.6rem;">
<div style="font-size:1.9rem;color:#9a3412;line-height:1;">➜</div>
<div class="leg" style="color:#9a3412;margin-top:4px;text-align:center;font-weight:600;">Stage1<br>仍超限<br>↓<br>Stage2<br>sota-LLM</div>
</div>
<div>
<div class="col-h">turn10 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip comp">S ▸ sota-LLM 摘要（折叠 FUP + 10 个 S · fb0…fb9）</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD10 ✗</div>
</div>
</div>

<div class="cap" style="max-width:60rem;margin:0.5rem auto 0;">Stage1 收集 S 后仍超上限 ➜ <b>Stage2</b> 调 sota-LLM 把 FUP + 所有 S + fb9 摘要成<b>单个</b> &lt;turn_summary&gt;；前缀只剩 SP。仅此极端情形需一次 LLM 调用，绝大多数压缩是零成本的 Stage1。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算（删除线=被丢弃）</span>
</div>

---
transition: slide-up
title: "05 · turn10 -> turn11"
---

<div class="page-title">05 · SCRIBE 上下文管理机制 - turn10 ➜ turn11（终止）</div>

<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:0.6rem;max-width:60rem;margin:0 auto;align-items:start;">
<div>
<div class="col-h">turn10 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">S ▸ sota-LLM 摘要</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;">REWARD10 ✗</div>
</div>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:1.8rem;">
<div style="font-size:2rem;color:#166534;line-height:1;">➜</div>
<div class="leg" style="color:#166534;margin-top:4px;text-align:center;font-weight:600;">append<br>命中</div>
</div>
<div>
<div class="col-h">turn11 输入（context）</div>
<div class="chip reuse">SP · system_prompt</div>
<div class="chip reuse">S ▸ sota-LLM 摘要</div>
<div class="chip new">T10 ▸ O·A·AR·S · fb10</div>
<div class="chip neu" style="margin-top:6px;">▸ rollout T/O/(A+AR)+R+S</div>
<div class="cap" style="margin-top:3px;color:#166534;font-weight:600;">REWARD11 ✓ 答案正确 / 达阈值</div>
</div>
</div>

<div class="cap" style="max-width:58rem;margin:0.5rem auto 0;">整段复用 + 追加 turn10；turn11 提交答案正确（或 REWARD11 达阈值）➜ <b>loop 终止</b> ✓。整条轨迹：12 个 turn，仅 5 次压缩事件，其余 6 次过渡为纯 append（KV-cache 命中）。</div>
<div style="display:flex;justify-content:center;gap:0.7rem;margin-top:0.35rem;" class="leg">
<span class="chip reuse inl">绿 = KV-cache 复用</span>
<span class="chip new inl">蓝 = 新增需计算</span>
<span class="chip comp inl">橙 = 压缩/重算（删除线=被丢弃）</span>
</div>

---
transition: slide-up
title: "05 · 上下文机制定义 (1/2)"
---

<div class="page-title">05 · 上下文机制定义（1/2）- Block 模型与压缩规则</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;max-width:64rem;margin:0 auto;">

<div class="p-3 rounded-lg" style="background:#eff6ff;border:1px solid #bfdbfe;">
<div style="font-weight:600;color:#1e40af;font-size:0.95rem;margin-bottom:0.3rem;">Block 标签与顺序</div>
<div style="font-size:0.78rem;color:#374151;line-height:1.5;">
T=think · A=tool_call · O=output · R=reflect · S=turn_summary · AR=tool_response<br>
一个 turn 的 rollout = <code>list[T/O/(list[A]+list[AR])] + R + S</code><br>
<b>step = 一次 LLM call</b>；N 个 tool_call ➜ N+1 步；最后非工具步产出 O+R+S 结束 turn<br>
（<b>注</b>：rollout 实际输入始终排除 T&R——它们一次性；此处列出全部 block 是为训练时完整记录）
</div>
</div>

<div class="p-3 rounded-lg" style="background:#f5f3ff;border:1px solid #ddd6fe;">
<div style="font-weight:600;color:#5b21b6;font-size:0.95rem;margin-bottom:0.3rem;">Block 层 vs 消息层</div>
<div style="font-size:0.78rem;color:#374151;line-height:1.5;">
标签在 <b>block 层</b>始终保留（供解析器切分 ScribeBlock）<br>
<b>消息层</b>（模型下次实际读到的）：<br>
• &lt;think&gt;/&lt;reflect&gt; <b>永不</b>出现在未来输入（一次性）<br>
• &lt;turn_summary&gt; 保留标签（recent 与压缩后 old 都留）<br>
• &lt;tool_call&gt; 在无 S 的 old turn 降级为 O+A 时保留<br>
• OUTPUT 裸文本；AR 由 role=tool 承载
</div>
</div>

<div class="p-3 rounded-lg" style="background:#ecfdf5;border:1px solid #a7f3d0;">
<div style="font-weight:600;color:#047857;font-size:0.95rem;margin-bottom:0.3rem;">压缩三规则</div>
<div style="font-size:0.78rem;color:#374151;line-height:1.5;">
① <b>T&R 一次性</b>：生成后在下一步前剥离，永不进未来输入<br>
② <b>recent-k 保留</b>：recent-k（默认 k=3）之前的 turn 只保留 S（标签完整）<br>
③ <b>turn 原子性</b>：压缩只在 rollout <b>前</b>触发（check-before-rollout），不在 turn 内中途触发——赌单 turn 不超长；首个大 turn 也难超当今 1M context
</div>
</div>

<div class="p-3 rounded-lg" style="background:#fffbeb;border:1px solid #fde68a;">
<div style="font-weight:600;color:#92400e;font-size:0.95rem;margin-bottom:0.3rem;">无 S 兜底</div>
<div style="font-size:0.78rem;color:#374151;line-height:1.5;">
若某 turn 无 S（rollout 格式错）：<br>
• 该 turn 老到只需保留 S 时 ➜ 改保留 <b>O + A</b><br>
&nbsp;&nbsp;（O 作裸文本，A 带 &lt;tool_call&gt; 标签）<br>
• recent 但无 S 的 turn ➜ 保留 O/A/AR<br>
下文示例均假设每 turn 正确生成 S
</div>
</div>

</div>

---
transition: slide-up
title: "05 · 上下文机制定义 (2/2)"
---

<div class="page-title">05 · 上下文机制定义（2/2）- 压缩集合、两阶段与硬边界</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;max-width:64rem;margin:0 auto;">

<div class="p-3 rounded-lg" style="background:#eff6ff;border:1px solid #bfdbfe;">
<div style="font-weight:600;color:#1e40af;font-size:0.95rem;margin-bottom:0.3rem;">两个集合（关键区分）</div>
<div style="font-size:0.78rem;color:#374151;line-height:1.5;">
<b>Token-count 集合</b>（决定「是否压缩」）：整个输入 context = system_prompt + first_user_prompt + 所有 turn-feedback + 所有 input blocks，一起计数（<b>SP 也算</b>）<br>
<b>Compression 集合</b>（决定「压缩什么」）：分阶段增长。<b>SP 永不压缩</b>（逐字保留——承载 task/protocol，须跨轨迹一致）
</div>
</div>

<div class="p-3 rounded-lg" style="background:#f5f3ff;border:1px solid #ddd6fe;">
<div style="font-weight:600;color:#5b21b6;font-size:0.95rem;margin-bottom:0.3rem;">两阶段压缩</div>
<div style="font-size:0.78rem;color:#374151;line-height:1.5;">
<b>Stage1（收集 S）</b>：只动 input blocks，旧 turn 折叠为 S；FUP + feedbacks 逐字保留<br>
<b>Stage2（sota-LLM，当 S 序列仍接近上限时触发）</b>：LLM 输入<b>包含</b> FUP + feedbacks，一起折叠进单个 &lt;turn_summary&gt;；返回 <code>{"summary":"..."}</code>，校验为恰好一个 turn_summary 块<br>
仅 SP 逐字穿过两阶段；压缩直到不再接近上限才 rollout；单块仍超则重试 sota-LLM（其重试/退避与 SCRIBE 解耦）
</div>
</div>

<div class="p-3 rounded-lg" style="background:#fef2f2;border:1px solid #fecaca;">
<div style="font-weight:600;color:#991b1b;font-size:0.95rem;margin-bottom:0.3rem;">硬边界（Hard Boundary）</div>
<div style="font-size:0.78rem;color:#374151;line-height:1.5;">
若<b>不可压缩前缀</b> = SP + FUP + 所有 feedbacks（所有 input block 已压成空）自身仍超上限 ➜ 压缩无能为力<br>
此时 SCRIBE <b>抛错</b>：轨迹配置不可行（逐字保留部分单独溢出 context window）
</div>
</div>

<div class="p-3 rounded-lg" style="background:#ecfdf5;border:1px solid #a7f3d0;">
<div style="font-weight:600;color:#047857;font-size:0.95rem;margin-bottom:0.3rem;">check-before-rollout 纪律</div>
<div style="font-size:0.78rem;color:#374151;line-height:1.5;">
压缩是 <b>turn 级</b>（非 turn 内级）：<br>
• 整个 input blocks 的 token 接近上限 ➜ rollout <b>前</b>自动触发压缩<br>
• 压缩到不再接近上限 ➜ 才进入 rollout<br>
• 该 turn 最终存储：压缩后 input + 新生成/rollout blocks + reward<br>
权衡：原子 turn 的 rollout 是对「该 turn 不至过长使 context 崩」的合理下注
</div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
---

<div class="sec-big">06 · reward mechanism of a turn</div>

<div class="sec-sub">16 个 metric（多维 reward 防 hacking）+ 权重配比</div>

---
transition: slide-up
title: "06 · Metric 1"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 1</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#dbeafe;border:1px solid #93c5fd;">
<div style="font-size:2.6rem;font-weight:300;color:#1e40af;line-height:1;">M1</div>
<div style="font-size:0.95rem;font-weight:600;color:#1e40af;margin-top:0.3rem;">最终步奖励</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">final-step reward</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #93c5fd;color:#1e40af;font-weight:700;font-size:0.8rem;">w = 0.52</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">核心信号 · answer correctness<br>判定：环境（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：本 turn 最终步的答案质量与终止方式<br>
<b>机制</b>：reward = correctness(1a) × natural_termination(1b) + truncation_penalty(1c)<br>
• <b>1a 答案正确性</b>：提交答案 == 正确答案 ➜ 1.0，否则 0.0（二元）<br>
• <b>1b 自然终止</b>：必须以最终非工具 LLM call 结束（terminated && done_reason=="no_tool_call"），否则该因子 0<br>
• <b>1c 截断惩罚</b>：被步数上限截断 ➜ −0.2<br>
<b>正交设计</b>：格式由 M4 管，M1 不查块格式 ➜ 便于消融<br>
仅「干净最终非工具步产出的正确答案」拿满分；原始区间 [−0.2, 1] 仿射缩放到 [0,1]
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 2"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 2</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#f1f5f9;border:1px solid #cbd5e1;">
<div style="font-size:2.6rem;font-weight:300;color:#475569;line-height:1;">M2</div>
<div style="font-size:0.95rem;font-weight:600;color:#475569;margin-top:0.3rem;">工具使用质量</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">tool-usage quality</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #cbd5e1;color:#475569;font-weight:700;font-size:0.8rem;">w = 0.08</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">基础设施 · infrastructure<br>判定：环境（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：本 turn rollout 的工具使用质量<br>
<b>机制</b>：score = 1 − 2a 惩罚 − 2b 惩罚<br>
• <b>2a submit 恰好一次</b>：偏离则 −0.15 / 单位<br>
• <b>2b 同一消息内并行相同工具调用</b>（name + 归一化 args 相同）：每个重复 −0.10<br>
<b>跨步重复不罚</b>（环境状态可能已变：run ➜ edit ➜ run）<br>
阶梯式扣分，单个小错只削分不归零
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 3"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 3</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#d1fae5;border:1px solid #6ee7b7;">
<div style="font-size:2.6rem;font-weight:300;color:#047857;line-height:1;">M3</div>
<div style="font-size:0.95rem;font-weight:600;color:#047857;margin-top:0.3rem;">步数长度</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">step length</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #6ee7b7;color:#047857;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">效率 · efficiency<br>纯统计（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：步数长度（A 块计数）<br>
<b>机制</b>：score = 1 / (1 + n_a)<br>
<b>目标</b>：在保持 M1 同水平的同时，让 M3 尽可能低<br>
更少 tool_call 步 ➜ 更高分（更高效）<br>
纯统计；与 M5（token 量）共同构成效率维度
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 4"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 4</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#ede9fe;border:1px solid #c4b5fd;">
<div style="font-size:2.6rem;font-weight:300;color:#5b21b6;line-height:1;">M4</div>
<div style="font-size:0.95rem;font-weight:600;color:#5b21b6;margin-top:0.3rem;">格式正确性</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">format correctness</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #c4b5fd;color:#5b21b6;font-weight:700;font-size:0.8rem;">w = 0.10</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">守门员 · gatekeeper<br>判定：解析器（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.8rem;color:#374151;line-height:1.55;">
<b>衡量</b>：格式正确性（守门员）<br>
<b>机制</b>：score = 1 − Σ 违约惩罚，阶梯式扣分<br>
• <b>4a</b> 期望块类型齐全（T/O/A/AR/R/S）<br>
• <b>4b</b> 最终非工具步以 R 再 S 结尾（倒数第二 R，最后 S）<br>
• <b>4c</b> R/S 只在最终步出现<br>
• <b>4d</b> 最终步不含 tool_call<br>
• <b>4e</b> 仅允许约定 SCRIBE 标签（禁 &lt;submit&gt;/&lt;bash&gt;/&lt;plan&gt;…）<br>
• <b>4f</b> R/S 必须在 submit 之后<br>
• <b>4g</b> 解析无效份额（嵌套标签 / 坏 JSON）
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 5"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 5</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#d1fae5;border:1px solid #6ee7b7;">
<div style="font-size:2.6rem;font-weight:300;color:#047857;line-height:1;">M5</div>
<div style="font-size:0.95rem;font-weight:600;color:#047857;margin-top:0.3rem;">rollout 总 token</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">total rollout tokens</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #6ee7b7;color:#047857;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">效率 · efficiency<br>纯统计（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：本 turn rollout 总 token（T/O/A/R/S，<b>AR 除外</b>）<br>
<b>机制</b>：score = ref / (n + ref)，ref = 8192<br>
更多 token ➜ 更低分；<b>软衰减</b>，不骤降至 0<br>
对正常长度 rollout 信号常驻；不应主导 reward
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 6"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 6</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#fef3c7;border:1px solid #fcd34d;">
<div style="font-size:2.6rem;font-weight:300;color:#92400e;line-height:1;">M6</div>
<div style="font-size:0.95rem;font-weight:600;color:#92400e;margin-top:0.3rem;">token 复用率</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">token reuse ratio</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #fcd34d;color:#92400e;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">summary 防抄袭 · 符号性<br>纯统计（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：summary 对 ground-truth 关键 token 的复用率<br>
<b>机制</b>：reuse = |S ∩ (O∪A)| / |O∪A|<br>
S = turn_summary token 集；分母仅 O∪A（S 故意排除，避免被自身长度膨胀）<br>
衡量「summary 复用了多少 ground-truth 关键 token」；纯统计
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 7"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 7</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#ccfbf1;border:1px solid #5eead4;">
<div style="font-size:2.6rem;font-weight:300;color:#0f766e;line-height:1;">M7</div>
<div style="font-size:0.95rem;font-weight:600;color:#0f766e;margin-top:0.3rem;">忠实度</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">faithfulness</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #5eead4;color:#0f766e;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">summary 质量 · judge<br>LLM judge（7-10 合一调用）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：summary 是否如实报告 O/A（忠实度）<br>
<b>机制</b>：LLM judge 对照 ground-truth(O+A) 评分<br>
高分 = 准确报告所思所做；低分 = 捏造 / 遗漏 / 矛盾<br>
<b>M7-M10 合并在一次 LLM 调用</b>返回四维分数
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 8"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 8</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#ccfbf1;border:1px solid #5eead4;">
<div style="font-size:2.6rem;font-weight:300;color:#0f766e;line-height:1;">M8</div>
<div style="font-size:0.95rem;font-weight:600;color:#0f766e;margin-top:0.3rem;">方向中立</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">direction neutrality</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #5eead4;color:#0f766e;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">summary 质量 · judge<br>LLM judge（7-10 合一调用）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：summary 是否保持回顾性（方向中立）<br>
<b>机制</b>：LLM judge；summary 只能回头看本 turn，<b>不得预测 / 规划下一 turn</b><br>
高分 = 纯回顾；低分 = 泄漏到未来规划<br>
保证 S 是「记录」而非「指令」，避免污染下游 turn
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 9"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 9</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#ccfbf1;border:1px solid #5eead4;">
<div style="font-size:2.6rem;font-weight:300;color:#0f766e;line-height:1;">M9</div>
<div style="font-size:0.95rem;font-weight:600;color:#0f766e;margin-top:0.3rem;">turn 聚焦</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">turn focus</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #5eead4;color:#0f766e;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">summary 质量 · judge<br>LLM judge（7-10 合一调用）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：summary 是否只描述本 turn（turn focus）<br>
<b>机制</b>：LLM judge；严格限定当前 turn，不得混入更早 turn 内容<br>
（后续 turn 内容构造上不会出现；此维只防「前序 turn」渗入）<br>
高分 = 仅本 turn；低分 = 折入前序 turn 行为
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 10"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 10</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#ccfbf1;border:1px solid #5eead4;">
<div style="font-size:2.6rem;font-weight:300;color:#0f766e;line-height:1;">M10</div>
<div style="font-size:0.95rem;font-weight:600;color:#0f766e;margin-top:0.3rem;">流畅度</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">fluency</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #5eead4;color:#0f766e;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">summary 质量 · judge<br>LLM judge（7-10 合一调用）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：summary 流畅度<br>
<b>机制</b>：LLM judge；防止为堆高 M6 复用率而生硬堆砌关键词<br>
高分 = 自然流畅；与 M6 协同（既要高复用又要可读）<br>
reflect 块是让后续 turn_summary 拿更高 reward 的 CoT 过程
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 11"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 11</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#fef3c7;border:1px solid #fcd34d;">
<div style="font-size:2.6rem;font-weight:300;color:#92400e;line-height:1;">M11</div>
<div style="font-size:0.95rem;font-weight:600;color:#92400e;margin-top:0.3rem;">压缩比</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">compression ratio</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #fcd34d;color:#92400e;font-weight:700;font-size:0.8rem;">w = 0.01</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">summary 防抄袭 · 符号性<br>纯统计（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：summary 压缩比<br>
<b>机制</b>：ratio = len(S tokens) / len(O∪A tokens)；score = 1 − ratio<br>
防止把整段 ground-truth 当 summary（会拖垮 M11）<br>
纯统计；与 M6（用集合）不同，此处用<b>长度</b>
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 12"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 12</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#fef3c7;border:1px solid #fcd34d;">
<div style="font-size:2.6rem;font-weight:300;color:#92400e;line-height:1;">M12</div>
<div style="font-size:0.95rem;font-weight:600;color:#92400e;margin-top:0.3rem;">跨块 n-gram 重叠</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">cross-block n-gram overlap</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #fcd34d;color:#92400e;font-weight:700;font-size:0.8rem;">w = 0.03</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">summary 防抄袭 · anti copy-paste<br>纯统计（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：跨块 n-gram 重叠（防 copy-paste）<br>
<b>机制</b>：n=4；overlap = |shared| / |ngrams(S)|；score = 1 − overlap<br>
#11 只挡整段照搬，#12 抓「部分逐字搬运」（连续短语）--能绕过 #11、甚至骗过 #10<br>
纯统计，廉价
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 13"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 13</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#fef3c7;border:1px solid #fcd34d;">
<div style="font-size:2.6rem;font-weight:300;color:#92400e;line-height:1;">M13</div>
<div style="font-size:0.95rem;font-weight:600;color:#92400e;margin-top:0.3rem;">块内 n-gram 重叠</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">intra-block n-gram overlap</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #fcd34d;color:#92400e;font-weight:700;font-size:0.8rem;">w = 0.03</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">summary 防抄袭 · anti loop<br>纯统计（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.8rem;color:#374151;line-height:1.55;">
<b>衡量</b>：块内 n-gram 重叠（防块内自循环）<br>
<b>机制</b>：对 T/O/R/S 块（A 结构化天然重复、AR 环境产出，均排除）<br>
intra = (total_ngrams − unique_ngrams) / total_ngrams；token 加权均值；score = 1 − penalty<br>
#12 只抓跨块抄袭，看不见块内自我重复（小模型 / 早期 RL 常见退化）
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 14"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 14</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#f1f5f9;border:1px solid #cbd5e1;">
<div style="font-size:2.6rem;font-weight:300;color:#475569;line-height:1;">M14</div>
<div style="font-size:0.95rem;font-weight:600;color:#475569;margin-top:0.3rem;">工具参数合法性</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">malformed tool-call penalty</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #cbd5e1;color:#475569;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">基础设施 · infrastructure<br>判定：执行层（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：工具调用参数合法性<br>
<b>机制</b>：每个 malformed_arguments / exec_error 调用 −0.10；clamp [0,1]<br>
例：JSON 不可解析、args 非对象、缺必填参（bash.command）、多出幻觉参（bash.stdout）<br>
完美 = 1.0；给 GRPO <b>连续梯度</b>抑制，而非任其循环到截断
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 15"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 15</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#f1f5f9;border:1px solid #cbd5e1;">
<div style="font-size:2.6rem;font-weight:300;color:#475569;line-height:1;">M15</div>
<div style="font-size:0.95rem;font-weight:600;color:#475569;margin-top:0.3rem;">跨步重复工具调用</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">repeated tool-call penalty</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #cbd5e1;color:#475569;font-weight:700;font-size:0.8rem;">w = 0.02</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">基础设施 · infrastructure<br>判定：执行层（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.82rem;color:#374151;line-height:1.6;">
<b>衡量</b>：跨步重复工具调用<br>
<b>机制</b>：每个与本 turn 早先执行过的 (name, 归一化 args) 完全相同的调用 −0.08；clamp [0,1]<br>
例：bash{ls} ➜ 结果 ➜ bash{ls}；与 M2b（单消息内并行重复）互补<br>
完美 = 1.0；与 M14 一起止住「重复 / 坏调用浪费步数到截断」
</div>
</div>
</div>

---
transition: slide-up
title: "06 · Metric 16"
---

<div class="page-title">06 · the reward mechanism of a turn - Metric 16</div>

<div style="display:grid;grid-template-columns:230px 1fr;gap:0.8rem;max-width:62rem;margin:0 auto;align-items:start;">
<div class="p-4 rounded-lg" style="background:#fee2e2;border:1px solid #fca5a5;">
<div style="font-size:2.6rem;font-weight:300;color:#991b1b;line-height:1;">M16</div>
<div style="font-size:0.95rem;font-weight:600;color:#991b1b;margin-top:0.3rem;">跨 turn 重复提交</div>
<div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">cross-turn duplicate submit</div>
<div style="margin-top:0.7rem;"><span style="background:white;padding:2px 8px;border-radius:4px;border:1px solid #fca5a5;color:#991b1b;font-weight:700;font-size:0.8rem;">w = 0.05</span></div>
<div style="font-size:0.7rem;color:#6b7280;margin-top:0.5rem;">防停滞 · anti stalling（唯一跨 turn）<br>判定：history_manager（无 judge）</div>
</div>
<div class="p-4 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;">
<div style="font-size:0.8rem;color:#374151;line-height:1.55;">
<b>衡量</b>：跨 turn 重复提交相同答案（<b>唯一跨 turn 惩罚</b>）<br>
<b>机制</b>：三条件全满足才触发：①本 turn submit ≥ 1 ②存在上一 turn 答案 ③本 turn 答案 == 上一 turn 答案<br>
触发：1.0 − 0.20；否则 1.0（含两次答案不同--正是「用反馈改进」）<br>
防「忽略反馈、逐 turn 重提相同答案」的 stalling；intra-turn 指标(2a/2b/15)看不见<br>
prev_turn_answer 由 history_manager 在 finalize 时读出
</div>
</div>
</div>

---
transition: slide-up
title: "06 · 16 metric 权重配比总结"
---

<div class="page-title">06 · the reward mechanism of a turn - 16 metric 权重配比总结</div>

<div style="display:grid;grid-template-columns:1.35fr 1fr;gap:0.7rem;max-width:66rem;margin:0 auto;align-items:start;">

<div style="font-size:0.66rem;">
<table style="border-collapse:collapse;width:100%;"><tbody>
<tr style="background:#f3f4f6;"><th style="padding:3px 6px;border:1px solid #d1d5db;text-align:left;">M</th><th style="padding:3px 6px;border:1px solid #d1d5db;text-align:left;">metric</th><th style="padding:3px 6px;border:1px solid #d1d5db;">w</th><th style="padding:3px 6px;border:1px solid #d1d5db;text-align:left;">类别</th></tr>
<tr><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M1</td><td style="padding:3px 6px;border:1px solid #d1d5db;">最终步奖励 answer+终止</td><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;color:#1e40af;">0.52</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#dbeafe;">核心</td></tr>
<tr style="background:#f9fafb;"><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M2</td><td style="padding:3px 6px;border:1px solid #d1d5db;">工具使用质量</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.08</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#f1f5f9;">基础设施</td></tr>
<tr><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M3</td><td style="padding:3px 6px;border:1px solid #d1d5db;">步数长度</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#d1fae5;">效率</td></tr>
<tr style="background:#f9fafb;"><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M4</td><td style="padding:3px 6px;border:1px solid #d1d5db;">格式正确性</td><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">0.10</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#ede9fe;">守门员</td></tr>
<tr><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M5</td><td style="padding:3px 6px;border:1px solid #d1d5db;">rollout 总 token</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#d1fae5;">效率</td></tr>
<tr style="background:#f9fafb;"><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M6</td><td style="padding:3px 6px;border:1px solid #d1d5db;">token 复用率</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#fef3c7;">防抄袭</td></tr>
<tr><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M7</td><td style="padding:3px 6px;border:1px solid #d1d5db;">忠实度 judge</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#ccfbf1;">summary 质量</td></tr>
<tr style="background:#f9fafb;"><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M8</td><td style="padding:3px 6px;border:1px solid #d1d5db;">方向中立 judge</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#ccfbf1;">summary 质量</td></tr>
<tr><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M9</td><td style="padding:3px 6px;border:1px solid #d1d5db;">turn 聚焦 judge</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#ccfbf1;">summary 质量</td></tr>
<tr style="background:#f9fafb;"><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M10</td><td style="padding:3px 6px;border:1px solid #d1d5db;">流畅度 judge</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#ccfbf1;">summary 质量</td></tr>
<tr><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M11</td><td style="padding:3px 6px;border:1px solid #d1d5db;">压缩比</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.01</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#fef3c7;">防抄袭</td></tr>
<tr style="background:#f9fafb;"><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M12</td><td style="padding:3px 6px;border:1px solid #d1d5db;">跨块 n-gram 重叠</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.03</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#fef3c7;">防抄袭</td></tr>
<tr><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M13</td><td style="padding:3px 6px;border:1px solid #d1d5db;">块内 n-gram 重叠</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.03</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#fef3c7;">防抄袭</td></tr>
<tr style="background:#f9fafb;"><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M14</td><td style="padding:3px 6px;border:1px solid #d1d5db;">工具参数合法</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#f1f5f9;">基础设施</td></tr>
<tr><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M15</td><td style="padding:3px 6px;border:1px solid #d1d5db;">跨步重复工具</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.02</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#f1f5f9;">基础设施</td></tr>
<tr style="background:#f9fafb;"><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:700;">M16</td><td style="padding:3px 6px;border:1px solid #d1d5db;">跨 turn 重复提交</td><td style="padding:3px 6px;border:1px solid #d1d5db;">0.05</td><td style="padding:3px 6px;border:1px solid #d1d5db;background:#fee2e2;">防停滞</td></tr>
<tr style="background:#e5e7eb;"><td style="padding:3px 6px;border:1px solid #d1d5db;" colspan="2"><b>合计 Σ wᵢ</b></td><td style="padding:3px 6px;border:1px solid #d1d5db;font-weight:800;color:#1e40af;">1.00</td><td style="padding:3px 6px;border:1px solid #d1d5db;">clamp [0,1]</td></tr>
</tbody></table>
</div>

<div class="p-3 rounded-lg" style="background:#f9fafb;border:1px solid #e5e7eb;font-size:0.74rem;color:#374151;line-height:1.5;">
<div style="font-weight:700;color:#5a7a8a;margin-bottom:0.3rem;">类别小计</div>
• 核心 answer：M1 = <b>0.52</b><br>
• 守门员 format：M4 = <b>0.10</b><br>
• 基础设施 tool：M2+M14+M15 = <b>0.12</b><br>
• 效率：M3+M5 = <b>0.04</b><br>
• summary 质量(judge)：M7-M10 = <b>0.08</b><br>
• summary 防抄袭(统计)：M6+M11+M12+M13 = <b>0.09</b><br>
• 防停滞：M16 = <b>0.05</b>
<div style="font-weight:700;color:#5a7a8a;margin:0.5rem 0 0.3rem;">设计哲学</div>
• answer 主导但不过激（0.52）<br>
• 防抄袭指标为符号/反抄袭信号，非主驱动<br>
• <b>多维 reward 防 hacking</b><br>
• M7-M10 合并一次 LLM 调用<br>
• total = clamp(Σ wᵢ·mᵢ, 0, 1)
<div style="font-weight:700;color:#5a7a8a;margin:0.5rem 0 0.3rem;">trajectory reward</div>
<code>trajectory_reward = mean(turn_reward) × decay^len(turns)</code><br>
用 mean 而非仅末 turn：早 turn 坏行为直接受罚；长度衰减奖励更少 turn 解决
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
---

<div class="sec-big">07 · abstract of our contribution</div>

<div class="sec-sub">可作为 A 类会议（如 ICLR）论文摘要</div>

---
transition: slide-up
title: "07 · abstract"
---

<div class="page-title">07 · abstract of our contribution</div>

<div style="max-width:66rem;margin:0 auto;">

<div class="p-4 rounded-lg" style="background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #3b82f6;font-size:0.86rem;color:#1f2937;line-height:1.65;">
We present <b>SCRIBE</b>, a small yet foundational change to the ReAct loop for long-horizon <b>loop-engineering</b> tasks: at the end of every turn the agent explicitly writes a <b>Reflect (R)</b> and a <b>Turn-Summary (S)</b> of what it just did. This single change induces a new <b>context-management mechanism</b> — disposable T/R blocks, recent-<i>k</i> full retention, and automatic <b>Stage-1</b> (collect-S) / <b>Stage-2</b> (sota-LLM) compression — which in turn unlocks four benefits:
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;margin-top:0.8rem;">

<div class="p-3 rounded-lg" style="background:#ecfdf5;border:1px solid #a7f3d0;">
<div style="font-size:1.4rem;font-weight:300;color:#047857;line-height:1;">01</div>
<div style="font-weight:600;color:#047857;font-size:0.9rem;margin-top:0.2rem;">Painless context extension</div>
<div style="font-size:0.76rem;color:#374151;line-height:1.5;margin-top:0.2rem;">Context grows across turns yet stays bounded, almost entirely by <b>zero-cost automatic compression</b> — only the rare Stage-2 needs one LLM call.</div>
</div>

<div class="p-3 rounded-lg" style="background:#ccfbf1;border:1px solid #5eead4;">
<div style="font-size:1.4rem;font-weight:300;color:#0f766e;line-height:1;">02</div>
<div style="font-weight:600;color:#0f766e;font-size:0.9rem;margin-top:0.2rem;">Token-level credit assignment</div>
<div style="font-size:0.76rem;color:#374151;line-height:1.5;margin-top:0.2rem;">Each turn's summary is a first-class, rewarded block, so GRPO credit flows <b>trajectory → turn → step → token</b> — markedly more sample-efficient than broad-ReAct.</div>
</div>

<div class="p-3 rounded-lg" style="background:#dbeafe;border:1px solid #93c5fd;">
<div style="font-size:1.4rem;font-weight:300;color:#1e40af;line-height:1;">03</div>
<div style="font-weight:600;color:#1e40af;font-size:0.9rem;margin-top:0.2rem;">Higher accuracy &amp; token efficiency</div>
<div style="font-size:0.76rem;color:#374151;line-height:1.5;margin-top:0.2rem;">Both accuracy and token efficiency on loop-engineering tasks <b>exceed the pre-training model and the ReAct-trained model</b>.</div>
</div>

<div class="p-3 rounded-lg" style="background:#fef3c7;border:1px solid #fcd34d;">
<div style="font-size:1.4rem;font-weight:300;color:#92400e;line-height:1;">04</div>
<div style="font-weight:600;color:#92400e;font-size:0.9rem;margin-top:0.2rem;">High KV-cache reuse</div>
<div style="font-size:0.76rem;color:#374151;line-height:1.5;margin-top:0.2rem;">Between consecutive turns only a small suffix is new while the <b>whole prefix is cache-reused</b>; compression is the only prefix-breaking event.</div>
</div>

</div>

<div style="text-align:center;margin-top:0.7rem;color:#5a7a8a;font-size:0.85rem;font-weight:500;">
A small &amp; beautiful change → a context-mechanism shift → many derived benefits.
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

<div style="font-size:4rem;font-weight:500;margin-bottom:1rem;">谢谢聆听</div>

<div style="font-size:1.8rem;font-weight:400;color:rgba(255,255,255,0.9);">Q &amp; A</div>

<div style="font-size:0.95rem;color:rgba(255,255,255,0.75);margin-top:2rem;">周浩洋 · JLU, SE · 2025.04.17 – Present</div>
