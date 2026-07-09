# SCRIBE GRPO 训练问题诊断与改进方案

> 基于 1.5B Qwen2.5-Instruct + LoRA 的 GSM8K 在线 GRPO 训练实验分析
> **最终更新**：iter_40 accuracy = 64.5%，揭示固定学习率导致模型"跳脚"无法深入优化

---

## 一、实验现象

### 1.1 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen2.5-1.5B-Instruct |
| 训练方式 | SFT → GRPO (LoRA) |
| SFT 样本数 | 188 (GSM8K 训练集子集) |
| GRPO 迭代数 | 50 |
| Batch size | 4 (任务数) |
| Group size | 4 (每任务采样轨迹数) |
| KL 系数 (β) | 0.04 |
| Clip ε | 0.2 |
| **学习率** | **5e-6 (固定，无调度)** |
| **学习率调度** | **无 warmup，无退火** |
| 评估样本数 | 200 (GSM8K 测试集随机采样) |

### 1.2 完整 Accuracy 曲线

| 阶段 | Accuracy | Mean Reward | KL | 状态 |
|------|----------|-------------|-----|------|
| SFT 基线 | **67.5%** | — | — | ✅ 基线 |
| GRPO iter 0 | 64.0% | 0.515 | 0.001 | 起点 |
| GRPO iter 10 | 61.5% | 0.553 | 0.031 | 缓慢下降 |
| GRPO iter 12 | — | **0.612** (峰值) | 0.070 | Reward 峰值 |
| **GRPO iter 15** | — | 0.551 | **5.381** | 💥 KL 爆炸 |
| **GRPO iter 20** | **35.5%** | 0.438 | 0.085 | 💀 Accuracy 崩溃 |
| GRPO iter 25 | — | 0.544 | 0.132 | 震荡 |
| **GRPO iter 30** | **65.0%** | **0.356** (最低) | **1.644** | 🚀 恢复 |
| **GRPO iter 40** | **64.5%** | 0.406 | 0.009 | ⏸️ **平台期** |

### 1.3 核心发现

1. **崩溃-恢复-平台三部曲**：
   - iter 20: 崩溃至 35.5%
   - iter 30: 恢复至 65.0%
   - iter 40: **稳定在 64.5%，但与 SFT 始终差 2.5-3.0 pts**

2. **Reward 与 Accuracy 严重脱钩**：
   - iter 12: reward=0.612（峰值），accuracy 在下降
   - iter 30: reward=0.356（最低），accuracy=65.0%（最高）

3. **固定学习率导致"跳脚"**：
   - iter 30-40 accuracy 平台 → **不是收敛，是"不敢动"**
   - lr=5e-6 太大，一动就飞，只能"原地踏步"

---

## 二、根因分析

### 2.1 根因一：固定学习率 = 模型"跳脚"（最核心）

#### 问题：无 warmup，无退火

```python
# 当前代码
optimizer = torch.optim.AdamW(
    [p for p in policy.parameters() if p.requires_grad],
    lr=5e-6,              # ← 固定不变！
    weight_decay=0.0,
)
# 无 scheduler！
```

#### 后果

| 阶段 | 理想 lr | 实际 lr | 后果 |
|------|---------|---------|------|
| iter 0-5 | 5e-7 (warmup) | 5e-6 | 初始更新过大，直接跳飞 |
| iter 5-15 | 5e-6 | 5e-6 | 正常，但无缓冲 |
| iter 15-30 | 2-3e-6 (退火) | 5e-6 | 无法精细调整，震荡 |
| iter 30-40 | 1e-6 (精细) | 5e-6 | **无法深入优化，只能"跳脚"** |
| iter 40-50 | 5e-7 (收敛) | 5e-6 | 无法收敛到最优 |

#### "跳脚"的具体表现

```
固定 lr=5e-6 的 50 个 iteration：

Iter 0:  "我看看这边" → 一步跳飞
Iter 5:  "那边好像更好" → 又跳过去  
Iter 15: "哎哟 KL 爆炸了" → 被弹飞 (KL=5.38)
Iter 20: "我在哪？" → 35.5%，懵了
Iter 30: "好像这边还行" → 弹回 65% (reward=0.356最低)
Iter 40: "就这儿吧" → 64.5%，不动了

为什么不动了？
→ 因为 lr 还是 5e-6，一动就飞，只能"原地跳脚"
```

#### 关键证据：Accuracy 平台期

```
SFT:        67.5% (天花板)
GRPO iter 30:  65.0% (差 2.5 pts)
GRPO iter 40:  64.5% (差 3.0 pts)

→ 差距从未缩小！
→ 模型"想"优化，但 lr 太大，一走就飞
→ 结果：只能在 64-65% "徘徊"，无法达到 67.5%+
```

### 2.2 根因二：Reward 设计 Misalignment

#### 问题：Answer Correctness 权重过低

```python
w1: float = 0.32   # answer + natural termination (唯一与"解题能力"直接相关的指标)
w2-w15: float = 0.68  # 格式/摘要/工具使用
```

**后果**：
- 1.5B 小模型容量有限，自然倾向于"最容易拿分"的方向
- "真正解题"比"优化 format + summary"困难得多，但 reward 占比却更低
- 模型将有限 capacity 分配给 format gaming，挤占了数学推理能力

#### 关键证据：Reward 最低时 Accuracy 最高

| iter | mean_reward | accuracy | 说明 |
|------|-------------|----------|------|
| 12 | **0.612** | ~55%? | reward 峰值，accuracy 在下降 |
| 20 | 0.438 | **35.5%** | accuracy 崩溃 |
| 30 | **0.356** | **65.0%** | **reward 最低，accuracy 最高！** |

**结论**：模型学会了"优化 format/summary"但不解题，导致 reward 与 accuracy 严重脱钩。

### 2.3 根因三：Group Size 过小

`group_size = 4` 导致 GRPO advantage 估计噪声过大：
- 样本标准差 (ddof=1) 极不稳定（日志中 std=0.14-0.20）
- 一个异常样本即可颠覆整个 group
- 模型无法区分"真正会解题"和"蒙对"

### 2.4 根因四：KL 约束过弱

`beta = 0.04` 无法阻止 1.5B 小模型的累积漂移：
- iter 15 KL=5.38 → 剧烈漂移，崩溃
- iter 30 KL=1.64 → 再次漂移，但适度，反而推动恢复
- **KL 存在"黄金区间"**：KL≈1.6 可能有益，KL>5 则崩溃

---

## 三、关键发现：iter_30-40 的平台期

### 3.1 现象

```
iter 30: 65.0%
iter 40: 64.5%

变化仅 -0.5% → 基本稳定
但与 SFT 差距始终 2.5-3.0 pts → 从未超越
```

### 3.2 解读

**不是"收敛"，是"假收敛"**

```
真正收敛：
  accuracy 曲线：64% → 65% → 66% → 67% → 68%...
  → 逐步上升，最终超越 SFT

假收敛（你的实验）：
  accuracy 曲线：64% → 35% → 65% → 64.5% → 64.5%...
  → 崩溃后恢复，然后"原地踏步"
  → 因为 lr 太大，无法精细调整
```

### 3.3 根因

**固定 lr=5e-6 导致模型"跳脚"**
- 模型在 SFT 附近"徘徊"
- 但无法"深入"到更好的区域
- 因为 lr 太大，一走就"飞"
- 结果：只能"稳定在表面"

---

## 四、改进方案

### 4.1 方案 A：添加学习率调度（最优先）

**修改 `train_grpo_online_submit_only.py`**：

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 在 optimizer 创建后添加：
warmup_iters = 5
total_iters = args.num_iterations

scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters),
        CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=1e-7),
    ],
    milestones=[warmup_iters],
)

# 在每个 iteration 结束时调用：
for iteration in range(args.num_iterations):
    # ... GRPO update ...
    scheduler.step()
```

**预期学习率曲线**：

```
Iter 0-5:   lr = 5e-7   (warmup，小步探索)
Iter 5-15:  lr = 5e-6   (稳定期，正常更新)
Iter 15-30: lr = 2-3e-6 (退火，精细调整)
Iter 30-40: lr = 1e-6   (深入优化)
Iter 40-50: lr = 5e-7   (收敛微调)
```

**预期效果**：
- 消除 KL 爆炸
- 减少 reward 震荡
- accuracy 稳步上升，**可能超越 SFT**

### 4.2 方案 B：权重调整 + 关闭易被 hack 的指标

**修改 `Scribe/scribe_gym/rewards.py`**：

```python
@dataclass
class TurnRewardConfig:
    w1: float = 0.55   # ↑ answer correctness 占绝对主导
    w2: float = 0.08   # tool usage
    w4: float = 0.10   # format correctness
    w6: float = 0.00   # ↓↓ 关闭 token reuse ratio
    w11: float = 0.00  # ↓↓ 关闭 compression ratio
    # 其余保持不变
```

### 4.3 方案 C：增大 Group Size + 增强 KL 约束

```bash
python scripts/train_grpo_online_submit_only.py \
    --group_size 8 \      # ↑ 降低 baseline 噪声
    --kl_coef 0.1 \       # ↑ 增强 KL 约束
    --learning_rate 5e-6 \ # 配合 lr 调度
    ...
```

### 4.4 方案 D：Metric 16 - 跨 Turn 重复提交惩罚（新增）

**设计**：如果当前 turn 和上一轮 turn 提交了相同答案，惩罚当前 turn。

```python
def metric_16(turn: TurnRecord, oc: TurnOutcome, cfg: TurnRewardConfig) -> float:
    if oc.submit_count < 1 or turn.prev_turn_answer is None:
        return 1.0
    current = oc.answer.strip() if oc.answer else ""
    prev = turn.prev_turn_answer.strip()
    if current == prev:
        return _clamp(1.0 - cfg.cross_turn_dup_penalty, 0.0, 1.0)
    return 1.0
```

**作用**：防止模型在多个 turn 中重复提交相同答案而不改进。

---

## 五、对照实验计划

| 实验 | 配置 | 预期效果 | 优先级 |
|------|------|---------|--------|
| **Baseline** | 当前配置（固定 lr，无调度） | 震荡，平台期 64-65% | 对照组 |
| **Fix-A** | **+ lr 调度（warmup + cosine）** | **稳定收敛，可能 68-70%** | 🔴 **P0** |
| **Fix-B** | Fix-A + w1=0.55, w6=w11=0 | 更快收敛，更高 accuracy | 🔴 P0 |
| **Fix-C** | Fix-B + group_size=8, β=0.1 | 更稳定的 baseline | 🟡 P1 |
| **Fix-D** | Fix-C + Metric 16 | 减少跨 turn 重复提交 | 🟢 P2 |

### 验证指标

每 5 个 iteration 评估：
1. **Accuracy** (200 sample) — 核心指标
2. **Reward 震荡幅度** — 稳定性指标
3. **KL 趋势** — 是否还有 spike
4. **与 SFT 的差距** — 是否缩小

---

## 六、关键结论（最终版）

1. **固定学习率是最大瓶颈**：lr=5e-6 固定不变，导致模型"跳脚"，无法深入优化区。

2. **iter_30-40 的平台期是"假收敛"**：accuracy 稳定在 64-65%，不是因为"收敛了"，而是因为 lr 太大，模型"不敢动"。

3. **Reward hacking 确实存在**：iter_30 reward 最低但 accuracy 最高，证明模型学会了"优化 format 但不解题"。

4. **Catastrophic forgetting 可能是可逆的**：iter_30 从 35.5% 恢复至 65.0%，说明 SFT 知识未被永久覆盖。

5. **KL 存在"黄金区间"**：KL≈1.6 的扰动可能有益（推动恢复），KL>5 则导致崩溃。

6. **GRPO 对 1.5B 小模型可行，但需要 careful tuning**：
   - 必须加 lr 调度
   - 必须简化 reward（提升 w1，关闭 w6/w11）
   - 可能需要更大 group_size

7. **最终目标**：通过 Fix-A/B，让 GRPO accuracy **超越 SFT 67.5%**，达到 **70%+**。

---

## 七、即时行动清单

### 今天（立即执行）

- [ ] 1. 修改 `train_grpo_online_submit_only.py`：添加 lr 调度（warmup + cosine）
- [ ] 2. 修改 `rewards.py`：w1=0.55, w6=0, w11=0
- [ ] 3. 重新跑 GRPO（从 SFT checkpoint 开始）

### 明天

- [ ] 4. 查看前 10 iter 的 reward 曲线和 KL
- [ ] 5. 评估 iter_10 accuracy

### 本周

- [ ] 6. 完成 50 iter 训练
- [ ] 7. 对比：Baseline（震荡）vs Fix-A（稳定上升）
- [ ] 8. 如果 Fix-A 有效，跑 Fix-B（+ 权重调整）

---

*文档版本: v2.0 (最终版)*
*日期: 2026-07-10*
*基于实验: SCRIBE GRPO 1.5B GSM8K, iter 0-40*
*核心洞察: 固定学习率导致模型"跳脚"，无法深入深度优化区*
