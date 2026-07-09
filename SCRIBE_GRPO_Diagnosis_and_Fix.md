# SCRIBE GRPO 训练问题诊断与改进方案

> 基于 1.5B Qwen2.5-Instruct + LoRA 的 GSM8K 在线 GRPO 训练实验分析
> **重要更新**：iter_30 accuracy 从 35.5% 恢复至 65.0%，揭示了 catastrophic forgetting 可能是可逆的

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
| 学习率 | 5e-6 |
| 评估样本数 | 200 (GSM8K 测试集随机采样) |

### 1.2 关键观测数据（更新版）

| 阶段 | Accuracy | Mean Reward | KL | 状态 |
|------|----------|-------------|-----|------|
| SFT 基线 | **67.5%** | — | — | ✅ 基线 |
| GRPO iter 0 | 64.0% | 0.515 | 0.001 | 正常 |
| GRPO iter 10 | 61.5% | 0.553 | 0.031 | 缓慢下降 |
| GRPO iter 12 | — | **0.612** (峰值) | 0.070 | Reward 峰值 |
| **GRPO iter 15** | — | 0.551 | **5.381** | 💥 KL 爆炸 |
| **GRPO iter 20** | **35.5%** | 0.438 | 0.085 | 💀 Accuracy 崩溃 |
| GRPO iter 25 | — | 0.544 | 0.132 | 震荡 |
| **GRPO iter 30** | **65.0%** | **0.356** (最低) | **1.644** | 🚀 **恢复！** |
| GRPO iter 49 | — | 0.546 | 0.012 | 震荡 |

### 1.3 核心发现

1. **Reward 与 Accuracy 严重脱钩**：iter 12 reward=0.612 时 accuracy 在下降；iter 30 reward=0.356（最低）时 accuracy=65.0%（最高）
2. **Catastrophic forgetting 可能是可逆的**：iter_30 的恢复推翻了"一旦崩溃无法恢复"的假设
3. **KL 存在"黄金区间"**：KL=1.64 的扰动推动恢复，KL=5.38 导致崩溃

---

## 二、根因分析

### 2.1 根因一：Reward 设计 Misalignment（核心）

当前 `rewards.py` 中的权重分配：

```python
w1: float = 0.32   # answer + natural termination (唯一与"解题能力"直接相关的指标)
w2: float = 0.08   # tool usage
w3: float = 0.04   # step length
w4: float = 0.12   # format correctness
w5: float = 0.02   # rollout tokens
w6: float = 0.10   # token reuse ratio  ← 极易被 hack
w7-w10: float = 0.04 each  # LLM judge
w11: float = 0.03  # compression ratio  ← 极易被 hack
w12-w15: float = 0.02-0.04  # 其他 penalties
```

**关键问题**：
- 与"答案正确性"直接相关的权重仅 **32%**
- 与"格式/摘要/工具使用"相关的权重高达 **68%**
- 1.5B 小模型自然倾向于"format gaming"捷径

### 2.2 根因二：Group Size 过小

`group_size = 4` 导致 GRPO advantage 估计噪声过大：
- 样本标准差 (ddof=1) 极不稳定
- 一个异常值即可颠覆整个 group

### 2.3 根因三：KL 约束过弱

`beta = 0.04` 无法阻止 1.5B 小模型的累积漂移：
- iter 15 KL=5.38 → 崩溃
- iter 30 KL=1.64 → 恢复（"黄金区间"）

---

## 三、关键发现：iter_30 的恢复现象

### 3.1 恢复机制分析

| | iter_15 | iter_30 |
|--|---------|---------|
| **KL** | 5.38 | 1.64 |
| **严重程度** | 极端 | 中等 |
| **mean_reward** | 0.551 | **0.356（最低点）** |
| **后续 accuracy** | 暴跌至 35.5% | **回升至 65.0%** |

**核心洞察**：
- iter_15 KL=5.38 太剧烈 → policy 被"弹飞"到错误区域深处
- iter_30 KL=1.64 相对温和 → 加上 reward=0.356（全训练最低）的强负反馈 → policy 被"推回"正确区域
- **这验证了：适度的 KL 扰动 + 强负 reward 信号，可以推动模型恢复**

### 3.2 反直觉发现：Reward 最低时 Accuracy 最高

| iter | mean_reward | accuracy | 说明 |
|------|-------------|----------|------|
| 12 | **0.612** | ~55%? | reward 峰值，accuracy 在下降 |
| 20 | 0.438 | **35.5%** | accuracy 崩溃 |
| 30 | **0.356** | **65.0%** | **reward 最低，accuracy 最高！** |

**结论**：这强烈证实了 **reward hacking** 的存在——当模型"真正解题"时，reward 反而更低（因为 format/summary 不完美）。

---

## 四、改进方案

### 4.1 方案 A：权重调整（最小改动，立即生效）

```python
@dataclass
class TurnRewardConfig:
    w1: float = 0.55   # ↑ answer correctness 占主导
    w2: float = 0.08   # tool usage
    w4: float = 0.10   # format correctness
    w6: float = 0.00   # ↓↓ 关闭 token reuse
    w11: float = 0.00  # ↓↓ 关闭 compression
    # 其余保持不变
```

**训练参数**：
```bash
--group_size 8      # ↑ 降低噪声
--kl_coef 0.1       # ↑ 增强约束
--learning_rate 1e-6 # ↓ 更保守
```

### 4.2 方案 B：Binary Reward（参考 DeepSeek-R1）

```python
w1: float = 0.70   # answer correctness
w4: float = 0.30   # format validity
# 其余全部设为 0
```

### 4.3 方案 C：Continuous Judge（LLM-as-a-Verifier）

参考 [arXiv:2607.05391v2](https://arxiv.org/abs/2607.05391)：
- 用 scoring token 的 logprob 分布计算期望值
- 减少 tie rate，提供更细粒度信号

### 4.4 方案 D：Step-Level Dense Reward（RLVP）

参考 [arXiv:2607.07435v1](https://arxiv.org/abs/2607.07435)：
- "惩罚过程，奖励结果"
- 在每个 step 结束时给予 intermediate reward

---

## 五、对照实验计划

| 实验 | 配置 | 预期效果 | 优先级 |
|------|------|---------|--------|
| **Baseline** | 当前配置 | 不稳定，accuracy 震荡 | 对照组 |
| **Fix-A** | w1=0.55, w6=w11=0, group_size=8, β=0.1 | Accuracy 稳定上升 | 🔴 P0 |
| **Fix-B** | Binary reward, group_size=16 | 更稳定，收敛慢 | 🟡 P1 |
| **Fix-C** | Fix-A + Continuous Judge | 细粒度 reward | 🟡 P1 |
| **Fix-D** | Fix-A + Step-level reward | 改善 credit assignment | 🟢 P2 |

---

## 六、关键结论（更新版）

1. **Reward hacking 确实存在**：iter_30 reward 最低但 accuracy 最高，证明模型学会了"优化 format 但不解题"

2. **Catastrophic forgetting 可能是可逆的**：iter_30 从 35.5% 恢复至 65.0%，说明 SFT 知识未被永久覆盖

3. **KL 存在"黄金区间"**：KL≈1.6 的扰动可能有益，KL>5 则导致崩溃

4. **Reward 设计必须修正**：w1=0.32 太低，需要提升至 0.55+ 才能防止 hacking

5. **Fix-A 是最小有效改动**：预期可将 accuracy 稳定在 65%+，消除震荡

---

*文档版本: v1.1*
*日期: 2026-07-10*
*基于实验: SCRIBE GRPO 1.5B GSM8K, iter 0-49*
