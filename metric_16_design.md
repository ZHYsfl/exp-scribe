# Metric 16: Cross-Turn Duplicate Submit Penalty

## 设计动机

在多 turn 的 SCRIBE 交互中，模型可能在多个 turn 中重复提交相同的答案而不进行实质性思考。这种行为：
- 浪费了 turn 预算
- 没有利用 feedback 改进答案
- 是一种"偷懒"策略

Metric 16 旨在惩罚这种行为，鼓励模型在收到 feedback 后真正改进答案。

## 设计规格

### 触发条件

同时满足以下三个条件时触发惩罚：
1. **当前 turn 有 submit**：`oc.submit_count >= 1`
2. **存在上一轮 turn**：`prev_turn_answer is not None`
3. **两轮答案相同**：`current_answer.strip() == prev_turn_answer.strip()`

### 计分规则

```python
def metric_16(turn: TurnRecord, oc: TurnOutcome, cfg: TurnRewardConfig) -> float:
    """Metric 16 — cross-turn duplicate submit penalty.
    
    如果当前 turn 和上一轮 turn 都提交了相同的答案，则惩罚当前 turn。
    这防止模型在多个 turn 中重复提交相同答案而不进行实质性思考。
    
    评分规则：
      - 无上一轮 turn 或上一轮无 submit -> 1.0 (无惩罚)
      - 当前 turn 无 submit -> 1.0 (无惩罚)
      - 两轮答案不同 -> 1.0 (无惩罚，鼓励改进)
      - 两轮答案相同 -> 1.0 - cfg.cross_turn_dup_penalty (惩罚)
    """
    # 条件1：当前 turn 必须有 submit
    if oc.submit_count < 1:
        return 1.0
    
    # 条件2：必须存在上一轮 turn 的答案
    if turn.prev_turn_answer is None:
        return 1.0
    
    # 条件3：比较两轮答案
    current_answer = oc.answer.strip() if oc.answer else ""
    prev_answer = turn.prev_turn_answer.strip()
    
    if current_answer == prev_answer:
        # 惩罚重复提交
        return _clamp(1.0 - cfg.cross_turn_dup_penalty, 0.0, 1.0)
    
    # 答案不同，无惩罚（鼓励改进）
    return 1.0
```

### 配置参数

在 `TurnRewardConfig` 中添加：

```python
@dataclass
class TurnRewardConfig:
    # ... 现有参数 ...
    
    w16: float = 0.05  # cross-turn duplicate submit penalty weight
    cross_turn_dup_penalty: float = 0.20  # penalty magnitude
```

### 在 `TurnRewardBreakdown` 中添加

```python
@dataclass
class TurnRewardBreakdown:
    # ... 现有 metrics ...
    metric_16: float = 0.0
    
    @property
    def metrics(self) -> List[float]:
        return [
            self.metric_1, self.metric_2, self.metric_3, self.metric_4,
            self.metric_5, self.metric_6, self.metric_7, self.metric_8,
            self.metric_9, self.metric_10, self.metric_11, self.metric_12,
            self.metric_13, self.metric_14, self.metric_15, self.metric_16,
        ]
    
    @property
    def weights(self) -> List[float]:
        c = DEFAULT_TURN_REWARD_CONFIG
        return [c.w1, c.w2, c.w3, c.w4, c.w5, c.w6,
                c.w7, c.w8, c.w9, c.w10, c.w11, c.w12, c.w13, c.w14, c.w15, c.w16]
```

## 与现有 metrics 的关系

| Metric | 惩罚范围 | 与 Metric 16 的区别 |
|--------|---------|-------------------|
| Metric 2a | 同一 turn 内多次 submit | Metric 16 跨 turn |
| Metric 2b | 同一 step 内并行重复 tool call | Metric 16 跨 turn |
| Metric 15 | 同一 turn 内跨 step 重复 tool call | Metric 16 跨 turn |
| **Metric 16** | **跨 turn 重复提交相同答案** | **唯一跨 turn 的惩罚** |

## 实现注意事项

### 1. `prev_turn_answer` 的传递

需要在 `HistoryManager` 或 `ScribeRunner` 中维护一个 `prev_turn_answer` 状态：

```python
class HistoryManager:
    def __init__(self, ...):
        # ... 现有代码 ...
        self._prev_turn_answer: Optional[str] = None
    
    def add_turn(self, turn: TurnRecord):
        # ... 现有代码 ...
        # 在创建新 turn 前，记录上一轮的答案
        if self._turns:
            last_turn = self._turns[-1]
            # 从 last_turn 中提取最终 submit 的答案
            self._prev_turn_answer = self._extract_final_answer(last_turn)
    
    def _extract_final_answer(self, turn: TurnRecord) -> Optional[str]:
        """从 turn 的 step_records 中提取最终 submit 的答案。"""
        for rec in reversed(turn.step_records):
            for b in rec.output_blocks:
                if b.type == ScribeBlockType.TOOL_CALL:
                    parsed = b.parsed
                    if isinstance(parsed, dict) and parsed.get("name") == "submit":
                        args = parsed.get("arguments", {})
                        return args.get("answer")
        return None
```

### 2. `TurnRecord` 的修改

```python
@dataclass
class TurnRecord:
    step_records: List[StepRecord]
    reward: float = 0.0
    feedback: Optional[str] = None
    prev_turn_answer: Optional[str] = None  # 新增
    reward_breakdown: Optional[Any] = None
```

### 3. `compute_turn_reward` 的修改

```python
async def compute_turn_reward(
    turn: TurnRecord,
    config: Optional[TurnRewardConfig] = None,
    *,
    outcome: TurnOutcome,
    judge: Optional[Judge] = None,
    token_counter: Any = None,
) -> TurnRewardBreakdown:
    # ... 现有代码 ...
    
    bd = TurnRewardBreakdown(
        metric_1=metric_1(turn, outcome, cfg),
        # ... 其他 metrics ...
        metric_15=metric_15(turn, outcome, cfg),
        metric_16=metric_16(turn, outcome, cfg),  # 新增
        judge_used=judge_used,
    )
    
    ws = [cfg.w1, cfg.w2, cfg.w3, cfg.w4, cfg.w5, cfg.w6,
          cfg.w7, cfg.w8, cfg.w9, cfg.w10, cfg.w11, cfg.w12, cfg.w13, cfg.w14, cfg.w15, cfg.w16]
    total = sum(w * v for w, v in zip(ws, bd.metrics))
    bd.total = _clamp(total, cfg.total_low, cfg.total_high)
    return bd
```

## 预期效果

### 惩罚前（无 Metric 16）

```
Turn 1: 思考 → submit(答案=42) → feedback("答案错误，请再试")
Turn 2: 思考 → submit(答案=42) → feedback("答案仍然错误")
Turn 3: 思考 → submit(答案=42) → feedback("你已经重复提交3次")

Reward: 每轮都获得较高的 format/summary 分数，但没有改进
```

### 惩罚后（有 Metric 16）

```
Turn 1: 思考 → submit(答案=42) → feedback("答案错误，请再试")
Turn 2: 思考 → submit(答案=42) → metric_16=0.8 (惩罚-0.2)
Turn 3: 思考 → submit(答案=42) → metric_16=0.8 (惩罚-0.2)

Reward: 重复提交的 turn 总 reward 降低，鼓励模型改进答案
```

## 与 Fix-A 的整合

在 Fix-A（权重调整）中，建议：

```python
w1: float = 0.55   # answer correctness (主导)
w16: float = 0.05  # cross-turn duplicate submit penalty
# ... 其他 weights ...
```

Metric 16 的权重不宜过高（0.05 足够），因为它是一个"辅助性"的惩罚信号，不应该主导 reward。但它的存在可以有效防止模型"躺平"重复提交。

## 总结

Metric 16 是一个**跨 turn 的重复提交惩罚**，旨在：
1. 防止模型在多个 turn 中重复提交相同答案
2. 鼓励模型利用 feedback 改进答案
3. 减少"format gaming"的空间（因为重复提交相同答案无法通过 format 优化来掩盖）

这是 SCRIBE 15+1 metrics 体系的重要补充，特别适用于多 turn 交互场景。
