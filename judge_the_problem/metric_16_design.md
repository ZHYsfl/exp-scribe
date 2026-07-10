# Metric 16: Cross-Turn Duplicate Submit Penalty

## 设计动机

在多 turn 的 SCRIBE 交互中，模型可能在多个 turn 中重复提交相同的答案而不进行实质性思考。这种行为：
- 浪费了 turn 预算
- 没有利用 feedback 改进答案
- 是一种"偷懒 / 躺平"策略

前 15 个 metric 全部是**轮内**信号（2a 同轮多次 submit、2b 同一 step 并行重复、15 同轮跨 step 重复 tool call），它们看不到"跨 turn 重复提交同一答案"这种停滞模式。Metric 16 是唯一的**跨 turn** 惩罚，专门堵这个漏洞。

## 设计规格

### 触发条件

同时满足以下三个条件时触发惩罚：
1. **当前 turn 有 submit**：`oc.submit_count >= 1`
2. **存在上一轮 turn，且上轮 turn 提交过答案**：`turn.prev_turn_answer is not None`
3. **两轮答案相同**（去空白后）：`(oc.answer or "").strip() == turn.prev_turn_answer.strip()`

> 故意把范围限定在"重复 submit 同一答案"，而不是泛化的"跨 turn 重复 tool call"。跨 turn 重跑某个工具常常是合法的（env 状态可能已变），但跨 turn 重复提交同一个最终答案几乎从不是好事。

### 计分规则

- 无上一轮 turn / 上一轮无 submit -> `1.0`（无惩罚）
- 当前 turn 无 submit -> `1.0`（无惩罚）
- 两轮答案不同 -> `1.0`（**这正是我们要奖励的"利用 feedback 改进"行为**）
- 两轮答案相同 -> `1.0 - cfg.cross_turn_dup_penalty`（惩罚，clamp 到 [0,1]）

### 实现代码（`Scribe/scribe_gym/rewards.py`）

```python
def metric_16(turn: TurnRecord, oc: TurnOutcome, cfg: TurnRewardConfig) -> float:
    """Metric 16 - cross-turn duplicate submit penalty.

    README 16: penalize re-submitting the *same* answer already submitted in
    the previous turn. Distinct from 2a (intra-turn multi-submit), 2b
    (parallel dup in one message), and 15 (cross-step repeated tool call
    within one turn) - all of those are intra-turn. This is the only
    cross-turn penalty.

    prev_turn_answer is set on the TurnRecord by the agent at finalization
    time (reads the previous turn's final submit from history_manager), so
    this metric stays a pure function of (turn, outcome, config).
    """
    if oc.submit_count < 1:
        return 1.0
    if turn.prev_turn_answer is None:
        return 1.0
    current_answer = (oc.answer or "").strip()
    prev_answer = turn.prev_turn_answer.strip()
    if not current_answer or not prev_answer:
        return 1.0
    if current_answer == prev_answer:
        return _clamp(1.0 - cfg.cross_turn_dup_penalty, 0.0, 1.0)
    return 1.0
```

### 配置参数（`TurnRewardConfig`）

```python
@dataclass
class TurnRewardConfig:
    # ... 现有参数 ...
    w16: float = 0.05                  # cross-turn duplicate submit penalty
    cross_turn_dup_penalty: float = 0.20  # penalty magnitude
```

### 在 `TurnRewardBreakdown` 中

```python
@dataclass
class TurnRewardBreakdown:
    # ... metric_1 .. metric_15 ...
    metric_16: float = 0.0

    @property
    def metrics(self) -> List[float]:
        return [..., self.metric_15, self.metric_16]   # 长度 16

    @property
    def weights(self) -> List[float]:
        c = DEFAULT_TURN_REWARD_CONFIG
        return [..., c.w15, c.w16]
```

`compute_turn_reward` 中加入 `metric_16=metric_16(turn, outcome, cfg)`，权重列表末尾加 `c.w16`，`total = clamp(Σ wi·metrici, 0..1)`。

## `prev_turn_answer` 的传递（关键：避免 off-by-one）

⚠️ **时机很重要**：`prev_turn_answer` 必须在 `finalize_turn` 中、**计算 reward 之前、且当前 turn 加入 history_manager 之前**设置。此时 `history_manager.last_turn` 才是真正的"上一轮"。

### 1. `TurnRecord` 新增字段与属性（`turn_record.py`）

```python
@dataclass
class TurnRecord:
    step_records: List[StepRecord]
    reward: float = 0.0
    feedback: Optional[str] = None
    reward_breakdown: Optional[Any] = None
    prev_turn_answer: Optional[str] = None   # 新增：上一轮的最终 submit 答案

    @property
    def final_submit_answer(self) -> Optional[str]:
        """本 turn 最后一次 submit 的 answer（逆序扫描，最近一次胜出，
        与 env 的 _answer 语义一致）。用于给下一轮 seed prev_turn_answer。"""
        for rec in reversed(self.step_records):
            for b in reversed(rec.output_blocks):
                if b.type != ScribeBlockType.TOOL_CALL:
                    continue
                parsed = b.parsed
                if isinstance(parsed, dict) and parsed.get("name") == "submit":
                    args = parsed.get("arguments", {})
                    if isinstance(args, dict) and args.get("answer") is not None:
                        return str(args["answer"])
        return None
```

### 2. `HistoryManager` 新增 `last_turn` 属性（`history_manager.py`）

```python
@property
def last_turn(self) -> Optional[TurnRecord]:
    """最近一次加入的 turn（add 之前读取即为"上一轮"）。"""
    return self._turns[-1] if self._turns else None
```

> 不在 `add_turn` 里维护 `_prev_turn_answer`：那样会把"上上轮"的答案错配给当前轮（off-by-one）。统一在 `finalize_turn` 里按需读取。

### 3. `finalize_turn` 中 seed（`openai_bridge.py`）

```python
async def finalize_turn(self, feedback):
    if self.history_manager is None:
        return None
    record = TurnRecord(step_records=list(self._turn_step_records),
                        reward=0.0, feedback=feedback)
    # 当前 turn 还没 add，所以 last_turn 是真正的上一轮
    prev = self.history_manager.last_turn
    record.prev_turn_answer = prev.final_submit_answer if prev is not None else None
    record.reward = await self._compute_turn_reward(record)   # metric_16 在这里读到
    self.history_manager.add_turn(record)
    ...
```

## 与现有 metrics 的关系

| Metric | 惩罚范围 | 与 Metric 16 的区别 |
|--------|---------|-------------------|
| Metric 2a | 同一 turn 内多次 submit | Metric 16 跨 turn |
| Metric 2b | 同一 step 内并行重复 tool call | Metric 16 跨 turn |
| Metric 15 | 同一 turn 内跨 step 重复 tool call | Metric 16 跨 turn |
| **Metric 16** | **跨 turn 重复提交相同答案** | **唯一跨 turn 的惩罚** |

## 预期效果

```
Turn 1: 思考 -> submit(42) -> feedback("答案错误，请再试")
Turn 2: 思考 -> submit(42) -> metric_16 = 1 - 0.20 = 0.80（惩罚）
Turn 3: 思考 -> submit(42) -> metric_16 = 0.80（惩罚）

Turn 4: 思考 -> submit(43) -> metric_16 = 1.0（答案变了，无惩罚 = 奖励改进）
```

重复提交的 turn 总 reward 降低（w16=0.05 贡献 -0.01），鼓励模型利用 feedback 改进答案。

## 与权重调整（Fix-A）的整合

实际采用的新权重（见 README / `TurnRewardConfig`）：

```python
w1:  float = 0.52   # answer correctness（主导，但不极端）
w16: float = 0.05   # cross-turn duplicate submit penalty
w6:  float = 0.02   # token reuse（象征性保留）
w11: float = 0.01   # compression（象征性保留）
# ... 详见 TurnRewardConfig，Σ wi = 1.0
```

Metric 16 的权重 0.05 不宜过高：它是一个"辅助性"惩罚信号，不应主导 reward，但足以防止"躺平"重复提交。0.05 与 w4(format, 0.10)、w2(tool, 0.08) 同量级，是一个有存在感但不喧宾夺主的守门信号。

## 单元测试（`Scribe/tests/test_rewards.py`）

覆盖：无上一轮 -> 1.0；同答案 -> 惩罚；不同答案 -> 1.0；无 submit -> 1.0；空白忽略 -> 惩罚；`final_submit_answer` 取最后一次 submit；无 submit 时返回 None。

## 总结

Metric 16 是一个**跨 turn 的重复提交惩罚**，旨在：
1. 防止模型在多个 turn 中重复提交相同答案
2. 鼓励模型利用 feedback 改进答案（答案变了就给 1.0）
3. 减少"format gaming"空间（重复提交相同答案无法靠 format 优化掩盖）

这是 SCRIBE 16-metric 体系的重要补充，特别适用于多 turn 交互场景。
