你知道有没有那种做agentic rl那种科研小实验特别特别方便的数据集或者环境啊？很容易验证想法，不是很重量级而且任务对算力还友好

我搜索一下最近的 lightweight agentic RL benchmark 和环境进展，给你最实用的建议。

有的！做 agentic RL 的小实验，最头疼的其实是**环境本身太重、reset 和 step 太慢、或者 action space 太大导致 rollout 成本爆炸**。下面这几类环境/数据集是社区里验证想法最常用、算力最友好的：

---

## 1. 文本类环境（最轻量，LLM agent 首选）

这类环境完全基于文本状态转移，没有图像渲染，step 极快，单卡甚至 CPU 就能跑 RL。

| 环境 | 特点 | 适合验证什么 |
|------|------|-------------|
| **ALFWorld** | 基于 TextWorld 的室内导航/操作模拟（厨房、客厅等），observation 和 action 都是文本 | 多步规划、工具使用、ReAct / Reflexion 风格的 agentic RL |
| **TextWorld** | 纯文本冒险游戏，轻量可定制 | 探索、课程学习、知识图谱推理 |
| **WebShop** | 电商购物模拟，多轮对话+搜索+选择 | 多轮决策、搜索策略、reward shaping |

**ALFWorld** 尤其推荐：它几乎是目前 LLM agent + RL 的 "hello world"，很多 agentic RL 论文（包括 Microsoft 的 Agent² RL-Bench）都用它做基线。装个 `alfworld` pip 包就能跑，一个 episode 几毫秒。

---

## 2. 离散网格世界（传统 RL agent 也适用）

如果你做的是非 LLM agent（或者用小 LM + 策略网络），grid world 是最经典的小实验沙盒。

| 环境 | 特点 |
|------|------|
| **MiniGrid** | Farama 维护的 2D 网格世界，observation 可设为局部视野或完整网格，action 只有 3-7 个离散动作 |
| **BabyAI** | 基于 MiniGrid 加入语言指令（"go to the red ball"），专门测试 agent 遵循指令和推理能力 |

**优势**：step 速度极快，reward 稀疏可配置，很容易构造需要**多步推理、子目标分解**的任务。用 JAX 重写后（如 `jax-minigrid`）还能在 GPU 上并行跑成千上万个环境，单卡一天能跑几亿步。

---

## 3. 轻量级工具使用 / Function Calling

如果你验证的是"agent 调用工具"的 RL 策略（比如搜索、计算器、代码解释器），**没必要上真实的复杂 API**：

| 环境 | 说明 |
|------|------|
| **BFCL (Berkeley Function Calling Leaderboard)** | 有 single-turn 和 multi-turn 版本，可以包装成 Gym 环境做 RL |
| **τ²-bench** | 航空/零售/电信的客服多轮对话，状态转移确定，适合验证带工具调用的 multi-turn RL |
| **Simple Calculator / WebQA** | 自己搭的极简环境（比如 2-3 个工具，固定状态机），验证 reward model 或 credit assignment 想法最快 |

**小窍门**：很多 idea 其实不需要完整的 MCP 或真实 API。自己写个 50 行 Python 的 `gymnasium.Env`，状态转移写死，observation 用文本或 JSON，反而最能快速验证核心算法。

---

## 4. 快速并行框架（降低算力门槛）

环境轻量还不够，agentic RL 往往需要大量 rollout。配合这些框架：

- **JAX-based envs**：如 `jax-minigrid`、`brax`（虽然 brax 偏连续控制，但思路类似），可以把环境 step 完全编译到 GPU，并行度拉满。
- **AgentFly** / **veRL**：近半年几个 agentic RL 训练框架（如论文里提到的）支持 multi-turn 轨迹的异步 rollout，减少 GPU 空等。
- **RD-Agent / AWorld-RL**：开源的 agentic RL 训练套件，内置了环境封装和 reward 函数模板。

---

## 5. 一句话建议

| 你想验证的方向 | 推荐环境 | 硬件要求 |
|-------------|---------|---------|
| LLM agent 多轮规划 / tool use | **ALFWorld** | 单卡 24G 可跑 7B model |
| 小模型 + 视觉/指令推理 | **MiniGrid / BabyAI** | CPU/GPU 均可，极快 |
| 纯文本探索 + curriculum | **TextWorld** | CPU 即可 |
| Tool-use / API calling RL | **τ²-bench** 或 **自研 50 行环境** | 单卡 |
| 超快速并行验证 | **JAX Minigrid** | 单卡 GPU 并行 10k+ env |

**最实用的组合**：先用 **MiniGrid**（或 BabyAI）验证算法能不能在简单离散空间里学会多步推理；再迁移到 **ALFWorld** 或自研文本环境上测 LLM agent。如果这两个都能 work，再考虑上 WebShop、MCP-Universe 或真实工具链。

需要我帮你找某个具体环境的安装/封装示例，或者写一个极简的自定义 agentic RL 环境模板吗？

今天已经2026年6月27日了ALFWorld 如何做agentic rl，ALFWORLD怎么使用，这一块你得继续调研啊

我来深度调研 ALFWorld 的 agentic RL 用法，从环境接口到实际训练代码。

我已经完成了深度调研。下面是关于 **ALFWorld 如何做 Agentic RL** 的完整报告，从环境原理到实际训练代码。

---

# ALFWorld Agentic RL 深度调研报告

## 一、ALFWorld 是什么（核心机制）

ALFWorld 是一个**基于 TextWorld 的文本化具身环境**，把 ALFRED 数据集的视觉导航/操作任务翻译成了纯文本交互。它只依赖文本状态转移，**不需要渲染、不需要 GPU 跑仿真**，单 CPU 就能秒级运行一个 episode。

### 六种任务类型
| ID | 任务类型 | 说明 |
|---|---|---|
| 1 | `pick_and_place_simple` | 拿起物品放到指定容器 |
| 2 | `look_at_obj_in_light` | 在光源下查看物品 |
| 3 | `pick_clean_then_place_in_recep` | 清洗后放置 |
| 4 | `pick_heat_then_place_in_recep` | 加热后放置 |
| 5 | `pick_cool_then_place_in_recep` | 冷却后放置 |
| 6 | `pick_two_obj_and_place` | 拿起两个物品放置 |

### 核心环境接口
- **Observation**: 文本描述（如 `You are in the kitchen. You see a fridge, a countertop...`）
- **Action**: 文本命令（如 `go to kitchen`, `pick up book`, `put book in drawer`）
- **Admissible Commands**: 环境每步提供当前可用的合法动作列表（`info['admissible_commands']`），这是关键约束
- **Reward**: 极度稀疏——`won=1` 时给 `10.0`，否则 `0`
- **Max Steps**: 默认 50 步截断
- **Done**: 任务成功/失败/超时

---

## 二、原生 ALFWorld 如何使用（底层 API）

### 安装
```bash
conda create -n alfworld python=3.9
pip install alfworld[full]          # 文本版；加 [vis] 才有 AI2-THOR 视觉版
export ALFWORLD_DATA=~/.cache/alfworld
alfworld-download                    # 下载 PDDL + game files
```

### 最简交互代码
```python
import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

# 加载配置
config = generic.load_config()  # 或传入 configs/base_config.yaml
env_type = config['env']['type']  # 'AlfredTWEnv' 或 'AlfredThorEnv'

# 初始化环境
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# 交互循环
obs, info = env.reset()
while True:
    # 从 info 中获取合法动作
    admissible_commands = list(info['admissible_commands'])
    random_actions = [np.random.choice(admissible_commands[0])]

    obs, scores, dones, infos = env.step(random_actions)
    print("Action:", random_actions[0], "Obs:", obs[0])

    if dones[0]:
        break
```

### 关键返回字段
```python
obs           # List[str] 文本观察
scores        # List[float] 当前 reward（基本是 0，最后一步可能是 10）
dones         # List[bool] 是否结束
infos         # List[Dict] 包含：
              #   'admissible_commands' -> 合法动作列表
              #   'won' -> 0/1 是否成功
              #   'extra.gamefile' -> 当前游戏文件路径
              #   'goal_condition_success_rate' -> 子目标完成率
```

---

## 三、如何把 ALFWorld 接入 LLM Agent RL 训练

原生 ALFWorld 的 API 不是为 LLM 设计的。要做 Agentic RL，需要**三层封装**：

### 第一层：环境并行封装（Ray Workers）

verl-agent 的做法是用 **Ray Remote Actor** 把每个 ALFWorld 实例封装成独立 worker，支持并行跑上百个环境：

```python
@ray.remote
class AlfworldWorker:
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)
        self.env.seed(seed)

    def step(self, action):
        actions = [action]
        obs, scores, dones, infos = self.env.step(actions)
        return obs, scores, dones, infos

    def reset(self):
        obs, infos = self.env.reset()
        return obs, infos
```

然后外层 `AlfworldEnvs` 管理一组 workers，实现 `reset()` 和 `step()` 的批量调用。

### 第二层：文本 Observation 构建（Prompt Engineering）

这是最关键的一层。LLM 不直接看原始 observation，而是看包装后的 prompt：

**Prompt 模板（verl-agent 实际使用的）**：
```python
"""
You are an expert agent operating in the ALFRED Embodied Environment. 
Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s). 
Below are the most recent {history_length} observations and actions: {action_history}

You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. 
This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action 
and present it within <action> </action> tags.
"""
```

**关键点**：
- 必须注入 `admissible_actions`，否则 LLM 会生成非法动作
- 必须包含 `task_description`（从首帧 observation 中提取）
- 支持 history（最近 N 步的 obs + action），但也可以无历史
- 要求 LLM 输出结构化格式：`<think>推理</think><action>动作</action>`

### 第三层：Action Projection（解析 LLM 输出）

LLM 输出是自由文本，需要解析并映射到环境合法动作：

```python
def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    valids = [0] * len(actions)
    
    for i in range(len(actions)):
        # 1. 提取 <action>...</action> 之间的内容
        start_idx = actions[i].find("<action>")
        end_idx = actions[i].find("</action>")
        
        if start_idx == -1 or end_idx == -1:
            actions[i] = actions[i][-30:]  # 非法，截取最后 30 字符
            continue
            
        extracted = actions[i][start_idx + len("<action>"):end_idx].strip().lower()
        actions[i] = extracted
        valids[i] = 1
        
        # 2. 检查是否包含 <think> 标签
        if "<think>" not in actions[i] or "</think>" not in actions[i]:
            valids[i] = 0
            
        # 3. 检查中文（防止模型乱输出）
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0
    
    return actions, valids
```

**注意**：verl-agent 实际不检查动作是否在 `admissible_commands` 里，而是靠环境本身在 `step()` 时拒绝非法动作。但 `valids` 标记用于后续给 **invalid action penalty**（默认 `-0.1`）。

---

## 四、Reward 设计

ALFWorld 原生 reward 极度稀疏（只有最后一步有信号），做 RL 时需要额外设计：

### 实际使用的 Reward（verl-agent / GiGPO 论文配置）

| 信号 | 值 | 说明 |
|---|---|---|
| 任务成功 | `+10.0` | `info['won'] == 1` |
| 任务失败 | `0` | `info['won'] == 0` |
| 非法动作 | `-0.1` | `use_invalid_action_penalty=True` |
| 子目标完成 | `+goal_condition_success_rate` | 仅 Thor 多模态版使用 |

### 关于 Dense Reward 的实验结论

根据 **Meow-Tea-Taro** 论文（2025 年 10 月，专门研究 ALFWorld 的 multi-turn RL）：
- 给每步加 dense reward（如基于 expert plan 的进度 reward）确实**加速收敛**
- 但 **dense reward 对不同 RL 算法影响差异极大**：PPO 和 GRPO 对 dense reward 的稳定性很敏感
- **最优做法**：先用 SFT warm-up，再上线 sparse reward 的 RL

---

## 五、RL 算法与超参配置

### 推荐训练流程（论文验证过的）

| 阶段 | 配置 | 说明 |
|---|---|---|
| **SFT Warm-up** | 1 epoch, 300 条数据, lr=1e-6 | 必须做！Base model 在 ALFWorld 上基本是 0% |
| **RL 训练** | 90-150 epochs | PPO/GRPO/RLOO/GiGPO |

### 关键超参（GiGPO / verl-agent 实际配置）

```yaml
# 模型
actor_rollout_ref.model.path: Qwen/Qwen2.5-1.5B-Instruct  # 或 7B
actor_rollout_ref.model.lora_rank: 64                      # 可选 LoRA
actor_rollout_ref.model.lora_alpha: 64

# 数据
data.max_prompt_length: 2048
data.max_response_length: 512

# 环境
env.env_name: alfworld/AlfredTWEnv
env.max_steps: 50
env.rollout.n: 8                     # GRPO group size

# RL
actor_rollout_ref.actor.optim.lr: 1e-6
actor_rollout_ref.actor.use_kl_loss: True
actor_rollout_ref.actor.kl_loss_coef: 0.01
algorithm.adv_estimator: grpo      # 或 gigpo, ppo, rloo

# 训练
trainer.n_gpus_per_node: 2
trainer.total_epochs: 150
trainer.test_freq: 5
```

### 不同算法在 ALFWorld 上的效果（GiGPO 论文）

使用 **Qwen2.5-7B-Instruct**，从 SFT 后继续 RL：

| 算法 | 平均成功率 |
|---|---|
| Base Prompting | ~14.8% |
| ReAct | ~31.2% |
| Reflexion | ~42.7% |
| **PPO** | ~80.4% |
| **RLOO** | ~75.5% |
| **GRPO** | ~77.6% |
| **GiGPO** | **~90.8%** |

### 单卡 3090 24GB 可行性

- **1.5B 模型 + GRPO + 2 GPU**：完全可行（verl-agent 官方 example）
- **7B 模型 + LoRA + 2 GPU**：可行，需调 `rollout.gpu_memory_utilization=0.6` 和 `group_size=8`
- 单卡 3090 24GB 跑 7B 全参数 GRPO：**不够**，必须用 LoRA 或 4-bit

---

## 六、可直接上手的最简代码

### 方案 A：用 verl-agent（最推荐，开箱即用）

```bash
# 1. 安装 verl-agent（包含 ALFWorld 环境）
git clone https://github.com/langfengq/verl-agent.git
cd verl-agent
pip install -e .

# 2. 下载 ALFWorld 数据
export ALFWORLD_DATA=~/.cache/alfworld
alfworld-download

# 3. 准备 dummy dataset（只用来说明 modality）
python -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128

# 4. 跑 GRPO（1.5B 模型，2 GPU）
bash examples/grpo_trainer/run_alfworld.sh
```

### 方案 B：自己写最小 Gym 封装（适合验证新算法）

如果你不想用 verl-agent 这种重型框架，可以只用 ALFWorld 原生 API + 一个薄 wrapper：

```python
import gymnasium as gym
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

class ALFWorldGymEnv(gym.Env):
    def __init__(self, config_path="configs/base_config.yaml"):
        config = generic.load_config(config_path)
        self.base_env = get_environment("AlfredTWEnv")(config, train_eval='train')
        self.env = self.base_env.init_env(batch_size=1)
        self.task_desc = None
        
    def reset(self, seed=None):
        obs, info = self.env.reset()
        self.task_desc = self._extract_task(obs[0])
        self.admissible = info['admissible_commands'][0]
        return self._build_obs(obs[0], init=True), info
        
    def step(self, action_text: str):
        # action_text: LLM 生成的原始字符串
        obs, scores, dones, infos = self.env.step([action_text])
        reward = 10.0 * infos[0]['won']  # sparse reward
        done = dones[0]
        return self._build_obs(obs[0], init=False), reward, done, False, infos[0]
        
    def _build_obs(self, raw_obs, init=False):
        if init:
            return f"Task: {self.task_desc}\nObs: {raw_obs}\nValid actions: {self.admissible}"
        return f"Obs: {raw_obs}\nValid actions: {self.admissible}"
    
    def _extract_task(self, obs):
        # 从首帧 obs 中提取 "Your task is to: ..."
        idx = obs.find("Your task is to: ")
        return obs[idx + len("Your task is to: "):].strip() if idx != -1 else ""
```

---

## 七、关键踩坑点

| 坑 | 解决方案 |
|---|---|
| Base model 在 ALFWorld 上 0% 准确率 | **必须先做 SFT warm-up**（约 300 条轨迹） |
| LLM 生成非法动作 | 在 prompt 中明确列出 `admissible_actions`，加 invalid action penalty |
| 中文乱输出 | projection 中检测并过滤中文，设 valid=0 |
| 忘记 `<think>` 或 `<action>` 标签 | projection 中检测标签存在性，否则 mark invalid |
| 多步历史导致 prompt 超长 | 限制 `history_length`（如 5），超长时截断 |
| 显存爆 | 7B 模型用 LoRA / QLoRA，vLLM `gpu_memory_utilization` 调低 |
| 训练不稳定（GRPO） | 调小 `group_size`（4-8），增大 `kl_loss_coef` |

---

## 八、总结：ALFWorld 做 Agentic RL 的“标准路径”

```
1. 安装 ALFWorld + 下载数据
   ↓
2. 写/抄 EnvironmentManager（封装 ALFWorld → LLM prompt）
   ↓
3. 准备 300 条 SFT 数据（可用专家轨迹或 ReAct 生成）
   ↓
4. SFT 1 epoch → 让模型学会基本动作格式
   ↓
5. 接入 RL 框架（verl-agent / TRL / 自研）
   ↓
6. 训练 90-150 epochs，观察 success_rate
   ↓
7. 调 reward / group_size / history_length / LR
```

如果你现在就想动手，**最快路径**是直接把 [verl-agent](https://github.com/langfengq/verl-agent) 跑起来，它的 `examples/grpo_trainer/run_alfworld.sh` 是已经验证过的完整 pipeline。如果你想自己从 0 搭一个轻量版，可以参考上面的 wrapper 代码。

需要我帮你写一份针对 3090 24GB 单卡的 LoRA GRPO 配置，或者生成一份 ALFWorld SFT 数据准备脚本吗？

