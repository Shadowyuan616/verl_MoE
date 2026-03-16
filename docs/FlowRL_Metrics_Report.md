# FlowRL 训练指标详细技术报告

## 概览

本报告基于 `verl` 框架中 FlowRL 算法训练日志，针对 **Qwen3-30B-A3B (MoE)** 模型的 RL 训练指标进行深入解读。报告涵盖以下几类指标：

1. **FlowRL 核心 Actor 指标** (`actor/*`)
2. **Rollout Correction 离策略指标** (`rollout_corr/*`)
3. **训练调试指标** (`training/*`)
4. **Critic/Reward 相关指标** (`critic/*`)
5. **性能与时序指标** (`perf/*`, `timing_s/*`)
6. **序列长度与响应统计** (`response_length/*`, `prompt_length/*`)

---

## 一、FlowRL 核心 Actor 指标 (`actor/*`)

### 1.0 FlowRL 算法背景

**FlowRL** ([arXiv:2509.15207](https://arxiv.org/abs/2509.15207)) 是一种基于 **流平衡（Flow Balance）** 的强化学习方法，其核心思想是 **匹配奖励分布（Matching Reward Distributions）** 而非单纯最大化奖励。这一设计促进了多样化探索和可泛化的推理轨迹。

#### 核心目标函数

$$
\mathcal{L}_{\text{FlowRL}} = w \cdot \left( \log Z_{\phi}(x) + \frac{1}{|y|} \log \pi_{\theta}(y \mid x) - \beta \hat{r}(x, y) - \frac{1}{|y|} \log \pi_{\text{ref}}(y \mid x) \right)^2
$$

简化表示为：
$$
\mathcal{L}_{\text{FlowRL}} = w \cdot \delta^2
$$

其中 **残差项 $\delta$** 定义为：
$$
\delta = \log Z_{\phi}(x) + \bar{\log \pi}_{\theta}(y|x) - \beta \cdot \bar{r}(x,y) - \bar{\log \pi}_{\text{ref}}(y|x)
$$

#### 公式中各项的物理意义

| 符号 | 含义 | 对应指标 |
|------|------|----------|
| $\log Z_{\phi}(x)$ | **配分函数的对数**：由 `ProjZModule` 网络从 prompt 的 hidden states 预测，用于流平衡中的归一化常数 | `actor/log_z` |
| $\bar{\log \pi}_{\theta}(y\|x)$ | **当前策略的平均 log 概率**：$\frac{1}{\|y\|} \sum_t \log \pi_\theta(y_t\|y_{<t}, x)$ | `actor/log_prob` |
| $\beta$ | **奖励缩放系数**：论文中设为 15.0，控制奖励信号在损失中的权重 | 硬编码 `flowrl_beta_coef=15.0` |
| $\bar{r}(x,y)$ | **平均奖励**：序列级奖励的 token 平均 | `actor/log_reward` |
| $\bar{\log \pi}_{\text{ref}}(y\|x)$ | **参考策略的平均 log 概率**：初始 SFT 模型的输出概率 | `actor/ref_log_prob` |
| $w$ | **重要性权重**：$\text{clamp}(\exp(\sum_t \log \frac{\pi_\theta}{\pi_{\text{old}}}), \max=10)$，用于 off-policy 校正 | `actor/importance_weight` |

#### 与传统 RL 方法的区别

| 特性 | PPO/GRPO | FlowRL |
|------|----------|--------|
| **优化目标** | 最大化期望奖励 | 匹配目标奖励分布 |
| **损失形式** | Clipped surrogate loss | 轨迹平衡残差平方 $\delta^2$ |
| **Value/Critic** | 需要 Value Network | 使用 $\log Z$ 替代 |
| **探索性** | 依赖熵正则 | 内建分布匹配促进多样性 |

---

### 1.1 `actor/log_prob` — 当前策略平均 log 概率

| 属性 | 说明 |
|------|------|
| **定义** | 当前策略 $\pi_\theta$ 在 response tokens 上的 **平均** log 概率：$\bar{\log \pi}_\theta = \frac{1}{\|y\|} \sum_t \log \pi_\theta(y_t\|y_{<t}, x)$ |
| **物理含义** | 对应 FlowRL 目标函数中的 $\frac{1}{\|y\|} \log \pi_{\theta}(y \mid x)$ 项，衡量当前模型对生成序列的「流入流量」|
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L472](recipe/flowrl/flowrl_actor.py#L472) |

```python
avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)  # 用于 loss 计算
"actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),  # 用于 logging
```

| 正常趋势 | 异常信号 |
|---------|---------|
| 训练初期略负（~-0.3~-0.5），随训练可能略微增加或保持稳定 | 急剧下降（趋向-inf）：策略崩溃，模型概率坍缩 |
| 与 `actor/ref_log_prob` 差值反映策略偏移程度 | 急剧上升（趋向0）：策略过拟合到特定模式 |

---

### 1.2 `actor/old_log_prob` — 旧策略 log 概率

| 属性 | 说明 |
|------|------|
| **定义** | Rollout 时的 actor 策略在 response tokens 上的平均 log 概率 |
| **物理含义** | 用于计算重要性权重 $w = \exp(\sum_t \log \frac{\pi_\theta}{\pi_{\text{old}}})$，衡量 on-policy vs off-policy 程度 |
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L473](recipe/flowrl/flowrl_actor.py#L473) |

| 正常趋势 | 异常信号 |
|---------|---------|
| 与 `actor/log_prob` 保持接近（差值小于 0.1） | 差距过大 (>0.5)：严重 off-policy，重要性权重可能爆炸 |
| 每个 step 开始时等于上一步的 `log_prob` | 差距为负且持续增大：策略在远离 rollout 时的分布 |

---

### 1.3 `actor/ref_log_prob` — 参考策略 log 概率

| 属性 | 说明 |
|------|------|
| **定义** | 参考策略（初始 SFT 模型）在 response tokens 上的平均 log 概率 |
| **物理含义** | 对应 FlowRL 目标函数中的 $\frac{1}{\|y\|} \log \pi_{\text{ref}}(y \mid x)$ 项，作为流平衡的「流出基准」|
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L474](recipe/flowrl/flowrl_actor.py#L474) |

**在 FlowRL 中的作用**：
- 提供一个稳定的参考点，防止策略偏离太远
- 当 $\log \pi_\theta \approx \log \pi_{\text{ref}}$ 时，残差 $\delta$ 简化为 $\log Z - \beta \cdot r$

| 正常趋势 | 异常信号 |
|---------|---------|
| 保持稳定（~-0.3~-0.4），因为 ref 模型参数冻结 | 值发生变化：ref 模型加载错误或被意外更新 |
| 不同样本间有正常波动 | 所有样本值完全相同：可能是计算 bug |

---

### 1.4 `actor/log_z` ⭐ FlowRL 核心 — 配分函数估计

| 属性 | 说明 |
|------|------|
| **定义** | 配分函数的对数估计 $\log Z_\phi(x)$，由 `ProjZModule`（3层 MLP）从 **prompt** 的 hidden states 预测 |
| **物理含义** | 类似于 PPO 中的 Value Function，但服务于流平衡而非优势估计。$Z(x)$ 表示给定 prompt $x$ 时，所有可能 response 的「归一化常数」|
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L269-L276](recipe/flowrl/flowrl_actor.py#L269-L276) |

**计算流程**：
```python
# 1. 获取 prompt 部分的最后一层 hidden states
prompts_last_hidden = full_last_hidden[:, : -response_length - 1]

# 2. 对 prompt tokens 做 masked mean pooling
avg_hidden = verl_F.masked_mean(prompts_last_hidden, prompt_attention_mask.unsqueeze(-1), axis=1)

# 3. 通过 ProjZModule 预测 log Z
log_z = self.actor_module.proj_z(avg_hidden)  # Shape: (B, 1)
```

**ProjZModule 结构**：
```python
class ProjZModule(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 3, dropout: float = 0.1):
        # 3层 MLP: Linear -> GELU -> LayerNorm -> Dropout -> ... -> Linear(hidden, 1)
```

**在 FlowRL 损失中的角色**：
$$
\delta = \underbrace{\log Z_\phi(x)}_{\text{可学习}} + \bar{\log \pi}_\theta - \beta \cdot \bar{r} - \bar{\log \pi}_{\text{ref}}
$$

$\log Z$ 学习的目标是使残差 $\delta \to 0$，即满足流平衡条件。

| 正常趋势 | 异常信号 |
|---------|---------|
| 训练过程中缓慢调整，随 reward 分布变化而变化 | 爆炸（趋向 ±inf）：`ProjZModule` 训练不稳定，考虑降低 lr |
| 绝对值适中（通常 ~-0.1~0.1，取决于 reward scale） | 恒为 0 或不变：`proj_z` 梯度可能被截断，检查是否参与反向传播 |
| 高 reward prompt 对应较大的 $\log Z$ | 与 reward 完全无关：模型未学到有意义的 $Z$ 估计 |

---

### 1.5 `actor/log_reward` — 平均奖励（归一化 advantage）

| 属性 | 说明 |
|------|------|
| **定义** | FlowRL 目标函数中的 $\bar{r}(x,y)$ 项，实际上是 **GRPO 归一化后的 advantage**，而非原始 reward |
| **物理含义** | 衡量当前样本相对于同 prompt 组内其他样本的相对质量，在损失中被 $\beta=15$ 缩放 |
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L451](recipe/flowrl/flowrl_actor.py#L451) |

#### 完整计算链路

```
reward_fn (Rule/RM)
        │
        ▼
token_level_scores (batch_size, response_length)  ← sparse reward，仅 EOS 位置有值
        │
        ▼ (若 use_kl_in_reward=True)
token_level_rewards = token_level_scores - β_kl * KL(π_old || π_ref)
        │
        ▼
GRPO Advantage Estimator
        │
        ▼
advantages (batch_size, response_length)  ← 组内归一化，扩展到 token 级
        │
        ▼
FlowRL: reward = advantages
```

#### Step 1: 原始 Reward 计算

原始 reward 由 `reward_fn`（如 rule-based math verifier）计算：

```python
# verl/trainer/ppo/ray_trainer.py#L1497
batch.batch["token_level_scores"] = reward_tensor  # Shape: (B, response_length)
```

对于数学任务，`reward_tensor` 通常是 **sparse reward**：
- 仅在 EOS token 位置有值（0 或 1）
- 其余位置为 0

#### Step 2: KL 惩罚（可选）

如果配置 `algorithm.use_kl_in_reward=True`：

```python
# verl/trainer/ppo/ray_trainer.py#L154
token_level_rewards = token_level_scores - beta * kld
```

否则直接使用：
```python
# verl/trainer/ppo/ray_trainer.py#L1509
batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
```

#### Step 3: GRPO Advantage 计算（核心）

FlowRL 默认使用 `adv_estimator=grpo`，计算逻辑如下：

```python
# verl/trainer/ppo/core_algos.py#L264-L328
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (B, response_length)
    response_mask: torch.Tensor,        # (B, response_length)
    index: np.ndarray,                  # uid 数组，标识同一 prompt 的样本
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    # Step 3.1: 序列级 score = sum of token rewards
    scores = token_level_rewards.sum(dim=-1)  # (B,) 对于 sparse reward，等于最后 token 的 reward
    
    # Step 3.2: 按 prompt uid 分组，计算组内均值和标准差
    for idx in unique_uids:
        group_scores = [scores[i] for i in range(B) if index[i] == idx]
        id2mean[idx] = torch.mean(torch.stack(group_scores))
        id2std[idx] = torch.std(torch.stack(group_scores))
    
    # Step 3.3: 组内归一化
    for i in range(B):
        if norm_adv_by_std_in_grpo:  # 默认 True
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:  # Dr.GRPO 模式
            scores[i] = scores[i] - id2mean[index[i]]
    
    # Step 3.4: 扩展到 token 级别
    advantages = scores.unsqueeze(-1) * response_mask  # (B, response_length)
    
    return advantages, advantages  # advantages 同时作为 returns
```

**GRPO 归一化的物理意义**：
- 同一 prompt 生成 `n=8` 个 responses（通过 `rollout.n=8` 配置）
- 组内 z-score 归一化使得 advantage 在 [-3, 3] 范围内
- 这避免了不同 prompt 难度带来的 reward scale 差异

#### Step 4: FlowRL 使用 Advantage 作为 Reward

```python
# recipe/flowrl/flowrl_actor.py#L378
policy_loss, flowrl_metrics = self.compute_flowrl(
    ...
    reward=advantages,  # ⚠️ 注意：这里传入的是 GRPO advantages，不是原始 reward
    ...
)
```

在 `compute_flowrl` 中：

```python
# recipe/flowrl/flowrl_actor.py#L451
seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)  # (B,) 序列级平均

# Logging: batch 级平均
"actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item()
```

#### 指标解读

| 属性 | 说明 |
|------|------|
| **数值范围** | 由于 GRPO z-score 归一化，通常在 **[-3, 3]** 范围内 |
| **与原始 reward 的关系** | `actor/log_reward ≈ z-score(原始 reward)`，反映相对质量而非绝对质量 |
| **为何命名为 "log_reward"** | FlowRL 论文中 $R = \exp(\text{advantage})$，因此 $\log R = \text{advantage}$ |

#### 正常趋势与异常信号

| 正常趋势 | 异常信号 |
|---------|---------|
| 均值接近 0（GRPO 归一化效果） | 均值显著偏离 0：归一化失效或 `uid` 分组错误 |
| 随训练略微上升（策略在学习更好的回答） | 恒为 0：所有样本 reward 完全相同，无法区分 |
| 有正有负，反映组内样本质量差异 | 全为正或全为负：`rollout.n=1` 导致无法组内归一化 |
| 方差适中（~0.8-1.2） | 方差过大 (>2)：reward 分布极端不平衡 |

#### 与 FlowRL 损失的关系

在 FlowRL 损失中，`actor/log_reward` 被 $\beta=15$ 缩放：

$$
\delta = \log Z + \bar{\log \pi}_\theta - \underbrace{15 \cdot \texttt{actor/log\_reward}}_{\text{缩放后的 reward 信号}} - \bar{\log \pi}_{\text{ref}}
$$

因此：
- 当 `actor/log_reward > 0`（好样本），$\delta$ 倾向为负，需要增大 $\log \pi_\theta$ 或 $\log Z$ 来平衡
- 当 `actor/log_reward < 0`（差样本），$\delta$ 倾向为正，需要减小 $\log \pi_\theta$ 来平衡

这就是 FlowRL "匹配奖励分布" 的核心机制：通过流平衡，使得高 reward 样本的 $\pi_\theta$ 增大，低 reward 样本的 $\pi_\theta$ 减小。

---

### 1.6 `actor/final_loss` ⭐ FlowRL 核心 — 轨迹平衡损失

| 属性 | 说明 |
|------|------|
| **定义** | FlowRL 的核心优化目标：$\mathcal{L} = \mathbb{E}[w \cdot \delta^2]$ |
| **物理含义** | 带重要性加权的轨迹平衡残差平方损失，目标是使流入=流出达到平衡 |
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L446-L463](recipe/flowrl/flowrl_actor.py#L446-L463) |

**完整计算流程**：
```python
# 1. 计算序列级平均 log 概率
avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)       # (B,)
avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, response_mask, axis=1)  # (B,)
seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)       # (B,)

# 2. 计算 FlowRL 残差 δ（流平衡条件）
delta = log_z + avg_log_prob - self.flowrl_beta_coef * seq_log_reward - avg_ref_log_prob
#       ↑ 流入项 ↑                    ↑ 目标流量 ↑                        ↑ 流出基准 ↑

# 3. 计算重要性权重 w（off-policy 校正）
log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
imp_w = torch.clamp(torch.exp(log_w).detach(), max=10)

# 4. 加权损失
weighted_losses = imp_w * (delta ** 2)
avg_loss = torch.mean(weighted_losses)
```

其中 `flowrl_beta_coef = 15.0` 是硬编码的奖励缩放系数。

**流平衡的直觉**：
- 当 $\delta = 0$ 时，流平衡成立：$\log Z + \log \pi_\theta = \beta r + \log \pi_{\text{ref}}$
- 即：当前策略的「流入」= 奖励加权的「流出基准」
- 优化目标是最小化 $\delta^2$，使所有轨迹都趋近流平衡

| 正常趋势 | 异常信号 |
|---------|---------|
| 训练初期较高（~200-400），逐步下降 | 持续上升：学习率过大 / reward 设计问题 / $\beta$ 不匹配 |
| 最终稳定在较低水平（~50-100） | 震荡剧烈：importance weight 不稳定 |
| 下降速度与 reward 提升相关 | 趋向 0 太快：可能过拟合到少数模式 |

---

### 1.7 `actor/importance_weight` — 重要性采样权重

| 属性 | 说明 |
|------|------|
| **定义** | Off-policy 重要性权重：$w = \text{clamp}\left(\exp\left(\sum_t \log \frac{\pi_\theta(y_t)}{\pi_{\text{old}}(y_t)}\right), \max=10\right)$ |
| **物理含义** | 序列级 importance sampling 比率，用于校正因策略更新导致的分布偏移 |
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L459-L461](recipe/flowrl/flowrl_actor.py#L459-L461) |

```python
# 计算序列级 log 重要性比率
log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)  # (B,)

# 转换为比率并 clamp
imp_w_raw = torch.exp(log_w).detach()  # detach 防止通过 w 传梯度
imp_w = torch.clamp(imp_w_raw, max=10)  # 上限截断，防止方差爆炸
```

**设计要点**：
- 使用 `detach()` 阻止梯度通过 importance weight 回传
- Clamp 上限设为 10（而非 PPO 的 1±ε），允许更大的权重范围
- 没有下限 clamp，允许低于 1 的权重（即 $\pi_\theta < \pi_{\text{old}}$ 的情况）

| 正常趋势 | 异常信号 |
|---------|---------|
| 均值接近 1~3，表示策略变化适中 | 均值频繁 >5：策略更新过于激进，考虑减小 lr |
| 分布集中，方差较小 | 均值 <0.5：策略可能在「逃离」旧分布 |
| | 大量样本触及上限 10：严重 off-policy |

---

### 1.8 `actor/ppo_kl` — 策略间 KL 散度（旧策略）

| 属性 | 说明 |
|------|------|
| **定义** | 当前策略与旧策略（rollout 时）的近似 KL 散度 |
| **计算公式** | $\text{KL}(\pi_{\text{old}} \| \pi_\theta) \approx \mathbb{E}_{y \sim \pi_{\text{old}}}[\log \pi_{\text{old}}(y) - \log \pi_\theta(y)]$ |
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L464-L466](recipe/flowrl/flowrl_actor.py#L464-L466) |

```python
negative_approx_kl = log_prob - old_log_prob  # 逐 token
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)  # 取负并平均
```

**注意**：这里用的是 **反向 KL** 的近似（$\pi_{\text{old}}$ 在期望内），正值表示当前策略概率下降。

| 正常趋势 | 异常信号 |
|---------|---------|
| 保持较小（~0.0001-0.001），表示策略更新稳定 | 持续 >0.01：更新步长过大，可能不稳定 |
| 与 importance weight 变化相关 | 负值且绝对值大：策略在特定 token 上概率急剧上升 |

---

### 1.9 `actor/ref_kl` — 策略间 KL 散度（参考策略）

| 属性 | 说明 |
|------|------|
| **定义** | 当前策略与参考策略（初始 SFT）的近似 KL 散度 |
| **物理含义** | 衡量策略偏离初始模型的程度，是 FlowRL 中隐式正则化的监控指标 |
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L468-L470](recipe/flowrl/flowrl_actor.py#L468-L470) |

```python
approx_kl_ref = log_prob - ref_log_prob
ref_kl = verl_F.masked_mean(-approx_kl_ref, response_mask)
```

**与 FlowRL 目标的关系**：
- FlowRL 损失中包含 $\log \pi_\theta - \log \pi_{\text{ref}}$，隐式约束策略不要偏离太远
- 当 `ref_kl` 过大时，残差 $\delta$ 的 $\log \pi_\theta - \log \pi_{\text{ref}}$ 项会增大，增加损失

| 正常趋势 | 异常信号 |
|---------|---------|
| 随训练缓慢增加（策略在学习）| 绝对值 >1：策略与 ref 差异过大，可能 reward hacking |
| 正值表示当前策略比 ref 更「不确定」| 急剧变化：检查是否有异常梯度 |
| 负值表示当前策略比 ref 更「确定」| |

---

### 1.10 `actor/grad_norm` — 梯度范数

| 属性 | 说明 |
|------|------|
| **定义** | Actor 模型参数梯度的 L2 范数（在 grad clip 之后的值） |
| **物理含义** | 训练稳定性的关键指标，反映损失函数对参数的敏感程度 |
| **代码位置** | [recipe/flowrl/flowrl_actor.py#L428](recipe/flowrl/flowrl_actor.py#L428) |

```python
grad_norm = self._optimizer_step()  # 内部执行 clip_grad_norm_ 并返回范数
```

当前配置：`actor.grad_clip=1.0`

| 正常趋势 | 异常信号 |
|---------|---------|
| 保持在合理范围（30B 模型通常 ~100-500） | 持续增大或突然飙升：梯度爆炸，需降低 lr |
| 训练初期可能较大，后期稳定 | 趋向 0：梯度消失，检查 loss 是否有效 |
| 偶尔 spike 是正常的 | 持续在 clip 阈值附近：可能需要调整 clip 值 |

---

### 1.11 `actor/entropy` — 策略熵

| 属性 | 说明 |
|------|------|
| **定义** | 策略输出分布的熵：$H(\pi) = -\sum_a \pi(a) \log \pi(a)$ |
| **物理含义** | 衡量策略的「探索程度」，高熵=输出多样，低熵=输出确定 |
| **代码位置** | 在 FlowRL 中 `calculate_entropy=False`，但仍从 forward 输出记录 |

**在 FlowRL 中的特殊性**：
- FlowRL 不使用显式的熵正则化（`entropy_coeff=0`）
- 流平衡机制本身通过分布匹配促进多样性
- 熵指标仅作为监控，不参与 loss 计算

| 正常趋势 | 异常信号 |
|---------|---------|
| 训练中逐步下降（策略变得更确定） | 趋向 0：策略坍缩，所有输出趋同 |
| 保持适度水平（~0.2-0.5） | 持续过高：策略未收敛 |
| 下降速度与 reward 提升相关 | 突然下降：可能发生 mode collapse |

---

### 1.12 `actor/lr` — 学习率

| 属性 | 说明 |
|------|------|
| **定义** | 当前优化器的学习率 |
| **当前配置** | `lr=1e-6`, `warmup_style=constant`, `warmup_steps=10` |

根据配置，warmup 10 步后保持恒定 `lr=1e-6`。

---

### 1.13 FlowRL 指标关系图

```
                    ┌──────────────────────────────────────────┐
                    │           FlowRL Loss 计算流程            │
                    └──────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
   ┌─────────┐                    ┌─────────┐                    ┌─────────┐
   │ log_z   │                    │log_prob │                    │log_reward│
   │(ProjZ)  │                    │(Actor)  │                    │(Reward) │
   └────┬────┘                    └────┬────┘                    └────┬────┘
        │                              │                              │
        │         ┌────────────────────┼────────────────────┐        │
        │         │                    ▼                    │        │
        │         │           ┌──────────────┐              │        │
        │         │           │ref_log_prob  │              │        │
        │         │           │(Frozen Ref)  │              │        │
        │         │           └──────┬───────┘              │        │
        │         │                  │                      │        │
        ▼         ▼                  ▼                      ▼        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  δ = log_z + avg_log_prob - β*seq_log_reward - avg_ref_log_prob   │
   │                        (残差/Flow Balance)                         │
   └────────────────────────────────────┬───────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
              ┌───────────┐                          ┌────────────┐
              │old_log_prob│                          │importance  │
              │(Rollout)   │                          │  weight    │
              └─────┬──────┘                          └──────┬─────┘
                    │                                        │
                    └────────────────┬───────────────────────┘
                                     ▼
                    ┌────────────────────────────────────────┐
                    │     final_loss = mean(w * δ²)          │
                    └────────────────────────────────────────┘
```

---

## 二、Rollout Correction 离策略诊断指标 (`rollout_corr/*`)

这些指标用于诊断 **训练策略 (FSDP/FP32) 与 rollout 策略 (vLLM/BF16)** 之间的分布差异，对于稳定训练至关重要。

### 2.1 `rollout_corr/kl` & `rollout_corr/k3_kl`

| 属性 | 说明 |
|------|------|
| **定义** | 训练策略与 rollout 策略的 KL 散度 |
| **计算方式** | `kl = E[log π_rollout - log π_train]`（直接估计器）<br>`k3_kl = E[exp(r) - r - 1]` 其中 `r = log π_train - log π_rollout`（K3 估计器，更稳定） |
| **代码位置** | [verl/trainer/ppo/rollout_corr_helper.py#L759-L771](verl/trainer/ppo/rollout_corr_helper.py#L759-L771) |

| 正常趋势 | 异常信号 |
|---------|---------|
| 保持很小（~0.001-0.002），表示训练与 rollout 一致 | >0.01：显著 off-policy，可能导致训练不稳定 |

---

### 2.2 `rollout_corr/training_ppl` & `rollout_corr/rollout_ppl`

| 属性 | 说明 |
|------|------|
| **定义** | 训练/rollout 策略在生成序列上的困惑度 (Perplexity) |
| **计算公式** | $\text{PPL} = \exp(-\frac{1}{|T|}\sum_t \log \pi(y_t|y_{<t}))$ |
| **代码位置** | [verl/trainer/ppo/rollout_corr_helper.py#L749-L777](verl/trainer/ppo/rollout_corr_helper.py#L749-L777) |

| 正常趋势 | 异常信号 |
|---------|---------|
| 两者接近（比值~1.0） | 差异过大（>1.1 或 <0.9）：精度不一致问题 |

---

### 2.3 `rollout_corr/log_ppl_diff` & `rollout_corr/ppl_ratio`

| 属性 | 说明 |
|------|------|
| **定义** | 训练与 rollout PPL 的差异 |
| **计算方式** | `log_ppl_diff = mean_log_prob_rollout - mean_log_prob_train`<br>`ppl_ratio = exp(log_ppl_diff)` |
| **代码位置** | [verl/trainer/ppo/rollout_corr_helper.py#L779-L794](verl/trainer/ppo/rollout_corr_helper.py#L779-L794) |

| 正常趋势 | 异常信号 |
|---------|---------|
| `ppl_ratio` 接近 1.0（~0.99-1.01） | 偏离1.0过多：训练和 rollout 精度不匹配 |

---

### 2.4 `rollout_corr/chi2_token` & `rollout_corr/chi2_seq`

| 属性 | 说明 |
|------|------|
| **定义** | 卡方散度 $\chi^2(\pi_{\text{train}} \| \pi_{\text{rollout}}) = \mathbb{E}[\rho^2] - 1$ |
| **物理含义** | 衡量 importance sampling 权重的方差，越大表示 off-policy 越严重 |
| **代码位置** | [verl/trainer/ppo/rollout_corr_helper.py#L796-L810](verl/trainer/ppo/rollout_corr_helper.py#L796-L810) |

```python
# Token-level
chi2_token = verl_F.masked_mean(rho_squared_token, response_mask) - 1.0

# Sequence-level
rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)
chi2_seq = rho_squared_seq.mean() - 1.0
```

| 正常趋势 | 异常信号 |
|---------|---------|
| `chi2_token` 很小（~0.002-0.003） | `chi2_token` > 0.1：token 级 off-policy 严重 |
| `chi2_seq` 适中（~10-50） | `chi2_seq` > 1000：序列级 IS 权重方差爆炸 |

---

### 2.5 `rollout_corr/rollout_is_veto_fraction` & `rollout_corr/rollout_is_catastrophic_token_fraction`

| 属性 | 说明 |
|------|------|
| **定义** | 被拒绝的样本/token 比例（catastrophic outlier veto 机制） |
| **物理含义** | 检测是否存在极端 off-policy 样本 |
| **代码位置** | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py) |

| 正常趋势 | 异常信号 |
|---------|---------|
| 始终为 0（当前日志显示正常） | >0.1：存在大量 catastrophic samples，训练可能不稳定 |

---

## 三、训练调试指标 (`training/*`)

### 3.1 `training/rollout_probs_diff_*`

| 指标 | 定义 | 代码位置 |
|------|------|----------|
| `rollout_probs_diff_valid` | 输入是否有效（1=有效） | [verl/utils/debug/metrics.py#L104](verl/utils/debug/metrics.py#L104) |
| `rollout_probs_diff_max` | rollout vs actor 概率差的最大值 | [verl/utils/debug/metrics.py#L105](verl/utils/debug/metrics.py#L105) |
| `rollout_probs_diff_mean` | rollout vs actor 概率差的均值 | [verl/utils/debug/metrics.py#L106](verl/utils/debug/metrics.py#L106) |
| `rollout_probs_diff_std` | rollout vs actor 概率差的标准差 | [verl/utils/debug/metrics.py#L107](verl/utils/debug/metrics.py#L107) |

计算方式：
```python
actor_probs = torch.exp(actor_old_log_probs)
rollout_probs = torch.exp(rollout_old_log_probs)
rollout_probs_diff = abs(actor_probs - rollout_probs)  # 在 valid tokens 上
```

| 正常趋势 | 异常信号 |
|---------|---------|
| `mean` < 0.005，表示训练与 rollout 精度一致 | `mean` > 0.01：精度问题，见 [FAQ](https://verl.readthedocs.io/en/latest/faq/faq.html) |

---

### 3.2 `training/rollout_actor_probs_pearson_corr`

| 属性 | 说明 |
|------|------|
| **定义** | rollout 概率与 actor 概率的 Pearson 相关系数 |
| **参考论文** | [arXiv:2506.13585](https://arxiv.org/pdf/2506.13585) |
| **代码位置** | [verl/utils/debug/metrics.py#L52-L57](verl/utils/debug/metrics.py#L52-L57) |

| 正常趋势 | 异常信号 |
|---------|---------|
| 接近 1.0（~0.998-0.999） | < 0.95：训练与 rollout 实现差异大 |

---

## 四、Critic/Reward 相关指标 (`critic/*`)

### 4.1 `critic/score/*` & `critic/rewards/*`

| 指标 | 定义 | 代码位置 |
|------|------|----------|
| `score/mean,max,min` | 原始 token-level scores 的序列总和统计 | [verl/trainer/ppo/metric_utils.py#L101-L119](verl/trainer/ppo/metric_utils.py#L101-L119) |
| `rewards/mean,max,min` | 应用 KL 惩罚后的 rewards 统计 | 同上 |

计算方式：
```python
sequence_score = batch.batch["token_level_scores"].sum(-1)
sequence_reward = batch.batch["token_level_rewards"].sum(-1)
```

| 正常趋势 | 异常信号 |
|---------|---------|
| 随训练逐步提升 | 恒定不变：reward function 未生效 |
| `min` 和 `max` 有明显差异 | 全为同一值：可能是 reward bug |

---

### 4.2 `critic/advantages/*` & `critic/returns/*`

| 指标 | 定义 |
|------|------|
| `advantages/mean,max,min` | 优势函数值的统计（用于策略梯度） |
| `returns/mean,max,min` | 回报值统计 |

由于 FlowRL 使用 GRPO advantage estimator (`adv_estimator=grpo`)，advantages 实际上就是归一化后的 rewards。

| 正常趋势 | 异常信号 |
|---------|---------|
| `advantages/mean` 接近 0（归一化效果） | 均值显著偏离0：归一化失效 |
| `max` 和 `min` 对称（如 2.47 和 -2.47） | 严重不对称：数据分布有问题 |

---

## 五、性能与时序指标 (`perf/*`, `timing_s/*`)

### 5.1 性能指标

| 指标 | 定义 | 当前值 | 分析 |
|------|------|--------|------|
| `perf/mfu/actor` | Actor 训练的 MFU (Model FLOPs Utilization) | ~0.053 | 正常偏低，MoE 模型 MFU 通常较低 |
| `perf/max_memory_allocated_gb` | 最大 GPU 显存分配 | ~88.6 GB | 接近 H100 80GB 上限，注意 OOM 风险 |
| `perf/throughput` | 每秒每 GPU 处理的 tokens | ~277 | MoE 30B 模型的合理吞吐量 |
| `perf/total_num_tokens` | 每步处理的总 tokens | ~9.68M | batch_size=256, n=8, avg_len~4700 |

### 5.2 时序分解

| 指标 | 含义 | 当前值 | 占比 |
|------|------|--------|------|
| `timing_s/gen` | rollout 生成时间 | ~509s | 23% |
| `timing_s/old_log_prob` | actor log prob 计算 | ~192s | 9% |
| `timing_s/ref` | ref log prob 计算 | ~212s | 10% |
| `timing_s/update_actor` | actor 参数更新 | ~1270s | 58% ⚠️ |
| `timing_s/step` | 总 step 时间 | ~2184s | 100% |

**瓶颈分析**: `update_actor` 占比过高（58%），主要原因：
1. MoE 模型的 all-to-all 通信开销
2. `ppo_mini_batch_size=64` 相对于 `train_batch_size=256` 需要 4 次 gradient accumulation

---

## 六、序列长度统计 (`response_length/*`, `prompt_length/*`)

| 指标 | 定义 | 当前值 |
|------|------|--------|
| `response_length/mean` | 平均生成长度 | ~4567 tokens |
| `response_length/max` | 最大生成长度 | 8192 (触及上限) |
| `response_length/clip_ratio` | 被截断的比例 | ~22.4% |
| `prompt_length/mean` | 平均 prompt 长度 | ~161 tokens |
| `response/aborted_ratio` | 被中止的比例 | 0 |

**分析**:
- 22.4% 的样本触及 `max_response_length=8192`，这些样本的回答可能被截断
- 建议：如果准确率受影响，考虑增加 `max_response_length` 或启用 `overlong_buffer`

---

## 七、MoE 模型特定指标

### 7.1 负载均衡损失 (Load Balancing Loss)

⚠️ **当前日志中未看到显式的 MoE auxiliary loss 指标**。可能原因：
1. 使用的是 Qwen3-30B-A3B (Qwen3 MoE)，其 auxiliary loss 可能默认关闭
2. verl 框架可能将其合并到主 loss 中

建议检查：
- `actor_rollout_ref.model.override_config` 中是否有 `aux_loss_alpha` 配置
- 模型 config 中 `output_router_logits` 是否开启

### 7.2 Router 相关监控

当前未看到 router 分布指标。如需监控 expert 路由：
- 可以在 [recipe/flowrl/flowrl_actor.py](recipe/flowrl/flowrl_actor.py) 中添加 router logits 的统计

---

## 八、诊断指南汇总

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| `actor/final_loss` 持续上升 | 学习率过大 / reward 设计问题 | 降低 `lr`，检查 reward function |
| `actor/importance_weight` 频繁触及10 | off-policy 程度严重 | 增加 rollout 频率，减小 batch_size |
| `rollout_probs_diff_mean` > 0.01 | 训练与推理精度不一致 | 参考 [FAQ](https://verl.readthedocs.io/en/latest/faq/faq.html) 调整 vLLM dtype |
| `chi2_seq` > 1000 | 序列级 IS 权重方差爆炸 | 启用 `rollout_correction.rollout_rs` |
| `response_length/clip_ratio` > 30% | 太多样本被截断 | 增加 `max_response_length` |
| `actor/entropy` 趋向 0 | 策略坍缩 | 增加 `entropy_coeff` |
| `actor/ref_kl` 绝对值过大 | 策略偏离 ref 太远 | 增加 `kl_loss_coef` |

---

## 九、参考文献

1. **FlowRL**: Flow Matching 在 RL 中的应用（具体论文待确认）
2. **Rollout Correction**: ["When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"](https://richardli.xyz/rl-collapse)
3. **DAPO**: [Dynamic Alignment Policy Optimization](https://dapo-sia.github.io/)
4. **verl 文档**: [https://verl.readthedocs.io](https://verl.readthedocs.io)
