# 四设备无线信道马尔可夫随机博弈 Demo 开发约束与规范

## 1. 项目定位

本项目的首要目标是实现一个可复现实验闭环，用于演示四设备在两态无线信道上的马尔可夫随机博弈过程，包括环境仿真、至少一种可运行学习算法、评估指标统计、参数敏感性分析和图表输出。

本项目不是先做“大而全”的 MARL 平台，而是先做一个可解释、可验证、可扩展的教学型 demo。第一阶段必须保证模型定义、转移逻辑、奖励逻辑和日志输出是正确的；算法数量和 UI 展示可以后补。

## 2. 数学模型约束

### 2.1 智能体与时间

- 智能体数量固定为 `N = 4`。
- 系统按离散时隙演化，`t = 0, 1, 2, ...`。
- 每个时隙内 4 个设备同时动作。

### 2.2 信道与状态空间

- 信道状态定义为 `c_t ∈ {0, 1}`。
- `c_t = 0` 表示空闲信道。
- `c_t = 1` 表示拥堵信道。
- 每个设备的缓存状态为 `b_{i,t} ∈ {0,1}`，其中 `1` 表示时隙开始时有待发送数据。
- 全局缓存向量定义为 `b_t = (b_{1,t}, b_{2,t}, b_{3,t}, b_{4,t}) ∈ {0,1}^4`。
- 全局状态定义为 `s_t = (c_t, b_t)`。
- 状态空间大小固定为 `|S| = 2 × 2^4 = 32`。

### 2.3 通信拓扑约束

- 拓扑用邻接矩阵 `A ∈ {0,1}^{4×4}` 表示。
- `A[i,j] = 1` 表示设备 `i` 可以向设备 `j` 发送。
- `A[i,i]` 必须恒为 `0`。
- 必须支持三种拓扑：
  - `all_to_all`：任意 `i != j` 都可发送。
  - `star`：0 号设备为网关，其他节点仅可与 0 号通信。
  - `ring`：每个节点只和左右邻居通信。
- 选择目标 `j` 但 `A[i,j] = 0` 时，动作视为“无效目标”，必须施加额外惩罚。

### 2.4 动作空间约束

- 每个设备动作集合固定为 5 个离散动作：
  - `0 = WAIT`
  - `1 = STANDBY`
  - `2, 3, 4 = TX(target)`
- 对于 `n=4`，动作 `2/3/4` 通过循环偏移映射到除自身外的 3 个目标节点。
- 代码级 canonical 语义应保持与附录一致：`decode_target(i, a)` 使用 `offset = a - 1`，目标为 `(i + offset) % n`。
- 首版 demo 不允许改成显式二维动作 `(是否发送, 目标)`，必须保持单整数动作接口，保证与算法输出维度一致。

### 2.5 观测与信息结构

- 默认采用部分可观测设定。
- 设备 `i` 的局部观测为 `o_{i,t} = (\tilde c_{i,t}, b_{i,t})`。
- 信道观测满足：
  - `Pr(\tilde c_{i,t} = c_t) = p_obs`
  - `Pr(\tilde c_{i,t} != c_t) = 1 - p_obs`
- 完全可观测简化版允许设置 `p_obs = 1`，但不得修改状态和奖励定义。
- 附录代码当前采用“所有 agent 共享同一个带噪信道观测样本”的实现；若后续改为各 agent 独立噪声，必须同步更新测试与文档。

### 2.6 发送数、碰撞与成功规则

- 时隙发送设备数定义为：

```math
k_t = \sum_{i=1}^4 \mathbf{1}[a_{i,t} \in \{TX(\cdot)\}]
```

- 若 `k_t >= 2`，则发生碰撞，首版 demo 中所有发送全部失败。
- 首版 demo 不实现捕获效应、多包接收或 SIC。
- 若 `k_t = 1`，唯一发送者成功概率依赖信道状态：

```math
Pr(\text{succ} \mid c_t = 0) = q_0,\quad
Pr(\text{succ} \mid c_t = 1) = q_1,\quad q_1 < q_0
```

- 代码级 canonical 语义还必须满足：
  - 成功发送要求 `b_{i,t} = 1`。
  - 成功发送要求目标合法。
  - 无包时选择发送，不能成功。
  - 无效目标动作在当前实现中仍计入 `k_t`，因此仍会影响碰撞与信道拥堵转移。

### 2.7 缓存转移规则

- 成功发送后，缓存清空：`b_{i,t}^{after} = 0`。
- 对于清空后的空缓存设备，在时隙末以 Bernoulli 到达补包：

```math
Pr(b_{i,t+1} = 1 \mid b_{i,t}^{after} = 0) = \lambda_i
```

- 若未成功发送且原本有包，缓存保持非空。
- 首版 demo 不实现队列长度大于 1 的缓存。

### 2.8 信道状态转移规则

- 信道拥堵概率采用线性截断模型：

```math
Pr(c_{t+1}=1 \mid c_t, k_t) =
\operatorname{clip}(p_{base} + p_{load} \cdot k_t + p_{persist} \cdot c_t,\ 0,\ 1)
```

- `clip` 必须严格截断到 `[0,1]`。
- 任何实现都不得直接使用未裁剪概率。

### 2.9 奖励函数约束

- 设备 `i` 的即时奖励定义为：

```math
r_i(s_t, a_t) =
R_{succ} \cdot \mathbf{1}[i\ \text{成功发送}]
- C_{coll} \cdot \mathbf{1}[i\ \text{发送且发生碰撞}]
- C_{fail} \cdot \mathbf{1}[i\ \text{发送但失败且}\ b_{i,t}=1]
- E(a_{i,t})
- C_{delay} \cdot b_{i,t}
- C_{inv} \cdot \mathbf{1}[\text{选择无效目标}]
```

- 能耗必须满足：

```math
E(WAIT)=e_w,\quad E(STANDBY)=e_s,\quad E(TX)=e_t,\quad e_t \gg e_w \ge e_s
```

- 代码级 canonical 语义必须保持：
  - `WAIT` 只扣 `e_wait`，若有包再扣 `c_delay`。
  - `STANDBY` 只扣 `e_sleep`，若有包再扣 `c_delay`。
  - `TX` 一定扣 `e_tx`。
  - `collision` 惩罚仅对 `b_i=1` 的发送者生效。
  - `c_fail` 仅对“发送失败且原本有包”的情况生效。
  - 空缓存发送者当前不会被扣 `c_fail`，只承担能耗和可能的无效目标惩罚。

### 2.10 折扣回报与均衡评估

- 每个设备的目标函数为：

```math
J_i = \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_i(s_t, a_t)\right],\quad \gamma \in (0,1)
```

- 随机博弈标准形式写为：

```math
G = \langle N, S, \{A_i\}_{i=1}^N, P, \{r_i\}_{i=1}^N, \gamma \rangle
```

- 实验中采用 `\epsilon`-Nash 近似判据：

```math
\Delta_i = \max_{\pi_i'} \hat J_i(\pi_i', \hat \pi_{-i}) - \hat J_i(\hat \pi)
```

- 当 `max_i Δ_i <= ε` 时，认为达到 `ε`-Nash。

### 2.11 简化阶段博弈约束

- 简化解析仅在以下前提下使用：
  - 固定 `c=0`
  - 所有设备持续有包 `b_{i,t}=1`
  - 动作只允许 `{TX, WAIT}`
- 其解析式必须按以下口径实现：

```math
u_{solo} = q_0 R_{succ} - e_t
```

```math
u_{coll} = -C_{coll} - e_t
```

```math
u_{wait} = -e_w - C_{delay}
```

```math
U_{TX}(p) = (1-p)^3 u_{solo} + [1-(1-p)^3] u_{coll}
```

```math
U_{WAIT}(p) = u_{wait}
```

- 对称混合策略均衡通过 `U_TX(p*) = U_WAIT(p*)` 数值求解。

## 3. 默认参数规范

以下默认值来自 `docx` 主文与附录 `config.yaml`，首版 demo 必须以它们为默认配置：

| 参数 | 默认值 |
| --- | --- |
| `n_agents` | `4` |
| `episode_len` | `200` |
| `gamma` | `0.95` |
| `arrival_p` | `0.3` |
| `succ_idle` | `0.95` |
| `succ_cong` | `0.60` |
| `p_base` | `0.05` |
| `p_load` | `0.20` |
| `p_persist` | `0.50` |
| `r_succ` | `1.0` |
| `c_coll` | `1.0` |
| `c_fail` | `0.3` |
| `c_delay` | `0.01` |
| `c_invalid` | `0.2` |
| `e_tx` | `0.05` |
| `e_wait` | `0.005` |
| `e_sleep` | `0.001` |
| `p_obs_correct` | `0.9` |
| `topology` | `all_to_all` |

## 4. 工程实现规范

### 4.1 推荐目录结构

```text
wireless_marl/
  config.yaml
  env.py
  algos/
    iql.py
    qmix.py
    mappo.py
  train.py
  eval.py
  plot.py
  utils.py
```

### 4.2 环境接口约束

- 环境类名固定为 `WirelessMarkovGame`。
- 参数类名固定为 `EnvParams`。
- `reset(seed)` 必须返回 `obs_dict`。
- `step(actions_dict)` 必须返回：
  - `obs_dict`
  - `reward_dict`
  - `terminated`
  - `truncated`
  - `info`
- `obs_dict` 和 `reward_dict` 的 key 必须是 `0..3`。
- 必须保留 `get_state_vec()` 供 CTDE 算法使用。

### 4.3 `info` 字段规范

`info` 至少包含以下字段：

- `t`
- `k_tx`
- `collision`
- `success_vec`
- `invalid_vec`
- `buffers`
- `channel`
- `p_congest_next`

### 4.4 工具函数规范

- 必须保留 `clip01()`。
- 必须保留 `topology_adjacency()`。
- 必须提供统一 `set_global_seed()`。
- 所有随机源必须可复现。

### 4.5 轻量化算力要求

本项目默认面向普通笔记本或课程实验机开发，必须遵循轻量化约束。当前 demo 范围固定为 `值迭代 / IQL / QMIX / IPPO(MAPPO)`。

- 默认训练与评估必须支持 `CPU` 运行，不得把 `GPU/CUDA` 设为必需依赖。
- 默认实现不得依赖分布式训练、参数服务器或多机框架。
- 单次训练任务应以“可演示优先”为原则，默认配置不得使用大规模网络、超大 replay buffer 或高频多轮更新。
- 多随机种子实验默认采用串行运行，不要求并行加速。
- 绘图、评估、参数扫描必须与训练解耦，避免重复训练造成无意义算力消耗。

建议的轻量化配置上限如下：

- `值迭代`：允许全状态与全联合动作枚举，不需要额外算力优化。
- `IQL`：默认使用表格型实现，不引入神经网络近似。
- `QMIX`：默认 agent hidden size 不超过 `64`，mixing hidden size 不超过 `32`，`batch_size` 不超过 `64`，`buffer_size` 不超过 `20000`。
- `IPPO/MAPPO`：默认 hidden size 不超过 `128`，`update_epochs` 不超过 `4`，`batch_size` 不超过 `1024`。
- 所有算法的默认实验配置都应能在轻量机器上完成单 seed 演示。

### 4.6 算法实现规范

本项目允许同时存在“博弈建模算法”和“工程训练算法”两类实现，但首版 demo 必须明确区分它们的用途：

- “解析/规划型算法”用于做理论对照、上界验证或小规模真值参考。
- “学习型算法”用于做主展示、训练曲线和敏感性分析。

#### 4.6.1 集中式值迭代

- 适用定位：完全可观测、集中控制、共享团队目标时的上界基线。
- 适用前提：
  - 使用全局状态 `s_t = (c_t, b_t)`。
  - 使用联合动作 `a_t = (a_{1,t}, a_{2,t}, a_{3,t}, a_{4,t})`。
  - 优化团队奖励 `r_{team} = Σ_i r_i`。
- 状态数固定为 `32`，联合动作数固定为 `5^4 = 625`，理论上可枚举。
- Bellman 更新必须按如下形式实现：

```math
V_{new}(s) =
\max_{a_{joint}}
\left[
R_{team}(s, a_{joint})
+ \gamma \sum_{s'} P(s' \mid s, a_{joint}) V(s')
\right]
```

- 实现约束：
  - 若实现值迭代，必须显式枚举全部状态和联合动作。
  - 必须区分“真实解析转移”与“采样估计转移”，不能混写。
  - 若使用采样近似转移，文件名和文档中不得仍称其为“精确值迭代”。
- 用途约束：
  - 值迭代不是首版主训练算法。
  - 若实现，它应作为 sanity check 和理论参考。

#### 4.6.2 IQL

- 适用定位：首版 demo 的主算法和最小可运行基线。
- 训练范式：分散训练，分散执行。
- 每个 agent 独立维护 `Q_i(o_i, a_i)`。
- 默认观测为局部观测 `o_i = (\tilde c_i, b_i)`，因此表格型 IQL 的最小状态数为 `4`，动作数为 `5`。
- 更新规则必须按如下口径实现：

```math
Q_i(o_i, a_i)
\leftarrow
(1-\alpha) Q_i(o_i, a_i)
+ \alpha \left[
r_i + \gamma \max_{a_i'} Q_i(o_i', a_i')
\right]
```

- 训练奖励使用各自个体奖励 `r_i`，不得替换为团队奖励。
- 动作选择默认用 `epsilon-greedy`。
- 实现约束：
  - 一个完整训练过程中，agent 不得在每次 `eval_every` 后重新初始化。
  - `epsilon` 必须是连续衰减而不是分段重置。
  - 评估阶段必须用 `greedy=True`。
  - 必须支持导出每个 agent 的动作直方图，供饼图分析使用。
- 预期作用：
  - 用于展示非平稳环境下的基线表现。
  - 用于说明局部观测下的偶然协调或不稳定性。

#### 4.6.3 QMIX

- 适用定位：协作设定下的 CTDE 主算法。
- 训练范式：集中训练，分散执行。
- 必须使用局部观测 `o_i` 作为 agent 网络输入，使用全局状态 `s` 作为 mixing network 输入。
- QMIX 训练目标必须基于团队奖励，而不是各自独立奖励。
- 团队奖励默认定义为：

```math
r_{team}(t) = \sum_{i=1}^4 r_i(s_t, a_t)
```

- 单 agent utility 网络输出 `Q_i(o_i, a_i)`。
- mixing network 输出 `Q_{tot}(s, a_1, ..., a_4)`，并保持对每个 `Q_i` 的单调性约束。
- 经验回放样本至少包含：
  - `state`
  - `obs_dict`
  - `action_dict`
  - `r_team`
  - `next_state`
  - `next_obs_dict`
  - `done`
- 实现约束：
  - 必须有 online / target 网络。
  - 必须有 replay buffer。
  - 必须有 soft update 或 hard update 机制。
  - 评估时只能使用各 agent 的局部观测贪心执行，不能读取全局状态选动作。
- 用途约束：
  - QMIX 只在协作口径下有明确意义。
  - 若实验采用一般和博弈奖励，QMIX 结果必须标注为“团队奖励近似”。

#### 4.6.4 IPPO / MAPPO

- 适用定位：策略梯度系方法中的稳定强基线。
- `IPPO` 表示各 agent 独立 PPO。
- `MAPPO` 表示多智能体 PPO，推荐参数共享 actor，并允许 critic 使用更多全局信息。
- 首版推荐口径：
  - actor 输入本地观测 `o_i`
  - critic 可用本地观测或全局状态，需在配置中显式声明
  - 协作实验默认优化团队回报
- PPO 更新必须遵循 clip objective，至少包含：
  - `clip_eps`
  - `gae_lambda`
  - `entropy_coef`
  - `value_loss_coef`
- 实现约束：
  - 必须明确是 on-policy，旧样本不可重复长期回放。
  - rollout buffer 中必须保存 `obs, act, logp, rew, done, val`。
  - 必须支持 GAE。
  - 参数共享与否必须可配置。
- 用途约束：
  - 若目标是“更稳的协作学习展示”，在保留算法中 `MAPPO` 是最强候选。

#### 4.6.5 算法选型优先级

| 算法 | 训练口径 | 主要输入 | 优化目标 | 首版是否必做 | 主要用途 |
| --- | --- | --- | --- | --- | --- |
| 值迭代 | 集中式规划 | 全局状态、联合动作 | 团队回报 | 否 | 理论上界/真值参考 |
| IQL | 分散训练 | 局部观测 | 个体回报 | 是 | 最小闭环基线 |
| QMIX | CTDE | 局部观测 + 全局状态 | 团队回报 | 否 | 协作主力算法 |
| IPPO/MAPPO | On-policy CTDE/分散 | 局部观测，可选全局 critic | 团队回报或个体回报 | 否 | 稳定策略梯度基线 |

### 4.7 算法分阶段约束

- 第一阶段必须跑通 `IQL` 的完整训练、评估、绘图闭环。
- 第二阶段再接入 `QMIX / IPPO(MAPPO)`。
- `值迭代` 可并行实现为理论对照模块，但不阻塞首版交付。

### 4.8 训练与评估规范

- 训练与评估必须严格分离。
- 训练时允许 `epsilon-greedy` 或采样噪声。
- 评估时必须固定为贪心或确定性策略。
- 每种算法至少跑 `3~10` 个随机种子。
- 指标汇报必须包含均值和标准差，或置信区间。

### 4.9 日志与结果输出规范

- 训练日志至少输出 CSV。
- 首版日志最少包含：
  - `episode`
  - `eps`
  - `throughput`
  - `collision`
  - `avg_reward_per_agent`
- 输出图目录固定为 `outputs/figs`。

### 4.10 图表规范

至少生成以下图：

- 训练回合 vs 吞吐折线图。
- 训练回合 vs 碰撞率折线图。
- 训练回合 vs 平均奖励折线图。
- 算法对比柱状图。
- 动作分布饼图。
- 参数敏感性热力图。

## 5. 评价指标与实验规范

必须统计以下指标：

- 平均吞吐量：`successes / slot`
- 碰撞率：`collision_slots / slot`
- 平均能耗：`energy / slot`
- 社会福利：`Σ_i r_i` 的折扣累计或每步平均
- 均衡性指标：`ε`-Nash 偏离优势上界
- 遗憾值：如后续实现无悔学习分析

必须做两类敏感性分析：

- 奖励参数扫描：`C_coll, C_delay, e_tx`
- 信道参数扫描：`p_load, p_persist, q_1`

## 6. 测试与验收规范

最小单元测试集必须覆盖以下内容：

- `reset/step` 返回维度和 key 正确。
- `seed` 固定时环境可复现。
- `clip01` 输出必在 `[0,1]`。
- `all_to_all/star/ring` 邻接矩阵正确。
- 单发送者在合法目标和有包条件下才可能成功。
- `k_t >= 2` 时必发生碰撞。
- 无效目标动作会触发 `c_invalid`。
- 奖励数值在合理范围内，不出现 `NaN/Inf`。
- `p_congest_next` 与 `k_t`、`c_t` 逻辑一致。

最小运行验收命令应为：

```bash
pytest -q
python train.py --algo iql
python plot.py
```

如果 CLI 最终仍采用 `python train.py --config config.yaml`，则需要同步更新文档与验收脚本，不允许文档和入口不一致。

## 7. 附录代码的当前状态与开发注意事项

`docx` 附录代码可以作为实现参考，但当前不能被视为“可直接交付的最终版本”，原因如下：

- `train.py` 当前只真正支持 `IQL`，其余算法未接入统一训练入口。
- `train.py` 中的 IQL 训练循环按 `eval_every` 分段时会重新初始化 agent，这会导致训练曲线不是连续训练结果，发布前必须修正。
- `eval.py` 仍是占位实现。
- `plot.py` 尚未记录动作直方图，因此饼图只是占位说明。
- 深度算法模块是教学版实现，工程上需要补齐模型保存、加载、日志、异常处理和训练状态恢复。

因此，后续开发必须遵循以下优先级：

- 先固定环境语义与测试。
- 再修复 IQL 持续训练与评估闭环。
- 然后补齐图表和参数扫描。
- 最后再扩展 QMIX / IPPO(MAPPO)。

## 8. Demo 第一阶段交付定义

首版 demo 交付物必须至少包括：

- 一个可复现的 `WirelessMarkovGame` 环境。
- 一个可跑通的 `IQL` 训练脚本。
- 一套默认参数配置。
- 一套基础指标日志。
- 至少 3 张自动生成图。
- 一份简化均衡分析说明。
- 一套可执行测试。

如果以上 7 项没有全部完成，不应进入展示包装或 UI 美化阶段。
