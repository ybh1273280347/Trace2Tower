---
type: source-material
title: "Trace2Tower: Transition-Aware EigenTrace Induction of Multi-Level Skills for LLM Agents"
aliases:
  - Trace2Tower 原始资料
  - Trace2Tower Obsidian 阅读版
source: "https://njn3pha612.feishu.cn/wiki/CBMawK7fqinyoFkWNYOcUMB2nvd"
created: 2026-06-28
tags:
  - trace2tower
  - source-material
  - llm-agent
  - hierarchical-skills
  - eigen-trace
---

# Trace2Tower: Transition-Aware EigenTrace Induction of Multi-Level Skills for LLM Agents

> [!abstract]
> 本资料提出 **Trace2Tower**，一种无需微调的大模型智能体技能分解框架，能够从原始执行轨迹中自动构造可复用的多层级技能结构。其核心是构建 **transition-aware EigenTrace graph**，将轨迹片段之间的语义相似性、时序依赖关系和成功/失败一致性共同编码到谱图中，从而诱导稳定的行为模式。

> [!info] 资料定位
> 这是一份从飞书导出的原始资料阅读版。内容保留原始资料主线，仅做 Obsidian 友好的结构化排版：标题层级、callout、表格、公式区块、代码块语言和列表整理。

## 0. 核心主张

### 0.1 EigenTrace 图构造

本文首先将 agent 的原始执行轨迹切分为事件级行为片段，并进一步构造 EigenTrace 图来刻画片段之间的关系。

不同于仅根据语义相似度进行轨迹聚类的方法，EigenTrace 同时考虑：

- 事件片段之间的语义相似性。
- 执行过程中的先后转移关系。
- 片段在成功或失败轨迹中的表现一致性。

该设计使技能发现能够显式利用 agent 行为中的执行顺序和结果反馈，从而更准确地识别具有复用价值的行为模式。

### 0.2 Spectrum-to-Tower 多层级技能诱导

本文基于 EigenTrace 图中的谱结构信息，从 agent 轨迹中自动诱导多层级技能塔。

| 技能层级 | 来源 | 表示对象 |
|---|---|---|
| Low-level skill | 重复出现的动作模板 | 原子操作 |
| Mid-level skill | 谱分解得到的稳定行为模式 | 可复用执行流程 |
| High-level skill | 中层技能之间的组合关系 | 跨任务策略结构 |

与已有扁平化 skill library 不同，Trace2Tower 能够同时表达原子动作、可复用流程和高层组合策略，从而更适合复杂任务中的技能复用与组合部署。

### 0.3 部署反馈驱动的 refinement

本文不依赖大模型微调，而是在外部技能结构层面进行持续优化。

部署过程中，系统根据每个技能的近期成功情况、任务收益、步骤节省效果和调用成本评估其实际效用，并据此对技能塔进行结构调整。

| 技能状态 | 结构化处理 |
|---|---|
| 表现不稳定 | 拆分 |
| 语义重复或功能重叠 | 合并 |
| 稳定有效的技能组合 | 提升为高层技能 |
| 低效或有害 | 删除或降权 |

### 0.4 实验环境

本文在 ALFWorld 和 WebShop 上系统评估 Trace2Tower 的多层级技能分解能力。

| 环境 | 验证重点 | 典型技能 |
|---|---|---|
| ALFWorld | household long-horizon task 中的操作型技能 | 物体定位、拿取、变换、放置 |
| WebShop | 网页交互轨迹中的决策型技能 | 搜索改写、候选筛选、属性验证、选项选择 |

通过这两个环境，本文验证 Trace2Tower 能够适配不同类型的 agent 执行轨迹，并在技能复用、组合泛化和 shortcut 抵抗方面优于原始轨迹检索、扁平技能总结和普通轨迹聚类方法。

## 1. 算法总览

> [!summary]
> Trace2Tower 的目标是从 LLM agent 的原始执行轨迹中自动诱导可复用、可组合、可部署的多层级技能结构。

与现有扁平化 skill library 不同，Trace2Tower：

- 不直接让 LLM 总结技能。
- 不只基于语义相似度检索历史经验。
- 先将执行轨迹转化为事件级行为片段。
- 再构造包含语义、时序和成败反馈的 EigenTrace 图。
- 通过对比式谱分解发现稳定行为模式。
- 最终形成 low / mid / high 三层技能塔。

整体算法可以概括为：

$$
\mathcal{D}_{trace} \rightarrow \mathcal{D}_{seg} \rightarrow G_{ET} \rightarrow Z_{ET} \rightarrow \mathcal{T} \rightarrow \mathcal{T}^{refined}
$$

| 符号 | 含义 |
|---|---|
| $\mathcal{D}_{trace}$ | 原始执行轨迹 |
| $\mathcal{D}_{seg}$ | 事件级轨迹片段 |
| $G_{ET}$ | Transition-Aware EigenTrace Graph |
| $Z_{ET}$ | EigenTrace 谱表示 |
| $\mathcal{T}$ | 多层级技能塔 |
| $\mathcal{T}^{refined}$ | 经过部署反馈修正后的技能塔 |

Trace2Tower 的整体流程可以分为 6 步：

1. 记录 Agent-Task-Trace 历史交互。
2. 构造 Event-Level Trajectory Segments。
3. 构造 Transition-Aware EigenTrace Graph。
4. 进行 Contrastive EigenTrace Decomposition。
5. 诱导 Multi-Level Skill Tower。
6. Skill-Augmented Deployment 与 Utility-Aware Refinement。

## 2. 记录 Agent-Task-Trace 历史交互

第一步记录 agent 在历史任务中的完整 step-level 交互轨迹。对于第 $i$ 个任务，记录为：

$$
\tau_i=\{x_{i,1},x_{i,2},...,x_{i,T_i}\}
$$

每一步交互为：

$$
x_{i,t}=(g_i,o_{i,t},A_{i,t},a_{i,t},r_{i,t},d_{i,t})
$$

这一步的核心作用是保留三类后续算法所需信号：

| 信号 | 用途 |
|---|---|
| 状态与动作语义 | 支持片段表示与语义相似性计算 |
| 执行顺序 | 支持时序转移强度建模 |
| 结果反馈 | 支持成功/失败一致性建模 |

## 3. 构造 Event-Level Trajectory Segments

原始 step-level 轨迹过细，不能直接作为技能。因此 Trace2Tower 将每条轨迹切分为事件级行为片段：

$$
\tau_i \rightarrow \mathcal{E}_i = \{e_{i,1},e_{i,2},...,e_{i,M_i}\}
$$

每个事件片段表示一个相对完整的局部行为，例如“定位物体”“拿取物体”“验证商品属性”“选择商品选项”。

### 3.1 ALFWorld 事件片段

原始动作序列：

```text
go to countertop -> take apple -> go to microwave -> heat apple -> put apple in fridge
```

可切分为：

```text
LocateObject -> AcquireObject -> TransformObject -> PlaceObject
```

### 3.2 WebShop 事件片段

WebShop 中，agent 需要根据文本 instruction 浏览多类网页并执行搜索、点击、选择和购买动作，这正好对应事件级分解。

原始动作序列：

```text
search[blue waterproof watch under 50] -> click[item] -> click[features] -> click[blue] -> click[buy now]
```

可切分为：

```text
QueryFormulation -> CandidateFiltering -> AttributeVerification -> OptionSelection -> PurchaseDecision
```

## 4. 构造 Transition-Aware EigenTrace Graph

> [!important]
> 这是 Trace2Tower 的第一个核心算法创新。

普通方法通常只根据 embedding 相似度做聚类。Trace2Tower 构造的是 **Transition-Aware EigenTrace Graph**，以 event segment 为节点，同时利用三类边权信息：

$$
W_{uv} = M_{uv} [ \alpha S_{uv} + \beta T_{uv} + \gamma O_{uv} ]
$$

| 符号 | 含义 |
|---|---|
| $S_{uv}$ | 语义相似性 |
| $T_{uv}$ | 时序转移强度 |
| $O_{uv}$ | 成败一致性 |
| $M_{uv}$ | 稀疏连接掩码 |
| $\alpha,\beta,\gamma$ | 三类信号的权重 |

### 4.1 语义相似性

$$
S_{uv}=\cos(h_u,h_v)
$$

其中 $h_u$ 和 $h_v$ 是两个事件片段的 embedding。

例如：

- ALFWorld 中 `heat apple` 与 `heat potato` 语义相近。
- WebShop 中 `click[features]` 与 `click[description]` 都可能属于属性验证行为。

### 4.2 时序转移强度

$$
T_{uv} = \frac{\#(u\rightarrow v)}{\#(u)+\epsilon}
$$

它刻画片段 $u$ 后面接片段 $v$ 的概率。

例如：

```text
AcquireObject -> TransformObject -> PlaceObject
QueryFormulation -> CandidateFiltering -> AttributeVerification
```

这一项使 Trace2Tower 能够学习“技能流程”，而不是只学习“技能类别”。

### 4.3 成败一致性

先定义片段 $u$ 的成功倾向：

$$
\rho_u=P(y=1|u\in\tau)
$$

再定义：

$$
O_{uv}=1-|\rho_u-\rho_v|
$$

如果两个片段都经常出现在成功轨迹中，它们连接更强；如果一个常来自成功轨迹，另一个常来自失败轨迹，它们连接变弱。

> [!warning] Shortcut failure
> 这对 WebShop 特别重要。例如表面上看起来合理的 `click title-matching item -> buy now`，如果 final reward 很低，就应被视为 shortcut failure，而不应和真正有效的购买片段强连接。

## 5. 进行 Contrastive EigenTrace Decomposition

> [!important]
> 这是 Trace2Tower 的第二个算法创新。

为了突出真正导向成功的行为模式，Trace2Tower 分别构造成功轨迹图和失败轨迹图：

$$
G^+,\quad G^-
$$

| 图 | 含义 |
|---|---|
| $G^+$ | 由成功轨迹中的片段和转移构成 |
| $G^-$ | 由失败轨迹中的片段和转移构成 |

然后构造对比邻接矩阵：

$$
W^{CE}=W^+-\lambda W^-
$$

其中 $\lambda$ 是失败模式惩罚权重。

再构造归一化图拉普拉斯：

$$
L^{CE}=I-D^{-\frac{1}{2}}W^{CE}D^{-\frac{1}{2}}
$$

并进行谱分解：

$$
L^{CE}q_k=\lambda_kq_k
$$

得到每个事件片段的 EigenTrace 表示：

$$
z_u=[q_1(u),q_2(u),...,q_r(u)]
$$

这一设计的意义是：

- 让成功轨迹中的稳定模式被保留。
- 让失败轨迹中的高频无效模式被削弱。
- 让片段聚类不只依赖语义 embedding，而是依赖“语义 + 转移 + 成败反馈”的谱结构。

## 6. 诱导 Multi-Level Skill Tower

Trace2Tower 将技能组织为三层：

$$
\mathcal{T} = \{ \mathcal{S}^{low}, \mathcal{S}^{mid}, \mathcal{S}^{high} \}
$$

其中，低层技能表示原子动作，中层技能表示可复用行为片段，高层技能表示由多个中层技能组成的任务级策略。

这一阶段不是简单使用一种聚类算法，而是采用多粒度技能诱导策略。

真正的核心聚类算法是：

```text
Transition-Aware Contrastive EigenTrace Spectral Clustering
```

也就是：先构造包含语义、时序和成败反馈的 EigenTrace 图，再进行谱分解，最后在谱空间中聚类事件片段。

### 6.1 Low-Level Skills：动作模板归纳

低层技能来自原始动作，不需要复杂聚类。系统将参数不同但动作语义相同的操作归为同一类。

这一层的目标是保证技能最终可以落回具体动作。例如，所有 `take X from Y` 类型的动作都可以归纳为：

```text
TakeObject(object, source)
```

### 6.2 Mid-Level Skills：EigenTrace 谱聚类

中层技能是 Trace2Tower 的核心。它不是直接对文本 embedding 做 K-means，而是先构造 **Transition-Aware EigenTrace Graph**。

这个图的每个节点是一个 event segment，边权同时考虑三类信息：

$$
W_{uv}=M_{uv}[\alpha S_{uv}+\beta T_{uv}+\gamma O_{uv}]
$$

这一步的创新在于，聚类不是只看“两个片段像不像”，而是看：

> [!note]
> 它们是否语义相似、是否经常在执行过程中相邻出现、是否共同导向成功。

然后，Trace2Tower 在这个图上进行对比式谱分解。系统会构造成功轨迹图和失败轨迹图，并削弱失败轨迹中的行为模式：

$$
W^{CE}=W^+-\lambda W^-
$$

这样可以让模型更关注导向成功的行为，而不是简单发现高频但无效的动作模式。

最后，对谱表示进行聚类，得到中层技能：

$$
\mathcal{S}^{mid}=Cluster(\{z_u\})
$$

这里的 `Cluster` 可以使用 K-means，但需要强调：**K-means 只是最后一步离散化，真正的算法核心是 EigenTrace 谱表示的构造。**

中层技能不是单个动作，而是一段可复用的行为阶段。

### 6.3 High-Level Skills：中层技能转移社区发现

高层技能不是直接从原始动作聚类得到，而是在中层技能基础上进一步归纳。

系统先统计中层技能之间的转移关系。例如：

```text
LocateObject -> AcquireObject -> TransformObject -> PlaceObject
QueryFormulation -> CandidateFiltering -> AttributeVerification -> OptionSelection -> PurchaseDecision
```

然后系统在中层技能转移图上做社区发现或路径挖掘，将经常共同出现、并且能够稳定导向成功的中层技能组合成高层技能。

例如，高层技能 `Multi-Attribute Purchase Routine` 会指导 agent 先搜索，再筛选，再验证属性，最后购买，而不是直接点击标题相似的商品。

## 7. Skill-Augmented Deployment 与 Utility-Aware Refinement

在新任务中，Trace2Tower 根据当前任务和状态检索相关技能：

$$
\mathcal{K}_t=Retrieve(g_t,o_t,\mathcal{T})
$$

为了避免 prompt 过长，可以只选择少量 high / mid / low 技能，例如：

```text
Top-1 high-level skill
Top-2 mid-level skills
Top-3 low-level skills
```

部署完成后，根据技能使用表现更新技能效用：

$$
U(s) = \eta_1SR(s) + \eta_2\Delta R(s) + \eta_3StepSave(s) - \eta_4Cost(s)
$$

然后执行四类结构化更新：

| 更新 | 作用 |
|---|---|
| Split | 拆分表现不稳定的技能 |
| Merge | 合并语义重复或功能重叠的技能 |
| Promote | 将稳定有效的技能组合提升为高层技能 |
| Prune / Downweight | 删除或降权低效、有害、高成本技能 |

## 8. 算法伪代码

```text
Algorithm: Trace2Tower

Input:
  D_trace: ALFWorld / WebShop agent trajectories
  f_phi: frozen segment encoder
  k: number of nearest neighbors
  r: spectral rank

Output:
  T = {S_low, S_mid, S_high}: multi-level skill tower

1. Record Agent-Task-Trace interactions:
   For each task i:
      collect tau_i = {(g_i, o_i,t, A_i,t, a_i,t, r_i,t, d_i,t)}
      record final outcome y_i and metadata m_i

2. Event-level segmentation:
   For each trajectory tau_i:
      split tau_i into event segments {e_i,1, ..., e_i,M_i}
      compute local reward gain and trajectory success label

3. Segment signature encoding:
   For each segment e_u:
      build compact signature sig_u
      compute embedding h_u = f_phi(sig_u)

4. Transition-aware EigenTrace graph construction:
   For each segment pair (u, v):
      compute semantic similarity S_uv
      compute transition strength T_uv
      compute outcome consistency O_uv
      build sparse edge W_uv

5. Contrastive EigenTrace decomposition:
   Build success graph G+ and failure graph G-
   Compute W_CE = W+ - lambda W-
   Construct normalized Laplacian L_CE
   Compute top-r eigenvectors q_1, ..., q_r
   Obtain segment representation z_u

6. Multi-level skill induction:
   S_low  = action-template clusters
   S_mid  = clusters over EigenTrace representations
   S_high = communities over mid-skill transition graph

7. Skill-augmented deployment:
   Retrieve high/mid/low skills for each new task state
   Inject selected skills into the agent prompt

8. Utility-aware refinement:
   Update skill success rate, reward gain, step saving, and cost
   Split, merge, promote, or prune skills

Return:
  T = {S_low, S_mid, S_high}
```

## 9. 核心算法创新

> [!important]
> Trace2Tower 的核心算法创新是：将 LLM agent 的原始执行轨迹转化为事件片段图，并在图边中同时编码语义相似性、时序转移关系和成败一致性；再通过成功/失败对比谱分解提取导向成功的 EigenTrace 行为模式，最终诱导出 low / mid / high 多层级技能塔，并在部署中持续 refinement。

## 10. 研究问题

> [!question] RQ1
> PUE 能否准确预测技能未来效用？

> [!question] RQ2
> PUE 是否优于频率、成功率、相似度等启发式策略？

> [!question] RQ3
> PUE 是否能提升长期任务表现和技能使用效率？

> [!question] RQ4
> PUE 中哪些因素最关键？

