# 指标如何支撑实验故事

这份文档解释当前实验指标能说明什么、不能说明什么，以及后续怎样用它们证明 Trace2Tower/PUE 强于 baseline。

## 当前故事主线

当前研究问题已经收束到 PUE：

1. PUE 能否准确预测技能未来效用？
2. PUE 是否优于频率、历史成功率、语义相似度等启发式策略？
3. PUE 是否能提升长期任务表现和技能使用效率？
4. PUE 中哪些因素最关键？

注意：当前代码里的 `pue_full` 是可解释 proxy，用来跑通评估链路。论文最终 PUE 可以替换它，但应保持同一套指标输出，方便和 baseline 对齐。

## 指标分层

### A. 任务执行指标

来源：

```text
experiments/*/summary.json
experiments/webshop_baseline_summary.csv
```

字段：

| 指标 | 越大越好 | 能说明什么 | 当前限制 |
|---|---:|---|---|
| `success_rate` | 是 | 最终任务完成率 | `smoke_random` 只能测链路；`llm_action` 可做 skill 注入 smoke |
| `avg_reward` | 是 | WebShop 等连续 reward 任务表现 | 小样本只能观察，不能写论文结论 |
| `avg_steps` | 否 | 执行效率，步数越少越好 | 需要结合成功率看，不能单独追求更少 |
| `invalid_action_rate` | 否 | 动作合法性、环境接口稳定性 | WebShop 当前基本为 0 |
| `loop_rate` | 否 | 是否陷入重复动作 | 可作为长程任务效率/鲁棒性侧证 |
| `avg_token_cost` | 否 | LLM action prompt/response token 成本 | 只在 LLM agent 下有意义 |
| `llm_action_fallback_rate` | 否 | LLM JSON 解析失败或非法动作比例 | 高了说明 prompt/action contract 不稳 |
| `episodes` | - | 样本数量 | 太小只能 smoke，不能写结论 |

这些指标未来用于回答 RQ3：PUE 是否提升长期任务表现和技能使用效率。

当前已经有 `llm_action` 让 skill 影响动作，但只有小样本 smoke。正式比较要在相同 LLM、相同 episode budget、相同环境 split 下跑 no-skill 和各个 skill baseline。

### B. 技能产物与检索指标

来源：

```text
experiments/*/model.json
experiments/*/retrieval.jsonl
experiments/*/deployment_retrieval.jsonl
experiments/webshop_baseline_summary.csv
```

字段：

| 指标 | 能说明什么 |
|---|---|
| `model_method` | 当前跑的是哪个 miner/baseline |
| `skill_count` | 方法产生多少技能或技能节点 |
| `avg_retrieved_skills` | 每个任务平均检索多少技能 |
| `avg_deployment_retrieved_skills` | 在线执行时每一步平均注入多少技能 |
| `retrieval_score` | selector 给单个 skill 的排序分数 |
| `retrieval_strategy` | 当前使用的检索策略 |

这些指标用于证明实验链路闭合：baseline 真的产生了技能，retriever 真的做了选择。

它们不能单独证明任务表现更强，只能解释后续任务表现差异来自哪里。

### C. PUE/selector 预测指标

来源：

```text
experiments/*/selectors/selector_metrics.json
experiments/*/selectors/selector_scores.jsonl
```

字段：

| 指标 | 越大越好 | 支撑哪个 RQ | 含义 |
|---|---:|---|---|
| `future_utility_pearson` | 是 | RQ1/RQ2 | 预测分数和未来效用的线性相关 |
| `future_utility_spearman` | 是 | RQ1/RQ2 | 排序相关，更适合 top-k skill selection |
| `positive_utility_auc` | 是 | RQ1/RQ2 | 区分正收益/非正收益 skill 的能力 |
| `avg_selected_utility` | 是 | RQ2/RQ3 | 被 selector 选中的 skill 平均未来效用 |
| `avg_oracle_utility` | 是 | RQ2 | oracle selector 的上界 |
| `regret` | 否 | RQ2/RQ3 | selector 相比 oracle 少拿多少 utility |
| `positive_selection_rate` | 是 | RQ2/RQ3 | 选中正收益 skill 的比例 |
| `avg_selected_token_cost` | 否 | RQ3/RQ4 | 被选 skill 的上下文成本 |

重点读法：

- 如果 PUE 的 `future_utility_spearman` 高于 frequency/success/similarity，说明它更会排序未来有用技能。
- 如果 PUE 的 `regret` 更低，说明它离 oracle 更近。
- 如果 PUE 的 `avg_selected_utility` 更高，说明它实际选中的技能更有价值。
- 如果 PUE 的 `avg_selected_token_cost` 没有明显升高，说明它不是靠塞更多上下文取胜。

## 如何证明强于 baseline

### 表 1：预测质量

目标：回答 RQ1/RQ2。

```text
Selector | Pearson | Spearman | AUC | Regret | Avg Selected Utility
Frequency
Success Rate
Similarity
Recent Reward Lift
PUE Full
```

结论条件：

- `PUE Full` 的 Spearman/AUC 最高或稳定接近最高。
- `PUE Full` 的 regret 最低。
- `PUE Full` 的 avg selected utility 最高。

如果 PUE 只在 Pearson 高但 regret 不低，说明预测整体趋势不错，但 top-k 选择不一定好；论文中应谨慎。

### 表 2：PUE 因素消融

目标：回答 RQ4。

```text
Variant | Spearman | Regret | Avg Utility | Token Cost
PUE Full
w/o cost
w/o recent
w/o similarity
w/o reward lift
w/o failure risk
w/o coverage
```

结论条件：

- 去掉某个因素后 Spearman 降、regret 升，说明该因素关键。
- 如果 `w/o cost` utility 相近但 token cost 明显升高，说明 cost 项主要贡献效率。
- 如果 `w/o similarity` 在 OOD 或新任务上掉得明显，说明 context relevance 很重要。

### 表 3：长期部署表现

目标：回答 RQ3。

```text
Method | Success Rate | Avg Reward | Avg Steps | Token Cost | Success/1k Tokens
No Skill
Raw Trajectory
Flat Skill
SkillX official
SkillLens official
Trace2Tower + PUE
```

结论条件：

- `Trace2Tower + PUE` 的 success/reward 更高。
- steps/token cost 不显著恶化，最好更低。
- success per token 更高，说明效率而非堆上下文。

当前已经可以用 `llm_action` 产出这张表的 smoke 版本，但不能用 1 episode 结果写结论。

## 当前代码里的 future utility proxy

`scripts/evaluate_selectors.py` 当前用离线 proxy 构造 future utility：

```text
skill-task match * future task value - token cost penalty
```

其中：

- `skill-task match` 来自 segment label 和 action template 的匹配。
- `future task value` 来自 reward、success、step efficiency。
- `token cost penalty` 防止长 skill 天然占便宜。

这个 proxy 用来先回答“selector 评估链路能否跑通”。正式论文可以替换为更强定义，例如真实 deployment 后的 success lift、reward lift、step saving 和 token cost。

## 哪些指标不能乱用

- 小样本 smoke 的 `success_rate` 不能写论文结论。
- 当前 smoke 的任务 reward 一样，不代表 baseline 没差异；可能是样本太小，或当前任务不需要技能。
- `skill_count` 多不代表强，可能只是冗余更多。
- `avg_retrieved_skills` 高不代表强，可能只是上下文更长。
- `avg_token_cost` 更高时，必须结合 success/reward lift 解释；否则只是花了更多上下文。
- `positive_utility_auc=0` 在小样本或全正/全负标签时不一定有意义，需要检查样本分布。

## 下一步补齐证据

1. 扩大 episodes，保证每个 selector 有足够正负样本。
2. 对所有 baseline 跑 `evaluate_selectors.py`，不要只看 SkillX。
3. 新增更多 PUE ablation。
4. 用 `llm_action` 做相同 agent 下的长期部署对比。
5. 报告长期 batch 曲线：随着 deployment 增加，PUE 是否降低 regret、提升 reward、降低 token cost。
