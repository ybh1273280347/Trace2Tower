# 面向 AI 接手的研究与实验简报

这份文档给后续 AI 助手快速接手 Trace2Tower/PUE 实验。先读本文，再读 `docs/ai_assist/05_experiment_skeleton_handoff.md` 和代码。

## 研究定位

Trace2Tower 的大故事是：现有方法已经证明 execution traces 可以生成 skill，多粒度 skill representation 有价值，已有 skill graph 可以支持更细粒度复用；但它们没有充分回答 skill tower 本身如何从 raw execution traces 中自动发现。

当前用户最关心的实验问题已经收束到 PUE：

1. RQ1：PUE 能否准确预测技能未来效用？
2. RQ2：PUE 是否优于频率、成功率、相似度等启发式策略？
3. RQ3：PUE 是否能提升长期任务表现和技能使用效率？
4. RQ4：PUE 中哪些因素最关键？

这里的 PUE 是 Predictive Utility Estimator，用于预测某个 skill 在未来任务或当前上下文中的期望效用。不要把当前代码里的 `pue_full` 视作论文最终算法；它只是让 selector 评估链路先跑通的可解释 baseline/proxy。

## 需要回答的实验证据

### RQ1：预测未来效用

核心指标：

- Pearson / Spearman：预测分数和未来真实 utility 的相关性。
- AUC：区分正收益 skill 与负收益 skill。
- Calibration：预测 utility bucket 与实际 utility 是否一致。
- Regret：相比 oracle utility selector 损失多少。

当前入口：

```bash
python scripts/evaluate_selectors.py \
  --config configs/baseline_skillx_webshop.json \
  --records experiments/baseline_skillx_webshop/records.jsonl \
  --output-dir experiments/baseline_skillx_webshop/selectors
```

后续真正 PUE 应替换或新增 selector，并保持同一输出格式。

### RQ2：对比启发式策略

必须比较：

- `frequency`
- `success_rate`
- `similarity`
- `recent_reward_lift`
- `pue_full`
- PUE ablation variants

结果表至少包含：

```text
selector, future_utility_corr, auc, regret, avg_selected_utility, avg_selected_token_cost
```

如果加入学习式 PUE，不要删除这些 heuristic baseline。

### RQ3：长期表现和效率

离线预测只回答“选得准不准”。长期表现需要 deployment batch：

```text
train traces -> induce skills -> deploy with selector -> collect future traces -> update feedback -> next batch
```

指标：

- success rate / avg reward
- avg steps
- selected token cost
- skill reuse rate
- positive utility selection rate
- regret over deployment batches

当前 `scripts/evaluate_selectors.py` 已输出 `batches` 字段，可作为长期评估格式雏形；真正 deployment 后要把 batch 从离线 split 扩展为真实时间顺序。

当前已经接入 `llm_action` agent，selector 检索到的 skill 可以通过 prompt 影响动作。它适合先跑 RQ3 的工程 smoke，但小样本结果不能直接写成论文结论。

### RQ4：PUE 因素消融

建议因素：

- recent success / recent trend
- reward lift
- transition confidence
- coverage / support
- semantic relevance
- token cost
- failure risk

当前已有 selector 名称：

- `pue_full`
- `pue_no_cost`
- `pue_no_recent`
- `pue_no_similarity`

后续可继续增加：

- `pue_no_reward_lift`
- `pue_no_failure_risk`
- `pue_no_coverage`
- `pue_no_transition`

## Baseline 语义

当前可运行 baseline miner：

- `no_skill`：无技能，下限。
- `raw_trajectory`：episodic memory，把每条轨迹作为可检索记忆。
- `flat_skill_summary`：按规则 segment label 归纳 flat skill。
- `skillx_official`：调用官方 SkillX 流程，包含 SkillX 的抽取、过滤、聚类/合并路径。
- `skilllens_official`：调用官方 SkillLens `skilllens extract`。

外部仓库在：

```text
.external/baselines/SkillX
.external/baselines/SkillLens
```

## 代码接入点

主流程：

```text
env -> agent -> segmentation -> mining -> retrieval -> evaluation
```

关键文件：

- `src/trace2tower/run.py`：主实验流水线。
- `src/trace2tower/registry.py`：组件注册。
- `src/trace2tower/mining/baselines.py`：当前 baseline miner。
- `src/trace2tower/retrieval/score_based.py`：selector/retriever scoring。
- `scripts/evaluate_selectors.py`：离线 PUE/selector 评估，批量计算用 `numpy`。
- `scripts/collect_results.py`：收集 summary。
- `src/trace2tower/agents/llm_action_agent.py`：LLM action selector，读取在线注入的 `retrieved_skills`。
- `src/trace2tower/llm.py`：OpenAI-compatible chat client。

新增 Trace2Tower 主算法时建议：

```text
src/trace2tower/mining/trace2tower.py
src/trace2tower/mining/graph.py
src/trace2tower/mining/spectral.py
src/trace2tower/mining/tower.py
```

数值计算优先用 `numpy`，图结构可用 `networkx`。不要在 Python list 循环里手搓大矩阵、谱分解或批量相似度。

## 禁止事项

- 不要重新引入伪算法 miner。
- 不要把当前 `pue_full` 写成论文最终 PUE。
- 不要在真实 benchmark 失败时回落到 dummy 环境。
- 不要重新加入本地 `*_style` 近似 baseline。
- 不要让新算法覆盖已有 baseline 配置；新增独立 config。

## 最小下一步

如果下一个 AI 要继续推进，推荐顺序：

1. 先扩大 WebShop smoke 数量，例如 30-50 episodes。
2. 跑当前 baseline 并用 `collect_results.py` 汇总。
3. 用 `evaluate_selectors.py` 在每个 baseline 的 records 上跑 selector 评估。
4. 实现真正的 PUE estimator，保持相同 selector metrics 输出。
5. 用已接入的 `llm_action` 跑 no-skill、baseline skill 和 Trace2Tower skill 的同 agent 对比，从而回答 RQ3。
