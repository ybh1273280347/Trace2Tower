# 核心代码位置与数据流

这份文档说明当前实验代码的主要入口、数据产物和后续接算法的位置。

## 主数据流

```text
config json
  -> trace2tower.run
  -> env reset/step
  -> agent act
  -> records.jsonl
  -> rule segmenter
  -> segments.jsonl
  -> miner
  -> model.json
  -> retriever/selector
  -> retrieval.jsonl
  -> evaluator
  -> summary.json
```

## 运行入口

### 单配置运行

文件：

```text
src/trace2tower/run.py
```

作用：

- 读取 config。
- 构造 env/agent/segmenter/miner/retriever/evaluator。
- 跑 episode。
- 写出 `trajectories.jsonl`、`records.jsonl`、`segments.jsonl`、`model.json`、`retrieval.jsonl`、`summary.json`。

命令：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.run \
  --config configs/baseline_skillx_webshop.json
```

### 多配置运行

文件：

```text
scripts/run_config_matrix.py
```

作用：

- 顺序跑多个 config。
- 失败时直接中断，避免无声污染实验表。

## 组件注册

文件：

```text
src/trace2tower/registry.py
```

作用：

- `build_segmenter()`
- `build_miner()`
- `build_retriever()`
- `build_pipeline_bundle()`

新增算法时通常必须改这里。

## 环境与 agent

环境：

```text
src/trace2tower/env_factory.py
src/trace2tower/envs/webshop_adapter.py
src/trace2tower/envs/alfworld_adapter.py
```

当前约束：

- 真实环境缺依赖时直接报错。
- 不回落到 dummy/toy。

Agent：

```text
src/trace2tower/agent_factory.py
src/trace2tower/agents/random_agent.py
src/trace2tower/agents/llm_action_agent.py
src/trace2tower/llm.py
```

当前 agent：

- `smoke_random`：只验证环境和日志链路，不是论文 baseline。
- `llm_action`：OpenAI-compatible LLM action selector，会从 admissible actions 中选一个动作。

LLM 配置需要：

```bash
export OPENAI_API_KEY=...
# 可选：export OPENAI_BASE_URL=https://api.openai.com/v1
```

`llm_action` 会读取 `info["retrieved_skills"]`，把技能卡作为 prompt guidance。

## 轨迹切分

文件：

```text
src/trace2tower/segmentation/rule.py
```

作用：

- 把 step action 映射成 segment label。
- 写入 `metadata`：trajectory id、env、goal、success、reward、action。

后续如果做更细分事件，可在这里新增规则，或新增 `LLMSegmenter`。

## 当前 baseline miner

文件：

```text
src/trace2tower/mining/baselines.py
```

已有 miner：

- `NoSkillMiner`
- `RawTrajectoryMiner`
- `FlatSkillSummaryMiner`
- `SkillXOfficialMiner`
- `SkillLensOfficialMiner`

这些 miner 负责把 `segments` 转成统一模型：

```json
{
  "method": "...",
  "nodes": [],
  "edges": [],
  "skills": []
}
```

每个 skill 应尽量带：

```text
support
coverage
success_rate
failure_rate
avg_reward
recent_success_rate
recent_reward_lift
token_cost
labels
action_templates
```

这些字段会被 selector/PUE 使用。

## 后续 Trace2Tower 主算法位置

建议新增：

```text
src/trace2tower/mining/trace2tower.py
src/trace2tower/mining/graph.py
src/trace2tower/mining/spectral.py
src/trace2tower/mining/tower.py
```

职责建议：

- `graph.py`：构造 S/T/O 图和边权矩阵。
- `spectral.py`：谱分解、聚类、eigengap/stability。
- `tower.py`：low/mid/high skill tower 诱导。
- `trace2tower.py`：实现 `Trace2TowerMiner.mine()`，组装上述步骤。

数值计算用 `numpy`，图结构用 `networkx`。不要在 Python 循环里手搓大矩阵。

## 检索与 selector

文件：

```text
src/trace2tower/retrieval/score_based.py
```

当前支持：

- `none`
- `topk`
- `frequency`
- `success_rate`
- `similarity`
- `recent_reward_lift`
- `pue_full`
- `pue_no_cost`
- `pue_no_recent`
- `pue_no_similarity`

这里的 `_pue_score()` 是在线/单任务检索用 proxy。离线批量评估在 `scripts/evaluate_selectors.py`。

## 在线 skill 注入

文件：

```text
src/trace2tower/run.py
```

配置：

```json
"runtime": {
  "skill_model_path": "experiments/baseline_skillx_webshop/model.json"
}
```

流程：

```text
read skill model
  -> retriever.retrieve(model, current task state)
  -> info["retrieved_skills"]
  -> LLMActionAgent prompt
  -> env.step(action)
```

在线检索记录：

```text
experiments/.../deployment_retrieval.jsonl
```

LLM token 与 action parse fallback 会写入 step info，并进入 `summary.json`：

```text
avg_token_cost
llm_action_fallback_rate
```

新增真正 PUE selector 时：

1. 在 `score_based.py` 新增打分策略。
2. 在 `registry.build_retriever()` 注册名字。
3. 在 `scripts/evaluate_selectors.py` 增加同名批量矩阵计算。
4. 新建 config，不覆盖已有 baseline。

## 离线 PUE/selector 评估

文件：

```text
scripts/evaluate_selectors.py
```

作用：

- 读取已有 `records.jsonl`。
- 按时间顺序切成 train/future。
- 用 train segments 重新 mine skill model。
- 为 skills 和 future tasks 构造矩阵特征。
- 用 `numpy` 批量计算 selector score、future utility、top-k、correlation、AUC、regret。
- 输出：
  - `selector_metrics.json`
  - `selector_scores.jsonl`

这是当前回答 RQ1/RQ2/RQ4 的主要脚本。

## 结果汇总

文件：

```text
scripts/collect_results.py
```

作用：

- 读取多个实验目录或 config。
- 提取 `summary.json`、`model.json`、`retrieval.jsonl`。
- 汇总为 CSV/JSON。

输出示例：

```text
experiments/webshop_baseline_summary.csv
experiments/webshop_baseline_summary.json
```

## 配置位置

当前常用配置：

```text
configs/webshop_small.json
configs/alfworld_text_only.json
configs/baseline_no_skill_webshop.json
configs/baseline_raw_trajectory_webshop.json
configs/baseline_flat_skill_webshop.json
configs/baseline_skillx_webshop.json
configs/baseline_skilllens_webshop.json
```

改实验规模：

```json
"runtime": {
  "episodes": 50,
  "max_steps": 30,
  "output_dir": "experiments/..."
}
```

改方法：

```json
"miner": {"name": "..."},
"retriever": {"name": "...", "top_k": 3}
```

## 输出文件怎么看

单个实验目录：

```text
trajectories.jsonl  dataclass 轨迹
records.jsonl       原始 step + segments，最常用
segments.jsonl      miner 输入
model.json          miner 输出的技能结构
retrieval.jsonl     每个 task 检索到哪些 skill
summary.json        基础任务指标
```

selector 评估目录：

```text
selector_metrics.json  总体指标和 batch 指标
selector_scores.jsonl  每个 future task 下各 selector 选了哪些 skill
```

## 最常见下一步

1. 扩大 WebShop episodes。
2. 跑当前 baseline。
3. 用 `collect_results.py` 汇总。
4. 对每个 baseline 跑 `evaluate_selectors.py`。
5. 实现新的 PUE selector 或 Trace2Tower miner。
6. 接 skill-aware agent，开始看真实任务表现。
