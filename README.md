# Trace2Tower

面向 ALFWorld / WebShop 的 Trace2Tower 实验平台。当前代码已经把 Trace2Tower 主算法落在 `src/trace2tower/mining/`：

- `trace2tower.py`：算法入口，串起 embedding、图构造、谱分解和技能塔诱导。
- `graph.py`：构造语义相似性、时序转移、成败一致性三类边权，并生成成功/失败对比图。
- `spectral.py`：执行 contrastive EigenTrace 谱分解和中层技能聚类。
- `tower.py`：诱导 low / mid / high 三层技能塔。
- `refinement.py`：根据部署反馈执行 downweight、split、merge、promote。

## 环境变量

根目录 `.env` 是 LLM 配置唯一事实源，并已被 `.gitignore` 忽略：

```bash
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
LLM_EMBEDDING_MODEL=...
```

代码不会静默 fallback 到默认模型。缺少这些变量时，LLM agent 或 official baseline 会直接报错。`OPENAI_API_KEY` / `OPENAI_BASE_URL` 只作为传给外部官方库的派生环境变量。

## Baseline 状态

- `trace2tower`：Transition-Aware Contrastive EigenTrace 主算法。
- `no_skill`：无技能控制组。
- `skillx_official`：调用 `.external/baselines/SkillX` 官方代码；embedding 使用 `LLM_BASE_URL` 同源服务和 `LLM_EMBEDDING_MODEL`。
- `skilllens_official`：调用 `.external/baselines/SkillLens` 官方 `skilllens extract`。

## 快速检查

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m compileall src scripts
```

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.runtime.run \
  --config configs/baseline_skillx_webshop.json
```

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.runtime.run \
  --config configs/baseline_skilllens_webshop.json
```

## 输出文件

每次运行会在 `runtime.output_dir` 下写出 `trajectories.jsonl`、`records.jsonl`、`segments.jsonl`、`model.json`、`retrieval.jsonl`、`deployment_retrieval.jsonl` 和 `summary.json`。Official baseline 的原始输入、日志和官方输出保存在对应实验目录下的 miner 子目录。

## RQ 实验流水线

已有共享训练 segments 后，可以单独跑多个 miner：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/mine_skill_models.py \
  configs/baseline_skillx_webshop.json \
  configs/baseline_skilllens_webshop.json \
  --segments experiments/baseline_no_skill_webshop/segments.jsonl \
  --records experiments/baseline_no_skill_webshop/records.jsonl \
  --output-root experiments/current_baselines/models
```

同一 LLM agent 下部署对比：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/deploy_skill_models.py \
  --config configs/llm_skillx_webshop.json \
  --output-root experiments/current_baselines/deployment \
  --include-no-skill \
  --models \
  skillx=experiments/current_baselines/models/baseline_skillx_webshop/model.json \
  skilllens=experiments/current_baselines/models/baseline_skilllens_webshop/model.json
```

pandas 生成对比表：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/analyze_experiment_table.py \
  experiments/current_baselines \
  --output-dir experiments/current_baselines/analysis
```

想把采集共享训练集、采矿、selector 评估、部署和分析串起来，用：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/run_experiment_suite.py \
  --train-config configs/baseline_no_skill_webshop.json \
  --miner-configs configs/trace2tower_webshop.json configs/baseline_skillx_webshop.json configs/baseline_skilllens_webshop.json \
  --deployment-config configs/llm_trace2tower_webshop.json \
  --output-root experiments/trace2tower_suite
```

suite 会生成：

- `models/`：同一批 segments 下的各 miner `model.json`。
- `selectors/`：RQ1/RQ2/RQ4 的 PUE 与 heuristic selector 指标。
- `deployment/`：RQ3 的同一 agent/env 部署对比与 `refined_model.json`。
- `analysis/`：pandas 汇总表和图。
