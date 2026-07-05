# Trace2Tower

面向 ALFWorld / WebShop 的实验骨架。当前重点是让真实环境、轨迹采集、切分、官方 baseline 接入、检索和评测闭环跑通；Trace2Tower 主算法还没有实现。

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

- `no_skill`、`raw_trajectory`、`flat_skill_summary`：本地基础对照。
- `skillx_official`：调用 `.external/baselines/SkillX` 官方代码；embedding 使用 `LLM_BASE_URL` 同源服务和 `LLM_EMBEDDING_MODEL`。
- `skilllens_official`：调用 `.external/baselines/SkillLens` 官方 `skilllens extract`。

## 快速检查

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m compileall src scripts
```

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.run \
  --config configs/baseline_skillx_webshop.json
```

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.run \
  --config configs/baseline_skilllens_webshop.json
```

## 输出文件

每次运行会在 `runtime.output_dir` 下写出 `trajectories.jsonl`、`records.jsonl`、`segments.jsonl`、`model.json`、`retrieval.jsonl`、`deployment_retrieval.jsonl` 和 `summary.json`。Official baseline 的原始输入、日志和官方输出保存在对应实验目录下的 miner 子目录。
