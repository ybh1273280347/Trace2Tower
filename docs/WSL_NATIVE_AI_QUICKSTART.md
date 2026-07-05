# WSL 原生 AI 辅助上手说明

这份给你后续在 WSL/Ubuntu 里直接开 AI 辅助时使用。

## 进入项目

```bash
cd ~/papers/Trace2Tower
```

统一 Python 环境：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python --version
```

预期是 Python 3.8.x。

## 每次让 AI 接手时先让它读

推荐提示词：

```text
先阅读 docs/ai_assist/06_research_experiment_brief_for_ai.md、
docs/ai_assist/05_experiment_skeleton_handoff.md、
docs/ai_assist/09_llm_integration_status.md、
README.md，然后继续当前任务。
不要实现伪算法，不要把当前 pue_full 当最终 PUE。
```

如果是接算法：

```text
Trace2Tower 主算法还没有实现。请新增真实 miner/retriever/selector，
保持现有 baseline 可运行，并用 numpy/networkx 做批量数值计算。
```

## 常用验证命令

编译检查：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m compileall src scripts
```

跑 WebShop 无技能 smoke：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.run \
  --config configs/webshop_small.json
```

跑 baseline 矩阵：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/run_config_matrix.py \
  configs/baseline_no_skill_webshop.json \
  configs/baseline_raw_trajectory_webshop.json \
  configs/baseline_flat_skill_webshop.json \
  configs/baseline_trace2skill_webshop.json \
  configs/baseline_skillx_webshop.json \
  configs/baseline_skilllens_webshop.json
```

汇总结果：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/collect_results.py \
  configs/baseline_no_skill_webshop.json \
  configs/baseline_raw_trajectory_webshop.json \
  configs/baseline_flat_skill_webshop.json \
  configs/baseline_trace2skill_webshop.json \
  configs/baseline_skillx_webshop.json \
  configs/baseline_skilllens_webshop.json \
  --output-json experiments/webshop_baseline_summary.json \
  --output-csv experiments/webshop_baseline_summary.csv
```

离线评估 selector/PUE：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/evaluate_selectors.py \
  --config configs/baseline_skillx_webshop.json \
  --records experiments/baseline_skillx_webshop/records.jsonl \
  --output-dir experiments/baseline_skillx_webshop/selectors
```

LLM action agent smoke：

```bash
export OPENAI_API_KEY=...
# 可选：export OPENAI_BASE_URL=https://api.openai.com/v1
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.run \
  --config configs/llm_no_skill_webshop.json
```

如果使用当前测试过的 proxy：

```bash
export OPENAI_BASE_URL=https://api.zhizengzeng.com/v1
```

注意当前 proxy 可用 model id 是 `gpt-4o-mini`，不是 `openai/gpt-4o-mini`。

LLM + 已有 SkillX-style skill 注入：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.run \
  --config configs/llm_skillx_webshop.json
```

## 当前能看的结果文件

- `experiments/webshop_baseline_summary.csv`
- `experiments/webshop_baseline_summary.json`
- `experiments/baseline_skillx_webshop/selectors/selector_metrics.json`
- `experiments/*/summary.json`
- `experiments/*/model.json`
- `experiments/*/retrieval.jsonl`
- `experiments/*/deployment_retrieval.jsonl`
- `experiments/llm_smoke_summary.csv`

`experiments/` 已被 git ignore，适合反复跑。

## 新增算法时改哪里

新增 miner：

```text
src/trace2tower/mining/your_method.py
src/trace2tower/mining/__init__.py
src/trace2tower/registry.py
configs/your_method_webshop.json
```

新增 selector/retriever：

```text
src/trace2tower/retrieval/score_based.py
src/trace2tower/registry.py
scripts/evaluate_selectors.py
```

新增评估汇总：

```text
src/trace2tower/evaluation/
scripts/collect_results.py
```

## 研究问题记忆版

当前重点不是“先把 Trace2Tower 算法写满”，而是让实验能回答：

1. PUE 预测未来效用准不准？
2. PUE 是否比频率、成功率、相似度更好？
3. PUE 是否提升长期任务表现和技能使用效率？
4. PUE 哪些因素最关键？

## 注意

- WebShop/ALFWorld 尽量从 WSL 里跑，不要用 Windows PowerShell 直接跑 `.tools/bin/micromamba`。
- 当前 baseline 的外在 reward 一样是正常的，因为 `smoke_random` 不读取 skill；先用它验证模型产物、检索和 selector 评估链路。
- 当前已经有 `llm_action` agent，可以用它跑 no-skill 和 skill-injected 两条主流程 smoke，让检索到的 skill 真正影响动作。
- 不要把 1 episode LLM smoke 当论文结果；它只说明 API、prompt、动作选择和日志链路通了。
- git 仓库已经初始化，当前配置是 LF 换行；大数据、环境和实验产物已 ignore。
