# LLM 主流程接入状态

这份文档记录当前 LLM 接入进度、可运行命令和 smoke 结果。它只说明工程状态，不代表论文实验结论。

## 当前已经接通

- `agent.name=llm_action` 已可用。
- LLM client 使用 OpenAI-compatible `/chat/completions`，不依赖 OpenAI SDK。
- `runtime.skill_model_path` 可加载已有 `model.json`，在在线执行每一步前检索 skill。
- 检索到的 skill 会写入 `info["retrieved_skills"]`，并进入 `LLMActionAgent` prompt。
- 每一步在线注入记录写到 `deployment_retrieval.jsonl`。
- `summary.json` 会记录 `avg_token_cost` 和 `llm_action_fallback_rate`。

## 关键代码位置

```text
src/trace2tower/llm.py
src/trace2tower/agents/llm_action_agent.py
src/trace2tower/agent_factory.py
src/trace2tower/run.py
src/trace2tower/evaluation/evaluator.py
```

对应配置：

```text
configs/llm_no_skill_webshop.json
configs/llm_skillx_webshop.json
```

## API 配置

根目录 `.env` 是唯一事实源：

```bash
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
LLM_EMBEDDING_MODEL=...
```

代码不会静默 fallback 到默认模型。外部官方库需要的 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 由运行时从 `LLM_API_KEY` / `LLM_BASE_URL` 派生。

## Smoke 命令

无技能 LLM：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.run \
  --config configs/llm_no_skill_webshop.json
```

SkillX official skill 注入：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m trace2tower.run \
  --config configs/llm_skillx_webshop.json
```

汇总两条 LLM smoke：

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/collect_results.py \
  configs/llm_no_skill_webshop.json \
  configs/llm_skillx_webshop.json \
  --output-csv experiments/llm_smoke_summary.csv
```

## 本次 smoke 结果

当前只跑了 1 episode、`max_steps=10`，只能说明主流程打通。

```text
llm_no_skill_webshop:
  success_rate = 1.0
  avg_steps = 3.0
  avg_reward = 0.7777777777777778
  avg_token_cost = 1571.0
  llm_action_fallback_rate = 0.0

llm_skillx_webshop:
  success_rate = 1.0
  avg_steps = 3.0
  avg_reward = 0.7777777777777778
  avg_token_cost = 2109.0
  llm_action_fallback_rate = 0.0
  deployment_model_method = skillx_official
  deployment_skill_count = 11
  avg_deployment_retrieved_skills = 3.0
```

结果文件：

```text
experiments/llm_no_skill_webshop/summary.json
experiments/llm_skillx_webshop/summary.json
experiments/llm_skillx_webshop/deployment_retrieval.jsonl
experiments/llm_smoke_summary.csv
```

## 还能说明什么

- 可以说明：LLM API、环境、动作选择、skill 检索、skill 注入、在线记录和 summary 汇总已经连起来。
- 不能说明：SkillX official 或未来 Trace2Tower + PUE 已经优于 no-skill。
- 下一步要扩大 episode，并把 no-skill、baseline skills、Trace2Tower skills 放到同一 LLM agent 下比较。
