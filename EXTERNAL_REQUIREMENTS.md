# External Requirements

Large benchmark assets and upstream baseline repositories are intentionally not tracked by git.
Prepare them under `.external/` before running the full experiments.

## Required Local Layout

```text
.external/
  webshop/
    data/items_ins_v2_1000.json
    search_engine/resources_1k/
    search_engine/indexes_1k/
    web_agent_site/
  alfworld/
    json_2.1.1/
    logic/alfred.pddl
    logic/alfred.twl2
    detectors/mrcnn_alfred_objects_sep13_004.pth
  baselines/
    SkillX/
    SkillLens/
```

## Python Environments

Main Trace2Tower environment:

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m pip install -e .
```

Official baseline environment:

```text
.envs/baselines-py312/
```

The baseline configs expect this Python executable:

```text
.envs/baselines-py312/bin/python
```

## LLM Variables

Create `.env` locally. Do not commit it.

```text
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
LLM_EMBEDDING_MODEL=qwen3-embedding-8b
```

## Smoke Commands

WebShop Trace2Tower suite:

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python scripts/run_experiment_suite.py \
  --output-root experiments/trace2tower_suite_smoke \
  --miner-configs configs/trace2tower_webshop.json \
  --deployment-config configs/llm_trace2tower_webshop.json \
  --segments experiments/baseline_no_skill_webshop/segments.jsonl \
  --records experiments/baseline_no_skill_webshop/records.jsonl \
  --exclude-no-skill
```

ALFWorld environment reset check:

```bash
.tools/bin/micromamba run -p .envs/trace2tower-py38 python - <<'PY'
from trace2tower.factories.env import build_env
env = build_env(
    "alfworld",
    "text",
    alfworld_config_path="configs/alfworld/base_config.yaml",
    alfworld_data_dir=".external/alfworld",
)
obs, info = env.reset()
print(obs[:120])
print(info["admissible_actions"][:5])
PY
```
