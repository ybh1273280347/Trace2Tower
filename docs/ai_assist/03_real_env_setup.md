# 真实环境安装与验证

## 统一环境

实验统一使用项目内 Python 3.8 环境：

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python --version
```

不要用系统 Python 3.12 跑正式实验。ALFWorld / TextWorld / WebShop 的依赖比较旧，混用新 Python 很容易出现隐性兼容问题。

## 已接入的数据

- ALFWorld：`.external/alfworld`
- WebShop small：`.external/webshop/data/items_shuffle_1000.json`
- WebShop 目标文件：`.external/webshop/data/items_human_ins.json`
- WebShop 搜索索引：`.external/webshop/search_engine/indexes_1k`

## 验证命令

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python -m compileall src
```

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python - <<'PY'
from trace2tower.envs.alfworld_adapter import build_alfworld_env
env = build_alfworld_env(config_path="configs/alfworld/base_config.yaml", data_dir=".external/alfworld")
obs, info = env.reset()
print(obs[:300])
print(info["admissible_actions"][:10])
PY
```

```bash
cd ~/papers/Trace2Tower
.tools/bin/micromamba run -p .envs/trace2tower-py38 python - <<'PY'
from trace2tower.envs.webshop_adapter import build_webshop_env
env = build_webshop_env(num_products=1000, webshop_root=".external/webshop")
obs, info = env.reset()
print(obs[:300])
print(info["admissible_actions"][:10])
PY
```
