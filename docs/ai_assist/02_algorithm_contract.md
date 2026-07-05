# 算法接入契约

当前仓库不再保留伪算法 miner。新算法应以可运行组件接入，即使第一版很简单，也要产生可解释、可落盘、可比较的输出。

## Miner 输出

```json
{
  "method": "method_name",
  "description": "...",
  "nodes": [],
  "edges": [],
  "skills": []
}
```

`skills` 至少包含：

```json
{
  "skill_id": "...",
  "name": "...",
  "granularity": "flat|trajectory|planning|functional|atomic|policy|strategy|procedure|primitive|low|mid|high",
  "members": ["segment_id"],
  "content": "...",
  "embedding_text": "...",
  "metadata": {
    "support": 3,
    "success_rate": 0.5,
    "avg_reward": 0.4,
    "token_cost": 80
  }
}
```

## 接入位置

1. 新增 miner：`src/trace2tower/mining/xxx.py`
2. 在 `src/trace2tower/mining/__init__.py` 导出。
3. 在 `registry.build_miner()` 注册配置名。
4. 新建单独 config，不覆盖已有 baseline config。
5. 跑 `compileall` 和一个小规模真实环境 smoke。

## 当前可运行 baseline

- `no_skill`
- `raw_trajectory`
- `flat_skill_summary`
- `skillx_official`
- `skilllens_official`

`skillx_official` 和 `skilllens_official` 会调用 `.external/baselines/` 下的官方仓库。不要重新加入本地 style/proxy baseline。
