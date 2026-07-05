# 实验代码骨架交接

## 当前状态

1. `run.py` 已支持：真实环境采集 -> 规则切分 -> 离线 miner -> retriever 演示 -> summary 输出。
2. `registry.py` 按配置选择 `agent`、`segmenter`、`miner`、`retriever`。
3. `RuleSegmenter` 输出带任务、环境、成败、奖励和动作的 segment metadata。
4. 旧伪算法 miner 已删除，不再生成伪算法产物。
5. 当前可运行 baseline miner：
   - `no_skill`
   - `raw_trajectory`
   - `flat_skill_summary`
   - `skillx_official`
   - `skilllens_official`
6. 当前可运行 retriever / selector：
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
7. 新增离线评估脚本：
   - `scripts/collect_results.py`
   - `scripts/evaluate_selectors.py`
8. 已接入 LLM action agent：
   - `agent.name=llm_action`
   - OpenAI-compatible `/chat/completions`
   - 通过 `runtime.skill_model_path` 加载已有 `model.json` 做在线 skill 注入
   - 当前 API/model 与 smoke 结果见 `09_llm_integration_status.md`

## 明确边界

- `trace2tower` 主算法还没有实现；不要把 baseline 近似实现写成 Trace2Tower 结果。
- SkillX / SkillLens baseline 当前接入官方仓库；不要重新加入本地 style/proxy baseline。
- 不要在真实 benchmark 失败时回落到 dummy 环境。
- LLM 配置只读 `.env` 里的 `LLM_*`，缺变量时应直接失败，不要静默 fallback。

## 下一步接 Trace2Tower 算法

建议新增：

```text
src/trace2tower/mining/trace2tower.py
```

然后在 `registry.build_miner()` 注册：

```text
"trace2tower": Trace2TowerMiner(...)
```

如果需要 embedding、图构造、谱分解、层级诱导，优先拆到 `src/trace2tower/mining/` 下的小模块；数值计算用 `numpy`，图结构可用 `networkx`。

## LLM 主流程

无技能 LLM：

```text
configs/llm_no_skill_webshop.json
```

加载已有 SkillX official skill model：

```text
configs/llm_skillx_webshop.json
```

在线 skill 注入由 `run.py` 完成：

```text
runtime.skill_model_path -> read model.json -> retriever.retrieve(...) -> info["retrieved_skills"] -> LLMActionAgent
```

在线检索记录输出到：

```text
deployment_retrieval.jsonl
```

## 输出契约

真实 miner 至少返回：

```json
{
  "method": "trace2tower",
  "nodes": [],
  "edges": [],
  "skills": []
}
```

单个 skill 建议保持：

```json
{
  "skill_id": "...",
  "name": "...",
  "granularity": "low|mid|high|flat",
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
