# 如何接入一个新算法

现在先不选具体 baseline。接入平台的方式是：先把候选方法作为一个独立层注册进来，能跑小规模 smoke，再决定是否纳入正式对比。

## 推荐顺序

1. 先确定方法属于哪一层：agent、segmentation、mining、retrieval、evaluation。
2. 新增一个类，继承该层的 base class。
3. 在对应 factory 或 registry 里加一个名字。
4. 新建一个 config，不改旧 config。
5. 先跑 1 到 3 个 episode，确认输出文件结构稳定。
6. 再补正式指标和批量实验脚本。

## 常见插入点

| 方法类型 | 位置 | 输出 |
|---|---|---|
| prompt / ReAct / LLM agent | `src/trace2tower/agents/` | action |
| 轨迹切分方法 | `src/trace2tower/segmentation/` | segments |
| Trace2Tower 技能诱导 | `src/trace2tower/mining/` | graph / skills |
| 技能检索和注入 | `src/trace2tower/retrieval/` | retrieved skills |
| 新评测指标 | `src/trace2tower/evaluation/` | summary 字段 |

## 配置习惯

每个新想法都单独建配置，例如：

```json
{
  "env": {"name": "webshop", "mode": "text", "num_products": 1000},
  "agent": {"name": "candidate_method"},
  "segmenter": {"name": "rule"},
  "miner": {"name": "trace2tower_candidate"},
  "retriever": {"name": "topk"},
  "evaluator": {"metrics": ["success_rate", "avg_reward"]},
  "runtime": {
    "episodes": 3,
    "max_steps": 30,
    "output_dir": "experiments/candidate_method_smoke"
  }
}
```

这样后续调研到新论文时，只需要新增一个模块和一份配置，不需要推翻已有实验平台。
