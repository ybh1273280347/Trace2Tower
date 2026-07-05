from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from trace2tower.config import load_config
from trace2tower.registry import build_miner
from trace2tower.text import action_template, tokenize


DEFAULT_SELECTORS = [
    "frequency",
    "success_rate",
    "similarity",
    "recent_reward_lift",
    "pue_no_cost",
    "pue_no_recent",
    "pue_no_similarity",
    "pue_full",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment config whose miner should be evaluated.")
    parser.add_argument("--records", required=True, help="records.jsonl produced by trace2tower.run.")
    parser.add_argument("--output-dir", required=True, help="Directory for selector metrics.")
    parser.add_argument("--future-ratio", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--selectors", nargs="*", default=DEFAULT_SELECTORS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = _read_jsonl(Path(args.records))
    if len(records) < 2:
        raise ValueError("At least two records are required for train/future selector evaluation.")

    # 离线评估按时间顺序切分：前段诱导技能，后段模拟“未来任务”。
    train_records, future_records = _split_records(records, future_ratio=args.future_ratio)
    config = load_config(args.config)
    miner_config = dict(config.miner_config)
    miner_config.setdefault("runtime_output_dir", str(Path(args.output_dir) / "miner"))
    miner = build_miner(config.miner_name, miner_config)
    train_segments = [
        segment
        for record in train_records
        for segment in record.get("segments", [])
    ]
    model = miner.mine(train_segments)
    skills = model.get("skills", [])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skills:
        result = _empty_result(config, train_records, future_records)
        _write_json(output_dir / "selector_metrics.json", result)
        _write_jsonl(output_dir / "selector_scores.jsonl", [])
        return

    # 先把 skill 和 future task 投到同一个离散特征空间，后续全部用 numpy 矩阵批量算。
    feature_space = _build_feature_space(skills, future_records)
    skill_features = _build_skill_features(skills, feature_space)
    task_features = _build_task_features(future_records, feature_space)
    utility = _future_utility_matrix(skill_features, task_features)
    selector_scores = {
        selector: _selector_score_matrix(selector, skill_features, task_features)
        for selector in args.selectors
    }

    metrics = {
        selector: _selector_metrics(
            scores=scores,
            utility=utility,
            token_cost=skill_features["token_cost"],
            top_k=args.top_k,
        )
        for selector, scores in selector_scores.items()
    }
    batches = _batch_metrics(
        selector_scores=selector_scores,
        utility=utility,
        token_cost=skill_features["token_cost"],
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    score_records = _score_records(
        future_records=future_records,
        skills=skills,
        selector_scores=selector_scores,
        utility=utility,
        top_k=args.top_k,
    )
    result = {
        "config": str(args.config),
        "records": str(args.records),
        "miner": config.miner_name,
        "model_method": model.get("method"),
        "split": {
            "train_records": len(train_records),
            "future_records": len(future_records),
            "train_segments": len(train_segments),
            "skill_count": len(skills),
            "future_ratio": args.future_ratio,
        },
        "top_k": args.top_k,
        "selectors": metrics,
        "batches": batches,
    }
    _write_json(output_dir / "selector_metrics.json", result)
    _write_jsonl(output_dir / "selector_scores.jsonl", score_records)


def _split_records(records: list[dict[str, Any]], future_ratio: float) -> tuple[list[dict], list[dict]]:
    if not 0 < future_ratio < 1:
        raise ValueError("future-ratio must be between 0 and 1.")
    split_at = max(1, int(round(len(records) * (1.0 - future_ratio))))
    split_at = min(split_at, len(records) - 1)
    return records[:split_at], records[split_at:]


def _build_feature_space(skills: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, list[str]]:
    labels: set[str] = set()
    templates: set[str] = set()
    vocab: set[str] = set()

    for skill in skills:
        metadata = skill.get("metadata", {})
        labels.update(str(item.get("value", "")) for item in metadata.get("labels", []))
        templates.update(str(item.get("value", "")) for item in metadata.get("action_templates", []))
        vocab.update(tokenize(_skill_text(skill)))

    for record in records:
        for segment in record.get("segments", []):
            labels.add(str(segment.get("label", "Unknown")))
            templates.add(action_template(_segment_action(segment)))
        vocab.update(tokenize(_task_text(record)))

    return {
        "labels": sorted(item for item in labels if item),
        "templates": sorted(item for item in templates if item),
        "vocab": sorted(item for item in vocab if item),
    }


def _build_skill_features(skills: list[dict[str, Any]], space: dict[str, list[str]]) -> dict[str, np.ndarray]:
    label_index = {value: index for index, value in enumerate(space["labels"])}
    template_index = {value: index for index, value in enumerate(space["templates"])}
    vocab_index = {value: index for index, value in enumerate(space["vocab"])}
    n = len(skills)

    labels = np.zeros((n, len(label_index)), dtype=np.float32)
    templates = np.zeros((n, len(template_index)), dtype=np.float32)
    text = np.zeros((n, len(vocab_index)), dtype=np.float32)
    support = np.zeros(n, dtype=np.float32)
    success_rate = np.zeros(n, dtype=np.float32)
    avg_reward = np.zeros(n, dtype=np.float32)
    coverage = np.zeros(n, dtype=np.float32)
    recent_success_rate = np.zeros(n, dtype=np.float32)
    recent_reward_lift = np.zeros(n, dtype=np.float32)
    failure_rate = np.zeros(n, dtype=np.float32)
    token_cost = np.zeros(n, dtype=np.float32)

    for row, skill in enumerate(skills):
        metadata = skill.get("metadata", {})
        for item in metadata.get("labels", []):
            value = str(item.get("value", ""))
            if value in label_index:
                labels[row, label_index[value]] = float(item.get("count", 1.0))
        for item in metadata.get("action_templates", []):
            value = str(item.get("value", ""))
            if value in template_index:
                templates[row, template_index[value]] = float(item.get("count", 1.0))
        for token, count in Counter(tokenize(_skill_text(skill))).items():
            if token in vocab_index:
                text[row, vocab_index[token]] = float(count)

        support[row] = float(metadata.get("support", len(skill.get("members", []))) or 0.0)
        success_rate[row] = float(metadata.get("success_rate", 0.0) or 0.0)
        avg_reward[row] = float(metadata.get("avg_reward", 0.0) or 0.0)
        coverage[row] = float(metadata.get("coverage", 0.0) or 0.0)
        recent_success_rate[row] = float(metadata.get("recent_success_rate", success_rate[row]) or 0.0)
        recent_reward_lift[row] = float(metadata.get("recent_reward_lift", 0.0) or 0.0)
        failure_rate[row] = float(metadata.get("failure_rate", 1.0 - success_rate[row]) or 0.0)
        token_cost[row] = float(metadata.get("token_cost", 0.0) or 0.0)

    return {
        "labels": _row_normalize(labels),
        "templates": _row_normalize(templates),
        "text": _row_normalize(text),
        "support": support,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "coverage": coverage,
        "recent_success_rate": recent_success_rate,
        "recent_reward_lift": recent_reward_lift,
        "failure_rate": failure_rate,
        "token_cost": token_cost,
    }


def _build_task_features(records: list[dict[str, Any]], space: dict[str, list[str]]) -> dict[str, np.ndarray]:
    label_index = {value: index for index, value in enumerate(space["labels"])}
    template_index = {value: index for index, value in enumerate(space["templates"])}
    vocab_index = {value: index for index, value in enumerate(space["vocab"])}
    m = len(records)

    labels = np.zeros((m, len(label_index)), dtype=np.float32)
    templates = np.zeros((m, len(template_index)), dtype=np.float32)
    text = np.zeros((m, len(vocab_index)), dtype=np.float32)
    reward = np.zeros(m, dtype=np.float32)
    success = np.zeros(m, dtype=np.float32)
    step_efficiency = np.zeros(m, dtype=np.float32)

    max_steps = max((len(record.get("steps", [])) for record in records), default=1) or 1
    for row, record in enumerate(records):
        for segment in record.get("segments", []):
            label = str(segment.get("label", "Unknown"))
            if label in label_index:
                labels[row, label_index[label]] += 1.0
            template = action_template(_segment_action(segment))
            if template in template_index:
                templates[row, template_index[template]] += 1.0
        for token, count in Counter(tokenize(_task_text(record))).items():
            if token in vocab_index:
                text[row, vocab_index[token]] = float(count)

        reward[row] = float(record.get("final_reward", 0.0) or 0.0)
        success[row] = 1.0 if record.get("success") else 0.0
        step_efficiency[row] = 1.0 - (len(record.get("steps", [])) / max_steps)

    return {
        "labels": _row_normalize(labels),
        "templates": _row_normalize(templates),
        "text": _row_normalize(text),
        "reward": reward,
        "success": success,
        "step_efficiency": step_efficiency,
    }


def _future_utility_matrix(skill_features: dict[str, np.ndarray], task_features: dict[str, np.ndarray]) -> np.ndarray:
    # 当前 utility 是 proxy：技能是否匹配未来任务 * 未来任务价值 - 技能上下文成本。
    # 后续接真实 deployment 后，可以替换成 success lift / reward lift / step saving。
    label_match = skill_features["labels"] @ task_features["labels"].T
    template_match = skill_features["templates"] @ task_features["templates"].T
    match = 0.55 * label_match + 0.45 * template_match
    future_value = (
        0.5 * task_features["reward"]
        + 0.35 * task_features["success"]
        + 0.15 * task_features["step_efficiency"]
    )
    cost_penalty = skill_features["token_cost"] / (skill_features["token_cost"] + 100.0)
    return match * future_value[np.newaxis, :] - 0.08 * cost_penalty[:, np.newaxis]


def _selector_score_matrix(
    selector: str,
    skill_features: dict[str, np.ndarray],
    task_features: dict[str, np.ndarray],
) -> np.ndarray:
    task_count = task_features["text"].shape[0]
    if selector == "frequency":
        return np.repeat(skill_features["support"][:, np.newaxis], task_count, axis=1)
    if selector == "success_rate":
        return np.repeat(skill_features["success_rate"][:, np.newaxis], task_count, axis=1)
    if selector == "recent_reward_lift":
        return np.repeat(skill_features["recent_reward_lift"][:, np.newaxis], task_count, axis=1)
    if selector == "similarity":
        return skill_features["text"] @ task_features["text"].T
    if selector == "pue_no_cost":
        return _pue_matrix(skill_features, task_features, use_cost=False)
    if selector == "pue_no_recent":
        return _pue_matrix(skill_features, task_features, use_recent=False)
    if selector == "pue_no_similarity":
        return _pue_matrix(skill_features, task_features, use_similarity=False)
    if selector in {"pue", "pue_full"}:
        return _pue_matrix(skill_features, task_features)
    raise ValueError(f"Unsupported selector: {selector}")


def _pue_matrix(
    skill_features: dict[str, np.ndarray],
    task_features: dict[str, np.ndarray],
    *,
    use_cost: bool = True,
    use_recent: bool = True,
    use_similarity: bool = True,
) -> np.ndarray:
    task_count = task_features["text"].shape[0]
    similarity = skill_features["text"] @ task_features["text"].T if use_similarity else 0.0
    cost = skill_features["token_cost"] / (skill_features["token_cost"] + 100.0) if use_cost else 0.0
    recent_success = skill_features["recent_success_rate"] if use_recent else 0.0
    recent_lift = skill_features["recent_reward_lift"] if use_recent else 0.0

    # 这里是可解释 PUE proxy，不是最终论文算法；保留它是为了和启发式 selector 同表比较。
    base = (
        1.2 * recent_success
        + 0.9 * skill_features["avg_reward"]
        + 0.6 * skill_features["coverage"]
        + 0.4 * recent_lift
        + 0.3 * np.minimum(1.0, skill_features["support"] / 20.0)
        - 0.8 * skill_features["failure_rate"]
        - 0.5 * cost
    )
    return np.repeat(base[:, np.newaxis], task_count, axis=1) + 0.5 * similarity


def _selector_metrics(
    *,
    scores: np.ndarray,
    utility: np.ndarray,
    token_cost: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    # selector 的核心不是整体分数高，而是 top-k 选出的技能是否接近 oracle utility 选择。
    selected = _topk_indices(scores, top_k)
    oracle = _topk_indices(utility, top_k)
    selected_utility = _mean_selected(utility, selected)
    oracle_utility = _mean_selected(utility, oracle)
    selected_cost = _mean_selected(token_cost[:, np.newaxis], selected)
    return {
        "future_utility_pearson": _pearson(scores.ravel(), utility.ravel()),
        "future_utility_spearman": _spearman(scores.ravel(), utility.ravel()),
        "positive_utility_auc": _auc(scores.ravel(), (utility.ravel() > 0).astype(np.float32)),
        "avg_selected_utility": float(np.mean(selected_utility)),
        "avg_oracle_utility": float(np.mean(oracle_utility)),
        "regret": float(np.mean(oracle_utility - selected_utility)),
        "positive_selection_rate": float(np.mean(selected_utility > 0)),
        "avg_selected_token_cost": float(np.mean(selected_cost)),
    }


def _batch_metrics(
    *,
    selector_scores: dict[str, np.ndarray],
    utility: np.ndarray,
    token_cost: np.ndarray,
    batch_size: int,
    top_k: int,
) -> list[dict[str, Any]]:
    # batch 指标先服务离线趋势分析；真实长期部署时可替换为实际 deployment batch。
    task_count = utility.shape[1]
    batches = []
    for start in range(0, task_count, max(1, batch_size)):
        end = min(task_count, start + max(1, batch_size))
        batch: dict[str, Any] = {
            "batch_index": len(batches),
            "task_start": start,
            "task_end": end,
            "task_count": end - start,
            "selectors": {},
        }
        for selector, scores in selector_scores.items():
            batch["selectors"][selector] = _selector_metrics(
                scores=scores[:, start:end],
                utility=utility[:, start:end],
                token_cost=token_cost,
                top_k=top_k,
            )
        batches.append(batch)
    return batches


def _score_records(
    *,
    future_records: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    selector_scores: dict[str, np.ndarray],
    utility: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    output = []
    for task_index, record in enumerate(future_records):
        selectors = {}
        for selector, scores in selector_scores.items():
            skill_indices = _topk_indices(scores[:, task_index : task_index + 1], top_k).ravel()
            selectors[selector] = [
                {
                    "skill_id": skills[int(index)].get("skill_id"),
                    "score": float(scores[int(index), task_index]),
                    "future_utility": float(utility[int(index), task_index]),
                }
                for index in skill_indices
            ]
        output.append(
            {
                "task_id": record.get("task_id"),
                "goal": record.get("goal", ""),
                "reward": record.get("final_reward", 0.0),
                "success": bool(record.get("success")),
                "selectors": selectors,
            }
        )
    return output


def _topk_indices(matrix: np.ndarray, top_k: int) -> np.ndarray:
    # argpartition 比完整排序更适合大技能库，只取每个 future task 的 top-k。
    k = min(max(1, top_k), matrix.shape[0])
    partition = np.argpartition(-matrix, kth=k - 1, axis=0)[:k]
    values = np.take_along_axis(matrix, partition, axis=0)
    order = np.argsort(-values, axis=0)
    return np.take_along_axis(partition, order, axis=0)


def _mean_selected(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if values.shape[1] == 1 and indices.shape[1] > 1:
        values = np.repeat(values, indices.shape[1], axis=1)
    selected = np.take_along_axis(values, indices, axis=0)
    return np.mean(selected, axis=0)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rank(left), _rank(right))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(values.size, dtype=np.float32)
    return ranks


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels > 0
    positive_count = int(np.sum(positives))
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return 0.0
    ranks = _rank(scores) + 1.0
    positive_rank_sum = float(np.sum(ranks[positives]))
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _empty_result(config: Any, train_records: list[dict], future_records: list[dict]) -> dict[str, Any]:
    return {
        "miner": config.miner_name,
        "model_method": "no_skill",
        "split": {
            "train_records": len(train_records),
            "future_records": len(future_records),
            "train_segments": 0,
            "skill_count": 0,
        },
        "selectors": {},
        "batches": [],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _skill_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(skill.get("name", "")),
            str(skill.get("granularity", "")),
            str(skill.get("content", "")),
            str(skill.get("embedding_text", "")),
        ]
    )


def _task_text(record: dict[str, Any]) -> str:
    pieces = [
        str(record.get("goal", "")),
        str(record.get("env", "")),
    ]
    for segment in record.get("segments", [])[:8]:
        pieces.append(str(segment.get("label", "")))
        pieces.append(str(segment.get("text", "")))
    return "\n".join(pieces)


def _segment_action(segment: dict[str, Any]) -> str:
    return str(segment.get("metadata", {}).get("action", ""))


if __name__ == "__main__":
    main()
