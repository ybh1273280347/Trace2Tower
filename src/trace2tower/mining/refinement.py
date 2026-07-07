from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from trace2tower.embedding import OpenAICompatibleEmbeddingClient
from trace2tower.mining.common import build_skill, segment_success


DEFAULT_UTILITY_WEIGHTS = {
    "success": 0.45,
    "reward": 0.35,
    "step_save": 0.15,
    "cost": 0.05,
}


def refine_skill_tower(
    model: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    deployment_retrieval: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = config or {}
    task_outcomes = _task_outcomes(records)
    usage = _skill_usage(deployment_retrieval, task_outcomes)
    co_usage = _mid_skill_co_usage(deployment_retrieval)
    weights = {
        **DEFAULT_UTILITY_WEIGHTS,
        **settings.get("utility_weights", {}),
    }

    segment_by_id = _segments_by_id(model)
    scored_skills, action_counts = _score_skills(
        model.get("skills", []),
        usage=usage,
        weights=weights,
        settings=settings,
    )
    split_skills, split_edges = _split_unstable_skills(
        scored_skills,
        segment_by_id=segment_by_id,
        settings=settings,
    )
    merged_skills, merge_edges, merge_summary = _merge_duplicate_skills(
        scored_skills + split_skills,
        segment_by_id=segment_by_id,
        settings=settings,
    )
    promoted_skills, promote_edges, reinforced_high_count = _promote_mid_skill_sets(
        merged_skills,
        segment_by_id=segment_by_id,
        co_usage=co_usage,
        settings=settings,
    )

    final_skills = merged_skills + promoted_skills
    refined = dict(model)
    refined["skills"] = final_skills
    refined["nodes"] = _nodes_with_refined_skills(model.get("nodes", []), final_skills)
    refined["edges"] = [
        *model.get("edges", []),
        *split_edges,
        *merge_edges,
        *promote_edges,
    ]
    refined["metadata"] = {
        **dict(model.get("metadata", {})),
        "refinement": {
            "record_count": len(records),
            "deployment_retrieval_count": len(deployment_retrieval),
            "utility_weights": weights,
            "action_counts": dict(action_counts),
            "structural_updates": {
                "split_child_count": len(split_skills),
                "merged_skill_count": merge_summary["merged_skill_count"],
                "promoted_high_skill_count": len(promoted_skills),
                "reinforced_high_skill_count": reinforced_high_count,
            },
            "merge_embedding": merge_summary["embedding"],
        },
    }
    return refined


def _score_skills(
    skills: list[dict[str, Any]],
    *,
    usage: dict[str, list[dict[str, float]]],
    weights: dict[str, float],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    scored = []
    actions: Counter[str] = Counter()
    for skill in skills:
        item = dict(skill)
        metadata = dict(item.get("metadata", {}))
        stats = usage.get(str(skill.get("skill_id", "")), [])
        update = _refinement_update(
            stats,
            token_cost=float(metadata.get("token_cost", 0.0) or 0.0),
            weights=weights,
            granularity=str(item.get("granularity", "")),
            settings=settings,
        )
        if update["refinement_action"] == "split_candidate":
            # 父技能保留为 provenance，但部署权重降低，真正部署优先使用 split child。
            update["deployment_weight"] = min(float(update["deployment_weight"]), 0.5)
            update["refinement_structure"] = "split_parent"

        metadata.update(update)
        actions[update["refinement_action"]] += 1
        item["metadata"] = metadata
        if update["refinement_action"] == "downweight" and settings.get("prune_downweighted", False):
            continue
        scored.append(item)
    return scored, actions


def _split_unstable_skills(
    skills: list[dict[str, Any]],
    *,
    segment_by_id: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    min_support = int(settings.get("min_split_support", 1))
    split_skills = []
    split_edges = []
    for skill in skills:
        if skill.get("metadata", {}).get("refinement_action") != "split_candidate":
            continue

        members = _member_segments(skill, segment_by_id)
        success_members = [segment for segment in members if segment_success(segment)]
        failure_members = [segment for segment in members if not segment_success(segment)]
        if len(success_members) < min_support or len(failure_members) < min_support:
            continue

        for split_name, split_members in [
            ("success", success_members),
            ("failure", failure_members),
        ]:
            child = build_skill(
                skill_id=f"{skill['skill_id']}_split_{split_name}",
                name=f"{skill['name']} [{split_name}]",
                granularity=str(skill.get("granularity", "")),
                segments=split_members,
                all_segment_count=max(len(segment_by_id), 1),
                source_method=str(skill.get("metadata", {}).get("source_method", "trace2tower")),
                content="\n".join(
                    [
                        f"Split child of {skill['name']}",
                        f"Outcome partition: {split_name}",
                        str(skill.get("content", "")),
                    ]
                ),
                extra_metadata={
                    "tower_level": skill.get("metadata", {}).get("tower_level", skill.get("granularity", "")),
                    "parent_skill_id": skill["skill_id"],
                    "split_group": split_name,
                    "refinement_action": "split_child",
                    "deployment_weight": max(
                        0.5,
                        float(skill.get("metadata", {}).get("deployment_weight", 1.0) or 1.0),
                    ),
                },
            )
            split_skills.append(child)
            split_edges.append(
                {
                    "source": child["skill_id"],
                    "target": skill["skill_id"],
                    "relation": "split_from",
                    "weight": 1.0,
                }
            )
    return split_skills, split_edges


def _merge_duplicate_skills(
    skills: list[dict[str, Any]],
    *,
    segment_by_id: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    threshold = float(settings.get("merge_similarity_threshold", 0.985))
    if len(skills) < 2:
        return skills, [], _merge_summary(0, "", 0, 0)

    client = OpenAICompatibleEmbeddingClient(
        timeout=int(settings.get("embedding_timeout", 120)),
        max_retries=int(settings.get("embedding_max_retries", 3)),
        retry_delay=float(settings.get("embedding_retry_delay", 1.0)),
        batch_size=int(settings.get("embedding_batch_size", 8)),
        batch_delay=float(settings.get("embedding_batch_delay", 0.0)),
    )
    result = client.embed([_skill_text(skill) for skill in skills])
    similarities = _cosine_matrix(result.embeddings)
    components = _merge_components(skills, similarities, threshold=threshold)
    if not components:
        return skills, [], _merge_summary(0, client.model, result.prompt_tokens, result.total_tokens)

    merged = []
    merged_indices = set()
    merge_edges = []
    for output_index, component in enumerate(components):
        merged_indices.update(component)
        component_skills = [skills[index] for index in component]
        merged_skill = _merged_skill(
            output_index,
            component_skills,
            segment_by_id=segment_by_id,
            threshold=threshold,
        )
        merged.append(merged_skill)
        for source in component_skills:
            merge_edges.append(
                {
                    "source": merged_skill["skill_id"],
                    "target": source["skill_id"],
                    "relation": "merged_from",
                    "weight": 1.0,
                }
            )

    final_skills = [
        skill
        for index, skill in enumerate(skills)
        if index not in merged_indices
    ] + merged
    return final_skills, merge_edges, _merge_summary(
        len(merged),
        client.model,
        result.prompt_tokens,
        result.total_tokens,
    )


def _promote_mid_skill_sets(
    skills: list[dict[str, Any]],
    *,
    segment_by_id: dict[str, dict[str, Any]],
    co_usage: Counter[tuple[str, ...]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    skill_by_id = {str(skill.get("skill_id", "")): skill for skill in skills}
    existing_high_by_set = {
        tuple(sorted(str(item) for item in skill.get("metadata", {}).get("child_skill_ids", []))): skill
        for skill in skills
        if skill.get("granularity") == "high" and skill.get("metadata", {}).get("child_skill_ids")
    }
    min_usage = int(settings.get("promote_min_usage", 2))
    min_utility = float(settings.get("promote_min_avg_utility", settings.get("promote_threshold", 0.55)))

    promoted = []
    promote_edges = []
    reinforced_high_count = 0
    for output_index, (mid_ids, count) in enumerate(sorted(co_usage.items())):
        if count < min_usage:
            continue
        child_skills = [
            skill_by_id[skill_id]
            for skill_id in mid_ids
            if skill_id in skill_by_id and skill_by_id[skill_id].get("granularity") == "mid"
        ]
        if len(child_skills) < 2:
            continue

        avg_utility = _mean(
            float(skill.get("metadata", {}).get("deployment_utility", 0.0) or 0.0)
            for skill in child_skills
        )
        if avg_utility < min_utility:
            continue

        if mid_ids in existing_high_by_set:
            high_skill = existing_high_by_set[mid_ids]
            metadata = dict(high_skill.get("metadata", {}))
            metadata.update(
                {
                    "deployment_co_usage_count": count,
                    "deployment_utility": max(
                        float(metadata.get("deployment_utility", 0.0) or 0.0),
                        avg_utility,
                    ),
                    "deployment_weight": max(
                        float(metadata.get("deployment_weight", 1.0) or 1.0),
                        1.0 + avg_utility,
                    ),
                    "refinement_action": "reinforced_high",
                }
            )
            high_skill["metadata"] = metadata
            reinforced_high_count += 1
            for child in child_skills:
                promote_edges.append(
                    {
                        "source": high_skill["skill_id"],
                        "target": child["skill_id"],
                        "relation": "reinforced_by_deployment",
                        "weight": float(count),
                    }
                )
            continue

        members = _unique_segments(child_skills, segment_by_id)
        content = "\n".join(
            [
                "Promoted high-level routine from deployment co-use.",
                f"Deployment co-use count: {count}",
                "Composed mid-level routines:",
                *[f"- {skill['name']}" for skill in child_skills],
            ]
        )
        skill = build_skill(
            skill_id=f"trace2tower_refined_high_{output_index:03d}",
            name=f"Refined High Routine {output_index}",
            granularity="high",
            segments=members,
            all_segment_count=max(len(segment_by_id), 1),
            source_method="trace2tower",
            content=content,
            extra_metadata={
                "tower_level": "high",
                "child_skill_ids": list(mid_ids),
                "deployment_co_usage_count": count,
                "deployment_utility": avg_utility,
                "deployment_weight": max(1.0, 1.0 + avg_utility),
                "refinement_action": "promoted_high",
            },
        )
        promoted.append(skill)
        for child in child_skills:
            promote_edges.append(
                {
                    "source": skill["skill_id"],
                    "target": child["skill_id"],
                    "relation": "promotes_mid_skill",
                    "weight": float(count),
                }
            )
    return promoted, promote_edges, reinforced_high_count


def _task_outcomes(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    max_steps = max((len(record.get("steps", [])) for record in records), default=1) or 1
    return {
        str(record.get("task_id")): {
            "success": 1.0 if record.get("success") else 0.0,
            "reward": float(record.get("final_reward", 0.0) or 0.0),
            "step_save": 1.0 - (len(record.get("steps", [])) / max_steps),
        }
        for record in records
    }


def _skill_usage(
    deployment_retrieval: list[dict[str, Any]],
    task_outcomes: dict[str, dict[str, float]],
) -> dict[str, list[dict[str, float]]]:
    usage: dict[str, list[dict[str, float]]] = defaultdict(list)
    seen = set()
    for record in deployment_retrieval:
        task_id = str(record.get("task_id", ""))
        outcome = task_outcomes.get(task_id)
        if not outcome:
            continue
        for skill in record.get("retrieved_skills", []):
            skill_id = str(skill.get("skill_id", ""))
            key = (task_id, skill_id)
            if not skill_id or key in seen:
                continue
            seen.add(key)
            usage[skill_id].append(outcome)
    return usage


def _mid_skill_co_usage(deployment_retrieval: list[dict[str, Any]]) -> Counter[tuple[str, ...]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for record in deployment_retrieval:
        mid_ids = sorted(
            {
                str(skill.get("skill_id", ""))
                for skill in record.get("retrieved_skills", [])
                if skill.get("granularity") == "mid" and skill.get("skill_id")
            }
        )
        if len(mid_ids) < 2:
            continue
        for size in range(2, len(mid_ids) + 1):
            counter.update(tuple(group) for group in combinations(mid_ids, size))
    return counter


def _refinement_update(
    stats: list[dict[str, float]],
    *,
    token_cost: float,
    weights: dict[str, float],
    granularity: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not stats:
        return {
            "deployment_usage_count": 0,
            "deployment_utility": 0.0,
            "deployment_weight": 1.0,
            "refinement_action": "unobserved",
        }

    success_rate = _mean(item["success"] for item in stats)
    avg_reward = _mean(item["reward"] for item in stats)
    step_save = _mean(item["step_save"] for item in stats)
    cost_penalty = token_cost / (token_cost + 100.0)
    utility = (
        weights["success"] * success_rate
        + weights["reward"] * avg_reward
        + weights["step_save"] * step_save
        - weights["cost"] * cost_penalty
    )
    action = _refinement_action(
        utility=utility,
        success_values=[item["success"] for item in stats],
        usage_count=len(stats),
        granularity=granularity,
        settings=settings,
    )
    return {
        "deployment_usage_count": len(stats),
        "deployment_success_rate": success_rate,
        "deployment_avg_reward": avg_reward,
        "deployment_step_save": step_save,
        "deployment_cost_penalty": cost_penalty,
        "deployment_utility": utility,
        "deployment_weight": _deployment_weight(utility, action),
        "refinement_action": action,
    }


def _refinement_action(
    *,
    utility: float,
    success_values: list[float],
    usage_count: int,
    granularity: str,
    settings: dict[str, Any],
) -> str:
    prune_threshold = float(settings.get("prune_threshold", 0.05))
    promote_threshold = float(settings.get("promote_threshold", 0.55))
    split_variance_threshold = float(settings.get("split_variance_threshold", 0.20))
    if utility < prune_threshold:
        return "downweight"
    if usage_count >= 3 and _variance(success_values) >= split_variance_threshold:
        return "split_candidate"
    if granularity == "mid" and utility >= promote_threshold:
        return "promote_candidate"
    return "stable"


def _deployment_weight(utility: float, action: str) -> float:
    if action == "downweight":
        return 0.25
    return max(0.5, min(2.0, 1.0 + utility))


def _segments_by_id(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    segments = {}
    for node in model.get("nodes", []):
        if node.get("metadata", {}).get("node_type") == "skill":
            continue
        segment_id = str(node.get("node_id", ""))
        if not segment_id:
            continue
        segments[segment_id] = {
            "segment_id": segment_id,
            "label": node.get("label", "Unknown"),
            "text": node.get("text", ""),
            "metadata": node.get("metadata", {}),
        }
    return segments


def _member_segments(
    skill: dict[str, Any],
    segment_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        segment_by_id[member]
        for member in skill.get("members", [])
        if member in segment_by_id
    ]


def _unique_segments(
    skills: list[dict[str, Any]],
    segment_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = set()
    segments = []
    for skill in skills:
        for segment in _member_segments(skill, segment_by_id):
            segment_id = segment["segment_id"]
            if segment_id in seen:
                continue
            seen.add(segment_id)
            segments.append(segment)
    return segments


def _merge_components(
    skills: list[dict[str, Any]],
    similarities: np.ndarray,
    *,
    threshold: float,
) -> list[list[int]]:
    parent = list(range(len(skills)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(skills)):
        for right in range(left + 1, len(skills)):
            if skills[left].get("granularity") != skills[right].get("granularity"):
                continue
            if similarities[left, right] >= threshold:
                union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(skills)):
        grouped[find(index)].append(index)
    return [
        indices
        for indices in grouped.values()
        if len(indices) > 1
    ]


def _merged_skill(
    output_index: int,
    skills: list[dict[str, Any]],
    *,
    segment_by_id: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    members = _unique_segments(skills, segment_by_id)
    granularity = str(skills[0].get("granularity", ""))
    content = "\n".join(
        [
            "Merged semantically overlapping skills.",
            "Merged sources:",
            *[f"- {skill['name']}" for skill in skills],
            "",
            *[str(skill.get("content", "")) for skill in skills],
        ]
    )
    return build_skill(
        skill_id=f"trace2tower_refined_merge_{output_index:03d}",
        name=f"Merged {granularity.title()} Skill {output_index}",
        granularity=granularity,
        segments=members,
        all_segment_count=max(len(segment_by_id), 1),
        source_method="trace2tower",
        content=content,
        extra_metadata={
            "tower_level": skills[0].get("metadata", {}).get("tower_level", granularity),
            "merged_skill_ids": [skill["skill_id"] for skill in skills],
            "merge_similarity_threshold": threshold,
            "deployment_utility": _mean(
                float(skill.get("metadata", {}).get("deployment_utility", 0.0) or 0.0)
                for skill in skills
            ),
            "deployment_weight": max(
                float(skill.get("metadata", {}).get("deployment_weight", 1.0) or 1.0)
                for skill in skills
            ),
            "refinement_action": "merged",
        },
    )


def _nodes_with_refined_skills(
    nodes: list[dict[str, Any]],
    skills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_skill_ids = {str(skill.get("skill_id", "")) for skill in skills}
    segment_nodes = [
        node
        for node in nodes
        if node.get("metadata", {}).get("node_type") != "skill"
    ]
    historical_skill_nodes = []
    for node in nodes:
        if node.get("metadata", {}).get("node_type") != "skill":
            continue
        if str(node.get("node_id", "")) in current_skill_ids:
            continue
        item = dict(node)
        metadata = dict(item.get("metadata", {}))
        metadata["historical_refinement_node"] = True
        item["metadata"] = metadata
        historical_skill_nodes.append(item)

    skill_nodes = [
        {
            "node_id": skill["skill_id"],
            "label": skill["name"],
            "text": skill.get("content", ""),
            "metadata": {
                "node_type": "skill",
                "granularity": skill.get("granularity", ""),
                "refinement_action": skill.get("metadata", {}).get("refinement_action", ""),
            },
        }
        for skill in skills
    ]
    return segment_nodes + historical_skill_nodes + skill_nodes


def _skill_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(skill.get("name", "")),
            str(skill.get("granularity", "")),
            str(skill.get("content", "")),
            str(skill.get("embedding_text", "")),
        ]
    )


def _cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norm, 1e-12)
    return normalized @ normalized.T


def _merge_summary(
    merged_skill_count: int,
    model: str,
    prompt_tokens: int,
    total_tokens: int,
) -> dict[str, Any]:
    return {
        "merged_skill_count": merged_skill_count,
        "embedding": {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return _mean((value - mean) ** 2 for value in values)
