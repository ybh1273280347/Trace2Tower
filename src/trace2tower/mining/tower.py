from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import numpy as np

from trace2tower.mining.common import (
    build_skill,
    group_by_trajectory,
    segment_action,
    segment_node,
)
from trace2tower.mining.graph import EigenTraceGraph, graph_edges
from trace2tower.mining.spectral import SpectralResult
from trace2tower.text import action_template


def induce_skill_tower(
    graph: EigenTraceGraph,
    spectral: SpectralResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not graph.segments:
        return [], [], []

    low_skills = _low_level_skills(graph.segments)
    mid_skills = _mid_level_skills(graph, spectral)
    high_skills, hierarchy_edges = _high_level_skills(graph, spectral, mid_skills)
    edges = graph_edges(graph) + hierarchy_edges
    nodes = [segment_node(segment) for segment in graph.segments] + _skill_nodes(low_skills + mid_skills + high_skills)
    return low_skills + mid_skills + high_skills, nodes, edges


def _low_level_skills(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[action_template(segment_action(segment))].append(segment)

    skills = []
    for index, (template, items) in enumerate(sorted(grouped.items())):
        content = "\n".join(
            [
                f"Low-level action template: {template}",
                "Representative actions:",
                *[f"- {action}" for action, _ in Counter(segment_action(item) for item in items).most_common(8)],
            ]
        )
        skills.append(
            build_skill(
                skill_id=f"trace2tower_low_{index:03d}",
                name=f"Low {template}",
                granularity="low",
                segments=items,
                all_segment_count=len(segments),
                source_method="trace2tower",
                content=content,
                extra_metadata={
                    "tower_level": "low",
                    "action_template": template,
                },
            )
        )
    return skills


def _mid_level_skills(graph: EigenTraceGraph, spectral: SpectralResult) -> list[dict[str, Any]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(spectral.labels):
        grouped[int(label)].append(index)

    skills = []
    for output_index, (cluster_id, indices) in enumerate(sorted(grouped.items())):
        items = [graph.segments[index] for index in indices]
        labels = Counter(str(item.get("label", "Unknown")) for item in items)
        templates = Counter(action_template(segment_action(item)) for item in items)
        centroid = np.mean(spectral.embedding[indices], axis=0).tolist()
        content = "\n".join(
            [
                f"Mid-level EigenTrace behavior pattern {cluster_id}",
                f"Dominant events: {_counter_text(labels)}",
                f"Dominant action templates: {_counter_text(templates)}",
                f"Typical transition context: {_transition_context(graph, set(indices))}",
            ]
        )
        skills.append(
            build_skill(
                skill_id=f"trace2tower_mid_{output_index:03d}",
                name=f"Mid EigenTrace Pattern {output_index}",
                granularity="mid",
                segments=items,
                all_segment_count=len(graph.segments),
                source_method="trace2tower",
                content=content,
                extra_metadata={
                    "tower_level": "mid",
                    "eigentrace_cluster": cluster_id,
                    "eigentrace_centroid": centroid,
                    "eigentrace_support_indices": indices,
                },
            )
        )
    return skills


def _high_level_skills(
    graph: EigenTraceGraph,
    spectral: SpectralResult,
    mid_skills: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transitions = _mid_transition_graph(graph, spectral)
    communities = _communities(transitions, node_count=len(mid_skills))
    skills = []
    hierarchy_edges = []
    for community_index, community in enumerate(communities):
        if not community:
            continue
        indices = _segments_for_mid_community(spectral.labels, community)
        items = [graph.segments[index] for index in indices]
        child_skills = [mid_skills[index] for index in sorted(community) if index < len(mid_skills)]
        content = "\n".join(
            [
                f"High-level skill community {community_index}",
                "Composed mid-level routines:",
                *[f"- {skill['name']}" for skill in child_skills],
                f"Common execution order: {_mid_path_text(transitions, community)}",
            ]
        )
        skill = build_skill(
            skill_id=f"trace2tower_high_{community_index:03d}",
            name=f"High Skill Routine {community_index}",
            granularity="high",
            segments=items,
            all_segment_count=len(graph.segments),
            source_method="trace2tower",
            content=content,
            extra_metadata={
                "tower_level": "high",
                "child_skill_ids": [skill["skill_id"] for skill in child_skills],
                "mid_community": sorted(int(item) for item in community),
            },
        )
        skills.append(skill)
        for child in child_skills:
            hierarchy_edges.append(
                {
                    "source": skill["skill_id"],
                    "target": child["skill_id"],
                    "relation": "contains_mid_skill",
                    "weight": 1.0,
                }
            )
    return skills, hierarchy_edges


def _mid_transition_graph(graph: EigenTraceGraph, spectral: SpectralResult) -> nx.DiGraph:
    transition_graph = nx.DiGraph()
    for label in sorted(set(int(item) for item in spectral.labels)):
        transition_graph.add_node(label)

    segment_index = {str(segment["segment_id"]): index for index, segment in enumerate(graph.segments)}
    for items in group_by_trajectory(graph.segments).values():
        ordered = sorted(items, key=lambda item: int(item.get("step_index", 0) or 0))
        for previous, current in zip(ordered, ordered[1:]):
            source = int(spectral.labels[segment_index[str(previous["segment_id"])]])
            target = int(spectral.labels[segment_index[str(current["segment_id"])]])
            if source == target:
                continue
            weight = transition_graph.get_edge_data(source, target, {}).get("weight", 0.0) + 1.0
            transition_graph.add_edge(source, target, weight=weight)
    return transition_graph


def _communities(graph: nx.DiGraph, *, node_count: int) -> list[set[int]]:
    if node_count == 0:
        return []
    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        return [{node} for node in range(node_count)]
    return [
        set(int(node) for node in community)
        for community in nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight")
    ]


def _segments_for_mid_community(labels: np.ndarray, community: set[int]) -> list[int]:
    return [
        index
        for index, label in enumerate(labels)
        if int(label) in community
    ]


def _skill_nodes(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": skill["skill_id"],
            "label": skill["name"],
            "text": skill.get("content", ""),
            "metadata": {
                "node_type": "skill",
                "granularity": skill.get("granularity", ""),
            },
        }
        for skill in skills
    ]


def _counter_text(counter: Counter[str]) -> str:
    return ", ".join(f"{value} ({count})" for value, count in counter.most_common(5)) or "none"


def _transition_context(graph: EigenTraceGraph, indices: set[int]) -> str:
    incoming = Counter()
    outgoing = Counter()
    for source, target in zip(*np.nonzero(graph.transition > 0)):
        if source in indices and target not in indices:
            outgoing[str(graph.segments[target].get("label", "Unknown"))] += 1
        if target in indices and source not in indices:
            incoming[str(graph.segments[source].get("label", "Unknown"))] += 1
    return f"incoming=[{_counter_text(incoming)}], outgoing=[{_counter_text(outgoing)}]"


def _mid_path_text(graph: nx.DiGraph, community: set[int]) -> str:
    edges = [
        (source, target, data.get("weight", 0.0))
        for source, target, data in graph.edges(data=True)
        if source in community and target in community
    ]
    if not edges:
        return "single routine or no repeated cross-mid transition"
    edges.sort(key=lambda item: item[2], reverse=True)
    return " -> ".join(f"{source}:{target}" for source, target, _ in edges[:6])

