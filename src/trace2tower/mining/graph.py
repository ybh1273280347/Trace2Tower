from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trace2tower.mining.common import group_by_trajectory, segment_action, segment_success
from trace2tower.text import action_template


@dataclass
class EigenTraceGraph:
    segments: list[dict[str, Any]]
    segment_ids: list[str]
    signatures: list[str]
    embeddings: np.ndarray
    semantic: np.ndarray
    transition: np.ndarray
    outcome: np.ndarray
    mask: np.ndarray
    adjacency: np.ndarray
    success_adjacency: np.ndarray
    failure_adjacency: np.ndarray
    contrastive_adjacency: np.ndarray
    spectral_affinity: np.ndarray
    outcome_tendency: np.ndarray


def segment_signature(segment: dict[str, Any]) -> str:
    metadata = segment.get("metadata", {})
    return "\n".join(
        [
            f"Label: {segment.get('label', 'Unknown')}",
            f"Action template: {action_template(segment_action(segment))}",
            f"Action: {segment_action(segment)}",
            f"Goal: {metadata.get('goal', '')}",
            f"Env: {metadata.get('env', '')}",
            f"Observation: {segment.get('text', '')}",
        ]
    )


def build_eigentrace_graph(
    segments: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    semantic_top_k: int,
    semantic_weight: float,
    transition_weight: float,
    outcome_weight: float,
    contrastive_lambda: float,
) -> EigenTraceGraph:
    if len(segments) != embeddings.shape[0]:
        raise ValueError("Segment count and embedding row count do not match.")
    if not segments:
        return _empty_graph()

    normalized_embeddings = _row_normalize(embeddings)
    semantic = np.maximum(normalized_embeddings @ normalized_embeddings.T, 0.0)
    np.fill_diagonal(semantic, 0.0)

    transition = _transition_matrix(segments)
    outcome_tendency = _outcome_tendency(segments)
    outcome = 1.0 - np.abs(outcome_tendency[:, np.newaxis] - outcome_tendency[np.newaxis, :])
    np.fill_diagonal(outcome, 0.0)

    mask = _sparse_mask(semantic, transition, top_k=semantic_top_k)
    adjacency = mask * (
        semantic_weight * semantic
        + transition_weight * transition
        + outcome_weight * outcome
    )
    adjacency = _symmetrize(adjacency)
    success_flags = np.asarray([segment_success(segment) for segment in segments], dtype=bool)
    if not np.any(success_flags):
        raise ValueError("Trace2Tower requires at least one successful segment to build G+.")

    success_adjacency = adjacency * np.outer(success_flags, success_flags)
    failure_adjacency = adjacency * np.outer(~success_flags, ~success_flags)
    contrastive_adjacency = success_adjacency - contrastive_lambda * failure_adjacency
    if not np.any(contrastive_adjacency):
        raise ValueError("Trace2Tower contrastive graph is empty after applying G+ - lambda G-.")
    # positive part 仅用于审计和可视化；真正谱分解使用 signed contrastive_adjacency。
    spectral_affinity = np.maximum(contrastive_adjacency, 0.0)

    return EigenTraceGraph(
        segments=segments,
        segment_ids=[str(segment["segment_id"]) for segment in segments],
        signatures=[segment_signature(segment) for segment in segments],
        embeddings=normalized_embeddings,
        semantic=semantic,
        transition=transition,
        outcome=outcome,
        mask=mask,
        adjacency=adjacency,
        success_adjacency=success_adjacency,
        failure_adjacency=failure_adjacency,
        contrastive_adjacency=contrastive_adjacency,
        spectral_affinity=spectral_affinity,
        outcome_tendency=outcome_tendency,
    )


def graph_edges(graph: EigenTraceGraph, *, min_weight: float = 1e-9) -> list[dict[str, Any]]:
    edges = []
    n = len(graph.segment_ids)
    for source in range(n):
        for target in range(source + 1, n):
            weight = float(graph.adjacency[source, target])
            if weight <= min_weight:
                continue
            edges.append(
                {
                    "source": graph.segment_ids[source],
                    "target": graph.segment_ids[target],
                    "relation": "eigentrace_affinity",
                    "weight": weight,
                    "metadata": {
                        "semantic": float(graph.semantic[source, target]),
                        "transition": float(graph.transition[source, target]),
                        "outcome_consistency": float(graph.outcome[source, target]),
                        "contrastive_weight": float(graph.contrastive_adjacency[source, target]),
                    },
                }
            )
    return edges


def _empty_graph() -> EigenTraceGraph:
    empty = np.zeros((0, 0), dtype=np.float32)
    return EigenTraceGraph(
        segments=[],
        segment_ids=[],
        signatures=[],
        embeddings=empty,
        semantic=empty,
        transition=empty,
        outcome=empty,
        mask=empty,
        adjacency=empty,
        success_adjacency=empty,
        failure_adjacency=empty,
        contrastive_adjacency=empty,
        spectral_affinity=empty,
        outcome_tendency=np.zeros(0, dtype=np.float32),
    )


def _transition_matrix(segments: list[dict[str, Any]]) -> np.ndarray:
    index = {str(segment["segment_id"]): row for row, segment in enumerate(segments)}
    counts = np.zeros((len(segments), len(segments)), dtype=np.float32)
    outgoing = np.zeros(len(segments), dtype=np.float32)
    for items in group_by_trajectory(segments).values():
        ordered = sorted(items, key=lambda item: int(item.get("step_index", 0) or 0))
        for previous, current in zip(ordered, ordered[1:]):
            source = index[str(previous["segment_id"])]
            target = index[str(current["segment_id"])]
            counts[source, target] += 1.0
            outgoing[source] += 1.0

    directed = counts / np.maximum(outgoing[:, np.newaxis], 1e-12)
    return _symmetrize(directed)


def _outcome_tendency(segments: list[dict[str, Any]]) -> np.ndarray:
    grouped: dict[tuple[str, str], list[float]] = {}
    for segment in segments:
        key = (
            str(segment.get("label", "Unknown")),
            action_template(segment_action(segment)),
        )
        grouped.setdefault(key, []).append(1.0 if segment_success(segment) else 0.0)

    tendency_by_key = {
        key: float(np.mean(values))
        for key, values in grouped.items()
    }
    return np.asarray(
        [
            tendency_by_key[
                (
                    str(segment.get("label", "Unknown")),
                    action_template(segment_action(segment)),
                )
            ]
            for segment in segments
        ],
        dtype=np.float32,
    )


def _sparse_mask(semantic: np.ndarray, transition: np.ndarray, *, top_k: int) -> np.ndarray:
    n = semantic.shape[0]
    mask = transition > 0
    if top_k > 0 and n > 1:
        k = min(top_k, n - 1)
        for row in range(n):
            candidates = np.argpartition(-semantic[row], kth=k)[:k]
            mask[row, candidates] = True
    mask = np.logical_or(mask, mask.T)
    np.fill_diagonal(mask, False)
    return mask.astype(np.float32)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return np.maximum(matrix, matrix.T).astype(np.float32)
