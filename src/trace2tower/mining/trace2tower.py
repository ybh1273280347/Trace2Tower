from __future__ import annotations

from typing import Any

from trace2tower.embedding import OpenAICompatibleEmbeddingClient
from trace2tower.mining.graph import build_eigentrace_graph
from trace2tower.mining.spectral import contrastive_spectral_clustering
from trace2tower.mining.tower import induce_skill_tower


class Trace2TowerMiner:
    method = "trace2tower"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def mine(self, segments: list[dict]) -> dict:
        if not segments:
            return {
                "method": self.method,
                "description": "Trace2Tower received no segments.",
                "nodes": [],
                "edges": [],
                "skills": [],
                "metadata": {"segment_count": 0},
            }

        embedding_client = OpenAICompatibleEmbeddingClient(
            timeout=int(self.config.get("embedding_timeout", 120)),
            max_retries=int(self.config.get("embedding_max_retries", 3)),
            retry_delay=float(self.config.get("embedding_retry_delay", 1.0)),
            batch_size=int(self.config.get("embedding_batch_size", 8)),
            batch_delay=float(self.config.get("embedding_batch_delay", 0.0)),
        )
        signatures = [
            _segment_signature_for_embedding(segment)
            for segment in segments
        ]
        embedding_result = embedding_client.embed(signatures)
        graph = build_eigentrace_graph(
            segments,
            embedding_result.embeddings,
            semantic_top_k=int(self.config.get("semantic_top_k", 8)),
            semantic_weight=float(self.config.get("semantic_weight", 0.45)),
            transition_weight=float(self.config.get("transition_weight", 0.35)),
            outcome_weight=float(self.config.get("outcome_weight", 0.20)),
            contrastive_lambda=float(self.config.get("contrastive_lambda", 0.65)),
        )
        spectral = contrastive_spectral_clustering(
            graph.contrastive_adjacency,
            rank=int(self.config.get("spectral_rank", 6)),
            cluster_count=_optional_int(self.config.get("mid_clusters")),
            min_clusters=int(self.config.get("min_mid_clusters", 2)),
            max_clusters=int(self.config.get("max_mid_clusters", 8)),
            random_state=int(self.config.get("random_state", 13)),
        )
        skills, nodes, edges = induce_skill_tower(graph, spectral)
        return {
            "method": self.method,
            "description": (
                "Transition-aware contrastive EigenTrace induction of low/mid/high "
                "skills from event-level execution segments."
            ),
            "nodes": nodes,
            "edges": edges,
            "skills": skills,
            "metadata": {
                "segment_count": len(segments),
                "embedding_model": embedding_client.model,
                "embedding_prompt_tokens": embedding_result.prompt_tokens,
                "embedding_total_tokens": embedding_result.total_tokens,
                "embedding_batch_size": int(self.config.get("embedding_batch_size", 8)),
                "semantic_top_k": int(self.config.get("semantic_top_k", 8)),
                "semantic_weight": float(self.config.get("semantic_weight", 0.45)),
                "transition_weight": float(self.config.get("transition_weight", 0.35)),
                "outcome_weight": float(self.config.get("outcome_weight", 0.20)),
                "contrastive_lambda": float(self.config.get("contrastive_lambda", 0.65)),
                "spectral_rank": int(self.config.get("spectral_rank", 6)),
                "mid_cluster_count": spectral.cluster_count,
                "eigengap": spectral.eigengap,
                "eigenvalues": spectral.eigenvalues,
                "algorithm_steps": [
                    "event_segment_signature_embedding",
                    "transition_aware_eigentrace_graph",
                    "success_failure_contrastive_decomposition",
                    "spectrum_to_low_mid_high_skill_tower",
                ],
            },
        }


def _segment_signature_for_embedding(segment: dict[str, Any]) -> str:
    metadata = segment.get("metadata", {})
    return "\n".join(
        [
            f"Label: {segment.get('label', 'Unknown')}",
            f"Action: {metadata.get('action', '')}",
            f"Goal: {metadata.get('goal', '')}",
            f"Environment: {metadata.get('env', '')}",
            f"Observation and local outcome: {segment.get('text', '')}",
        ]
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    return int(value)
