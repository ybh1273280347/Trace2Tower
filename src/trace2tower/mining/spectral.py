from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans


@dataclass
class SpectralResult:
    eigenvalues: list[float]
    embedding: np.ndarray
    labels: np.ndarray
    cluster_count: int
    eigengap: float


def contrastive_spectral_clustering(
    affinity: np.ndarray,
    *,
    rank: int,
    cluster_count: int | None,
    min_clusters: int,
    max_clusters: int,
    random_state: int,
) -> SpectralResult:
    n = affinity.shape[0]
    if n == 0:
        return SpectralResult([], np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int32), 0, 0.0)
    if n == 1:
        return SpectralResult([0.0], np.ones((1, 1), dtype=np.float32), np.zeros(1, dtype=np.int32), 1, 0.0)

    sym_affinity = _symmetrize_signed(affinity)
    graph_laplacian = (
        _signed_normalized_laplacian(sym_affinity)
        if np.any(sym_affinity < 0)
        else laplacian(sym_affinity, normed=True)
    )
    eigenvalues, eigenvectors = eigh(graph_laplacian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    auto_clusters = cluster_count or _auto_cluster_count(
        eigenvalues,
        n=n,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )
    actual_rank = min(max(rank, auto_clusters), n)
    embedding = eigenvectors[:, 1:actual_rank] if actual_rank > 1 else eigenvectors[:, :1]
    if embedding.shape[1] == 0:
        embedding = eigenvectors[:, :1]
    embedding = _row_normalize(embedding.astype(np.float32))

    labels = KMeans(
        n_clusters=auto_clusters,
        n_init=20,
        random_state=random_state,
    ).fit_predict(embedding)
    gap = _eigengap(eigenvalues, auto_clusters)
    return SpectralResult(
        eigenvalues=[float(value) for value in eigenvalues[: min(len(eigenvalues), max(actual_rank + 1, auto_clusters + 1))]],
        embedding=embedding,
        labels=labels.astype(np.int32),
        cluster_count=auto_clusters,
        eigengap=gap,
    )


def _auto_cluster_count(
    eigenvalues: np.ndarray,
    *,
    n: int,
    min_clusters: int,
    max_clusters: int,
) -> int:
    upper = min(max_clusters, n)
    lower = min(max(1, min_clusters), upper)
    if upper <= 1:
        return 1
    if lower == upper:
        return lower

    candidates = range(max(2, lower), upper + 1)
    best_k = max(2, lower)
    best_gap = -1.0
    for k in candidates:
        gap = _eigengap(eigenvalues, k)
        if gap > best_gap:
            best_gap = gap
            best_k = k
    return best_k


def _eigengap(eigenvalues: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(eigenvalues):
        return 0.0
    return float(eigenvalues[k] - eigenvalues[k - 1])


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def _symmetrize_signed(matrix: np.ndarray) -> np.ndarray:
    return ((matrix + matrix.T) / 2.0).astype(np.float32)


def _signed_normalized_laplacian(matrix: np.ndarray) -> np.ndarray:
    # Signed graph 使用绝对权重度数归一化，保留失败图负边对谱空间的排斥作用。
    degree = np.sum(np.abs(matrix), axis=1)
    scale = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = matrix * scale[:, np.newaxis] * scale[np.newaxis, :]
    return np.eye(matrix.shape[0], dtype=np.float32) - normalized
