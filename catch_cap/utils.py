from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_entropy(embeddings: Iterable[Iterable[float]]) -> tuple[float, List[List[float]]]:
    """Compute semantic entropy from embeddings."""

    matrix = np.array(list(embeddings))
    if matrix.ndim != 2:
        raise ValueError("Embeddings must form a 2D array")
    similarity = cosine_similarity(matrix)
    upper_triangle = similarity[np.triu_indices_from(similarity, k=1)]
    avg_similarity = float(np.mean(upper_triangle)) if upper_triangle.size else 1.0
    entropy = 1.0 - avg_similarity
    return entropy, similarity.tolist()


def ratio_above_threshold(values: Iterable[float], threshold: float) -> float:
    """Return the ratio of values above the provided threshold."""

    vals = list(values)
    if not vals:
        return 0.0
    count = sum(1 for value in vals if value <= threshold)
    return count / len(vals)


