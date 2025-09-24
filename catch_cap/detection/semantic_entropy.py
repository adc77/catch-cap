"""Semantic entropy detection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..config import SemanticEntropyConfig
from ..types import GenerationResult, SemanticEntropyAnalysis
from ..utils import compute_semantic_entropy


@dataclass
class SemanticEntropyDetector:
    """Detect confabulation using semantic entropy across responses."""

    config: SemanticEntropyConfig

    async def analyse(
        self,
        responses: Iterable[GenerationResult],
        embeddings: Iterable[Iterable[float]],
    ) -> SemanticEntropyAnalysis:
        entropy, matrix = compute_semantic_entropy(embeddings)
        is_confident = entropy < self.config.threshold
        return SemanticEntropyAnalysis(
            entropy_score=entropy,
            similarity_matrix=matrix,
            is_confident=is_confident,
        )


