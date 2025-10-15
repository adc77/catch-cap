"""Tests for semantic entropy detection."""

import pytest
import numpy as np
from catch_cap.utils import compute_semantic_entropy


def test_compute_semantic_entropy_identical_embeddings():
    """Identical embeddings should have very low entropy."""
    embeddings = [[1.0, 0.0, 0.0]] * 5  # All identical
    entropy, similarity_matrix = compute_semantic_entropy(embeddings)

    assert entropy < 0.01, "Identical embeddings should have near-zero entropy"
    assert len(similarity_matrix) == 5
    assert len(similarity_matrix[0]) == 5


def test_compute_semantic_entropy_diverse_embeddings():
    """Very different embeddings should have high entropy."""
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    entropy, similarity_matrix = compute_semantic_entropy(embeddings)

    assert entropy > 0.5, "Diverse embeddings should have high entropy"


def test_compute_semantic_entropy_single_embedding():
    """Single embedding should handle gracefully."""
    embeddings = [[1.0, 0.0, 0.0]]
    entropy, similarity_matrix = compute_semantic_entropy(embeddings)

    # With one embedding, upper triangle is empty, so entropy should be 0 (1 - 1.0)
    assert 0.0 <= entropy <= 1.0


def test_compute_semantic_entropy_invalid_input():
    """Invalid input should raise ValueError."""
    with pytest.raises(ValueError):
        compute_semantic_entropy([[1.0]])  # 1D array
