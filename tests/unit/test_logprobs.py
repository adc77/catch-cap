"""Tests for log probability detection."""

import pytest
from catch_cap.config import LogProbConfig
from catch_cap.detection.logprobs import LogProbDetector
from catch_cap.types import GenerationResult


def test_logprob_detector_no_logprobs():
    """Detector should handle missing logprobs gracefully."""
    config = LogProbConfig(enabled=True, min_logprob=-4.0, fraction_threshold=0.15)
    detector = LogProbDetector(config)

    response = GenerationResult(text="test", logprobs=None)
    analysis = detector.analyse(response)

    assert analysis.flagged_token_ratio == 0.0
    assert analysis.flagged_token_count == 0
    assert analysis.total_tokens == 0
    assert not analysis.is_suspicious


def test_logprob_detector_all_high_confidence():
    """All high-confidence tokens should not be flagged."""
    config = LogProbConfig(enabled=True, min_logprob=-2.0, fraction_threshold=0.15)
    detector = LogProbDetector(config)

    # All tokens have high log probability (close to 0)
    response = GenerationResult(text="test", logprobs=[-0.5, -0.3, -0.8, -0.4])
    analysis = detector.analyse(response)

    assert analysis.flagged_token_ratio == 0.0
    assert analysis.flagged_token_count == 0
    assert not analysis.is_suspicious


def test_logprob_detector_some_suspicious():
    """Some suspicious tokens should be flagged."""
    config = LogProbConfig(
        enabled=True,
        min_logprob=-3.0,
        fraction_threshold=0.3,
        min_flagged_tokens=2,
    )
    detector = LogProbDetector(config)

    # Mix of high and low confidence tokens
    response = GenerationResult(
        text="test",
        logprobs=[-0.5, -5.0, -0.8, -4.5, -1.0],  # 2/5 are suspicious
    )
    analysis = detector.analyse(response)

    assert analysis.flagged_token_count == 2
    assert analysis.flagged_token_ratio == 0.4  # 2/5
    assert analysis.is_suspicious  # Meets min_flagged_tokens threshold


def test_logprob_detector_threshold_logic():
    """Test both threshold conditions (ratio and count)."""
    config = LogProbConfig(
        enabled=True,
        min_logprob=-3.0,
        fraction_threshold=0.5,  # 50% threshold
        min_flagged_tokens=10,  # Or at least 10 tokens
    )
    detector = LogProbDetector(config)

    # Only 2 suspicious tokens out of 10 (20%) - below ratio threshold
    # But also below min_flagged_tokens - should NOT be flagged
    logprobs = [-1.0] * 8 + [-5.0] * 2
    response = GenerationResult(text="test", logprobs=logprobs)
    analysis = detector.analyse(response)

    assert not analysis.is_suspicious

    # 11 suspicious tokens - meets min_flagged_tokens even if ratio is low
    logprobs = [-1.0] * 50 + [-5.0] * 11
    response2 = GenerationResult(text="test", logprobs=logprobs)
    analysis2 = detector.analyse(response2)

    assert analysis2.is_suspicious  # Meets min_flagged_tokens
