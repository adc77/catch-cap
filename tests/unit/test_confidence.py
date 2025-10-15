"""Tests for confidence scoring."""

import pytest
from catch_cap.confidence import compute_confidence_score, interpret_confidence
from catch_cap.types import (
    CatchCapResult,
    GenerationResult,
    SemanticEntropyAnalysis,
    LogProbAnalysis,
    JudgeVerdict,
)


def test_confidence_with_high_entropy():
    """High semantic entropy should increase confidence in hallucination detection."""
    result = CatchCapResult(
        query="test query",
        responses=[GenerationResult(text="test")],
        semantic_entropy=SemanticEntropyAnalysis(
            entropy_score=0.8,  # High entropy
            is_confident=False,
        ),
        logprob_analysis=None,
        judge_verdict=None,
        confabulation_detected=True,
        corrected_answer=None,
        web_answer=None,
        metadata={},
    )

    confidence = compute_confidence_score(result)
    assert confidence > 0.5, "High entropy should give high confidence"
    assert interpret_confidence(confidence) in ["Medium", "High", "Very High"]


def test_confidence_with_low_logprobs():
    """High ratio of low-probability tokens should increase confidence."""
    result = CatchCapResult(
        query="test query",
        responses=[GenerationResult(text="test")],
        semantic_entropy=None,
        logprob_analysis=LogProbAnalysis(
            flagged_token_ratio=0.3,  # 30% suspicious tokens
            flagged_token_count=15,
            total_tokens=50,
            is_suspicious=True,
        ),
        judge_verdict=None,
        confabulation_detected=True,
        corrected_answer=None,
        web_answer=None,
        metadata={},
    )

    confidence = compute_confidence_score(result)
    assert confidence > 0.3, "Suspicious logprobs should increase confidence"


def test_confidence_with_judge_verdict():
    """Judge verdict should be the strongest signal."""
    result = CatchCapResult(
        query="test query",
        responses=[GenerationResult(text="test")],
        semantic_entropy=None,
        logprob_analysis=None,
        judge_verdict=JudgeVerdict(
            verdict="INCONSISTENT",
            raw_response="The response is INCONSISTENT with evidence",
            is_consistent=False,
        ),
        confabulation_detected=True,
        corrected_answer=None,
        web_answer=None,
        metadata={},
    )

    confidence = compute_confidence_score(result)
    assert confidence > 0.7, "Judge verdict should give very high confidence"
    assert interpret_confidence(confidence) in ["High", "Very High"]


def test_confidence_with_no_signals():
    """No detection signals should give low confidence."""
    result = CatchCapResult(
        query="test query",
        responses=[GenerationResult(text="test")],
        semantic_entropy=None,
        logprob_analysis=None,
        judge_verdict=None,
        confabulation_detected=False,
        corrected_answer=None,
        web_answer=None,
        metadata={},
    )

    confidence = compute_confidence_score(result)
    assert confidence < 0.3, "No signals should give low confidence"
    assert interpret_confidence(confidence) in ["Very Low", "Low"]


def test_interpret_confidence():
    """Test confidence level interpretation."""
    assert interpret_confidence(0.95) == "Very High"
    assert interpret_confidence(0.75) == "High"
    assert interpret_confidence(0.55) == "Medium"
    assert interpret_confidence(0.35) == "Low"
    assert interpret_confidence(0.15) == "Very Low"
