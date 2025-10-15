"""Confidence scoring for hallucination detection results."""

from __future__ import annotations

from typing import Optional

from .types import CatchCapResult, SemanticEntropyAnalysis, LogProbAnalysis, JudgeVerdict


def compute_confidence_score(result: CatchCapResult) -> float:
    """
    Compute a 0-1 confidence score for the hallucination detection.

    A higher score means we're more confident that:
    - If confabulation_detected=True: the response is indeed hallucinated
    - If confabulation_detected=False: the response is factually accurate

    Args:
        result: The detection result to score

    Returns:
        Confidence score between 0.0 (no confidence) and 1.0 (very confident)
    """
    signals = []
    weights = []

    # Signal 1: Semantic entropy (higher entropy = more confident in hallucination detection)
    if result.semantic_entropy:
        entropy = result.semantic_entropy.entropy_score
        # Normalize to 0-1 range (assuming typical entropy range 0-1)
        # High entropy (>0.5) strongly suggests hallucination
        if entropy > 0.5:
            signals.append(min(entropy, 1.0))
            weights.append(2.0)  # Strong signal
        elif entropy < 0.2:
            # Low entropy suggests confidence in the response
            if not result.confabulation_detected:
                signals.append(1.0 - entropy)
                weights.append(1.5)

    # Signal 2: Log probabilities (higher ratio of low-prob tokens = more confident in hallucination)
    if result.logprob_analysis:
        flagged_ratio = result.logprob_analysis.flagged_token_ratio
        if flagged_ratio > 0.1:  # Significant number of suspicious tokens
            signals.append(min(flagged_ratio * 5, 1.0))  # Scale up
            weights.append(1.5)

    # Signal 3: Judge verdict (strongest signal if available)
    if result.judge_verdict:
        if result.judge_verdict.verdict in ("CONSISTENT", "INCONSISTENT"):
            # Judge gave a clear verdict - very high confidence
            signals.append(1.0)
            weights.append(3.0)  # Judge is strongest signal
        else:
            # Judge uncertain (UNKNOWN)
            signals.append(0.3)
            weights.append(1.0)

    # Signal 4: Web search availability (weak signal, but helpful)
    if result.web_answer:
        # Having web grounding increases confidence
        signals.append(0.7)
        weights.append(0.5)

    # Weighted average
    if signals:
        weighted_sum = sum(s * w for s, w in zip(signals, weights))
        total_weight = sum(weights)
        confidence = weighted_sum / total_weight
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]

    # No signals available - low confidence
    return 0.1


def interpret_confidence(score: float) -> str:
    """
    Convert numerical confidence score to human-readable interpretation.

    Args:
        score: Confidence score between 0.0 and 1.0

    Returns:
        Human-readable confidence level
    """
    if score >= 0.9:
        return "Very High"
    elif score >= 0.7:
        return "High"
    elif score >= 0.5:
        return "Medium"
    elif score >= 0.3:
        return "Low"
    else:
        return "Very Low"
