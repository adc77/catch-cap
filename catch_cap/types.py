from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class GenerationResult:
    """Represents a single generated response."""

    text: str
    logprobs: Optional[List[float]] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SemanticEntropyAnalysis:
    """Summary of semantic entropy computation."""

    entropy_score: float
    similarity_matrix: Optional[List[List[float]]] = None
    is_confident: bool = False


@dataclass
class LogProbAnalysis:
    """Summary of log-probability analysis."""

    flagged_token_ratio: float
    flagged_token_count: int
    total_tokens: int
    flagged_tokens: Sequence[str] = field(default_factory=tuple)
    is_suspicious: bool = False


@dataclass
class JudgeVerdict:
    """Represents the judgment from the chosen LLM judge."""

    verdict: str
    raw_response: str
    is_consistent: bool


# @dataclass
# class CatchCapResult:
#     """Aggregated result from the CatchCap pipeline."""

#     query: str
#     responses: Sequence[GenerationResult]
#     semantic_entropy: Optional[SemanticEntropyAnalysis]
#     logprob_analysis: Optional[LogProbAnalysis]
#     judge_verdict: Optional[JudgeVerdict]
#     confabulation_detected: bool
#     corrected_answer: Optional[str]
#     web_answer: Optional[str]
#     metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class CatchCapResult:
    """Aggregated result from the CatchCap pipeline."""

    query: str
    responses: Sequence[GenerationResult]
    semantic_entropy: Optional[SemanticEntropyAnalysis]
    logprob_analysis: Optional[LogProbAnalysis]
    judge_verdict: Optional[JudgeVerdict]
    confabulation_detected: bool
    corrected_answer: Optional[str]
    web_answer: Optional[str]  # This is now the synthesized answer
    raw_search_results: Optional[Sequence] = None  # Raw search results
    metadata: Dict[str, str] = field(default_factory=dict)


