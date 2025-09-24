from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..config import LogProbConfig
from ..types import GenerationResult, LogProbAnalysis


@dataclass
class LogProbDetector:
    """Detect hallucination risk using token log probabilities."""

    config: LogProbConfig

    def analyse(self, response: GenerationResult) -> LogProbAnalysis:
        logprobs = response.logprobs or []
        total_tokens = len(logprobs)
        if total_tokens == 0:
            return LogProbAnalysis(
                flagged_token_ratio=0.0,
                flagged_token_count=0,
                total_tokens=0,
                flagged_tokens=tuple(),
                is_suspicious=False,
            )

        threshold = self.config.min_logprob
        flagged_indices = [i for i, value in enumerate(logprobs) if value < threshold]
        flagged_ratio = len(flagged_indices) / total_tokens
        is_suspicious = (
            flagged_ratio >= self.config.fraction_threshold
            or len(flagged_indices) >= self.config.min_flagged_tokens
        )

        return LogProbAnalysis(
            flagged_token_ratio=flagged_ratio,
            flagged_token_count=len(flagged_indices),
            total_tokens=total_tokens,
            flagged_tokens=tuple(str(index) for index in flagged_indices),
            is_suspicious=is_suspicious,
        )


