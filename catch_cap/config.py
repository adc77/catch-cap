"""Configuration objects for catch_cap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence


ProviderName = Literal["openai", "gemini", "groq"]
WebSearchProviderName = Literal["tavily", "searxng", "none"]


@dataclass
class ModelConfig:
    """Configuration for an LLM model used to generate text."""

    provider: ProviderName
    name: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    extra_args: dict = field(default_factory=dict)


@dataclass
class SemanticEntropyConfig:
    """Settings for semantic entropy based detection."""

    enabled: bool = True
    n_responses: int = 3
    threshold: float = 0.25
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: ProviderName = "openai"


@dataclass
class LogProbConfig:
    """Settings for log-probability based detection."""

    enabled: bool = True
    min_logprob: float = -4.5
    fraction_threshold: float = 0.2
    min_flagged_tokens: int = 5

@dataclass
class WebSearchConfig:
    """Configuration for web search grounding."""

    provider: WebSearchProviderName = "tavily"
    max_results: int = 5
    timeout_seconds: int = 20
    searxng_url: str = "https://searxng.wstf.tech/search"
    synthesizer_model: Optional[ModelConfig] = None  # LLM to synthesize web results


@dataclass
class JudgeConfig:
    """Configuration for the LLM-as-a-judge step."""

    model: ModelConfig
    instructions: str = (
        "Compare the model response with the web-synthesized answer for factual accuracy. "
        "For counting or numerical questions, the numbers must match exactly. "
        "If the model response is factually correct according to the web evidence, return CONSISTENT. "
        "If the model response contains any factual errors, wrong numbers, or contradicts the web evidence, return INCONSISTENT. "
        "Be strict about accuracy. Return only CONSISTENT or INCONSISTENT."
    )
    acceptable_labels: Sequence[str] = field(default_factory=lambda: ("CONSISTENT", "INCONSISTENT"))


@dataclass
class CatchCapConfig:
    """Top-level configuration for the CatchCap pipeline."""

    generator: ModelConfig
    semantic_entropy: SemanticEntropyConfig = field(default_factory=SemanticEntropyConfig)
    logprobs: LogProbConfig = field(default_factory=LogProbConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    judge: Optional[JudgeConfig] = None
    enable_correction: bool = True


