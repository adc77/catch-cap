"""Base client abstractions for model providers."""

from __future__ import annotations

import abc
from typing import Iterable, List, Optional, Sequence

from ..types import GenerationResult


class BaseModelClient(abc.ABC):
    """Interface for model provider clients."""

    @abc.abstractmethod
    async def generate(  # pragma: no cover - interface definition only
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        n: int = 1,
        return_logprobs: bool = False,
        extra_args: Optional[dict] = None,
    ) -> Sequence[GenerationResult]:
        """Generate responses for the given prompt."""

    @abc.abstractmethod
    async def embed(  # pragma: no cover - interface definition only
        self, texts: Iterable[str], *, model: str
    ) -> List[List[float]]:
        """Return embeddings for the given texts."""


