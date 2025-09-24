"""Base classes for web search adapters."""

from __future__ import annotations

import abc
from typing import Iterable, Optional


class WebSearchResult:
    """Simple representation of a web search result."""

    def __init__(self, title: str, url: str, content: str, score: float = 0.0):
        self.title = title
        self.url = url
        self.content = content
        self.score = score


class BaseWebSearch(abc.ABC):
    """Interface for web search services."""

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout: Optional[int] = None,
    ) -> Iterable[WebSearchResult]:
        """Perform a web search and return aggregated results."""


