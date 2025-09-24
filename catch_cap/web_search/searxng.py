from __future__ import annotations

from typing import Iterable, Optional

import aiohttp

from ..exceptions import WebSearchError
from .base import BaseWebSearch, WebSearchResult


class SearXNGSearch(BaseWebSearch):
    """Adapter for SearXNG metasearch."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def search(
        self, query: str, *, max_results: int, timeout: Optional[int] = None
    ) -> Iterable[WebSearchResult]:
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "results_count": max_results,
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    payload = await response.json()
        except Exception as exc:  # pragma: no cover - network errors
            raise WebSearchError(str(exc)) from exc

        results = []
        for item in payload.get("results", [])[:max_results]:
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                )
            )
        return results


