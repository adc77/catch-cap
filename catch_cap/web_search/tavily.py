from __future__ import annotations

from typing import Iterable, Optional

from tavily import AsyncTavilyClient

from ..exceptions import WebSearchError
from .base import BaseWebSearch, WebSearchResult


class TavilySearch(BaseWebSearch):
    """Adapter for Tavily search API."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncTavilyClient(api_key)

    async def search(
        self, query: str, *, max_results: int, timeout: Optional[int] = None
    ) -> Iterable[WebSearchResult]:
        try:
            response = await self.client.search(
                query,
                max_results=max_results,
                include_answer=True,
                include_raw_content=True,
            )
        except Exception as exc:  # pragma: no cover - network error branch
            raise WebSearchError(str(exc)) from exc

        results = []
        answer_text = None

        if isinstance(response, dict):
            raw_results = response.get("results", []) or []
            answer_text = response.get("answer")
            for item in raw_results[:max_results]:
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("raw_content") or item.get("content") or ""
                score = float(item.get("score", 0.0))
                results.append(WebSearchResult(title=title, url=url, content=content, score=score))
        else:
            raw_results = getattr(response, "results", []) or []
            answer_text = getattr(response, "answer", None)
            for item in raw_results[:max_results]:
                title = getattr(item, "title", "")
                url = getattr(item, "url", "")
                content = getattr(item, "raw_content", None) or getattr(item, "content", "")
                score = getattr(item, "score", 0.0)
                results.append(WebSearchResult(title=title, url=url, content=content or "", score=score))

        if answer_text:
            results.insert(
                0,
                WebSearchResult(
                    title="Answer",
                    url="",
                    content=str(answer_text),
                    score=1.0,
                ),
            )
        return results


