from __future__ import annotations

from typing import Iterable

from ..clients.base import BaseModelClient
from ..config import ModelConfig
from .base import WebSearchResult


WEB_SYNTHESIS_PROMPT = """Based on the following web search results, provide a clear and accurate answer to the query: "{query}"

Web Search Results:
{search_results}

Instructions:
- Count carefully and double-check any numerical answers
- Only use information that appears in the search results
- If search results contain conflicting information, mention this
- Be precise and factual
- Do not mention "based on search results" or similar phrases
- For counting questions, be extra careful to count correctly

Answer:"""


class WebResultSynthesizer:
    """Synthesizes web search results into a coherent answer using an LLM."""

    def __init__(self, client: BaseModelClient, model_config: ModelConfig):
        self.client = client
        self.model_config = model_config

    async def synthesize(self, query: str, results: Iterable[WebSearchResult]) -> str:
        """Synthesize web search results into a coherent answer."""
        
        # Format search results
        search_content = []
        for i, result in enumerate(results, 1):
            if result.content.strip():
                # Clean and limit content
                clean_content = result.content.replace("\n", " ").strip()
                if len(clean_content) > 300:
                    clean_content = clean_content[:300] + "..."
                search_content.append(f"{i}. {result.title}\n   Content: {clean_content}")
        
        if not search_content:
            return "No relevant web content found to answer the query."
        
        formatted_results = "\n\n".join(search_content)
        
        prompt = WEB_SYNTHESIS_PROMPT.format(
            query=query,
            search_results=formatted_results
        )
        
        responses = await self.client.generate(
            prompt,
            temperature=0.1,  # Lower temperature for factual synthesis
            top_p=0.9,
            max_tokens=150,
            n=1,
            return_logprobs=False,
            model_config=self.model_config,
        )
        
        return responses[0].text if responses else "Unable to synthesize web results."