"""Groq client implementation."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from groq import AsyncGroq

from ..config import ModelConfig
from ..types import GenerationResult
from .base import BaseModelClient


class GroqModelClient(BaseModelClient):
    """Client wrapper around Groq async SDK."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncGroq(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        n: int = 1,
        return_logprobs: bool = False,
        extra_args: Optional[dict] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> Sequence[GenerationResult]:
        params = {
            "model": model_config.name if model_config else "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,  # Groq only supports n=1
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if extra_args:
            params.update(extra_args)

        results: List[GenerationResult] = []
        
        # Make multiple calls since Groq doesn't support n > 1
        for _ in range(n):
            response = await self.client.chat.completions.create(**params)
            
            for choice in response.choices:
                text = choice.message.content or ""
                logprobs = None
                if return_logprobs and choice.logprobs:
                    logprobs = [token.logprob for token in choice.logprobs.content]
                results.append(GenerationResult(text=text.strip(), logprobs=logprobs))
        
        return results

    async def embed(self, texts: Iterable[str], *, model: str) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            response = await self.client.embeddings.create(
                model=model,
                input=text,
            )
            embeddings.append(response.data[0].embedding)
        return embeddings


