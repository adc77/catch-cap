"""OpenAI client implementation."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from openai import AsyncOpenAI

from ..config import ModelConfig
from ..types import GenerationResult
from .base import BaseModelClient


class OpenAIModelClient(BaseModelClient):
    """Client wrapper around OpenAI async SDK."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key)

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
            "model": model_config.name if model_config else "gpt-4.1-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
        }
        params["n"] = max(1, n)
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens
        if extra_args:
            params.update(extra_args)
        if return_logprobs:
            params["logprobs"] = True

        response = await self.client.chat.completions.create(**params)
        results: List[GenerationResult] = []

        for choice in response.choices[:n]:
            output_text = choice.message.content or ""
            logprobs = None
            if return_logprobs and choice.logprobs:
                token_logprobs = [token.logprob for token in choice.logprobs.content]
                logprobs = token_logprobs
            results.append(GenerationResult(text=output_text.strip(), logprobs=logprobs))

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


