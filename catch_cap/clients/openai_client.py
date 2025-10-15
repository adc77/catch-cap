"""OpenAI client implementation."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..config import ModelConfig
from ..exceptions import CatchCapError
from ..types import GenerationResult
from .base import BaseModelClient


class OpenAIModelClient(BaseModelClient):
    """Client wrapper around OpenAI async SDK."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        self.client = AsyncOpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
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
        try:
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
        except Exception as e:
            raise CatchCapError(f"OpenAI generation failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def embed(self, texts: Iterable[str], *, model: str) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        Batches requests for significant cost and performance improvements.
        OpenAI supports up to 2048 texts per batch.
        """
        try:
            texts_list = list(texts)
            if not texts_list:
                return []

            # Single batch if small enough (OpenAI limit: 2048 texts per request)
            if len(texts_list) <= 2048:
                response = await self.client.embeddings.create(
                    model=model,
                    input=texts_list,
                )
                return [item.embedding for item in response.data]

            # For larger batches, split into chunks
            embeddings: List[List[float]] = []
            chunk_size = 2048
            for i in range(0, len(texts_list), chunk_size):
                chunk = texts_list[i:i + chunk_size]
                response = await self.client.embeddings.create(
                    model=model,
                    input=chunk,
                )
                embeddings.extend([item.embedding for item in response.data])

            return embeddings
        except Exception as e:
            raise CatchCapError(f"OpenAI embedding failed: {str(e)}") from e


