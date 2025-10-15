"""Gemini client implementation."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

from google import genai
from google.genai import types

from ..config import ModelConfig
from ..types import GenerationResult
from .base import BaseModelClient


class GeminiModelClient(BaseModelClient):
    """Client wrapper around Gemini async SDK."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            # Try GEMINI_API_KEY first, then GOOGLE_API_KEY as fallback
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        self.client = genai.Client(api_key=api_key)

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
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            candidate_count=n,
        )
        if max_tokens is not None:
            config.max_output_tokens = max_tokens
        if extra_args:
            for key, value in extra_args.items():
                setattr(config, key, value)

        response = await self.client.aio.models.generate_content(
            model=model_config.name if model_config else "gemini-2.0-flash",
            contents=[prompt],
            config=config,
        )

        results: List[GenerationResult] = []
        for candidate in response.candidates[:n]:
            text = candidate.content.parts[0].text if candidate.content.parts else ""
            logprobs = None
            if return_logprobs and candidate.generation_config and candidate.generation_config.logprobs:
                logprobs = candidate.generation_config.logprobs
            results.append(GenerationResult(text=text.strip(), logprobs=logprobs))
            # results.append(GenerationResult(text=text.strip()))
        return results

    async def embed(self, texts: Iterable[str], *, model: str) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        Batches requests for improved performance.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Gemini supports batch embedding
        response = await self.client.aio.models.embed_content(
            model=model,
            contents=texts_list,
        )

        return [emb.values for emb in response.embeddings]


