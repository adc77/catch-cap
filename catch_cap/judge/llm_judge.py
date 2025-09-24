from __future__ import annotations

from ..clients.base import BaseModelClient
from ..config import JudgeConfig
from ..types import JudgeVerdict
from ..exceptions import JudgeError
from .base import BaseJudge


JUDGE_PROMPT_TEMPLATE = """You are asked to evaluate a model response for factual consistency.
Query: {query}

Model response:
{response}

Reference answer:
{reference}

{instructions}
"""


class LLMJudge(BaseJudge):
    """Judge powered by a chosen LLM client."""

    def __init__(self, config: JudgeConfig, client: BaseModelClient):
        self.config = config
        self.client = client

    async def evaluate(self, query: str, response_text: str, reference_text: str) -> JudgeVerdict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            response=response_text,
            reference=reference_text,
            instructions=self.config.instructions,
        )

        generations = await self.client.generate(
            prompt,
            temperature=0.3,
            top_p=1.0,
            max_tokens=120,
            n=1,
            return_logprobs=False,
            extra_args=self.config.model.extra_args
        )
        if not generations:
            raise JudgeError("Judge model returned no output")

        verdict_text = generations[0].text.strip()
        normalized = verdict_text.upper()
        
        # Fix: Check INCONSISTENT first since it contains CONSISTENT
        if "INCONSISTENT" in normalized:
            label = "INCONSISTENT"
        elif "CONSISTENT" in normalized:
            label = "CONSISTENT"
        else:
            label = "UNKNOWN"
            
        is_consistent = label == "CONSISTENT"
        return JudgeVerdict(verdict=label, raw_response=verdict_text, is_consistent=is_consistent)