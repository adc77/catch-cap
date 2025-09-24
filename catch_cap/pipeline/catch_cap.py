from __future__ import annotations

from typing import List, Optional

from ..clients.base import BaseModelClient
from ..clients.gemini_client import GeminiModelClient
from ..clients.groq_client import GroqModelClient
from ..clients.openai_client import OpenAIModelClient
from ..config import CatchCapConfig, ModelConfig
from ..detection.logprobs import LogProbDetector
from ..detection.semantic_entropy import SemanticEntropyDetector
from ..exceptions import ProviderNotAvailableError
from ..judge.llm_judge import LLMJudge
from ..types import CatchCapResult
from ..web_search.base import BaseWebSearch
from ..web_search.searxng import SearXNGSearch
from ..web_search.tavily import TavilySearch
from ..web_search.synthesizer import WebResultSynthesizer


PROVIDER_CLIENTS = {
    "openai": OpenAIModelClient,
    "gemini": GeminiModelClient,
    "groq": GroqModelClient,
}


class CatchCap:
    """Main entry point for confabulation detection."""

    def __init__(self, config: CatchCapConfig):
        self.config = config
        self.generator_client = self._build_client(config.generator)
        self.embedding_client = self._build_client(
            ModelConfig(provider=config.semantic_entropy.embedding_provider, name=config.semantic_entropy.embedding_model)
        )
        self.semantic_detector = SemanticEntropyDetector(config.semantic_entropy)
        self.logprob_detector = LogProbDetector(config.logprobs)
        self.web_search = self._build_web_search()
        self.web_synthesizer = self._build_web_synthesizer()
        self.judge = self._build_judge()

    def _build_client(self, model_config: ModelConfig) -> BaseModelClient:
        client_cls = PROVIDER_CLIENTS.get(model_config.provider)
        if not client_cls:
            raise ProviderNotAvailableError(f"Provider {model_config.provider} is not supported")
        return client_cls()

    def _build_web_search(self) -> Optional[BaseWebSearch]:
        if self.config.web_search.provider == "tavily":
            return TavilySearch()
        if self.config.web_search.provider == "searxng":
            return SearXNGSearch(self.config.web_search.searxng_url)
        return None
    
    def _build_web_synthesizer(self) -> Optional[WebResultSynthesizer]:
        if not self.config.web_search.synthesizer_model:
            return None
        client = self._build_client(self.config.web_search.synthesizer_model)
        return WebResultSynthesizer(client, self.config.web_search.synthesizer_model)

    def _build_judge(self) -> Optional[LLMJudge]:
        if not self.config.judge:
            return None
        client = self._build_client(self.config.judge.model)
        return LLMJudge(self.config.judge, client)

    async def run(self, query: str) -> CatchCapResult:
        responses = await self.generator_client.generate(
            query,
            temperature=self.config.generator.temperature,
            top_p=self.config.generator.top_p,
            max_tokens=self.config.generator.max_tokens,
            n=self.config.semantic_entropy.n_responses,
            return_logprobs=self.config.logprobs.enabled,
            extra_args=self.config.generator.extra_args,
            model_config=self.config.generator,
        )
        semantic_analysis = None
        if self.config.semantic_entropy.enabled:
            embeddings = await self.embedding_client.embed(
                [response.text for response in responses],
                model=self.config.semantic_entropy.embedding_model,
            )
            semantic_analysis = await self.semantic_detector.analyse(responses, embeddings)

        logprob_analysis = None
        if self.config.logprobs.enabled:
            primary_response = responses[0]
            logprob_analysis = self.logprob_detector.analyse(primary_response)

        web_answer = None
        search_content = []
        if self.web_search:
            search_results = list(await self.web_search.search(
                query,
                max_results=self.config.web_search.max_results,
                timeout=self.config.web_search.timeout_seconds,
            ))
            # for result in search_results:
            #     search_content.append(result.content)
            # web_answer = "\n".join(search_content[:3]) if search_content else None
            # Synthesize web results into coherent answer
            if self.web_synthesizer and search_results:
                web_answer = await self.web_synthesizer.synthesize(query, search_results)
            elif search_results:
                # Fallback: concatenate first few results
                web_answer = "\n".join(r.content for r in search_results[:3] if r.content)

        judge_verdict = None
        if self.judge and web_answer:
            judge_verdict = await self.judge.evaluate(query, responses[0].text, web_answer)
            print(judge_verdict)

        confabulation_detected = False
        reasons: List[str] = []
        if semantic_analysis and not semantic_analysis.is_confident:
            confabulation_detected = True
            reasons.append("High entropy")
        if logprob_analysis and logprob_analysis.is_suspicious:
            confabulation_detected = True
            reasons.append("Low log probabilities")
        if judge_verdict and not judge_verdict.is_consistent:
            confabulation_detected = True
            reasons.append("Judge marked inconsistent")

        corrected_answer = None
        if self.config.enable_correction and confabulation_detected and web_answer:
            corrected_answer = web_answer

        metadata = {"reasons": ", ".join(reasons)}

        return CatchCapResult(
            query=query,
            responses=responses,
            semantic_entropy=semantic_analysis,
            logprob_analysis=logprob_analysis,
            judge_verdict=judge_verdict,
            confabulation_detected=confabulation_detected,
            corrected_answer=corrected_answer,
            web_answer=web_answer,
            metadata=metadata,
        )


