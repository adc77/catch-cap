from __future__ import annotations

import time
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ..clients.base import BaseModelClient
from ..clients.gemini_client import GeminiModelClient
from ..clients.groq_client import GroqModelClient
from ..clients.openai_client import OpenAIModelClient
from ..confidence import compute_confidence_score, interpret_confidence
from ..config import CatchCapConfig, ModelConfig
from ..detection.logprobs import LogProbDetector
from ..detection.semantic_entropy import SemanticEntropyDetector
from ..exceptions import ProviderNotAvailableError
from ..judge.llm_judge import LLMJudge
from ..logging import logger
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

    def __init__(self, config: CatchCapConfig, auto_load_dotenv: bool = True):
        # Automatically load .env file if available and not disabled
        if auto_load_dotenv and load_dotenv is not None:
            load_dotenv()

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

        # Optional rate limiting
        self.rate_limiter = None
        if config.rate_limit_rpm:
            try:
                from aiolimiter import AsyncLimiter
                # Convert RPM to requests per second for aiolimiter
                max_rate = config.rate_limit_rpm
                time_period = 60  # seconds
                self.rate_limiter = AsyncLimiter(max_rate, time_period)
                logger.info(f"Rate limiting enabled: {max_rate} requests per minute")
            except ImportError:
                logger.warning("aiolimiter not installed - rate limiting disabled")

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
        """
        Run the full hallucination detection pipeline.
        Implements graceful degradation - continues even if some components fail.
        """
        # Apply rate limiting if configured
        if self.rate_limiter:
            async with self.rate_limiter:
                return await self._run_internal(query)
        else:
            return await self._run_internal(query)

    async def _run_internal(self, query: str) -> CatchCapResult:
        """Internal implementation of the detection pipeline."""
        start_time = time.time()
        logger.info(f"Starting detection pipeline for query: {query[:100]}...")

        detection_methods = []
        errors = []

        # Step 1: Generate responses (critical - cannot continue without this)
        try:
            logger.debug(f"Generating {self.config.semantic_entropy.n_responses} responses...")
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
        except Exception as e:
            # Generation failed - return error state
            logger.error(f"Generation failed: {str(e)}")
            return CatchCapResult(
                query=query,
                responses=[],
                semantic_entropy=None,
                logprob_analysis=None,
                judge_verdict=None,
                confabulation_detected=True,  # Fail-safe: assume hallucination
                corrected_answer=None,
                web_answer=None,
                metadata={
                    "error": str(e),
                    "stage": "generation",
                    "reasons": "Generation failed"
                }
            )

        # Step 2: Semantic entropy analysis (optional - graceful degradation)
        semantic_analysis = None
        if self.config.semantic_entropy.enabled:
            try:
                embeddings = await self.embedding_client.embed(
                    [response.text for response in responses],
                    model=self.config.semantic_entropy.embedding_model,
                )
                semantic_analysis = await self.semantic_detector.analyse(responses, embeddings)
                detection_methods.append("semantic_entropy")
                logger.debug(f"Semantic entropy: {semantic_analysis.entropy_score:.3f}")
            except Exception as e:
                error_msg = f"Semantic entropy failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

        # Step 3: Log probability analysis (optional - graceful degradation)
        logprob_analysis = None
        if self.config.logprobs.enabled:
            try:
                primary_response = responses[0]
                logprob_analysis = self.logprob_detector.analyse(primary_response)
                detection_methods.append("logprobs")
                logger.debug(f"Log prob flagged ratio: {logprob_analysis.flagged_token_ratio:.2%}")
            except Exception as e:
                error_msg = f"Log probability analysis failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

        # Step 4: Web search (optional - graceful degradation)
        web_answer = None
        if self.web_search:
            try:
                search_results = list(await self.web_search.search(
                    query,
                    max_results=self.config.web_search.max_results,
                    timeout=self.config.web_search.timeout_seconds,
                ))
                # Synthesize web results into coherent answer
                if self.web_synthesizer and search_results:
                    try:
                        web_answer = await self.web_synthesizer.synthesize(query, search_results)
                    except Exception as e:
                        errors.append(f"Web synthesis failed: {str(e)}")
                        # Fallback: concatenate results
                        web_answer = "\n".join(r.content for r in search_results[:3] if r.content)
                elif search_results:
                    # Fallback: concatenate first few results
                    web_answer = "\n".join(r.content for r in search_results[:3] if r.content)

                if web_answer:
                    detection_methods.append("web_search")
                    logger.debug("Web search completed successfully")
            except Exception as e:
                error_msg = f"Web search failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

        # Step 5: Judge evaluation (optional - graceful degradation)
        judge_verdict = None
        if self.judge and web_answer:
            try:
                judge_verdict = await self.judge.evaluate(query, responses[0].text, web_answer)
                detection_methods.append("judge")
                logger.debug(f"Judge verdict: {judge_verdict.verdict}")
            except Exception as e:
                error_msg = f"Judge evaluation failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

        # Step 6: Aggregate results
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

        # Build preliminary result for confidence scoring
        preliminary_result = CatchCapResult(
            query=query,
            responses=responses,
            semantic_entropy=semantic_analysis,
            logprob_analysis=logprob_analysis,
            judge_verdict=judge_verdict,
            confabulation_detected=confabulation_detected,
            corrected_answer=corrected_answer,
            web_answer=web_answer,
            metadata={},
        )

        # Compute confidence score
        confidence = compute_confidence_score(preliminary_result)
        confidence_level = interpret_confidence(confidence)

        # Log final results
        duration = time.time() - start_time
        if confabulation_detected:
            logger.warning(
                f"Confabulation detected! Reasons: {', '.join(reasons)}. "
                f"Confidence: {confidence_level} ({confidence:.2f}). "
                f"Duration: {duration:.2f}s"
            )
        else:
            logger.info(
                f"No confabulation detected. "
                f"Confidence: {confidence_level} ({confidence:.2f}). "
                f"Duration: {duration:.2f}s"
            )

        metadata = {
            "reasons": ", ".join(reasons) if reasons else "No issues detected",
            "detection_methods": detection_methods,
            "errors": errors if errors else None,
            "detection_time_seconds": round(duration, 3),
            "confidence_score": round(confidence, 3),
            "confidence_level": confidence_level,
        }

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


