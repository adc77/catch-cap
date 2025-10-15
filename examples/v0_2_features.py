"""
Example showcasing new features in v0.2.2.

Demonstrates:
- Confidence scoring
- Rate limiting
- Structured logging
- Error handling and graceful degradation
"""

import asyncio
import logging
from catch_cap import (
    CatchCap,
    CatchCapConfig,
    ModelConfig,
    SemanticEntropyConfig,
    WebSearchConfig,
    JudgeConfig,
)
from catch_cap.logging import setup_logger


async def main():
    # Enable debug logging to see detailed pipeline execution
    setup_logger(level=logging.DEBUG)

    # Configure with new v0.2.2 features
    config = CatchCapConfig(
        generator=ModelConfig(
            provider="openai",
            name="gpt-4.1-mini",
            temperature=0.6
        ),
        semantic_entropy=SemanticEntropyConfig(
            n_responses=3,
            threshold=0.3
        ),
        web_search=WebSearchConfig(
            provider="tavily",
            max_results=5,
            synthesizer_model=ModelConfig(provider="openai", name="gpt-4.1-nano")
        ),
        judge=JudgeConfig(
            model=ModelConfig(provider="openai", name="gpt-4.1-nano")
        ),
        # NEW in v0.2.0: Rate limiting
        rate_limit_rpm=30,  # Max 30 requests per minute
    )

    detector = CatchCap(config)

    # Test query
    query = "How many r's are in the word strawberry?"

    print("="*60)
    print("Running hallucination detection with v0.2.2 features...")
    print("="*60)

    result = await detector.run(query)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nQuery: {result.query}")
    print(f"Model Response: {result.responses[0].text}")

    # NEW in v0.2.2: Confidence scoring
    print(f"\n[CONFIDENCE ANALYSIS]")
    print(f"  Confidence Level: {result.metadata.get('confidence_level')}")
    print(f"  Confidence Score: {result.metadata.get('confidence_score')}")

    # Detection results
    print(f"\n[DETECTION RESULTS]")
    print(f"  Confabulation Detected: {result.confabulation_detected}")
    print(f"  Reasons: {result.metadata.get('reasons')}")

    # NEW in v0.2.2: Detection methods used
    print(f"\n[DETECTION METHODS USED]")
    for method in result.metadata.get('detection_methods', []):
        print(f"  ✓ {method}")

    # Semantic entropy
    if result.semantic_entropy:
        print(f"\n[SEMANTIC ENTROPY]")
        print(f"  Entropy Score: {result.semantic_entropy.entropy_score:.3f}")
        print(f"  Model Confident: {result.semantic_entropy.is_confident}")

    # Log probabilities
    if result.logprob_analysis:
        print(f"\n[LOG PROBABILITIES]")
        print(f"  Flagged Token Ratio: {result.logprob_analysis.flagged_token_ratio:.2%}")
        print(f"  Flagged Token Count: {result.logprob_analysis.flagged_token_count}")

    # Judge verdict
    if result.judge_verdict:
        print(f"\n[JUDGE VERDICT]")
        print(f"  Verdict: {result.judge_verdict.verdict}")
        print(f"  Is Consistent: {result.judge_verdict.is_consistent}")

    # Web answer
    if result.web_answer:
        print(f"\n[WEB-GROUNDED ANSWER]")
        print(f"  {result.web_answer}")

    # Corrected answer
    if result.corrected_answer:
        print(f"\n[CORRECTED ANSWER]")
        print(f"  {result.corrected_answer}")

    # NEW in v0.2.2: Timing and error tracking
    print(f"\n[PERFORMANCE]")
    print(f"  Detection Time: {result.metadata.get('detection_time_seconds')}s")

    if result.metadata.get('errors'):
        print(f"\n[WARNINGS/ERRORS]")
        for error in result.metadata.get('errors', []):
            print(f"  ⚠ {error}")

    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(main())
