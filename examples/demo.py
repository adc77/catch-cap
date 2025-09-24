import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catch_cap import (
    CatchCap,
    CatchCapConfig,
    ModelConfig,
    SemanticEntropyConfig,
    LogProbConfig,
    WebSearchConfig,
    JudgeConfig,
)
from dotenv import load_dotenv


async def main():
    load_dotenv()
    config = CatchCapConfig(
        generator=ModelConfig(provider="gemini", name="gemini-2.5-flash-preview-05-20", temperature=0.6),
        semantic_entropy=SemanticEntropyConfig(n_responses=4, threshold=0.5),
        logprobs=LogProbConfig(enabled=False, min_logprob=-5.0, fraction_threshold=0.15),
        web_search=WebSearchConfig(
            provider="tavily", 
            max_results=5,
            synthesizer_model=ModelConfig(provider="gemini", name="gemini-2.5-flash-preview-05-20", temperature=0.1)
        ),
        judge=JudgeConfig(
            model=ModelConfig(provider="gemini", name="gemini-2.5-flash-preview-05-20")
        ),
    )
    detector = CatchCap(config)
    result = await detector.run("How many r's are there in strawberry?")

    print("=== ANALYSIS RESULTS ===")
    print("Query:", result.query)
    print("Model response:", result.responses[0].text)
    print()
    print("=== WEB SEARCH ===")
    print("Web answer:", result.web_answer)
    print()
    print("=== JUDGE EVALUATION ===")
    if result.judge_verdict:
        print("Judge raw response:", repr(result.judge_verdict.raw_response))
        print("Judge verdict:", result.judge_verdict.verdict)
        print("Is consistent:", result.judge_verdict.is_consistent)
    print()
    print("=== FINAL VERDICT ===")
    print("Confabulation detected:", result.confabulation_detected)
    if result.corrected_answer:
        print("Corrected answer:", result.corrected_answer)


asyncio.run(main())