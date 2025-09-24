# catch_cap

CatchCap helps you detect and reduce LLM hallucinations. It offers two complementary detection tracks and optional auto-correction by grounding answers in the open web.

## Features
- Supports OpenAI, Google Gemini, and Groq text models
- Semantic entropy analysis across n sampled responses
- Real-time log-probability monitoring for token-level suspicion
- Configurable web search grounding with Tavily or SearXNG
- Pluggable LLM-as-a-judge for factual consistency checks
- Optional handoff to grounded web answer when hallucination is detected

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from catch_cap import (
    CatchCap,
    CatchCapConfig,
    ModelConfig,
    SemanticEntropyConfig,
    LogProbConfig,
    WebSearchConfig,
    JudgeConfig,
)


async def main():
    config = CatchCapConfig(
        generator=ModelConfig(provider="openai", name="gpt-4.1-mini", temperature=0.6),
        semantic_entropy=SemanticEntropyConfig(n_responses=4, threshold=0.3),
        logprobs=LogProbConfig(min_logprob=-5.0, fraction_threshold=0.15),
        web_search=WebSearchConfig(provider="tavily", max_results=4),
        judge=JudgeConfig(
            model=ModelConfig(provider="openai", name="gpt-4.1-nano"),
            instructions="Return CONSISTENT or INCONSISTENT only.",
        ),
    )
    detector = CatchCap(config)
    result = await detector.run("Who won the 2025 Nobel Prize in Physics?")

    print("Query:", result.query)
    print("Confabulation detected:", result.confabulation_detected)
    if result.semantic_entropy:
        print("Entropy score:", result.semantic_entropy.entropy_score)
    if result.logprob_analysis:
        print("Flagged ratio:", result.logprob_analysis.flagged_token_ratio)
    if result.judge_verdict:
        print("Judge verdict:", result.judge_verdict.verdict)
    if result.corrected_answer:
        print("Corrected answer:", result.corrected_answer)


asyncio.run(main())
```

## Configuration Notes
- `ModelConfig`: choose any supported provider (`openai`, `gemini`, or `groq`) and model name.
- `SemanticEntropyConfig`: toggles the n-samples workflow; set `enabled=False` to skip.
- `LogProbConfig`: controls token log-prob thresholding; disable by setting `enabled=False`.
- `WebSearchConfig`: select `tavily`, `searxng`, or `none` for grounding.
- `JudgeConfig`: optional; LLM-as-a-judge step runs only when provided.

## Architecture Overview
1. Generate `n` responses using the configured model and collect log probabilities.
2. Compute semantic entropy over response embeddings to estimate confidence.
3. Inspect token log probabilities for low-confidence spans.
4. Run web search (if enabled) for external evidence.
5. Ask the judge model to compare the primary response with the web evidence.
6. Flag confabulation when any detector or judge indicates risk.
7. Optionally surface the web-backed answer as a corrective response.

## Environment
Set the relevant API keys before running:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

## Extending
- Add new providers by subclassing `BaseModelClient` and plugging into `PROVIDER_CLIENTS`.
- Extend web providers by implementing `BaseWebSearch` adapters.
- Customize judging prompts via `JudgeConfig.instructions`.


