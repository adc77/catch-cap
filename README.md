# catch-cap

for detecting and reducing hallucinations in Large Language Model responses through semantic entropy analysis, log-probability monitoring, and web-grounded fact-checking.

## Installation

```bash
pip install catch-cap
```

## API Keys Setup

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key" 
export GROQ_API_KEY="your-groq-key"
export TAVILY_API_KEY="your-tavily-key"
```

Or use a `.env` file:
```env
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
TAVILY_API_KEY=your-tavily-key
```

## What's New in v0.2.0

**catch-cap v0.2.0** is now available – a major update transforming catch-cap into a production-ready hallucination detection middleware. This release brings reliability, speed, and better insight for anyone working with LLM outputs.

**Highlights:**
- **Confidence Scoring:** Each detection gives a 0–1 confidence score and a human-readable level ("High", "Medium", "Low").
- **Rate Limiting:** Throttle model/API usage to prevent overages and stay within quotas.
- **Structured Logging:** New logging for full pipeline observability and debug support.
- **Graceful Degradation:** If a component fails (e.g., web search times out), detection keeps going using available data.
- **Automatic Retries:** All API/network calls retry on transient errors, with exponential backoff.
- **10x Faster Embeddings:** Embeddings are batched for performance and cost-efficiency.
- **Extensive Metadata:** Results include reasons, detection time, and methods used.

For full details, new configuration options, and migration guidance, **see the [v0.2.0 Release Notes](V0.2.0_RELEASE_NOTES.md)**.


## Quick Start

```python
import asyncio
from catch_cap import CatchCap, CatchCapConfig, ModelConfig

async def main():
    config = CatchCapConfig(
        generator=ModelConfig(provider="openai", name="gpt-4.1-mini"),
    )
    
    detector = CatchCap(config)
    result = await detector.run("How many r's are there in strawberry?")
    
    print(f"Confabulation detected: {result.confabulation_detected}")
    if result.corrected_answer:
        print(f"Corrected answer: {result.corrected_answer}")

asyncio.run(main())
```

## Supported Models

### OpenAI
**Text Generation:** all models except thinking models
**Embeddings:** `text-embedding-3-large`, `text-embedding-3-small`  
**Log Probabilities:** Supported

### Google Gemini
**Text Generation:** all models except thinking models
**Embeddings:** `text-embedding-004`, `embedding-001`  
**Log Probabilities:** Not supported

### Groq
**Text Generation:** all models except thinking models
**Embeddings:** Use OpenAI or Gemini  
**Log Probabilities:** Limited support

## Configuration

### Basic Configuration
```python
from catch_cap import CatchCapConfig, ModelConfig

config = CatchCapConfig(
    generator=ModelConfig(provider="openai", name="gpt-4.1-mini")
)
```

### Full Configuration
```python
from catch_cap import *

config = CatchCapConfig(
    generator=ModelConfig(
        provider="openai", 
        name="gpt-4.1-mini",
        temperature=0.7,
        max_tokens=500
    ),
    semantic_entropy=SemanticEntropyConfig(
        enabled=True,
        n_responses=3,
        threshold=0.3,
        embedding_model="text-embedding-3-large",
        embedding_provider="openai"
    ),
    logprobs=LogProbConfig(
        enabled=True,
        min_logprob=-5.0,
        fraction_threshold=0.15,
        min_flagged_tokens=5
    ),
    web_search=WebSearchConfig(
        provider="tavily",
        max_results=3,
        synthesizer_model=ModelConfig(provider="openai", name="gpt-4.1-nano")
    ),
    judge=JudgeConfig(
        model=ModelConfig(provider="openai", name="gpt-4.1-nano"),
        instructions="Compare responses for factual accuracy. Return CONSISTENT or INCONSISTENT only."
    ),
    enable_correction=True
)
```

## Usage Examples

### Minimal Setup
```python
config = CatchCapConfig(
    generator=ModelConfig(provider="openai", name="gpt-4.1-mini")
)
detector = CatchCap(config)
result = await detector.run("How many r's are there in strawberry?")
```

### Semantic Entropy Only
```python
config = CatchCapConfig(
    generator=ModelConfig(provider="gemini", name="gemini-2.0-flash"),
    semantic_entropy=SemanticEntropyConfig(n_responses=5, threshold=0.2),
    logprobs=LogProbConfig(enabled=False),
    web_search=WebSearchConfig(provider="none")
)
```

### Maximum Detection
```python
config = CatchCapConfig(
    generator=ModelConfig(provider="openai", name="gpt-4.1-mini"),
    semantic_entropy=SemanticEntropyConfig(n_responses=5, threshold=0.2),
    logprobs=LogProbConfig(min_logprob=-4.0, fraction_threshold=0.1),
    web_search=WebSearchConfig(
        provider="tavily",
        synthesizer_model=ModelConfig(provider="openai", name="gpt-4.1-nano")
    ),
    judge=JudgeConfig(model=ModelConfig(provider="openai", name="gpt-4"))
)
```

### Production Setup
```python
from dotenv import load_dotenv
load_dotenv()

config = CatchCapConfig(
    generator=ModelConfig(provider="openai", name="gpt-4.1-mini", temperature=0.7),
    semantic_entropy=SemanticEntropyConfig(n_responses=3, threshold=0.3),
    web_search=WebSearchConfig(
        provider="tavily",
        synthesizer_model=ModelConfig(provider="openai", name="gpt-4.1-nano")
    ),
    judge=JudgeConfig(model=ModelConfig(provider="openai", name="gpt-4.1-nano"))
)
```

## Result Analysis

```python
result = await detector.run("Your_query_here")

# Basic results
print(f"Query: {result.query}")
print(f"Confabulation detected: {result.confabulation_detected}")
print(f"Original response: {result.responses[0].text}")

# Semantic entropy analysis
if result.semantic_entropy:
    print(f"Entropy score: {result.semantic_entropy.entropy_score}")
    print(f"Model confident: {result.semantic_entropy.is_confident}")

# Log probability analysis  
if result.logprob_analysis:
    print(f"Suspicious tokens ratio: {result.logprob_analysis.flagged_token_ratio}")
    print(f"Total flagged tokens: {result.logprob_analysis.flagged_token_count}")

# Judge verdict
if result.judge_verdict:
    print(f"Judge verdict: {result.judge_verdict.verdict}")
    print(f"Factually consistent: {result.judge_verdict.is_consistent}")

# Corrections
if result.corrected_answer:
    print(f"Corrected answer: {result.corrected_answer}")
```

## Error Handling

```python
from catch_cap.exceptions import CatchCapError, ProviderNotAvailableError

try:
    result = await detector.run(query)
except ProviderNotAvailableError:
    print("Model provider not available")
except CatchCapError as e:
    print(f"Detection error: {e}")
```

## Batch Processing

```python
queries = ["Query 1", "Query 2", "Query 3"]
results = []

for query in queries:
    result = await detector.run(query)
    results.append(result)
    print(f"Query: {query}")
    print(f"Confabulation detected: {result.confabulation_detected}")
```

## Configuration Reference

### ModelConfig
```python
ModelConfig(
    provider="openai",     # "openai", "gemini", or "groq"
    name="gpt-4.1-mini",         # Model name
    temperature=0.7,      # Sampling temperature (0.0-2.0)
    top_p=0.9,           # Nucleus sampling
    max_tokens=1000,     # Max output tokens
    extra_args={}        # Provider-specific args
)
```

### SemanticEntropyConfig
```python
SemanticEntropyConfig(
    enabled=True,                    # Enable semantic entropy detection
    n_responses=5,                   # Number of responses to generate
    threshold=0.25,                  # Entropy threshold (lower = more confident)
    embedding_model="text-embedding-3-small",
    embedding_provider="openai"
)
```

### LogProbConfig
```python
LogProbConfig(
    enabled=True,              # Enable log-prob analysis
    min_logprob=-4.5,         # Token log-prob threshold
    fraction_threshold=0.2,    # Fraction of flagged tokens needed
    min_flagged_tokens=5       # Minimum flagged tokens to trigger
)
```

### WebSearchConfig
```python
WebSearchConfig(
    provider="tavily",         # "tavily", "searxng", or "none"
    max_results=5,             # Number of search results
    timeout_seconds=20,        # Search timeout
    searxng_url="http://localhost:8080/search",  # For SearXNG
    synthesizer_model=ModelConfig(provider="openai", name="gpt-4.1-mini")
)
```

### JudgeConfig
```python
JudgeConfig(
    model=ModelConfig(provider="openai", name="gpt-4.1-mini"),
    instructions="Compare responses for factual accuracy. Return CONSISTENT or INCONSISTENT only.",
    acceptable_labels=("CONSISTENT", "INCONSISTENT")
)
```

## License

MIT License
