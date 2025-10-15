# Changelog

All notable changes to catch-cap will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-15 (Beta)

### Added
- **Structured Logging**: Comprehensive logging throughout the pipeline with configurable levels
- **Confidence Scoring**: Automatic confidence scores (0-1) for detection results with human-readable interpretations
- **Rate Limiting**: Optional rate limiting support via `rate_limit_rpm` config parameter
- **Test Suite**: Foundation test suite with unit tests for confidence, semantic entropy, and log probabilities
- **Graceful Degradation**: Pipeline continues even if individual components fail (e.g., web search timeout)
- **Enhanced Error Handling**: Automatic retry logic with exponential backoff for API calls
- **Detailed Metadata**: Results now include detection methods used, errors encountered, timing, and confidence scores

### Changed
- **BREAKING**: Batch embedding optimization - up to 10x faster embedding generation (automatically batches requests)
- **Improved Judge Parsing**: Robust verdict extraction with regex word boundaries to prevent false matches
- **Enhanced Metadata**: `CatchCapResult.metadata` now includes `detection_time_seconds`, `confidence_score`, `confidence_level`, `detection_methods`, and `errors`

### Fixed
- Judge verdict parsing now correctly handles responses containing both "CONSISTENT" and "INCONSISTENT"
- Removed debug `print()` statement from pipeline (replaced with proper logging)
- Typo in web synthesizer error message (catched_cap/web_search/synthesizer.py:47)

### Performance
- **10x Faster Embeddings**: OpenAI embeddings now batched (up to 2048 texts per request)
- **Reduced API Calls**: Gemini embeddings also batched for significant cost savings
- **Retry Logic**: Failed requests now automatically retry (3 attempts with exponential backoff)

### Documentation
- Added CLAUDE.md for AI assistant guidance
- Added CHANGELOG.md (this file)
- Updated README with v0.2.0 examples

## [0.1.4] - 2025-09-25

### Added
- Initial PyPI release
- Multi-provider support (OpenAI, Gemini, Groq)
- Semantic entropy detection
- Log probability monitoring
- Web search grounding (Tavily, SearXNG)
- LLM-as-a-judge evaluation
- Auto-correction with web-grounded answers

---

## Upgrade Guide (0.1.4 → 0.2.0)

### No Breaking Changes for Basic Usage
If you're using the default configuration, your code will continue to work:

```python
# This still works exactly the same
from catch_cap import CatchCap, CatchCapConfig, ModelConfig

config = CatchCapConfig(
    generator=ModelConfig(provider="openai", name="gpt-4.1-mini")
)
detector = CatchCap(config)
result = await detector.run("query")
```

### New Features to Adopt

#### 1. Confidence Scores
Results now include confidence scoring in metadata:

```python
result = await detector.run("query")
print(f"Confidence: {result.metadata['confidence_level']}")  # "High", "Medium", etc.
print(f"Score: {result.metadata['confidence_score']}")  # 0.0-1.0
```

#### 2. Rate Limiting
Add rate limiting to avoid hitting API limits:

```python
config = CatchCapConfig(
    generator=ModelConfig(provider="openai", name="gpt-4.1-mini"),
    rate_limit_rpm=60,  # Max 60 requests per minute
)
```

#### 3. Logging
Enable debug logging to see pipeline details:

```python
from catch_cap.logging import setup_logger
import logging

setup_logger(level=logging.DEBUG)  # See detailed pipeline execution
```

#### 4. Error Tracking
Check for partial failures:

```python
result = await detector.run("query")
if result.metadata.get("errors"):
    print(f"Warnings: {result.metadata['errors']}")
```

### Performance Improvements
No code changes needed - embeddings are automatically batched for 10x speedup!

### Dependencies
If you manually manage dependencies, add:
- `tenacity>=8.0.0` (retry logic)
- `aiolimiter>=1.1.0` (rate limiting, optional)
