class CatchCapError(Exception):
    """Base exception for the library."""


class ProviderNotAvailableError(CatchCapError):
    """Raised when a requested provider is not available."""


class WebSearchError(CatchCapError):
    """Raised when web search fails."""


class JudgeError(CatchCapError):
    """Raised when LLM-as-a-judge fails."""


