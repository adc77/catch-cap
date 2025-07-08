from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SearchResult:
    """Represents a search result from the vector database."""
    content: str
    metadata: Dict[str, Any]
    score: float
    id: str

@dataclass
class ResearchReport:
    """Represents a comprehensive research report."""
    query: str
    summary: str
    detailed_analysis: str
    sources: List[Dict[str, Any]]
    sub_queries: List[str]
    confidence_score: float