# from tavily import AsyncTavilyClient
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = AsyncTavilyClient(os.getenv("TAVILY_API_KEY"))

# async def main():
#     response = await client.search("Who is Cristiano Ronaldo?", include_answer=True)
#     print(response)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

# # pip install tavily-python 



import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tavily import AsyncTavilyClient
from dotenv import load_dotenv

from models.token_tracker import TokenTracker

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TavilySearchResult:
    """Enhanced Tavily search result with metadata"""
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None

@dataclass
class TavilyResponse:
    """Structured Tavily response with comprehensive data"""
    query: str
    answer: str
    results: List[TavilySearchResult]
    follow_up_questions: Optional[List[str]]
    images: List[str]
    response_time: float
    request_id: str
    total_results: int
    error: Optional[str] = None

class TavilySearchTool:
    """
    Robust Tavily search tool with error handling, logging, and token tracking
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Tavily search tool
        
        Args:
            api_key: Tavily API key (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        self.client = AsyncTavilyClient(self.api_key)
        self.token_tracker = TokenTracker()
        
    async def search(self, 
                    query: str,
                    search_depth: str = "basic",
                    topic: str = "general",
                    max_results: int = 5,
                    include_images: bool = False,
                    include_answer: bool = True,
                    include_raw_content: bool = False,
                    include_domains: Optional[List[str]] = None,
                    exclude_domains: Optional[List[str]] = None) -> TavilyResponse:
        """
        Perform a comprehensive Tavily search with robust error handling
        
        Args:
            query: Search query
            search_depth: Search depth ("basic" or "advanced")
            topic: Search topic for optimization
            max_results: Maximum number of results
            include_images: Whether to include images
            include_answer: Whether to include generated answer
            include_raw_content: Whether to include raw HTML content
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            
        Returns:
            TavilyResponse object with structured results
        """
        try:
            logger.info(f"Starting Tavily search for: {query}")
            
            # Build search parameters
            search_params = {
                "query": query,
                "search_depth": search_depth,
                "topic": topic,
                "max_results": max_results,
                "include_images": include_images,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content
            }
            
            # Add domain filters if provided
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
            # Perform the search
            response = await self.client.search(**search_params)
            
            # Track token usage (estimated)
            if include_answer and response.get('answer'):
                self.token_tracker.track_openai_chat(
                    input_text=query,
                    output_text=response['answer'],
                    model="tavily-search"
                )
            
            # Process and structure the response
            tavily_response = self._process_response(response, query)
            
            logger.info(f"Search completed: {len(tavily_response.results)} results found")
            return tavily_response
            
        except Exception as e:
            logger.error(f"Error in Tavily search: {e}")
            return TavilyResponse(
                query=query,
                answer=f"Error occurred while searching: {str(e)}",
                results=[],
                follow_up_questions=None,
                images=[],
                response_time=0.0,
                request_id="",
                total_results=0,
                error=str(e)
            )
    
    def _process_response(self, response: Dict[str, Any], query: str) -> TavilyResponse:
        """Process raw Tavily response into structured format"""
        try:
            # Extract results
            results = []
            raw_results = response.get('results', [])
            
            for result in raw_results:
                tavily_result = TavilySearchResult(
                    title=result.get('title', ''),
                    url=result.get('url', ''),
                    content=result.get('content', ''),
                    score=result.get('score', 0.0),
                    raw_content=result.get('raw_content')
                )
                results.append(tavily_result)
            
            # Create structured response
            return TavilyResponse(
                query=response.get('query', query),
                answer=response.get('answer', ''),
                results=results,
                follow_up_questions=response.get('follow_up_questions'),
                images=response.get('images', []),
                response_time=response.get('response_time', 0.0),
                request_id=response.get('request_id', ''),
                total_results=len(results)
            )
            
        except Exception as e:
            logger.error(f"Error processing Tavily response: {e}")
            return TavilyResponse(
                query=query,
                answer="Error processing search results",
                results=[],
                follow_up_questions=None,
                images=[],
                response_time=0.0,
                request_id="",
                total_results=0,
                error=f"Processing error: {str(e)}"
            )
    
    async def search_with_context(self, 
                                 query: str,
                                 context: str,
                                 **kwargs) -> TavilyResponse:
        """
        Perform search with additional context
        
        Args:
            query: Search query
            context: Additional context to improve search
            **kwargs: Additional search parameters
            
        Returns:
            TavilyResponse object
        """
        enhanced_query = f"{query}. Context: {context}"
        return await self.search(enhanced_query, **kwargs)
    
    async def multi_query_search(self, 
                                queries: List[str],
                                **kwargs) -> List[TavilyResponse]:
        """
        Perform multiple searches concurrently
        
        Args:
            queries: List of search queries
            **kwargs: Additional search parameters
            
        Returns:
            List of TavilyResponse objects
        """
        tasks = [self.search(query, **kwargs) for query in queries]
        return await asyncio.gather(*tasks)
    
    def format_response(self, response: TavilyResponse, 
                       show_detailed: bool = False,
                       show_metadata: bool = True) -> str:
        """
        Format Tavily response for display
        
        Args:
            response: TavilyResponse object
            show_detailed: Whether to show detailed content
            show_metadata: Whether to show metadata
            
        Returns:
            Formatted string
        """
        if response.error:
            return f"Error searching for '{response.query}': {response.error}"
        
        # Start with query and answer
        output = f"**Query:** {response.query}\n\n"
        
        if response.answer:
            output += f"**Answer:** {response.answer}\n\n"
        
        # Add metadata if requested
        if show_metadata:
            output += (
                f"**Search Metadata:**\n"
                f"Total results: {response.total_results}\n"
                f"Response time: {response.response_time:.2f}s\n"
                f"Request ID: {response.request_id}\n\n"
            )
        
        # Add results
        if response.results:
            output += "**Sources:**\n"
            for i, result in enumerate(response.results, 1):
                output += f"{i}. **{result.title}**\n"
                output += f"   URL: {result.url}\n"
                output += f"   Score: {result.score:.3f}\n"
                
                if show_detailed and result.content:
                    # Truncate content for display
                    content = result.content[:300] + "..." if len(result.content) > 300 else result.content
                    output += f"   Content: {content}\n"
                
                output += "\n"
        
        # Add follow-up questions if available
        if response.follow_up_questions:
            output += "**Follow-up Questions:**\n"
            for question in response.follow_up_questions:
                output += f"- {question}\n"
            output += "\n"
        
        # Add images if available
        if response.images:
            output += f"**Images:** {len(response.images)} images found\n"
        
        return output
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return self.token_tracker.get_detailed_breakdown()
    
    def print_token_summary(self):
        """Print token usage summary"""
        self.token_tracker.print_summary()

async def main():
    """Example usage of the TavilySearchTool"""
    
    try:
        # Initialize the tool
        search_tool = TavilySearchTool()
        
        # Example queries
        queries = [
            "How many R's are there in strawberry?"
        ]
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"SEARCHING: {query}")
            print('='*80)
            
            # Perform search
            response = await search_tool.search(
                query=query,
                max_results=5,
                include_answer=True,
                search_depth="basic"
            )
            
            # Format and display results
            formatted_output = search_tool.format_response(
                response, 
                show_detailed=True,
                show_metadata=True
            )
            print(formatted_output)
        
        # Show token usage summary
        print(f"\n{'='*80}")
        print("TOKEN USAGE SUMMARY")
        print('='*80)
        search_tool.print_token_summary()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
