# import asyncio
# import logging
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass

# from searxng_service import SearXNGClient
# from browsing_agent import BrowsingAgent
# from clients.openai_client import OpenAIClient
# from clients.gemini_client import GeminiClient
# from models.token_tracker import TokenTracker

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# @dataclass
# class WebSearchResult:
#     """Enhanced web search result with relevance scoring"""
#     title: str
#     url: str
#     snippet: str
#     engine: str
#     relevance_score: float = 0.0
#     content: Optional[str] = None
#     is_relevant: bool = False

# class IntelligentWebSearchTool:
#     """
#     Intelligent web search tool that combines SearXNG search with content relevance filtering
#     """
    
#     def __init__(self, 
#                  searxng_url: str = "https://2wvev6223syg.share.zrok.io/",
#                  relevance_threshold: float = 0.6,
#                  max_concurrent: int = 5,
#                  initial_chars: int = 2000):
#         """
#         Initialize the intelligent web search tool
        
#         Args:
#             searxng_url: SearXNG instance URL
#             relevance_threshold: Minimum relevance score for content filtering
#             max_concurrent: Maximum concurrent requests for content analysis
#             initial_chars: Number of characters for initial relevance check
#         """
#         # Initialize token tracker
#         self.token_tracker = TokenTracker()
        
#         # Initialize clients
#         self.openai_client = OpenAIClient(token_tracker=self.token_tracker)
#         self.gemini_client = GeminiClient(token_tracker=self.token_tracker)
        
#         # Initialize services
#         self.searxng_client = SearXNGClient(searxng_url=searxng_url)
#         self.browsing_agent = BrowsingAgent(
#             openai_client=self.openai_client,
#             gemini_client=self.gemini_client,
#             initial_chars=initial_chars,
#             relevance_threshold=relevance_threshold,
#             max_concurrent=max_concurrent
#         )
        
#         self.relevance_threshold = relevance_threshold
        
#     async def search(self, 
#                     query: str, 
#                     num_results: int = 10, 
#                     engines: str = None,
#                     relevance_threshold: Optional[float] = None,
#                     include_content: bool = True) -> Dict[str, Any]:
#         """
#         Perform intelligent web search with content relevance filtering
        
#         Args:
#             query: Search query
#             num_results: Number of initial search results to fetch
#             engines: Specific search engines to use (comma-separated)
#             relevance_threshold: Override default relevance threshold
#             include_content: Whether to include extracted content in results
#             max_content_length: Maximum content length to include in results
            
#         Returns:
#             Dictionary containing search results and metadata
#         """
#         try:
#             logger.info(f"Starting intelligent web search for: {query}")
            
#             # Use provided threshold or default
#             threshold = relevance_threshold or self.relevance_threshold
#             self.browsing_agent.relevance_threshold = threshold
            
#             # Step 1: Get initial search results from SearXNG
#             logger.info("Fetching search results from SearXNG...")
#             search_results = await self.searxng_client.search_searxng(
#                 query=query, 
#                 num_results=num_results, 
#                 engines=engines
#             )
            
#             if not search_results:
#                 return {
#                     "query": query,
#                     "total_found": 0,
#                     "relevant_results": [],
#                     "search_results": [],
#                     "summary": {
#                         "urls_analyzed": 0,
#                         "relevant_count": 0,
#                         "average_relevance": 0.0,
#                         "threshold_used": threshold
#                     },
#                     "error": "No search results found"
#                 }
            
#             logger.info(f"Found {len(search_results)} initial search results")
            
#             # Step 2: Extract URLs for content analysis
#             urls = [result['link'] for result in search_results if 'link' in result]
            
#             if not urls:
#                 return {
#                     "query": query,
#                     "total_found": len(search_results),
#                     "relevant_results": [],
#                     "search_results": search_results,
#                     "summary": {
#                         "urls_analyzed": 0,
#                         "relevant_count": 0,
#                         "average_relevance": 0.0,
#                         "threshold_used": threshold
#                     },
#                     "error": "No valid URLs found in search results"
#                 }
            
#             # Step 3: Analyze content relevance using BrowsingAgent
#             logger.info(f"Analyzing content relevance for {len(urls)} URLs...")
#             scraping_results = await self.browsing_agent.scrape_urls(urls, query)
            
#             # Step 4: Filter relevant results
#             relevant_results = self.browsing_agent.get_relevant_results(scraping_results)
          
#             # Step 5: Combine search metadata with relevance data
#             enhanced_results = self._combine_results(search_results, scraping_results, include_content)
            
#             # Step 6: Calculate summary statistics
#             summary = self._calculate_summary(scraping_results, threshold)
            
#             logger.info(f"Search completed: {len(relevant_results)} relevant results found")
            
#             return {
#                 "query": query,
#                 "total_found": len(search_results),
#                 "relevant_results": [r for r in enhanced_results if r.is_relevant],
#                 "all_results": enhanced_results,
#                 "search_results": search_results,  # Original search results for fallback
#                 "summary": summary
#             }
            
#         except Exception as e:
#             logger.error(f"Error in intelligent web search: {e}")
#             return {
#                 "query": query,
#                 "total_found": 0,
#                 "relevant_results": [],
#                 "search_results": [],
#                 "summary": {
#                     "urls_analyzed": 0,
#                     "relevant_count": 0,
#                     "average_relevance": 0.0,
#                     "threshold_used": threshold
#                 },
#                 "error": str(e)
#             }
    
#     def _combine_results(self, 
#                         search_results: List[Dict], 
#                         scraping_results: List, 
#                         include_content: bool) -> List[WebSearchResult]:
#         """Combine search results with scraping analysis"""
#         combined = []
        
#         # Create URL to scraping result mapping
#         scraping_map = {result.url: result for result in scraping_results}
        
#         for search_result in search_results:
#             url = search_result['link']
#             scraping_result = scraping_map.get(url)
            
#             # Create enhanced result
#             enhanced = WebSearchResult(
#                 title=search_result['title'],
#                 url=url,
#                 snippet=search_result.get('snippet', ''),
#                 engine=search_result.get('engine', ''),
#                 relevance_score=scraping_result.relevance_score if scraping_result else 0.0,
#                 is_relevant=scraping_result.is_relevant if scraping_result else False
#             )
            
#             # Add content if requested and available
#             if include_content and scraping_result and scraping_result.content:
#                 content = scraping_result.content
#                 enhanced.content = content
            
#             combined.append(enhanced)
        
#         # Sort by relevance score (highest first)
#         combined.sort(key=lambda x: x.relevance_score, reverse=True)
        
#         return combined
    
#     def _calculate_summary(self, scraping_results: List, threshold: float) -> Dict[str, Any]:
#         """Calculate summary statistics"""
#         relevant_results = [r for r in scraping_results if r.is_relevant]
        
#         return {
#             "urls_analyzed": len(scraping_results),
#             "relevant_count": len(relevant_results),
#             "average_relevance": sum(r.relevance_score for r in relevant_results) / len(relevant_results) if relevant_results else 0.0,
#             "threshold_used": threshold,
#             "success_rate": len(relevant_results) / len(scraping_results) if scraping_results else 0.0
#         }
    
#     def format_results(self, results: Dict[str, Any], show_all: bool = False) -> str:
#         """Format search results for display"""
#         query = results['query']
#         relevant_results = results['relevant_results']
#         summary = results['summary']
        
#         if 'error' in results:
#             return f"Error searching for '{query}': {results['error']}"
        
#         if not relevant_results and not show_all:
#             # Show search results as fallback
#             search_results = results['search_results'][:5]
#             formatted_results = []
#             for i, result in enumerate(search_results, 1):
#                 formatted_results.append(
#                     f"{i}. **{result['title']}**\n"
#                     f"   URL: {result['link']}\n"
#                     f"   Snippet: {result.get('snippet', 'No snippet available')}\n"
#                     f"   Engine: {result.get('engine', 'Unknown')}\n"
#                 )
            
#             return (
#                 f"**Web Search Results for '{query}'**\n"
#                 f"No pages met relevance threshold ({summary['threshold_used']:.1f})\n"
#                 f"Analyzed {summary['urls_analyzed']} URLs\n\n"
#                 f"**Fallback Search Results:**\n\n" + 
#                 "\n".join(formatted_results)
#             )
        
#         # Format relevant results
#         results_to_show = results['all_results'] if show_all else relevant_results
#         formatted_results = []
        
#         for i, result in enumerate(results_to_show, 1):
#             relevance_emoji = "✅" if result.is_relevant else "❌"
#             formatted_result = (
#                 f"{i}. {relevance_emoji} **{result.title}**\n"
#                 f"   URL: {result.url}\n"
#                 f"   Relevance: {result.relevance_score:.2f}\n"
#                 f"   Engine: {result.engine}\n"
#             )
            
#             if result.content:
#                 formatted_result += f"   Content Preview: {result.content[:200]}...\n"
#             elif result.snippet:
#                 formatted_result += f"   Snippet: {result.snippet}\n"
                
#             formatted_results.append(formatted_result)
        
#         header = (
#             f"**Intelligent Web Search Results for '{query}'**\n\n"
#             f"**Summary:**\n"
#             f"   URLs analyzed: {summary['urls_analyzed']}\n"
#             f"   Relevant results: {summary['relevant_count']}\n"
#             f"   Success rate: {summary['success_rate']:.1%}\n"
#             f"   Average relevance: {summary['average_relevance']:.2f}\n"
#             f"   Threshold used: {summary['threshold_used']:.1f}\n\n"
#             f"**Results:**\n\n"
#         )
        
#         return header + "\n".join(formatted_results)

# async def main():
#     """Example usage of the IntelligentWebSearchTool"""
    
#     # Initialize the tool
#     search_tool = IntelligentWebSearchTool(
#         relevance_threshold=0.5,  # Lower threshold for demo
#         max_concurrent=3
#     )
    
#     # Example queries
#     queries = [
#         "How many R's are there in strawberry?"
#     ]
    
#     for query in queries:
#         print(f"\n{'='*80}")
#         print(f"SEARCHING: {query}")
#         print('='*80)
        
#         # Perform search
#         results = await search_tool.search(
#             query=query,
#             num_results=10,
#             include_content=True,
#             # max_content_length=500
#         )
        
#         # Format and display results
#         formatted_output = search_tool.format_results(results)
#         print(formatted_output)
        
#         # Show token usage
#         print(f"\nToken Usage: {search_tool.token_tracker.get_total_tokens()} total tokens used")

# if __name__ == "__main__":
#     asyncio.run(main())



import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from searxng_service import SearXNGClient
from browsing_agent import BrowsingAgent
from clients.openai_client import OpenAIClient
from clients.gemini_client import GeminiClient
from models.token_tracker import TokenTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    """Enhanced web search result with relevance scoring"""
    title: str
    url: str
    snippet: str
    engine: str
    relevance_score: float = 0.0
    content: Optional[str] = None
    is_relevant: bool = False

class IntelligentWebSearchTool:
    """
    Intelligent web search tool that combines SearXNG search with content relevance filtering
    """
    
    def __init__(self, 
                 searxng_url: str = "https://2wvev6223syg.share.zrok.io/",
                 relevance_threshold: float = 0.6,
                 max_concurrent: int = 5,
                 initial_chars: int = 2000):
        """
        Initialize the intelligent web search tool
        
        Args:
            searxng_url: SearXNG instance URL
            relevance_threshold: Minimum relevance score for content filtering
            max_concurrent: Maximum concurrent requests for content analysis
            initial_chars: Number of characters for initial relevance check
        """
        # Initialize token tracker
        self.token_tracker = TokenTracker()
        
        # Initialize clients
        self.openai_client = OpenAIClient(token_tracker=self.token_tracker)
        self.gemini_client = GeminiClient(token_tracker=self.token_tracker)
        
        # Initialize services
        self.searxng_client = SearXNGClient(searxng_url=searxng_url)
        self.browsing_agent = BrowsingAgent(
            openai_client=self.openai_client,
            gemini_client=self.gemini_client,
            initial_chars=initial_chars,
            relevance_threshold=relevance_threshold,
            max_concurrent=max_concurrent
        )
        
        self.relevance_threshold = relevance_threshold
        
    async def search(self, 
                    query: str, 
                    num_results: int = 10, 
                    engines: str = None,
                    relevance_threshold: Optional[float] = None,
                    include_content: bool = True) -> Dict[str, Any]:
        """
        Perform intelligent web search with content relevance filtering and answer generation
        
        Args:
            query: Search query
            num_results: Number of initial search results to fetch
            engines: Specific search engines to use (comma-separated)
            relevance_threshold: Override default relevance threshold
            include_content: Whether to include extracted content in results
            
        Returns:
            Dictionary containing search results, answer, and metadata
        """
        try:
            logger.info(f"Starting intelligent web search for: {query}")
            
            # Use provided threshold or default
            threshold = relevance_threshold or self.relevance_threshold
            self.browsing_agent.relevance_threshold = threshold
            
            # Step 1: Get initial search results from SearXNG
            logger.info("Fetching search results from SearXNG...")
            search_results = await self.searxng_client.search_searxng(
                query=query, 
                num_results=num_results, 
                engines=engines
            )
            
            if not search_results:
                return {
                    "query": query,
                    "answer": "No search results found to answer the query.",
                    "total_found": 0,
                    "relevant_results": [],
                    "search_results": [],
                    "summary": {
                        "urls_analyzed": 0,
                        "relevant_count": 0,
                        "average_relevance": 0.0,
                        "threshold_used": threshold
                    },
                    "error": "No search results found"
                }
            
            logger.info(f"Found {len(search_results)} initial search results")
            
            # Step 2: Extract URLs for content analysis
            urls = [result['link'] for result in search_results if 'link' in result]
            
            if not urls:
                return {
                    "query": query,
                    "answer": "No valid URLs found in search results to analyze.",
                    "total_found": len(search_results),
                    "relevant_results": [],
                    "search_results": search_results,
                    "summary": {
                        "urls_analyzed": 0,
                        "relevant_count": 0,
                        "average_relevance": 0.0,
                        "threshold_used": threshold
                    },
                    "error": "No valid URLs found in search results"
                }
            
            # Step 3: Analyze content relevance using BrowsingAgent
            logger.info(f"Analyzing content relevance for {len(urls)} URLs...")
            scraping_results = await self.browsing_agent.scrape_urls(urls, query)
            
            # Step 4: Filter relevant results
            relevant_results = self.browsing_agent.get_relevant_results(scraping_results)
          
            # Step 5: Generate answer from relevant content
            answer = await self._generate_answer(query, relevant_results)
            
            # Step 6: Combine search metadata with relevance data
            enhanced_results = self._combine_results(search_results, scraping_results, include_content)
            
            # Step 7: Calculate summary statistics
            summary = self._calculate_summary(scraping_results, threshold)
            
            logger.info(f"Search completed: {len(relevant_results)} relevant results found")
            
            return {
                "query": query,
                "answer": answer,
                "total_found": len(search_results),
                "relevant_results": [r for r in enhanced_results if r.is_relevant],
                "all_results": enhanced_results,
                "search_results": search_results,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent web search: {e}")
            return {
                "query": query,
                "answer": f"Error occurred while searching: {str(e)}",
                "total_found": 0,
                "relevant_results": [],
                "search_results": [],
                "summary": {
                    "urls_analyzed": 0,
                    "relevant_count": 0,
                    "average_relevance": 0.0,
                    "threshold_used": threshold
                },
                "error": str(e)
            }
    
    async def _generate_answer(self, query: str, relevant_results: List) -> str:
        """Generate a comprehensive answer from relevant content"""
        if not relevant_results:
            return "No relevant content found to answer the query."
        
        # Combine content from all relevant results
        combined_content = []
        for result in relevant_results:
            if result.content:
                combined_content.append(f"Source: {result.url}\nContent: {result.content}")
        
        if not combined_content:
            return "No content available from relevant results."
        
        # Create prompt for answer generation
        content_text = "\n\n".join(combined_content)
        
        prompt = f"""Based on the following web content, provide a comprehensive and accurate answer to the query: "{query}"

        Web Content:
        {content_text}

        Instructions:
        - Use only the information provided in the web content
        - Be factual and concise
        - If the content doesn't fully answer the query, state what information is available
        - Synthesize information from multiple sources when relevant
        - Do not include URLs or source references in the answer

        Answer:"""
        
        try:
            # Use OpenAI for answer generation
            response = await self.openai_client.chat_completion(prompt, model="gpt-4.1-nano")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to Gemini if OpenAI fails
            try:
                response = await self.gemini_client.generate_content(
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.3
                )
                return response.strip()
            except Exception as e2:
                logger.error(f"Error with Gemini fallback: {e2}")
                return "Unable to generate answer due to API errors."
    
    def _combine_results(self, 
                        search_results: List[Dict], 
                        scraping_results: List, 
                        include_content: bool) -> List[WebSearchResult]:
        """Combine search results with scraping analysis"""
        combined = []
        
        # Create URL to scraping result mapping
        scraping_map = {result.url: result for result in scraping_results}
        
        for search_result in search_results:
            url = search_result['link']
            scraping_result = scraping_map.get(url)
            
            # Create enhanced result
            enhanced = WebSearchResult(
                title=search_result['title'],
                url=url,
                snippet=search_result.get('snippet', ''),
                engine=search_result.get('engine', ''),
                relevance_score=scraping_result.relevance_score if scraping_result else 0.0,
                is_relevant=scraping_result.is_relevant if scraping_result else False
            )
            
            # Add content if requested and available
            if include_content and scraping_result and scraping_result.content:
                content = scraping_result.content
                enhanced.content = content
            
            combined.append(enhanced)
        
        # Sort by relevance score (highest first)
        combined.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return combined
    
    def _calculate_summary(self, scraping_results: List, threshold: float) -> Dict[str, Any]:
        """Calculate summary statistics"""
        relevant_results = [r for r in scraping_results if r.is_relevant]
        
        return {
            "urls_analyzed": len(scraping_results),
            "relevant_count": len(relevant_results),
            "average_relevance": sum(r.relevance_score for r in relevant_results) / len(relevant_results) if relevant_results else 0.0,
            "threshold_used": threshold,
            "success_rate": len(relevant_results) / len(scraping_results) if scraping_results else 0.0
        }
    
    def format_results(self, results: Dict[str, Any], show_all: bool = False) -> str:
        """Format search results for display"""
        query = results['query']
        answer = results['answer']
        relevant_results = results['relevant_results']
        summary = results['summary']
        
        if 'error' in results:
            return f"Error searching for '{query}': {results['error']}"
        
        # Start with the generated answer
        output = f"**Question:** {query}\n\n**Answer:** {answer}\n\n"
        
        # Add summary
        output += (
            f"**Search Summary:**\n"
            f"URLs analyzed: {summary['urls_analyzed']}\n"
            f"Relevant results: {summary['relevant_count']}\n"
            f"Success rate: {summary['success_rate']:.1%}\n"
            f"Average relevance: {summary['average_relevance']:.2f}\n\n"
        )
        
        if relevant_results:
            output += "**Sources:**\n"
            for i, result in enumerate(relevant_results, 1):
                output += (
                    f"{i}. {result.title}\n"
                    f"   URL: {result.url}\n"
                    f"   Relevance: {result.relevance_score:.2f}\n\n"
                )
        
        return output

async def main():
    """Example usage of the IntelligentWebSearchTool"""
    
    # Initialize the tool
    search_tool = IntelligentWebSearchTool(
        relevance_threshold=0.5,
        max_concurrent=3
    )
    
    # Example queries
    queries = [
        "How many R's are there in strawberry?"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"SEARCHING: {query}")
        print('='*80)
        
        # Perform search
        results = await search_tool.search(
            query=query,
            num_results=5,
            include_content=True
        )
        
        # Format and display results
        formatted_output = search_tool.format_results(results)
        print(formatted_output)
        
        # Show token usage
        print(f"\nToken Usage: {search_tool.token_tracker.get_total_tokens()} total tokens used")

if __name__ == "__main__":
    asyncio.run(main())