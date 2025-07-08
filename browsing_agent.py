import asyncio
import aiohttp
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import logging
import json
import os
from dotenv import load_dotenv
from clients.openai_client import OpenAIClient
from clients.gemini_client import GeminiClient
from models.token_tracker import TokenTracker

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Result of URL scraping operation"""
    url: str
    is_relevant: bool
    relevance_score: float
    content: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None
    processing_time: float = 0.0

class BrowsingAgent:
    """
    Intelligent browsing agent that scrapes URLs based on content relevance
    """
    
    def __init__(self, openai_client: OpenAIClient, gemini_client: GeminiClient, initial_chars: int = 2000, 
                 relevance_threshold: float = 0.6, max_concurrent: int = 5):
        """
        Initialize the browsing agent
        
        Args:
            openai_api_key: OpenAI API key
            initial_chars: Number of characters to fetch for initial relevance check
            relevance_threshold: Minimum relevance score to proceed with full scraping
            max_concurrent: Maximum concurrent requests
        """
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        self.initial_chars = initial_chars
        self.relevance_threshold = relevance_threshold
        self.max_concurrent = max_concurrent
        
        # Request headers to appear more like a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Timeout settings
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)

    async def _fetch_content(self, session: aiohttp.ClientSession, url: str, 
                           limit_chars: Optional[int] = None) -> Tuple[str, str]:
        """
        Fetch content from URL with optional character limit
        
        Returns:
            Tuple of (text_content, title)
        """
        try:
            async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                response.raise_for_status()
                
                # Check content type to avoid processing binary files
                content_type = response.headers.get('content-type', '').lower()
                
                # Skip binary files like PDFs, images, videos, etc.
                binary_types = [
                    'application/pdf', 'application/octet-stream',
                    'image/', 'video/', 'audio/',
                    'application/zip', 'application/x-rar',
                    'application/msword', 'application/vnd.ms-excel',
                    'application/vnd.openxmlformats'
                ]
                
                if any(binary_type in content_type for binary_type in binary_types):
                    logger.info(f"Skipping binary file: {url} (type: {content_type})")
                    return "", f"Binary file: {url.split('/')[-1]}"
                
                # Read content with potential limit
                try:
                    if limit_chars:
                        content = b""
                        async for chunk in response.content.iter_chunked(1024):
                            content += chunk
                            if len(content) >= limit_chars * 3:  # Accounting for HTML overhead
                                break
                        html_content = content.decode('utf-8', errors='ignore')
                    else:
                        html_content = await response.text(errors='ignore')
                except UnicodeDecodeError:
                    # If UTF-8 decoding fails, try with error handling
                    content = await response.read()
                    html_content = content.decode('utf-8', errors='ignore')
                
                # Parse HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                
                # Extract title
                title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
                
                # Extract text content
                text_content = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text_content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Apply character limit if specified
                if limit_chars and len(text_content) > limit_chars:
                    text_content = text_content[:limit_chars] + "..."
                
                return text_content, title
                
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {str(e)}")
            return "", f"Error: {url.split('/')[-1]}"
    
    async def _check_relevance_by_openai(self, content: str, query: str, url: str) -> float:
        """
        Check content relevance using OpenAI GPT-4.1-nano
        
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:

            prompt = f"""
            You are an expert in research and information retrieval.

            Evaluate the relevance of the following web content to the given query.

            QUERY: {query}

            WEB CONTENT:
            {content}

            Instructions:
            - Consider both thematic relevance (is the content addressing the theme or issue in the query?) and semantic relevance (is the content meaningfully answering or contributing to the query?).
            - Think like an expert in research and information retrieval.
            - Focus on contextual relevance, and usefulness.
            - Return only the relevance score.

            Format:
            {{"relevance_score": 0.0}}  # score should be a float between 0.0 and 1.0

            """

            response = await self.openai_client.chat_completion(prompt, model="gpt-4.1-nano", temperature=0.2)
            
            result_text = response.strip()
            # result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                score = float(result.get('relevance_score', 0.0))
                logger.info(f"Relevance for {url}: {score:.2f}")
                return max(0.0, min(1.0, score))  # Ensure score is between 0.0 and 1.0
            except json.JSONDecodeError:
                # Fallback: try to extract number from response
                import re
                numbers = re.findall(r'[0-9]*\.?[0-9]+', result_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))
                else:
                    logger.warning(f"Could not parse relevance score from: {result_text}")
                    return 0.0
                    
        except Exception as e:
            logger.error(f"Error checking relevance for {url}: {str(e)}")
            return 0.0

    async def _check_relevance_by_gemini(self, content: str, query: str, url: str) -> float:
        """
        Check content relevance using Gemini
        """
        try:
            prompt = f"""
            You are an expert in research and information retrieval.

            Evaluate the relevance of the following web content to the given query.

            QUERY: {query}

            WEB CONTENT:
            {content}

            Instructions:
            - Consider both thematic relevance (is the content addressing the theme or issue in the query?) and semantic relevance (is the content meaningfully answering or contributing to the query?).
            - Think like an expert in research and information retrieval.
            - Focus on contextual relevance, and usefulness.
            - Return only the relevance score.

            Format:
            {{"relevance_score": 0.0}}  # score should be a float between 0.0 and 1.0

            """

            response = await self.gemini_client.generate_content(prompt, model="gemini-2.0-flash", temperature=0.2)

            result_text = response.strip()

            # Parse JSON response
            try:
                result = json.loads(result_text)
                score = float(result.get('relevance_score', 0.0))
                logger.info(f"Relevance for {url}: {score:.2f}")
                return max(0.0, min(1.0, score))  # Ensure score is between 0.0 and 1.0
            except json.JSONDecodeError:
                # Fallback: try to extract number from response     
                import re
                numbers = re.findall(r'[0-9]*\.?[0-9]+', result_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))
                else:
                    logger.warning(f"Could not parse relevance score from: {result_text}")
                    return 0.0

        except Exception as e:
            logger.error(f"Error checking relevance for {url}: {str(e)}")
            return 0.0

    async def _process_url(self, session: aiohttp.ClientSession, url: str, query: str) -> ScrapingResult:
        """
        Process a single URL through the relevance-based scraping pipeline
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing URL: {url}")
            
            # Step 1: Fetch full content directly
            full_content, title = await self._fetch_content(session, url)

            # # store full_content in a .txt file in the current directory in a folder called scraped content
            # os.makedirs("scraped_content", exist_ok=True)
            # with open(f"scraped_content/{url.split('/')[-1]}.txt", "w", encoding='utf-8') as f:
            #     f.write(full_content)
            
            if not full_content.strip():
                return ScrapingResult(
                    url=url, 
                    is_relevant=False, 
                    relevance_score=0.0,
                    error="No content found",
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Check relevance with LLM using full content
            # relevance_score = await self._check_relevance_by_openai(full_content, query, url)
            relevance_score = await self._check_relevance_by_gemini(full_content, query, url)

            # Step 3: Decide whether to return content based on relevance
            if relevance_score >= self.relevance_threshold:
                logger.info(f"URL {url} is relevant (score: {relevance_score:.2f}). Returning full content...")
                
                # # Save full content to file
                # os.makedirs("full_content", exist_ok=True)
                # with open(f"full_content/{url.split('/')[-1]}.txt", "w", encoding='utf-8') as f:
                #     f.write(full_content)
                
                return ScrapingResult(
                    url=url,
                    is_relevant=True,
                    relevance_score=relevance_score,
                    content=full_content,
                    title=title,
                    processing_time=time.time() - start_time
                )
            else:
                logger.info(f"URL {url} not relevant enough (score: {relevance_score:.2f}). Skipping...")
                
                return ScrapingResult(
                    url=url,
                    is_relevant=False,
                    relevance_score=relevance_score,
                    content=None,  # Don't return content for irrelevant pages
                    title=title,
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return ScrapingResult(
                url=url,
                is_relevant=False,
                relevance_score=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def scrape_urls(self, urls: List[str], query: str) -> List[ScrapingResult]:
        """
        Main method to scrape multiple URLs based on relevance to query
        
        Args:
            urls: List of URLs to process
            query: Search query for relevance checking
            
        Returns:
            List of ScrapingResult objects
        """
        logger.info(f"Starting to process {len(urls)} URLs for query: '{query}'")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(session, url):
            async with semaphore:
                return await self._process_url(session, url, query)
        
        # Process URLs concurrently
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=20)) as session:
            tasks = [process_with_semaphore(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ScrapingResult(
                    url=urls[i],
                    is_relevant=False,
                    relevance_score=0.0,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        # Log summary
        relevant_count = sum(1 for r in final_results if r.is_relevant)
        avg_processing_time = sum(r.processing_time for r in final_results) / len(final_results)
        
        logger.info(f"Processing complete. {relevant_count}/{len(urls)} URLs were relevant. "
                   f"Average processing time: {avg_processing_time:.2f}s")
        
        return final_results
    
    def get_relevant_results(self, results: List[ScrapingResult]) -> List[ScrapingResult]:
        """Filter and return only relevant results"""
        return [r for r in results if r.is_relevant]
    
    def print_summary(self, results: List[ScrapingResult]):
        """Print a summary of scraping results"""
        print(f"\n{'='*60}")
        print(f"SCRAPING SUMMARY")
        print(f"{'='*60}")
        
        relevant_results = self.get_relevant_results(results)
        
        print(f"Total URLs processed: {len(results)}")
        print(f"Relevant URLs found: {len(relevant_results)}")
        print(f"Success rate: {len(relevant_results)/len(results)*100:.1f}%")
        
        if relevant_results:
            print(f"\nRELEVANT CONTENT FOUND:")
            print(f"{'-'*40}")
            
            for i, result in enumerate(relevant_results, 1):
                print(f"{i}. {result.title}")
                print(f"   URL: {result.url}")
                print(f"   Relevance Score: {result.relevance_score:.2f}")
                print(f"   Content Length: {len(result.content) if result.content else 0} characters")
                print(f"   Processing Time: {result.processing_time:.2f}s")
                print()


# Example usage and testing function
async def main():
    """
    Example usage of the BrowsingAgent
    """
    
    # Example URLs and query
    urls = [
        "https://www.legalserviceindia.com/legal/article-2017-joint-responsibility-in-crime.html",
        "https://lawjurist.com/index.php/2024/12/28/a-study-of-principle-of-joint-liability-with-relevant-case-laws/",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3429038",
        "https://www.researchgate.net/publication/342130148_Juxtaposition_between_the_Theory_of_Joint_Criminal_Enterprise_JCE_and_the_Theory_of_CommandSuperior_Responsibility_under_International_Law",
        "https://www.jstor.org/stable/25659914",
        "https://www.diva-portal.org/smash/get/diva2:661882/fulltext01.pdf",
        "https://scholarship.law.cornell.edu/cgi/viewcontent.cgi?article=1029&context=facpub",
        "https://cld.irmct.org/notions/show/488/joint-criminal-enterprise",
        "https://www.aijcrnet.com/journals/Vol_7_No_3_September_2017/9.pdf",
        "https://legal.economictimes.indiatimes.com/news/international/supreme-court-to-hear-chevron-appeal-on-louisiana-coastal-damage-lawsuits/121897188"
    ]

    
    query = "How has the theory of joint criminal enterprise been utilized in Indian criminal law?"
    
    # Initialize agent
    agent = BrowsingAgent(
        openai_client=OpenAIClient(token_tracker=TokenTracker()),
        gemini_client=GeminiClient(token_tracker=TokenTracker()),
        initial_chars=None,
        relevance_threshold=0.5,
        max_concurrent=3
    )
    
    try:
        # Scrape URLs
        results = await agent.scrape_urls(urls, query)
        
        # Print summary
        agent.print_summary(results)
        
        # Get only relevant results
        relevant_results = agent.get_relevant_results(results)
        
        # Print detailed results for relevant URLs
        for result in relevant_results:
            print(f"\n{'='*80}")
            print(f"CONTENT FROM: {result.url}")
            print(f"TITLE: {result.title}")
            print(f"RELEVANCE SCORE: {result.relevance_score:.2f}")
            print(f"{'='*80}")
            print(result.content[:1000] + "..." if len(result.content) > 1000 else result.content)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())