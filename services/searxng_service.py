import asyncio
import requests
from typing import List, Dict, Any
import logging
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from clients.gemini_client import GeminiClient
from clients.openai_client import OpenAIClient
from models.token_tracker import TokenTracker

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

class SearXNGClient:
    """Client for SearXNG web search."""
    
    def __init__(self, searxng_url: str = "https://3zvqq5x2pyyz.share.zrok.io/"):
        self.searxng_url = searxng_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ReasoningAssistant/1.0'
        })
        self.openai_client = OpenAIClient(token_tracker=TokenTracker())
        self.gemini_client = GeminiClient(token_tracker=TokenTracker())

    async def search_searxng(self, query: str, num_results: int = 20, engines: str = None) -> List[Dict[str, Any]]:
        """
        Search SearXNG and return the top results
        
        Parameters:
        query (str): The search query
        num_results (int): Number of results to return
        engines (str): Specific engine(s) to use for the search
        
        Returns:
        list: Top search results
        """
        # if engines is None:
        #     engines = 'azbpartners, bar and bench, clpr, cyrilshroff, ikigailaw, legal wires, live law, mondaq, nishithdesai, nlsir, nlsrepo, pudr'
        
        engines = 'google, bing, duckduckgo'
        params = {
            'q': query,
            'format': 'json',
            # 'categories': 'law', 
            'engines': engines,
            'language': 'en',
            'pageno': 1,
            'results_count': num_results
        }
        
        try:
            response = self.session.get(self.searxng_url, params=params, timeout=30)
            response.raise_for_status()  
            
            results_json = response.json()
            results = results_json.get('results', [])[:num_results]
            
            valid_items = []
            for item in results:
                if 'url' in item and 'title' in item:
                    valid_item = {
                        'link': item['url'],
                        'title': item['title'],
                        'snippet': item.get('content', ''),
                        'engine': item.get('engine', '')
                    }
                    valid_items.append(valid_item)
            
            # Simple ranking by relevance score if available
            # ranked_results = await self.rerank_results_by_openai(valid_items, query)
            ranked_results = await self.rerank_results_by_gemini(valid_items, query)
            return ranked_results[:10]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching SearXNG: {e}")
            return []

    async def rerank_results_by_openai(self, results: List[Dict], query: str) -> List[Dict]:
        """Rerank results using OpenAI"""
        try:
            prompt = f"""You are an expert in research and information retrieval.

            Evaluate and rank the following search results based on their relevance to the given query.

            QUERY: {query}

            SEARCH RESULTS:
            {json.dumps(results, indent=2)}

            Instructions:
            - Rank the results from most relevant (1) to least relevant
            - Consider thematic alignment, semantic similarity, and context
            - Return ONLY a valid JSON array with the results in ranked order
            - Do not include any explanations or additional text
            - Preserve all original fields (link, title, snippet, engine) for each result

            IMPORTANT: Your response must be a valid JSON array that starts with [ and ends with ]. Do not include any other text.
            """

            response = await self.openai_client.chat_completion(prompt, model="gpt-4.1-nano", temperature=0.1)
            
            result_text = response.strip()
            
            # Clean the response - remove any markdown formatting or extra text
            result_text = self.clean_json_response(result_text)
            
            # Parse the JSON response
            ranked_results = self.parse_json(result_text)
            
            if ranked_results and isinstance(ranked_results, list) and len(ranked_results) > 0:
                # Validate that each item has required fields
                valid_results = []
                for item in ranked_results:
                    if isinstance(item, dict) and 'link' in item and 'title' in item:
                        valid_results.append(item)
                
                if valid_results:
                    logger.info(f"OpenAI reranking successful: {len(valid_results)} results")
                    return valid_results
                else:
                    logger.warning("OpenAI returned list but items missing required fields")
                    return results
            else:
                logger.warning("OpenAI did not return a valid list, falling back to original order")
                return results
                    
        except Exception as e:
            logger.error(f"Error reranking results with OpenAI: {e}")
            return results

    async def rerank_results_by_gemini(self, results: List[Dict], query: str) -> List[Dict]:
        """Rerank results using Gemini"""
        try:
            prompt = f"""You are an expert in research and information retrieval.

            Evaluate and rank the following search results based on their relevance to the given query.

            QUERY: {query}

            SEARCH RESULTS:
            {json.dumps(results, indent=2)}

            Instructions:
            - Rank the results from most relevant (1) to least relevant
            - Consider thematic alignment, semantic similarity, and context
            - Return ONLY a valid JSON array with the results in ranked order
            - Do not include any explanations or additional text
            - Preserve all original fields (link, title, snippet, engine) for each result

            IMPORTANT: Your response must be a valid JSON array that starts with [ and ends with ]. Do not include any other text.
            """

            response = await self.gemini_client.generate_content(prompt, model="gemini-2.0-flash", temperature=0.1)

            result_text = response.strip()
            
            # Clean the response - remove any markdown formatting or extra text
            result_text = self.clean_json_response(result_text)

            # Parse the JSON response
            ranked_results = self.parse_json(result_text)
            
            if ranked_results and isinstance(ranked_results, list) and len(ranked_results) > 0:
                # Validate that each item has required fields
                valid_results = []
                for item in ranked_results:
                    if isinstance(item, dict) and 'link' in item and 'title' in item:
                        valid_results.append(item)
                
                if valid_results:
                    logger.info(f"Gemini reranking successful: {len(valid_results)} results")
                    return valid_results
                else:
                    logger.warning("Gemini returned list but items missing required fields")
                    return results
            else:
                logger.warning("Gemini did not return a valid list, falling back to original order")
                return results

        except Exception as e:
            logger.error(f"Error reranking results with Gemini: {e}")
            return results

    def clean_json_response(self, response_text: str) -> str:
        """Clean the response text to extract valid JSON"""
        # Remove markdown code blocks if present
        if "```json" in response_text:
            # Extract content between ```json and ```
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        elif "```" in response_text:
            # Extract content between ``` and ```
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        
        # Find the JSON array bounds
        start_bracket = response_text.find('[')
        end_bracket = response_text.rfind(']')
        
        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            response_text = response_text[start_bracket:end_bracket + 1]
        
        return response_text.strip()

    def parse_json(self, json_string: str):
        """Enhanced JSON parsing with better error handling"""
        if not json_string or not isinstance(json_string, str):
            logger.error("Invalid input to parse_json: empty or non-string")
            return None
            
        try:
            # Attempt to parse the JSON string
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}")
            logger.debug(f"Problematic JSON (first 500 chars): {json_string[:500]}")
            
            # Try to fix common JSON issues
            try:
                # Remove trailing commas
                fixed_json = json_string.replace(',]', ']').replace(',}', '}')
                data = json.loads(fixed_json)
                logger.info("Fixed JSON by removing trailing commas")
                return data
            except json.JSONDecodeError:
                logger.error("Could not fix JSON automatically")
                return None

async def main():
    client = SearXNGClient()
    results = await client.search_searxng("How many R's are there in strawberry?")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
