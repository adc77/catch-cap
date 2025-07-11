from openai import AsyncOpenAI
from typing import List
import logging

from config.settings import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from models.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for OpenAI API interactions."""
    
    def __init__(self, token_tracker: TokenTracker):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.token_tracker = token_tracker
    
    async def generate_embedding(self, text: str, model: str = OPENAI_EMBEDDING_MODEL) -> List[float]:
        """Generate embedding for text."""
        try:
            # Track token usage
            self.token_tracker.track_openai_embedding(text, model)
            
            response = await self.client.embeddings.create(
                model=model,
                dimensions=1024,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def chat_completion(self, prompt: str, model: str, temperature: float = 0.5, max_tokens: int = None) -> str:
        """Generate chat completion."""
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            
            if max_tokens:
                kwargs["max_completion_tokens"] = max_tokens
            
            response = await self.client.chat.completions.create(**kwargs)
            
            # Check if response is valid
            if response and response.choices and len(response.choices) > 0:
                output_text = response.choices[0].message.content
                if output_text is not None:
                    output_text = output_text.strip()
                    self.token_tracker.track_openai_chat(prompt, output_text, model)
                    return output_text
            
            logger.error("Received an invalid response from OpenAI API.")
            return ""
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    async def openai_deep_research(self, prompt: str, model: str = "o4-mini-deep-research", 
                                 tools: List[dict] = None, output_file: str = None) -> str:
        """Generate deep research response using OpenAI's o3-deep-research model."""
        try:
            # Default tools for deep research
            if tools is None:
                tools = [
                    {"type": "web_search_preview"},
                    {"type": "code_interpreter", "container": {"type": "auto"}},
                    # {
                    #     "type": "mcp",
                    #     "server_label": "mcp_search_service",
                    #     "server_url": "http://localhost:8000/sse",
                    #     "require_approval": "never",
                    # },
                ]
            
            # Create the response using the responses API
            response = await self.client.responses.create(
                model=model,
                background = True,
                reasoning={
                    "summary": "auto",
                },
                tools=tools,
                input=prompt,
            )
            
            # Get the output text
            output_text = response.output_text if hasattr(response, 'output_text') else str(response)
            
            # Save to file if specified
            if output_file:
                try:
                    with open(output_file, "w", encoding="utf-8") as file:
                        file.write(output_text)
                    logger.info(f"Response saved to {output_file}")
                except Exception as e:
                    logger.warning(f"Failed to save response to file: {e}")
            # Track token usage (approximate for deep research)
            self.token_tracker.track_openai_chat(prompt, output_text, model)
            
            return output_text
            
        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            raise

    async def generate_n_samples(self, prompt: str, n_samples: int, model: str = "gpt-4.1-nano", temperature: float = 1.0, top_p: float = 0.9) -> List[str]:
        """Generate n samples of text."""
        try:
            responses = []
            for _ in range(n_samples):
                response = await self.client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    top_p=top_p,
                )
                responses.append(response.output_text)
            return responses
        except Exception as e:
            logger.error(f"Error in generate_n_samples: {e}")
            raise

    async def generate_n_embeddings(self, prompt: str, n_samples: int, model: str = "text-embedding-3-small") -> List[List[float]]:
        """Generate n embeddings for text."""
        try:
            embeddings = []
            for _ in range(n_samples):
                embedding = await self.client.embeddings.create(
                    model=model,
                    dimensions=1024,
                    input=prompt,
                    encoding_format="float"
                )
                embeddings.append(embedding.data[0].embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error in generate_n_embeddings: {e}")
            raise

    async def compare_responses(self, responses: List[str], web_answer: str, query: str) -> str:
        """Compare responses with web answer."""
        prompt = f"""
        You are a helpful assistant that compares responses with web answer for the given query and give a reasoning why the responses are hallucinated or confabulated.
        The query is:
        {query}
        The responses are:
        {responses}
        The web answer is:
        {web_answer}
        Give a reasoning why the responses are hallucinated or confabulated.
        Do not mention responses and web search in your response just give a reasoning why the responses are hallucinated or confabulated.
        Return only the reasoning without any additional text.
        Compare all `responses` with the `web_answer` and return a single reasoning why the responses are hallucinated or confabulated.
        """
        try:
            response = await self.client.responses.create(
                model="gpt-4.1-nano",
                input=prompt,
            )   
            return response.output_text
        except Exception as e:
            logger.error(f"Error in compare_responses: {e}")
            raise





