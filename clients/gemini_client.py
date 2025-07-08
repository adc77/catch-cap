from google import genai
from google.genai import types
import logging

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from models.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Gemini API interactions."""
    
    def __init__(self, token_tracker: TokenTracker):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.token_tracker = token_tracker
    
    async def generate_content(self, prompt: str, model: str = GEMINI_MODEL, 
                        temperature: float = 0.7, max_output_tokens: int = None) -> str:
        """Generate content using Gemini."""
        try:
            config_kwargs = {"temperature": temperature}
            if max_output_tokens:
                config_kwargs["max_output_tokens"] = max_output_tokens
            
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(**config_kwargs)
            )
             # Check if response and response.text are valid
            if response is None:
                logger.error("Gemini API returned None response")
                return ""
            
            if not hasattr(response, 'text') or response.text is None:
                logger.error("Gemini response has no text or text is None")
                return ""
            
            output_text = response.text.strip()
            
            # Only track if we have valid output
            if output_text:
                self.token_tracker.track_gemini_generation(prompt, output_text, model)
            
            return output_text
            
        except Exception as e:
            logger.error(f"Error in Gemini content generation: {e}")
            raise