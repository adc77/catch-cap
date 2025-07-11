from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import logging
import numpy as np
from sem_ent import detect_confabulation

# Import OpenAI client dependencies
from clients.openai_client import OpenAIClient
from models.token_tracker import TokenTracker
from config.settings import OPENAI_CHAT_MODEL, OPENAI_MINI_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Confabulation Detection API",
    description="API for detecting confabulation in AI responses using semantic entropy and web search validation",
    version="1.0.0"
)

# Request models
class ConfabulationRequest(BaseModel):
    query: str
    entropy_threshold: float = 0.3

class GenerateRequest(BaseModel):
    query: str
    model: Optional[str] = OPENAI_CHAT_MODEL
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class DeepResearchRequest(BaseModel):
    query: str
    model: Optional[str] = "o4-mini-deep-research"
    output_file: Optional[str] = None

# Response models
class ConfabulationSummary(BaseModel):
    query: str
    entropy_score: float
    entropy_threshold: float
    is_confident: bool
    confabulation_detected: bool
    analysis_result: str
    comparison_summary: str
    model_response_sample: str
    web_answer_sample: str
    total_tokens_used: int
    web_search_summary: Dict[str, Any]

class ConfabulationResponse(BaseModel):
    success: bool
    summary: Optional[ConfabulationSummary] = None
    error: Optional[str] = None

class GenerateResponse(BaseModel):
    success: bool
    query: str
    response: str
    model: str
    tokens_used: Dict[str, int]
    cost: float
    error: Optional[str] = None

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def extract_total_tokens(token_data):
    """Extract total tokens from token tracker data"""
    if isinstance(token_data, dict):
        return int(token_data.get('total_tokens', 0))
    elif isinstance(token_data, (int, np.integer)):
        return int(token_data)
    else:
        return 0

def sanitize_results(results):
    """Convert all numpy types in results to native Python types"""
    sanitized = {}
    for key, value in results.items():
        if key == 'similarity_matrix':
            # Skip similarity matrix for API responses as it's large
            continue
        sanitized[key] = convert_numpy_types(value)
    return sanitized

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Confabulation Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Detect confabulation for a given query",
            "/detect/detailed": "POST - Detect confabulation with detailed results",
            "/generate": "POST - Generate response using OpenAI directly",
            "/generate/deep-research": "POST - Generate response using OpenAI deep research",
            "/health": "GET - Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Confabulation Detection API is running"}

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    Generate a response directly using OpenAI client
    
    Args:
        request: GenerateRequest containing query, model, temperature, and max_tokens
        
    Returns:
        GenerateResponse with the generated response and token usage information
    """
    try:
        logger.info(f"Received generate request for query: {request.query}")
        
        # Validate input
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.temperature is not None and not 0.0 <= request.temperature <= 2.0:
            raise HTTPException(status_code=400, detail="Temperature must be between 0.0 and 2.0")
        
        if request.max_tokens is not None and request.max_tokens <= 0:
            raise HTTPException(status_code=400, detail="Max tokens must be positive")
        
        # Initialize token tracker and OpenAI client
        token_tracker = TokenTracker()
        openai_client = OpenAIClient(token_tracker)
        
        # Generate response
        response = await openai_client.chat_completion(
            prompt=request.query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Get token usage and cost information
        token_breakdown = token_tracker.get_detailed_breakdown()
        
        logger.info(f"Generated response for query: {request.query}")
        logger.info(f"Tokens used: {token_breakdown['totals']['total_tokens']}, Cost: ${token_breakdown['total_cost']:.4f}")
        
        return GenerateResponse(
            success=True,
            query=request.query,
            response=response,
            model=request.model,
            tokens_used=token_breakdown['totals'],
            cost=token_breakdown['total_cost']
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/detect", response_model=ConfabulationResponse)
async def detect_confabulation_endpoint(request: ConfabulationRequest):
    """
    Detect confabulation for a given query
    
    Args:
        request: ConfabulationRequest containing query and optional entropy_threshold
        
    Returns:
        ConfabulationResponse with detection results and summary
    """
    try:
        logger.info(f"Received confabulation detection request for query: {request.query}")
        
        # Validate input
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not 0.0 <= request.entropy_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Entropy threshold must be between 0.0 and 1.0")
        
        # Perform confabulation detection
        raw_results = await detect_confabulation(
            prompt=request.query,
            entropy_threshold=request.entropy_threshold
        )
        
        # Sanitize results to remove numpy types
        results = sanitize_results(raw_results)
        
        # Extract total tokens properly
        total_tokens = extract_total_tokens(results["total_tokens_used"])
        
        # Create summary with proper type conversion
        summary = ConfabulationSummary(
            query=str(results["query"]),
            entropy_score=float(results["entropy_score"]),
            entropy_threshold=float(results["entropy_threshold"]),
            is_confident=bool(results["is_confident"]),
            confabulation_detected=bool(results["confabulation_detected"]),
            analysis_result=str(results["analysis_result"]),
            comparison_summary=str(results["comparison_result"])[:200] + "..." if len(str(results["comparison_result"])) > 200 else str(results["comparison_result"]),
            model_response_sample=str(results["model_responses"][0])[:150] + "..." if len(str(results["model_responses"][0])) > 150 else str(results["model_responses"][0]),
            web_answer_sample=str(results["web_answer"])[:150] + "..." if len(str(results["web_answer"])) > 150 else str(results["web_answer"]),
            total_tokens_used=total_tokens,
            web_search_summary=results["web_search_summary"],
            reasoning=str(results["reasoning"])
        )
        
        logger.info(f"Confabulation detection completed for query: {request.query}")
        logger.info(f"Result: {results['analysis_result']}, Confabulation: {results['confabulation_detected']}")
        
        return ConfabulationResponse(
            success=True,
            summary=summary
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing confabulation detection: {str(e)}")
        # Return error response without summary
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/detect/detailed")
async def detect_confabulation_detailed(request: ConfabulationRequest):
    """
    Detect confabulation with detailed results
    
    Args:
        request: ConfabulationRequest containing query and optional entropy_threshold
        
    Returns:
        Complete detection results including all model responses and detailed analysis
    """
    try:
        logger.info(f"Received detailed confabulation detection request for query: {request.query}")
        
        # Validate input
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not 0.0 <= request.entropy_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Entropy threshold must be between 0.0 and 1.0")
        
        # Perform confabulation detection
        raw_results = await detect_confabulation(
            prompt=request.query,
            entropy_threshold=request.entropy_threshold
        )
        
        # Sanitize results to remove numpy types
        results = sanitize_results(raw_results)
        
        # Extract total tokens properly
        total_tokens = extract_total_tokens(results["total_tokens_used"])
        
        # Return complete results with proper type conversion
        response_data = {
            "success": True,
            "query": str(results["query"]),
            "entropy_score": float(results["entropy_score"]),
            "entropy_threshold": float(results["entropy_threshold"]),
            "is_confident": bool(results["is_confident"]),
            "confabulation_detected": bool(results["confabulation_detected"]),
            "analysis_result": str(results["analysis_result"]),
            "model_responses": [str(resp) for resp in results["model_responses"]],
            "web_answer": str(results["web_answer"]),
            "comparison_result": str(results["comparison_result"]),
            "web_search_summary": results["web_search_summary"],
            "total_tokens_used": total_tokens,
            "reasoning": str(results["reasoning"])
        }
        
        logger.info(f"Detailed confabulation detection completed for query: {request.query}")
        
        return response_data
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing detailed confabulation detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")