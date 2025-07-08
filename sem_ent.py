import asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from config.settings import N_SAMPLES
from clients.openai_client import OpenAIClient
from models.token_tracker import TokenTracker
from web_search_tool import IntelligentWebSearchTool
import logging

openai_client = OpenAIClient(token_tracker=TokenTracker())
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_semantic_entropy(embeddings):
    """Compute semantic entropy from embeddings"""
    sim_matrix = cosine_similarity(embeddings)
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    avg_similarity = np.mean(upper_triangle)
    entropy_score = 1 - avg_similarity
    return entropy_score, sim_matrix

async def compare_responses(model_responses, web_answer, query):
    """Compare model responses with web search answer to detect confabulation"""
    
    # Create a prompt to compare responses
    model_responses_text = "\n".join([f"Response {i+1}: {resp}" for i, resp in enumerate(model_responses)])
    
    comparison_prompt = f"""Compare the following model responses with the web search answer for the query: "{query}"

    Model Responses:
    {model_responses_text}

    Web Search Answer:
    {web_answer}

    Instructions:
    - Determine if the model responses are consistent with the web search answer
    - Consider the core facts and main points
    - Ignore minor differences in phrasing or style
    - Return only "CONSISTENT" or "INCONSISTENT" 
    - No explanation is needed
    """

    try:
        comparison_result = await openai_client.chat_completion(comparison_prompt, model="gpt-4.1-nano")
        is_consistent = "CONSISTENT" in comparison_result.upper() and "INCONSISTENT" not in comparison_result.upper()
        print(comparison_result)
        print(is_consistent)
        print("########################################################COMPARISON RESULT########################################################")
        return is_consistent, comparison_result
    except Exception as e:
        logger.error(f"Error in response comparison: {e}")
        return False, f"Error during comparison: {str(e)}"

async def detect_confabulation(prompt, entropy_threshold=0.3):
    """
    Main function to detect confabulation using semantic entropy and web search validation
    
    Args:
        prompt: The query to analyze
        entropy_threshold: Threshold below which model is considered confident
    
    Returns:
        Dictionary containing analysis results
    """
    
    logger.info(f"Starting confabulation detection for: {prompt}")
    
    # Initialize clients
    token_tracker = TokenTracker()
    openai_client = OpenAIClient(token_tracker=token_tracker)
    web_search_tool = IntelligentWebSearchTool(relevance_threshold=0.5, max_concurrent=3)
    
    # Step 1: Generate N responses from the model
    logger.info("Generating model responses...")
    responses = await openai_client.generate_n_samples(prompt, N_SAMPLES)
    
    # Step 2: Generate embeddings for the responses
    logger.info("Generating embeddings...")
    embeddings = await openai_client.generate_n_embeddings(responses, N_SAMPLES)
    
    # Step 3: Compute semantic entropy
    logger.info("Computing semantic entropy...")
    entropy_score, similarity_matrix = compute_semantic_entropy(embeddings)
    
    # Step 4: Analyze confidence level
    is_confident = entropy_score < entropy_threshold
    
    logger.info(f"Entropy score: {entropy_score:.4f}, Confident: {is_confident}")
    
    # Step 5: Perform web search for grounding
    logger.info("Performing web search for grounding...")
    web_results = await web_search_tool.search(prompt, num_results=5, include_content=True)
    web_answer = web_results.get("answer", "No answer found")
    
    # Step 6: Determine confabulation status
    confabulation_detected = False
    analysis_result = ""
    comparison_result = ""
    
    if not is_confident:
        # High entropy = low confidence = potential confabulation
        confabulation_detected = True
        analysis_result = "HIGH_ENTROPY_CONFABULATION"
        comparison_result = "Model shows low confidence (high entropy), indicating potential confabulation"
    else:
        # Low entropy = high confidence, need to check accuracy
        logger.info("Comparing confident responses with web search answer...")
        is_consistent, comparison_details = await compare_responses(responses, web_answer, prompt)
        
        if is_consistent:
            confabulation_detected = False
            analysis_result = "NO_CONFABULATION"
            comparison_result = f"Model responses are consistent with web search. {comparison_details}"
        else:
            confabulation_detected = True
            analysis_result = "CONFIDENT_BUT_WRONG_CONFABULATION"
            comparison_result = f"Model is confident but wrong compared to web search. {comparison_details}"
    
    # Compile results
    results = {
        "query": prompt,
        "entropy_score": entropy_score,
        "entropy_threshold": entropy_threshold,
        "is_confident": is_confident,
        "confabulation_detected": confabulation_detected,
        "analysis_result": analysis_result,
        "model_responses": responses,
        "web_answer": web_answer,
        "comparison_result": comparison_result,
        "similarity_matrix": similarity_matrix,
        "web_search_summary": web_results.get("summary", {}),
        "total_tokens_used": token_tracker.get_total_tokens()
    }
    
    return results

def save_results(results, filename="confabulation_analysis.txt"):
    """Save analysis results to file"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write("CONFABULATION DETECTION ANALYSIS\n")
        file.write("=" * 50 + "\n\n")
        
        file.write(f"Query: {results['query']}\n")
        file.write(f"Entropy Score: {results['entropy_score']:.4f}\n")
        file.write(f"Entropy Threshold: {results['entropy_threshold']:.4f}\n")
        file.write(f"Model Confident: {results['is_confident']}\n")
        file.write(f"Confabulation Detected: {results['confabulation_detected']}\n")
        file.write(f"Analysis Result: {results['analysis_result']}\n\n")
        
        file.write("COMPARISON ANALYSIS:\n")
        file.write(f"{results['comparison_result']}\n\n")
        
        file.write("MODEL RESPONSES:\n")
        for i, response in enumerate(results['model_responses']):
            file.write(f"[{i+1}] {response}\n\n")
        
        file.write("WEB SEARCH ANSWER:\n")
        file.write(f"{results['web_answer']}\n\n")
        
        file.write("WEB SEARCH SUMMARY:\n")
        summary = results['web_search_summary']
        file.write(f"URLs analyzed: {summary.get('urls_analyzed', 'N/A')}\n")
        file.write(f"Relevant results: {summary.get('relevant_count', 'N/A')}\n")
        file.write(f"Success rate: {summary.get('success_rate', 'N/A'):.1%}\n\n")
        
        file.write("SIMILARITY MATRIX:\n")
        for row in results['similarity_matrix']:
            file.write(" ".join(f"{val:.4f}" for val in row) + "\n")
        
        file.write(f"\nTotal Tokens Used: {results['total_tokens_used']}\n")

def print_summary(results):
    """Print a summary of the analysis"""
    print("\nCONFABULATION DETECTION SUMMARY")
    print("=" * 50)
    print(f"Query: {results['query']}")
    print(f"Entropy Score: {results['entropy_score']:.4f}")
    print(f"Model Confident: {results['is_confident']}")
    print(f"Confabulation Detected: {results['confabulation_detected']}")
    print(f"Analysis: {results['analysis_result']}")
    print(f"\nComparison: {results['comparison_result'][:200]}...")
    print(f"\nModel Response Sample: {results['model_responses'][0][:150]}...")
    print(f"\nWeb Answer Sample: {results['web_answer'][:150]}...")
    print(f"\nTokens Used: {results['total_tokens_used']}")

async def main():
    """Main function to run confabulation detection"""
    
    # Test queries - mix of factual and potentially confabulated
    test_queries = [
        "How many R's are there in strawberry?",
        "Give me some papers on the relationship between homeschooling and neuroplasticity?",
        "Who won the 2025 Nobel Prize in Physics?",
        "What color is the sky on Mars?"
    ]
    
    for query in test_queries:
        print(f"\nAnalyzing: {query}")
        print("-" * 50)
        
        try:
            results = await detect_confabulation(query, entropy_threshold=0.3)
            print_summary(results)
            # Save detailed results
            filename = f"analysis_{query.replace('?', '').replace(' ', '_')[:30]}.txt"
            save_results(results, filename)
            
            # Optional: visualize similarity matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(results['similarity_matrix'], cmap='viridis')
            plt.colorbar(label='Cosine Similarity')
            plt.title(f"Semantic Similarity Matrix\nQuery: {query[:50]}...")
            plt.tight_layout()
            plt.savefig(f"similarity_{query.replace('?', '').replace(' ', '_')[:30]}.png")
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing query '{query}': {e}")
            logger.error(f"Error in main analysis: {e}")

if __name__ == "__main__":
    asyncio.run(main())