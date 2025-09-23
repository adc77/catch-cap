import asyncio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import matplotlib.pyplot as plt
from config.settings import N_SAMPLES
from clients.openai_client import OpenAIClient
from models.token_tracker import TokenTracker
from services.web_search_tool import IntelligentWebSearchTool
import logging

openai_client = OpenAIClient(token_tracker=TokenTracker())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_cluster_entropy(embeddings, n_clusters=None):
    """
    Compute Shannon entropy from K-means clustering of embeddings
    
    Args:
        embeddings: Array of embedding vectors
        n_clusters: Number of clusters (if None, uses optimal number)
    
    Returns:
        entropy_score: Shannon entropy of cluster distribution
        cluster_labels: Cluster assignments for each response
        cluster_probabilities: Probability distribution over clusters
    """
    n_samples = len(embeddings)
    
    # Determine optimal number of clusters if not specified
    if n_clusters is None:
        # Use elbow method or set reasonable default
        max_clusters = min(n_samples // 2, 5)  # Max 5 clusters or half the samples
        n_clusters = max(2, max_clusters)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute cluster probabilities
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_probabilities = counts / n_samples
    
    # Compute Shannon entropy
    shannon_entropy = entropy(cluster_probabilities, base=2)
    
    return shannon_entropy, cluster_labels, cluster_probabilities

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
        return is_consistent, comparison_result
    except Exception as e:
        logger.error(f"Error in response comparison: {e}")
        return False, f"Error during comparison: {str(e)}"

async def detect_confabulation(prompt, entropy_threshold=1.0, n_clusters=None):
    """
    Main function to detect confabulation using K-means clustering and Shannon entropy
    
    Args:
        prompt: The query to analyze
        entropy_threshold: Threshold above which model is considered uncertain
        n_clusters: Number of clusters for K-means (if None, auto-determined)
    
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
    
    # Step 3: Perform K-means clustering and compute Shannon entropy
    logger.info("Computing cluster entropy...")
    shannon_entropy, cluster_labels, cluster_probabilities = compute_cluster_entropy(embeddings, n_clusters)
    
    # Step 4: Analyze uncertainty level
    is_uncertain = shannon_entropy > entropy_threshold
    
    logger.info(f"Shannon entropy: {shannon_entropy:.4f}, Uncertain: {is_uncertain}")
    logger.info(f"Cluster probabilities: {cluster_probabilities}")
    
    # Step 5: Perform web search for grounding
    logger.info("Performing web search for grounding...")
    web_results = await web_search_tool.search(prompt, num_results=5, include_content=True)
    web_answer = web_results.get("answer", "No answer found")
    
    # Step 6: Determine confabulation status
    confabulation_detected = False
    analysis_result = ""
    comparison_result = ""
    
    if is_uncertain:
        # High entropy = uncertainty = confabulation
        confabulation_detected = True
        analysis_result = "HIGH_ENTROPY_CONFABULATION"
        comparison_result = "Model shows high uncertainty (high entropy), indicating confabulation"
    else:
        # Low entropy = certainty, need to check accuracy against web search
        logger.info("Comparing certain responses with web search answer...")
        is_consistent, comparison_details = await compare_responses(responses, web_answer, prompt)
        
        if is_consistent:
            confabulation_detected = False
            analysis_result = "NO_CONFABULATION_DETECTED"
            comparison_result = f"{comparison_details}"
        else:
            confabulation_detected = True
            analysis_result = "MODEL_CERTAIN_BUT_WRONG_ANSWER-CONFABULATION_DETECTED"
            comparison_result = f"{comparison_details}"
    
    # Compile results
    results = {
        "query": prompt,
        "shannon_entropy": shannon_entropy,
        "entropy_threshold": entropy_threshold,
        "is_uncertain": is_uncertain,
        "confabulation_detected": confabulation_detected,
        "analysis_result": analysis_result,
        "model_responses": responses,
        "web_answer": web_answer,
        "comparison_result": comparison_result,
        "cluster_labels": cluster_labels,
        "cluster_probabilities": cluster_probabilities,
        "n_clusters": len(cluster_probabilities),
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
        file.write(f"Shannon Entropy: {results['shannon_entropy']:.4f}\n")
        file.write(f"Entropy Threshold: {results['entropy_threshold']:.4f}\n")
        file.write(f"Model Uncertain: {results['is_uncertain']}\n")
        file.write(f"Confabulation Detected: {results['confabulation_detected']}\n")
        file.write(f"Analysis Result: {results['analysis_result']}\n")
        file.write(f"Number of Clusters: {results['n_clusters']}\n")
        file.write(f"Cluster Probabilities: {results['cluster_probabilities']}\n\n")
        
        file.write("COMPARISON ANALYSIS:\n")
        file.write(f"{results['comparison_result']}\n\n")
        
        file.write("MODEL RESPONSES:\n")
        for i, (response, cluster) in enumerate(zip(results['model_responses'], results['cluster_labels'])):
            file.write(f"[{i+1}] Cluster {cluster}: {response}\n\n")
        
        file.write("WEB SEARCH ANSWER:\n")
        file.write(f"{results['web_answer']}\n\n")
        
        file.write("WEB SEARCH SUMMARY:\n")
        summary = results['web_search_summary']
        file.write(f"URLs analyzed: {summary.get('urls_analyzed', 'N/A')}\n")
        file.write(f"Relevant results: {summary.get('relevant_count', 'N/A')}\n")
        file.write(f"Success rate: {summary.get('success_rate', 'N/A'):.1%}\n\n")
        
        file.write(f"Total Tokens Used: {results['total_tokens_used']}\n")

def print_summary(results):
    """Print a summary of the analysis"""
    print("\nCONFABULATION DETECTION SUMMARY")
    print("=" * 50)
    print(f"Query: {results['query']}")
    print(f"Shannon Entropy: {results['shannon_entropy']:.4f}")
    print(f"Model Uncertain: {results['is_uncertain']}")
    print(f"Confabulation Detected: {results['confabulation_detected']}")
    print(f"Analysis: {results['analysis_result']}")
    print(f"Clusters: {results['n_clusters']}")
    print(f"Cluster Distribution: {results['cluster_probabilities']}")
    print(f"\nComparison: {results['comparison_result'][:200]}...")
    print(f"\nModel Response Sample: {results['model_responses'][0][:150]}...")
    print(f"\nWeb Answer Sample: {results['web_answer'][:150]}...")
    print(f"\nTokens Used: {results['total_tokens_used']}")

def visualize_clusters(embeddings, cluster_labels, cluster_probabilities, query, filename=None):
    """Visualize clustering results"""
    from sklearn.decomposition import PCA
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, (label, color) in enumerate(zip(unique_labels, colors)):
        mask = cluster_labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[color], label=f'Cluster {label} ({cluster_probabilities[i]:.2f})', 
                   alpha=0.7, s=100)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'Response Clustering\nQuery: {query[:50]}...')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

async def main():
    """Main function to run confabulation detection"""
    
    # Test queries
    test_queries = [
        "How many R's are there in strawberry?",
        # "Can an LLP give a loan to a private limited company, if the directors are the same?"
    ]
    
    for query in test_queries:
        print(f"\nAnalyzing: {query}")
        print("-" * 50)
        
        try:
            results = await detect_confabulation(query, entropy_threshold=1.0)
            print_summary(results)
            
            # Save detailed results
            filename = f"analysis_{query.replace('?', '').replace(' ', '_')[:30]}.txt"
            save_results(results, filename)
            
            # # Visualize clusters
            # embeddings = await OpenAIClient(token_tracker=TokenTracker()).generate_n_embeddings(results['model_responses'], N_SAMPLES)
            # viz_filename = f"clusters_{query.replace('?', '').replace(' ', '_')[:30]}.png"
            # visualize_clusters(embeddings, results['cluster_labels'], results['cluster_probabilities'], query, viz_filename)
            
        except Exception as e:
            print(f"Error analyzing query '{query}': {e}")
            logger.error(f"Error in main analysis: {e}")

if __name__ == "__main__":
    asyncio.run(main())