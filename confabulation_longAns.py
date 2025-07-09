# import asyncio
# import numpy as np
# from sklearn.cluster import KMeans
# from scipy.stats import entropy
# import logging
# from typing import List, Dict, Any
# from config.settings import N_SAMPLES
# from clients.openai_client import OpenAIClient
# from models.token_tracker import TokenTracker
# from services.web_search_tool import IntelligentWebSearchTool
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class LongFormConfabulationDetector:
#     """Detects confabulation in long-form responses using claim-based analysis"""
    
#     def __init__(self, entropy_threshold=1.0, risk_threshold=0.6):
#         self.entropy_threshold = entropy_threshold
#         self.risk_threshold = risk_threshold
#         self.token_tracker = TokenTracker()
#         self.openai_client = OpenAIClient(token_tracker=self.token_tracker)
#         self.web_search_tool = IntelligentWebSearchTool(relevance_threshold=0.5, max_concurrent=3)
    
#     async def extract_claims(self, paragraph: str) -> List[str]:
#         """Extract individual claims from a paragraph response"""
        
#         claims_prompt = f"""Extract all factual claims from the following paragraph. Each claim should be a single, verifiable statement.

#         Paragraph:
#         {paragraph}

#         Instructions:
#         - Extract each distinct factual claim as a separate item
#         - Each claim should be complete and standalone
#         - Focus on factual statements, not opinions or general statements
#         - Return each claim on a new line with no numbering or bullets
#         - If no factual claims exist, return "NO_CLAIMS"

#         Claims:"""

#         try:
#             response = await self.openai_client.chat_completion(claims_prompt, model="gpt-4.1-nano")
            
#             if "NO_CLAIMS" in response.upper():
#                 return []
            
#             # Split response into individual claims and clean
#             claims = [claim.strip() for claim in response.split('\n') if claim.strip()]
#             claims = [claim for claim in claims if len(claim) > 10]  # Filter out very short claims
            
#             logger.info(f"Extracted {len(claims)} claims from paragraph")
#             return claims
            
#         except Exception as e:
#             logger.error(f"Error extracting claims: {e}")
#             return []
    
#     async def generate_questions_for_claim(self, claim: str) -> List[str]:
#         """Generate 3 factoid questions that can be answered by the given claim"""
        
#         questions_prompt = f"""Generate exactly 3 factoid questions that can be answered by the following claim. Each question should be specific and verifiable.

#         Claim:
#         {claim}

#         Instructions:
#         - Generate exactly 3 questions
#         - Each question should be answerable by the claim
#         - Questions should be factual and specific
#         - Return each question on a new line
#         - No numbering or bullets

#         Questions:"""

#         try:
#             response = await self.openai_client.chat_completion(questions_prompt, model="gpt-4.1-nano")
            
#             # Split and clean questions
#             questions = [q.strip() for q in response.split('\n') if q.strip()]
#             questions = [q for q in questions if '?' in q]  # Ensure they are questions
            
#             # Ensure we have exactly 3 questions
#             if len(questions) > 3:
#                 questions = questions[:3]
#             elif len(questions) < 3:
#                 # Pad with generic questions if needed
#                 while len(questions) < 3:
#                     questions.append(f"What does the claim state about this topic?")
            
#             return questions
            
#         except Exception as e:
#             logger.error(f"Error generating questions for claim: {e}")
#             return [f"What does the claim state?"] * 3
    
#     def compute_cluster_entropy(self, embeddings, n_clusters=None):
#         """Compute Shannon entropy from K-means clustering of embeddings"""
#         n_samples = len(embeddings)
        
#         if n_clusters is None:
#             max_clusters = min(n_samples // 2, 5)
#             n_clusters = max(2, max_clusters)
        
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         cluster_labels = kmeans.fit_predict(embeddings)
        
#         unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#         cluster_probabilities = counts / n_samples
        
#         shannon_entropy = entropy(cluster_probabilities, base=2)
        
#         return shannon_entropy, cluster_labels, cluster_probabilities
    
#     async def analyze_question_entropy(self, question: str) -> float:
#         """Analyze entropy for a single question using the short-form method"""
        
#         try:
#             # Generate N responses for the question
#             responses = await self.openai_client.generate_n_samples(question, N_SAMPLES)
            
#             # Generate embeddings
#             embeddings = await self.openai_client.generate_n_embeddings(responses, N_SAMPLES)
            
#             # Compute entropy
#             shannon_entropy, _, _ = self.compute_cluster_entropy(embeddings)
            
#             return shannon_entropy
            
#         except Exception as e:
#             logger.error(f"Error analyzing question entropy: {e}")
#             return float('inf')  # Return high entropy on error
    
#     async def analyze_claim_entropy(self, claim: str) -> Dict[str, Any]:
#         """Analyze entropy for a claim by generating questions and computing average entropy"""
        
#         logger.info(f"Analyzing claim: {claim[:100]}...")
        
#         # Generate questions for the claim
#         questions = await self.generate_questions_for_claim(claim)
        
#         # Analyze entropy for each question
#         question_entropies = []
#         for i, question in enumerate(questions):
#             logger.info(f"  Analyzing question {i+1}/3: {question[:60]}...")
#             entropy_score = await self.analyze_question_entropy(question)
#             question_entropies.append(entropy_score)
        
#         # Compute average entropy for the claim
#         claim_entropy = np.mean(question_entropies)
        
#         return {
#             "claim": claim,
#             "questions": questions,
#             "question_entropies": question_entropies,
#             "claim_entropy": claim_entropy
#         }
    
#     def compute_risk_score(self, paragraph_entropy: float) -> float:
#         """Compute risk score based on paragraph entropy"""
#         # Normalize entropy to risk score (0-1 scale)
#         # Higher entropy = higher risk
#         risk_score = min(paragraph_entropy / 2.0, 1.0)  # Assuming max reasonable entropy is 2.0
#         return risk_score
    
#     async def compare_with_web_search(self, paragraph: str, claims: List[str]) -> Dict[str, Any]:
#         """Compare paragraph content with web search results"""
        
#         # Create a summary query from the main topic of the paragraph
#         summary_prompt = f"""Create a concise search query (max 10 words) that captures the main topic of this paragraph:

#         {paragraph}

#         Query:"""
        
#         try:
#             search_query = await self.openai_client.chat_completion(summary_prompt, model="gpt-4.1-nano")
#             search_query = search_query.strip()
            
#             # Perform web search
#             web_results = await self.web_search_tool.search(search_query, num_results=5, include_content=True)
#             web_answer = web_results.get("answer", "No answer found")
            
#             # Compare paragraph with web answer
#             comparison_prompt = f"""Compare the following paragraph with the web search answer for consistency.

#             Original Paragraph:
#             {paragraph}

#             Web Search Answer:
#             {web_answer}

#             Instructions:
#             - Determine if the paragraph is factually consistent with the web search answer
#             - Consider the main claims and facts
#             - Ignore minor differences in phrasing or style
#             - Return only "CONSISTENT" or "INCONSISTENT"

#             Result:"""
            
#             comparison_result = await self.openai_client.chat_completion(comparison_prompt, model="gpt-4.1-nano")
#             is_consistent = "CONSISTENT" in comparison_result.upper() and "INCONSISTENT" not in comparison_result.upper()
            
#             return {
#                 "search_query": search_query,
#                 "web_answer": web_answer,
#                 "is_consistent": is_consistent,
#                 "comparison_result": comparison_result,
#                 "web_summary": web_results.get("summary", {})
#             }
            
#         except Exception as e:
#             logger.error(f"Error in web search comparison: {e}")
#             return {
#                 "search_query": "error",
#                 "web_answer": "Error during search",
#                 "is_consistent": False,
#                 "comparison_result": f"Error: {str(e)}",
#                 "web_summary": {}
#             }
    
#     async def detect_confabulation(self, paragraph: str) -> Dict[str, Any]:
#         """Main function to detect confabulation in long-form responses"""
        
#         logger.info(f"Starting long-form confabulation detection for paragraph of {len(paragraph)} characters")
        
#         # Step 1: Extract claims from paragraph
#         logger.info("Extracting claims from paragraph...")
#         claims = await self.extract_claims(paragraph)
        
#         if not claims:
#             return {
#                 "paragraph": paragraph,
#                 "confabulation_detected": False,
#                 "analysis_result": "NO_CLAIMS_FOUND",
#                 "risk_score": 0.0,
#                 "claims_analysis": [],
#                 "paragraph_entropy": 0.0,
#                 "web_comparison": {},
#                 "total_tokens_used": self.token_tracker.get_total_tokens()
#             }
        
#         # Step 2: Analyze entropy for each claim
#         logger.info(f"Analyzing entropy for {len(claims)} claims...")
#         claims_analysis = []
        
#         for i, claim in enumerate(claims):
#             logger.info(f"Processing claim {i+1}/{len(claims)}")
#             claim_analysis = await self.analyze_claim_entropy(claim)
#             claims_analysis.append(claim_analysis)
        
#         # Step 3: Compute paragraph entropy (mean of all claim entropies)
#         claim_entropies = [analysis["claim_entropy"] for analysis in claims_analysis]
#         paragraph_entropy = np.mean(claim_entropies)
        
#         # Step 4: Compute risk score
#         risk_score = self.compute_risk_score(paragraph_entropy)
        
#         logger.info(f"Paragraph entropy: {paragraph_entropy:.4f}, Risk score: {risk_score:.4f}")
        
#         # Step 5: Determine confabulation based on risk score
#         confabulation_detected = False
#         analysis_result = ""
#         web_comparison = {}
        
#         if risk_score > self.risk_threshold:
#             # High risk - flag confabulation directly
#             confabulation_detected = True
#             analysis_result = "HIGH_RISK_CONFABULATION"
#         else:
#             # Low risk - check with web search
#             logger.info("Low risk detected, comparing with web search...")
#             web_comparison = await self.compare_with_web_search(paragraph, claims)
            
#             if web_comparison["is_consistent"]:
#                 confabulation_detected = False
#                 analysis_result = "LOW_RISK_CONSISTENT_WITH_WEB"
#             else:
#                 confabulation_detected = True
#                 analysis_result = "LOW_RISK_INCONSISTENT_WITH_WEB"
        
#         # Compile results
#         results = {
#             "paragraph": paragraph,
#             "confabulation_detected": confabulation_detected,
#             "analysis_result": analysis_result,
#             "risk_score": risk_score,
#             "paragraph_entropy": paragraph_entropy,
#             "entropy_threshold": self.entropy_threshold,
#             "risk_threshold": self.risk_threshold,
#             "num_claims": len(claims),
#             "claims_analysis": claims_analysis,
#             "web_comparison": web_comparison,
#             "total_tokens_used": self.token_tracker.get_total_tokens()
#         }
        
#         logger.info(f"Analysis complete: {analysis_result}, Confabulation: {confabulation_detected}")
        
#         return results
    
#     def save_results(self, results: Dict[str, Any], filename: str = "longform_analysis.txt"):
#         """Save detailed analysis results to file"""
        
#         with open(filename, "w", encoding="utf-8") as file:
#             file.write("LONG-FORM CONFABULATION DETECTION ANALYSIS\n")
#             file.write("=" * 60 + "\n\n")
            
#             file.write(f"Paragraph Length: {len(results['paragraph'])} characters\n")
#             file.write(f"Number of Claims: {results['num_claims']}\n")
#             file.write(f"Paragraph Entropy: {results['paragraph_entropy']:.4f}\n")
#             file.write(f"Risk Score: {results['risk_score']:.4f}\n")
#             file.write(f"Risk Threshold: {results['risk_threshold']:.4f}\n")
#             file.write(f"Confabulation Detected: {results['confabulation_detected']}\n")
#             file.write(f"Analysis Result: {results['analysis_result']}\n\n")
            
#             file.write("ORIGINAL PARAGRAPH:\n")
#             file.write(f"{results['paragraph']}\n\n")
            
#             file.write("CLAIMS ANALYSIS:\n")
#             file.write("-" * 40 + "\n")
            
#             for i, claim_analysis in enumerate(results['claims_analysis']):
#                 file.write(f"\nClaim {i+1}:\n")
#                 file.write(f"Text: {claim_analysis['claim']}\n")
#                 file.write(f"Entropy: {claim_analysis['claim_entropy']:.4f}\n")
#                 file.write("Questions:\n")
#                 for j, (question, entropy) in enumerate(zip(claim_analysis['questions'], claim_analysis['question_entropies'])):
#                     file.write(f"  {j+1}. {question} (Entropy: {entropy:.4f})\n")
#                 file.write("\n")
            
#             if results['web_comparison']:
#                 file.write("WEB SEARCH COMPARISON:\n")
#                 file.write("-" * 40 + "\n")
#                 file.write(f"Search Query: {results['web_comparison']['search_query']}\n")
#                 file.write(f"Consistent: {results['web_comparison']['is_consistent']}\n")
#                 file.write(f"Comparison Result: {results['web_comparison']['comparison_result']}\n")
#                 file.write(f"Web Answer: {results['web_comparison']['web_answer'][:500]}...\n\n")
            
#             file.write(f"Total Tokens Used: {results['total_tokens_used']}\n")
    
#     def print_summary(self, results: Dict[str, Any]):
#         """Print a concise summary of the analysis"""
        
#         print("\nLONG-FORM CONFABULATION DETECTION SUMMARY")
#         print("=" * 60)
#         print(f"Paragraph Length: {len(results['paragraph'])} characters")
#         print(f"Number of Claims: {results['num_claims']}")
#         print(f"Paragraph Entropy: {results['paragraph_entropy']:.4f}")
#         print(f"Risk Score: {results['risk_score']:.4f}")
#         print(f"Confabulation Detected: {results['confabulation_detected']}")
#         print(f"Analysis Result: {results['analysis_result']}")
        
#         if results['web_comparison']:
#             print(f"Web Consistency: {results['web_comparison']['is_consistent']}")
        
#         claim_entropies = [f"{ca['claim_entropy']:.3f}" for ca in results['claims_analysis']]
#         print(f"\nClaim Entropies: {claim_entropies}")
#         print(f"Tokens Used: {results['total_tokens_used']}")

# async def main():
#     """Example usage of the LongFormConfabulationDetector"""
    
#     # Initialize detector
#     detector = LongFormConfabulationDetector(
#         entropy_threshold=1.0,
#         risk_threshold=0.6
#     )
    
#     test_paragraphs = [
#          """
#         Yes, an LLP (Limited Liability Partnership) can generally give a loan to a private limited company, even if some of the same individuals are directors or members of both entities. However, there are important legal and regulatory considerations to keep in mind:

#         1. **Legal Authority and Governing Documents:**
#         - The LLP’s partnership agreement and the resolution of its members should explicitly authorize the LLP to lend money to other entities.
#         - Similarly, the private limited company’s articles of association and board resolutions should approve the acceptance of the loan.

#         2. **Related Party Transactions:**
#         - Since the same individuals are involved in both entities, the loan could be classified as a related party transaction.
#         - Such transactions must be conducted at arm’s length, meaning the terms should be comparable to those with an unrelated third party.

#         3. **Compliance with Laws and Regulations:**
#         - In many jurisdictions, including India (under the Companies Act, 2013), there are restrictions on loans given by a company to its directors or related parties.
#         - For LLPs, applicable laws (such as the Limited Liability Partnership Act, 2008, and relevant regulations) should be checked to ensure there are no prohibitions or restrictions.

#         4. **Tax Implications:**
#         - The transaction may have tax implications, such as withholding taxes or transfer pricing considerations, especially if the entities are in different jurisdictions or if the loan terms are not at arm’s length.

#         5. **Disclosure and Reporting:**
#         - Proper disclosures should be made in the financial statements of both entities, especially if the loan is significant.

#         **Summary:**
#         While there is no outright prohibition against an LLP lending to a private limited company with overlapping directors, it is crucial to ensure that the transaction is properly authorized, conducted at arm’s length, and compliant with applicable laws. Consulting with a legal or financial advisor is advisable to navigate specific legal nuances and ensure compliance.
#         """
#     ]
    
#     for i, paragraph in enumerate(test_paragraphs):
#         print(f"\n{'='*80}")
#         print(f"ANALYZING PARAGRAPH {i+1}")
#         print('='*80)
        
#         try:
#             results = await detector.detect_confabulation(paragraph.strip())
#             detector.print_summary(results)
            
#             # Save detailed results
#             filename = f"longform_analysis_{i+1}.txt"
#             detector.save_results(results, filename)
            
#         except Exception as e:
#             print(f"Error analyzing paragraph {i+1}: {e}")
#             logger.error(f"Error in main analysis: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
import logging
from typing import List, Dict, Any
from config.settings import N_SAMPLES
from clients.openai_client import OpenAIClient
from models.token_tracker import TokenTracker
from services.web_search_tool import IntelligentWebSearchTool
import re
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastLongFormConfabulationDetector:
    """Fast version with parallel processing for claim-based confabulation detection"""
    
    def __init__(self, entropy_threshold=1.0, risk_threshold=0.6, max_concurrent_claims=3, max_concurrent_questions=5):
        self.entropy_threshold = entropy_threshold
        self.risk_threshold = risk_threshold
        self.max_concurrent_claims = max_concurrent_claims
        self.max_concurrent_questions = max_concurrent_questions
        self.token_tracker = TokenTracker()
        self.openai_client = OpenAIClient(token_tracker=self.token_tracker)
        self.web_search_tool = IntelligentWebSearchTool(relevance_threshold=0.5, max_concurrent=3)
    
    async def extract_claims(self, paragraph: str) -> List[str]:
        """Extract individual claims from a paragraph response"""
        
        claims_prompt = f"""Extract all factual claims from the following paragraph. Each claim should be a single, verifiable statement.

        Paragraph:
        {paragraph}

        Instructions:
        - Extract each distinct factual claim as a separate item
        - Each claim should be complete and standalone
        - Focus on factual statements, not opinions or general statements
        - Return each claim on a new line with no numbering or bullets
        - If no factual claims exist, return "NO_CLAIMS"

        Claims:"""

        try:
            response = await self.openai_client.chat_completion(claims_prompt, model="gpt-4o-mini")
            
            if "NO_CLAIMS" in response.upper():
                return []
            
            # Split response into individual claims and clean
            claims = [claim.strip() for claim in response.split('\n') if claim.strip()]
            claims = [claim for claim in claims if len(claim) > 10]  # Filter out very short claims
            
            logger.info(f"Extracted {len(claims)} claims from paragraph")
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []
    
    async def generate_questions_for_claim_batch(self, claims: List[str]) -> List[List[str]]:
        """Generate questions for multiple claims in parallel"""
        
        async def generate_single_claim_questions(claim: str) -> List[str]:
            questions_prompt = f"""Generate exactly 3 factoid questions that can be answered by the following claim. Each question should be specific and verifiable.

            Claim:
            {claim}

            Instructions:
            - Generate exactly 3 questions
            - Each question should be answerable by the claim
            - Questions should be factual and specific
            - Return each question on a new line
            - No numbering or bullets

            Questions:"""

            try:
                response = await self.openai_client.chat_completion(questions_prompt, model="gpt-4o-mini")
                
                # Split and clean questions
                questions = [q.strip() for q in response.split('\n') if q.strip()]
                questions = [q for q in questions if '?' in q]  # Ensure they are questions
                
                # Ensure we have exactly 3 questions
                if len(questions) > 3:
                    questions = questions[:3]
                elif len(questions) < 3:
                    # Pad with generic questions if needed
                    while len(questions) < 3:
                        questions.append(f"What does the claim state about this topic?")
                
                return questions
                
            except Exception as e:
                logger.error(f"Error generating questions for claim: {e}")
                return [f"What does the claim state?"] * 3
        
        # Process claims in parallel batches
        semaphore = asyncio.Semaphore(self.max_concurrent_claims)
        
        async def process_claim_with_semaphore(claim):
            async with semaphore:
                return await generate_single_claim_questions(claim)
        
        # Run all claim question generation in parallel
        logger.info(f"Generating questions for {len(claims)} claims in parallel...")
        start_time = time.time()
        
        tasks = [process_claim_with_semaphore(claim) for claim in claims]
        all_questions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_questions = []
        for i, result in enumerate(all_questions):
            if isinstance(result, Exception):
                logger.error(f"Error processing claim {i}: {result}")
                processed_questions.append([f"What does the claim state?"] * 3)
            else:
                processed_questions.append(result)
        
        logger.info(f"Generated questions for {len(claims)} claims in {time.time() - start_time:.2f} seconds")
        return processed_questions
    
    def compute_cluster_entropy(self, embeddings, n_clusters=None):
        """Compute Shannon entropy from K-means clustering of embeddings"""
        n_samples = len(embeddings)
        
        if n_clusters is None:
            max_clusters = min(n_samples // 2, 5)
            n_clusters = max(2, max_clusters)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_probabilities = counts / n_samples
        
        shannon_entropy = entropy(cluster_probabilities, base=2)
        
        return shannon_entropy, cluster_labels, cluster_probabilities
    
    async def analyze_question_entropy_batch(self, questions: List[str]) -> List[float]:
        """Analyze entropy for multiple questions in parallel"""
        
        async def analyze_single_question(question: str) -> float:
            try:
                # Generate N responses for the question
                responses = await self.openai_client.generate_n_samples(question, N_SAMPLES)
                
                # Generate embeddings
                embeddings = await self.openai_client.generate_n_embeddings(responses, N_SAMPLES)
                
                # Compute entropy
                shannon_entropy, _, _ = self.compute_cluster_entropy(embeddings)
                
                return shannon_entropy
                
            except Exception as e:
                logger.error(f"Error analyzing question entropy: {e}")
                return float('inf')  # Return high entropy on error
        
        # Process questions in parallel with semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent_questions)
        
        async def process_question_with_semaphore(question):
            async with semaphore:
                return await analyze_single_question(question)
        
        # Run all question entropy analysis in parallel
        logger.info(f"Analyzing entropy for {len(questions)} questions in parallel...")
        start_time = time.time()
        
        tasks = [process_question_with_semaphore(question) for question in questions]
        entropies = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_entropies = []
        for i, result in enumerate(entropies):
            if isinstance(result, Exception):
                logger.error(f"Error processing question {i}: {result}")
                processed_entropies.append(float('inf'))
            else:
                processed_entropies.append(result)
        
        logger.info(f"Analyzed {len(questions)} questions in {time.time() - start_time:.2f} seconds")
        return processed_entropies
    
    async def analyze_all_claims_parallel(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Analyze all claims in parallel using batched processing"""
        
        logger.info(f"Starting parallel analysis of {len(claims)} claims...")
        total_start_time = time.time()
        
        # Step 1: Generate questions for all claims in parallel
        all_questions_lists = await self.generate_questions_for_claim_batch(claims)
        
        # Step 2: Flatten all questions and track which claim they belong to
        all_questions = []
        question_to_claim_map = []
        
        for claim_idx, questions in enumerate(all_questions_lists):
            for question in questions:
                all_questions.append(question)
                question_to_claim_map.append(claim_idx)
        
        # Step 3: Analyze entropy for all questions in parallel
        all_question_entropies = await self.analyze_question_entropy_batch(all_questions)
        
        # Step 4: Group results back by claim
        claims_analysis = []
        
        for claim_idx, claim in enumerate(claims):
            # Find questions and entropies for this claim
            claim_questions = all_questions_lists[claim_idx]
            
            # Find corresponding entropies
            claim_question_entropies = []
            question_idx = 0
            for i, mapped_claim_idx in enumerate(question_to_claim_map):
                if mapped_claim_idx == claim_idx:
                    claim_question_entropies.append(all_question_entropies[i])
                    question_idx += 1
                    if question_idx >= 3:  # We only have 3 questions per claim
                        break
            
            # Compute average entropy for the claim
            claim_entropy = np.mean(claim_question_entropies)
            
            claims_analysis.append({
                "claim": claim,
                "questions": claim_questions,
                "question_entropies": claim_question_entropies,
                "claim_entropy": claim_entropy
            })
        
        logger.info(f"Completed parallel analysis of {len(claims)} claims in {time.time() - total_start_time:.2f} seconds")
        return claims_analysis
    
    def compute_risk_score(self, paragraph_entropy: float) -> float:
        """Compute risk score based on paragraph entropy"""
        # Normalize entropy to risk score (0-1 scale)
        # Higher entropy = higher risk
        risk_score = min(paragraph_entropy / 2.0, 1.0)  # Assuming max reasonable entropy is 2.0
        return risk_score
    
    async def compare_with_web_search(self, paragraph: str, claims: List[str]) -> Dict[str, Any]:
        """Compare paragraph content with web search results"""
        
        # Create a summary query from the main topic of the paragraph
        summary_prompt = f"""Create a concise search query (max 10 words) that captures the main topic of this paragraph:

        {paragraph}

        Query:"""
        
        try:
            # Run query generation and web search in parallel
            search_query_task = self.openai_client.chat_completion(summary_prompt, model="gpt-4o-mini")
            
            search_query = await search_query_task
            search_query = search_query.strip()
            
            # Perform web search
            web_results = await self.web_search_tool.search(search_query, num_results=5, include_content=True)
            web_answer = web_results.get("answer", "No answer found")
            
            # Compare paragraph with web answer
            comparison_prompt = f"""Compare the following paragraph with the web search answer for consistency.

            Original Paragraph:
            {paragraph}

            Web Search Answer:
            {web_answer}

            Instructions:
            - Determine if the paragraph is factually consistent with the web search answer
            - Consider the main claims and facts
            - Ignore minor differences in phrasing or style
            - Return only "CONSISTENT" or "INCONSISTENT"

            Result:"""
            
            comparison_result = await self.openai_client.chat_completion(comparison_prompt, model="gpt-4o-mini")
            is_consistent = "CONSISTENT" in comparison_result.upper() and "INCONSISTENT" not in comparison_result.upper()
            
            return {
                "search_query": search_query,
                "web_answer": web_answer,
                "is_consistent": is_consistent,
                "comparison_result": comparison_result,
                "web_summary": web_results.get("summary", {})
            }
            
        except Exception as e:
            logger.error(f"Error in web search comparison: {e}")
            return {
                "search_query": "error",
                "web_answer": "Error during search",
                "is_consistent": False,
                "comparison_result": f"Error: {str(e)}",
                "web_summary": {}
            }
    
    async def detect_confabulation(self, paragraph: str) -> Dict[str, Any]:
        """Main function to detect confabulation in long-form responses with parallel processing"""
        
        logger.info(f"Starting FAST long-form confabulation detection for paragraph of {len(paragraph)} characters")
        overall_start_time = time.time()
        
        # Step 1: Extract claims from paragraph
        logger.info("Extracting claims from paragraph...")
        claims = await self.extract_claims(paragraph)
        
        if not claims:
            return {
                "paragraph": paragraph,
                "confabulation_detected": False,
                "analysis_result": "NO_CLAIMS_FOUND",
                "risk_score": 0.0,
                "claims_analysis": [],
                "paragraph_entropy": 0.0,
                "web_comparison": {},
                "total_tokens_used": self.token_tracker.get_total_tokens(),
                "processing_time": time.time() - overall_start_time
            }
        
        # Step 2: Analyze entropy for all claims in parallel
        logger.info(f"Starting parallel analysis of {len(claims)} claims...")
        claims_analysis = await self.analyze_all_claims_parallel(claims)
        
        # Step 3: Compute paragraph entropy (mean of all claim entropies)
        claim_entropies = [analysis["claim_entropy"] for analysis in claims_analysis]
        paragraph_entropy = np.mean(claim_entropies)
        
        # Step 4: Compute risk score
        risk_score = self.compute_risk_score(paragraph_entropy)
        
        logger.info(f"Paragraph entropy: {paragraph_entropy:.4f}, Risk score: {risk_score:.4f}")
        
        # Step 5: Determine confabulation based on risk score
        confabulation_detected = False
        analysis_result = ""
        web_comparison = {}
        
        if risk_score > self.risk_threshold:
            # High risk - flag confabulation directly
            confabulation_detected = True
            analysis_result = "HIGH_RISK_CONFABULATION"
        else:
            # Low risk - check with web search
            logger.info("Low risk detected, comparing with web search...")
            web_comparison = await self.compare_with_web_search(paragraph, claims)
            
            if web_comparison["is_consistent"]:
                confabulation_detected = False
                analysis_result = "LOW_RISK_CONSISTENT_WITH_WEB"
            else:
                confabulation_detected = True
                analysis_result = "LOW_RISK_INCONSISTENT_WITH_WEB"
        
        total_processing_time = time.time() - overall_start_time
        
        # Compile results
        results = {
            "paragraph": paragraph,
            "confabulation_detected": confabulation_detected,
            "analysis_result": analysis_result,
            "risk_score": risk_score,
            "paragraph_entropy": paragraph_entropy,
            "entropy_threshold": self.entropy_threshold,
            "risk_threshold": self.risk_threshold,
            "num_claims": len(claims),
            "claims_analysis": claims_analysis,
            "web_comparison": web_comparison,
            "total_tokens_used": self.token_tracker.get_total_tokens(),
            "processing_time": total_processing_time
        }
        
        logger.info(f"FAST analysis complete in {total_processing_time:.2f} seconds: {analysis_result}, Confabulation: {confabulation_detected}")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a concise summary of the analysis"""
        
        print("\nFAST LONG-FORM CONFABULATION DETECTION SUMMARY")
        print("=" * 60)
        print(f"Processing Time: {results.get('processing_time', 0):.2f} seconds")
        print(f"Paragraph Length: {len(results['paragraph'])} characters")
        print(f"Number of Claims: {results['num_claims']}")
        print(f"Paragraph Entropy: {results['paragraph_entropy']:.4f}")
        print(f"Risk Score: {results['risk_score']:.4f}")
        print(f"Confabulation Detected: {results['confabulation_detected']}")
        print(f"Analysis Result: {results['analysis_result']}")
        
        if results['web_comparison']:
            print(f"Web Consistency: {results['web_comparison']['is_consistent']}")
        
        claim_entropies = [f"{ca['claim_entropy']:.3f}" for ca in results['claims_analysis']]
        print(f"\nClaim Entropies: {claim_entropies}")
        print(f"Tokens Used: {results['total_tokens_used']}")

# Performance comparison function
async def compare_performance():
    """Compare performance between original and fast versions"""
    
    from confabulation_longAns import LongFormConfabulationDetector as OriginalDetector
    
    test_paragraph = """
    Yes, an LLP (Limited Liability Partnership) can generally give a loan to a private limited company, even if some of the same individuals are directors or members of both entities. However, there are important legal and regulatory considerations to keep in mind:

    1. **Legal Authority and Governing Documents:**
    - The LLP’s partnership agreement and the resolution of its members should explicitly authorize the LLP to lend money to other entities.
    - Similarly, the private limited company’s articles of association and board resolutions should approve the acceptance of the loan.

    2. **Related Party Transactions:**
    - Since the same individuals are involved in both entities, the loan could be classified as a related party transaction.
    - Such transactions must be conducted at arm’s length, meaning the terms should be comparable to those with an unrelated third party.

    3. **Compliance with Laws and Regulations:**
    - In many jurisdictions, including India (under the Companies Act, 2013), there are restrictions on loans given by a company to its directors or related parties.
    - For LLPs, applicable laws (such as the Limited Liability Partnership Act, 2008, and relevant regulations) should be checked to ensure there are no prohibitions or restrictions.

    4. **Tax Implications:**
    - The transaction may have tax implications, such as withholding taxes or transfer pricing considerations, especially if the entities are in different jurisdictions or if the loan terms are not at arm’s length.

    5. **Disclosure and Reporting:**
    - Proper disclosures should be made in the financial statements of both entities, especially if the loan is significant.

    **Summary:**
    While there is no outright prohibition against an LLP lending to a private limited company with overlapping directors, it is crucial to ensure that the transaction is properly authorized, conducted at arm’s length, and compliant with applicable laws. Consulting with a legal or financial advisor is advisable to navigate specific legal nuances and ensure compliance.
    """
    
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Test fast version
    print("\nTesting FAST version...")
    fast_detector = FastLongFormConfabulationDetector()
    start_time = time.time()
    fast_results = await fast_detector.detect_confabulation(test_paragraph.strip())
    fast_time = time.time() - start_time
    
    # Test original version
    # print("\nTesting ORIGINAL version...")
    # original_detector = OriginalDetector()
    # start_time = time.time()
    # original_results = await original_detector.detect_confabulation(test_paragraph.strip())
    # original_time = time.time() - start_time
    
    print(f"\nPerformance Results:")
    print(f"Original version: {original_time:.2f} seconds")
    print(f"Fast version: {fast_time:.2f} seconds")
    print(f"Speed improvement: {original_time/fast_time:.2f}x faster")
    
    print(f"\nResults comparison:")
    print(f"Original confabulation detected: {original_results['confabulation_detected']}")
    print(f"Fast confabulation detected: {fast_results['confabulation_detected']}")

async def main():
    """Example usage of the FastLongFormConfabulationDetector"""
    
    # Initialize fast detector
    detector = FastLongFormConfabulationDetector(
        entropy_threshold=1.0,
        risk_threshold=0.6,
        max_concurrent_claims=5,
        max_concurrent_questions=5 
    )
    
    test_paragraphs = [
        """
       Yes, an LLP (Limited Liability Partnership) can generally give a loan to a private limited company, even if some of the same individuals are directors or members of both entities. However, there are important legal and regulatory considerations to keep in mind:

        1. **Legal Authority and Governing Documents:**
        - The LLP’s partnership agreement and the resolution of its members should explicitly authorize the LLP to lend money to other entities.
        - Similarly, the private limited company’s articles of association and board resolutions should approve the acceptance of the loan.

        2. **Related Party Transactions:**
        - Since the same individuals are involved in both entities, the loan could be classified as a related party transaction.
        - Such transactions must be conducted at arm’s length, meaning the terms should be comparable to those with an unrelated third party.

        3. **Compliance with Laws and Regulations:**
        - In many jurisdictions, including India (under the Companies Act, 2013), there are restrictions on loans given by a company to its directors or related parties.
        - For LLPs, applicable laws (such as the Limited Liability Partnership Act, 2008, and relevant regulations) should be checked to ensure there are no prohibitions or restrictions.

        4. **Tax Implications:**
        - The transaction may have tax implications, such as withholding taxes or transfer pricing considerations, especially if the entities are in different jurisdictions or if the loan terms are not at arm’s length.

        5. **Disclosure and Reporting:**
        - Proper disclosures should be made in the financial statements of both entities, especially if the loan is significant.

        **Summary:**
        While there is no outright prohibition against an LLP lending to a private limited company with overlapping directors, it is crucial to ensure that the transaction is properly authorized, conducted at arm’s length, and compliant with applicable laws. Consulting with a legal or financial advisor is advisable to navigate specific legal nuances and ensure compliance.
        """
    ]
    
    for i, paragraph in enumerate(test_paragraphs):
        print(f"\n{'='*80}")
        print(f"ANALYZING PARAGRAPH {i+1} (FAST VERSION)")
        print('='*80)
        
        try:
            results = await detector.detect_confabulation(paragraph.strip())
            detector.print_summary(results)
            
        except Exception as e:
            print(f"Error analyzing paragraph {i+1}: {e}")
            logger.error(f"Error in main analysis: {e}")
    
    # # Run performance comparison
    # print(f"\n{'='*80}")
    # await compare_performance()

if __name__ == "__main__":
    asyncio.run(main())