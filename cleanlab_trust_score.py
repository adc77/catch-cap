# import os
# from cleanlab_tlm import TLM
# from sklearn.metrics import accuracy_score
# import requests
# from dotenv import load_dotenv
# load_dotenv()
# CLEANLAB_TLM_API_KEY = os.getenv("CLEANLAB_TLM_API_KEY")
# tlm = TLM(options={"log": ["explanation"], "model": "gpt-4o-mini"}) 

# # question = "How many r's are there in word strawberry?"
# question = "whats bigger 9.11 or 9.9 ?"
# output = tlm.prompt(question)

# print(f'Response: {output["response"]}')
# print(f'Trustworthiness Score: {output["trustworthiness_score"]}')
# print(f'Explanation: {output["log"]["explanation"]}')



import os
from cleanlab_tlm import TLM
import time
from dotenv import load_dotenv

load_dotenv()
CLEANLAB_TLM_API_KEY = os.getenv("CLEANLAB_TLM_API_KEY")
tlm = TLM(options={"log": ["explanation"], "model": "gpt-4.1-mini"}) 

def run_cleanlab_analysis():
    """Run Cleanlab TLM analysis on all test queries and save to file"""
    
    # Test queries from wstf_trust_score.py
    test_queries = [
        "How many R's are there in strawberry?",
        "Cite the Supreme Court's 2023 judgment Zeta v. Yama on the doctrine of lex loci delicti in Indian tort law.",
        "According to Section 56B of the Real Estate (Regulation and Development) Act, 2016, what are the penalties for delaying handover for more than six months?",
        "Summarize the recent central government notification GSR 987 (E), 2025, exempting e-vehicles from GST.",
        "In Guru Kabir v. State of U.P., 2024, how did the Allahabad High Court redefine the principle of sedition under IPC Section 124B?",
        "What are the unique stages under the Indian Arbitration and Conciliation (Amendment) Act, 2025, for fast-track domestic arbitration?",
        "Explain the ratio of the fictitious Supreme Court decision Hansa v. Hindustan, 2022 SCC 301, dealing with third-party insurance claims.",
        "Can an LLP give a loan to a private limited company, if the directors are the same?",
        "In 2000 was it legal for a karta in a Hindu family to disinherit his daughter; i.e., did she have no right to inheritance in absence of a will?",
        "Could talaq be recited thrice in 2000 and would it be considered as divorce under Muslim Personal Law in India?",
        "What is the Pinnel Rule?"
    ]
    
    # Initialize the analysis file
    filename = "cleanlab_trust_score_analysis.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write("CLEANLAB TLM TRUSTWORTHINESS ANALYSIS - ALL QUERIES\n")
        file.write("="*80 + "\n")
        file.write(f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Model: gpt-4.1-mini\n")
        file.write("="*80 + "\n")
    
    total_start_time = time.time()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nAnalyzing Query {i}/{len(test_queries)}: {query[:60]}...")
        print("-" * 60)
        
        try:
            # Time the query
            query_start_time = time.time()
            
            # Get TLM output
            output = tlm.prompt(query)
            
            query_end_time = time.time()
            query_duration = query_end_time - query_start_time
            
            # Extract results
            response = output["response"]
            trustworthiness_score = output["trustworthiness_score"]
            explanation = output["log"]["explanation"]
            
            # Print summary
            print(f"Response: {response[:100]}...")
            print(f"Trustworthiness Score: {trustworthiness_score:.4f}")
            print(f"Time taken: {query_duration:.2f} seconds")
            
            # Append to file
            with open(filename, "a", encoding="utf-8") as file:
                file.write(f"\n{'='*80}\n")
                file.write(f"QUERY {i}: {query}\n")
                file.write("="*50 + "\n\n")
                file.write(f"RESPONSE:\n{response}\n\n")
                file.write(f"TRUSTWORTHINESS SCORE: {trustworthiness_score:.4f}\n\n")
                file.write(f"EXPLANATION:\n{explanation}\n\n")
                file.write(f"QUERY EXECUTION TIME: {query_duration:.2f} seconds\n")
                file.write("="*80 + "\n")
                
        except Exception as e:
            error_msg = f"Error analyzing query '{query}': {str(e)}"
            print(error_msg)
            
            # Log error to file
            with open(filename, "a", encoding="utf-8") as file:
                file.write(f"\n{'='*80}\n")
                file.write(f"QUERY {i}: {query}\n")
                file.write("="*50 + "\n\n")
                file.write(f"ERROR: {error_msg}\n")
                file.write("="*80 + "\n")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Add completion summary
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"\n{'='*80}\n")
        file.write("ANALYSIS SUMMARY\n")
        file.write("="*50 + "\n")
        file.write(f"Total queries processed: {len(test_queries)}\n")
        file.write(f"Total analysis time: {total_duration:.2f} seconds\n")
        file.write(f"Average time per query: {total_duration/len(test_queries):.2f} seconds\n")
        file.write(f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("="*80 + "\n")
    
    print(f"\nAnalysis complete! Results saved to: {filename}")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"Average time per query: {total_duration/len(test_queries):.2f} seconds")

def analyze_single_query(query):
    """Analyze a single query and return results"""
    start_time = time.time()
    output = tlm.prompt(query)
    end_time = time.time()
    
    return {
        'query': query,
        'response': output["response"],
        'trustworthiness_score': output["trustworthiness_score"],
        'explanation': output["log"]["explanation"],
        'execution_time': end_time - start_time
    }

if __name__ == "__main__":
    run_cleanlab_analysis()