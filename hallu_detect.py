import asyncio
import tiktoken
import logging
from typing import Dict, Any, Tuple
from clients.openai_client import OpenAIClient
from models.token_tracker import TokenTracker

# Import confabulation detection modules
from confabulation_shortAns import detect_confabulation as detect_short_confabulation
from confabulation_longAns import FastLongFormConfabulationDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfabulationAwareResponseGenerator:
    """
    Generates responses with automatic confabulation detection and correction
    """
    
    def __init__(self, short_token_threshold: int = 30):
        self.short_token_threshold = short_token_threshold
        self.token_tracker = TokenTracker()
        self.openai_client = OpenAIClient(token_tracker=self.token_tracker)
        self.long_form_detector = FastLongFormConfabulationDetector(
            entropy_threshold=1.0,
            risk_threshold=0.6,
            max_concurrent_claims=5,
            max_concurrent_questions=5
        )
        
        # Initialize tokenizer for length measurement
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def determine_response_type(self, response: str) -> str:
        """Determine if response is short or long based on token count"""
        token_count = self.count_tokens(response)
        return "short" if token_count < self.short_token_threshold else "long"
    
    async def generate_initial_response(self, question: str) -> str:
        """Generate initial response to the user's question"""
        try:
            response = await self.openai_client.chat_completion(
                prompt=question,
                model="gpt-4.1-nano"
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating initial response: {e}")
            return "Error: Unable to generate response"
    
    async def generate_corrected_response(self, question: str, web_answer: str, confabulation_details: str) -> str:
        """Generate a corrected response using web search results"""
        
        correction_prompt = f"""You are tasked with providing an accurate answer to a question. The original AI response contained confabulation (false information). Use the provided web search information to give a factually correct answer.

        Question: {question}

        Web Search Information:
        {web_answer}

        Confabulation Analysis:
        {confabulation_details}

        Instructions:
        - Provide a factually accurate answer based on the web search information
        - Only include information that can be verified from the web search results
        - If the web search doesn't provide enough information, acknowledge this limitation
        - Do not mention the confabulation detection process in your response
        - Do not mention about web search in your response

        Corrected Answer:"""

        try:
            corrected_response = await self.openai_client.chat_completion(
                prompt=correction_prompt,
                model="gpt-4.1-nano"
            )
            return corrected_response.strip()
        except Exception as e:
            logger.error(f"Error generating corrected response: {e}")
            return f"Error generating correction. Web information: {web_answer[:200]}..."
    
    async def detect_short_form_confabulation(self, question: str) -> Dict[str, Any]:
        """Detect confabulation in short-form responses"""
        try:
            results = await detect_short_confabulation(
                prompt=question,
                entropy_threshold=1.0
            )
            return results
        except Exception as e:
            logger.error(f"Error in short-form confabulation detection: {e}")
            return {
                "confabulation_detected": True,
                "analysis_result": "ERROR_IN_DETECTION",
                "web_answer": "Error during analysis",
                "comparison_result": f"Error: {str(e)}"
            }
    
    async def detect_long_form_confabulation(self, response: str) -> Dict[str, Any]:
        """Detect confabulation in long-form responses"""
        try:
            results = await self.long_form_detector.detect_confabulation(response)
            return results
        except Exception as e:
            logger.error(f"Error in long-form confabulation detection: {e}")
            return {
                "confabulation_detected": True,
                "analysis_result": "ERROR_IN_DETECTION",
                "web_comparison": {"web_answer": "Error during analysis"},
                "comparison_result": f"Error: {str(e)}"
            }
    
    def extract_web_answer(self, confab_results: Dict[str, Any], response_type: str) -> str:
        """Extract web answer from confabulation results based on response type"""
        if response_type == "short":
            return confab_results.get("web_answer", "No web answer available")
        else:
            web_comparison = confab_results.get("web_comparison", {})
            return web_comparison.get("web_answer", "No web answer available")
    
    def format_confabulation_summary(self, confab_results: Dict[str, Any], response_type: str) -> str:
        """Format confabulation analysis summary"""
        if response_type == "short":
            return (
                f"Analysis: {confab_results.get('analysis_result', 'Unknown')}\n"
                f"Shannon Entropy: {confab_results.get('shannon_entropy', 0):.4f}\n"
                f"Model Uncertain: {confab_results.get('is_uncertain', False)}\n"
                f"Comparison: {confab_results.get('comparison_result', 'No comparison')[:100]}..."
            )
        else:
            return (
                f"Analysis: {confab_results.get('analysis_result', 'Unknown')}\n"
                f"Paragraph Entropy: {confab_results.get('paragraph_entropy', 0):.4f}\n"
                f"Risk Score: {confab_results.get('risk_score', 0):.4f}\n"
                f"Number of Claims: {confab_results.get('num_claims', 0)}\n"
                f"Web Consistency: {confab_results.get('web_comparison', {}).get('is_consistent', False)}"
            )
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        Main function to process a question with confabulation detection and correction
        
        Returns:
            Dictionary containing original response, confabulation analysis, and corrected response if needed
        """
        logger.info(f"Processing question: {question}")
        
        # Step 1: Generate initial response
        logger.info("Generating initial response...")
        original_response = await self.generate_initial_response(question)
        
        if original_response.startswith("Error:"):
            return {
                "question": question,
                "original_response": original_response,
                "response_type": "error",
                "token_count": 0,
                "confabulation_detected": False,
                "analysis_results": {},
                "corrected_response": None,
                "final_response": original_response
            }
        
        # Step 2: Determine response type
        token_count = self.count_tokens(original_response)
        response_type = self.determine_response_type(original_response)
        
        logger.info(f"Response type: {response_type} ({token_count} tokens)")
        
        # Step 3: Perform confabulation detection based on response type
        logger.info(f"Performing {response_type}-form confabulation detection...")
        
        if response_type == "short":
            confab_results = await self.detect_short_form_confabulation(question)
        else:
            confab_results = await self.detect_short_form_confabulation(original_response)
        
        confabulation_detected = confab_results.get("confabulation_detected", False)
        
        logger.info(f"Confabulation detected: {confabulation_detected}")
        
        # Step 4: Generate corrected response if confabulation detected
        corrected_response = None
        final_response = original_response
        
        if confabulation_detected:
            logger.info("Generating corrected response...")
            web_answer = self.extract_web_answer(confab_results, response_type)
            confab_summary = self.format_confabulation_summary(confab_results, response_type)
            
            corrected_response = await self.generate_corrected_response(
                question, web_answer, confab_summary
            )
            final_response = corrected_response
        
        # Step 5: Compile results
        results = {
            "question": question,
            "original_response": original_response,
            "response_type": response_type,
            "token_count": token_count,
            "confabulation_detected": confabulation_detected,
            "analysis_results": confab_results,
            "corrected_response": corrected_response,
            "final_response": final_response,
            "total_tokens_used": self.token_tracker.get_total_tokens()
        }
        
        logger.info("Processing complete")
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results"""
        print("\n" + "="*80)
        print("CONFABULATION-AWARE RESPONSE GENERATION")
        print("="*80)
        
        print(f"\nQuestion: {results['question']}")
        print(f"Response Type: {results['response_type']} ({results['token_count']} tokens)")
        print(f"Confabulation Detected: {results['confabulation_detected']}")
        
        print(f"\nOriginal Response:")
        print("-" * 40)
        print(f"{results['original_response']}")
        
        if results['confabulation_detected'] and results['corrected_response']:
            print(f"\nCorrected Response:")
            print("-" * 40)
            print(f"{results['corrected_response']}")
            
            print(f"\nConfabulation Analysis:")
            print("-" * 40)
            analysis = results['analysis_results']
            if results['response_type'] == "short":
                print(f"Analysis: {analysis.get('analysis_result', 'Unknown')}")
                print(f"Shannon Entropy: {analysis.get('shannon_entropy', 0):.4f}")
                print(f"Model Uncertain: {analysis.get('is_uncertain', False)}")
            else:
                print(f"Analysis: {analysis.get('analysis_result', 'Unknown')}")
                print(f"Paragraph Entropy: {analysis.get('paragraph_entropy', 0):.4f}")
                print(f"Risk Score: {analysis.get('risk_score', 0):.4f}")
                print(f"Number of Claims: {analysis.get('num_claims', 0)}")
        
        print(f"\nFinal Response:")
        print("-" * 40)
        print(f"{results['final_response']}")
        
        print(f"\nTokens Used: {results['total_tokens_used']}")
        print("="*80)
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to file"""
        if filename is None:
            safe_question = results['question'].replace('?', '').replace(' ', '_')[:30]
            filename = f"detection_results_{safe_question}.txt"
        
        with open(filename, "w", encoding="utf-8") as file:
            file.write("CONFABULATION-AWARE RESPONSE GENERATION RESULTS\n")
            file.write("=" * 60 + "\n\n")
            
            file.write(f"Question: {results['question']}\n")
            file.write(f"Response Type: {results['response_type']}\n")
            file.write(f"Token Count: {results['token_count']}\n")
            file.write(f"Confabulation Detected: {results['confabulation_detected']}\n\n")
            
            file.write("ORIGINAL RESPONSE:\n")
            file.write("-" * 40 + "\n")
            file.write(f"{results['original_response']}\n\n")
            
            if results['corrected_response']:
                file.write("CORRECTED RESPONSE:\n")
                file.write("-" * 40 + "\n")
                file.write(f"{results['corrected_response']}\n\n")
            
            file.write("FINAL RESPONSE:\n")
            file.write("-" * 40 + "\n")
            file.write(f"{results['final_response']}\n\n")
            
            file.write("DETAILED ANALYSIS RESULTS:\n")
            file.write("-" * 40 + "\n")
            file.write(f"{results['analysis_results']}\n\n")
            
            file.write(f"Total Tokens Used: {results['total_tokens_used']}\n")

async def main():
    """Example usage and interactive mode"""
    
    # Initialize the response generator
    generator = ConfabulationAwareResponseGenerator(short_token_threshold=60)
    
    # Test questions
    test_questions = [
        "How many R's are there in strawberry?"
    ]
    
    print("CONFABULATION-AWARE RESPONSE GENERATOR")
    print("="*60)
    print("Choose mode:")
    print("1. Test with predefined questions")
    print("2. Interactive mode")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            # Test mode
            for i, question in enumerate(test_questions):
                print(f"\n\nTEST {i+1}/{len(test_questions)}")
                results = await generator.process_question(question)
                generator.print_results(results)
                
                # Save results
                generator.save_results(results, f"test_result_{i+1}.txt")
        
        elif choice == "2":
            # Interactive mode
            print("\nInteractive Mode - Enter questions (type 'quit' to exit)")
            
            while True:
                question = input("\nEnter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    print("Please enter a valid question.")
                    continue
                
                results = await generator.process_question(question)
                generator.print_results(results)
                
                # Ask if user wants to save results
                save_choice = input("\nSave results to file? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    generator.save_results(results)
                    print("Results saved!")
        
        else:
            print("Invalid choice. Please run again and select 1 or 2.")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())