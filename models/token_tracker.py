from dataclasses import dataclass, field
from typing import List, Dict, Any
from tiktoken import encoding_for_model
import logging

from config.settings import MODEL_PRICING

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Represents token usage for a single API call."""
    input_tokens: int
    output_tokens: int
    model: str
    call_type: str  # "embedding", "chat_completion", "gemini_generation"
    cost: float = 0.0

@dataclass
class TokenTracker:
    """Tracks token usage across all API calls."""
    openai_calls: List[TokenUsage] = field(default_factory=list)
    gemini_calls: List[TokenUsage] = field(default_factory=list)
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count tokens in text using tiktoken."""
        try:
            enc = encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            # Fallback estimation: ~4 chars per token
            return len(text) // 4
    
    def count_gemini_tokens(self, text: str) -> int:
        """Estimate tokens for Gemini (rough approximation)."""
        return len(text) // 4
    
    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost for a token usage."""
        if usage.model not in MODEL_PRICING:
            return 0.0
        
        pricing = MODEL_PRICING[usage.model]
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def track_openai_embedding(self, input_text: str, model: str) -> TokenUsage:
        """Track OpenAI embedding API call."""
        input_tokens = self.count_tokens(input_text, "gpt-4o")
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=0,
            model=model,
            call_type="embedding"
        )
        usage.cost = self.calculate_cost(usage)
        self.openai_calls.append(usage)
        return usage
    
    def track_openai_chat(self, input_text: str, output_text: str, model: str) -> TokenUsage:
        """Track OpenAI chat completion API call."""
        input_tokens = self.count_tokens(input_text, "gpt-4o")
        output_tokens = self.count_tokens(output_text, "gpt-4o")
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            call_type="chat_completion"
        )
        usage.cost = self.calculate_cost(usage)
        self.openai_calls.append(usage)
        return usage
    
    def track_gemini_generation(self, input_text: str, output_text: str, model: str) -> TokenUsage:
        """Track Gemini generation API call."""
        input_tokens = self.count_gemini_tokens(input_text)
        output_tokens = self.count_gemini_tokens(output_text)
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            call_type="gemini_generation"
        )
        usage.cost = self.calculate_cost(usage)
        self.gemini_calls.append(usage)
        return usage
    
    def get_total_tokens(self) -> Dict[str, int]:
        """Get total input and output tokens across all calls."""
        total_input = sum(call.input_tokens for call in self.openai_calls + self.gemini_calls)
        total_output = sum(call.output_tokens for call in self.openai_calls + self.gemini_calls)
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output
        }
    
    def get_total_cost(self) -> float:
        """Get total cost across all API calls."""
        return sum(call.cost for call in self.openai_calls + self.gemini_calls)
    
    def get_detailed_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of token usage and costs."""
        totals = self.get_total_tokens()
        
        # Breakdown by provider
        openai_input = sum(call.input_tokens for call in self.openai_calls)
        openai_output = sum(call.output_tokens for call in self.openai_calls)
        openai_cost = sum(call.cost for call in self.openai_calls)
        
        gemini_input = sum(call.input_tokens for call in self.gemini_calls)
        gemini_output = sum(call.output_tokens for call in self.gemini_calls)
        gemini_cost = sum(call.cost for call in self.gemini_calls)
        
        # Breakdown by call type
        embedding_calls = [call for call in self.openai_calls if call.call_type == "embedding"]
        chat_calls = [call for call in self.openai_calls if call.call_type == "chat_completion"]
        gemini_generation_calls = [call for call in self.gemini_calls if call.call_type == "gemini_generation"]
        
        return {
            "totals": totals,
            "total_cost": self.get_total_cost(),
            "by_provider": {
                "openai": {
                    "input_tokens": openai_input,
                    "output_tokens": openai_output,
                    "total_tokens": openai_input + openai_output,
                    "cost": openai_cost,
                    "call_count": len(self.openai_calls)
                },
                "gemini": {
                    "input_tokens": gemini_input,
                    "output_tokens": gemini_output,
                    "total_tokens": gemini_input + gemini_output,
                    "cost": gemini_cost,
                    "call_count": len(self.gemini_calls)
                }
            },
            "by_call_type": {
                "embedding": {
                    "call_count": len(embedding_calls),
                    "input_tokens": sum(call.input_tokens for call in embedding_calls),
                    "cost": sum(call.cost for call in embedding_calls)
                },
                "chat_completion": {
                    "call_count": len(chat_calls),
                    "input_tokens": sum(call.input_tokens for call in chat_calls),
                    "output_tokens": sum(call.output_tokens for call in chat_calls),
                    "cost": sum(call.cost for call in chat_calls)
                },
                "gemini_generation": {
                    "call_count": len(gemini_generation_calls),
                    "input_tokens": sum(call.input_tokens for call in gemini_generation_calls),
                    "output_tokens": sum(call.output_tokens for call in gemini_generation_calls),
                    "cost": sum(call.cost for call in gemini_generation_calls)
                }
            }
        }
    
    def print_summary(self):
        """Print a formatted summary of token usage and costs."""
        breakdown = self.get_detailed_breakdown()
        
        print("\n" + "="*60)
        print("TOKEN USAGE AND COST SUMMARY")
        print("="*60)
        
        print(f"Total Input Tokens: {breakdown['totals']['total_input_tokens']:,}")
        print(f"Total Output Tokens: {breakdown['totals']['total_output_tokens']:,}")
        print(f"Total Tokens: {breakdown['totals']['total_tokens']:,}")
        print(f"Total Cost: ${breakdown['total_cost']:.4f}")
        
        print("\nBy Provider:")
        print(f"  OpenAI: {breakdown['by_provider']['openai']['call_count']} calls, "
              f"{breakdown['by_provider']['openai']['total_tokens']:,} tokens, "
              f"${breakdown['by_provider']['openai']['cost']:.4f}")
        print(f"  Gemini: {breakdown['by_provider']['gemini']['call_count']} calls, "
              f"{breakdown['by_provider']['gemini']['total_tokens']:,} tokens, "
              f"${breakdown['by_provider']['gemini']['cost']:.4f}")
        
        print("\nBy Call Type:")
        for call_type, data in breakdown['by_call_type'].items():
            if data['call_count'] > 0:
                print(f"  {call_type.replace('_', ' ').title()}: {data['call_count']} calls, "
                      f"${data['cost']:.4f}")
        
        print("="*60)