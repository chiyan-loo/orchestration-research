from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class ChainOfThoughtAgent:
    """
    Ultra-simple chain of thought agent with direct LLM invoke
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    def generate_response(self, question: str, context: str = "") -> str:
        """
        Generate a chain of thought response to the given question
        
        Args:
            question: The problem or question to solve
            context: Optional context information to help with reasoning
            
        Returns:
            String containing the step-by-step reasoning and answer
        """

        system_prompt = """You are a helpful assistant that solves problems using step-by-step chain of thought reasoning.

Always break down your thinking into clear steps:
1. Understand what the problem is asking
2. Identify the key information given (including any context)
3. Work through the solution logically step by step
4. State your final answer clearly within "<answer>...</answer>"

Think through each step carefully and show your work."""

        user_message = f"""
{context}

Problem: {question}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse the response to extract only the answer
        full_response = response.content
        
        # Find answer tags and extract content
        start_tag = "<answer>"
        end_tag = "</answer>"
        
        if start_tag in full_response and end_tag in full_response:
            start_idx = full_response.find(start_tag) + len(start_tag)
            end_idx = full_response.find(end_tag)
            return full_response[start_idx:end_idx].strip()
        else:
            # Fallback if tags not found
            return full_response


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = ChainOfThoughtAgent()
    
    # Test with context
    context = """
    Sarah works at a bakery that sells cupcakes for $3 each and cookies for $2 each.
    The bakery gives a 10% discount for orders over $20.
    Sarah gets a 20% employee discount on top of any other discounts.
    """
    
    question = "If Sarah buys 5 cupcakes and 8 cookies, how much will she pay?"
    
    print("Context:", context)
    print(f"Question: {question}\n")
    
    response = agent.generate_response(question, context)
    print("Agent Response:")
    print(response)