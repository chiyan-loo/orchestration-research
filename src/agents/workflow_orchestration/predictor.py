from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
import re

class Predictor:
    def __init__(self, model: str):
        self.llm = ChatOllama(model=model, temperature=0.7)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response to a query with optional context using chain of thought reasoning
        
        Args:
            query: The question or prompt to respond to
            context: Optional context information
            
        Returns:
            Generated response string extracted from XML tags
        """
        system_prompt = f"""You are an analytical agent that uses step-by-step reasoning to answer queries directly and accurately.

Context: {context if context else "No specific context provided"}

REQUIRED RESPONSE FORMAT:
You must structure your response using these XML tags:

<reasoning>
Provide your step-by-step thinking process:
1. QUERY ANALYSIS: Break down what the query is asking
2. CONTEXT EVALUATION: How does the provided context help (if any)
3. KNOWLEDGE APPLICATION: What relevant information do you know
4. LOGICAL STEPS: Walk through your reasoning process
5. VERIFICATION: Double-check your logic and conclusion
</reasoning>

<answer>
Provide your final, concise answer here
</answer>

IMPORTANT: Always include both reasoning and answer sections. The answer should be direct and focused."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        full_response = response.content.strip()
        
        print(f"Full reasoning response: {full_response}")
        
        # Extract answer from XML tags
        extracted_answer = self._extract_answer(full_response)
        
        return extracted_answer
    
    def _extract_answer(self, response: str) -> str:
        """
        Extract the answer from XML tags, with fallback to full response
        """
        # Try to extract answer from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        
        if answer_match:
            answer = answer_match.group(1).strip()
            return answer
        else:
            print("Warning: No <answer> tags found, returning full response")
            return response


# Example usage
if __name__ == "__main__":
    agent = Predictor(model="mistral:7b")
    
    query = "What is the capital of France?"
    context = "France is a country in Western Europe."
    
    result = agent.generate_response(query, context)
    print(f"Query: {query}")
    print(f"Response: {result}")