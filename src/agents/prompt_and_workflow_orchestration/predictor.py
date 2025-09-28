from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
import re
from pydantic import BaseModel, Field
import json
from typing import Optional
from langchain_core.language_models.base import BaseLanguageModel


class ResponseSchema(BaseModel):
    """Pydantic schema for structured response with reasoning and answer"""
    reasoning: str = Field(description="Step-by-step thinking process")
    answer: str = Field(description="Final, concise and direct answer to the query.")

class Predictor:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm.with_structured_output(ResponseSchema)
    
    def generate_response(self, query: str, context: str, system_prompt: str) -> str:
        """
        Generate a response to a query with optional context using chain of thought reasoning
        
        Args:
            query: The question or prompt to respond to
            context: Optional context information
            
        Returns:
            Generated response string extracted from XML tags
        """
        system_prompt = f"""{system_prompt}

Context: {context if context else "No specific context provided"}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        
        print(f"Full reasoning response: {response}")
        
        return response.answer

# Example usage
if __name__ == "__main__":
    llm = ChatOllama(model="mistral:7b", temperature=0.3)

    agent = Predictor(llm=llm)
    
    query = "What is the capital of France?"
    context = "France is a country in Western Europe."
    
    result = agent.generate_response(query, context, system_prompt="""You are an analytical agent that uses step-by-step reasoning to answer queries directly and accurately.
Provide your step-by-step thinking process:
1. QUERY ANALYSIS: Break down what the query is asking
2. CONTEXT EVALUATION: How does the provided context help (if any)
3. KNOWLEDGE APPLICATION: What relevant information do you know
4. LOGICAL STEPS: Walk through your reasoning process
5. VERIFICATION: Double-check your logic and conclusion
""")
    print(f"Query: {query}")
    print(f"Response: {result}")