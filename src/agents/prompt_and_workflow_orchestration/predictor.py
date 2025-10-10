from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Dict, Optional
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.exceptions import OutputParserException


class ResponseSchema(BaseModel):
    """Pydantic schema for structured response with reasoning and answer"""
    reasoning: str = Field(description="Required step-by-step thinking process")
    answer: str = Field(description="Final, concise and direct answer to the query.")


class Predictor:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm.with_structured_output(ResponseSchema)
        self.llm_unstructured = llm  # Keep unstructured version as fallback
    
    def generate_response(self, query: str, context: str, system_prompt: str, callback: UsageMetadataCallbackHandler) -> Dict:
        """
        Generate a response to a query with optional context using chain of thought reasoning
        
        Args:
            query: The question or prompt to respond to
            context: Optional context information
            system_prompt: Custom system prompt for the predictor
            callback: Callback handler for usage metadata
            
        Returns:
            Dict containing:
                - content: The final answer
                - structured: Boolean indicating if structured output succeeded
        """
        full_system_prompt = f"""{system_prompt}

Context: {context if context else "No specific context provided"}

Provide a single, concise final answer, no explanations. There is always sufficient information."""
        
        messages = [
            SystemMessage(content=full_system_prompt),
            HumanMessage(content=query)
        ]
        
        try:
            # Try structured output first
            response = self.llm.invoke(messages, config={"callbacks": [callback]})
            print(f"response: {response.answer}")
            
            return {
                "content": response.answer,
                "structured": True
            }
        
        except (OutputParserException, ValueError, AttributeError) as e:
            print(f"Structured output failed: {e}. Falling back to unstructured output.")
            
            # Fallback to unstructured output
            response = self.llm_unstructured.invoke(messages, config={"callbacks": [callback]})
            
            # Extract content from unstructured response
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "content": content,
                "structured": False
            }


# Example usage
if __name__ == "__main__":
    llm = ChatOllama(model="mistral:7b", temperature=0.3)

    callback = UsageMetadataCallbackHandler()

    agent = Predictor(llm=llm)
    
    query = "What is the capital of France?"
    context = "France is a country in Western Europe."
    
    result = agent.generate_response(query, context, callback=callback, system_prompt="""You are an analytical agent that uses step-by-step reasoning to answer queries directly and accurately.
Provide your step-by-step thinking process:
1. QUERY ANALYSIS: Break down what the query is asking
2. CONTEXT EVALUATION: How does the provided context help (if any)
3. KNOWLEDGE APPLICATION: What relevant information do you know
4. LOGICAL STEPS: Walk through your reasoning process
5. VERIFICATION: Double-check your logic and conclusion
""")
    print(f"Query: {query}")
    print(f"Response content: {result['content']}")
    print(f"Input tokens: {callback}")