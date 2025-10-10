from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.exceptions import OutputParserException

class ResponseSchema(BaseModel):
    """Pydantic schema for structured response with reasoning and answer"""
    reasoning: str = Field(description="Step-by-step thinking process")
    answer: str = Field(description="Final, concise and direct answer to the query.")

class Summarizer:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm.with_structured_output(ResponseSchema)
        self.llm_unstructured = llm  # Keep unstructured version as fallback

    def generate_response(self, query: str, context: str, system_prompt: str, callback: UsageMetadataCallbackHandler) -> dict:
        """
        Summarizes and refines the context to make it more directly answer the query.
        Takes raw context and makes it more focused and relevant to the specific question.
        """
        
        human_prompt = f"""Refine the following context:

Query: {query}
Original Context: {context}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            # Try structured output first
            response = self.llm.invoke(messages, config={"callbacks": [callback]})
            
            return {
                "content": response.answer.strip(),
                "reasoning": response.reasoning,
                "structured": True
            }
        
        except (OutputParserException, ValueError, AttributeError) as e:
            print(f"Structured output failed: {e}. Falling back to unstructured output.")
            
            # Fallback to unstructured output
            response = self.llm_unstructured.invoke(messages, config={"callbacks": [callback]})
            
            # Extract content from unstructured response
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "content": content.strip(),
                "reasoning": None,
                "structured": False
            }