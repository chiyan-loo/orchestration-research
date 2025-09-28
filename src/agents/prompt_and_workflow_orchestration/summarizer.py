from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.language_models.base import BaseLanguageModel

class ResponseSchema(BaseModel):
    """Pydantic schema for structured response with reasoning and answer"""
    reasoning: str = Field(description="Step-by-step thinking process")
    answer: str = Field(description="Final, concise and direct answer to the query.")

class Summarizer:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm.with_structured_output(ResponseSchema)

    def generate_response(self, query: str, context: str, system_prompt: str) -> str:
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
        
        response = self.llm.invoke(messages)
        return response.answer.strip()