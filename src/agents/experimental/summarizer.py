from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

class Summarizer:
    def __init__(self, model: str):
        self.llm = ChatOllama(
            model=model,
            temperature=0.1  # Low temperature for conservative, precise summarization
        )

    def generate_response(self, query: str, context: str) -> str:
        """
        Summarizes and refines the context to make it more directly answer the query.
        Takes raw context and makes it more focused and relevant to the specific question.
        """
        
        system_prompt = f"""You are a context refinement specialist. Your job is to reorganize and highlight information from the context that directly addresses the query, while preserving ALL important details, nuances, and supporting information.

CRITICAL RULES:
1. PRESERVE important details, numbers, dates, names, technical specifications, and qualifications
2. RETAIN context that provides nuance, caveats, or important background
3. REORGANIZE information to put the most relevant details first
4. EXPAND on key points if they directly answer the query
5. DO NOT oversimplify complex topics - maintain necessary complexity
6. INCLUDE supporting evidence, sources, or reasoning when present

APPROACH:
- Start with the most direct answer to the query
- Follow with important supporting details and context
- Include relevant background information that aids understanding
- Preserve specific data points, measurements, or technical details
- Maintain important qualifications or limitations

Now refine the following context:

Query: {query}
Original Context: {context}

Refined Context:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Refine the context following the principles above - preserve important details while organizing information to directly address the query.")
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()