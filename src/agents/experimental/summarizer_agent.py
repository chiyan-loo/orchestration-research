from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

class SummarizerAgent:
    def __init__(self, model: str):
        self.llm = ChatOllama(model=model)

    def generate_response(self, query: str, context: str) -> str:
        """
        Summarizes and refines the context to make it more directly answer the query.
        Takes raw context and makes it more focused and relevant to the specific question.
        """
        
        system_prompt = f"""You are a summarizer agent that refines context to better answer specific queries. Avoid removing information important to the query.

EXAMPLES:

Example 1:
Query: "What is the capital of France?"  
Original Context: "France is a country in Western Europe. It has a rich history dating back centuries. Paris, the capital city, is known for the Eiffel Tower and Louvre Museum. The country has a population of about 67 million people. French cuisine is famous worldwide."
Refined Context: "Paris is the capital city of France."

Example 2:
Query: "When was the iPhone first released?"
Original Context: "Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company has released many products over the years. The iPhone was first introduced by Steve Jobs at the Macworld Conference & Expo on January 9, 2007, and was released to the public on June 29, 2007. It revolutionized the smartphone industry."
Refined Context: "The iPhone was first introduced on January 9, 2007, and released to the public on June 29, 2007."

Now refine the following:

Query: {query}
Original Context: {context}

Refined Context:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Refine the context to better address the query following the pattern shown in the examples.")
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()