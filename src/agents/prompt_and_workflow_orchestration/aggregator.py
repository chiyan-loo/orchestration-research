from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.language_models.base import BaseLanguageModel


class Aggregator:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
    
    def aggregate_messages(self, messages: List[str], query: str) -> str:
        """
        Aggregate multiple messages and generate a singular response based on consistent information
        
        Args:
            messages: List of message strings to analyze
            query: Optional specific query to answer based on the messages
        
        Returns:
            A single aggregated response
        """
        if not messages:
            return "No messages provided for aggregation."
        
        # Prepare messages for analysis
        numbered_messages = "\n".join([
            f"Message {i+1}: {message}" 
            for i, message in enumerate(messages)
        ])
        
        # Create aggregation prompt
        aggregation_prompt = f"""Analyze these multiple messages and only return the final answer, no explanations. Provide a single, short, concise final answer that best reflects the consensus or most accurate result.

Query: {query}

Messages to analyze:
{numbered_messages}

Instructions:
1. Identify information that appears consistently across multiple messages
2. Focus on facts and key points that are mentioned or supported by more than one message
3. Ignore contradictory or outlier information that only appears in one message
4. Synthesize the consistent information into a clear, direct answer to the query
5. If there is an equal number of contradictory opinions, find the most accurate, well-thought response

Provide your aggregated response:"""

        
        # Generate aggregated response
        messages_for_llm = [
            SystemMessage(content="You are an expert at analyzing multiple pieces of information and identifying consistent patterns to create accurate, synthesized responses."),
            HumanMessage(content=aggregation_prompt)
        ]
        
        response = self.llm.invoke(messages_for_llm)
        return response.content.strip()


# Example usage
if __name__ == "__main__":
    llm = ChatOllama(model="mistral:7b", temperature=0.3)

    aggregator = Aggregator(llm=llm)
    
    sample_messages = [
        "Paris is the capital of France and has about 2.1 million people.",
        "The capital city of France is Paris, known for the Eiffel Tower.",
        "Paris, France's capital, is home to roughly 2 million residents and many famous landmarks.",
        "Berlin is the capital of Germany, not France. Paris is France's capital."
    ]
    
    query = "What is the population of Paris?"
    result = aggregator.aggregate_messages(sample_messages, query)
    print(f"Result: {result}")