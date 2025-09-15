from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


class RefinerState(TypedDict):
    query: str
    current_response: str
    context: str
    critique: str
    improved_response: str


class Refiner:
    def __init__(self, model: str):
        self.llm = ChatOllama(
            model=model,
            temperature=0.4,  # Lower temperature for more focused, accurate responses
        )
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RefinerState)
        
        workflow.add_node("critique", self._critique)
        workflow.add_node("improve", self._improve)
        
        workflow.add_edge(START, "critique")
        workflow.add_edge("critique", "improve")
        workflow.add_edge("improve", END)
        
        return workflow.compile()

    def _critique(self, state: RefinerState) -> RefinerState:
        """
        Analyzes the current response focusing heavily on accuracy and conciseness
        """
        query = state.get("query", "")
        current_response = state.get("current_response", "")
        context = state.get("context", "")
        
        system_prompt = """You are a precision-focused critic. Your PRIMARY goals are:
1. ELIMINATE HALLUCINATIONS: Flag any claim not directly supported by context
2. MAXIMIZE CONCISENESS: Identify unnecessary words, redundancy, and verbosity
3. ENSURE ACCURACY: Verify every factual claim against the context
4. FOCUS ON RELEVANCE: Remove information not directly answering the query

Be ruthless in identifying what should be removed or edited."""

        critique_prompt = f"""
Query: {query}

Current Response: {current_response}

Context: {context if context else "NO CONTEXT PROVIDED - All claims must be general knowledge only"}

CRITICAL ANALYSIS REQUIRED:

HALLUCINATION CHECK:
- List every factual claim in the response
- Mark which claims are NOT supported by the context
- Identify any speculation or assumptions

CONCISENESS AUDIT:
- Identify redundant phrases or repetitive information  
- Flag unnecessarily complex language that could be simplified
- Find sentences that add no value

ACCURACY VERIFICATION:
- Check every number, date, or specific fact against the context
- Identify any contradictions with the provided context
- Flag vague or imprecise language that should be more specific

RELEVANCE FILTER:
- Mark information that doesn't directly answer the query
- Identify tangential details that should be removed

FINAL DIRECTIVE: Focus on what to CUT and SIMPLIFY, not what to add."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=critique_prompt)
        ]
        
        response = self.llm.invoke(messages)

        state["critique"] = response.content.strip()
        
        return state

    def _improve(self, state: RefinerState) -> RefinerState:
        """
        Creates a more concise and accurate response by removing/fixing issues
        """
        query = state.get("query", "")
        current_response = state.get("current_response", "")
        context = state.get("context", "")
        critique = state.get("critique", "")
        
        system_prompt = """You are a precision editor focused on accuracy.

CORE PRINCIPLES:
- SHORTER IS BETTER: Cut unnecessary words ruthlessly
- ONLY VERIFIABLE FACTS: If it's not in the context, don't claim it
- DIRECT ANSWERS: Get straight to the point
- NO SPECULATION: Remove uncertain or assumptive language
- SIMPLE LANGUAGE: Use the clearest, most direct phrasing

DO NOT:
- Add new information not in the original response
- Include background information unless directly relevant
- Use hedging language like "it seems" or "possibly"

DO:
- Remove unsupported claims entirely
- Shorten wordy explanations
- Use precise, specific language
- Answer the query directly and stop
- Acknowledge limitations clearly when context is insufficient"""

        improvement_prompt = f"""
Query: {query}

Original Response: {current_response}

Context: {context if context else "NO CONTEXT - Use only general knowledge"}

Critique: {critique}

Create a more accurate response by:
1. Removing all unsupported claims identified in the critique
2. Cutting unnecessary words and redundancy
3. Using only information that directly answers the query
4. Making language more precise and direct

TARGET: The improved response should be more concise while being more accurate."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=improvement_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["improved_response"] = response.content.strip()
        
        return state

    def generate_response(self, query: str, current_response: str, context: str) -> str:
        """
        Main method to generate an improved response through critique and improvement
        
        Args:
            query: The original user query
            current_response: The response to be improved
            context: Available context from retrieval or previous agents
            
        Returns:
            Improved response with reduced hallucinations and increased conciseness
        """
        result = self.graph.invoke({
            "query": query,
            "current_response": current_response,
            "context": context,
            "critique": "",
            "improved_response": ""
        })
        
        return result["improved_response"]
    
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            
            with open("reflector_workflow.png", "wb") as f:
                f.write(png_data)
                        
        except Exception as e:
            print(f"Install graphviz: pip install graphviz")


if __name__ == "__main__":
    reflector = Refiner(model="mistral:7b")
    
    # Test with a verbose response containing hallucinations
    query = "What is the capital of France?"
    current_response = """The capital of France is Paris, which is a beautiful and historic city located in the northern part of the country. Paris has a population of approximately 15 million people in the greater metropolitan area and was founded in 1889 as a major European capital. The city is known for its many attractions including the Eiffel Tower, which was built for the World's Fair, and numerous museums and cultural sites that attract millions of visitors each year."""
    context = "Paris is the capital and most populous city of France. The city has a population of approximately 2.1 million residents."
    
    improved_response = reflector.generate_response(query, current_response, context)
    print(f"Original Response Length: {len(current_response)} characters")
    print(f"Improved Response: {improved_response}")
    print(f"Improved Response Length: {len(improved_response)} characters")