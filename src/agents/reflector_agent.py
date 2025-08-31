from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


class ReflectorState(TypedDict):
    query: str
    current_response: str
    context: str
    critique: str
    improved_response: str


class ReflectorAgent:
    def __init__(self, model: str):
        self.llm = ChatOllama(
            model=model
        )
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ReflectorState)
        
        workflow.add_node("critique", self._critique)
        workflow.add_node("improve", self._improve)
        
        workflow.add_edge(START, "critique")
        workflow.add_edge("critique", "improve")
        workflow.add_edge("improve", END)
        
        return workflow.compile()

    def _critique(self, state: ReflectorState) -> ReflectorState:
        """
        Analyzes the current response for potential issues and hallucinations
        """
        query = state.get("query", "")
        current_response = state.get("current_response", "")
        context = state.get("context", "")
        
        system_prompt = """You are a critical analysis agent. Your job is to identify issues with responses including:
        1. Hallucinations or unsupported claims
        2. Information not grounded in the provided context
        3. Factual inaccuracies or speculation
        4. Missing important information from the context
        5. Clarity and coherence issues

        Be thorough but concise in your critique."""

        critique_prompt = f"""
        Original Query: {query}

        Current Response: {current_response}

        Available Context: {context}

        Provide a detailed critique identifying:
        - Any claims not supported by the context
        - Missing relevant information from the context
        - Potential hallucinations or inaccuracies
        - Areas for improvement in clarity or completeness"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=critique_prompt)
        ]
        
        response = self.llm.invoke(messages)

        print("critique: ", response.content)

        state["critique"] = response.content.strip()
        
        return state

    def _improve(self, state: ReflectorState) -> ReflectorState:
        """
        Generates an improved response based on the critique
        """
        query = state.get("query", "")
        current_response = state.get("current_response", "")
        context = state.get("context", "")
        critique = state.get("critique", "")
        
        system_prompt = """You are a response improvement agent. Using the provided critique, create a better response that:
        1. Removes hallucinations and unsupported claims
        2. Grounds all statements in the provided context
        3. Addresses the issues identified in the critique
        4. Maintains accuracy and acknowledges limitations
        5. Improves clarity and completeness

        Only make claims that can be directly supported by the context."""

        improvement_prompt = f"""
        Original Query: {query}

        Current Response: {current_response}

        Available Context: {context}

        Critique: {critique}

        Based on the critique, provide an improved response that addresses all identified issues while staying grounded in the available context"""

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
            Improved response with reduced hallucinations
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
    reflector = ReflectorAgent(model="mistral:7b")
    
    # Test with a response containing hallucinations
    query = "What is the capital of France?"
    current_response = "The capital of France is Paris. Paris has a population of 15 million people and was founded in 1889."
    context = "Paris is the capital and most populous city of France. The city has a population of approximately 2.1 million residents."
    
    improved_response = reflector.generate_response(query, current_response, context)
    print(f"Improved Response: {improved_response}")