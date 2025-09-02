from typing import TypedDict, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
import re


class AggregatorState(TypedDict):
    query: str
    context: str
    responses: List[str]
    analysis: str
    final_response: str


class AggregatorAgent:
    def __init__(self, model: str, num_samples: int = 3):
        self.llm = ChatOllama(model=model)
        self.num_samples = num_samples
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AggregatorState)
        
        workflow.add_node("generate", self._generate_responses)
        workflow.add_node("judge", self._judge_consistency)
        workflow.add_node("synthesize", self._synthesize_final)
        
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "judge")
        workflow.add_edge("judge", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _generate_responses(self, state: AggregatorState) -> AggregatorState:
        """Generate multiple responses"""
        query = state["query"]
        context = state["context"]
        
        system_prompt = f"""Answer the query accurately and concisely.
        Context: {context if context else "None"}"""
        
        responses = []
        
        for i in range(self.num_samples):
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            response = self.llm.invoke(messages)
            responses.append(response.content.strip())
            print(f"Generated response {i+1}/{self.num_samples}")
            print(f"Generated response: {response.content}")
        
        state["responses"] = responses
        return state
    
    def _judge_consistency(self, state: AggregatorState) -> AggregatorState:
        """Simple consistency checking - just get judge's analysis"""
        responses = state["responses"]
        query = state["query"]
        
        # Prepare responses for judgment
        numbered_responses = "\n".join([
            f"Response {i+1}: {response}" 
            for i, response in enumerate(responses)
        ])
        
        judge_prompt = f"""Look at these multiple responses to the same query and write a summary of the consistent information. Focus on facts that multiple responses agree on.

Query: {query}

Responses:
{numbered_responses}
"""

        messages = [
            SystemMessage(content="You are analyzing responses to identify consistent information."),
            HumanMessage(content=judge_prompt)
        ]
        
        response = self.llm.invoke(messages)
        judge_analysis = response.content.strip()
        
        print(f"Judge analysis: {judge_analysis}")
        
        state["analysis"] = judge_analysis
            
        return state
    
    def _synthesize_final(self, state: AggregatorState) -> AggregatorState:
        """Synthesize final response from consistent facts"""
        query = state["query"]
        analysis = state["analysis"]
        
        synthesis_prompt = f"""Based on these key facts that appeared consistently across multiple responses, create a clear and accurate answer:

Query: {query}

Key Facts:
{analysis}

Please provide a natural, well-written response that incorporates these facts to directly answer the query."""
        
        messages = [
            SystemMessage(content="You are creating a final answer based on consistent facts from multiple responses."),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["final_response"] = response.content.strip()
        
        return state
    
    def generate_response(self, query: str, context: str = "") -> str:
        """Generate self-consistent response"""
        result = self.graph.invoke({
            "query": query,
            "context": context,
            "responses": [],
            "analysis": "",
            "final_response": ""
        })
        
        return result["final_response"]
    
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            
            with open("aggregator_workflow.png", "wb") as f:
                f.write(png_data)
                        
        except Exception as e:
            print(f"Install graphviz: pip install graphviz")


if __name__ == "__main__":
    aggregator = AggregatorAgent(model="mistral:7b")
        
    query = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France. The city has a population of approximately 2.1 million residents."
    
    result = aggregator.generate_response(query, context)
    
    print(f"\nFinal Response:\n{result}")