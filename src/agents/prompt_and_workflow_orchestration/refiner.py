from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

class EditorResponseSchema(BaseModel):
    """Pydantic schema for structured response with reasoning and answer"""
    reasoning: str = Field(description="Step-by-step thinking process")
    answer: str = Field(description="Final concise and direct answer to the query")

class RefinerState(TypedDict):
    query: str
    current_response: str
    context: str
    critique: str
    improved_response: str
    critic_system_prompt: str
    editor_system_prompt: str

class Refiner:
    def __init__(self, model: str):
        self.llm = ChatOllama(
            model=model,
            temperature=0.4,
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
        """Analyzes the current response focusing heavily on accuracy and conciseness"""
        query = state["query"]
        current_response = state["current_response"]
        context = state["context"]
        system_prompt = state["critic_system_prompt"]

        critique_prompt = f"""Thoroughly critique the current response given the query and context. Provide a thorough list potential flaws of the current response and make suggestions. Be ruthless in your critique. 
Query: {query}

Current Response: {current_response}

Context: {context}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=critique_prompt)
        ]
        response = self.llm.invoke(messages)

        print(response.content)

        state["critique"] = response.content
        
        return state

    def _improve(self, state: RefinerState) -> RefinerState:
        """Creates a more concise and accurate response by removing/fixing issues"""
        query = state["query"]
        current_response = state["current_response"]
        context = state["context"]
        critique = state["critique"]
        system_prompt = state["editor_system_prompt"]
        
        improvement_prompt = f"""
Query: {query}

Original Response: {current_response}

Context: {context}

Critique: {critique}

Create a more accurate and concise response by addressing the critiques of the original response."""

        structured_llm = self.llm.with_structured_output(EditorResponseSchema)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=improvement_prompt)
        ]
        response = structured_llm.invoke(messages)

        print(response)
        state["improved_response"] = response.answer
        
        return state

    def generate_response(self, query: str, current_response: str, context: str, 
                         critic_system_prompt: str, editor_system_prompt: str) -> str:
        """
        Main method to generate an improved response through critique and improvement
        """
        result = self.graph.invoke({
            "query": query,
            "current_response": current_response,
            "context": context,
            "critique": "",
            "improved_response": "",
            "critic_system_prompt": critic_system_prompt,
            "editor_system_prompt": editor_system_prompt
        })
        
        return result["improved_response"]
    
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        png_data = self.graph.get_graph().draw_mermaid_png()
        
        with open("refiner_workflow.png", "wb") as f:
            f.write(png_data)


if __name__ == "__main__":
    refiner = Refiner(model="mistral:7b")
    
    query = "What is the capital of France?"
    current_response = """The capital of the United Kingdom is London"""
    context = "Paris is the capital and most populous city of France. The city has a population of approximately 2.1 million residents."
    
    custom_critic = """You are a geographic fact-checker. Think through each claim step by step:
    1. Identify all geographic and demographic claims
    2. Check each claim against the provided context
    3. Reason through what information is accurate vs inaccurate
    4. Provide specific recommendations for corrections
    
    Show your reasoning process clearly before giving your final critique."""
    
    custom_editor = """You are a geography textbook editor. Use step-by-step reasoning:
    1. Review the critique carefully
    2. Think through each correction needed
    3. Consider what makes a good geographic answer
    4. Create a precise, factual response
    
    Show your editing reasoning before providing the final improved response."""
    
    improved_response = refiner.generate_response(
        query, current_response, context, 
        critic_system_prompt=custom_critic,
        editor_system_prompt=custom_editor
    )
    print(f"Improved Response: {improved_response}")
