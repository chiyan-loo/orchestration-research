from typing import TypedDict, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.exceptions import OutputParserException

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
    callback: UsageMetadataCallbackHandler
    structured: bool  # Track if structured output succeeded

class Refiner:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        
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
        callback = state.get("callback")

        critique_prompt = f"""Thoroughly critique the current response given the query and context. Provide a thorough list potential flaws of the current response and make suggestions. Value shortness and concisenesss as a criteria for the original response, but make the critique itself thorough and extensive. The original response should have no extra context, no explanations, no clarifications.
        Be ruthless in your critique.
Query: {query}

Current Response: {current_response}

Context: {context}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=critique_prompt)
        ]
        
        # Invoke with callback if provided
        config = {"callbacks": [callback]} if callback else {}
        response = self.llm.invoke(messages, config=config)

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
        callback = state.get("callback")
        
        improvement_prompt = f"""
Query: {query}

Original Response: {current_response}

Context: {context}

Critique: {critique}

Create a more accurate single, short, concise final answer by addressing the critiques of the original response. Only return the final answer, no explanations, no clarifications."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=improvement_prompt)
        ]
        
        # Invoke with callback if provided
        config = {"callbacks": [callback]} if callback else {}
        
        try:
            # Try structured output first
            structured_llm = self.llm.with_structured_output(EditorResponseSchema)
            response = structured_llm.invoke(messages, config=config)
            
            print(f"Structured response: {response}")
            state["improved_response"] = response.answer
            state["structured"] = True
            
        except (OutputParserException, ValueError, AttributeError) as e:
            print(f"Structured output failed in _improve: {e}. Falling back to unstructured output.")
            
            # Fallback to unstructured output
            response = self.llm.invoke(messages, config=config)
            
            # Extract content from unstructured response
            content = response.content if hasattr(response, 'content') else str(response)
            
            state["improved_response"] = content
            state["structured"] = False
        
        return state

    def generate_response(self, query: str, current_response: str, context: str, 
                         critic_system_prompt: str, editor_system_prompt: str,
                         callback: UsageMetadataCallbackHandler = None) -> Dict:
        """
        Main method to generate an improved response through critique and improvement
        
        Args:
            query: The question being asked
            current_response: The response to be improved
            context: Additional context information
            critic_system_prompt: Custom system prompt for critic sub-agent
            editor_system_prompt: Custom system prompt for editor sub-agent
            callback: Optional callback handler for tracking token usage
        
        Returns:
            Dict containing:
                - content: The improved response
                - structured: Whether structured output succeeded
        """
        result = self.graph.invoke({
            "query": query,
            "current_response": current_response,
            "context": context,
            "critique": "",
            "improved_response": "",
            "critic_system_prompt": critic_system_prompt,
            "editor_system_prompt": editor_system_prompt,
            "callback": callback,  # Pass callback in state
            "structured": True,  # Initialize as True
        })
        
        return {
            "content": result["improved_response"],
            "structured": result.get("structured", False),
        }
    
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        png_data = self.graph.get_graph().draw_mermaid_png()
        
        with open("refiner_workflow.png", "wb") as f:
            f.write(png_data)


if __name__ == "__main__":
    llm = ChatOllama(model="mistral:7b", temperature=0.3)
    callback = UsageMetadataCallbackHandler()

    refiner = Refiner(llm=llm)
    
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
    
    result = refiner.generate_response(
        query, current_response, context, 
        critic_system_prompt=custom_critic,
        editor_system_prompt=custom_editor,
        callback=callback
    )
    print(f"Improved Response: {result['content']}")
    print(f"Structured output succeeded: {result['structured']}")
    print(f"Token Usage: {callback.usage_metadata}")