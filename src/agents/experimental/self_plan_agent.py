from typing import TypedDict, List, Annotated, Literal, Union, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from reflector import Reflector
from aggregator import Aggregator
from debator import Debator
from predictor import Predictor
from summarizer import Summarizer

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    context: str
    workflow_plan: List[str]
    current_step: int
    next_agent: Literal["aggregator", "debator", "reflector", "summarizer", "predictor", "end"]

class SelfPlanAgent():
    def __init__(self, model: str):
        self.llm = ChatOllama(
            model=model
        )

        self.predictor = Predictor(model=model)
        self.reflector = Reflector(model=model)
        self.aggregator = Aggregator(model=model)
        self.debator = Debator(model=model)
        self.summarizer = Summarizer(model=model)
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self._plan_workflow)
        workflow.add_node("executor", self._execute_workflow)
        workflow.add_node("predict", self._predict)
        workflow.add_node("aggregate", self._aggregate)
        workflow.add_node("debate", self._debate)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("summarize", self._summarize)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        
        workflow.add_conditional_edges(
            "executor", lambda state: state["next_agent"], 
            {
                "end": END,
                "predictor": "predict",
                "aggregator": "aggregate",
                "debator": "debate",
                "reflector": "reflect",
                "summarizer": "summarize",
            }
        )

        workflow.add_edge("predict", "executor")
        workflow.add_edge("aggregate", "executor")
        workflow.add_edge("debate", "executor")
        workflow.add_edge("reflect", "executor")
        workflow.add_edge("summarize", "executor")

        return workflow.compile()

    def _plan_workflow(self, state: AgentState) -> AgentState:
        """
        Plans the complete workflow based on query and context
        """
        query = state.get("query", "")
        context = state.get("context", "")
        
        system_prompt = f"""You are a workflow planner for a multi-agent system. Based on the query and context, create an optimal sequence of agents.

QUERY: {query}
CONTEXT: {context if context else "No context provided"}

AVAILABLE AGENTS:
- predictor: Simple base agent that directly responds to queries (fastest, basic responses)
- summarizer: Summarizes context to make it more clearly answer the query
- aggregator: Analyzes multiple messages and synthesizes consistent information from them
- debator: Generates responses from multiple perspectives for complex/controversial topics
- reflector: Reviews and improves existing responses to reduce hallucination

WORKFLOW PLANNING RULES:
1. Start with 'summarizer' if context needs refinement before other processing
2. Use 'predictor' to respond to simple, straightforward questions that don't need special processing
3. Use 'debator' to respond to complex/controversial questions requiring multiple perspectives
4. Use 'aggregator' when you need to find consensus from multiple responses in the workflow
5. Use 'reflector' to improve quality of generated responses
6. Adapt workflows to the query. For complex queries, make complex workflows. For simple queries, make simple workflows.

Analyze the query complexity and determine the optimal workflow.

FORMAT: Provide only the agent names, one per line, no explanations."""

        planner_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create an optimal workflow plan for this query.")
        ]
        
        response = self.llm.invoke(planner_messages)
        workflow_steps = [step.strip() for step in response.content.strip().split('\n') if step.strip()]
        
        # Filter valid agents
        valid_agents = ["predictor", "summarizer", "aggregator", "debator", "reflector"]
        workflow_plan = [step for step in workflow_steps if step in valid_agents]
        
        print(f"Planned workflow: {workflow_plan}")
        
        state["workflow_plan"] = workflow_plan
        state["current_step"] = 0
        
        return state

    def _execute_workflow(self, state: AgentState) -> AgentState:
        """
        Executes the next step in the planned workflow
        """
        workflow_plan = state.get("workflow_plan", [])
        current_step = state.get("current_step", 0)
        
        if current_step >= len(workflow_plan):
            state["next_agent"] = "end"
            return state
        
        next_agent = workflow_plan[current_step]
        state["next_agent"] = next_agent
        state["current_step"] = current_step + 1
        
        print(f"Executing step {current_step + 1}/{len(workflow_plan)}: {next_agent}")
        
        return state

    def _predict(self, state: AgentState) -> AgentState:
        """
        Simple predictor agent that directly responds to queries
        """
        print("Predictor executing")
        
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        response = self.predictor.respond(query=query, context=context)
        
        print(f"Predictor generated response: {response[:100]}...")
        
        messages.append(AIMessage(content=response))
        state["messages"] = messages
        
        return state

    def _summarize(self, state: AgentState) -> AgentState:
        """
        Summarizer agent that refines context to better answer the query
        """
        print("Summarizer executing")
        
        query = state.get("query", "")
        context = state.get("context", "")
        
        refined_context = self.summarizer.generate_response(
            query=query,
            context=context
        )

        print(f"Refined context: {refined_context[:100]}...")
        state["context"] = refined_context
        
        return state

    def _aggregate(self, state: AgentState) -> AgentState:
        """
        Aggregator agent that synthesizes consistent information from multiple messages
        """
        print("Aggregator executing")
        
        query = state.get("query", "")
        messages = state.get("messages", [])
        
        # Extract content from AI messages to use as input messages for aggregator
        message_contents = [msg.content for msg in messages if isinstance(msg, AIMessage)]
        
        # If no AI messages yet, use the query itself
        if not message_contents:
            message_contents = [query]
        
        aggregated_response = self.aggregator.aggregate_messages(
            messages=message_contents,
            query=query
        )
        
        print(f"Aggregator generated response: {aggregated_response[:100]}...")
        
        messages.append(AIMessage(content=aggregated_response))
        state["messages"] = messages
        
        return state
    
    def _debate(self, state: AgentState) -> AgentState:
        """
        Debator agent that generates multiple perspectives and conducts structured debate
        """
        print("Debator executing")
        
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        debate_response = self.debator.generate_response(
            query=query,
            context=context
        )
        
        print(f"Debator generated response: {debate_response[:100]}...")
        
        messages.append(AIMessage(content=debate_response))
        state["messages"] = messages
        
        return state
    
    def _reflect(self, state: AgentState) -> AgentState:
        """
        Reflection agent that criticizes the current response and improves it
        """
        print("Reflector executing")
        
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        # Get the current response (last AI message)
        current_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                current_response = msg.content
                break
        
        improved_response = self.reflector.generate_response(
            query=query,
            current_response=current_response,
            context=context
        )
        
        print(f"Reflector generated improved response: {improved_response[:100]}...")
        
        messages.append(AIMessage(content=improved_response))
        state["messages"] = messages
        
        return state

    def generate_response(self, query: str, context: str):
        initial_state = {
            "query": query,
            "context": context,
            "messages": [],
            "workflow_plan": [],
            "current_step": 0,
            "next_agent": None,
        }
        
        print(f"Starting workflow-based orchestration for query: {query}")
        response = self.graph.invoke(initial_state)

        return {
            "content": response["messages"][-1].content if response["messages"] else "No response generated",
            "workflow_plan": response["workflow_plan"],
            "final_context": response["context"]
        }

    def save_workflow_image(self):
        """Save workflow as PNG image"""
        png_data = self.graph.get_graph().draw_mermaid_png()
        
        with open("orchestration_workflow.png", "wb") as f:
            f.write(png_data)


if __name__ == "__main__":
    orchestration_agent = SelfPlanAgent(
        model="mistral:7b", 
    )

    orchestration_agent.save_workflow_image()
    
    response = orchestration_agent.generate_response(
        query="Which magazine was started first Arthur's Magazine or First for Women?", 
        context="..."
    )
    print(f"Response:\n{response}")