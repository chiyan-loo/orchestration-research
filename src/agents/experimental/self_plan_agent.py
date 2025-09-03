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
    workflow_plan: List[Union[str, List[str]]]  # Can now contain lists for parallel execution
    current_step: int
    next_agent: Literal["aggregator", "debator", "reflector", "summarizer", "predictor", "parallel", "end"]
    parallel_results: List[str]  # Store results from parallel execution

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
        workflow.add_node("parallel_executor", self._execute_parallel)

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
                "parallel": "parallel_executor",
            }
        )

        workflow.add_edge("predict", "executor")
        workflow.add_edge("aggregate", "executor")
        workflow.add_edge("debate", "executor")
        workflow.add_edge("reflect", "executor")
        workflow.add_edge("summarize", "executor")
        workflow.add_edge("parallel_executor", "executor")

        return workflow.compile()

    def _plan_workflow(self, state: AgentState) -> AgentState:
        """
        Plans the complete workflow based on query and context using chain of thought reasoning
        """
        query = state.get("query", "")
        context = state.get("context", "")
        
        system_prompt = f"""You are a workflow planner for a multi-agent system. Use chain of thought reasoning to create an optimal sequence of agents.

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
6. You can specify multiple instances of the same agent for parallel execution by using format: [agent1, agent2, agent3]
7. After parallel execution, you typically want to use 'aggregator' to synthesize the results
8. Adapt workflows to the query. For complex queries, make complex workflows. For simple queries, make simple workflows.

PARALLEL EXECUTION EXAMPLES:
- Simple query: predictor
- Fact-checking query: summarizer, predictor, reflector
- Complex debate: summarizer, [debator, debator, debator], aggregator, reflector
- Comparative analysis: [predictor, debator], aggregator
- Multi-perspective research: summarizer, [debator, debator, predictor, predictor], aggregator, reflector

CHAIN OF THOUGHT REASONING PROCESS:
Think step by step through these questions:

1. QUERY ANALYSIS: What type of question is this?
   - Simple factual (dates, names, basic definitions)
   - Complex analytical (requires deep thinking, comparisons)
   - Controversial/subjective (multiple valid viewpoints)
   - Creative (open-ended, imaginative)
   - Research-based (needs information synthesis)

2. PROCESSING REQUIREMENTS: What does this query need?
   - Does the context need summarization/refinement first?
   - Would multiple perspectives improve the answer quality?
   - Is there potential for hallucination that needs fact-checking?
   - Would parallel processing provide better coverage or quality?

3. WORKFLOW OPTIMIZATION: What's the best sequence?
   - Start with context refinement if needed (summarizer)
   - Use parallel execution for diverse perspectives or redundancy
   - Always aggregate after parallel execution
   - End with reflection for quality improvement

4. PARALLEL DECISION: When to use parallel execution?
   - Use 2-4 debators for complex/controversial topics
   - Use multiple predictors for fact verification
   - Mix different agents for comprehensive coverage
   - Always follow parallel execution with aggregator

Now analyze this specific query:

<reasoning>
Let me think through this step by step:

1. QUERY ANALYSIS: [Analyze the query type and complexity]

2. PROCESSING REQUIREMENTS: [Determine what processing is needed]

3. WORKFLOW OPTIMIZATION: [Design the optimal sequence]

4. PARALLEL DECISION: [Decide if and where to use parallel execution]
</reasoning>

<workflow>
[Provide the final workflow plan]
</workflow>"""

        planner_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create an optimal workflow plan for this query using chain of thought reasoning.")
        ]
        
        response = self.llm.invoke(planner_messages)
        response_content = response.content.strip()
        
        # Extract workflow from the response
        workflow_section = ""
        if "<workflow>" in response_content and "</workflow>" in response_content:
            start = response_content.find("<workflow>") + len("<workflow>")
            end = response_content.find("</workflow>")
            workflow_section = response_content[start:end].strip()
        
        # Parse the workflow plan
        workflow_plan = self._parse_workflow_plan(workflow_section)
        
        print(f"Planned workflow: {workflow_plan}")
        
        state["workflow_plan"] = workflow_plan
        state["current_step"] = 0
        state["parallel_results"] = []
        
        return state

    def _parse_workflow_plan(self, workflow_text: str) -> List[Union[str, List[str]]]:
        """
        Parse workflow text into a structured plan that can include parallel execution
        """
        workflow_plan = []
        lines = [line.strip() for line in workflow_text.split('\n') if line.strip()]
        
        valid_agents = ["predictor", "summarizer", "aggregator", "debator", "reflector"]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a parallel execution block [agent1, agent2, agent3]
            if line.startswith('[') and line.endswith(']'):
                # Parse parallel agents
                agents_str = line[1:-1]  # Remove brackets
                parallel_agents = [agent.strip() for agent in agents_str.split(',')]
                # Filter valid agents
                valid_parallel_agents = [agent for agent in parallel_agents if agent in valid_agents]
                if valid_parallel_agents:
                    workflow_plan.append(valid_parallel_agents)
            else:
                # Single agent
                if line in valid_agents:
                    workflow_plan.append(line)
        
        return workflow_plan

    def _execute_workflow(self, state: AgentState) -> AgentState:
        """
        Executes the next step in the planned workflow
        """
        workflow_plan = state.get("workflow_plan", [])
        current_step = state.get("current_step", 0)
        
        if current_step >= len(workflow_plan):
            state["next_agent"] = "end"
            return state
        
        next_step = workflow_plan[current_step]
        
        # Check if this step is parallel execution (list of agents)
        if isinstance(next_step, list):
            state["next_agent"] = "parallel"
            state["parallel_agents"] = next_step
        else:
            # Single agent execution
            state["next_agent"] = next_step
        
        state["current_step"] = current_step + 1
        
        print(f"Executing step {current_step + 1}/{len(workflow_plan)}: {next_step}")
        
        return state

    def _execute_parallel(self, state: AgentState) -> AgentState:
        """
        Execute multiple agents in parallel and store their results
        """
        parallel_agents = state.get("parallel_agents", [])
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        print(f"Executing {len(parallel_agents)} agents in parallel: {parallel_agents}")
        
        parallel_results = []
        
        for agent_name in parallel_agents:
            print(f"  Running {agent_name}...")
            
            if agent_name == "predictor":
                result = self.predictor.generate_response(query=query, context=context)
            elif agent_name == "debator":
                result = self.debator.generate_response(query=query, context=context)
            elif agent_name == "summarizer":
                result = self.summarizer.generate_response(query=query, context=context)
            elif agent_name == "reflector":
                # For reflector, use the last AI message if available
                current_response = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        current_response = msg.content
                        break
                result = self.reflector.generate_response(
                    query=query, 
                    current_response=current_response, 
                    context=context
                )
            elif agent_name == "aggregator":
                # For aggregator, use existing AI messages
                message_contents = [msg.content for msg in messages if isinstance(msg, AIMessage)]

                result = self.aggregator.aggregate_messages(
                    messages=message_contents,
                    query=query
                )
            else:
                result = f"Unknown agent: {agent_name}"
            
            parallel_results.append(result)
            print(f"  {agent_name} result: {result[:100]}...")
        
        # Add all parallel results as separate AI messages
        for i, result in enumerate(parallel_results):
            messages.append(AIMessage(content=f"[{parallel_agents[i]}]: {result}"))
        
        state["messages"] = messages
        state["parallel_results"] = parallel_results
        
        return state

    def _predict(self, state: AgentState) -> AgentState:
        """
        Simple predictor agent that directly responds to queries
        """
        print("Predictor executing")
        
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        response = self.predictor.generate_response(query=query, context=context)
        
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
            "parallel_results": [],
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
    
    response = orchestration_agent.generate_response(
        query="Which magazine was started first Arthur's Magazine or First for Women?", 
        context="..."
    )
    print(f"Response:\n{response}")