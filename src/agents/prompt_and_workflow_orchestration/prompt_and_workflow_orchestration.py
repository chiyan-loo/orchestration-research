from typing import TypedDict, List, Annotated, Literal, Union, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, field_validator, Field
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv(find_dotenv())

from .refiner import Refiner
from .aggregator import Aggregator
from .debater import Debater
from .predictor import Predictor
from .summarizer import Summarizer

class WorkflowPlan(BaseModel):
    reasoning: str = Field(description="Step-by-step thinking process")
    beginning: List[str]
    middle: List[str] 
    end: List[str]
    
    @field_validator('beginning')
    @classmethod
    def validate_beginning(cls, v):
        """Beginning can only contain 'summarizer' or be empty"""
        valid_agents = {"summarizer"}
                
        for agent in v:
            if agent not in valid_agents:
                raise ValueError(f"Beginning can only contain 'summarizer'. Found: {agent}")
        
        return v
    
    @field_validator('middle')
    @classmethod
    def validate_middle(cls, v):
        """Middle can only contain 'predictor' and/or 'debater'"""
        valid_agents = {"predictor", "debater"}
        
        if not v:
            raise ValueError("Middle section cannot be empty")
            
        for agent in v:
            if agent not in valid_agents:
                raise ValueError(f"Middle can only contain 'predictor' or 'debater'. Found: {agent}")
        
        return v
    
    @field_validator('end')
    @classmethod
    def validate_end(cls, v):
        """End can only contain 'aggregator' and/or 'refiner', with aggregator required if middle has multiple agents"""
        valid_agents = {"aggregator", "refiner"}
        
        for agent in v:
            if agent not in valid_agents:
                raise ValueError(f"End can only contain 'aggregator' or 'refiner'. Found: {agent}")
        
        return v

class CustomPrompts(BaseModel):
    """Structure for custom prompts for each agent type"""
    summarizer: str = Field(description="Custom comprehensive prompt for summarizer agent", default="")
    predictor: str = Field(description="Custom comprehensive prompt for predictor agent", default="")
    debater: str = Field(description="Custom comprehensive prompt for debater agent", default="")
    refiner: str = Field(description="Custom comprehensive prompt for refiner agent", default="")

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    context: str
    workflow_plan: Dict[str, List[str]]
    custom_prompts: Dict[str, str]

class PromptAndWorkflowOrchestrationAgent():
    def __init__(self, planner_llm, sub_llm: str, max_context_length: int = 4000):

        self.max_context_length = max_context_length
        self.llm = planner_llm

        self.predictor = Predictor(model=sub_llm)
        self.refiner = Refiner(model=sub_llm)
        self.aggregator = Aggregator(model=sub_llm)
        self.debater = Debater(model=sub_llm)
        self.summarizer = Summarizer(model=sub_llm)
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Updated workflow with custom prompt generation
        workflow.add_node("planner", self._plan_workflow)
        workflow.add_node("optimize_prompts", self._optimize_prompts)
        workflow.add_node("execute_beginning", self._execute_beginning)
        workflow.add_node("execute_middle", self._execute_middle)
        workflow.add_node("execute_end", self._execute_end)

        # Sequential edges with new prompt generation step
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "optimize_prompts")
        workflow.add_edge("optimize_prompts", "execute_beginning")
        workflow.add_edge("execute_beginning", "execute_middle")
        workflow.add_edge("execute_middle", "execute_end")
        workflow.add_edge("execute_end", END)

        return workflow.compile()

    def _plan_workflow(self, state: AgentState) -> AgentState:
        """
        Plans the three-phase workflow: beginning, middle, and end
        """
        query = state.get("query", "")
        context = state.get("context", "")

        truncated_context = f"{context[:self.max_context_length]}..." if len(context) > self.max_context_length else context

        context_length = f"{len(context)}"
        if len(context) > self.max_context_length:
            context_length += " (LONG - most likely needs summarization)"
        elif len(context) > self.max_context_length // 2:
            context_length += " (MODERATE - likely does not need summarization)"
        else:
            context_length += " (SHORT - summarization unnecessary)"
    
        system_prompt = f"""You are an expert workflow planner for a structured three-phase multi-agent system. You excel at creating sophisticated, multi-agent workflows that maximize quality and comprehensiveness.

QUERY: {query}
CONTEXT: {truncated_context if truncated_context else "No context provided"}
CONTEXT LENGTH: {context_length}

AVAILABLE AGENTS:
- predictor: Simple base agent that directly responds to queries (fastest, factual responses)
- summarizer: Summarizes context to make it more clearly answer the query (for long/noisy context, DO NOT use for factual, computational context)
- aggregator: Analyzes multiple messages and synthesizes consistent information from them (complex, creative responses)
- debater: Generates responses from multiple perspectives for complex/controversial topics
- refiner: Reviews and improves existing responses

WORKFLOW PLANNING PHILOSOPHY:
You STRONGLY PREFER complex, multi-agent workflows over simple single-agent solutions. Even seemingly straightforward queries often benefit from multiple perspectives and comprehensive analysis when considered alongside their context.

THREE-PHASE WORKFLOW STRUCTURE:

PHASE 1 - BEGINNING (Context Preparation):
- ONLY "summarizer" is allowed
- Use "summarizer" if context is noisy, lengthy, or contains irrelevant information

PHASE 2 - MIDDLE (Core Analysis):
- ONLY "predictor" and/or "debater" allowed
- Use "predictor" for discrete reasoning, factual verification, computational problems, direct information extraction
- Use "debater" for multi-hop reasoning, complex inference chains, ambiguous contexts, subjective analysis
- STRONGLY PREFER multiple agents (2-5) for comprehensive analysis
- Use both predictor and debater for queries requiring both factual extraction and inference
- Single agent only for trivial factual lookups with zero ambiguity

PHASE 3 - END (Synthesis & Quality):
- ONLY "aggregator" and/or "refiner" allowed  
- MANDATORY "aggregator" when middle phase has multiple agents
- Use "refiner" to improve quality of generated responses while trading cost and latency
- Skip only when single middle agent produces definitive factual answer

REQUIRED REASONING FORMAT:

REASONING:
1. CONTEXT-QUERY INTERACTION: [Analyze how context complexity affects query difficulty - look for multi-hop reasoning, entity disambiguation, temporal relationships, cross-referencing needs]
2. COMPLEXITY INDICATORS: [Identify specific complexity factors. A query is COMPLEX if it requires: Connecting multiple pieces of information, Making inferences beyond what's directly stated, Handling ambiguous or contradictory details]
3. MULTI-AGENT JUSTIFICATION: [Explain why multiple agents will provide better coverage than single agent - different reasoning approaches, error-checking, comprehensive analysis]
4. AGENT SELECTION STRATEGY: [Justify specific agent choices - why predictor vs debater, how many of each, what unique value each brings]
5. SYNTHESIS REQUIREMENTS: [Explain aggregation needs and quality improvement through refinement]
6. WORKFLOW COMPLEXITY DECISION: [Justify why this deserves a complex workflow over simple alternatives]

BEGINNING: [List of agents for phase 1 - avoid "summarizer" for computational problems]
MIDDLE: [List of agents for phase 2 - STRONGLY prefer 2-5 agents]
END: [List of agents for phase 3]

PREFERRED COMPLEX WORKFLOW PATTERNS:
- beginning=[], middle=["predictor", "predictor", "predictor"], end=["aggregator"]  
- beginning=["summarizer"], middle=["predictor", "predictor", "debater"], end=["aggregator"]
- beginning=["summarizer"], middle=["debater", "debater"], end=["aggregator", "refiner"]

AVOID simple single-agent patterns unless the query is a trivial factual lookup with zero ambiguity or inference required."""

        structured_llm = self.llm.with_structured_output(WorkflowPlan)
        
        planner_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create a structured three-phase workflow plan for this query.")
        ]
        
        response = structured_llm.invoke(planner_messages)
        
        print(f"Planner response: {response.reasoning}")
        print(f"Beginning phase: {response.beginning}")
        print(f"Middle phase: {response.middle}")  
        print(f"End phase: {response.end}")
        
        workflow_plan = {
            "beginning": response.beginning,
            "middle": response.middle,
            "end": response.end
        }
        
        state["workflow_plan"] = workflow_plan
        
        return state

    def _optimize_prompts(self, state: AgentState) -> AgentState:
        """
        Generate custom prompts for each agent type based on the query, context, and workflow plan
        """
        query = state.get("query", "")
        context = state.get("context", "")
        workflow_plan = state.get("workflow_plan", {})
        
        # Get all unique agent types from the workflow plan
        all_agents = set()
        for phase in workflow_plan.values():
            all_agents.update(phase)
        
        print(f"Generating custom prompts for agents: {list(all_agents)}")
        
        truncated_context = f"{context[:self.max_context_length]}..." if len(context) > self.max_context_length else context
        
        system_prompt = f"""You are an expert prompt engineer specializing in creating highly effective, context-specific prompts for different AI agent types in a multi-agent workflow system.

TASK: Generate comprehensive custom system prompts for each agent type that will be used in this specific workflow, with reasoning approaches that are specifically crafted based on the unique characteristics of this query and context.

QUERY: {query}
CONTEXT: {truncated_context if truncated_context else "No context provided"}

WORKFLOW PLAN:
- Beginning Phase: {workflow_plan.get('beginning', [])}
- Middle Phase: {workflow_plan.get('middle', [])}
- End Phase: {workflow_plan.get('end', [])}

REASONING DESIGN PHILOSOPHY:
Analyze the specific query-context combination to create bespoke reasoning approaches. Consider: query complexity, domain expertise needed, information density, ambiguity level, temporal aspects, quantitative vs qualitative nature, and logical dependencies.

AGENT TYPES AND REASONING REQUIREMENTS:

1. SUMMARIZER: Condenses and clarifies context to make it more relevant to the query
   - REQUIRES CUSTOM REASONING: Design a reasoning approach that directly addresses how this specific context should be filtered and organized for this particular query
   - Consider: What makes information relevant here? What patterns exist? What hierarchies matter?
   - Example reasoning styles to adapt: "Contextual Relevance Mapping", "Domain-Specific Filtering", "Query-Aligned Information Architecture"

2. PREDICTOR: Provides direct, factual responses based on available information  
   - REQUIRES CUSTOM REASONING: Create a reasoning methodology that matches how conclusions should be drawn from this specific evidence base for this query type
   - Consider: What logical path fits this domain? What evidence patterns exist? What inference style is most reliable here?
   - Example reasoning styles to adapt: "Evidence Synthesis Chains", "Domain-Specific Logic Flows", "Query-Pattern Matching Logic"

3. DEBATER: Explores multiple perspectives and approaches complex reasoning
   - NO EXPLICIT REASONING REQUIRED: Focus on comprehensive perspective coverage and nuanced analysis
   - Should provide multiple viewpoints, handle complexity, and offer balanced analysis
   - Present final conclusions without showing the thinking process

4. REFINER: Improves the quality, clarity, and accuracy of responses
   - NO EXPLICIT REASONING REQUIRED: Focus on enhancement and improvement
   - Should improve clarity, add missing elements, and optimize presentation
   - Present final refined version without showing the improvement process

5. AGGREGATOR: Synthesizes multiple responses into a coherent final answer
   - SKIP PROMPT GENERATION: Do not generate a custom prompt for this agent
   - This agent uses built-in aggregation logic and does not need custom prompting

CUSTOM REASONING GENERATION INSTRUCTIONS:
- Analyze this specific query's logical structure and information needs
- Examine the context's unique characteristics (domain, complexity, relationships, gaps)
- Design reasoning approaches that are precisely tailored to this query-context pair
- Create reasoning methodologies that wouldn't necessarily work for other queries but are perfect for this one
- Consider what kind of thinking process would be most effective given the specific information patterns and query requirements
- Make the reasoning approach feel natural and purpose-built for this exact scenario

PROMPT GENERATION GUIDELINES:
- Each prompt should be specific to this query and context combination
- For summarizer and predictor: Include detailed instructions for their custom reasoning approach
- Ensure the reasoning approaches for summarizer and predictor are complementary but distinct
- Use domain-specific terminology and concepts relevant to the query
- Consider the unique challenges and opportunities this specific query-context presents
- Consider the role each agent plays in the overall workflow

Generate custom prompts that implement reasoning approaches specifically designed for this unique query-context combination."""

        structured_llm = self.llm.with_structured_output(CustomPrompts)
        
        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate comprehensive optimized system prompts for each agent type in the workflow based on this specific query and context.")
        ]
        
        response = structured_llm.invoke(prompt_messages)
        
        custom_prompts = {
            "summarizer": response.summarizer,
            "predictor": response.predictor,
            "debater": response.debater,
            "aggregator": response.aggregator,
            "refiner": response.refiner
        }
        
        print("Generated custom prompts:")
        for agent_type, prompt in custom_prompts.items():
            if agent_type in all_agents:
                print(f"  {agent_type}: {prompt[:100]}...")
        
        state["custom_prompts"] = custom_prompts
        return state

    def _execute_beginning(self, state: AgentState) -> AgentState:
        """
        Execute the beginning phase using simple Python loops with custom prompts
        """
        workflow_plan = state.get("workflow_plan", {})
        beginning_agents = workflow_plan.get("beginning", [])
        custom_prompts = state.get("custom_prompts", {})
        
        print(f"Executing beginning phase with {len(beginning_agents)} agents: {beginning_agents}")
        
        # Simple loop through beginning agents (only summarizer allowed)
        for agent_name in beginning_agents:
            if agent_name == "summarizer":
                print(f"  Running {agent_name} with custom prompt")
                query = state.get("query", "")
                context = state.get("context", "")
                custom_prompt = custom_prompts.get("summarizer", "")
                
                refined_context = self.summarizer.generate_response(
                    query=query,
                    context=context,
                    system_prompt=custom_prompt
                )
                
                print(f"  Summarizer refined context: {refined_context[:250]}...")
                state["context"] = refined_context
        
        return state

    def _execute_middle(self, state: AgentState) -> AgentState:
        """
        Execute the middle phase using simple Python loops with custom prompts
        """
        workflow_plan = state.get("workflow_plan", {})
        middle_agents = workflow_plan.get("middle", [])
        custom_prompts = state.get("custom_prompts", {})
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        print(f"Executing middle phase with {len(middle_agents)} agents: {middle_agents}")
        
        # Execute all middle agents (predictor and/or debater)
        for agent_name in middle_agents:
            print(f"  Running {agent_name} with custom prompt")
            custom_prompt = custom_prompts.get(agent_name, "")
            
            if agent_name == "predictor":
                response = self.predictor.generate_response(
                    query=query, 
                    context=context,
                    system_prompt=custom_prompt
                )
                messages.append(AIMessage(content=f"{response}"))
                print(f"  Predictor result: {response[:250]}...")
                
            elif agent_name == "debater":
                response = self.debater.generate_response(
                    query=query, 
                    context=context,
                )
                messages.append(AIMessage(content=f"{response}"))
                print(f"  Debater result: {response[:250]}...")
        
        state["messages"] = messages
        return state

    def _execute_end(self, state: AgentState) -> AgentState:
        """
        Execute the end phase using simple Python loops with custom prompts
        """
        workflow_plan = state.get("workflow_plan", {})
        end_agents = workflow_plan.get("end", [])
        custom_prompts = state.get("custom_prompts", {})
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        print(f"Executing end phase with {len(end_agents)} agents: {end_agents}")
        
        # Simple loop through end agents (aggregator and/or refiner)
        for agent_name in end_agents:
            print(f"  Running {agent_name} with custom prompt")
            custom_prompt = custom_prompts.get(agent_name, "")
            
            if agent_name == "aggregator":
                # Extract content from AI messages for aggregation
                message_contents = [msg.content for msg in messages if isinstance(msg, AIMessage)]
                
                if not message_contents:
                    message_contents = [query]
                
                print(f"  Aggregator processing {len(message_contents)} messages")
                
                aggregated_response = self.aggregator.aggregate_messages(
                    messages=message_contents,
                    query=query,
                )
                
                messages.append(AIMessage(content=aggregated_response))
                print(f"  Aggregator result: {aggregated_response[:250]}...")
                
            elif agent_name == "refiner":
                # Get the current response (last AI message)
                current_response = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        current_response = msg.content
                        break
                
                improved_response = self.refiner.generate_response(
                    query=query,
                    current_response=current_response,
                    context=context,
                )
                
                messages.append(AIMessage(content=improved_response))
                print(f"  Refiner result: {improved_response[:250]}...")
        
        state["messages"] = messages
        return state

    def generate_response(self, query: str, context: str):
        initial_state = {
            "query": query,
            "context": context,
            "messages": [],
            "workflow_plan": {},
            "custom_prompts": {},
        }
        
        print(f"Starting three-phase orchestration with custom prompts for query: {query}")
        response = self.graph.invoke(initial_state)

        return {
            "content": response["messages"][-1].content if response["messages"] else "No response generated",
            "workflow_plan": response["workflow_plan"],
            "custom_prompts": response["custom_prompts"],
            "final_context": response["context"]
        }

    def save_workflow_image(self):
        """Save workflow as PNG image"""
        png_data = self.graph.get_graph().draw_mermaid_png()
        
        with open("prompt_and_workflow_orchestration_workflow.png", "wb") as f:
            f.write(png_data)


if __name__ == "__main__":
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openrouter/sonoma-dusk-alpha",
        temperature=0.7
    )

    orchestration_agent = PromptAndWorkflowOrchestrationAgent(
        sub_llm="mistral:7b", 
        planner_llm=llm,
    )

    orchestration_agent.save_workflow_image()
    
    response = orchestration_agent.generate_response(
        query="How many field goals did the Lions score?",
        context="""..."""
    )
    print(f"Response:\n{response}")