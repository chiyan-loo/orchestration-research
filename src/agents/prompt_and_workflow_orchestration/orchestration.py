from typing import TypedDict, List, Annotated, Literal, Union, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, field_validator, Field, create_model
import os
import asyncio
from dotenv import load_dotenv, find_dotenv

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

class DebaterPrompts(BaseModel):
    """Structure for debater sub-agent prompts"""
    advocate: str = Field(description="Custom prompt for the advocate sub-agent", default="")
    critic: str = Field(description="Custom prompt for the critic sub-agent", default="")

class RefinerPrompts(BaseModel):
    """Structure for refiner sub-agent prompts"""
    critic: str = Field(description="Custom prompt for the critic sub-agent", default="")
    editor: str = Field(description="Custom prompt for the editor sub-agent", default="")

class CustomPrompts(BaseModel):
    """Structure for custom prompts for each agent type"""
    summarizer: str = Field(description="Custom comprehensive prompt for summarizer agent", default="")
    predictor: str = Field(description="Custom comprehensive prompt for predictor agent", default="")
    debater: DebaterPrompts = Field(description="Custom prompts for debater sub-agents", default_factory=DebaterPrompts)
    refiner: RefinerPrompts = Field(description="Custom prompts for refiner sub-agents", default_factory=RefinerPrompts)

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    context: str
    workflow_plan: Dict[str, List[str]]
    planner_reasoning: str
    custom_prompts: Dict[str, Union[str, Dict[str, str]]]

class OrchestrationAgent():
    def __init__(self, 
                 planner_llm: BaseLanguageModel, 
                 high_temp_llm: BaseLanguageModel,
                 medium_temp_llm: BaseLanguageModel,
                 low_temp_llm: BaseLanguageModel, 
                 max_context_length: int = 4000):

        self.max_context_length = max_context_length
        self.llm = planner_llm

        self.predictor = Predictor(llm=high_temp_llm)
        self.refiner = Refiner(llm=medium_temp_llm)
        self.aggregator = Aggregator(llm=low_temp_llm)
        self.debater = Debater(predictor_llm=high_temp_llm, synthesizer_llm=medium_temp_llm)
        self.summarizer = Summarizer(llm=low_temp_llm)
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Use async methods directly
        workflow.add_node("planner", self._plan_workflow_async)
        workflow.add_node("optimize_prompts", self._optimize_prompts_async)
        workflow.add_node("execute_beginning", self._execute_beginning_async)
        workflow.add_node("execute_middle", self._execute_middle_async)
        workflow.add_node("execute_end", self._execute_end_async)

        # Sequential edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "optimize_prompts")
        workflow.add_edge("optimize_prompts", "execute_beginning")
        workflow.add_edge("execute_beginning", "execute_middle")
        workflow.add_edge("execute_middle", "execute_end")
        workflow.add_edge("execute_end", END)

        return workflow.compile()

    async def _plan_workflow_async(self, state: AgentState) -> AgentState:
        """
        Plans the three-phase workflow: beginning, middle, and end (async version)
        """
        query = state.get("query", "")
        context = state.get("context", "")

        truncated_context = f"{context[:self.max_context_length]}..." if len(context) > self.max_context_length else context

        context_length = f"{len(context)}"
        if len(context) > self.max_context_length:
            context_length += " (LONG - needs summarization)"
        elif len(context) > self.max_context_length // 2:
            context_length += " (MODERATE - likely does not need summarization)"
        else:
            context_length += " (SHORT - summarization unnecessary)"
    
        system_prompt = f"""You are an expert workflow planner for a structured three-phase multi-agent system. You excel at creating sophisticated, multi-agent workflows that maximize quality and accuracy.

QUERY: {query}
CONTEXT: {truncated_context if truncated_context else "No context provided"}
CONTEXT LENGTH: {context_length}

AVAILABLE AGENTS:
- predictor: Directly responds to queries (use for: calculations, lookups)
- summarizer: Summarizes context to make it more clearly answer the query (for long/noisy context, DO NOT use for factual, computational context)
- aggregator: Synthesizes multiple agent outputs
- debater: Multi-perspective analysis (use for: ambiguous topics, multi-hop reasoning)
- refiner: Improves response accuracy

WORKFLOW PLANNING PHILOSOPHY:
You PREFER complex, multi-agent workflows over simple single-agent solutions. Even seemingly straightforward queries often benefit from multiple perspectives and comprehensive analysis when considered alongside their context.

THREE-PHASE WORKFLOW STRUCTURE:

PHASE 1 - BEGINNING (Context Preparation):
- ONLY "summarizer" is allowed
- Use "summarizer" if context is noisy, lengthy, or contains irrelevant information

PHASE 2 - MIDDLE (Core Analysis):
- ONLY "predictor" and/or "debater" allowed
- Use multiple "predictor" for discrete reasoning, factual verification, computational problems
- Use multiple "debater" for multi-hop reasoning, complex inference chains, ambiguous contexts, subjective analysis
- STRONGLY PREFER multiple agents (3-4) for comprehensive analysis
- Single agent only for trivial factual lookups with zero ambiguity

PHASE 3 - END (Synthesis & Quality):
- ONLY "aggregator" and/or "refiner" allowed  
- MANDATORY "aggregator" when middle phase has multiple agents
- Use "refiner" to improve quality of generated responses and reduce hallucinations
- Skip only when single middle agent produces definitive factual answer

REQUIRED REASONING FORMAT:

REASONING:
1. CONTEXT-QUERY INTERACTION: [Analyze how context complexity affects query difficulty - look for multi-hop reasoning, entity disambiguation, temporal relationships, cross-referencing needs]
2. MULTI-AGENT JUSTIFICATION: [If query is complex, explain why multiple agents will provide better coverage. Default to 3-4 agents for complex queries.]
3. SUMMARIZER OR NOT?
   - YES: Context is noisy, lengthy, contains irrelevant information
   - NO: Context is clean, concise, or computational/factual
4. PREDICTOR OR DEBATER OR BOTH?
   - PREDICTOR: fact retrieval, calculations, discrete reasoning
   - DEBATER: multi-hop reasoning, connecting multiple facts
5. AGGREGATOR: Required if middle phase has multiple agents (synthesizes all outputs)
6. REFINER OR NOT?
   - YES: Response requires verification, quality improvement, or hallucination risk (computations, multi-step inferences, synthesized information)
   - NO: Direct retrieval from clear context with low hallucination risk

BEGINNING: [List of agents for phase 1 - avoid "summarizer" for computational problems]
MIDDLE: [List of agents for phase 2 - STRONGLY prefer 3-4 agents]
END: [List of agents for phase 3]

AVOID simple single-agent patterns unless the query is a trivial factual lookup with zero ambiguity or inference required."""

        structured_llm = self.llm.with_structured_output(WorkflowPlan)
        
        planner_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create a structured three-phase workflow plan for this query.")
        ]
        
        response = await structured_llm.ainvoke(planner_messages)
        
        print(f"Planner reasoning: {response.reasoning}")
        print(f"Beginning phase: {response.beginning}")
        print(f"Middle phase: {response.middle}")  
        print(f"End phase: {response.end}")
        
        workflow_plan = {
            "beginning": response.beginning,
            "middle": response.middle,
            "end": response.end
        }
        
        state["planner_reasoning"] = response.reasoning
        state["workflow_plan"] = workflow_plan
        
        return state

    async def _optimize_prompts_async(self, state: AgentState) -> AgentState:
        """
        Generate custom prompts for each agent type based on the query, context, and workflow plan (async version)
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
        
        # Build agent-specific sections only for agents in the workflow (excluding aggregator)
        agent_sections = []
        
        if "summarizer" in all_agents:
            agent_sections.append("""1. SUMMARIZER: Condenses and clarifies context to make it more relevant to the query
    - REQUIRES CUSTOM REASONING: Design a reasoning approach that directly addresses how this specific context should be filtered and organized for this particular query
    - Consider: What makes information relevant here? What information must be kept?""")
        
        if "predictor" in all_agents:
            agent_sections.append("""2. PREDICTOR: Provides direct, factual responses based on available information
    - REQUIRES CUSTOM REASONING: Create a reasoning methodology that matches how conclusions should be drawn from this specific evidence base for this query type
    - Consider: What logical path fits this domain? What evidence patterns exist? What inference style is most reliable here?""")
        
        if "debater" in all_agents:
            agent_sections.append("""3. DEBATER: Multi-agent system that explores different perspectives and approaches
    - ADVOCATE: Develops one comprehensive approach to solving the query independently (no predetermined conclusions)
    - CRITIC: Explores alternative approaches or identifies potential limitations independently (no predetermined positions)
    - CRITICAL: Let both agents discover their own reasoning processes and conclusions based on the query-context""")
        
        if "refiner" in all_agents:
            agent_sections.append("""4. REFINER: Multi-agent system that fact checks responses
    - CRITIC: Identifies accuracy and hallucination issues specific to this query type, values conciseness
    - EDITOR: Implements improvements based on critique specific to the query type to reduce increase accuracy, values conciseness""")
        
        agent_types_text = "\n\n".join(agent_sections)
        
        # Build guidelines specific to agents in workflow (excluding aggregator)
        guidelines = []
        guidelines.append("- Each prompt should be specific to this query and context combination")
        guidelines.append("- Include detailed instructions for their custom reasoning approach")
        
        if "debater" in all_agents:
            guidelines.extend([
                "- For debater: Create two distinct prompts that encourage independent exploration of the problem space",
                "  * Neither should be told what specific conclusions to reach",
                "  * Advocate should focus on developing one comprehensive approach to the problem",
                "  * Critic should focus on exploring alternative approaches or identifying potential limitations"
            ])
        
        if "refiner" in all_agents:
            guidelines.append("- For refiner: Create complementary prompts where critic identifies domain-specific issues and editor applies domain-specific improvements")
        
        guidelines.extend([
            "- Use domain-specific terminology and concepts relevant to the query",
            "- Consider the unique challenges and opportunities this specific query-context presents",
            "- Design reasoning approaches that feel natural and purpose-built for this exact scenario"
        ])
        
        guidelines_text = "\n".join(guidelines)
        
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

    AGENT TYPES:

    {agent_types_text}

    PROMPT GENERATION GUIDELINES:
    {guidelines_text}

    Generate custom prompts that implement reasoning approaches specifically designed for this unique query-context combination."""
        
        fields = {}
        if "summarizer" in all_agents:
            fields["summarizer"] = (str, Field(description="Custom comprehensive prompt for summarizer agent", default=""))
        if "predictor" in all_agents:
            fields["predictor"] = (str, Field(description="Custom comprehensive prompt for predictor agent", default=""))
        if "debater" in all_agents:
            fields["debater"] = (DebaterPrompts, Field(description="Custom prompts for debater sub-agents", default_factory=DebaterPrompts))
        if "refiner" in all_agents:
            fields["refiner"] = (RefinerPrompts, Field(description="Custom prompts for refiner sub-agents", default_factory=RefinerPrompts))
        
        DynamicCustomPrompts = create_model('DynamicCustomPrompts', **fields)
        
        structured_llm = self.llm.with_structured_output(DynamicCustomPrompts)
        
        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate comprehensive optimized system prompts for each agent type in the workflow based on this specific query and context.")
        ]
        
        response = await structured_llm.ainvoke(prompt_messages)
        
        # Structure the custom prompts to handle the nested debater and refiner structures (excluding aggregator)
        custom_prompts = {}
        
        if "summarizer" in all_agents:
            custom_prompts["summarizer"] = response.summarizer
        if "predictor" in all_agents:
            custom_prompts["predictor"] = response.predictor
        if "debater" in all_agents:
            custom_prompts["debater"] = {
                "advocate": response.debater.advocate,
                "critic": response.debater.critic
            }
        if "refiner" in all_agents:
            custom_prompts["refiner"] = {
                "critic": response.refiner.critic,
                "editor": response.refiner.editor
            }
        
        print("Generated custom prompts:")
        for agent_type, prompt in custom_prompts.items():
            if agent_type in ["debater", "refiner"]:
                print(f"  {agent_type}:")
                for sub_agent, sub_prompt in prompt.items():
                    print(f"    {sub_agent}: {sub_prompt[:100]}...")
            else:
                print(f"  {agent_type}: {prompt[:100]}...")
        
        state["custom_prompts"] = custom_prompts
        return state

    async def _execute_beginning_async(self, state: AgentState) -> AgentState:
        """
        Execute the beginning phase using async with custom prompts
        """
        workflow_plan = state.get("workflow_plan", {})
        beginning_agents = workflow_plan.get("beginning", [])
        custom_prompts = state.get("custom_prompts", {})
        
        print(f"Executing beginning phase with {len(beginning_agents)} agents: {beginning_agents}")
        
        # Simple loop through beginning agents (only summarizer allowed)
        # This phase is sequential because context needs to be refined before middle phase
        for agent_name in beginning_agents:
            if agent_name == "summarizer":
                print(f"  Running {agent_name} with custom prompt")
                query = state.get("query", "")
                context = state.get("context", "")
                custom_prompt = custom_prompts.get("summarizer", "")
                
                # Wrap synchronous call in executor to make it async
                loop = asyncio.get_event_loop()
                refined_context = await loop.run_in_executor(
                    None,
                    self.summarizer.generate_response,
                    query,
                    context,
                    custom_prompt
                )
                
                print(f"  Summarizer refined context: {refined_context[:250]}...")
                state["context"] = refined_context
        
        return state

    async def _execute_middle_async(self, state: AgentState) -> AgentState:
        """
        Execute the middle phase in PARALLEL using asyncio.gather with custom prompts
        """
        workflow_plan = state.get("workflow_plan", {})
        middle_agents = workflow_plan.get("middle", [])
        custom_prompts = state.get("custom_prompts", {})
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        print(f"Executing middle phase with {len(middle_agents)} agents IN PARALLEL: {middle_agents}")
        
        # Create async tasks for all middle agents
        async def run_predictor():
            custom_prompt = custom_prompts.get("predictor", "")
            print(f"  Running predictor with custom prompt: {custom_prompt[:50]}...")
            
            # Wrap synchronous call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.predictor.generate_response,
                query,
                context,
                custom_prompt
            )
            print(f"  Predictor result: {response[:250]}...")
            return ("predictor", response)
        
        async def run_debater():
            debater_prompts = custom_prompts.get("debater", {})
            advocate_prompt = debater_prompts.get("advocate", "")
            critic_prompt = debater_prompts.get("critic", "")
            
            print(f"  Running debater with custom advocate prompt: {advocate_prompt[:50]}...")
            print(f"  Running debater with custom critic prompt: {critic_prompt[:50]}...")
            
            # Wrap synchronous call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.debater.generate_response(
                    query=query,
                    context=context,
                    advocate_system_prompt=advocate_prompt,
                    critic_system_prompt=critic_prompt
                )
            )
            print(f"  Debater result: {response[:250]}...")
            return ("debater", response)
        
        # Build list of tasks to run in parallel
        tasks = []
        for agent_name in middle_agents:
            if agent_name == "predictor":
                tasks.append(run_predictor())
            elif agent_name == "debater":
                tasks.append(run_debater())
        
        # Execute all middle agents in parallel
        results = await asyncio.gather(*tasks)
        
        # Add results to messages in the order they completed
        for agent_name, response in results:
            messages.append(AIMessage(content=f"{response}"))
        
        state["messages"] = messages
        return state

    async def _execute_end_async(self, state: AgentState) -> AgentState:
        """
        Execute the end phase using async with custom prompts
        """
        workflow_plan = state.get("workflow_plan", {})
        end_agents = workflow_plan.get("end", [])
        custom_prompts = state.get("custom_prompts", {})
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        print(f"Executing end phase with {len(end_agents)} agents: {end_agents}")
        
        # End phase must be sequential because:
        # 1. Aggregator must process all middle messages first
        # 2. Refiner needs the aggregated (or last) response to refine
        for agent_name in end_agents:
            print(f"  Running {agent_name} with custom prompt(s)")
            
            if agent_name == "aggregator":
                # Extract content from AI messages for aggregation
                message_contents = [msg.content for msg in messages if isinstance(msg, AIMessage)]
                
                if not message_contents:
                    message_contents = [query]
                
                print(f"  Aggregator processing {len(message_contents)} messages")
                
                # Wrap synchronous call in executor
                loop = asyncio.get_event_loop()
                aggregated_response = await loop.run_in_executor(
                    None,
                    self.aggregator.aggregate_messages,
                    message_contents,
                    query
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
                
                # Get custom prompts for refiner sub-agents
                refiner_prompts = custom_prompts.get("refiner", {})
                critic_prompt = refiner_prompts.get("critic", "")
                editor_prompt = refiner_prompts.get("editor", "")
                
                print(f"    Using custom critic prompt: {critic_prompt[:50]}...")
                print(f"    Using custom editor prompt: {editor_prompt[:50]}...")
                
                # Wrap synchronous call in executor
                loop = asyncio.get_event_loop()
                improved_response = await loop.run_in_executor(
                    None,
                    lambda: self.refiner.generate_response(
                        query=query,
                        current_response=current_response,
                        context=context,
                        critic_system_prompt=critic_prompt,
                        editor_system_prompt=editor_prompt
                    )
                )
                
                messages.append(AIMessage(content=improved_response))
                print(f"  Refiner result: {improved_response[:250]}...")
        
        state["messages"] = messages
        return state

    def generate_response(self, query: str, context: str):
        """Synchronous version - creates new event loop"""
        return asyncio.run(self.generate_response_async(query, context))

    async def generate_response_async(self, query: str, context: str):
        """Fully async version of generate_response"""
        initial_state = {
            "query": query,
            "context": context,
            "messages": [],
            "workflow_plan": {},
            "planner_reasoning": "",
            "custom_prompts": {},
        }
        
        print(f"Starting three-phase async orchestration with custom prompts for query: {query}")
        response = await self.graph.ainvoke(initial_state)

        return {
            "content": response["messages"][-1].content if response["messages"] else "No response generated",
            "workflow_plan": response["workflow_plan"],
            "planner_reasoning": response["planner_reasoning"],
            "custom_prompts": response["custom_prompts"],
            "input_tokens": response.response_metadata.get("input_tokens", 0),
            "output_tokens": response.response_metadata.get("output_tokens", 0),
        }

    def save_workflow_image(self):
        """Save workflow as PNG image"""
        png_data = self.graph.get_graph().draw_mermaid_png()
        
        with open("prompt_and_workflow_orchestration_workflow.png", "wb") as f:
            f.write(png_data)


async def main():
    load_dotenv(find_dotenv())

    planner_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    high_temp_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.8)
    medium_temp_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    low_temp_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

    orchestration_agent = OrchestrationAgent(
        planner_llm=planner_llm,
        high_temp_llm=high_temp_llm,
        medium_temp_llm=medium_temp_llm,
        low_temp_llm=low_temp_llm,
    )
    
    response = await orchestration_agent.generate_response_async(
        query="How many field goals were scored in the first quarter?",
        context="""To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007"""
    )
    print(f"Response:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())