from typing import TypedDict, List, Annotated, Literal, Union, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, field_validator
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv(find_dotenv())

from reflector import Reflector
from aggregator import Aggregator
from debater import Debater
from predictor import Predictor
from summarizer import Summarizer

class WorkflowPlan(BaseModel):
    reasoning: str
    workflow: List[Union[str, List[str]]]
    
    @field_validator('workflow')
    @classmethod
    def validate_workflow(cls, v):
        valid_agents = {"predictor", "summarizer", "aggregator", "debater", "reflector"}
        
        # Auto-correct common misspellings
        def fix_agent_name(agent_name):
            corrections = {
                "debator": "debater",
            }
            return corrections.get(agent_name, agent_name)
        
        print(f"Validating workflow: {v}")
        
        # Process and fix the workflow
        corrected_workflow = []
        for i, item in enumerate(v):
            if isinstance(item, str):
                corrected_agent = fix_agent_name(item)
                if corrected_agent not in valid_agents:
                    raise ValueError(f"Invalid agent: {item} (corrected to: {corrected_agent})")
                corrected_workflow.append(corrected_agent)
                
            elif isinstance(item, list):
                corrected_parallel = []
                for j, agent in enumerate(item):
                    corrected_agent = fix_agent_name(agent)
                    if corrected_agent not in valid_agents:
                        raise ValueError(f"Invalid agent: {agent} (corrected to: {corrected_agent})")
                    corrected_parallel.append(corrected_agent)
                corrected_workflow.append(corrected_parallel)
        
        print(f"Final corrected workflow: {corrected_workflow}")
        return corrected_workflow

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    context: str
    workflow_plan: List[Union[str, List[str]]]  # Can now contain lists for parallel execution
    current_step: int
    next_block: Literal["aggregator", "debater", "reflector", "summarizer", "predictor", "parallel", "end"] | List[str]
    parallel_results: List[str]  # Store results from parallel execution

class OrchestrationAgent():
    def __init__(self, llm, model: str, max_context_length: int = 4000):

        self.max_context_length = max_context_length

        # self.llm = ChatOllama(
        #     model=model,
        #     temperature=0.7 # For a balance between consistent and creative workflows
        # )

        self.llm = llm

        self.predictor = Predictor(model=model)
        self.reflector = Reflector(model=model)
        self.aggregator = Aggregator(model=model)
        self.debater = Debater(model=model)
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
            "executor", self._choose_next_step, 
            {
                "end": END,
                "predictor": "predict",
                "aggregator": "aggregate",
                "debater": "debate",
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

        truncated_context = f"{context[:self.max_context_length]}..." if len(context) > self.max_context_length else context

        context_length = f"{len(context)}"
        if len(context) > self.max_context_length:
            context_length += " (LONG - requires summarization)"
        elif len(context) < self.max_context_length and len(context) > self.max_context_length // 2:
            context_length += " (MODERATE - summarization unnecessary)"
        else:
            context_length += " (SHORT - do not summarize)"
    
        system_prompt = f"""You are an expert workflow planner for a multi-agent system. You excel at creating sophisticated, multi-layered workflows that maximize quality and comprehensiveness.

QUERY: {query}
CONTEXT: {truncated_context if truncated_context else "No context provided"}
CONTEXT LENGTH: {context_length}

AVAILABLE AGENTS:
- predictor: Simple base agent that directly responds to queries (fastest, factual responses)
- summarizer: Summarizes context to make it more clearly answer the query (for long/noisy context, DO NOT use for factual, direct context)
- aggregator: Analyzes multiple messages and synthesizes consistent information from them (complex, creative responses)
- debater: Generates responses from multiple perspectives for complex/controversial topics
- reflector: Reviews and improves existing responses

WORKFLOW PLANNING PHILOSOPHY:
You STRONGLY PREFER complex, multi-step workflows that leverage parallel processing and quality enhancement. Simple single-agent solutions should be RARE exceptions only for trivial queries.

WORKFLOW PLANNING RULES:
1. For parallel execution, use NESTED LISTS with multiple agents inside: ["step1", ["parallel_agent1", "parallel_agent2"], "step3"]
2. Start with 'summarizer' if context is noisy and needs refinement before other processing
3. Use 'debater' for open-ended questions, long context understanding, multi-hop reasoning, subjective analysis
4. Use 'predictor' for mathematical calculations, discrete reasoning, factual verification, computational problems
5. Use 'reflector' to improve quality of generated responses while trading cost and latency
6. ALWAYS use 'aggregator' after parallel execution to synthesize results
7. Parallel blocks can ONLY include 'predictor' and 'debater' - NO 'reflector', 'aggregator', or 'summarizer' in parallel executors
8. When a workflow includes more than one 'predictor' or 'debater' agent, ALWAYS use 'aggregater' to synthesis outputs 
9. DEFAULT to using 2-4 agents in parallel blocks for comprehensive analysis
10. BIAS towards complex workflows: prefer ["summarizer", ["debater", "debater"], "aggregator"] over simple alternatives

REQUIRED CHAIN OF THOUGHT REASONING:
You must provide detailed step-by-step analysis in this exact format:

REASONING:
1. QUERY ANALYSIS: [Analyze query complexity, ambiguity, domain, and reasoning requirements]
2. CONTEXT EVALUATION: [Assess context length, relevance, and need for summarization]
3. COMPLEXITY ASSESSMENT: [Determine why this query deserves a complex multi-agent approach]
4. PARALLEL STRATEGY: [Explain your parallel execution strategy - why multiple agents and how many]
5. AGENT SELECTION: [Justify each agent choice and their specific roles]
6. WORKFLOW JUSTIFICATION: [Explain each step and why simpler alternatives are insufficient]
7. QUALITY MEASURES: [Explain how aggregation and reflection will improve the final output]

WORKFLOW: [Your final workflow plan]

PREFERRED COMPLEX WORKFLOW PATTERNS:
- [["predictor", "predictor"], "aggregator"]
- [["debater", "debater", "debater", "debater"], "aggregator", "reflector"]
- ["summarizer", ["predictor", "debater", "debater"], "aggregator"]

AVOID SIMPLE PATTERNS like ["predictor"] or ["predictor", "reflector"] unless the query is trivial."""

        structured_llm = self.llm.with_structured_output(WorkflowPlan)
        
        planner_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create an optimal workflow plan for this query.")
        ]
        
        response = structured_llm.invoke(planner_messages)
        
        print(f"Planner reasoning: {response.reasoning}")
        print(f"Parsed workflow: {response.workflow}")
        
        state["workflow_plan"] = response.workflow
        state["current_step"] = 0
        state["parallel_results"] = []
        
        return state
        
    def _choose_next_step(self, state: AgentState) -> str:
        if isinstance(state["next_block"], list):
            return "parallel"
        else:
            return state["next_block"]

    def _execute_workflow(self, state: AgentState) -> AgentState:
        """
        Executes the next step in the planned workflow
        """
        workflow_plan = state.get("workflow_plan", [])
        current_step = state.get("current_step", 0)
        
        if current_step >= len(workflow_plan):
            state["next_block"] = "end"
            return state
        
        next_step = workflow_plan[current_step]
        
        state["next_block"] = next_step
        
        state["current_step"] = current_step + 1
        
        print(f"Executing step {current_step + 1}/{len(workflow_plan)}: {next_step}")
        
        return state

    def _execute_parallel(self, state: AgentState) -> AgentState:
        """
        Execute multiple agents in parallel and store their results
        """
        parallel_agents = state["next_block"]
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        print(f"Executing {len(parallel_agents)} agents in parallel: {parallel_agents}")
        
        parallel_results = []
        
        for agent_name in parallel_agents:
            print(f"  Running {agent_name}")
            
            if agent_name == "predictor":
                result = self.predictor.generate_response(query=query, context=context)
            elif agent_name == "debater":
                result = self.debater.generate_response(query=query, context=context)
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
            print(f"  {agent_name} result: {result}")
        
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
        
        print(f"Predictor generated response: {response}")
        
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

        print(f"Refined context: {refined_context}")
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

        print("aggregator message contents", message_contents)
        
        aggregated_response = self.aggregator.aggregate_messages(
            messages=message_contents,
            query=query
        )
        
        print(f"Aggregator generated response: {aggregated_response}")
        
        messages.append(AIMessage(content=aggregated_response))
        state["messages"] = messages
        
        return state
    
    def _debate(self, state: AgentState) -> AgentState:
        """
        Debater agent that generates multiple perspectives and conducts structured debate
        """
        print("Debater executing")
        
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        debate_response = self.debater.generate_response(
            query=query,
            context=context
        )
        
        print(f"Debater generated response: {debate_response}")
        
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
        
        print(f"Reflector generated improved response: {improved_response}")
        
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
            "next_block": None,
            "parallel_results": [],
        }
        
        print(f"Starting workflow-based orchestration for query: {query}")
        response = self.graph.invoke(initial_state)

        return {
            "content": response["messages"][-1].content,
            "workflow_plan": response["workflow_plan"],
            "final_context": response["context"]
        }

    def save_workflow_image(self):
        """Save workflow as PNG image"""
        png_data = self.graph.get_graph().draw_mermaid_png()
        
        with open("self_plan_workflow.png", "wb") as f:
            f.write(png_data)


if __name__ == "__main__":
    # llm = ChatOllama(
    #     model="mistral:7b",
    #     temperature=0.3 # For a balance between consistent and creative workflows
    # )

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-oss-20b:free",
        temperature=0.7
    )

    orchestration_agent = OrchestrationAgent(
        model="mistral:7b", 
        llm=llm,
    )
    
    response = orchestration_agent.generate_response(
        query='What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?',
        context=""""sentences": [
[
"Meet Corliss Archer, a program from radio's Golden Age, ran from January 7, 1943 to September 30, 1956.",
" Although it was CBS's answer to NBC's popular \"A Date with Judy\", it was also broadcast by NBC in 1948 as a summer replacement for \"The Bob Hope Show\".",
" From October 3, 1952 to June 26, 1953, it aired on ABC, finally returning to CBS.",
" Despite the program's long run, fewer than 24 episodes are known to exist."
],
[
"Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938.",
" As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."
],
[
"Janet Marie Waldo (February 4, 1920 – June 12, 2016) was an American radio and voice actress.",
" She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."
],
[
"Meet Corliss Archer is an American television sitcom that aired on CBS (July 13, 1951 - August 10, 1951) and in syndication via the Ziv Company from April to December 1954.",
" The program was an adaptation of the radio series of the same name, which was based on a series of short stories by F. Hugh Herbert."
],
[
"The post of Lord High Treasurer or Lord Treasurer was an English government position and has been a British government position since the Acts of Union of 1707.",
" A holder of the post would be the third-highest-ranked Great Officer of State, below the Lord High Steward and the Lord High Chancellor."
],
[
"A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale.",
" It stars Shirley Temple in her final starring role as well as her final film appearance.",
" It is a sequel to the 1945 film \"Kiss and Tell\".",
" \"A Kiss for Corliss\" was retitled \"Almost a Bride\" before release and this title appears in the title sequence.",
" The film was released on November 25, 1949, by United Artists."
],
[
"Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.",
" In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys.",
" The parents' bickering about which girl is the worse influence causes more problems than it solves."
],
[
"The office of Secretary of State for Constitutional Affairs was a British Government position, created in 2003.",
" Certain functions of the Lord Chancellor which related to the Lord Chancellor's Department were transferred to the Secretary of State.",
" At a later date further functions were also transferred to the Secretary of State for Constitutional Affairs from the First Secretary of State, a position within the government held by the Deputy Prime Minister."
],
[
"The Village Accountant (variously known as \"Patwari\", \"Talati\", \"Patel\", \"Karnam\", \"Adhikari\", \"Shanbogaru\",\"Patnaik\" etc.) is an administrative government position found in rural parts of the Indian sub-continent.",
" The office and the officeholder are called the \"patwari\" in Telangana, Bengal, North India and in Pakistan while in Sindh it is called \"tapedar\".",
" The position is known as the \"karnam\" in Andhra Pradesh, \"patnaik\" in Orissa or \"adhikari\" in Tamil Nadu, while it is commonly known as the \"talati\" in Karnataka, Gujarat and Maharashtra.",
" The position was known as the \"kulkarni\" in Northern Karnataka and Maharashtra.",
" The position was known as the \"shanbogaru\" in South Karnataka."
],
[
"Charles Craft (May 9, 1902 – September 19, 1968) was an English-born American film and television editor.",
" Born in the county of Hampshire in England on May 9, 1902, Craft would enter the film industry in Hollywood in 1927.",
" The first film he edited was the Universal Pictures silent film, \"Painting the Town\".",
" Over the next 25 years, Craft would edit 90 feature-length films.",
" In the early 1950s he would switch his focus to the small screen, his first show being \"Racket Squad\", from 1951–53, for which he was the main editor, editing 93 of the 98 episodes.",
" He would work on several other series during the 1950s, including \"Meet Corliss Archer\" (1954), \"Science Fiction Theatre\" (1955–56), and \"Highway Patrol\" (1955–57).",
" In the late 1950s and early 1960s he was one of the main editors on \"Sea Hunt\", starring Lloyd Bridges, editing over half of the episodes.",
" His final film work would be editing \"Flipper's New Adventure\" (1964, the sequel to 1963's \"Flipper\".",
" When the film was made into a television series, Craft would begin the editing duties on that show, editing the first 28 episodes before he retired in 1966.",
" Craft died on September 19, 1968 in Los Angeles, California."
]
]"""
    )
    print(f"Response:\n{response}")