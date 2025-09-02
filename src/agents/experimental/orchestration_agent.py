from typing import TypedDict, List, Annotated, Literal,  Union, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from reflector_agent import ReflectorAgent
from aggregator_agent import AggregatorAgent
from debator_agent import DebatorAgent
from summarizer_agent import SummarizerAgent

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    context: str
    loop_history: List[Dict[str, str]]
    next_agent: Literal["aggregator", "debator", "reflector", "summarizer", "end"]

class OrchestrationAgent():
    def __init__(self, model: str):
        self.llm = ChatOllama(
            model=model
        )

        self.reflector_agent = ReflectorAgent(model=model)
        self.aggregator_agent = AggregatorAgent(model=model)
        self.debator_agent = DebatorAgent(model=model)
        self.summarizer_agent = SummarizerAgent(model=model)
        
        self.graph = self._build_graph()

    def _build_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("aggregate", self._aggregate)
        workflow.add_node("debate", self._debate)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("summarize", self._summarize)

        workflow.add_node("orchestrator", self._orchestrate)

        workflow.add_edge(START, "orchestrator")
        workflow.add_conditional_edges(
            "orchestrator", lambda state: state["next_agent"], 
            {
                "end": END,
                "aggregator": "aggregate",
                "debator": "debate",
                "reflector": "reflect",
                "summarizer": "summarize",  # Add this edge
            }
        )

        workflow.add_edge("aggregate", "orchestrator")
        workflow.add_edge("debate", "orchestrator")
        workflow.add_edge("reflect", "orchestrator")
        workflow.add_edge("summarize", "orchestrator")  # Add this edge

        return workflow.compile()

    def _orchestrate(self, state: AgentState) -> AgentState:
        """
        Orchestrator that decides which agent to call next using chain-of-thought reasoning
        """
        messages = state.get("messages", [])
        loop_history = state.get("loop_history", [])
        query = state.get("query", "")
        context = state.get("context", "")
        

        # Get last response
        current_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                current_response = msg.content
                break

        # Count how many times each agent has been used
        agent_counts = {}
        for entry in loop_history:
            agent = entry.get("agent", "")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        print(f"""
    Current query: {query}
    Current context: {context if context else "None"}
    Agents usage history: {[entry["agent"] for entry in loop_history]}
    Agent usage counts: {agent_counts}
    Last agent: {loop_history[-1]["agent"] if loop_history else "None"}
    Total loops: {len(loop_history)}
        """)

        system_prompt = f"""You are an orchestrator deciding which agent to use next in a multi-agent system.

CURRENT SITUATION:
- Query: {query}
- Current response: {current_response if current_response else "None"}
- Context available: {context if context else "None"}
- Agents used so far: {[entry["agent"] for entry in loop_history]}
- Last agent used: {loop_history[-1]["agent"] if loop_history else "None"}
- Agent usage counts: {agent_counts}
- Total loops completed: {len(loop_history)}

AGENT PURPOSES:
- summarizer: Summarizes the context to make it more clearly answer the query
- debator: Generates a response based on multiple perspectives and conducts structured debate for complex/controversial topics
- aggregator: Generates multiple answers and synthesizes consistent information into a single response
- reflector: Reviews and improves existing responses to reduce hallucination and increase quality

DECISION RULES:
- Use 'summarizer' to make the context more clearly support the question, calling after already generating a response is unnecessary 
- Use 'debator' for complex, controversial, or opinion-based topics that benefit from multiple perspectives
- Use 'aggregator' when you have context and need a consistent, fact-based response
- Use 'reflector' to improve quality after generating an initial response
- Limit calling the same agent to under 2-3 times
- Choose 'end' when you have an accurate, high-quality final response

THINK STEP BY STEP:

Step 1 - Check current context:
- Is the context clear and direct? If not, 

Step 2 - Analyze the query type:
- Is this a complex/controversial topic needing multiple perspectives?
- Is this a straightforward question needing a direct answer?

Step 3 - Check current response quality:
- Is there is a current response? Is it accurate and complete?

Step 4 - Review agent usage history:
- Which agents have I already used and how many times?
- Am I reaching agent usage limits?

Step 5 - Determine next best action:
- Should I call the 'summarizer' to make the context more concise before generating a response?
- Should I call an agent to generate a starting response?
- Should I call the 'reflector' to improve an initial response?
- Should I end because I have a good enough answer?

FORMAT YOUR RESPONSE AS:
Reasoning: [Your step-by-step analysis from steps 1-5]
Decision: [summarizer, debator, aggregator, reflector, or end]"""

        orchestrator_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze the situation and decide which agent to use next. Follow the step-by-step thinking process.")
        ]
        
        response = self.llm.invoke(orchestrator_messages)
        response_content = response.content.strip()
        
        # Extract the decision from the response
        decision_line = ""
        reasoning = ""
        
        # Parse the structured response
        lines = response_content.split('\n')
        capture_reasoning = False
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('reasoning:'):
                capture_reasoning = True
                reasoning = line.replace('reasoning:', '').strip()
            elif line.lower().startswith('decision:'):
                decision_line = line.lower().replace('decision:', '').strip()
                capture_reasoning = False
            elif capture_reasoning and line:
                reasoning += " " + line
        
        print(f"Orchestrator raw output: {response.content}")
        
        if "summarizer" in decision_line:
            chosen_agent = "summarizer"
        elif "debator" in decision_line:
            chosen_agent = "debator"
        elif "aggregator" in decision_line:
            chosen_agent = "aggregator"
        elif "reflector" in decision_line:
            # Additional check: only use reflector if we have a previous response to improve
            has_previous_response = any(isinstance(msg, AIMessage) for msg in messages)
            if has_previous_response:
                chosen_agent = "reflector"
            else:
                print("Reflector chosen but no previous response to improve")
                chosen_agent = None
        else:
            chosen_agent = "end"
        
        state["next_agent"] = chosen_agent
        
        return state

    def _summarize(self, state: AgentState) -> AgentState:
        """
        Summarizer agent that refines context to better answer the query
        """
        print("Summarizer called")
        
        # Update loop history
        loop_history = state.get("loop_history", [])
        loop_history.append({"agent": "summarizer"})
        state["loop_history"] = loop_history
        
        query = state.get("query", "")
        context = state.get("context", "")
        
        # Use summarizer agent to refine the context
        refined_context = self.summarizer_agent.generate_response(
            query=query,
            context=context
        )

        print(f"Refined context: {refined_context[:100]}")
                
        # Update the context in the state with the refined version
        state["context"] = refined_context
        
        return state

    def _aggregate(self, state: AgentState) -> AgentState:
        """
        Aggregator agent that generates multiple responses and synthesizes consistent information
        """
        print("Aggregator called")
        
        # Update loop history
        loop_history = state.get("loop_history", [])
        loop_history.append({"agent": "aggregator"})
        state["loop_history"] = loop_history
        
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        # Use aggregator agent to generate consistent response
        aggregated_response = self.aggregator_agent.generate_response(
            query=query,
            context=context
        )
        
        print(f"Aggregator generated response: {aggregated_response[:100]}...")
        
        # Add aggregated response to messages
        messages.append(AIMessage(content=f"{aggregated_response}"))
        state["messages"] = messages
        
        return state
    
    def _debate(self, state: AgentState) -> AgentState:
        """
        Debator agent that generates multiple perspectives and conducts structured debate
        """
        print("Debator called")
        
        # Update loop history
        loop_history = state.get("loop_history", [])
        loop_history.append({"agent": "debator"})
        state["loop_history"] = loop_history
        
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        # Use debator agent to generate multi-perspective response
        debate_response = self.debator_agent.generate_response(
            query=query,
            context=context
        )
        
        print(f"Debator generated response: {debate_response[:100]}...")
        
        # Add debate response to messages
        messages.append(AIMessage(content=f"{debate_response}"))
        state["messages"] = messages
        
        return state
    
    def _reflect(self, state: AgentState) -> AgentState:
        """
        Reflection agent that criticizes the current response and improves it
        """
        print("Reflector called")

        # Update loop history
        loop_history = state.get("loop_history", [])
        loop_history.append({"agent": "reflector"})
        state["loop_history"] = loop_history
        
        
        query = state.get("query", "")
        context = state.get("context", "")
        messages = state.get("messages", [])
        
        # Get the current response (last AI message)
        current_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                current_response = msg.content
                break
        
        if not current_response:
            print("No previous response to reflect on")
            return state
        
        # Use reflector agent to improve the response
        improved_response = self.reflector_agent.generate_response(
            query=query,
            current_response=current_response,
            context=context
        )
        
        print(f"Reflector generated improved response: {improved_response[:100]}...")
        
        # Add improved response to messages
        messages.append(AIMessage(content=f"{improved_response}"))
        state["messages"] = messages
        
        return state

    def generate_response(self, query: str, context: str):
        # Initialize state properly
        initial_state = {
            "query": query,
            "context": context,
            "messages": [],
            "loop_history": [],
            "next_agent": None,
        }
        
        print(f"Starting orchestration for query: {query}")
        response = self.graph.invoke(initial_state)

        return {
            "content": response["messages"][-1].content,
            "loop_history": response["loop_history"],
            "final_context": response["context"]  # Include the potentially refined context
        }

    def save_workflow_image(self):
        """Save workflow as PNG image"""
        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            
            with open("orchestration_workflow.png", "wb") as f:
                f.write(png_data)
                        
        except Exception as e:
            print(f"Install graphviz: pip install graphviz")


if __name__ == "__main__":
    orchestration_agent = OrchestrationAgent(
        model="mistral:7b", 
    )

    orchestration_agent.save_workflow_image()
    
    response = orchestration_agent.generate_response(
        query="Which magazine was started first Arthur's Magazine or First for Women?", 
        context="""{
"sentences": [
[
"Radio City is India's first private FM radio station and was started on 3 July 2001.",
" It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).",
" It plays Hindi, English and regional songs.",
" It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.",
" Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.",
" The Radio station currently plays a mix of Hindi and Regional music.",
" Abraham Thomas is the CEO of the company."
],
[
"Football in Albania existed before the Albanian Football Federation (FSHF) was created.",
" This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) .",
" Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946.",
" In 1932, Albania joined FIFA (during the 12–16 June convention ) And in 1954 she was one of the founding members of UEFA."
],
[
"Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.",
" Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.",
" Echosmith started first as \"Ready Set Go!\"",
" until they signed to Warner Bros.",
" Records in May 2012.",
" They are best known for their hit song \"Cool Kids\", which reached number 13 on the \"Billboard\" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia.",
" The song was Warner Bros.",
" Records' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold.",
" The band's debut album, \"Talking Dreams\", was released on October 8, 2013."
],
[
"Women's colleges in the Southern United States refers to undergraduate, bachelor's degree–granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States.",
" Many started first as girls' seminaries or academies.",
" Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women.",
" Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level."
],
[
"The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum."
],
[
"Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
" Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.",
" In May 1846 it was merged into \"Godey's Lady's Book\"."
],
[
"The 2014–15 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship.",
" Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues.",
" Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014.",
" The regular season included just 12 rounds, where all the teams went to the semifinals.",
" In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk."
],
[
"First for Women is a woman's magazine published by Bauer Media Group in the USA.",
" The magazine was started in 1989.",
" It is based in Englewood Cliffs, New Jersey.",
" In 2011 the circulation of the magazine was 1,310,696 copies."
],
[
"The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California.",
" The fire started as two separate fires on November 15, 2008.",
" The \"Freeway Fire\" started first shortly after 9am with the \"Landfill Fire\" igniting approximately 2 hours later.",
" These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda."
],
[
"William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala.",
" It is most known for their premium jeans.",
" On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line.",
" The label also produces other clothing items such as jackets and tops.",
" The company started first as a denim line, later evolving into a men’s and women’s clothing line."
]
]
}"""
    )
    print(f"Response:\n{response}")