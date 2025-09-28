from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.base import BaseLanguageModel

class DebateState(TypedDict):
    query: str
    context: str
    round: int
    max_rounds: int
    advocate_messages: List[str]
    critic_messages: List[str]
    final_synthesis: str
    advocate_system_prompt: str
    critic_system_prompt: str


class Debater:
    def __init__(self, synthesizer_llm: BaseLanguageModel, predictor_llm: BaseLanguageModel, max_rounds: int = 2):
        self.advocate_llm = predictor_llm
        self.critic_llm = predictor_llm
        self.synthesizer_llm = synthesizer_llm
        self.max_rounds = max_rounds
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(DebateState)
        
        workflow.add_node("debate_round", self._run_debate_round)
        workflow.add_node("synthesize", self._synthesize_final)
        
        workflow.add_edge(START, "debate_round")
        workflow.add_conditional_edges(
            "debate_round",
            self._should_continue,
            {
                "continue": "debate_round",
                "end": "synthesize"
            }
        )
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _run_advocate(self, state: DebateState) -> str:
        # Use custom system prompt if provided, otherwise use default
        system_prompt = state.get("advocate_system_prompt") or "You are an Advocate. Present strong supporting arguments and evidence."
        
        if state["round"] == 1:
            prompt = f"""Present your approach to addressing this query using the available context.
            
            Context: {state['context']}
            Query: {state['query']}
            
            Develop your analysis and conclusions in 1-2 paragraphs."""
        else:
            recent_messages = self._get_recent_context(state)
            prompt = f"""Continue developing your approach to this query.
            
            Query: {state['query']}
            Previous discussion: {recent_messages}
            
            Round {state['round']}: Build upon the discussion and refine your analysis."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.advocate_llm.invoke(messages)
        return response.content.strip()
    
    def _run_critic(self, state: DebateState) -> str:
        # Use custom system prompt if provided, otherwise use default
        system_prompt = state.get("critic_system_prompt") or "You are a Critic. Identify potential issues and explore alternative approaches."
        
        if state["round"] == 1:
            prompt = f"""Analyze this query from your perspective using the available context.
            
            Context: {state['context']}
            Query: {state['query']}
            
            Develop your analysis and conclusions in 1-2 paragraphs."""
        else:
            recent_messages = self._get_recent_context(state)
            prompt = f"""Continue your analysis of this query.
            
            Query: {state['query']}
            Previous discussion: {recent_messages}
            
            Round {state['round']}: Further develop your perspective and analysis."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.critic_llm.invoke(messages)
        return response.content.strip()
    
    def _get_recent_context(self, state: DebateState) -> str:
        context_parts = []
        
        if state["advocate_messages"]:
            context_parts.append(f"Advocate: {state['advocate_messages'][-1]}")
        if state["critic_messages"]:
            context_parts.append(f"Critic: {state['critic_messages'][-1]}")
        
        return "\n\n".join(context_parts)
    
    def _run_debate_round(self, state: DebateState) -> DebateState:
        print(f"\n=== Round {state['round']} ===")
        
        # Run advocate first
        advocate_response = self._run_advocate(state)
        state["advocate_messages"].append(advocate_response)
        print(f"Advocate: {advocate_response[:100]}...")
        
        # Run critic second (can now see advocate's response)
        critic_response = self._run_critic(state)
        state["critic_messages"].append(critic_response)
        print(f"Critic: {critic_response[:100]}...")
        
        state["round"] += 1
        return state
    
    def _should_continue(self, state: DebateState) -> str:
        if state["round"] <= state["max_rounds"]:
            return "continue"
        return "end"
    
    def _synthesize_final(self, state: DebateState) -> DebateState:
        
        # Compile all debate content
        debate_content = ""
        for i in range(len(state["advocate_messages"])):
            debate_content += f"\nRound {i+1}:\n"
            debate_content += f"Advocate: {state['advocate_messages'][i]}\n\n"
            if i < len(state["critic_messages"]):
                debate_content += f"Critic: {state['critic_messages'][i]}\n\n"
        
        prompt = f"""Synthesize this multi-agent debate into a concise, comprehensive response that directly answers the query.

Query: {state['query']}

Debate Content: {debate_content}

Provide a balanced final answer that integrates consistent information from both perspectives."""
        
        messages = [
            SystemMessage(content="You synthesize multi-agent debates into comprehensive responses."),
            HumanMessage(content=prompt)
        ]
        
        response = self.synthesizer_llm.invoke(messages)
        state["final_synthesis"] = response.content.strip()
        
        return state
    
    def generate_response(self, query: str, context: str = "", 
                         advocate_system_prompt: str = "", 
                         critic_system_prompt: str = "") -> str:
        """
        Generate response with optional custom system prompts for sub-agents
        
        Args:
            query: The question or topic to analyze
            context: Additional context information
            advocate_system_prompt: Custom system prompt for advocate sub-agent
            critic_system_prompt: Custom system prompt for critic sub-agent
        """
        response = self.graph.invoke({
            "query": query,
            "context": context,
            "round": 1,
            "max_rounds": self.max_rounds,
            "advocate_messages": [],
            "critic_messages": [],
            "final_synthesis": "",
            "advocate_system_prompt": advocate_system_prompt,
            "critic_system_prompt": critic_system_prompt
        })
        
        return response["final_synthesis"]


if __name__ == "__main__":
    aggregator_llm = ChatOllama(model="mistral:7b", temperature=0.5)
    predictor_llm = ChatOllama(model="mistral:7b", temperature=0.5)

    debater = Debater(synthesizer_llm=aggregator_llm, predictor_llm=predictor_llm, max_rounds=2)
    
    query = "Should AI development be regulated?"
    context = "AI capabilities are advancing rapidly with potential benefits and risks."
    
    # Example with custom prompts
    advocate_prompt = "You are an expert policy analyst who develops comprehensive regulatory frameworks. Focus on practical implementation and stakeholder considerations."
    critic_prompt = "You are a technology innovation researcher who examines potential limitations and unintended consequences of policy proposals."
    
    response = debater.generate_response(
        query=query, 
        context=context,
        advocate_system_prompt=advocate_prompt,
        critic_system_prompt=critic_prompt
    )
    print(f"\n=== FINAL SYNTHESIS ===\n{response}")
    
    # Example without custom prompts (uses defaults)
    response_default = debater.generate_response(query=query, context=context)
    print(f"\n=== DEFAULT PROMPTS SYNTHESIS ===\n{response_default}")