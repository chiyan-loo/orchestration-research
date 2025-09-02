from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


class DebateState(TypedDict):
    query: str
    context: str
    round: int
    max_rounds: int
    advocate_messages: List[str]
    critic_messages: List[str]
    final_synthesis: str


class Debator:
    def __init__(self, model: str, max_rounds: int = 2):
        self.advocate_llm = ChatOllama(model=model)
        self.critic_llm = ChatOllama(model=model)
        self.synthesizer_llm = ChatOllama(model=model)
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
        if state["round"] == 1:
            prompt = f"""You are an Advocate in a debate. Present strong supporting arguments.
            
            Context: {state['context']}
            Topic: {state['query']}
            
            Present your initial position in 2-3 paragraphs."""
        else:
            recent_messages = self._get_recent_context(state)
            prompt = f"""You are an Advocate. Present strong supporting arguments.
            
            Topic: {state['query']}
            Recent debate: {recent_messages}
            
            Round {state['round']}: Respond to other viewpoints and strengthen your position."""
        
        messages = [
            SystemMessage(content="You are an Advocate. Present strong supporting arguments and evidence."),
            HumanMessage(content=prompt)
        ]
        
        response = self.advocate_llm.invoke(messages)
        return response.content.strip()
    
    def _run_critic(self, state: DebateState) -> str:
        if state["round"] == 1:
            prompt = f"""You are a Critic in a debate. Identify flaws and present counterarguments.
            
            Context: {state['context']}
            Topic: {state['query']}
            
            Present your critical analysis in 2-3 paragraphs."""
        else:
            recent_messages = self._get_recent_context(state)
            prompt = f"""You are a Critic. Identify flaws and present counterarguments.
            
            Topic: {state['query']}
            Recent debate: {recent_messages}
            
            Round {state['round']}: Challenge other viewpoints and present counterarguments."""
        
        messages = [
            SystemMessage(content="You are a Critic. Identify flaws and present counterarguments."),
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

        print(context_parts)
        
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
        
        prompt = f"""Synthesize this multi-agent debate into a comprehensive response.

Query: {state['query']}

Debate Content: {debate_content}

Provide a balanced final answer that integrates both perspectives."""
        
        messages = [
            SystemMessage(content="You synthesize multi-agent debates into comprehensive responses."),
            HumanMessage(content=prompt)
        ]
        
        response = self.synthesizer_llm.invoke(messages)
        state["final_synthesis"] = response.content.strip()
        
        return state
    
    def generate_response(self, query: str, context: str = "") -> str:
        response = self.graph.invoke({
            "query": query,
            "context": context,
            "round": 1,
            "max_rounds": self.max_rounds,
            "advocate_messages": [],
            "critic_messages": [],
            "final_synthesis": ""
        })
        
        return response["final_synthesis"]


if __name__ == "__main__":
    debate = Debator(max_rounds=2)
    
    query = "Should AI development be regulated?"
    context = "AI capabilities are advancing rapidly with potential benefits and risks."
    
    response = debate.generate_response(query, context)
    print(f"\n=== FINAL SYNTHESIS ===\n{response}")