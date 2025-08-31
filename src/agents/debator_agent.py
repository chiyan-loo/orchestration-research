from typing import TypedDict, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


class DebateState(TypedDict):
    query: str
    context: str
    perspectives: List[str]
    debate_rounds: List[Dict[str, str]]
    final_synthesis: str


class DebatorAgent:
    def __init__(self, model: str, num_perspectives: int = 3, debate_rounds: int = 2):
        self.llm = ChatOllama(model=model)
        self.num_perspectives = num_perspectives
        self.debate_rounds = debate_rounds
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(DebateState)
        
        workflow.add_node("generate_perspectives", self._generate_perspectives)
        workflow.add_node("conduct_debate", self._conduct_debate)
        workflow.add_node("synthesize", self._synthesize_final)
        
        workflow.add_edge(START, "generate_perspectives")
        workflow.add_edge("generate_perspectives", "conduct_debate")
        workflow.add_edge("conduct_debate", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _generate_perspectives(self, state: DebateState) -> DebateState:
        """Generate different perspectives on the query"""
        query = state["query"]
        context = state["context"]
        
        perspectives = []
        
        # Define different perspective roles
        perspective_roles = [
            "Advocate: Present the strongest arguments in favor",
            "Critic: Present the strongest counter-arguments and limitations", 
            "Neutral Analyst: Present a balanced, objective analysis",
            "Practical Expert: Focus on real-world implications and applications",
            "Historical Context Provider: Provide historical background and precedents"
        ]
        
        for i in range(min(self.num_perspectives, len(perspective_roles))):
            role = perspective_roles[i]
            
            system_prompt = f"""You are taking the role of a {role.split(':')[0]}. 
            {role.split(':')[1]}
            
            Context: {context if context else "None"}
            
            Provide your perspective in 2-3 paragraphs. Be specific and use evidence where possible."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"What is your perspective on: {query}")
            ]
            
            response = self.llm.invoke(messages)
            perspectives.append({
                "role": role.split(':')[0],
                "content": response.content.strip()
            })
            
            print(f"Generated {role.split(':')[0]} perspective")
        
        state["perspectives"] = [p["content"] for p in perspectives]
        return state
    
    def _conduct_debate(self, state: DebateState) -> DebateState:
        """Conduct rounds of debate between perspectives"""
        query = state["query"]
        perspectives = state["perspectives"]
        context = state["context"]
        
        debate_rounds = []
        
        for round_num in range(self.debate_rounds):
            print(f"Conducting debate round {round_num + 1}")
            
            # Each round, have perspectives respond to each other
            round_responses = []
            
            for i, current_perspective in enumerate(perspectives):
                # Get other perspectives for this agent to respond to
                other_perspectives = [p for j, p in enumerate(perspectives) if j != i]
                other_text = "\n\n".join([f"Perspective {j+1}: {p}" for j, p in enumerate(other_perspectives)])
                
                system_prompt = f"""You are continuing a debate about: {query}

Your original position was: {current_perspective}

Other perspectives presented:
{other_text}

Now provide a response that:
1. Addresses key points from other perspectives
2. Strengthens or refines your position
3. Finds common ground where possible
4. Raises new important considerations

Keep your response focused and 1-2 paragraphs."""
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content="Provide your response to the other perspectives.")
                ]
                
                response = self.llm.invoke(messages)
                round_responses.append(response.content.strip())
            
            debate_rounds.append({
                "round": round_num + 1,
                "responses": round_responses
            })
            
            # Update perspectives with the new responses for next round
            perspectives = round_responses
        
        state["debate_rounds"] = debate_rounds
        return state
    
    def _synthesize_final(self, state: DebateState) -> DebateState:
        """Synthesize the debate into a comprehensive final response"""
        query = state["query"]
        context = state["context"]
        original_perspectives = state["perspectives"]
        debate_rounds = state["debate_rounds"]
        
        # Prepare all debate content for synthesis
        debate_summary = "Original Perspectives:\n"
        for i, perspective in enumerate(original_perspectives):
            debate_summary += f"\nPerspective {i+1}: {perspective}\n"
        
        debate_summary += "\nDebate Evolution:\n"
        for round_data in debate_rounds:
            debate_summary += f"\nRound {round_data['round']}:\n"
            for i, response in enumerate(round_data['responses']):
                debate_summary += f"Response {i+1}: {response}\n"
        
        synthesis_prompt = f"""Based on this multi-perspective debate, create a comprehensive and balanced final response.

        Query: {query}
        Context: {context if context else "None"}

        Debate Content:
        {debate_summary}

        Please synthesize this into a well-structured response that:
        1. Directly answers the original query
        2. Incorporates the strongest points from all perspectives
        3. Acknowledges different viewpoints and their validity
        4. Provides a nuanced, balanced conclusion
        5. Highlights areas of consensus and remaining disagreements

        Structure your response clearly with an introduction, main points, and conclusion."""
        
        messages = [
            SystemMessage(content="You are synthesizing a multi-perspective debate into a comprehensive final answer."),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["final_synthesis"] = response.content.strip()
        
        return state
    
    def generate_response(self, query: str, context: str = "") -> str:
        """Generate a debate-informed response"""
        result = self.graph.invoke({
            "query": query,
            "context": context,
            "perspectives": [],
            "debate_rounds": [],
            "final_synthesis": ""
        })
        
        return result["final_synthesis"]
    
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            
            with open("debator_workflow.png", "wb") as f:
                f.write(png_data)
                        
        except Exception as e:
            print(f"Install graphviz: pip install graphviz")


if __name__ == "__main__":
    debator = DebatorAgent(model="mistral:7b")
    
    query = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France. The city has a population of approximately 2.1 million residents."
    
    result = debator.generate_response(query, context)
    
    print(f"\nFinal Debate Synthesis:\n{result}")