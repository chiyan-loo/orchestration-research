from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    query: str
    context: str
    predictions: List[str]
    aggregated: str


class SelfConsistency:
    def __init__(self, model: str, num_predictors: int = 3):
        self.model = model
        self.num_predictors = num_predictors
        self.predictor_llms = [ChatOllama(model=model, temperature=0.7) for _ in range(num_predictors)]
        self.aggregator_llm = ChatOllama(model=model, temperature=0.3)
        self.graph = self._build_graph()

    # Build the workflow graph
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("predictors", self._predictors_step)
        workflow.add_node("aggregate", self._aggregate_step)

        workflow.add_edge(START, "predictors")
        workflow.add_edge("predictors", "aggregate")
        workflow.add_edge("aggregate", END)

        return workflow.compile()
    
    # Run a single predictor
    def _run_predictor(self, query: str, context: str, llm: ChatOllama) -> str:
        system_prompt = f"""Answer the query clearly and concisely by using the following context. Only return the final answer, no explanations.

Context: {context if context else "No specific context provided"}"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        response = llm.invoke(messages)
        return response.content.strip()

    def _predictors_step(self, state: AgentState) -> AgentState:
        predictions = []
        for llm in self.predictor_llms:
            pred = self._run_predictor(state["query"], state["context"], llm)
            predictions.append(pred)
        state["predictions"] = predictions
        return state

    def _aggregate_step(self, state: AgentState) -> AgentState:
        joined_preds = "\n".join([f"- {p}" for p in state["predictions"]])
        prompt = f"""You are an aggregator.
Multiple predictors gave answers to the same query. Only return the aggregated final answer, no explanations.

Query: {state['query']}
Context: {state['context']}
Predictions:
{joined_preds}

Provide a single, concise final answer that best reflects the consensus or most accurate result."""
        messages = [
            SystemMessage(content="You combine multiple predictions into one final answer."),
            HumanMessage(content=prompt)
        ]
        response = self.aggregator_llm.invoke(messages)
        state["aggregated"] = response.content.strip()
        return state

    # Public method
    def generate_response(self, query: str, context: str = "") -> str:
        state = {
            "query": query,
            "context": context,
            "predictions": [],
            "aggregated": ""
        }
        result = self.graph.invoke(state)
        return result["aggregated"]


# ---- Example Usage ----
if __name__ == "__main__":
    workflow = SelfConsistency(model="mistral:7b", num_predictors=3)

    query = "What is the capital of France?"
    context = "France is a country in Western Europe."

    final_answer = workflow.generate_response(query, context)
    print("\n=== FINAL ANSWER ===")
    print(final_answer)
