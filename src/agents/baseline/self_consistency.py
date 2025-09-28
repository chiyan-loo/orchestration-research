import asyncio
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import find_dotenv, load_dotenv
import os


class AgentState(TypedDict):
    query: str
    context: str
    predictions: List[str]
    aggregated: str

class SelfConsistency:
    def __init__(
        self, 
        predictor_llm: BaseLanguageModel,
        aggregator_llm: BaseLanguageModel,
        num_predictors: int = 3
    ):
        """
        Initialize SelfConsistency with custom LLM objects.
        
        Args:
            predictor_llm: LLM object to use for all predictors
            aggregator_llm: LLM object for aggregation
            num_predictors: Number of predictor instances to create
        """
        self.predictor_llms = [predictor_llm for _ in range(num_predictors)]
        self.aggregator_llm = aggregator_llm
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the workflow graph"""
        workflow = StateGraph(AgentState)
        workflow.add_node("predictors", self._predictors_step)
        workflow.add_node("aggregate", self._aggregate_step)
        workflow.add_edge(START, "predictors")
        workflow.add_edge("predictors", "aggregate")
        workflow.add_edge("aggregate", END)
        return workflow.compile()
    
    async def _run_predictor(self, query: str, context: str, llm: BaseLanguageModel) -> str:
        """Run a single predictor asynchronously"""
        system_prompt = f"""Answer the query clearly and concisely by using the following context. Only return the final answer, no explanations.

Context: {context if context else "No specific context provided"}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        # Check if LLM supports async invoke
        if hasattr(llm, 'ainvoke'):
            response = await llm.ainvoke(messages)
        else:
            # Fallback to sync invoke in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, llm.invoke, messages)
        
        return response.content.strip()

    async def _predictors_step(self, state: AgentState) -> AgentState:
        """Run all predictors asynchronously"""
        tasks = [
            self._run_predictor(state["query"], state["context"], llm)
            for llm in self.predictor_llms
        ]
        predictions = await asyncio.gather(*tasks)
        state["predictions"] = predictions
        return state

    async def _aggregate_step(self, state: AgentState) -> AgentState:
        """Aggregate predictions into final answer"""
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
        
        # Check if LLM supports async invoke
        if hasattr(self.aggregator_llm, 'ainvoke'):
            response = await self.aggregator_llm.ainvoke(messages)
        else:
            # Fallback to sync invoke in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.aggregator_llm.invoke, messages)
        
        state["aggregated"] = response.content.strip()
        return state

    async def generate_response(self, query: str, context: str = "") -> str:
        """Generate response asynchronously"""
        state = {
            "query": query,
            "context": context,
            "predictions": [],
            "aggregated": ""
        }
        # Run predictors async, then aggregate
        state = await self._predictors_step(state)
        state = await self._aggregate_step(state)
        return state["aggregated"]


# ---- Example Usage ----
if __name__ == "__main__":
    async def main():
        load_dotenv(find_dotenv())

        predictor_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="x-ai/grok-4-fast:free",
            temperature=0.8
        )
        aggregator_llm = ChatOllama(model="mistral:7b", temperature=0.2)
        
        workflow = SelfConsistency(
            predictor_llm=predictor_llm,
            aggregator_llm=aggregator_llm,
            num_predictors=3
        )
        
        query = "What is the capital of France?"
        context = "France is a country in Western Europe."
        
        final_answer = await workflow.generate_response(query, context)
        print("FINAL ANSWER:", final_answer)
    
    asyncio.run(main())