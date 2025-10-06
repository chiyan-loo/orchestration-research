import asyncio
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import find_dotenv, load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI



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
        self.predictor_llms = [predictor_llm for _ in range(num_predictors)]
        self.aggregator_llm = aggregator_llm
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("predictors", self._predictors_step)
        workflow.add_node("aggregate", self._aggregate_step)

        workflow.add_edge(START, "predictors")
        workflow.add_edge("predictors", "aggregate")
        workflow.add_edge("aggregate", END)
        
        return workflow.compile()
    
    async def _run_predictor(self, query: str, context: str, llm: BaseLanguageModel) -> str:
        system_prompt = f"""Answer the query clearly and concisely by using the following context. Only return the final answer, no explanations.

Context: {context if context else "No specific context provided"}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = await llm.ainvoke(messages)
        return response.content.strip()

    async def _predictors_step(self, state: AgentState) -> AgentState:
        tasks = [
            self._run_predictor(state["query"], state["context"], llm)
            for llm in self.predictor_llms
        ]
        predictions = await asyncio.gather(*tasks)
        state["predictions"] = predictions
        return state

    async def _aggregate_step(self, state: AgentState) -> AgentState:
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
        
        response = await self.aggregator_llm.ainvoke(messages)
        state["aggregated"] = response.content.strip()
        return state

    async def generate_response(self, query: str, context: str = "") -> dict:
        initial_state = {
            "query": query,
            "context": context,
            "predictions": [],
            "aggregated": ""
        }
        final_state = await self.graph.ainvoke(initial_state)
        return {
            "content": final_state["aggregated"],
            "predictions": final_state["predictions"]
        }


if __name__ == "__main__":
    async def main():
        load_dotenv(find_dotenv())

        # predictor_llm = ChatOpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=os.getenv("OPENROUTER_API_KEY"),
        #     model="x-ai/grok-4-fast:free",
        #     temperature=0.8
        # )

        predictor_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.8)
        aggregator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        
        workflow = SelfConsistency(
            predictor_llm=predictor_llm,
            aggregator_llm=aggregator_llm,
            num_predictors=3
        )
        
        query = "How many field goals were scored in the first quarter?"
        context = "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007"
        
        final_answer = await workflow.generate_response(query, context)
        print("FINAL ANSWER:", final_answer["content"])

        print("Predictions: ", final_answer["predictions"])
    
    asyncio.run(main())