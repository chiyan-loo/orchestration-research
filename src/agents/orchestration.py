from typing import TypedDict, List, Annotated, Literal,  Union, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    context: str
    loop_history: List[Dict]
    next_agent: Union[Literal["aggregator", "debator", "reflector", "retriever"], None]
    query_for_retrieval: str

class OrchestrationAgent():
    def __init__(self):
        self.llm = ChatOllama(
            model="mistral:7b"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        self.graph = self._build_graph()

    def _build_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("aggregate", self._aggregate)
        workflow.add_node("debate", self._debate)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("retrieve", self._retrieve)

        workflow.add_node("orchestrator", self._orchestrate)

        workflow.add_edge(START, "orchestrator")
        workflow.add_conditional_edges(
            "orchestrator", self._choose_agent, 
            {
                "end": END,
                "aggregator": "aggregate",
                "debator": "debate",
                "reflector": "reflect",
                "retriever": "retrieve",   
            }
        )

        workflow.add_edge("aggregate", "orchestrator")
        workflow.add_edge("debate", "orchestrator")
        workflow.add_edge("reflect", "orchestrator")
        workflow.add_edge("retrieve", "orchestrator")

    def _orchestrate(self, state: AgentState) -> AgentState:
        """
        Orchestrator that decides which agent to call next based on conversation state
        """
        messages = state["messages"]
        loop_history = state.get("loop_history", [])

        system_prompt = SystemMessage(content="")

        all_messages = [system_prompt] + messages

        self.llm.invoke(all_messages)
        
        return state

    def _aggregate(self, state: AgentState) -> AgentState:
        return state
    
    def _debate(self, state: AgentState) -> AgentState:
        return state
    
    def _reflect(self, state: AgentState) -> AgentState:
        return state
    
    def _retrieve(self, state: AgentState) -> AgentState:
        """
        Retrieval agent that searches vector database using orchestrator's query
        """
        loop_history = state.get("loop_history", [])
        query = state.get("query_for_retrieval")
        
        # Add to loop history with metadata
        retrieval_entry = {
            "agent": "retriever",
            "query": query
        }
        loop_history.append(retrieval_entry)
        state["loop_history"] = loop_history
        
        # Perform vector search
        docs = self.vectorstore.similarity_search(query, k=3)
        
        # Build context
        if docs:
            context_parts = [f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)]
            context = f"Query: '{query}'\n\n" + "\n\n".join(context_parts)
        else:
            context = f"Query: '{query}'\n\nNo relevant documents found."
        
        state["context"] = context
        state["query_for_retrieval"] = ""
        
        return state
    
    def _choose_agent(self, state: AgentState) -> AgentState:
        if state["next_agent"]:
            return state["next_agent"]
        else:
            return "end"

    def generate_response(self, query: str):
        response = self.graph.invoke({
            "query": query,
        })

        print("Agent Response: ", response)

        return response


