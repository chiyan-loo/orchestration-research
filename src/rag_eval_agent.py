from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

load_dotenv()


try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    pdf_loader = PyPDFLoader("llms/Stock_Market_Performance_2024.pdf")
    documents = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents_split = text_splitter.split_documents(documents=documents)

    vector_store = FAISS.from_documents(documents_split, embeddings)
except Exception as e:
    print(f"Error initializing vector store: {e}")



@tool
def search_documents(query: str, k: int = 3) -> str:
    """
    Search through the document Stock Market Performance 2024 collection for relevant information.
    
    Args:
        query: The search query or question
        k: Number of relevant chunks to retrieve (default: 3)
    
    Returns:
        Relevant document chunks as a formatted string
    """

    print(f"search_documents() called with query: '{query}'")

    if vector_store is None:
        return "Error: Vector store not initialized. Please load documents first."
    
    try:
        relevant_docs = vector_store.similarity_search(query, k=k)

        results = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source', 'Unknown source')
            filename = os.path.basename(source) if source != 'Unknown source' else 'Unknown file'
            results.append(f"Source {i} ({filename}):\n{doc.page_content}")
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching vector store: {e}"


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class RagEvalAgent:
    def __init__(self):

        self.tools = [search_documents]
        self.llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(self.tools)

        self.memory = MemorySaver()

        self.graph = self._build_graph()

    def _build_graph(self):
        """Function that builds graph"""
        workflow = StateGraph(AgentState)

        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("tools", ToolNode(tools=self.tools))

        workflow.add_edge(START, "generate_response")
        workflow.add_conditional_edges(
            "generate_response", self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "generate_response")

        return workflow.compile(checkpointer=self.memory)

    def _generate_response(self, state: AgentState) -> AgentState:
        """Function that generates response"""
        system_prompt = SystemMessage(
            content="You are my AI assistant, please answer my query to the best of your ability"
        )
        response = self.llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}
    
    def _should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "continue"
        else:
            return "end"
    
    def get_response(self, user_message: str, session_id: str = "default") -> str:
        thread_config = {"configurable": {"thread_id": session_id}}

        new_message = HumanMessage(content=user_message)

        current_state = self.graph.get_state(thread_config)

        if current_state.values and "messages" in current_state.values:
            existing_messages = current_state.values["messages"]
            all_messages = existing_messages + [new_message]
        else:
            all_messages = [new_message]

        response = self.graph.invoke(
            {"messages": all_messages},
            config=thread_config
        )

        return response["messages"][-1].content
    
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            
            with open("therapy_workflow.png", "wb") as f:
                f.write(png_data)
            
            print("Workflow saved as therapy_workflow.png")
            
        except Exception as e:
            print(f"Install graphviz: pip install graphviz")


if __name__ == "__main__":
    agent = RagEvalAgent()

    while (True):
        user_input = input("You: ")
        print("Agent: ", agent.get_response(user_input))



