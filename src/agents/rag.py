from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from vector_store import vector_manager

class RAG:
    def __init__(self):
        self.llm = ChatOllama(model="mistral:7b")

    def generate_response(self, query):
        """Generate an answer for a given query using the vector database."""
        # Get most relevant docs from vector database
        relevant_docs = vector_manager.search(query, k=3)
        
        # Combine the document content
        relevant_doc_content = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"question: {query}\n\nDocuments: {relevant_doc_content}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        return {"content": ai_msg.content, "relevant_docs": relevant_docs}