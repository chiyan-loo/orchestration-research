from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class RAG:
    def __init__(self):
        self.llm = ChatOllama(model="mistral:7b")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def generate_response(self, query):
        """Generate an answer for a given query using the vector database."""
        # Get most relevant docs from vector database
        relevant_docs = self.vectorstore.similarity_search(query=query, k=3)
        
        # Combine the document content
        relevant_doc_content = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"question: {query}\n\nDocuments: {relevant_doc_content}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        return {
            "content": ai_msg.content, 
            "relevant_docs": relevant_docs
        }