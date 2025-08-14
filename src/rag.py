from typing import List
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class RAG:
    def __init__(self, chunk_size=1000, chunk_overlap=200, max_docs=5):
        self.llm = ChatOllama(model="mistral:7b")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Text splitter for large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_store = None
        self.max_docs = max_docs

    def load_documents(self, documents: List[str]):
        """Load documents, split them into chunks, and create vector store."""
        all_chunks = []
        
        for i, doc in enumerate(documents):
            # Split large documents into smaller chunks
            chunks = self.text_splitter.split_text(doc)
            
            # Convert to Document objects with metadata
            doc_chunks = [
                Document(page_content=chunk, metadata={"doc_id": i, "chunk_id": j}) 
                for j, chunk in enumerate(chunks)
            ]
            all_chunks.extend(doc_chunks)
        
        # Create vector store with all chunks
        self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        print(f"Loaded {len(all_chunks)} chunks from {len(documents)} documents")

    def get_most_relevant_docs(self, query: str) -> List[str]:
        """Find the most relevant document chunks for a given query."""
        if not self.vector_store:
            raise ValueError("Documents not loaded. Call load_documents() first.")
        
        # Get relevant chunks using vector similarity
        docs = self.vector_store.similarity_search(query, k=self.max_docs)
        return [doc.page_content for doc in docs]

    def generate_answer(self, query: str, relevant_docs: List[str]) -> str:
        """Generate an answer for a given query based on relevant documents."""
        # Combine relevant documents into context
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(relevant_docs)])
        
        # Truncate if context is too long
        max_length = 3000
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        prompt = f"Question: {query}\n\nDocuments:\n{context}"
        
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on given documents only."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content