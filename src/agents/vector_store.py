import os
import threading
from pathlib import Path
from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self, persist_directory: str = "../chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        self.lock = threading.Lock()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    
    def load_documents_from_folder(self, folder_path: str):
        """Load all text files from a folder into the vector database"""
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} doesn't exist")
            return
        
        # Load documents
        loader = DirectoryLoader(
            folder_path, 
            glob="**/*.txt",  # Change to "**/*" for all file types
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        if not documents:
            print("No documents found")
            return
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        with self.lock:
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()
        
        print(f"Added {len(chunks)} chunks from {len(documents)} documents")
    
    def search(self, query: str, k: int = 4):
        """Search the vector database"""
        with self.lock:
            return self.vectorstore.similarity_search(query, k=k)

# Global instance
vector_manager = VectorStoreManager("../chroma_db")

# Simple functions to use
def save_files_to_db(data_folder: str = "../data"):
    """Save all files from data folder to vector database"""
    vector_manager.load_documents_from_folder(data_folder)

def search_db(query: str, k: int = 4):
    """Search the vector database"""
    return vector_manager.search(query, k)