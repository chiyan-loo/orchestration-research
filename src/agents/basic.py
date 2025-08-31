from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

class Basic:
    def __init__(self):
        self.llm = ChatOllama(model="mistral:7b")

    def generate_response(self, query):
        """Generate an answer for a given query"""
        
        prompt = f"{query}"
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on given documents only."),
            HumanMessage(content=prompt),
        ]

        ai_msg = self.llm.invoke(messages)
        
        return ai_msg.content