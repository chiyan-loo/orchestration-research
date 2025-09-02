from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


class Predictor:
    def __init__(self, model: str):
        self.llm = ChatOllama(model=model)
    
    def respond(self, query: str, context: str = "") -> str:
        """
        Generate a response to a query with optional context
        
        Args:
            query: The question or prompt to respond to
            context: Optional context information
            
        Returns:
            Generated response string
        """
        system_prompt = f"""Answer the query accurately and concisely.
        
        Context: {context if context else "None"}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()


# Example usage
if __name__ == "__main__":
    agent = Predictor(model="mistral:7b")
    
    query = "What is the capital of France?"
    context = "France is a country in Western Europe."
    
    result = agent.respond(query, context)
    print(f"Query: {query}")
    print(f"Response: {result}")