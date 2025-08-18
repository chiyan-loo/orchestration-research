from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from vector_store import vector_manager

class AgentState(TypedDict):
    query: str
    relevant_docs: List
    response: str
    is_valid: bool
    final_answer: str
    retry_count: int
    feedback: str

class SelfRAG:
    def __init__(self):
        self.llm = ChatOllama(model="mistral:7b")
        self.max_retries = 2
        self.workflow = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self._retrieve_docs)
        workflow.add_node("generate", self._generate_answer)
        workflow.add_node("validate", self._validate_answer)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        
        # Conditional edge: retry generation or finish
        workflow.add_conditional_edges(
            "validate",
            self._should_retry,
            {
                "retry_generate": "generate", 
                "finish": END
            }
        )
        
        
        return workflow.compile()
    
    def _retrieve_docs(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents from vector database"""
        relevant_docs = vector_manager.search(state["query"], k=3)
        state["relevant_docs"] = relevant_docs
        state["retry_count"] = 0
        return state
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer using retrieved docs"""
        relevant_doc_content = "\n\n".join([doc.page_content for doc in state["relevant_docs"]])
        
        # Add feedback context if this is a retry
        feedback_context = ""
        if state.get("feedback") and state.get("retry_count", 0) > 0:
            feedback_context = f"\n\nPrevious feedback: {state['feedback']}\nPlease address this feedback in your response."
        
        prompt = f"""question: {state['query']}\n\nDocuments: {relevant_doc_content}{feedback_context}
        
        Answer strictly based on the provided documents. Do not add information not found in the documents."""
        
        messages = [
            ("system", "You are a helpful assistant that answers questions based ONLY on given documents. Do not hallucinate or add external knowledge."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        state["response"] = ai_msg.content
        return state
    
    def _validate_answer(self, state: AgentState) -> AgentState:
        """Validate if the answer is good and provide detailed feedback"""
        validation_prompt = f"""
        Query: {state['query']}
        Answer: {state['response']}
        Available Documents: {[doc.page_content + "..." for doc in state['relevant_docs']]}
        
        Evaluate this answer and respond in this exact format:
        VALID: yes/no
        ISSUE: [if not valid, specify: "hallucination", "irrelevant_docs", or "poor_answer"]
        FEEDBACK: [specific feedback on what needs to be improved]
        """
        
        messages = [
            ("system", "You are a strict validator. Check if the answer is supported by the documents and relevant to the query."),
            ("human", validation_prompt),
        ]
        
        validation_response = self.llm.invoke(messages)
        response_text = validation_response.content
        
        # Parse validation response
        is_valid = "VALID: yes" in response_text.lower()
        
        # Extract feedback
        feedback = ""
        if "FEEDBACK:" in response_text:
            feedback = response_text.split("FEEDBACK:")[-1].strip()
        
        if is_valid:
            state["final_answer"] = state["response"]
        else:
            state["final_answer"] = "Processing..."
            state["retry_count"] = state.get("retry_count", 0) + 1
            
        state["is_valid"] = is_valid
        state["feedback"] = feedback
        return state
    
    def _should_retry(self, state: AgentState) -> str:
        """Decide whether to retry generation"""
        if state["is_valid"] or state.get("retry_count", 0) >= self.max_retries:
            return "finish"
        
        return "retry_generate"
    
    def generate_response(self, question: str):
        """Query the RAG system"""
        result = self.workflow.invoke({"query": question})
        
        return {
            "content": result["final_answer"],
            "relevant_docs": result["relevant_docs"],
            "is_valid": result["is_valid"],
            "retry_count": result.get("retry_count", 0),
            "feedback": result.get("feedback", "")
        }
    
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        try:
            png_data = self.workflow.get_graph().draw_mermaid_png()
            
            with open("self_rag_workflow.png", "wb") as f:
                f.write(png_data)
            
            print("Workflow saved as therapy_workflow.png")
            
        except Exception as e:
            print(f"Install graphviz: pip install graphviz")


if __name__ == "__main__":
    validation_rag = SelfRAG()

    response = validation_rag.generate_response("Hi")
    print(response["content"])

    validation_rag.save_workflow_image()