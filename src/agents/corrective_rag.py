from typing import TypedDict, List
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class CorrAgentState(TypedDict):
    query: str
    original_query: str
    expanded_queries: List[str]
    relevant_docs: List
    relevance_score: float
    corrective_docs: List
    response: str
    correction_applied: bool
    final_answer: str

class CorrectiveRAG:
    def __init__(self):
        self.llm = ChatOllama(model="mistral:7b")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        self.relevance_threshold = 0.6
        self.max_query_expansions = 3
        self.workflow = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(CorrAgentState)
        
        workflow.add_node("retrieve", self._retrieve_docs)
        workflow.add_node("evaluate_relevance", self._evaluate_relevance)
        workflow.add_node("expand_query", self._expand_query)
        workflow.add_node("retrieve_expanded", self._retrieve_expanded)
        workflow.add_node("knowledge_refinement", self._knowledge_refinement)
        workflow.add_node("generate", self._generate_answer)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "evaluate_relevance")
        
        # Conditional edge: expand query or refine knowledge
        workflow.add_conditional_edges(
            "evaluate_relevance",
            self._should_expand_query,
            {
                "expand": "expand_query",
                "refine": "knowledge_refinement"
            }
        )
        
        workflow.add_edge("expand_query", "retrieve_expanded")
        workflow.add_edge("retrieve_expanded", "knowledge_refinement")
        workflow.add_edge("knowledge_refinement", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _retrieve_docs(self, state: CorrAgentState) -> CorrAgentState:
        """Retrieve relevant documents from vector database"""
        relevant_docs = self.vectorstore.similarity_search_with_score(
            state["query"], k=5
        )
        
        # Separate documents and scores
        docs = [doc for doc, score in relevant_docs]
        scores = [score for doc, score in relevant_docs]
        
        state["relevant_docs"] = docs
        state["original_query"] = state["query"]
        state["relevance_score"] = sum(scores) / len(scores) if scores else 0.0
        state["correction_applied"] = False
        state["expanded_queries"] = []
        
        return state
    
    def _evaluate_relevance(self, state: CorrAgentState) -> CorrAgentState:
        """Evaluate relevance of retrieved documents"""
        if not state["relevant_docs"]:
            state["search_needed"] = True
            return state
            
        evaluation_prompt = f"""
        Query: {state['query']}
        
        Retrieved Documents:
        {chr(10).join([f"Doc {i+1}: {doc.page_content[:200]}..." for i, doc in enumerate(state['relevant_docs'])])}
        
        Rate the overall relevance of these documents to the query on a scale of 0-1:
        - 1.0: Highly relevant, directly answers the query
        - 0.5: Moderately relevant, partially related
        - 0.0: Not relevant, completely unrelated
        
        Respond with only a decimal number between 0 and 1.
        """
        
        messages = [
            ("system", "You are a document relevance evaluator. Provide only a decimal score."),
            ("human", evaluation_prompt),
        ]
        
        relevance_response = self.llm.invoke(messages)
        
        try:
            relevance_score = float(relevance_response.content.strip())
            state["relevance_score"] = max(0.0, min(1.0, relevance_score))
        except ValueError:
            state["relevance_score"] = 0.3  # Default to moderate relevance on parse error
            
        state["search_needed"] = state["relevance_score"] < self.relevance_threshold
        
        return state
    
    def _expand_query(self, state: CorrAgentState) -> CorrAgentState:
        """Generate alternative query formulations for better retrieval"""
        expansion_prompt = f"""
        Original Query: {state['original_query']}
        
        The initial retrieval didn't find sufficiently relevant documents. 
        Generate {self.max_query_expansions} alternative ways to phrase this query that might find better matches:
        
        1. Use synonyms and alternative terminology
        2. Break complex queries into simpler parts
        3. Add related context or domain-specific terms
        4. Rephrase as different question types
        
        Format your response as:
        1. [alternative query 1]
        2. [alternative query 2]  
        3. [alternative query 3]
        """
        
        messages = [
            ("system", "You are a query expansion expert. Generate alternative phrasings to improve document retrieval."),
            ("human", expansion_prompt),
        ]
        
        expansion_response = self.llm.invoke(messages)
        response_text = expansion_response.content
        
        # Parse expanded queries
        expanded_queries = []
        for line in response_text.split('\n'):
            if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, self.max_query_expansions + 1)):
                query = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                if query:
                    expanded_queries.append(query)
        
        state["expanded_queries"] = expanded_queries[:self.max_query_expansions]
        
        return state
    
    def _retrieve_expanded(self, state: CorrAgentState) -> CorrAgentState:
        """Retrieve documents using expanded queries"""
        all_corrective_docs = []
        
        for expanded_query in state["expanded_queries"]:
            try:
                expanded_docs = self.vectorstore.similarity_search_with_score(
                    expanded_query, k=3
                )
                docs = [doc for doc, score in expanded_docs if score > 0.5]  # Filter by score
                all_corrective_docs.extend(docs)
            except Exception as e:
                print(f"Error retrieving for expanded query '{expanded_query}': {e}")
                continue
        
        # Remove duplicates based on content similarity
        unique_docs = []
        for doc in all_corrective_docs:
            is_duplicate = False
            for existing_doc in unique_docs:
                if doc.page_content[:100] == existing_doc.page_content[:100]:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append(doc)
        
        state["corrective_docs"] = unique_docs[:5]  # Limit corrective docs
        state["correction_applied"] = len(unique_docs) > 0
        
        return state
    
    def _knowledge_refinement(self, state: CorrAgentState) -> CorrAgentState:
        """Refine knowledge by combining retrieved docs with corrective information"""
        refined_docs = state["relevant_docs"].copy()
        
        # If we have corrective docs from expanded queries, integrate them
        if state.get("corrective_docs"):
            # Score and rank corrective documents for relevance
            scoring_prompt = f"""
            Original Query: {state['original_query']}
            
            Original Documents Quality: {state['relevance_score']:.2f} (below threshold)
            
            Corrective Documents:
            {chr(10).join([f"Doc {i+1}: {doc.page_content[:200]}..." for i, doc in enumerate(state['corrective_docs'])])}
            
            Rate each corrective document's relevance to the original query (0-1 scale).
            Respond with: Doc1: 0.X, Doc2: 0.Y, etc.
            """
            
            messages = [
                ("system", "You are a document relevance scorer. Rate each document's relevance to the query."),
                ("human", scoring_prompt),
            ]
            
            scoring_response = self.llm.invoke(messages)
            scoring_text = scoring_response.content.lower()
            
            # Add high-scoring corrective docs
            for i, doc in enumerate(state["corrective_docs"]):
                doc_key = f"doc{i+1}:"
                if doc_key in scoring_text:
                    try:
                        score_part = scoring_text.split(doc_key)[1].split(',')[0].split()[0]
                        score = float(score_part.replace('doc', '').replace(':', '').strip())
                        if score > 0.6:  # Only add high-relevance docs
                            refined_docs.append(doc)
                    except (ValueError, IndexError):
                        continue
        
        # If original docs were low relevance, prioritize corrective docs
        if state["relevance_score"] < self.relevance_threshold and state.get("corrective_docs"):
            # Reorder to prioritize corrective information
            original_docs = state["relevant_docs"][:2]  # Keep top 2 original
            corrective_docs = [doc for doc in refined_docs if doc not in original_docs]
            refined_docs = corrective_docs + original_docs
        
        state["relevant_docs"] = refined_docs[:6]  # Limit total docs
        
        return state
    
    def _generate_answer(self, state: CorrAgentState) -> CorrAgentState:
        """Generate answer using refined knowledge"""
        if not state["relevant_docs"]:
            state["final_answer"] = "I couldn't find relevant information to answer your query."
            return state
            
        relevant_doc_content = "\n\n".join([
            f"Source: {doc.page_content}" 
            for doc in state["relevant_docs"]
        ])
        
        correction_note = ""
        if state["correction_applied"]:
            correction_note = "\n\nNote: This answer includes information from expanded query searches to provide more comprehensive results."
        
        prompt = f"""Question: {state['query']}
        
        Available Information:
        {relevant_doc_content}
        
        Based on the available information, provide a comprehensive and accurate answer. 
        If the information is insufficient, clearly state what is missing.
        Prioritize accuracy over completeness.{correction_note}
        """
        
        messages = [
            ("system", "You are a knowledgeable assistant that provides accurate answers based on available information. Be honest about limitations."),
            ("human", prompt),
        ]
        
        ai_msg = self.llm.invoke(messages)
        state["response"] = ai_msg.content
        state["final_answer"] = ai_msg.content
        
        return state
    
    def _should_expand_query(self, state: CorrAgentState) -> str:
        """Decide whether to expand query for better retrieval"""
        if state["relevance_score"] < self.relevance_threshold:
            return "expand"
        return "refine"
    
    def generate_response(self, question: str):
        """Query the Corrective RAG system"""
        result = self.workflow.invoke({"query": question})
        
        return {
            "content": result["final_answer"],
            "relevant_docs": result["relevant_docs"],
            "relevance_score": result["relevance_score"],
            "correction_applied": result["correction_applied"],
            "expanded_queries": result.get("expanded_queries", [])
        }
    
    def save_workflow_image(self):
        """Save workflow as PNG image"""
        try:
            png_data = self.workflow.get_graph().draw_mermaid_png()
            
            with open("corrective_rag_workflow.png", "wb") as f:
                f.write(png_data)
            
            print("Workflow saved as therapy_workflow.png")
            
        except Exception as e:
            print(f"Install graphviz: pip install graphviz")

if __name__ == "__main__":
    corrective_rag = CorrectiveRAG()

    corrective_rag.save_workflow_image()

