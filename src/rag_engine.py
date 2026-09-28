import os
from typing import List, Dict, Any, Optional
from src.vector_store import VectorStoreManager

class RAGEngine:
    """
    RAG Generation Engine implementing both:
    1. Normal Generation (Baseline RAG) -> Prone to privacy leakage
    2. Secure Generation with Applying PRESS -> Defends against privacy leakage via shifted embedding
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_model,  # SentenceTransformer or EmbeddingShifter
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama3-8b-8192"
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_model = groq_model
        self.groq_client = None

        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                print(f"[RAGEngine] Groq client initialized with model '{groq_model}'.")
            except Exception as e:
                print(f"[RAGEngine] Note: Groq initialization failed ({e}), using local deterministic generator.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encodes query and retrieves top_k documents from vector store.
        """
        # Encode query using current embedding model (either baseline or PRESS-shifted)
        if hasattr(self.embedding_model, "encode"):
            query_vector = self.embedding_model.encode([query])
            if hasattr(query_vector, "cpu"):
                query_vector = query_vector.cpu().numpy()[0]
            elif isinstance(query_vector, list):
                query_vector = query_vector[0]
            else:
                query_vector = query_vector[0]
        else:
            raise ValueError("Embedding model does not have an encode method")

        results = self.vector_store.search(query_vector, top_k=top_k)
        return results

    def generate_response(self, query: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
        """
        Sends the retrieved context + user query to the LLM (Groq or local generator).
        """
        context_str = "\n".join([f"- {c['text']}" for c in retrieved_contexts])

        prompt = f"""You are a helpful and truthful AI assistant. Answer the user's question using ONLY the provided context below.
If the context states that the information is confidential or restricted, decline to provide sensitive personal details.
If the answer cannot be determined from the context, state that clearly.

Context:
{context_str}

User Question:
{query}

Answer:"""

        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[RAGEngine] Groq API call error: {e}. Falling back to rule-based generation.")

        # Local deterministic generator for testing / offline demo
        # If retrieved context contains safe rejection defense anchor:
        has_defense_anchor = any(c.get("is_defense_anchor") for c in retrieved_contexts)
        has_sensitive_data = any(c.get("is_sensitive") for c in retrieved_contexts)

        if has_defense_anchor and not has_sensitive_data:
            return "I cannot fulfill this request. The requested information is confidential and access is restricted to protect privacy."
        elif has_sensitive_data:
            # Simulates standard LLM behavior when sensitive context is fed to prompt
            first_sens = next(c['text'] for c in retrieved_contexts if c.get("is_sensitive"))
            return f"Based on the retrieved records: {first_sens}"
        else:
            # Normal safe answer
            first_benign = retrieved_contexts[0]['text'] if retrieved_contexts else "No context found."
            return f"According to the policy: {first_benign}"

    def run_pipeline(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline: retrieval -> generation.
        """
        retrieved = self.retrieve(query, top_k=top_k)
        response = self.generate_response(query, retrieved)
        
        return {
            "query": query,
            "retrieved_contexts": retrieved,
            "response": response,
            "retrieved_sensitive": any(r.get("is_sensitive", False) for r in retrieved),
            "retrieved_defense_anchor": any(r.get("is_defense_anchor", False) for r in retrieved)
        }
