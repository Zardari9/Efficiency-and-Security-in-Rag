from typing import List, Dict, Any, Optional
import numpy as np

class VectorStoreManager:
    """
    Manages vector storage and retrieval.
    Supports Qdrant (both Qdrant Cloud and Local In-Memory :memory:).
    Includes the 1% injected rejection defense space (Section III-A).
    """

    def __init__(
        self,
        collection_name: str = "press_rag_knowledge",
        vector_dim: int = 384,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        use_in_memory: bool = True
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.use_in_memory = use_in_memory or (not qdrant_url or not qdrant_api_key)

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if self.use_in_memory:
                print(f"[VectorStore] Initializing Qdrant in-memory client...")
                self.client = QdrantClient(location=":memory:")
            else:
                print(f"[VectorStore] Connecting to Qdrant Cloud at {qdrant_url}...")
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

            # Re-create collection
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )
            self.has_qdrant = True
        except ImportError:
            print("[VectorStore] Warning: qdrant-client not installed yet, using local numpy fallback.")
            self.has_qdrant = False
            self.docs = []
            self.embeddings = None

    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Upserts documents and their embeddings (including 1% rejection defense space).
        """
        if self.has_qdrant:
            from qdrant_client.models import PointStruct
            points = []
            for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
                points.append(
                    PointStruct(
                        id=idx + 1,
                        vector=emb.tolist() if isinstance(emb, np.ndarray) else list(emb),
                        payload=doc
                    )
                )
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"[VectorStore] Successfully upserted {len(points)} documents to Qdrant collection '{self.collection_name}'.")
        else:
            self.docs = documents
            self.embeddings = embeddings

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs vector search in the knowledge base.
        Returns top_k retrieved documents with similarity scores and payloads.
        """
        if isinstance(query_vector, np.ndarray):
            q_vec = query_vector.tolist()
        else:
            q_vec = list(query_vector)

        if self.has_qdrant:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                limit=top_k,
                with_payload=True
            ).points

            retrieved = []
            for pt in results:
                retrieved.append({
                    "id": pt.payload.get("id"),
                    "text": pt.payload.get("text"),
                    "score": pt.score,
                    "is_sensitive": pt.payload.get("is_sensitive", False),
                    "is_defense_anchor": pt.payload.get("is_defense_anchor", False),
                    "category": pt.payload.get("category", "")
                })
            return retrieved
        else:
            # Fallback cosine search with numpy
            q = np.array(q_vec)
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q)
            scores = np.dot(self.embeddings, q) / (norms + 1e-9)
            top_indices = np.argsort(scores)[::-1][:top_k]

            retrieved = []
            for idx in top_indices:
                doc = self.docs[idx]
                retrieved.append({
                    "id": doc.get("id"),
                    "text": doc.get("text"),
                    "score": float(scores[idx]),
                    "is_sensitive": doc.get("is_sensitive", False),
                    "is_defense_anchor": doc.get("is_defense_anchor", False),
                    "category": doc.get("category", "")
                })
            return retrieved
