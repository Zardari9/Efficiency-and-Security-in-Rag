import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PRESSConfig:
    """
    Configuration parameters strictly adhering to the PRESS research paper:
    'PRESS: Defending Privacy in Retrieval-Augmented Generation via Embedding Space Shifting' (ICASSP 2025).
    """
    # Embedding Model Parameters
    # Paper uses BGE-base-en-v1.5 (768-dim) / BGE-large-en-v1.5 (1024-dim)
    # Defaulting to all-MiniLM-L6-v2 or bge-base-en-v1.5
    base_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    shifted_model_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "press_shifted_model")
    
    # Training Hyperparameters (Section III-A)
    # Learning rate = 2e-5, Batch size = 64 (or 16 for CPU efficiency), Temperature tau = 0.4
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 5
    temperature: float = 0.4  # tau in Equation (6)
    
    # Dataset & Defense Space (Section III-A)
    # 1% rejection dataset injection acts as defense space
    rejection_ratio: float = 0.01
    sample_ratio: float = 0.01  # Randomly sample 1% of corpus for generating pairs
    
    # Reward Definition (Section II-B)
    kl_divergence_penalty: float = 0.1  # eta in Equation (4)
    
    # Evaluation Metrics (Section III-B)
    top_k: int = 5  # Recall@5, nDCG@5
    repeat_prompt_token_threshold: int = 10  # 10 tokens threshold for attack success
    
    # Vector Database Settings
    collection_name: str = "press_rag_knowledge"
    qdrant_url: Optional[str] = os.getenv("QDRANT_URL")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    use_in_memory_qdrant: bool = True  # Fallback to local memory if credentials not provided
    
    # LLM Settings (Section III-A)
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    groq_model: str = "llama3-8b-8192"  # or openai/gpt-oss-120b
