import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

class ShiftingTripletDataset(Dataset):
    """
    Dataset for D_emb = {(x^p_i, y^q_i, y^{q'}_i)} from Section II-D.
    """
    def __init__(self, triplets: List[Dict[str, Any]]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": item["negative"]
        }

class EmbeddingShifter:
    """
    Implements Section II-D (Embedding Shifting Fine-tuning):
    Minimizes the InfoNCE loss Lemb on dataset Demb:
    
    Lemb = sum_{i=1}^{Nemb} -log [ e^(<ep, eq> / tau) / ( e^(<ep, eq> / tau) + sum_{q' in Q'} e^(<ep, eq'> / tau) ) ]
    
    where ep, eq, eq' are text embeddings for query x^p, positive target y^q, and negative target y^q'.
    tau = 0.4 is temperature parameter (Section III-A).
    Learning rate = 2e-5 (Section III-A).
    """

    def __init__(self, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PRESS] Loading base embedding model '{model_name_or_path}' on device '{self.device}'...")
        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        self.model_name_or_path = model_name_or_path

    def compute_infonce_loss(self, ep: torch.Tensor, eq: torch.Tensor, eq_prime: torch.Tensor, temperature: float = 0.4) -> torch.Tensor:
        """
        Calculates Lemb from Equation (6).
        ep: (batch_size, dim) query embeddings
        eq: (batch_size, dim) positive target embeddings
        eq_prime: (batch_size, dim) negative target embeddings
        """
        # L2 normalize embeddings for cosine similarity <ep, eq>
        ep_norm = F.normalize(ep, p=2, dim=1)
        eq_norm = F.normalize(eq, p=2, dim=1)
        eq_prime_norm = F.normalize(eq_prime, p=2, dim=1)

        # Dot products divided by temperature tau
        sim_pos = torch.sum(ep_norm * eq_norm, dim=-1, keepdim=True) / temperature # (batch_size, 1)
        sim_neg = torch.sum(ep_norm * eq_prime_norm, dim=-1, keepdim=True) / temperature # (batch_size, 1)

        # In-batch cross-entropy: positive target is class 0
        logits = torch.cat([sim_pos, sim_neg], dim=1) # (batch_size, 2)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=ep.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def fine_tune(
        self,
        triplets: List[Dict[str, Any]],
        output_dir: str,
        epochs: int = 5,
        lr: float = 2e-5,
        batch_size: int = 16,
        temperature: float = 0.4
    ):
        """
        Fine-tunes the embedding model to shift sensitive queries towards safe target space.
        Parameters strictly follow Section III-A (lr=2e-5, tau=0.4).
        """
        print(f"[PRESS] Starting Embedding Shifting Fine-Tuning with {len(triplets)} triplets...")
        self.model.train()

        # Extract underlying PyTorch transformer module
        modules = list(self.model.children())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        dataset = ShiftingTripletDataset(triplets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()

                queries = batch["query"]
                positives = batch["positive"]
                negatives = batch["negative"]

                # Tokenize and forward through transformer
                features_p = self.model.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
                features_p = {k: v.to(self.device) for k, v in features_p.items()}
                ep = self.model(features_p)["sentence_embedding"]

                features_q = self.model.tokenizer(positives, padding=True, truncation=True, return_tensors="pt")
                features_q = {k: v.to(self.device) for k, v in features_q.items()}
                eq = self.model(features_q)["sentence_embedding"]

                features_q_prime = self.model.tokenizer(negatives, padding=True, truncation=True, return_tensors="pt")
                features_q_prime = {k: v.to(self.device) for k, v in features_q_prime.items()}
                eq_prime = self.model(features_q_prime)["sentence_embedding"]

                loss = self.compute_infonce_loss(ep, eq, eq_prime, temperature=temperature)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(dataloader))
            print(f"[PRESS] Epoch {epoch + 1}/{epochs} - Shifting Loss: {avg_loss:.4f}")

        # Save privacy-secured shifted embedding model
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)
        print(f"[PRESS] Shifted Embedding Model successfully saved to: {output_dir}")
        self.model.eval()

    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encodes texts into vectors using the embedding model.
        """
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=normalize)
        return embeddings
