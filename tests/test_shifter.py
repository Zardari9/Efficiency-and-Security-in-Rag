import pytest
import torch
from src.embedding_shifter import EmbeddingShifter

def test_infonce_loss():
    # Test InfoNCE loss computation directly
    shifter = EmbeddingShifter.__new__(EmbeddingShifter)
    shifter.device = "cpu"

    # ep: query embedding, eq: positive target, eq_prime: negative target
    # Case 1: ep perfectly aligned with eq (positive) and orthogonal to eq_prime (negative)
    ep = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    eq = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    eq_prime = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    loss_aligned = shifter.compute_infonce_loss(ep, eq, eq_prime, temperature=0.4)

    # Case 2: ep aligned with negative target eq_prime instead of positive eq
    loss_misaligned = shifter.compute_infonce_loss(ep, eq_prime, eq, temperature=0.4)

    # Misaligned loss must be significantly higher than aligned loss
    assert loss_misaligned.item() > loss_aligned.item()
    assert loss_aligned.item() >= 0.0

def test_infonce_gradient():
    shifter = EmbeddingShifter.__new__(EmbeddingShifter)
    shifter.device = "cpu"

    ep = torch.randn(4, 16, requires_grad=True)
    eq = torch.randn(4, 16, requires_grad=True)
    eq_prime = torch.randn(4, 16, requires_grad=True)

    loss = shifter.compute_infonce_loss(ep, eq, eq_prime, temperature=0.4)
    loss.backward()

    assert ep.grad is not None
    assert torch.any(ep.grad != 0)
