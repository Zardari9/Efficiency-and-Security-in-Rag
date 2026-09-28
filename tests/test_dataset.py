import pytest
from src.dataset_builder import PRESSDatasetBuilder

def test_corpus_construction():
    corpus = PRESSDatasetBuilder.get_full_corpus(include_defense_space=True)
    assert len(corpus) > 0
    sensitive_docs = [d for d in corpus if d.get("is_sensitive")]
    defense_anchors = [d for d in corpus if d.get("is_defense_anchor")]
    benign_docs = [d for d in corpus if not d.get("is_sensitive") and not d.get("is_defense_anchor")]

    assert len(sensitive_docs) == 10
    assert len(defense_anchors) == len(PRESSDatasetBuilder.SAFE_REJECTION_ANCHORS)
    assert len(benign_docs) == 5

def test_triplets_generation():
    triplets = PRESSDatasetBuilder.generate_shifting_triplets()
    assert len(triplets) > 0

    # Check pair distribution: 2 targeted, 1 untargeted, 2 safe per record (total 10 records = 50 triplets)
    targeted = [t for t in triplets if t["type"] == "targeted_attack"]
    untargeted = [t for t in triplets if t["type"] == "untargeted_attack"]
    safe = [t for t in triplets if t["type"] == "safe"]

    assert len(targeted) == 20
    assert len(untargeted) == 10
    assert len(safe) == 20

    # Ensure each triplet has query, positive, negative
    for t in triplets:
        assert "query" in t and len(t["query"]) > 0
        assert "positive" in t and len(t["positive"]) > 0
        assert "negative" in t and len(t["negative"]) > 0

def test_evaluation_queries():
    eval_suite = PRESSDatasetBuilder.get_evaluation_query_suite()
    assert len(eval_suite) >= 6
    types = {q["type"] for q in eval_suite}
    assert "targeted" in types
    assert "untargeted" in types
    assert "safe" in types
