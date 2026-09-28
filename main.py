import os
import sys
import argparse
from typing import Dict, Any

from src.config import PRESSConfig
from src.dataset_builder import PRESSDatasetBuilder
from src.privacy_classifier import PrivacyBinaryClassifier
from src.embedding_shifter import EmbeddingShifter
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine
from src.evaluation import PRESSEvaluator
from notebooks_and_vis.visualize_shifting import plot_embedding_space_shifting

def print_banner():
    print("=" * 80)
    print(" PRESS: Defending Privacy in Retrieval-Augmented Generation via Embedding Space Shifting")
    print(" ICASSP 2025 Implementation")
    print("=" * 80)

def run_press_pipeline(args):
    print_banner()
    config = PRESSConfig()
    
    # 1. Dataset Construction (Section II-A & Section III-A)
    print("\n[Step 1] Constructing Knowledge Corpus & 1% Defense Space...")
    corpus = PRESSDatasetBuilder.get_full_corpus(include_defense_space=True)
    sensitive_count = sum(1 for d in corpus if d.get("is_sensitive"))
    benign_count = sum(1 for d in corpus if not d.get("is_sensitive") and not d.get("is_defense_anchor"))
    defense_count = sum(1 for d in corpus if d.get("is_defense_anchor"))
    print(f"  - Total Documents: {len(corpus)}")
    print(f"  - Sensitive Documents (Healthcare & Enron Email): {sensitive_count}")
    print(f"  - Benign Public Documents: {benign_count}")
    print(f"  - Injected 1% Defense Rejection Space: {defense_count}")

    # 2. Base Embedding Model & Vector Storage Setup
    print("\n[Step 2] Initializing Base Embedding Model & Vector DB...")
    base_shifter = EmbeddingShifter(model_name_or_path=config.base_embedding_model)
    doc_texts = [d["text"] for d in corpus]
    doc_embeddings = base_shifter.encode(doc_texts)

    vector_store = VectorStoreManager(
        collection_name=config.collection_name,
        vector_dim=doc_embeddings.shape[1],
        qdrant_url=config.qdrant_url,
        qdrant_api_key=config.qdrant_api_key,
        use_in_memory=True if not config.qdrant_url else False
    )
    vector_store.upsert_documents(corpus, doc_embeddings.cpu().numpy() if hasattr(doc_embeddings, "cpu") else doc_embeddings)

    baseline_engine = RAGEngine(
        vector_store=vector_store,
        embedding_model=base_shifter,
        groq_api_key=config.groq_api_key,
        groq_model=config.groq_model
    )

    # 3. Baseline Evaluation (Unprotected)
    print("\n[Step 3] Evaluating Baseline (Unprotected) RAG System...")
    test_queries = PRESSDatasetBuilder.get_evaluation_query_suite()
    baseline_metrics = PRESSEvaluator.evaluate_system(baseline_engine, test_queries, top_k=config.top_k)

    # 4. Generate Shifting Sample Pairs (Section II-A & Section III-A)
    print("\n[Step 4] Generating Shifting Triplet Pairs for Embedding Shifter...")
    triplets = PRESSDatasetBuilder.generate_shifting_triplets()
    print(f"  - Generated {len(triplets)} shifting triplets (Targeted, Untargeted, Safe)")

    # Reward & Privacy Scoring verification (Section II-B)
    classifier = PrivacyBinaryClassifier(eta=config.kl_divergence_penalty)
    print("  - Verifying Privacy Binary Classifier on sample sensitive snippet:")
    sample_text = corpus[0]["text"]
    sample_privacy_score = classifier.predict_privacy_score(sample_text)
    sample_reward = classifier.compute_reward("", sample_text)
    print(f"    Sample: '{sample_text[:60]}...'")
    print(f"    f_private score: {sample_privacy_score:.4f} | Reward R(x,y): {sample_reward:.4f}")

    # 5. Embedding Space Shifting Fine-Tuning (Section II-D, Eq. 6)
    print("\n[Step 5] Training Privacy-Secured Embedding Model via InfoNCE Shifting (Eq. 6)...")
    shifted_shifter = EmbeddingShifter(model_name_or_path=config.base_embedding_model)
    shifted_shifter.fine_tune(
        triplets=triplets,
        output_dir=config.shifted_model_dir,
        epochs=args.epochs,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        temperature=config.temperature
    )

    # Re-index with shifted model
    print("\n[Step 6] Updating Vector Index with Privacy-Secured Embeddings...")
    shifted_doc_embeddings = shifted_shifter.encode(doc_texts)
    vector_store.upsert_documents(corpus, shifted_doc_embeddings.cpu().numpy() if hasattr(shifted_doc_embeddings, "cpu") else shifted_doc_embeddings)

    press_engine = RAGEngine(
        vector_store=vector_store,
        embedding_model=shifted_shifter,
        groq_api_key=config.groq_api_key,
        groq_model=config.groq_model
    )

    # 7. PRESS Evaluation (Protected)
    print("\n[Step 7] Evaluating PRESS Privacy-Secured RAG System...")
    press_metrics = PRESSEvaluator.evaluate_system(press_engine, test_queries, top_k=config.top_k)

    # 8. Display Comparative Results (Matching Tables I, II, III in Paper)
    print("\n" + "=" * 80)
    print(" EXPERIMENTAL EVALUATION RESULTS (Comparison matching ICASSP 2025 Paper)")
    print("=" * 80)

    print("\nTABLE I & II: Retrieval Defense Effectiveness (Recall@5 & nDCG@5)")
    print(f"{'Method':<20} | {'Target Recall':<14} {'Target nDCG':<12} | {'Untarget Recall':<15} {'Untarget nDCG':<13} | {'Safe Recall':<12} {'Safe nDCG':<10}")
    print("-" * 105)
    print(f"{'Baseline (Origin)':<20} | {baseline_metrics['target']['Recall@5']:<14.2f} {baseline_metrics['target']['nDCG@5']:<12.2f} | {baseline_metrics['untarget']['Recall@5']:<15.2f} {baseline_metrics['untarget']['nDCG@5']:<13.2f} | {baseline_metrics['safe']['Recall@5']:<12.2f} {baseline_metrics['safe']['nDCG@5']:<10.2f}")
    print(f"{'PRESS (Ours)':<20} | {press_metrics['target']['Recall@5']:<14.2f} {press_metrics['target']['nDCG@5']:<12.2f} | {press_metrics['untarget']['Recall@5']:<15.2f} {press_metrics['untarget']['nDCG@5']:<13.2f} | {press_metrics['safe']['Recall@5']:<12.2f} {press_metrics['safe']['nDCG@5']:<10.2f}")
    print("-" * 105)

    print("\nTABLE III: Defense Performance & Generation Leakage (Repeat Prompt Failure Count)")
    print(f"{'Method':<20} | {'Target Repeat Prompt':<22} | {'Untarget Repeat Prompt':<24} | {'Time (seconds)':<15}")
    print("-" * 90)
    print(f"{'Baseline (Origin)':<20} | {baseline_metrics['target']['Repeat_Prompt']:<22} | {baseline_metrics['untarget']['Repeat_Prompt']:<24} | {baseline_metrics['elapsed_time_sec']:<15.2f}")
    print(f"{'PRESS (Ours)':<20} | {press_metrics['target']['Repeat_Prompt']:<22} | {press_metrics['untarget']['Repeat_Prompt']:<24} | {press_metrics['elapsed_time_sec']:<15.2f}")
    print("-" * 90)

    # 9. Embedding Space PCA Visualization (Figure 4)
    print("\n[Step 8] Generating PCA Embedding Space Shifting Visualization (Fig. 4)...")
    vis_path = os.path.join(os.path.dirname(__file__), "pca_embedding_space.png")
    plot_embedding_space_shifting(base_shifter, shifted_shifter, test_queries, output_path=vis_path)

    # 10. Interactive Query Mode if requested
    if args.interactive:
        print("\n" + "=" * 80)
        print(" Interactive Mode: Type a query to see Baseline vs PRESS defense in real-time.")
        print(" Type 'exit' or 'quit' to stop.")
        print("=" * 80)
        while True:
            try:
                user_q = input("\nEnter Query: ").strip()
                if not user_q or user_q.lower() in ["exit", "quit"]:
                    break
                print("\n--> [Baseline RAG Retrieval]:")
                base_res = baseline_engine.run_pipeline(user_q, top_k=3)
                for idx, c in enumerate(base_res["retrieved_contexts"]):
                    flag = "[SENSITIVE LEAK]" if c["is_sensitive"] else ("[DEFENSE ANCHOR]" if c["is_defense_anchor"] else "[BENIGN]")
                    print(f"  {idx+1}. {flag} (Score: {c['score']:.4f}) {c['text'][:80]}...")
                print(f"Response: {base_res['response']}")

                print("\n--> [PRESS Secured RAG Retrieval]:")
                press_res = press_engine.run_pipeline(user_q, top_k=3)
                for idx, c in enumerate(press_res["retrieved_contexts"]):
                    flag = "[SENSITIVE LEAK]" if c["is_sensitive"] else ("[DEFENSE ANCHOR]" if c["is_defense_anchor"] else "[BENIGN]")
                    print(f"  {idx+1}. {flag} (Score: {c['score']:.4f}) {c['text'][:80]}...")
                print(f"Response: {press_res['response']}")
            except (KeyboardInterrupt, EOFError):
                break

    print("\n[Done] PRESS execution completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRESS: Privacy-Preserving RAG via Embedding Space Shifting")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs for embedding shifting (default: 5)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prompt to test queries")
    args = parser.parse_args()
    run_press_pipeline(args)
