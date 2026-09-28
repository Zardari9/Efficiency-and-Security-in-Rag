import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any

def plot_embedding_space_shifting(
    source_model,
    shifted_model,
    queries_data: List[Dict[str, Any]],
    output_path: str = "pca_embedding_space.png"
):
    """
    Reproduces Figure 4 from the PRESS paper:
    Visualization of embedding space using Principal Component Analysis (PCA) projection.
    (a) Source Embedding Space
    (b) Shifted Embedding Space
    Colors strictly matching Figure 4:
    - Red represents untargeted attack queries
    - Purple represents targeted attack queries
    - Blue represents safe queries
    """
    texts = [q["query"] for q in queries_data]
    types = [q["type"] for q in queries_data]

    # Encode with Source (Original) Model
    emb_source = source_model.encode(texts)
    if hasattr(emb_source, "cpu"):
        emb_source = emb_source.cpu().numpy()

    # Encode with Shifted (PRESS) Model
    emb_shifted = shifted_model.encode(texts)
    if hasattr(emb_shifted, "cpu"):
        emb_shifted = emb_shifted.cpu().numpy()

    # Fit PCA
    pca_source = PCA(n_components=2, random_state=42)
    coords_source = pca_source.fit_transform(emb_source)

    pca_shifted = PCA(n_components=2, random_state=42)
    coords_shifted = pca_shifted.fit_transform(emb_shifted)

    # Color map matching paper Fig 4
    # red = untargeted, purple = targeted, blue = safe
    color_map = {
        "untargeted": "#e74c3c",  # Red
        "targeted": "#8e44ad",    # Purple
        "safe": "#2980b9"         # Blue
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot (a) Source Embedding Space
    ax1 = axes[0]
    for q_type, color in color_map.items():
        indices = [i for i, t in enumerate(types) if t == q_type or (q_type == "untargeted" and "untarget" in t) or (q_type == "targeted" and "target" in t)]
        if indices:
            ax1.scatter(
                coords_source[indices, 0],
                coords_source[indices, 1],
                c=color,
                label=f"{q_type.capitalize()} Queries",
                alpha=0.85,
                s=80,
                edgecolors="none"
            )
    ax1.set_title("(a) Source Embedding Space", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("PCA Component 1", fontsize=11)
    ax1.set_ylabel("PCA Component 2", fontsize=11)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Plot (b) Shifted Embedding Space
    ax2 = axes[1]
    for q_type, color in color_map.items():
        indices = [i for i, t in enumerate(types) if t == q_type or (q_type == "untargeted" and "untarget" in t) or (q_type == "targeted" and "target" in t)]
        if indices:
            ax2.scatter(
                coords_shifted[indices, 0],
                coords_shifted[indices, 1],
                c=color,
                label=f"{q_type.capitalize()} Queries",
                alpha=0.85,
                s=80,
                edgecolors="none"
            )
    ax2.set_title("(b) Shifted Embedding Space (PRESS)", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("PCA Component 1", fontsize=11)
    ax2.set_ylabel("PCA Component 2", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("PRESS: Embedding Space Shifting for Privacy Defense in RAG (Fig. 4 Reproduction)", fontsize=15, y=0.98)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Figure saved successfully to {output_path}")

if __name__ == "__main__":
    from src.dataset_builder import PRESSDatasetBuilder
    from src.embedding_shifter import EmbeddingShifter

    source = EmbeddingShifter("sentence-transformers/all-MiniLM-L6-v2")
    # For standalone test, source is compared
    queries = PRESSDatasetBuilder.get_evaluation_query_suite()
    plot_embedding_space_shifting(source, source, queries, "notebooks_and_vis/pca_embedding_space_preview.png")
