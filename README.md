# Research Paper Classifier
It classifies scientific research papers into academic categories and retrieves semantically relevant papers using contrastive learning, multi-label classification, hybrid BM25–FAISS search, and cross-encoder reranking. 
[Google Colab](https://colab.research.google.com/drive/1wr-WZDou0-7SPTISn5Jq3I3isemWcrzy)

### Key Components

1. **Domain-Adaptive Contrastive Alignment (`MNRL`):**
   * Pre-trained `all-MiniLM-L6-v2` fine-tuned on scientific title-abstract pairs using `MultipleNegativesRankingLoss` (InfoNCE).
   * Maps paper titles and abstracts into a shared semantic latent space ($d=384$) optimized for academic prose.

2. **Dual-Tier Extreme Multi-Label Classifier:**
   * **Supervised Head (Top-50 classes):** A PyTorch MLP with dropout regularizers ($0.4$) trained using positive-class frequency weighting (`pos_weight = (N - P) / P`, clipped to $[1.0, 10.0]$) to prevent dominance by high-frequency classes (`cs.CV`, `cs.LG`).
   * **Zero-Shot Cold-Start Tail (100+ classes):** Directly projects abstracts against L2-normalized taxonomy descriptions using cosine similarity, resolving zero-sample cold starts without generating synthetic noise.

3. **Hybrid Search & Ranking Engine:**
   * **Sparse Retrieval:** `BM25Okapi` indexing title n-grams for exact keyword match.
   * **Dense Retrieval:** `FAISS HNSWFlat` (Inner Product, $M=32$, $efSearch=64$) for sub-10ms approximate nearest neighbor semantic search.
   * **Fusion & Reranking:** Reciprocal Rank Fusion ($k=60$) combines dense and sparse candidate pools, followed by a cross-attentive Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for sequence-level scoring.

## Benchmark Performance

The system was evaluated on a held-out test distribution using standard Extreme Multi-Label Ranking ($P@K, R@K, \text{NDCG}@K$) and global multi-label metrics.

### Ranking & Recommendation Performance

| Cutoff ($K$) | Precision@K | Recall@K | NDCG@K | Engineering Significance |
| :---: | :---: | :---: | :---: | :--- |
| **$K = 1$** | **94.36%** | 60.04% | **94.36%** | The primary subject category is correctly predicted on the very first recommendation. |
| **$K = 3$** | 51.34% | **83.91%** | **86.98%** | Over 83% of all true paper categories are retrieved within the top 3 slots. |
| **$K = 5$** | 35.31% | **92.12%** | **90.19%** | **92%+ ground-truth tag coverage** with position-discounted ranking ($\text{NDCG}@5$) exceeding 90%. |

### Core Classification Reliability

| Evaluation Metric | Score | Key Takeaway |
| :--- | :---: | :--- |
| **Weighted F1** | **71.40%** | Demonstrates strong predictive reliability across the true real-world query distribution. |
| **Micro Recall** | **78.90%** | Comprehensive tag capture across the complete 150+ category taxonomy. |
| **Precision@1** | **94.36%** | High-precision single-tag automation with near-zero false positive rate. |
| **Top-5 Category Recall** | **92.12%** | Eliminates manual tag search by surfacing 9 out of 10 intended tags on the primary screen. |
