from collections import defaultdict
from typing import List, Tuple
import numpy as np
import torch
from src.data import ArtifactLoader

class HybridSearchEngine:
    def __init__(self, loader: ArtifactLoader):
        self.loader = loader
        self.cfg = loader.cfg

    @torch.no_grad()
    def search(self, title: str, top_k: int = 5) -> List[Tuple[str, float]]:
        pool_size = self.cfg.POOL_SIZE
        rrf_k = self.cfg.RRF_K

        # Dense Search (FAISS)
        q_emb = self.loader.st_model.encode([title], normalize_embeddings=True, convert_to_numpy=True)
        _, faiss_ids = self.loader.faiss_index.search(
            np.ascontiguousarray(q_emb, dtype=np.float32), pool_size
        )
        dense_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(faiss_ids[0])}

        # Sparse Search (BM25)
        tokens = title.lower().split()
        bm25_scores = self.loader.bm25.get_scores(tokens)
        sparse_ids = np.argsort(bm25_scores)[::-1][:pool_size]
        sparse_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(sparse_ids)}

        # RRF Fusion
        candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        rrf_scores = defaultdict(float)
        for doc_id in candidate_ids:
            if doc_id in dense_ranks:
                rrf_scores[doc_id] += 1.0 / (rrf_k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                rrf_scores[doc_id] += 1.0 / (rrf_k + sparse_ranks[doc_id])

        sorted_pool = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:pool_size]
        pool_ids = [doc_id for doc_id, _ in sorted_pool]

        # Cross-Encoder Reranking
        pairs = [[title, self.loader.corpus_titles[cid]] for cid in pool_ids]
        scores = self.loader.cross_encoder.predict(pairs)
        ranked = sorted(zip(pool_ids, scores), key=lambda x: x[1], reverse=True)

        results = []
        for cid, score in ranked:
            if self.loader.corpus_titles[cid].strip().lower() != title.strip().lower():
                results.append((self.loader.corpus_titles[cid], float(score)))
            if len(results) == top_k:
                break
        return results