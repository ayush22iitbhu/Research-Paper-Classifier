import os
import pickle
from typing import Dict, List
import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from src.config import Config
from src.model import ProductionMLP


def build_and_save_artifacts(
    st_model: SentenceTransformer,
    mlp_model: ProductionMLP,
    mlb: MultiLabelBinarizer,
    sentences: List[str],
    title_embeddings: np.ndarray,
    arxiv_taxonomy: Dict[str, str],
    cfg: Config = Config,
):
    os.makedirs(cfg.ARTIFACTS_DIR, exist_ok=True)

    # Rare Categories (Zero-Shot Tail)
    top_set = set(mlb.classes_)
    rare_codes = [code for code in arxiv_taxonomy.keys() if code not in top_set]
    rare_descriptions = [arxiv_taxonomy[code] for code in rare_codes]
    rare_embeddings = st_model.encode(rare_descriptions, convert_to_tensor=True, device=cfg.DEVICE)

    with open(cfg.RARE_CATEGORIES_PATH, "wb") as f:
        pickle.dump(
            {
                "rare_codes": rare_codes,
                "rare_descriptions": rare_descriptions,
                "rare_category_embeddings": rare_embeddings.cpu().numpy(),
            },
            f,
        )

    # BM25 Sparse Index & Taxonomy
    tokenized_corpus = [doc.lower().split() for doc in sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(cfg.BM25_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25_model": bm25,
                "corpus_titles": sentences,
                "arxiv_taxonomy": arxiv_taxonomy,
            },
            f,
        )

    # FAISS HNSW Dense Index
    dimension = title_embeddings.shape[1]
    faiss_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    faiss_index.hnsw.efSearch = 64
    faiss_index.add(np.ascontiguousarray(title_embeddings, dtype=np.float32))
    faiss.write_index(faiss_index, cfg.FAISS_PATH)

    # PyTorch MLP Head & MultiLabelBinarizer
    torch.save(
        {
            "state_dict": mlp_model.state_dict(),
            "input_dim": 384,
            "num_classes": len(mlb.classes_),
            "classes": mlb.classes_,
        },
        cfg.MLP_WEIGHTS_PATH,
    )

    with open(cfg.MLB_PATH, "wb") as f:
        pickle.dump(mlb, f)

    return rare_codes, rare_embeddings