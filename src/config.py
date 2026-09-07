import os
import torch

class Config:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
    
    # Model and Index Paths
    BI_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "bi_encoder")
    FAISS_PATH = os.path.join(ARTIFACTS_DIR, "faiss_hnsw.index")
    BM25_PATH = os.path.join(ARTIFACTS_DIR, "bm25_corpus.pkl")
    MLP_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "head_mlp_weights.pt")
    MLB_PATH = os.path.join(ARTIFACTS_DIR, "mlb_head.pkl")
    RARE_CATEGORIES_PATH = os.path.join(ARTIFACTS_DIR, "rare_categories.pkl")
    CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Search & Inference Parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MLP_THRESHOLD = 0.50
    TAIL_THRESHOLD = 0.40
    RRF_K = 60
    POOL_SIZE = 20