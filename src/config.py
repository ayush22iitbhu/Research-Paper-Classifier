import os
import re
import torch


class Config:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    # Raw Dataset Paths (used by train.py)
    PAPERS_CSV = os.path.join(DATA_DIR, "arxiv_data_210930-054931.csv")
    TAXONOMY_CSV = os.path.join(DATA_DIR, "arxiv.csv")

    # Serialized Artifact Paths
    BI_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "bi_encoder")
    FAISS_PATH = os.path.join(ARTIFACTS_DIR, "faiss_hnsw.index")
    BM25_PATH = os.path.join(ARTIFACTS_DIR, "bm25_corpus.pkl")
    MLP_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "head_mlp_weights.pt")
    MLB_PATH = os.path.join(ARTIFACTS_DIR, "mlb_head.pkl")
    RARE_CATEGORIES_PATH = os.path.join(ARTIFACTS_DIR, "rare_categories.pkl")

    # Pretrained Models
    BASE_BI_ENCODER = "all-MiniLM-L6-v2"
    CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Taxonomy Sanitization Rules
    VALID_STANDALONE = {"quant-ph", "hep-th", "hep-ph", "hep-lat", "hep-ex", "gr-qc"}
    ARXIV_MODERN_REGEX = re.compile(r"^[a-zA-Z\-]+\.[a-zA-Z\-]+$")

    # Training & Inference Hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    TEST_SIZE = 0.10
    TOP_K_HEAD_CLASSES = 50
    MNRL_EPOCHS = 1
    MNRL_BATCH_SIZE = 32
    MNRL_LR = 2e-5
    MLP_EPOCHS = 10
    MLP_BATCH_SIZE = 128
    MLP_LR = 1e-3

    # Inference Thresholds
    MLP_THRESHOLD = 0.50
    TAIL_THRESHOLD = 0.40
    RRF_K = 60
    POOL_SIZE = 20