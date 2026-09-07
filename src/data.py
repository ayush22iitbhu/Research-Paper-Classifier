import os
import pickle
import faiss
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from src.config import Config
from src.model import ProductionMLP

class ArtifactLoader:
    """Loads and encapsulates all serialized binaries and model weights."""
    def __init__(self, cfg: Config = Config):
        self.cfg = cfg
        print(f"Loading artifacts onto {self.cfg.DEVICE}...")

        # 1. Encoders
        self.st_model = SentenceTransformer(self.cfg.BI_ENCODER_PATH, device=self.cfg.DEVICE)
        self.cross_encoder = CrossEncoder(self.cfg.CROSS_ENCODER_NAME, device=self.cfg.DEVICE)

        # 2. Sparse & Dense Indices
        self.faiss_index = faiss.read_index(self.cfg.FAISS_PATH)
        with open(self.cfg.BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25_model"]
            self.corpus_titles = bm25_data["corpus_titles"]
            self.taxonomy = bm25_data["arxiv_taxonomy"]

        # 3. Supervised MLP Head
        checkpoint = torch.load(self.cfg.MLP_WEIGHTS_PATH, map_location=self.cfg.DEVICE)
        self.mlp = ProductionMLP(checkpoint["input_dim"], checkpoint["num_classes"]).to(self.cfg.DEVICE)
        self.mlp.load_state_dict(checkpoint["state_dict"])
        self.mlp.eval()

        with open(self.cfg.MLB_PATH, "rb") as f:
            self.mlb = pickle.load(f)

        # 4. Zero-Shot Tail Embeddings
        with open(self.cfg.RARE_CATEGORIES_PATH, "rb") as f:
            rare_data = pickle.load(f)
            self.rare_codes = rare_data["rare_codes"]
            self.rare_descriptions = rare_data["rare_descriptions"]
            self.rare_embeddings = torch.tensor(
                rare_data["rare_category_embeddings"], 
                dtype=torch.float32, 
                device=self.cfg.DEVICE
            )
        print("All artifacts successfully loaded.\n")