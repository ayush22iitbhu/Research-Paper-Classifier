from typing import Any, Dict, List
import torch
from sentence_transformers import util
from src.data import ArtifactLoader

class DualTierClassifier:
    def __init__(self, loader: ArtifactLoader):
        self.loader = loader
        self.cfg = loader.cfg

    @torch.no_grad()
    def predict(self, abstract: str) -> List[Dict[str, Any]]:
        abstract_emb = self.loader.st_model.encode(
            abstract, convert_to_tensor=True, device=self.cfg.DEVICE
        )
        predictions = []

        # Head Branch (Supervised MLP)
        logits = self.loader.mlp(abstract_emb.unsqueeze(0))
        probs = torch.sigmoid(logits).squeeze()
        for idx in (probs >= self.cfg.MLP_THRESHOLD).nonzero(as_tuple=True)[0]:
            code = self.loader.mlb.classes_[idx.item()]
            predictions.append({
                "code": code,
                "name": self.loader.taxonomy.get(code, code),
                "conf": f"{probs[idx].item() * 100:.1f}%",
                "tier": "Head (Supervised)"
            })

        # Tail Branch (Zero-Shot Semantic Match)
        tail_scores = util.cos_sim(abstract_emb, self.loader.rare_embeddings)[0]
        for idx in (tail_scores >= self.cfg.TAIL_THRESHOLD).nonzero(as_tuple=True)[0]:
            code = self.loader.rare_codes[idx.item()]
            predictions.append({
                "code": code,
                "name": self.loader.rare_descriptions[idx.item()],
                "conf": f"{tail_scores[idx].item() * 100:.1f}%",
                "tier": "Tail (Zero-Shot)"
            })

        predictions.sort(key=lambda x: float(x["conf"].replace("%", "")), reverse=True)
        return predictions