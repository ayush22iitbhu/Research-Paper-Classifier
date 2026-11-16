import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from torch.utils.data import DataLoader, Dataset
from src.config import Config
from src.model import ProductionMLP


class DenseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_bi_encoder(arxiv_data: pd.DataFrame, cfg: Config = Config) -> SentenceTransformer:
    os.makedirs(cfg.ARTIFACTS_DIR, exist_ok=True)
    train_subset = arxiv_data.sample(len(arxiv_data), random_state=cfg.RANDOM_SEED)
    train_hf_data = HFDataset.from_dict({
        "anchor": train_subset["titles"].tolist(),
        "positive": train_subset["abstracts"].tolist(),
    })

    base_model = SentenceTransformer(cfg.BASE_BI_ENCODER)
    train_loss = MultipleNegativesRankingLoss(base_model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir="./arxiv-finetuned-minilm",
        num_train_epochs=cfg.MNRL_EPOCHS,
        per_device_train_batch_size=cfg.MNRL_BATCH_SIZE,
        learning_rate=cfg.MNRL_LR,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=250,
    )

    trainer = SentenceTransformerTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_hf_data,
        loss=train_loss,
    )
    trainer.train()

    base_model.save_pretrained(cfg.BI_ENCODER_PATH)
    return SentenceTransformer(cfg.BI_ENCODER_PATH).to(cfg.DEVICE)


def train_mlp_head(
    X_train: np.ndarray, y_train_encoded: np.ndarray, cfg: Config = Config
) -> ProductionMLP:
    train_loader = DataLoader(
        DenseDataset(X_train, y_train_encoded),
        batch_size=cfg.MLP_BATCH_SIZE,
        shuffle=True,
    )
    model = ProductionMLP(input_dim=X_train.shape[1], num_classes=y_train_encoded.shape[1]).to(cfg.DEVICE)

    class_positives = y_train_encoded.sum(axis=0)
    class_negatives = len(y_train_encoded) - class_positives
    raw_weights = class_negatives / np.maximum(class_positives, 1)
    pos_weights = torch.tensor(np.clip(raw_weights, a_min=1.0, a_max=10.0), dtype=torch.float32).to(cfg.DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.MLP_LR)

    for epoch in range(1, cfg.MLP_EPOCHS + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(cfg.DEVICE)), targets.to(cfg.DEVICE))
            loss.backward()
            optimizer.step()

    return model