from ast import literal_eval
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from src.config import Config


def filter_standard_arxiv_terms(terms_list: List[str], cfg: Config = Config) -> List[str]:
    return [
        t.strip()
        for t in terms_list
        if (cfg.ARXIV_MODERN_REGEX.match(t.strip()) or t.strip() in cfg.VALID_STANDALONE)
        and not any(c in t for c in [";", " ", ":", ","])
    ]


def load_and_clean_data(cfg: Config = Config) -> Tuple[pd.DataFrame, Dict[str, str]]:
    arxiv_data = pd.read_csv(cfg.PAPERS_CSV)
    arxiv_data = arxiv_data.drop_duplicates(subset=["titles"]).reset_index(drop=True)
    arxiv_data["terms"] = arxiv_data["terms"].apply(literal_eval)
    arxiv_data["terms"] = arxiv_data["terms"].apply(lambda x: filter_standard_arxiv_terms(x, cfg))
    arxiv_data = arxiv_data[arxiv_data["terms"].map(len) > 0].reset_index(drop=True)

    taxonomy = pd.read_csv(cfg.TAXONOMY_CSV)
    arxiv_taxonomy = dict(zip(taxonomy["code"], taxonomy["name"]))
    return arxiv_data, arxiv_taxonomy


def prepare_head_splits(
    abstract_embeddings: np.ndarray,
    y_raw: List[List[str]],
    cfg: Config = Config,
):
    indices = np.arange(len(abstract_embeddings))
    X_train_full, X_test, idx_train, idx_test = train_test_split(
        abstract_embeddings, indices, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED
    )

    train_raw_terms = [y_raw[i] for i in idx_train]
    all_train_tags = [tag for tags in train_raw_terms for tag in tags]
    top_tags = [tag for tag, _ in Counter(all_train_tags).most_common(cfg.TOP_K_HEAD_CLASSES)]

    y_train_pruned = [[t for t in tags if t in top_tags] for tags in train_raw_terms]
    valid_mask = np.array([len(tags) > 0 for tags in y_train_pruned])

    X_train = X_train_full[valid_mask]
    y_train_clean = [y_train_pruned[i] for i in range(len(y_train_pruned)) if valid_mask[i]]

    mlb = MultiLabelBinarizer(classes=top_tags)
    y_train_encoded = mlb.fit_transform(y_train_clean)

    return X_train, y_train_encoded, X_test, idx_test, mlb