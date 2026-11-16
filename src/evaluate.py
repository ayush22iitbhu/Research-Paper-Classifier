from typing import List
import numpy as np
import torch
from sentence_transformers import util
from sklearn.metrics import f1_score, ndcg_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from src.config import Config
from src.model import ProductionMLP


def compute_ranking_metrics_at_k(y_true: np.ndarray, y_scores: np.ndarray, k_values=(1, 3, 5)):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    valid_mask = y_true.sum(axis=1) > 0
    y_true_f = y_true[valid_mask]
    y_scores_f = y_scores[valid_mask]

    n_samples = y_true_f.shape[0]
    top_indices = np.argsort(y_scores_f, axis=1)[:, ::-1][:, : max(k_values)]

    results = {}
    for k in k_values:
        top_k = top_indices[:, :k]
        hits_matrix = y_true_f[np.arange(n_samples)[:, None], top_k]

        hits_per_sample = hits_matrix.sum(axis=1)
        actual_per_sample = y_true_f.sum(axis=1)

        prec = hits_per_sample / k
        rec = hits_per_sample / actual_per_sample
        denom = prec + rec
        numer = 2.0 * prec * rec
        f1 = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

        results[f"P@{k}"] = np.mean(prec) * 100
        results[f"R@{k}"] = np.mean(rec) * 100
        results[f"F1@{k}"] = np.mean(f1) * 100
        results[f"NDCG@{k}"] = ndcg_score(y_true_f, y_scores_f, k=k) * 100

    return results


@torch.no_grad()
def evaluate_dual_tier_system(
    model: ProductionMLP,
    mlb: MultiLabelBinarizer,
    rare_codes: List[str],
    rare_embeddings: torch.Tensor,
    X_test: np.ndarray,
    test_raw_terms: List[List[str]],
    cfg: Config = Config,
):
    top_set = set(mlb.classes_)
    all_categories = sorted(list(top_set.union(set(rare_codes))))
    mlb_all = MultiLabelBinarizer(classes=all_categories)
    mlb_all.fit([all_categories])
    y_test_all = mlb_all.transform(test_raw_terms)

    model.eval()
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(cfg.DEVICE)
    mlp_probs = torch.sigmoid(model(test_tensor)).cpu().numpy()
    rare_sim_matrix = util.cos_sim(test_tensor, rare_embeddings).cpu().numpy()

    full_score_matrix = np.zeros((len(X_test), len(all_categories)), dtype=np.float32)
    y_pred_all = np.zeros((len(X_test), len(all_categories)), dtype=int)

    top_to_master = [all_categories.index(tag) for tag in mlb.classes_]
    rare_to_master = [all_categories.index(code) for code in rare_codes]

    for mlp_idx, master_idx in enumerate(top_to_master):
        full_score_matrix[:, master_idx] = mlp_probs[:, mlp_idx]
        y_pred_all[:, master_idx] = (mlp_probs[:, mlp_idx] >= cfg.MLP_THRESHOLD).astype(int)

    for rare_idx, master_idx in enumerate(rare_to_master):
        full_score_matrix[:, master_idx] = rare_sim_matrix[:, rare_idx]
        y_pred_all[:, master_idx] = (rare_sim_matrix[:, rare_idx] >= cfg.TAIL_THRESHOLD).astype(int)

    print(f"Weighted F1 : {f1_score(y_test_all, y_pred_all, average='weighted', zero_division=0)*100:.2f}%")
    print(f"Micro F1    : {f1_score(y_test_all, y_pred_all, average='micro', zero_division=0)*100:.2f}%")
    print(f"Micro Recall: {recall_score(y_test_all, y_pred_all, average='micro', zero_division=0)*100:.2f}%")
    print(f"Macro F1    : {f1_score(y_test_all, y_pred_all, average='macro', zero_division=0)*100:.2f}%")

    test_support = y_test_all.sum(axis=0)
    for cutoff in [1, 3, 5, 10]:
        valid_classes = np.where(test_support >= cutoff)[0]
        score = f1_score(y_test_all[:, valid_classes], y_pred_all[:, valid_classes], average="macro", zero_division=0) * 100
        print(f"Macro F1 (support >= {cutoff:2d}, n={len(valid_classes):2d} classes): {score:.2f}%")

    metrics = compute_ranking_metrics_at_k(y_test_all, full_score_matrix, k_values=(1, 3, 5))
    for k in (1, 3, 5):
        print(
            f"K={k} | P@{k}: {metrics[f'P@{k}']:.2f}% | R@{k}: {metrics[f'R@{k}']:.2f}% | "
            f"F1@{k}: {metrics[f'F1@{k}']:.2f}% | NDCG@{k}: {metrics[f'NDCG@{k}']:.2f}%"
        )