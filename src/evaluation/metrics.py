"""Ranking and rating metrics."""
import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items retrieved in the top-k."""
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Binary-relevance NDCG@k."""
    gains = [1.0 if item in relevant else 0.0 for item in recommended[:k]]
    dcg = sum(g / np.log2(rank + 2) for rank, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def intra_list_diversity(genre_matrix: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity of genre vectors within one list."""
    norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True)
    unit = genre_matrix / np.where(norms == 0, 1, norms)
    sim = unit @ unit.T
    n = len(genre_matrix)
    off_diag = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(1.0 - off_diag)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
