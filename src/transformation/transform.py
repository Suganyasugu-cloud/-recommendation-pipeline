"""Stage 4 - Data transformation.

Builds derived datasets from the TRAIN split only (no leakage):
user activity summary, item popularity and the sparse user-item matrix.
"""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from src.common import get_logger, resolve_path

LOGGER = get_logger("transformation")


def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Per-user interaction counts, rating stats and activity window."""
    grouped = train.groupby("user_id")
    summary = grouped.agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"),
                          rating_std=("rating", "std"), first_ts=("timestamp", "min"),
                          last_ts=("timestamp", "max"))
    summary["rating_std"] = summary["rating_std"].fillna(0.0)
    return summary.reset_index()


def item_popularity(train: pd.DataFrame, positive_threshold: int) -> pd.DataFrame:
    """Per-item rating counts, positive counts and mean rating, ranked by popularity."""
    pos = (train["rating"] >= positive_threshold).astype(int)
    agg = train.assign(positive=pos).groupby("item_id").agg(
        n_ratings=("rating", "size"), n_positive=("positive", "sum"), mean_rating=("rating", "mean"))
    agg["popularity_rank"] = agg["n_positive"].rank(ascending=False, method="first").astype(int)
    return agg.reset_index().sort_values("popularity_rank")


def interaction_matrix(train: pd.DataFrame):
    """Return a CSR user x item rating matrix plus id <-> index mappings."""
    users = np.sort(train["user_id"].unique())
    items = np.sort(train["item_id"].unique())
    user_idx = pd.Series(np.arange(len(users)), index=users)
    item_idx = pd.Series(np.arange(len(items)), index=items)
    matrix = sparse.csr_matrix(
        (train["rating"].astype(float), (user_idx[train["user_id"]].to_numpy(),
                                         item_idx[train["item_id"]].to_numpy())),
        shape=(len(users), len(items)))
    return matrix, users, items


def run(config: dict) -> dict:
    """Write derived datasets to data/processed."""
    processed = resolve_path(config, "processed")
    train = pd.read_parquet(processed / "train.parquet")
    summary = user_activity_summary(train)
    popularity = item_popularity(train, config["features"]["positive_rating_threshold"])
    matrix, users, items = interaction_matrix(train)

    summary.to_parquet(processed / "user_activity_summary.parquet", index=False)
    popularity.to_parquet(processed / "item_popularity.parquet", index=False)
    sparse.save_npz(processed / "interaction_matrix.npz", matrix)
    (processed / "matrix_index.json").write_text(
        json.dumps({"user_ids": users.tolist(), "item_ids": items.tolist()}))
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    LOGGER.info("Interaction matrix %s, density %.4f", matrix.shape, density)
    return {"matrix_shape": list(matrix.shape), "density": round(float(density), 5)}
