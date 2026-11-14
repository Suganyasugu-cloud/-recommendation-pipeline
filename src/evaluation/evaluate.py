"""Stage 7 - Model evaluation: compare all trained models on the held-out test period."""
import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.common import get_logger, resolve_path
from src.evaluation import metrics as M
from src.features.feature_store import load_feature_store

LOGGER = get_logger("evaluation")


def top_k_lists(model, users, k: int) -> dict:
    """Top-k unseen items per user."""
    scores = model.score_users(users).astype(float)
    scores[model.seen_mask(users)] = -np.inf
    top = np.argsort(-scores, axis=1)[:, :k]
    return {u: model.item_ids[idx].tolist() for u, idx in zip(users, top)}


def evaluate_model(model, test, store, threshold: int, k: int) -> dict:
    """Compute ranking, catalogue and rating metrics for one model."""
    relevant = test[test["rating"] >= threshold].groupby("user_id")["item_id"].agg(set)
    users = [u for u in relevant.index if u in model.user_pos]
    lists = top_k_lists(model, users, k)
    prec = [M.precision_at_k(lists[u], relevant[u], k) for u in users]
    rec = [M.recall_at_k(lists[u], relevant[u], k) for u in users]
    ndcg = [M.ndcg_at_k(lists[u], relevant[u], k) for u in users]
    div = [M.intra_list_diversity(store.get_item_genres(lists[u]).to_numpy()) for u in users]
    recommended = {i for items in lists.values() for i in items}
    rating_scores = {"rmse": float("nan"), "mae": float("nan")}
    if model.supports_rating:
        pairs = test[test["user_id"].isin(model.user_pos) & test["item_id"].isin(model.item_pos)]
        preds = model.predict_rating(pairs["user_id"].tolist(), pairs["item_id"].tolist())
        rating_scores = {"rmse": M.rmse(pairs["rating"], preds), "mae": M.mae(pairs["rating"], preds)}
    return {
        f"precision@{k}": float(np.mean(prec)), f"recall@{k}": float(np.mean(rec)),
        f"ndcg@{k}": float(np.mean(ndcg)), "coverage": len(recommended) / len(model.item_ids),
        f"diversity@{k}": float(np.mean(div)),
        **rating_scores, "n_eval_users": len(users),
    }


def run(config: dict) -> dict:
    """Evaluate every saved model, log to MLflow and write the comparison table."""
    processed, models_dir = resolve_path(config, "processed"), resolve_path(config, "models")
    reports = resolve_path(config, "reports")
    k, threshold = config["training"]["top_k"], config["features"]["positive_rating_threshold"]
    test = pd.read_parquet(processed / "test.parquet")
    store = load_feature_store(config)
    run_ids = json.loads((models_dir / "run_ids.json").read_text())
    mlflow.set_tracking_uri(config["training"]["mlflow_tracking_uri"])

    results = {}
    for name, run_id in run_ids.items():
        model = joblib.load(models_dir / f"{name}.joblib")
        results[name] = evaluate_model(model, test, store, threshold, k)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({key.replace("@", "_at_"): val for key, val in results[name].items()
                                    if not np.isnan(val)})
        LOGGER.info("%s: %s", name, {m: round(v, 4) for m, v in results[name].items()})

    table = pd.DataFrame(results).T.round(4)
    table.index.name = "model"
    table.to_csv(reports / "model_comparison.csv")
    note = ("RMSE/MAE: popularity = item mean rating; svd_mf = bias + latent factors; item_knn is "
            "ranking-only (NaN). Users evaluated: those with at "
            f"least one test rating >= {threshold} who appear in train. Seen items are excluded.")
    (reports / "model_comparison.md").write_text(
        f"# Model Comparison (K={k})\n\n{table.astype(object).where(table.notna(), "n/a").to_markdown()}\n\n{note}\n")
    with mlflow.start_run(run_name="comparison"):
        mlflow.log_artifact(str(reports / "model_comparison.csv"))
    return results
