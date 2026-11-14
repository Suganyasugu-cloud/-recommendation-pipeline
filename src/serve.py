"""Prediction-time example: recommend for one user using the same FeatureStore and saved model."""
import argparse

import joblib

from src.common import load_config, resolve_path
from src.features.feature_store import load_feature_store


def recommend(user_id: int, k: int = 10, model_name: str = "item_knn") -> list:
    """Return top-k unseen item titles for a known user."""
    config = load_config()
    model = joblib.load(resolve_path(config, "models") / f"{model_name}.joblib")
    store = load_feature_store(config)
    scores = model.score_users([user_id]).astype(float)
    scores[model.seen_mask([user_id])] = float("-inf")
    top_items = model.item_ids[scores[0].argsort()[::-1][:k]]
    return store.items.loc[top_items, "title"].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id", type=int)
    parser.add_argument("--model", default="item_knn")
    args = parser.parse_args()
    print("\n".join(recommend(args.user_id, model_name=args.model)))
