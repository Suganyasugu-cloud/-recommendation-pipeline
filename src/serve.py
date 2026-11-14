"""Prediction-time example: recommend for one user using the same FeatureStore and saved model."""
import argparse

import joblib

from src.common import load_config, resolve_path
from src.features.feature_store import load_feature_store


def recommend(user_id: int, k: int = 10, model_name: str = "item_knn"):
    """Return the user's feature profile and top-k unseen items with their item features.

    Uses the same FeatureStore definitions as training, so prediction-time features match.
    """
    config = load_config()
    model = joblib.load(resolve_path(config, "models") / f"{model_name}.joblib")
    store = load_feature_store(config)
    scores = model.score_users([user_id]).astype(float)
    scores[model.seen_mask([user_id])] = float("-inf")
    top_items = model.item_ids[scores[0].argsort()[::-1][:k]]
    profile = store.get_user_features([user_id])
    recs = store.get_interaction_features([user_id] * len(top_items), top_items)
    recs["title"] = store.items.loc[top_items, "title"].to_numpy()
    item_cols = ["item_popularity", "item_avg_rating"]
    recs[item_cols] = store.get_item_features(top_items)[item_cols].to_numpy()
    return profile, recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id", type=int)
    parser.add_argument("--model", default="item_knn")
    args = parser.parse_args()
    user_profile, recommendations = recommend(args.user_id, model_name=args.model)
    print("User features:\n", user_profile.T.to_string(), "\n")
    print(recommendations.round(3).to_string(index=False))
