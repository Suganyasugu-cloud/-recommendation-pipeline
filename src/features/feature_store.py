"""Stage 5 - Feature management (lightweight custom feature layer).

A single FeatureStore class is used by training, evaluation and serving, so the
same feature definitions apply offline and online. Features are computed from
the TRAIN split relative to an `as_of` timestamp (the end of the training
window) and materialised to data/features/*.parquet.
"""
import pandas as pd

from src.common import get_logger, resolve_path

LOGGER = get_logger("features")
SECONDS_PER_DAY = 86400

FEATURE_CATALOG = [
    # entity, name, dtype, description, definition
    ("user", "user_activity_count", "int", "Number of ratings given by the user", "count(train ratings of user)"),
    ("user", "user_avg_rating", "float", "Mean rating given by the user", "mean(rating) per user"),
    ("user", "user_rating_std", "float", "Rating spread of the user", "std(rating) per user, 0 if single rating"),
    ("user", "user_positive_ratio", "float", "Share of the user's ratings that are positive", "mean(rating >= threshold)"),
    ("user", "user_preferred_genre", "str", "Genre the user rates positively most often", "argmax genre count over positive ratings"),
    ("user", "user_days_since_last_interaction", "float", "Recency of the user's last rating", "(as_of - max(timestamp)) / 86400"),
    ("user", "age / gender / occupation", "int/str", "Demographic attributes", "from validated users table"),
    ("item", "item_popularity", "int", "Number of ratings received", "count(train ratings of item)"),
    ("item", "item_positive_count", "int", "Number of positive ratings received", "count(rating >= threshold)"),
    ("item", "item_avg_rating", "float", "Mean rating received", "mean(rating) per item"),
    ("item", "item_popularity_rank", "int", "Rank by positive count (1 = most popular)", "rank(item_positive_count, desc)"),
    ("item", "item_days_since_last_rating", "float", "Recency of the item's last rating", "(as_of - max(timestamp)) / 86400"),
    ("item", "release_year / item_age_years", "int", "Release year and age at as_of", "year(as_of) - release_year"),
    ("item", "n_genres", "int", "Number of genres tagged on the item", "sum(genre flags)"),
    ("interaction", "user_genre_affinity", "float", "How strongly the user's genre profile matches the item", "mean over item genres of user's positive-rating share in that genre"),
    ("interaction", "expected_rating_bias", "float", "Baseline rating estimate", "global_mean + (user_avg - global_mean) + (item_avg - global_mean)"),
    ("interaction", "item_popularity_percentile", "float", "Item popularity percentile in [0,1]", "1 - (rank - 1) / n_items"),
]


class FeatureStore:
    """Computes, persists and serves user, item and interaction features."""

    def __init__(self, train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame,
                 positive_threshold: int, as_of_ts: int = None):
        self.train = train
        self.users = users.set_index("user_id")
        self.items = items.set_index("item_id")
        self.threshold = positive_threshold
        self.as_of_ts = int(as_of_ts if as_of_ts is not None else train["timestamp"].max())
        self.genre_cols = [c for c in items.columns
                           if c not in ("item_id", "title", "release_year", "release_year_imputed")]
        self.global_mean = float(train["rating"].mean())
        self._user_genre_profile = self._build_user_genre_profile()
        self.user_features = self._build_user_features()
        self.item_features = self._build_item_features()

    def _build_user_genre_profile(self) -> pd.DataFrame:
        """Per-user share of positive ratings falling in each genre."""
        positives = self.train[self.train["rating"] >= self.threshold]
        genres = self.items[self.genre_cols].reindex(positives["item_id"]).fillna(0).to_numpy()
        counts = pd.DataFrame(genres, columns=self.genre_cols, index=positives["user_id"].to_numpy())
        counts = counts.groupby(level=0).sum()
        return counts.div(counts.sum(axis=1).replace(0, 1), axis=0)

    def _build_user_features(self) -> pd.DataFrame:
        grouped = self.train.groupby("user_id")
        feats = pd.DataFrame({
            "user_activity_count": grouped.size(),
            "user_avg_rating": grouped["rating"].mean(),
            "user_rating_std": grouped["rating"].std().fillna(0.0),
            "user_positive_ratio": grouped["rating"].apply(lambda r: (r >= self.threshold).mean()),
            "user_days_since_last_interaction":
                (self.as_of_ts - grouped["timestamp"].max()) / SECONDS_PER_DAY,
        })
        feats["user_preferred_genre"] = self._preferred_genre()
        return feats.join(self.users[["age", "gender", "occupation"]], how="left")

    def _preferred_genre(self) -> pd.Series:
        return self._user_genre_profile.idxmax(axis=1).rename("user_preferred_genre")

    def _build_item_features(self) -> pd.DataFrame:
        positive = (self.train["rating"] >= self.threshold).astype(int)
        grouped = self.train.assign(positive=positive).groupby("item_id")
        feats = pd.DataFrame({
            "item_popularity": grouped.size(),
            "item_positive_count": grouped["positive"].sum(),
            "item_avg_rating": grouped["rating"].mean(),
            "item_days_since_last_rating": (self.as_of_ts - grouped["timestamp"].max()) / SECONDS_PER_DAY,
        })
        feats["item_popularity_rank"] = feats["item_positive_count"].rank(ascending=False, method="first").astype(int)
        as_of_year = pd.to_datetime(self.as_of_ts, unit="s").year
        meta = self.items[["release_year"]].copy()
        meta["item_age_years"] = as_of_year - meta["release_year"]
        meta["n_genres"] = self.items[self.genre_cols].sum(axis=1)
        return feats.join(meta, how="outer")

    # ----- serving API (identical for training and prediction) -----
    def get_user_features(self, user_ids) -> pd.DataFrame:
        """Return user features; unknown users get NaN rows (cold start)."""
        return self.user_features.reindex(user_ids)

    def get_item_features(self, item_ids) -> pd.DataFrame:
        """Return item features for the requested items."""
        return self.item_features.reindex(item_ids)

    def get_item_genres(self, item_ids) -> pd.DataFrame:
        """Return the binary genre matrix for items."""
        return self.items[self.genre_cols].reindex(item_ids).fillna(0)

    def get_interaction_features(self, user_ids, item_ids) -> pd.DataFrame:
        """Compute pair-level features for aligned lists of user and item ids."""
        users = self.user_features.reindex(user_ids)
        items = self.item_features.reindex(item_ids)
        profile = self._user_genre_profile.reindex(user_ids).fillna(0).to_numpy()
        genres = self.items[self.genre_cols].reindex(item_ids).fillna(0).to_numpy()
        n_genres = genres.sum(axis=1).clip(min=1)
        n_items = max(len(self.item_features.dropna(subset=["item_popularity"])), 1)
        out = pd.DataFrame({
            "user_id": list(user_ids), "item_id": list(item_ids),
            "user_genre_affinity": (profile * genres).sum(axis=1) / n_genres,
            "expected_rating_bias": (self.global_mean
                                     + (users["user_avg_rating"].fillna(self.global_mean).to_numpy() - self.global_mean)
                                     + (items["item_avg_rating"].fillna(self.global_mean).to_numpy() - self.global_mean)),
            "item_popularity_percentile": 1 - (items["item_popularity_rank"].fillna(n_items).to_numpy() - 1) / n_items,
        })
        return out

    def materialize(self, config: dict) -> None:
        """Persist feature tables and the human-readable catalog."""
        out = resolve_path(config, "features")
        self.user_features.reset_index().rename(columns={"index": "user_id"}).to_parquet(out / "user_features.parquet", index=False)
        self.item_features.reset_index().rename(columns={"index": "item_id"}).to_parquet(out / "item_features.parquet", index=False)
        write_catalog(resolve_path(config, "reports"))


def write_catalog(reports_dir) -> None:
    """Write the feature catalog as Markdown and CSV."""
    catalog = pd.DataFrame(FEATURE_CATALOG, columns=["entity", "feature", "dtype", "description", "definition"])
    catalog.to_csv(reports_dir / "feature_catalog.csv", index=False)
    lines = ["# Feature Catalog", "",
             "| Entity | Feature | Type | Description | Definition |", "|---|---|---|---|---|"]
    lines += [f"| {r.entity} | `{r.feature}` | {r.dtype} | {r.description} | {r.definition} |"
              for r in catalog.itertuples()]
    (reports_dir / "feature_catalog.md").write_text("\n".join(lines))


def load_feature_store(config: dict) -> FeatureStore:
    """Build a FeatureStore from the processed train split (shared entry point)."""
    processed = resolve_path(config, "processed")
    return FeatureStore(pd.read_parquet(processed / "train.parquet"),
                        pd.read_parquet(processed / "users.parquet"),
                        pd.read_parquet(processed / "items.parquet"),
                        config["features"]["positive_rating_threshold"])


def run(config: dict) -> dict:
    """Materialise features to disk."""
    store = load_feature_store(config)
    store.materialize(config)
    LOGGER.info("Features: %d users, %d items", len(store.user_features), len(store.item_features))
    return {"user_features": len(store.user_features), "item_features": len(store.item_features)}
