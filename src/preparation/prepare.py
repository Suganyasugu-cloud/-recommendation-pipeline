"""Stage 3 - Data preparation.

Cleans validated tables (types, formats, residual missing values, duplicate
user-item pairs) and creates a time-based train/test split.
"""
import numpy as np
import pandas as pd

from src.common import get_logger, resolve_path

LOGGER = get_logger("preparation")


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Standardise types and formats; impute residual missing ages with the median."""
    users = users.copy()
    users["user_id"] = users["user_id"].astype(int)
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users["age"] = users["age"].fillna(users["age"].median()).astype(int)
    users["gender"] = users["gender"].str.strip().str.upper().fillna("UNKNOWN")
    users["occupation"] = users["occupation"].str.strip().str.lower().fillna("other")
    users["zip_code"] = users["zip_code"].astype(str).str.strip().str.upper()
    return users.drop_duplicates("user_id").reset_index(drop=True)


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    """Standardise item metadata, derive release_year and impute the missing ones."""
    items = items.copy()
    items["item_id"] = items["item_id"].astype(int)
    items["title"] = items["title"].str.strip()
    release = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce")
    items["release_year"] = release.dt.year
    items["release_year_imputed"] = items["release_year"].isna()
    items["release_year"] = items["release_year"].fillna(items["release_year"].median()).astype(int)
    genre_cols = [c for c in items.columns if c not in
                  ("item_id", "title", "release_date", "release_year", "release_year_imputed")]
    items[genre_cols] = items[genre_cols].astype(int)
    return items.drop(columns=["release_date"]).drop_duplicates("item_id").reset_index(drop=True)


def clean_interactions(inter: pd.DataFrame) -> pd.DataFrame:
    """Cast types and keep only the latest rating per user-item pair."""
    inter = inter.copy()
    inter["user_id"] = inter["user_id"].astype(int)
    inter["item_id"] = inter["item_id"].astype(int)
    inter["rating"] = inter["rating"].astype(float).astype(int)
    inter["timestamp"] = inter["timestamp"].astype(float).astype("int64")
    inter = inter.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    inter["interaction_time"] = pd.to_datetime(inter["timestamp"], unit="s", utc=True)
    return inter.reset_index(drop=True)


def time_split(inter: pd.DataFrame, test_fraction: float, min_ratings: int = 5):
    """Per-user chronological split.

    Each user's most recent `test_fraction` of interactions go to the test set, so every
    evaluated user has history in train that strictly precedes their test events.
    A single global cut-off would leave most users with no test data in MovieLens 100K
    (ratings span only ~7 months and most users rate in one short session).
    Users with fewer than `min_ratings` interactions are kept entirely in train.
    """
    inter = inter.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    rank = inter.groupby("user_id").cumcount()
    size = inter.groupby("user_id")["item_id"].transform("size")
    n_test = np.floor(size * test_fraction).astype(int)
    is_test = (rank >= size - n_test) & (size >= min_ratings)
    train, test = inter[~is_test], inter[is_test]
    return train.reset_index(drop=True), test.reset_index(drop=True), int(train["timestamp"].max())


def run(config: dict) -> dict:
    """Run preparation and write parquet files into data/processed."""
    processed = resolve_path(config, "processed")
    users = clean_users(pd.read_csv(processed / "users_validated.csv", dtype=str))
    items = clean_items(pd.read_csv(processed / "items_validated.csv", dtype=str))
    inter = clean_interactions(pd.read_csv(processed / "interactions_validated.csv", dtype=str))
    train, test, cutoff = time_split(inter, config["split"]["test_fraction"])

    for name, frame in (("users", users), ("items", items), ("interactions", inter),
                        ("train", train), ("test", test)):
        frame.to_parquet(processed / f"{name}.parquet", index=False)
    stats = {"users": len(users), "items": len(items), "interactions": len(inter),
             "train": len(train), "test": len(test), "train_end_timestamp": cutoff,
             "cold_start_test_users": int((~test["user_id"].isin(train["user_id"])).sum())}
    LOGGER.info("Prepared data: %s", stats)
    return stats
