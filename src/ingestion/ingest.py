"""Stage 1 - Data ingestion.

Downloads MovieLens 100K and writes three raw CSV files (users, items,
interactions). Optionally injects a seeded set of faults into the raw copy so
the validation stage has something to detect.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import requests

from src.common import get_logger, resolve_path, set_seed

LOGGER = get_logger("ingestion")

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download_dataset(config: dict) -> None:
    """Download and unzip the dataset into data/raw if not already present."""
    raw_dir = resolve_path(config, "raw")
    target = raw_dir / config["ingestion"]["dataset_dir"]
    if target.exists():
        LOGGER.info("Dataset already present at %s", target)
        return
    LOGGER.info("Downloading %s", config["ingestion"]["dataset_url"])
    response = requests.get(config["ingestion"]["dataset_url"], timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(raw_dir)


def read_source(config: dict) -> dict:
    """Read the MovieLens source files into DataFrames."""
    base = resolve_path(config, "raw") / config["ingestion"]["dataset_dir"]
    users = pd.read_csv(
        base / "u.user", sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"], dtype=str,
    )
    items = pd.read_csv(
        base / "u.item", sep="|", encoding="latin-1", dtype=str,
        names=["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES,
    ).drop(columns=["video_release_date", "imdb_url"])
    interactions = pd.read_csv(
        base / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"], dtype=str,
    )
    return {"users": users, "items": items, "interactions": interactions}


def inject_faults(tables: dict, config: dict) -> dict:
    """Corrupt a copy of the raw tables with a documented, seeded set of faults."""
    rng = np.random.default_rng(config["seed"])
    counts = config["ingestion"]["fault_counts"]
    users, items = tables["users"].copy(), tables["items"].copy()
    inter = tables["interactions"].copy()

    def pick(n: int) -> np.ndarray:
        return rng.choice(inter.index.to_numpy(), size=n, replace=False)

    inter.loc[pick(counts["null_ratings"]), "rating"] = np.nan
    inter.loc[pick(counts["invalid_ratings"]), "rating"] = rng.choice(["0", "6", "-1", "9"], counts["invalid_ratings"])
    inter.loc[pick(counts["invalid_timestamps"]), "timestamp"] = rng.choice(
        ["-5", "0", "99999999999", "not_a_time"], counts["invalid_timestamps"])
    inter.loc[pick(counts["orphan_users"]), "user_id"] = "99999"
    inter.loc[pick(counts["orphan_items"]), "item_id"] = "88888"
    duplicates = inter.sample(counts["duplicate_rows"], random_state=config["seed"])
    inter = pd.concat([inter, duplicates], ignore_index=True)
    users.loc[rng.choice(users.index.to_numpy(), counts["bad_age_type"], replace=False), "age"] = "unknown"
    LOGGER.info("Injected faults: %s", counts)
    return {"users": users, "items": items, "interactions": inter}


def run(config: dict) -> dict:
    """Execute ingestion and write data/raw/{users,items,interactions}.csv."""
    set_seed(config["seed"])
    download_dataset(config)
    tables = read_source(config)
    if config["ingestion"]["inject_faults"]:
        tables = inject_faults(tables, config)
    raw_dir = resolve_path(config, "raw")
    for name, frame in tables.items():
        frame.to_csv(raw_dir / f"{name}.csv", index=False)
        LOGGER.info("Ingested %s: %d rows", name, len(frame))
    return {name: len(frame) for name, frame in tables.items()}
