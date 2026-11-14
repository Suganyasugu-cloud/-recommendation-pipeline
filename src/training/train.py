"""Stage 6 - Model training with MLflow experiment tracking."""
import json

import joblib
import mlflow
import pandas as pd

from src.common import get_logger, resolve_path, set_seed
from src.features.feature_store import load_feature_store
from src.training.models import ItemKNNRecommender, PopularityRecommender, SVDRecommender

LOGGER = get_logger("training")


def build_models(config: dict) -> list:
    """Instantiate the models to train from config."""
    cfg, threshold = config["training"], config["features"]["positive_rating_threshold"]
    return [
        PopularityRecommender(),
        ItemKNNRecommender(neighbors=cfg["knn_neighbors"], positive_threshold=threshold),
        SVDRecommender(factors=cfg["svd_factors"], seed=config["seed"]),
    ]


def run(config: dict) -> dict:
    """Fit all models, save them under models/ and record params/artifacts in MLflow."""
    set_seed(config["seed"])
    processed, models_dir = resolve_path(config, "processed"), resolve_path(config, "models")
    train = pd.read_parquet(processed / "train.parquet")
    store = load_feature_store(config)
    mlflow.set_tracking_uri(config["training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["training"]["mlflow_experiment"])

    run_ids = {}
    for model in build_models(config):
        with mlflow.start_run(run_name=model.name) as active:
            mlflow.log_params({**model.get_params(), "train_rows": len(train), "seed": config["seed"]})
            model.fit(train, store)
            path = models_dir / f"{model.name}.joblib"
            joblib.dump(model, path)
            mlflow.log_artifact(str(path))
            mlflow.log_artifact(str(resolve_path(config, "reports") / "feature_catalog.csv"))
            run_ids[model.name] = active.info.run_id
            LOGGER.info("Trained %s (run %s)", model.name, active.info.run_id)
    (models_dir / "run_ids.json").write_text(json.dumps(run_ids, indent=2))
    return run_ids
