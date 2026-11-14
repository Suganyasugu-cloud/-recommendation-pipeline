"""Prefect orchestration of the recommendation pipeline.

Run once:         python -m dags.prefect_flow
Serve on a cron:  python -m dags.prefect_flow --serve   (daily 02:00)

Dependencies are explicit through `wait_for`:
ingest -> validate -> prepare -> transform -> build_features -> train_models -> evaluate_models
"""
import argparse

from prefect import flow, task

from src.common import load_config
from src.evaluation import evaluate as evaluate_stage
from src.features import feature_store
from src.ingestion import ingest as ingest_stage
from src.preparation import prepare as prepare_stage
from src.training import train as train_stage
from src.transformation import transform as transform_stage
from src.validation import validate as validate_stage


@task(name="ingest", retries=1, retry_delay_seconds=5)
def ingest(config):
    """Download and write raw CSVs."""
    return ingest_stage.run(config)


@task(name="validate")
def validate(config):
    """Run validation rules and write error datasets."""
    return validate_stage.run(config)


@task(name="prepare")
def prepare(config):
    """Clean data and split by time."""
    return prepare_stage.run(config)


@task(name="transform")
def transform(config):
    """Build derived datasets and the interaction matrix."""
    return transform_stage.run(config)


@task(name="build_features")
def build_features(config):
    """Materialise the feature tables and catalog."""
    return feature_store.run(config)


@task(name="train_models")
def train_models(config):
    """Train all recommenders and track them in MLflow."""
    return train_stage.run(config)


@task(name="evaluate_models")
def evaluate_models(config):
    """Evaluate all models on the test period."""
    return evaluate_stage.run(config)


@flow(name="recsys-pipeline", log_prints=True)
def recsys_pipeline():
    """End-to-end recommendation pipeline."""
    config = load_config()
    ingested = ingest.submit(config)
    validated = validate.submit(config, wait_for=[ingested])
    prepared = prepare.submit(config, wait_for=[validated])
    transformed = transform.submit(config, wait_for=[prepared])
    featured = build_features.submit(config, wait_for=[transformed])
    trained = train_models.submit(config, wait_for=[featured])
    return evaluate_models.submit(config, wait_for=[trained]).result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve with a daily 02:00 schedule")
    args = parser.parse_args()
    if args.serve:
        recsys_pipeline.serve(name="recsys-daily", cron="0 2 * * *")
    else:
        recsys_pipeline()
