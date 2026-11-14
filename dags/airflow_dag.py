"""Apache Airflow equivalent of the Prefect flow (same stage modules, same dependencies).

Not required to run the project; provided so the pipeline can be deployed on Airflow.
Requires `apache-airflow` and PYTHONPATH pointing at the project root.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.common import load_config
from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

STAGES = [("ingest", ingest), ("validate", validate), ("prepare", prepare),
          ("transform", transform), ("build_features", feature_store),
          ("train_models", train), ("evaluate_models", evaluate)]

with DAG(
    dag_id="recsys_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=5)},
) as dag:
    previous = None
    for task_id, module in STAGES:
        current = PythonOperator(task_id=task_id, python_callable=lambda m=module: m.run(load_config()))
        if previous:
            previous >> current
        previous = current
