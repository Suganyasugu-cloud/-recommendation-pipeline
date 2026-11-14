"""Plain-Python entry point: runs all stages in dependency order."""
import argparse

from src.common import load_config
from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

STAGES = [("ingestion", ingest), ("validation", validate), ("preparation", prepare),
          ("transformation", transform), ("features", feature_store),
          ("training", train), ("evaluation", evaluate)]


def main() -> None:
    """CLI: `python -m src.pipeline [--from-stage NAME]`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", choices=[s for s, _ in STAGES], default=STAGES[0][0])
    args = parser.parse_args()
    config = load_config()
    names = [s for s, _ in STAGES]
    for name, module in STAGES[names.index(args.from_stage):]:
        module.run(config)


if __name__ == "__main__":
    main()
