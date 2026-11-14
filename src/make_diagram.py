"""Render the architecture diagram to reports/architecture.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.common import PROJECT_ROOT

STAGES = [
    ("1 Ingestion", "MovieLens 100K\nusers / items / ratings\n(+ seeded faults)", "data/raw"),
    ("2 Validation", "type, null, range,\ntimestamp, duplicate,\nreferential checks", "data/errors\nreports/validation_*"),
    ("3 Preparation", "clean, standardise,\nper-user time split", "data/processed\ntrain / test"),
    ("4 Transformation", "user activity,\nitem popularity,\nuser-item matrix", "data/processed"),
    ("5 Features", "FeatureStore\nuser / item /\ninteraction", "data/features\nfeature_catalog"),
    ("6 Training", "Popularity, ItemKNN,\nSVD-MF\n(MLflow tracking)", "models/ mlruns/"),
    ("7 Evaluation", "P@K R@K NDCG@K\ncoverage diversity\nRMSE MAE", "reports/\nmodel_comparison"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 4.6))
    ax.set_xlim(0, 7 * 2.3)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    for i, (title, body, out) in enumerate(STAGES):
        x = i * 2.3 + 0.1
        ax.add_patch(FancyBboxPatch((x, 1.6), 2.0, 1.9, boxstyle="round,pad=0.05", fc="#dbe9f7", ec="#1f4e79"))
        ax.text(x + 1.0, 3.2, title, ha="center", fontweight="bold", fontsize=10)
        ax.text(x + 1.0, 2.4, body, ha="center", va="center", fontsize=8)
        ax.add_patch(FancyBboxPatch((x, 0.35), 2.0, 0.9, boxstyle="round,pad=0.05", fc="#fdf2d0", ec="#8a6d00"))
        ax.text(x + 1.0, 0.8, out, ha="center", va="center", fontsize=7.5)
        ax.annotate("", xy=(x + 1.0, 1.28), xytext=(x + 1.0, 1.58), arrowprops=dict(arrowstyle="->"))
        if i < len(STAGES) - 1:
            ax.annotate("", xy=(x + 2.3, 2.55), xytext=(x + 2.05, 2.55), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(7 * 2.3 / 2, 4.3, "Orchestrated by Prefect flow (dags/prefect_flow.py); Airflow equivalent in dags/airflow_dag.py",
            ha="center", fontsize=10, style="italic")
    fig.savefig(PROJECT_ROOT / "reports" / "architecture.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
