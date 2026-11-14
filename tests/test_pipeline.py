"""Unit tests for validation rules, split logic and metrics."""
import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.preparation.prepare import time_split
from src.validation.validate import validate_interactions

CFG = {"rating_min": 1, "rating_max": 5, "min_timestamp": 631152000, "max_timestamp": 1893456000}


def _tables():
    users = pd.DataFrame({"user_id": ["1", "2"]})
    items = pd.DataFrame({"item_id": ["10", "11"]})
    return users, items


def test_validation_flags_each_fault():
    users, items = _tables()
    inter = pd.DataFrame({
        "user_id": ["1", "1", "9", "2", "2", "1"],
        "item_id": ["10", "11", "10", "99", "10", "10"],
        "rating": ["5", "0", "3", "3", None, "5"],
        "timestamp": ["900000000", "900000000", "900000000", "900000000", "900000000", "900000000"],
    })
    inter.loc[5] = inter.loc[0]  # exact duplicate
    rules = set(validate_interactions(inter, users, items, CFG)["rule"])
    assert {"invalid_rating", "orphan_user_id", "orphan_item_id", "missing_value", "duplicate_row"} <= rules


def test_split_is_chronological_per_user():
    inter = pd.DataFrame({"user_id": [1] * 10, "item_id": range(10), "rating": 4, "timestamp": range(10)})
    train, test, _ = time_split(inter, 0.2)
    assert len(test) == 2 and train["timestamp"].max() < test["timestamp"].min()


def test_ranking_metrics():
    assert M.precision_at_k([1, 2, 3], {1, 3}, 3) == 2 / 3
    assert M.recall_at_k([1, 2, 3], {1, 3, 7, 8}, 3) == 0.5
    assert M.ndcg_at_k([1, 2], {1}, 2) == 1.0
    assert np.isclose(M.intra_list_diversity(np.array([[1, 0], [0, 1]])), 1.0)
