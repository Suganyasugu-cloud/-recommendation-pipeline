"""Stage 2 - Data validation.

Rule-based checks over users, items and interactions. Every failing record is
written to data/errors/<table>_errors.csv with the rule that rejected it; clean
records go to data/processed/<table>_validated.csv. A JSON and Markdown report
summarise rule counts.
"""
import json

import pandas as pd

from src.common import get_logger, resolve_path

LOGGER = get_logger("validation")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def validate_users(users: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return a DataFrame of (index, rule) failures for the users table."""
    age = _numeric(users["age"])
    rules = {
        "missing_value": users.isna().any(axis=1),
        "wrong_dtype_age": age.isna() & users["age"].notna(),
        "age_out_of_range": age.notna() & ~age.between(cfg["age_min"], cfg["age_max"]),
        "duplicate_user_id": users.duplicated("user_id", keep="first"),
    }
    return _collect(rules)


def validate_items(items: pd.DataFrame) -> pd.DataFrame:
    """Return failures for the items table."""
    rules = {
        "missing_title": items["title"].isna(),
        "duplicate_item_id": items.duplicated("item_id", keep="first"),
        "invalid_release_date": items["release_date"].notna()
        & pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce").isna(),
    }
    return _collect(rules)


def validate_interactions(inter: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return failures for interactions, including referential integrity."""
    rating, ts = _numeric(inter["rating"]), _numeric(inter["timestamp"])
    rules = {
        "missing_value": inter.isna().any(axis=1),
        "wrong_dtype_rating": rating.isna() & inter["rating"].notna(),
        "invalid_rating": rating.notna() & ~rating.between(cfg["rating_min"], cfg["rating_max"]),
        "invalid_timestamp": inter["timestamp"].notna()
        & ~ts.between(cfg["min_timestamp"], cfg["max_timestamp"]),
        "orphan_user_id": ~inter["user_id"].isin(users["user_id"]),
        "orphan_item_id": ~inter["item_id"].isin(items["item_id"]),
        "duplicate_row": inter.duplicated(keep="first"),
    }
    return _collect(rules)


def _collect(rules: dict) -> pd.DataFrame:
    """Flatten boolean rule masks into long-form (row_index, rule) failures."""
    parts = [pd.DataFrame({"row_index": mask[mask].index, "rule": name}) for name, mask in rules.items()]
    return pd.concat(parts, ignore_index=True)


def _split(frame: pd.DataFrame, failures: pd.DataFrame):
    """Split a table into clean rows and an error table with reasons."""
    reasons = failures.groupby("row_index")["rule"].agg(lambda r: ";".join(sorted(set(r))))
    bad = frame.loc[reasons.index].copy()
    bad["error_reasons"] = reasons
    return frame.drop(index=reasons.index), bad


REPAIRABLE_USER_RULES = {"wrong_dtype_age", "age_out_of_range"}


def _split_users(users: pd.DataFrame, failures: pd.DataFrame):
    """Quarantine unusable users; keep users whose only fault is a bad age (age set to NaN).

    Dropping such users would orphan all of their interactions, so the age is blanked
    (imputed later in preparation) while the original row is still logged as an error.
    """
    reasons = failures.groupby("row_index")["rule"].agg(set)
    repairable = reasons[reasons.apply(lambda r: r <= REPAIRABLE_USER_RULES)].index
    hard = failures[~failures["row_index"].isin(repairable)]
    clean, bad = _split(users, hard)
    logged = users.loc[repairable].copy()
    logged["error_reasons"] = reasons.loc[repairable].apply(lambda r: ";".join(sorted(r))) + " (repaired: age blanked)"
    clean = clean.copy()
    clean.loc[clean.index.isin(repairable), "age"] = pd.NA
    return clean, pd.concat([bad, logged])


def run(config: dict) -> dict:
    """Validate raw CSVs, write clean/error datasets and the validation report."""
    raw, processed = resolve_path(config, "raw"), resolve_path(config, "processed")
    errors, reports = resolve_path(config, "errors"), resolve_path(config, "reports")
    cfg = config["validation"]
    tables = {n: pd.read_csv(raw / f"{n}.csv", dtype=str) for n in ("users", "items", "interactions")}

    # Dimension tables are validated first so interactions can be checked against clean keys.
    fail_users = validate_users(tables["users"], cfg)
    clean_users, bad_users = _split_users(tables["users"], fail_users)
    fail_items = validate_items(tables["items"])
    clean_items, bad_items = _split(tables["items"], fail_items)
    fail_inter = validate_interactions(tables["interactions"], clean_users, clean_items, cfg)
    clean_inter, bad_inter = _split(tables["interactions"], fail_inter)

    summary = {}
    for name, clean, bad, fails in (
        ("users", clean_users, bad_users, fail_users),
        ("items", clean_items, bad_items, fail_items),
        ("interactions", clean_inter, bad_inter, fail_inter),
    ):
        clean.to_csv(processed / f"{name}_validated.csv", index=False)
        bad.to_csv(errors / f"{name}_errors.csv", index=False)
        summary[name] = {
            "rows_in": len(tables[name]),
            "rows_valid": len(clean),
            "rows_rejected": len(bad),
            "rule_violations": fails["rule"].value_counts().to_dict(),
        }
        LOGGER.info("%s: %d valid, %d rejected", name, len(clean), len(bad))

    (reports / "validation_report.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(summary, reports / "validation_report.md")
    return summary


def _write_markdown(summary: dict, path) -> None:
    """Render the validation summary as a Markdown report."""
    lines = ["# Data Validation Report", ""]
    for table, info in summary.items():
        lines += [f"## {table}", "",
                  f"- Rows in: {info['rows_in']}", f"- Rows passed downstream: {info['rows_valid']}",
                  f"- Rows logged to error dataset: {info['rows_rejected']}", "",
                  "| Rule | Violations |", "|---|---|"]
        rows = [f"| {rule} | {count} |" for rule, count in info["rule_violations"].items()]
        lines += rows or ["| none | 0 |"]
        lines.append("")
    lines.append("Note: a row can violate several rules, so rule counts may exceed row counts. Users with only an invalid age are repaired (age blanked, imputed later) and kept.")
    path.write_text("\n".join(lines))
