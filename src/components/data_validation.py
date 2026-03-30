from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.common import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "frp",
    "daynight",
    "type",
]


def _calculate_missing_ratios(df: pd.DataFrame) -> dict[str, float]:
    return {col: float(df[col].isna().mean()) for col in df.columns}


def _validate_required_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def _validate_ranges(df: pd.DataFrame) -> dict[str, Any]:
    issues: dict[str, Any] = {}

    if "latitude" in df.columns:
        invalid_lat = int(((df["latitude"] < -90) | (df["latitude"] > 90)).fillna(False).sum())
        if invalid_lat > 0:
            issues["latitude_out_of_range"] = invalid_lat

    if "longitude" in df.columns:
        invalid_lon = int(((df["longitude"] < -180) | (df["longitude"] > 180)).fillna(False).sum())
        if invalid_lon > 0:
            issues["longitude_out_of_range"] = invalid_lon

    if "frp" in df.columns:
        invalid_frp = int((df["frp"] < 0).fillna(False).sum())
        if invalid_frp > 0:
            issues["negative_frp"] = invalid_frp

    return issues


def validate_data(df: pd.DataFrame, config: dict, params: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    validated_path = Path(config["data"]["validated_path"])
    report_path = Path(config["artifacts"]["validation_report_path"])
    max_missing_ratio = params["validation"]["max_missing_ratio"]

    logger.info("Starting data validation on dataframe with shape: %s", df.shape)

    missing_required_columns = _validate_required_columns(df)
    if missing_required_columns:
        raise ValueError(f"Missing required columns: {missing_required_columns}")

    missing_ratios = _calculate_missing_ratios(df)
    range_issues = _validate_ranges(df)

    high_missing_columns = {
        col: ratio
        for col, ratio in missing_ratios.items()
        if ratio > max_missing_ratio
    }

    duplicate_count = int(df.duplicated().sum())

    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required_columns": missing_required_columns,
        "missing_ratios": missing_ratios,
        "high_missing_columns": high_missing_columns,
        "duplicate_rows": duplicate_count,
        "range_issues": range_issues,
    }

    if duplicate_count > 0:
        logger.warning("Validation found %s duplicate rows", duplicate_count)

    if high_missing_columns:
        logger.warning("Columns above missing-value threshold: %s", high_missing_columns)

    if range_issues:
        logger.warning("Range validation issues found: %s", range_issues)

    validated_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(validated_path, index=False)
    save_json(report, report_path)

    logger.info("Validated dataset saved to %s", validated_path)
    logger.info("Validation report saved to %s", report_path)

    return df, report