from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_positive_fire_base(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    output_path = Path(config["data"]["positive_base_path"])

    required_columns = ["acq_date", "latitude", "longitude"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for dataset building: {missing}")

    positive_df = df.copy()

    positive_df["date"] = pd.to_datetime(
        positive_df["acq_date"], errors="coerce"
    ).dt.date

    positive_df["fire_occurred"] = 1

    columns_to_keep = [
        "date",
        "latitude",
        "longitude",
        "fire_occurred",
        "acq_time",
        "acq_timestamp",
        "frp",
        "confidence",
        "daynight",
        "type",
        "source_file",
    ]

    available_columns = [col for col in columns_to_keep if col in positive_df.columns]
    positive_df = positive_df[available_columns].copy()

    positive_df = positive_df.dropna(subset=["date", "latitude", "longitude"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    positive_df.to_csv(output_path, index=False)

    logger.info("Positive fire base dataset saved to %s", output_path)
    logger.info("Positive fire base shape: %s", positive_df.shape)

    return positive_df
