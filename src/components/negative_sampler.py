from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def sample_negative_examples(
    positive_df: pd.DataFrame, config: dict, params: dict
) -> pd.DataFrame:
    output_path = Path(config["data"]["negative_samples_path"])
    sampling_cfg = params["negative_sampling"]

    required_columns = ["date", "latitude", "longitude", "fire_occurred"]
    missing = [col for col in required_columns if col not in positive_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in positive dataframe: {missing}")

    positive_df = positive_df.copy()
    positive_df["date"] = pd.to_datetime(positive_df["date"], errors="coerce")

    positive_df = positive_df.dropna(subset=["date", "latitude", "longitude"])

    random_state = params["split"]["random_state"]
    rng = np.random.default_rng(random_state)

    multiplier = float(sampling_cfg["multiplier"])
    sample_size = int(len(positive_df) * multiplier)

    if sample_size <= 0:
        raise ValueError("Negative sample size must be greater than 0.")

    logger.info("Generating %s negative samples", sample_size)

    sampled_dates = (
        positive_df["date"]
        .sample(
            n=sample_size,
            replace=True,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    min_lat = float(sampling_cfg["min_latitude"])
    max_lat = float(sampling_cfg["max_latitude"])
    min_lon = float(sampling_cfg["min_longitude"])
    max_lon = float(sampling_cfg["max_longitude"])
    rounding_decimals = int(sampling_cfg["rounding_decimals"])

    negative_df = pd.DataFrame(
        {
            "date": sampled_dates,
            "latitude": rng.uniform(min_lat, max_lat, sample_size),
            "longitude": rng.uniform(min_lon, max_lon, sample_size),
            "fire_occurred": 0,
        }
    )

    negative_df["latitude"] = negative_df["latitude"].round(rounding_decimals)
    negative_df["longitude"] = negative_df["longitude"].round(rounding_decimals)

    positive_keys = set(
        zip(
            positive_df["date"].dt.strftime("%Y-%m-%d"),
            positive_df["latitude"].round(rounding_decimals),
            positive_df["longitude"].round(rounding_decimals),
        )
    )

    negative_df["_date_key"] = pd.to_datetime(negative_df["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    negative_df["_lat_key"] = negative_df["latitude"].round(rounding_decimals)
    negative_df["_lon_key"] = negative_df["longitude"].round(rounding_decimals)

    before_filter = len(negative_df)
    negative_df = negative_df[
        ~negative_df.apply(
            lambda row: (row["_date_key"], row["_lat_key"], row["_lon_key"])
            in positive_keys,
            axis=1,
        )
    ].copy()
    after_filter = len(negative_df)

    logger.info(
        "Removed %s negative samples overlapping with positive keys",
        before_filter - after_filter,
    )

    negative_df = negative_df.drop(columns=["_date_key", "_lat_key", "_lon_key"])
    negative_df = negative_df.drop_duplicates().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    negative_df.to_csv(output_path, index=False)

    logger.info("Negative samples saved to %s", output_path)
    logger.info("Negative samples shape: %s", negative_df.shape)

    return negative_df
