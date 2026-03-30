from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_training_dataset(config: dict) -> pd.DataFrame:
    positive_base_path = Path(config["data"]["positive_base_path"])
    negative_samples_path = Path(config["data"]["negative_samples_path"])
    output_path = Path(config["data"]["training_dataset_path"])

    if not positive_base_path.exists():
        raise FileNotFoundError(
            f"Positive fire base dataset not found: {positive_base_path}"
        )

    if not negative_samples_path.exists():
        raise FileNotFoundError(
            f"Negative samples dataset not found: {negative_samples_path}"
        )

    logger.info("Loading positive fire base from %s", positive_base_path)
    positive_df = pd.read_csv(positive_base_path)

    logger.info("Loading negative samples from %s", negative_samples_path)
    negative_df = pd.read_csv(negative_samples_path)

    logger.info("Positive dataset shape: %s", positive_df.shape)
    logger.info("Negative dataset shape: %s", negative_df.shape)

    required_columns = ["date", "latitude", "longitude", "fire_occurred"]

    for name, df in [("positive", positive_df), ("negative", negative_df)]:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {name} dataset: {missing}")

    positive_core = positive_df[required_columns].copy()
    negative_core = negative_df[required_columns].copy()

    positive_core["date"] = pd.to_datetime(positive_core["date"], errors="coerce")
    negative_core["date"] = pd.to_datetime(negative_core["date"], errors="coerce")

    training_df = pd.concat([positive_core, negative_core], ignore_index=True)

    training_df = training_df.dropna(
        subset=["date", "latitude", "longitude", "fire_occurred"]
    )

    training_df = training_df.drop_duplicates().sort_values(
        ["date", "latitude", "longitude"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_path, index=False)

    logger.info("Training dataset saved to %s", output_path)
    logger.info("Training dataset shape: %s", training_df.shape)
    logger.info(
        "Training dataset class balance: %s",
        training_df["fire_occurred"].value_counts().to_dict(),
    )

    return training_df