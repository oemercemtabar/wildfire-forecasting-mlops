from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_features(config: dict, params: dict):
    input_path = Path(config["data"]["weather_enriched_path"])
    artifacts = config["artifacts"]

    if not input_path.exists():
        raise FileNotFoundError(f"Weather-enriched dataset not found: {input_path}")

    logger.info("Loading weather-enriched dataset from %s", input_path)
    df = pd.read_csv(input_path)

    required_columns = [
        "date",
        "latitude",
        "longitude",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "fire_occurred",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    df = df.dropna(subset=required_columns + ["month", "day_of_year"]).reset_index(
        drop=True
    )

    feature_columns = [
        "latitude",
        "longitude",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "month",
        "day_of_year",
    ]

    X = df[feature_columns].copy()
    y = df["fire_occurred"].copy()

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    Path(artifacts["X_train_path"]).parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(artifacts["X_train_path"], index=False)
    X_test.to_csv(artifacts["X_test_path"], index=False)
    y_train.to_csv(artifacts["y_train_path"], index=False)
    y_test.to_csv(artifacts["y_test_path"], index=False)

    logger.info("Feature engineering completed")
    logger.info("X_train shape: %s", X_train.shape)
    logger.info("X_test shape: %s", X_test.shape)
    logger.info("y_train distribution: %s", y_train.value_counts().to_dict())
    logger.info("y_test distribution: %s", y_test.value_counts().to_dict())

    return X_train, X_test, y_train, y_test
