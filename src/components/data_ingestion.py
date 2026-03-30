from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def _build_acquisition_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "acq_date" not in df.columns or "acq_time" not in df.columns:
        return df

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    acq_time_str = (
        df["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )

    hours = acq_time_str.str[:2]
    minutes = acq_time_str.str[2:4]

    timestamp_str = df["acq_date"].dt.strftime("%Y-%m-%d") + " " + hours + ":" + minutes

    df["acq_timestamp"] = pd.to_datetime(timestamp_str, errors="coerce")

    return df


def ingest_data(config: dict) -> pd.DataFrame:
    raw_paths = [Path(path) for path in config["data"]["raw_paths"]]
    interim_path = Path(config["data"]["interim_path"])

    missing_files = [str(path) for path in raw_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Raw data file(s) not found: {', '.join(missing_files)}"
        )

    dataframes: list[pd.DataFrame] = []

    for raw_path in raw_paths:
        logger.info("Loading raw data from %s", raw_path)
        df = pd.read_csv(raw_path)
        logger.info("Loaded %s with shape %s", raw_path.name, df.shape)

        df = _normalize_column_names(df)
        df["source_file"] = raw_path.name

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    logger.info("Combined dataset shape before parsing: %s", combined_df.shape)

    combined_df = _build_acquisition_timestamp(combined_df)

    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    after_dedup = len(combined_df)

    logger.info("Removed %s duplicate rows", before_dedup - after_dedup)
    logger.info("Combined dataset shape after deduplication: %s", combined_df.shape)

    interim_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(interim_path, index=False)

    logger.info("Ingested dataset saved to %s", interim_path)

    return combined_df
