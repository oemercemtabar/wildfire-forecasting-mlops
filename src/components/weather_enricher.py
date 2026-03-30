from pathlib import Path
from typing import Any
import json

import pandas as pd

from src.components.weather_client import OpenMeteoHistoricalClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_weather_key(date: str, latitude: float, longitude: float) -> str:
    return f"{date}_{latitude:.2f}_{longitude:.2f}"


def _cache_file_path(
    cache_dir: Path, date: str, latitude: float, longitude: float
) -> Path:
    key = _build_weather_key(date, latitude, longitude)
    return cache_dir / f"{key}.json"


def _load_cached_weather(cache_file: Path) -> dict[str, Any] | None:
    if not cache_file.exists():
        return None

    with cache_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_cached_weather(cache_file: Path, data: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with cache_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _parse_daily_weather_response(
    response: dict[str, Any],
    daily_variables: list[str],
) -> dict[str, Any]:
    daily_data = response.get("daily", {})

    parsed: dict[str, Any] = {}
    for variable in daily_variables:
        values = daily_data.get(variable, [])
        parsed[variable] = values[0] if values else None

    return parsed


def enrich_with_weather(config: dict, params: dict) -> pd.DataFrame:
    training_dataset_path = Path(config["data"]["training_dataset_path"])
    output_path = Path(config["data"]["weather_enriched_path"])
    cache_dir = Path(config["artifacts"]["weather_cache_dir"])

    if not training_dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {training_dataset_path}")

    weather_cfg = params["weather"]
    daily_variables = weather_cfg["daily_variables"]
    timezone = weather_cfg.get("timezone", "GMT")
    rounding_decimals = int(weather_cfg.get("coordinate_rounding_decimals", 2))
    timeout = int(weather_cfg.get("request_timeout_seconds", 30))
    max_rows = weather_cfg.get("max_rows")
    max_unique_requests = weather_cfg.get("max_unique_requests")

    logger.info("Loading training dataset from %s", training_dataset_path)
    df = pd.read_csv(training_dataset_path)

    required_columns = ["date", "latitude", "longitude", "fire_occurred"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training dataset: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "latitude", "longitude"]).reset_index(drop=True)

    if max_rows is not None:
        df = df.head(int(max_rows)).copy()
        logger.info("Using only first %s rows for enrichment test", len(df))

    df["latitude_rounded"] = df["latitude"].round(rounding_decimals)
    df["longitude_rounded"] = df["longitude"].round(rounding_decimals)
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    unique_requests = (
        df[["date_str", "latitude_rounded", "longitude_rounded"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if max_unique_requests is not None:
        unique_requests = unique_requests.head(int(max_unique_requests)).copy()
        logger.info(
            "Limiting enrichment to first %s unique weather requests",
            len(unique_requests),
        )

    logger.info("Unique weather requests to process: %s", len(unique_requests))

    allowed_keys = set(
        zip(
            unique_requests["date_str"],
            unique_requests["latitude_rounded"],
            unique_requests["longitude_rounded"],
        )
    )

    df = df[
        df.apply(
            lambda row: (
                row["date_str"],
                row["latitude_rounded"],
                row["longitude_rounded"],
            )
            in allowed_keys,
            axis=1,
        )
    ].copy()

    client = OpenMeteoHistoricalClient(timeout=timeout)
    weather_cache: dict[str, dict[str, Any]] = {}
    weather_rows: list[dict[str, Any]] = []

    cache_hits_memory = 0
    cache_hits_disk = 0
    api_calls = 0

    for idx, row in unique_requests.iterrows():
        date_str = row["date_str"]
        latitude = float(row["latitude_rounded"])
        longitude = float(row["longitude_rounded"])

        cache_key = _build_weather_key(date_str, latitude, longitude)
        cache_file = _cache_file_path(cache_dir, date_str, latitude, longitude)

        if cache_key in weather_cache:
            parsed_weather = weather_cache[cache_key]
            cache_hits_memory += 1
        else:
            cached_weather = _load_cached_weather(cache_file)

            if cached_weather is not None:
                parsed_weather = cached_weather
                weather_cache[cache_key] = parsed_weather
                cache_hits_disk += 1
            else:
                try:
                    response = client.fetch_daily_weather(
                        latitude=latitude,
                        longitude=longitude,
                        date=date_str,
                        daily_variables=daily_variables,
                        timezone=timezone,
                    )
                    parsed_weather = _parse_daily_weather_response(
                        response=response,
                        daily_variables=daily_variables,
                    )
                    weather_cache[cache_key] = parsed_weather
                    _save_cached_weather(cache_file, parsed_weather)
                    api_calls += 1
                except Exception as exc:
                    logger.warning(
                        "Weather request failed for one weather lookup on date=%s: %s",
                        date_str,
                        exc,
                    )
                    parsed_weather = {variable: None for variable in daily_variables}
                    weather_cache[cache_key] = parsed_weather
                    _save_cached_weather(cache_file, parsed_weather)

        weather_rows.append(
            {
                "date_str": date_str,
                "latitude_rounded": latitude,
                "longitude_rounded": longitude,
                **parsed_weather,
            }
        )

        if (idx + 1) % 100 == 0:
            logger.info(
                "Processed %s / %s weather requests",
                idx + 1,
                len(unique_requests),
            )

    weather_df = pd.DataFrame(weather_rows)

    enriched_df = df.merge(
        weather_df,
        on=["date_str", "latitude_rounded", "longitude_rounded"],
        how="left",
    )

    enriched_df = enriched_df.drop(
        columns=["date_str", "latitude_rounded", "longitude_rounded"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(output_path, index=False)

    logger.info("Weather-enriched dataset saved to %s", output_path)
    logger.info("Weather-enriched dataset shape: %s", enriched_df.shape)
    logger.info(
        "Weather cache stats | memory_hits=%s disk_hits=%s api_calls=%s",
        cache_hits_memory,
        cache_hits_disk,
        api_calls,
    )

    return enriched_df
