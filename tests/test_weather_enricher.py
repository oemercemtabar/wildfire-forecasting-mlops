import pandas as pd

from src.components.weather_enricher import enrich_with_weather


class DummyWeatherClient:
    def __init__(self, timeout=30):
        self.timeout = timeout

    def fetch_daily_weather(self, latitude, longitude, date, daily_variables, timezone="GMT"):
        return {
            "daily": {
                "temperature_2m_mean": [12.5],
                "relative_humidity_2m_mean": [70],
                "precipitation_sum": [0.0],
                "wind_speed_10m_max": [11.2],
            }
        }


def test_weather_enricher_adds_weather_columns(monkeypatch, sample_positive_base_df, test_config, test_params):
    training_df = sample_positive_base_df.copy()
    training_df.to_csv(test_config["data"]["training_dataset_path"], index=False)

    monkeypatch.setattr(
        "src.components.weather_enricher.OpenMeteoHistoricalClient",
        DummyWeatherClient,
    )

    enriched_df = enrich_with_weather(test_config, test_params)

    assert "temperature_2m_mean" in enriched_df.columns
    assert "relative_humidity_2m_mean" in enriched_df.columns
    assert "precipitation_sum" in enriched_df.columns
    assert "wind_speed_10m_max" in enriched_df.columns
    assert enriched_df["temperature_2m_mean"].notna().all()

class FailingWeatherClient:
    def __init__(self, timeout=30):
        self.timeout = timeout

    def fetch_daily_weather(self, latitude, longitude, date, daily_variables, timezone="GMT"):
        raise RuntimeError("API failure")


def test_weather_enricher_handles_api_failure(monkeypatch, sample_positive_base_df, test_config, test_params):
    training_df = sample_positive_base_df.copy()
    training_df.to_csv(test_config["data"]["training_dataset_path"], index=False)

    monkeypatch.setattr(
        "src.components.weather_enricher.OpenMeteoHistoricalClient",
        FailingWeatherClient,
    )

    enriched_df = enrich_with_weather(test_config, test_params)

    assert enriched_df["temperature_2m_mean"].isna().all()
    assert enriched_df["relative_humidity_2m_mean"].isna().all()