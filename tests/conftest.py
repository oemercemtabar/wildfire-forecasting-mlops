from pathlib import Path
import pandas as pd
import pytest
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def sample_fire_events_df():
    return pd.DataFrame(
        {
            "latitude": [41.9, 42.0],
            "longitude": [12.5, 12.6],
            "bright_ti4": [300.1, 305.2],
            "scan": [0.4, 0.4],
            "track": [0.5, 0.5],
            "acq_date": ["2023-01-01", "2023-01-02"],
            "acq_time": [134, 245],
            "satellite": ["N", "N"],
            "instrument": ["VIIRS", "VIIRS"],
            "confidence": ["n", "n"],
            "version": [2, 2],
            "bright_ti5": [280.1, 281.2],
            "frp": [1.2, 1.5],
            "daynight": ["N", "N"],
            "type": [2, 3],
            "source_file": ["file1.csv", "file2.csv"],
            "acq_timestamp": ["2023-01-01 01:34:00", "2023-01-02 02:45:00"],
        }
    )


@pytest.fixture
def sample_positive_base_df():
    return pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "latitude": [41.9, 42.0],
            "longitude": [12.5, 12.6],
            "fire_occurred": [1, 1],
        }
    )


@pytest.fixture
def sample_training_df():
    return pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "latitude": [41.9, 42.0, 42.1, 42.2],
            "longitude": [12.5, 12.6, 12.7, 12.8],
            "fire_occurred": [1, 0, 1, 0],
            "temperature_2m_mean": [12.0, 10.0, 13.5, 9.5],
            "relative_humidity_2m_mean": [70, 75, 65, 80],
            "precipitation_sum": [0.0, 1.2, 0.0, 2.0],
            "wind_speed_10m_max": [12.0, 15.0, 10.0, 18.0],
        }
    )


@pytest.fixture
def test_config(tmp_path: Path):
    data_dir = tmp_path / "data"
    artifacts_dir = data_dir / "artifacts"

    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return {
        "data": {
            "raw_paths": [],
            "interim_path": str(data_dir / "interim.csv"),
            "validated_path": str(data_dir / "validated.csv"),
            "positive_base_path": str(data_dir / "positive_fire_base.csv"),
            "negative_samples_path": str(data_dir / "negative_samples.csv"),
            "training_dataset_path": str(data_dir / "training_dataset.csv"),
            "weather_enriched_path": str(data_dir / "weather_enriched.csv"),
        },
        "artifacts": {
            "validation_report_path": str(artifacts_dir / "validation_report.json"),
            "model_path": str(artifacts_dir / "model.joblib"),
            "metrics_path": str(artifacts_dir / "metrics.json"),
            "X_train_path": str(artifacts_dir / "X_train.csv"),
            "X_test_path": str(artifacts_dir / "X_test.csv"),
            "y_train_path": str(artifacts_dir / "y_train.csv"),
            "y_test_path": str(artifacts_dir / "y_test.csv"),
            "weather_cache_dir": str(artifacts_dir / "weather_cache"),
            "weather_summary_path": str(
                artifacts_dir / "weather_enrichment_summary.json"
            ),
            "failed_weather_requests_path": str(
                artifacts_dir / "failed_weather_requests.csv"
            ),
        },
    }


@pytest.fixture
def test_params():
    return {
        "split": {"test_size": 0.5, "random_state": 42},
        "model": {"name": "logistic_regression", "max_iter": 1000},
        "validation": {"max_missing_ratio": 0.2},
        "negative_sampling": {
            "multiplier": 1.0,
            "min_latitude": 36.0,
            "max_latitude": 47.5,
            "min_longitude": 6.0,
            "max_longitude": 18.8,
            "rounding_decimals": 2,
        },
        "weather": {
            "daily_variables": [
                "temperature_2m_mean",
                "relative_humidity_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
            ],
            "timezone": "GMT",
            "coordinate_rounding_decimals": 2,
            "request_timeout_seconds": 30,
            "max_rows": None,
            "max_unique_requests": None,
        },
    }
