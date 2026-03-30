
from src.components.dataset_builder import build_positive_fire_base
from src.components.feature_eng import build_features
from src.components.model_trainer import train_model
from src.components.negative_sampler import sample_negative_examples
from src.components.training_dataset_builder import build_training_dataset
from src.components.weather_enricher import enrich_with_weather


class DummyWeatherClient:
    def __init__(self, timeout=30):
        self.timeout = timeout

    def fetch_daily_weather(
        self,
        latitude,
        longitude,
        date,
        daily_variables,
        timezone="GMT",
    ):
        return {
            "daily": {
                "temperature_2m_mean": [12.5],
                "relative_humidity_2m_mean": [70],
                "precipitation_sum": [0.0],
                "wind_speed_10m_max": [11.2],
            }
        }


def test_pipeline_integration_runs_end_to_end(
    monkeypatch,
    sample_fire_events_df,
    test_config,
    test_params,
):
    validated_df = sample_fire_events_df.copy()

    positive_df = build_positive_fire_base(validated_df, test_config)
    assert not positive_df.empty
    assert positive_df["fire_occurred"].eq(1).all()

    negative_df = sample_negative_examples(positive_df, test_config, test_params)
    assert not negative_df.empty
    assert negative_df["fire_occurred"].eq(0).all()

    training_df = build_training_dataset(test_config)
    assert not training_df.empty
    assert set(training_df["fire_occurred"].unique()) == {0, 1}

    monkeypatch.setattr(
        "src.components.weather_enricher.OpenMeteoHistoricalClient",
        DummyWeatherClient,
    )

    enriched_df = enrich_with_weather(test_config, test_params)
    assert not enriched_df.empty
    assert "temperature_2m_mean" in enriched_df.columns

    X_train, X_test, y_train, y_test = build_features(test_config, test_params)
    assert not X_train.empty
    assert not X_test.empty

    model, benchmark_payload = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        test_config,
        test_params,
    )

    assert model is not None
    assert "best_model_name" in benchmark_payload
    assert "models" in benchmark_payload

    best_model_name = benchmark_payload["best_model_name"]
    best_metrics = benchmark_payload["models"][best_model_name]

    assert "accuracy" in best_metrics
    assert "f1" in best_metrics
    assert "roc_auc" in best_metrics