from src.components.feature_eng import build_features
from src.components.model_trainer import train_model


def test_model_trainer_saves_model_and_metrics(
    sample_training_df, test_config, test_params
):
    sample_training_df.to_csv(
        test_config["data"]["weather_enriched_path"],
        index=False,
    )

    X_train, X_test, y_train, y_test = build_features(test_config, test_params)
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
    assert "selection_metric" in benchmark_payload
    assert "models" in benchmark_payload
    assert len(benchmark_payload["models"]) >= 1

    best_model_name = benchmark_payload["best_model_name"]
    best_metrics = benchmark_payload["models"][best_model_name]

    assert "accuracy" in best_metrics
    assert "precision" in best_metrics
    assert "recall" in best_metrics
    assert "f1" in best_metrics
    assert "roc_auc" in best_metrics
