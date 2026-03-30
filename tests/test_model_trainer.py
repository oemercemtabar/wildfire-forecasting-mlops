from src.components.feature_eng import build_features
from src.components.model_trainer import train_model


def test_model_trainer_saves_model_and_metrics(sample_training_df, test_config, test_params):
    sample_training_df.to_csv(test_config["data"]["weather_enriched_path"], index=False)

    X_train, X_test, y_train, y_test = build_features(test_config, test_params)
    model, metrics = train_model(X_train, X_test, y_train, y_test, test_config, test_params)

    assert model is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics