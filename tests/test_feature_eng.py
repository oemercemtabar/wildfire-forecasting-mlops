from src.components.feature_eng import build_features


def test_feature_engineering_creates_expected_splits(sample_training_df, test_config, test_params):
    sample_training_df.to_csv(test_config["data"]["weather_enriched_path"], index=False)

    X_train, X_test, y_train, y_test = build_features(test_config, test_params)

    assert not X_train.empty
    assert not X_test.empty
    assert "month" in X_train.columns
    assert "day_of_year" in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})