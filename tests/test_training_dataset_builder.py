import pandas as pd

from src.components.training_dataset_builder import build_training_dataset


def test_build_training_dataset_combines_positive_and_negative(test_config):
    positive_df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "latitude": [41.9, 42.0],
            "longitude": [12.5, 12.6],
            "fire_occurred": [1, 1],
        }
    )

    negative_df = pd.DataFrame(
        {
            "date": ["2023-01-03", "2023-01-04"],
            "latitude": [42.1, 42.2],
            "longitude": [12.7, 12.8],
            "fire_occurred": [0, 0],
        }
    )

    positive_df.to_csv(test_config["data"]["positive_base_path"], index=False)
    negative_df.to_csv(test_config["data"]["negative_samples_path"], index=False)

    training_df = build_training_dataset(test_config)

    assert not training_df.empty
    assert set(training_df["fire_occurred"].unique()) == {0, 1}
    assert training_df.shape[0] == 4
