from src.components.dataset_builder import build_positive_fire_base


def test_build_positive_fire_base_creates_positive_labels(
    sample_fire_events_df,
    test_config,
):
    positive_df = build_positive_fire_base(sample_fire_events_df, test_config)

    assert not positive_df.empty
    assert positive_df["fire_occurred"].eq(1).all()
    assert {"date", "latitude", "longitude", "fire_occurred"}.issubset(
        positive_df.columns
    )