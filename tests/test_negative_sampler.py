from src.components.negative_sampler import sample_negative_examples


def test_negative_sampler_creates_zero_class_rows(
    sample_positive_base_df, test_config, test_params
):
    negative_df = sample_negative_examples(
        sample_positive_base_df, test_config, test_params
    )

    assert not negative_df.empty
    assert negative_df["fire_occurred"].eq(0).all()
    assert {"date", "latitude", "longitude", "fire_occurred"}.issubset(
        negative_df.columns
    )


def test_negative_sampler_avoids_positive_overlap(
    sample_positive_base_df, test_config, test_params
):
    negative_df = sample_negative_examples(
        sample_positive_base_df, test_config, test_params
    )

    positive_keys = set(
        zip(
            sample_positive_base_df["date"],
            sample_positive_base_df["latitude"].round(2),
            sample_positive_base_df["longitude"].round(2),
        )
    )
    negative_keys = set(
        zip(
            negative_df["date"].astype(str),
            negative_df["latitude"].round(2),
            negative_df["longitude"].round(2),
        )
    )

    assert positive_keys.isdisjoint(negative_keys)
