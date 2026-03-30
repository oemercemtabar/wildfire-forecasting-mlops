
import pandas as pd
import pytest

from src.components.data_validation import validate_data


def test_validate_data_passes_with_valid_dataframe(sample_fire_events_df, test_config, test_params):
    validated_df, report = validate_data(sample_fire_events_df, test_config, test_params)

    assert validated_df.shape[0] == 2
    assert report["missing_required_columns"] == []
    assert report["duplicate_rows"] == 0
    assert report["range_issues"] == {}



def test_validate_data_raises_for_missing_required_columns(test_config, test_params):
    df = pd.DataFrame({"latitude": [41.9], "longitude": [12.5]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df, test_config, test_params)