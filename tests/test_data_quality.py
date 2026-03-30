from src.components.data_validation import validate_data


def test_data_quality_report_has_expected_keys(
    sample_fire_events_df,
    test_config,
    test_params,
):
    _, report = validate_data(sample_fire_events_df, test_config, test_params)

    expected_keys = {
        "row_count",
        "column_count",
        "missing_required_columns",
        "missing_ratios",
        "high_missing_columns",
        "duplicate_rows",
        "range_issues",
    }

    assert expected_keys.issubset(report.keys())
    assert report["missing_required_columns"] == []
