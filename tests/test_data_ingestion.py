import pandas as pd

from src.components.data_ingestion import ingest_data


def test_ingest_data_combines_and_creates_timestamp(tmp_path, test_config):
    raw_1 = tmp_path / "fire_1.csv"
    raw_2 = tmp_path / "fire_2.csv"

    df1 = pd.DataFrame(
        {
            "latitude": [41.9],
            "longitude": [12.5],
            "bright_ti4": [300.1],
            "scan": [0.4],
            "track": [0.5],
            "acq_date": ["2023-01-01"],
            "acq_time": [134],
            "satellite": ["N"],
            "instrument": ["VIIRS"],
            "confidence": ["n"],
            "version": [2],
            "bright_ti5": [280.1],
            "frp": [1.2],
            "daynight": ["N"],
            "type": [2],
        }
    )

    df2 = pd.DataFrame(
        {
            "latitude": [42.0],
            "longitude": [12.6],
            "bright_ti4": [305.2],
            "scan": [0.4],
            "track": [0.5],
            "acq_date": ["2023-01-02"],
            "acq_time": [245],
            "satellite": ["N"],
            "instrument": ["VIIRS"],
            "confidence": ["n"],
            "version": [2],
            "bright_ti5": [281.2],
            "frp": [1.5],
            "daynight": ["N"],
            "type": [3],
        }
    )

    df1.to_csv(raw_1, index=False)
    df2.to_csv(raw_2, index=False)

    test_config["data"]["raw_paths"] = [str(raw_1), str(raw_2)]

    ingested_df = ingest_data(test_config)

    assert ingested_df.shape[0] == 2
    assert "acq_timestamp" in ingested_df.columns
    assert "source_file" in ingested_df.columns
