from src.utils.config import load_yaml


def test_load_config_files():
    config = load_yaml("configs/config.yaml")
    params = load_yaml("configs/params.yaml")

    assert "data" in config
    assert "artifacts" in config
    assert "split" in params
    assert "model" in params