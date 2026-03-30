import time

from src.components.weather_client import OpenMeteoHistoricalClient


def test_weather_client_initialization_is_fast():
    start = time.perf_counter()
    client = OpenMeteoHistoricalClient(timeout=30)
    elapsed = time.perf_counter() - start

    assert client is not None
    assert elapsed < 0.1
