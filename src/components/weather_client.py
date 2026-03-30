import requests


class OpenMeteoHistoricalClient:
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch_daily_weather(
        self,
        latitude: float,
        longitude: float,
        date: str,
        daily_variables: list[str],
        timezone: str = "GMT",
    ) -> dict:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date,
            "end_date": date,
            "daily": ",".join(daily_variables),
            "timezone": timezone,
        }

        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
