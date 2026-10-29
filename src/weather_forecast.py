"""Fetch daily weather forecast data for irrigation planning."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch_daily_forecast(location: str, forecast_days: int = 7) -> tuple[str, list[dict[str, float | str]]]:
    """Resolve an Indian location and fetch its daily Open-Meteo forecast."""
    location = location.strip()
    if not location:
        raise ValueError("Enter a city, district, or town.")
    if not 1 <= forecast_days <= 16:
        raise ValueError("Forecast days must be between 1 and 16.")

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
        {"name": location, "count": 1, "language": "en", "format": "json", "countryCode": "IN"}
    )
    try:
        with urlopen(geocoding_url, timeout=15) as response:
            geocoding = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Location lookup failed: {error}") from error

    results = geocoding.get("results") or []
    if not results:
        raise ValueError(f"No Indian location matched '{location}'. Try a nearby city or district.")

    place = results[0]
    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urlencode(
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": (
                "temperature_2m_max,relative_humidity_2m_mean,precipitation_sum,"
                "et0_fao_evapotranspiration"
            ),
            "forecast_days": forecast_days,
            "timezone": "auto",
        }
    )
    try:
        with urlopen(forecast_url, timeout=15) as response:
            forecast = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Weather forecast could not be loaded: {error}") from error

    daily = forecast.get("daily") or {}
    required = (
        "time",
        "temperature_2m_max",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
    )
    if any(key not in daily for key in required):
        raise RuntimeError("The weather service returned incomplete daily forecast data.")

    days: list[dict[str, float | str]] = []
    for values in zip(*(daily[key] for key in required)):
        day, temperature, humidity, rainfall, et0 = values
        if any(value is None for value in (temperature, humidity, rainfall, et0)):
            continue
        days.append(
            {
                "date": str(day),
                "temperature_c": float(temperature),
                "humidity_pct": float(humidity),
                "rain_mm": float(rainfall),
                "et0_mm": float(et0),
            }
        )
    if not days:
        raise RuntimeError("The weather service returned no complete forecast days.")

    place_name = ", ".join(
        part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part
    )
    return place_name, days
