"""Weather data ingestion from Open-Meteo API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import httpx

from app.core.config import settings


# Open-Meteo hourly variables we need
_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "direct_radiation",
    "uv_index",
]


async def fetch_current_weather(lat: float, lon: float) -> dict:
    """Fetch current + next-24h hourly weather from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_HOURLY_VARS),
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_gusts_10m",
            "cloud_cover",
            "surface_pressure",
            "precipitation",
            "uv_index",
        ]),
        "forecast_days": 2,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{settings.open_meteo_base_url}/forecast", params=params)
        resp.raise_for_status()
        return resp.json()


def parse_current_conditions(data: dict) -> dict:
    """Extract current conditions into a flat dict matching our feature names."""
    cur = data.get("current", {})
    return {
        "temp_c": cur.get("temperature_2m", 0.0),
        "humidity_pct": cur.get("relative_humidity_2m", 0.0),
        "wind_speed_ms": cur.get("wind_speed_10m", 0.0) / 3.6,  # km/h → m/s
        "wind_gust_ms": cur.get("wind_gusts_10m", 0.0) / 3.6,
        "cloud_cover_pct": cur.get("cloud_cover", 0.0),
        "precip_mm": cur.get("precipitation", 0.0),
        "pressure_hpa": cur.get("surface_pressure", 1013.25),
        "uv_index": cur.get("uv_index", 0.0),
    }


def parse_hourly_forecast(data: dict, hours: int = 24) -> list[dict]:
    """Parse the hourly forecast into a list of condition dicts."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    now = datetime.utcnow()

    results = []
    for i, time_str in enumerate(times):
        t = datetime.fromisoformat(time_str)
        if t < now:
            continue
        if len(results) >= hours:
            break
        results.append({
            "time": t.isoformat(),
            "temp_c": hourly.get("temperature_2m", [0.0])[i],
            "humidity_pct": hourly.get("relative_humidity_2m", [0.0])[i],
            "wind_speed_ms": hourly.get("wind_speed_10m", [0.0])[i] / 3.6,
            "wind_gust_ms": hourly.get("wind_gusts_10m", [0.0])[i] / 3.6,
            "cloud_cover_pct": hourly.get("cloud_cover", [0.0])[i],
            "solar_radiation_wm2": hourly.get("direct_radiation", [0.0])[i],
            "precip_mm": hourly.get("precipitation", [0.0])[i],
            "pressure_hpa": hourly.get("surface_pressure", [1013.25])[i],
            "dewpoint_c": hourly.get("dewpoint_2m", [0.0])[i],
            "uv_index": hourly.get("uv_index", [0.0])[i],
        })
    return results


def compute_pressure_delta(hourly_data: list[dict], current_idx: int = 0) -> float:
    """Compute 3-hour pressure change from hourly data."""
    if len(hourly_data) < 4:
        return 0.0
    current_p = hourly_data[current_idx].get("pressure_hpa", 1013.25)
    # Look 3 hours back if available
    past_idx = max(0, current_idx - 3)
    past_p = hourly_data[past_idx].get("pressure_hpa", current_p)
    return current_p - past_p
