"""Feature engineering: compute derived features for the comfort model."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from app.models.schemas import FeatureVector, DeviceSensors


def compute_heat_index(temp_c: float, rh: float) -> Optional[float]:
    """Rothfusz regression for heat index. Only valid when T >= 27 C and RH >= 40%."""
    if temp_c < 27 or rh < 40:
        return None
    t_f = temp_c * 9 / 5 + 32
    hi = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 6.83783e-3 * t_f**2
        - 5.481717e-2 * rh**2
        + 1.22874e-3 * t_f**2 * rh
        + 8.5282e-4 * t_f * rh**2
        - 1.99e-6 * t_f**2 * rh**2
    )
    return (hi - 32) * 5 / 9


def compute_wind_chill(temp_c: float, wind_speed_kmh: float) -> Optional[float]:
    """North American wind chill index. Valid when T <= 10 C and wind >= 4.8 km/h."""
    if temp_c > 10 or wind_speed_kmh < 4.8:
        return None
    wc = (
        13.12
        + 0.6215 * temp_c
        - 11.37 * wind_speed_kmh**0.16
        + 0.3965 * temp_c * wind_speed_kmh**0.16
    )
    return wc


def compute_standard_apparent_temp(temp_c: float, humidity_pct: float, wind_speed_ms: float) -> float:
    """Simple apparent temperature combining wind chill and heat index."""
    wind_kmh = wind_speed_ms * 3.6
    wc = compute_wind_chill(temp_c, wind_kmh)
    if wc is not None:
        return wc
    hi = compute_heat_index(temp_c, humidity_pct)
    if hi is not None:
        return hi
    return temp_c


def build_feature_vector(
    temp_c: float,
    humidity_pct: float = 0.0,
    wind_speed_ms: float = 0.0,
    wind_gust_ms: float = 0.0,
    cloud_cover_pct: float = 0.0,
    solar_radiation_wm2: float = 0.0,
    precip_mm: float = 0.0,
    pressure_hpa: float = 1013.25,
    pressure_delta_3h: float = 0.0,
    dewpoint_c: float = 0.0,
    uv_index: float = 0.0,
    hour_of_day: Optional[int] = None,
    device_sensors: Optional[DeviceSensors] = None,
) -> FeatureVector:
    """Assemble a full feature vector with derived features computed."""
    if hour_of_day is None:
        hour_of_day = datetime.utcnow().hour

    apparent = compute_standard_apparent_temp(temp_c, humidity_pct, wind_speed_ms)
    hi = compute_heat_index(temp_c, humidity_pct)
    wc = compute_wind_chill(temp_c, wind_speed_ms * 3.6)

    fv = FeatureVector(
        temp_c=temp_c,
        humidity_pct=humidity_pct,
        wind_speed_ms=wind_speed_ms,
        wind_gust_ms=wind_gust_ms,
        cloud_cover_pct=cloud_cover_pct,
        solar_radiation_wm2=solar_radiation_wm2,
        precip_mm=precip_mm,
        pressure_hpa=pressure_hpa,
        pressure_delta_3h=pressure_delta_3h,
        dewpoint_c=dewpoint_c,
        uv_index=uv_index,
        hour_of_day=hour_of_day,
        heat_index=hi,
        wind_chill=wc,
        apparent_temp_delta=apparent - temp_c,
    )

    if device_sensors:
        fv.local_pressure_hpa = device_sensors.local_pressure_hpa
        fv.elevation_m = device_sensors.elevation_m
        fv.ambient_light = device_sensors.ambient_light
        fv.steps_last_30min = device_sensors.steps_last_30min
        fv.indoor_outdoor = device_sensors.indoor_outdoor

    return fv


def feature_vector_to_model_input(fv: FeatureVector) -> list[float]:
    """Convert a FeatureVector to a flat list of floats for model inference.

    The order must match the training feature order exactly.
    """
    return [
        fv.temp_c,
        fv.humidity_pct,
        fv.wind_speed_ms,
        fv.wind_gust_ms,
        fv.cloud_cover_pct,
        fv.solar_radiation_wm2,
        fv.precip_mm,
        fv.pressure_hpa,
        fv.pressure_delta_3h,
        fv.dewpoint_c,
        fv.uv_index,
        fv.hour_of_day,
        fv.heat_index or 0.0,
        fv.wind_chill or 0.0,
        fv.apparent_temp_delta or 0.0,
        fv.steps_last_30min or 0,
        fv.elevation_m or 0.0,
    ]


# The ordered feature names (must stay in sync with feature_vector_to_model_input)
FEATURE_NAMES = [
    "temp_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_gust_ms",
    "cloud_cover_pct",
    "solar_radiation_wm2",
    "precip_mm",
    "pressure_hpa",
    "pressure_delta_3h",
    "dewpoint_c",
    "uv_index",
    "hour_of_day",
    "heat_index",
    "wind_chill",
    "apparent_temp_delta",
    "steps_last_30min",
    "elevation_m",
]
