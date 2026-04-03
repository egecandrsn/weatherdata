"""Pydantic request/response schemas for the API."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ComfortLabel(str, Enum):
    freezing = "freezing"
    cold = "cold"
    chilly = "chilly"
    cool = "cool"
    comfortable = "comfortable"
    warm = "warm"
    hot = "hot"
    sweltering = "sweltering"


class ClothingRec(str, Enum):
    heavy_coat = "heavy_coat"
    jacket = "jacket"
    hoodie = "hoodie"
    light_layer = "light_layer"
    tshirt = "tshirt"
    tank_top = "tank_top"


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

class DeviceSensors(BaseModel):
    local_pressure_hpa: Optional[float] = None
    elevation_m: Optional[float] = None
    ambient_light: Optional[float] = None
    steps_last_30min: Optional[int] = None
    indoor_outdoor: Optional[str] = None  # "indoor" | "outdoor" | "unknown"


class FeatureVector(BaseModel):
    # Weather
    temp_c: float
    humidity_pct: float = 0.0
    wind_speed_ms: float = 0.0
    wind_gust_ms: float = 0.0
    cloud_cover_pct: float = 0.0
    solar_radiation_wm2: float = 0.0
    precip_mm: float = 0.0
    pressure_hpa: float = 1013.25
    pressure_delta_3h: float = 0.0
    dewpoint_c: float = 0.0
    uv_index: float = 0.0

    # Device sensors
    local_pressure_hpa: Optional[float] = None
    elevation_m: Optional[float] = None
    ambient_light: Optional[float] = None

    # User context
    steps_last_30min: Optional[int] = None
    hour_of_day: int = 12
    minutes_since_wake: Optional[int] = None
    days_at_location: Optional[int] = None
    indoor_outdoor: Optional[str] = None

    # Derived (computed server-side)
    heat_index: Optional[float] = None
    wind_chill: Optional[float] = None
    apparent_temp_delta: Optional[float] = None


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    user_id: uuid.UUID
    lat: float
    lon: float
    device_sensors: Optional[DeviceSensors] = None


class FeedbackRequest(BaseModel):
    user_id: uuid.UUID
    prediction_id: uuid.UUID
    comfort_score: float = Field(..., ge=-1.0, le=1.0)
    tags: Optional[list[str]] = None
    clothing_tag: Optional[str] = None
    latency_ms: Optional[int] = None


class OnboardingRequest(BaseModel):
    user_id: uuid.UUID
    quiz_answers: dict  # {hot_cold_slider, barometric_sensitivity, climate_zone}
    home_lat: float
    home_lon: float
    timezone: str


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class HourlyForecast(BaseModel):
    time: datetime
    comfort_score: float
    comfort_label: ComfortLabel
    clothing_rec: ClothingRec
    temp_c: float


class PredictResponse(BaseModel):
    prediction_id: uuid.UUID
    comfort_score: float
    comfort_label: ComfortLabel
    clothing_rec: ClothingRec
    confidence: float
    description: str
    hourly_forecast: list[HourlyForecast] = []
    transition_alerts: list["TransitionAlertResponse"] = []


class FeedbackResponse(BaseModel):
    received: bool = True
    total_feedback: int
    next_model_update: Optional[str] = None


class OnboardingResponse(BaseModel):
    assigned_cluster: int
    initial_model_version: int


class UserProfileResponse(BaseModel):
    user_id: uuid.UUID
    cluster_id: Optional[int]
    model_version: int
    feedback_count: int
    created_at: datetime


class SensitivityProfile(BaseModel):
    cold_sensitivity: float = 0.0  # -1 to 1
    heat_sensitivity: float = 0.0
    wind_sensitivity: float = 0.0
    humidity_sensitivity: float = 0.0
    pressure_sensitivity: float = 0.0


class PersonalityCard(BaseModel):
    title: str
    description: str
    sensitivity_profile: SensitivityProfile


class TransitionAlertResponse(BaseModel):
    time: str
    message: str
    from_label: str
    to_label: str
    clothing_change: Optional[str] = None


class InsightsResponse(BaseModel):
    personality_card: Optional[PersonalityCard] = None
    accuracy_trend: list[float] = []
    discoveries: list[str] = []


class ModelDeliveryResponse(BaseModel):
    user_id: str
    model_version: int
    model_type: str  # "base" | "personalized"
    cluster_id: Optional[int] = None
    feedback_count: int = 0
