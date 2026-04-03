"""SQLAlchemy ORM models matching the data model spec."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    JSON,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class UserProfile(Base):
    __tablename__ = "user_profile"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    onboarding_answers = Column(JSON, nullable=True)
    cluster_id = Column(Integer, nullable=True)
    personal_model_ver = Column(Integer, default=0)
    feedback_count = Column(Integer, default=0)
    home_lat = Column(Float, nullable=True)
    home_lon = Column(Float, nullable=True)
    timezone = Column(String, nullable=True)
    preferences = Column(JSON, nullable=True)


class Prediction(Base):
    __tablename__ = "prediction"

    prediction_id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user_profile.user_id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    feature_vector = Column(JSON, nullable=False)
    comfort_score = Column(Float, nullable=False)
    comfort_label = Column(String, nullable=False)
    clothing_rec = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user_profile.user_id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    feature_vector = Column(JSON, nullable=False)
    comfort_score = Column(Float, nullable=False)
    tags = Column(ARRAY(String), nullable=True)
    predicted_score = Column(Float, nullable=False)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("prediction.prediction_id"), nullable=False)
    clothing_tag = Column(String, nullable=True)
    latency_ms = Column(Integer, nullable=True)


class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    temp_c = Column(Float)
    humidity_pct = Column(Float)
    wind_speed_ms = Column(Float)
    wind_gust_ms = Column(Float)
    cloud_cover_pct = Column(Float)
    solar_radiation_wm2 = Column(Float)
    precip_mm = Column(Float)
    pressure_hpa = Column(Float)
    dewpoint_c = Column(Float)
    uv_index = Column(Float)
    conditions = Column(String, nullable=True)
    raw_data = Column(JSON, nullable=True)
