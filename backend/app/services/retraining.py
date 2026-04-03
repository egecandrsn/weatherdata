"""Model retraining service — triggers personal model updates from accumulated feedback."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from app.services.features import FEATURE_NAMES, feature_vector_to_model_input
from app.ml.reward_model import (
    RewardNetwork,
    train_reward_model,
    save_reward_model,
    compute_ground_truth_reward,
    encode_tags,
)
from app.core.config import settings


def should_retrain(user: dict) -> bool:
    """Check if a user has enough new feedback since last training to trigger a retrain."""
    feedback_count = user.get("feedback_count", 0)
    model_ver = user.get("personal_model_ver", 0)
    feedbacks_since_last = feedback_count - (model_ver * settings.retrain_feedback_threshold)
    return feedbacks_since_last >= settings.retrain_feedback_threshold


def retrain_reward_model(
    user_id: str,
    feedback_records: list[dict],
    existing_model: Optional[RewardNetwork] = None,
) -> tuple[RewardNetwork, dict]:
    """Retrain the reward model for a specific user using their accumulated feedback.

    Returns the trained model and training metrics.
    """
    user_feedbacks = [f for f in feedback_records if f["user_id"] == user_id]

    if len(user_feedbacks) < settings.retrain_feedback_threshold:
        return existing_model or RewardNetwork(), {
            "status": "insufficient_data",
            "n_samples": len(user_feedbacks),
            "required": settings.retrain_feedback_threshold,
        }

    # Apply exponential decay weighting — recent feedback counts more
    weighted_feedbacks = _apply_decay_weights(user_feedbacks)

    model, metrics = train_reward_model(
        weighted_feedbacks,
        model=existing_model,
        epochs=30,
        lr=1e-3,
    )

    # Save user-specific reward model
    model_dir = os.path.join(settings.model_dir, "users", user_id)
    save_reward_model(model, os.path.join(model_dir, "reward_model.pt"))

    return model, metrics


def _apply_decay_weights(feedbacks: list[dict], decay_factor: float = 0.95) -> list[dict]:
    """Apply exponential decay so recent feedback matters more.

    Most recent feedback gets weight 1.0, each older one decays by decay_factor.
    We duplicate recent records proportionally to their weight.
    """
    n = len(feedbacks)
    if n <= 1:
        return feedbacks

    # Sort by timestamp (most recent last)
    sorted_fb = sorted(feedbacks, key=lambda f: f.get("timestamp", ""))

    weighted = []
    for i, fb in enumerate(sorted_fb):
        weight = decay_factor ** (n - 1 - i)
        # Include feedback if weight is significant
        if weight > 0.3:
            weighted.append(fb)

    return weighted if weighted else feedbacks


def compute_user_sensitivity_profile(feedbacks: list[dict]) -> dict:
    """Analyze user feedback to compute their sensitivity profile.

    Returns sensitivity scores for cold, heat, wind, humidity, and pressure.
    Each score ranges from -1 (insensitive) to +1 (very sensitive).
    """
    if len(feedbacks) < 5:
        return {
            "cold_sensitivity": 0.0,
            "heat_sensitivity": 0.0,
            "wind_sensitivity": 0.0,
            "humidity_sensitivity": 0.0,
            "pressure_sensitivity": 0.0,
        }

    cold_errors = []
    heat_errors = []
    wind_correlations = []
    humidity_correlations = []
    pressure_correlations = []

    for fb in feedbacks:
        fv = fb.get("feature_vector", {})
        predicted = fb.get("predicted_score", 0.0)
        actual = fb.get("comfort_score", 0.0)
        error = actual - predicted  # positive = user feels warmer than predicted

        temp = fv.get("temp_c", 20.0)
        wind = fv.get("wind_speed_ms", 0.0)
        humidity = fv.get("humidity_pct", 50.0)
        pressure_delta = fv.get("pressure_delta_3h", 0.0)

        # Cold sensitivity: when it's cold, does user feel colder than predicted?
        if temp < 10:
            cold_errors.append(error)

        # Heat sensitivity: when it's hot, does user feel warmer than predicted?
        if temp > 25:
            heat_errors.append(error)

        # Wind correlation: does wind amplify discomfort beyond prediction?
        if wind > 3:
            wind_correlations.append(abs(error) * (1 if error < 0 else -0.5))

        # Humidity: discomfort in humid conditions
        if humidity > 65:
            humidity_correlations.append(error)

        # Pressure: sensitivity to pressure changes
        if abs(pressure_delta) > 2:
            pressure_correlations.append(abs(error))

    def _mean_clamp(values: list[float]) -> float:
        if not values:
            return 0.0
        return max(-1.0, min(1.0, np.mean(values) * 2))

    return {
        "cold_sensitivity": _mean_clamp(cold_errors) * -1,  # negative error in cold = sensitive
        "heat_sensitivity": _mean_clamp(heat_errors),
        "wind_sensitivity": _mean_clamp(wind_correlations),
        "humidity_sensitivity": _mean_clamp(humidity_correlations),
        "pressure_sensitivity": _mean_clamp(pressure_correlations),
    }
