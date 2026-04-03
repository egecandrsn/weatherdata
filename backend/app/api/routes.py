"""Core API endpoints for FeelsLike AI."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    PredictRequest,
    PredictResponse,
    FeedbackRequest,
    FeedbackResponse,
    OnboardingRequest,
    OnboardingResponse,
    UserProfileResponse,
    InsightsResponse,
    HourlyForecast,
    PersonalityCard,
    SensitivityProfile,
    TransitionAlertResponse,
    ModelDeliveryResponse,
)
from app.services.weather import fetch_current_weather, parse_current_conditions, parse_hourly_forecast
from app.services.features import build_feature_vector, feature_vector_to_model_input
from app.services.comfort import score_to_comfort_label, score_to_clothing_rec, generate_description
from app.ml.inference import get_predictor
from app.core.config import settings
from app.ml.clustering import (
    assign_cluster,
    apply_cluster_adjustment,
    CLUSTER_PERSONALITIES,
)
from app.services.alerts import detect_transitions
from app.services.retraining import (
    should_retrain,
    retrain_reward_model,
    compute_user_sensitivity_profile,
)

router = APIRouter(prefix="/v1")

# ---------------------------------------------------------------------------
# In-memory stores (replace with DB in production)
# ---------------------------------------------------------------------------
_users: dict[str, dict] = {}
_predictions: dict[str, dict] = {}
_feedbacks: list[dict] = []


# ---------------------------------------------------------------------------
# POST /v1/predict
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Generate a personalized comfort prediction for the user."""
    user_id = str(req.user_id)
    user = _users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Complete onboarding first.")

    # Fetch weather
    try:
        weather_data = await fetch_current_weather(req.lat, req.lon)
        current = parse_current_conditions(weather_data)
        hourly = parse_hourly_forecast(weather_data, hours=24)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather API error: {e}")

    # Build feature vector
    fv = build_feature_vector(
        **current,
        device_sensors=req.device_sensors,
    )

    # Run base model inference
    predictor = get_predictor()
    try:
        base_score, confidence = predictor.predict(fv)
    except (RuntimeError, FileNotFoundError):
        # Fallback: use simple heuristic if model not trained yet
        base_score = (fv.temp_c - 20.0) / 22.0
        base_score = max(-1.0, min(1.0, base_score))
        confidence = 0.3

    # Apply cluster personalization
    cluster_id = user.get("cluster_id", 3)
    score = apply_cluster_adjustment(
        base_score,
        cluster_id,
        fv.temp_c,
        fv.wind_speed_ms,
        fv.pressure_delta_3h,
        fv.humidity_pct,
    )

    # Adjust confidence based on user's feedback history
    feedback_count = user.get("feedback_count", 0)
    if feedback_count < 5:
        confidence *= 0.6  # low confidence for new users
    elif feedback_count < 20:
        confidence *= 0.8

    label = score_to_comfort_label(score)
    clothing = score_to_clothing_rec(score)
    description = generate_description(label, clothing, fv.temp_c)

    prediction_id = uuid.uuid4()

    # Store prediction for feedback linkage
    _predictions[str(prediction_id)] = {
        "prediction_id": prediction_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "feature_vector": fv.model_dump(),
        "comfort_score": score,
        "comfort_label": label.value,
        "clothing_rec": clothing.value,
        "confidence": confidence,
    }

    # Build hourly forecast
    hourly_forecasts = []
    for h in hourly[:12]:  # next 12 hours
        h_fv = build_feature_vector(**{k: v for k, v in h.items() if k != "time"})
        try:
            h_score, _ = predictor.predict(h_fv)
        except (RuntimeError, FileNotFoundError):
            h_score = (h_fv.temp_c - 20.0) / 22.0
            h_score = max(-1.0, min(1.0, h_score))
        h_score = apply_cluster_adjustment(h_score, cluster_id, h_fv.temp_c)
        hourly_forecasts.append(
            HourlyForecast(
                time=datetime.fromisoformat(h["time"]),
                comfort_score=h_score,
                comfort_label=score_to_comfort_label(h_score),
                clothing_rec=score_to_clothing_rec(h_score),
                temp_c=h_fv.temp_c,
            )
        )

    # Detect transition alerts from hourly forecasts
    forecast_dicts = [
        {"time": hf.time.isoformat(), "comfort_score": hf.comfort_score}
        for hf in hourly_forecasts
    ]
    raw_alerts = detect_transitions(forecast_dicts)
    transition_alerts = [
        TransitionAlertResponse(
            time=a.time,
            message=a.message,
            from_label=a.from_label,
            to_label=a.to_label,
            clothing_change=a.clothing_change,
        )
        for a in raw_alerts
    ]

    return PredictResponse(
        prediction_id=prediction_id,
        comfort_score=round(score, 3),
        comfort_label=label,
        clothing_rec=clothing,
        confidence=round(confidence, 3),
        description=description,
        hourly_forecast=hourly_forecasts,
        transition_alerts=transition_alerts,
    )


# ---------------------------------------------------------------------------
# POST /v1/feedback
# ---------------------------------------------------------------------------
@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """Submit user feedback on a prediction."""
    user_id = str(req.user_id)
    user = _users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    prediction = _predictions.get(str(req.prediction_id))
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    feedback_entry = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "feature_vector": prediction["feature_vector"],
        "comfort_score": req.comfort_score,
        "tags": req.tags,
        "predicted_score": prediction["comfort_score"],
        "prediction_id": str(req.prediction_id),
        "clothing_tag": req.clothing_tag,
        "latency_ms": req.latency_ms,
    }
    _feedbacks.append(feedback_entry)

    user["feedback_count"] = user.get("feedback_count", 0) + 1
    total = user["feedback_count"]

    # Determine next model update
    remaining = settings.retrain_feedback_threshold - (total % settings.retrain_feedback_threshold)
    if remaining == settings.retrain_feedback_threshold:
        remaining = 0

    # Trigger retraining if threshold met
    if should_retrain(user):
        try:
            _, metrics = retrain_reward_model(user_id, _feedbacks)
            user["personal_model_ver"] = user.get("personal_model_ver", 0) + 1
            next_update = f"Model updated to v{user['personal_model_ver']}"
        except Exception:
            next_update = f"After {remaining} more feedback(s)" if remaining > 0 else "Queued for next training cycle"
    else:
        next_update = f"After {remaining} more feedback(s)" if remaining > 0 else "Queued for next training cycle"

    return FeedbackResponse(
        received=True,
        total_feedback=total,
        next_model_update=next_update,
    )


# ---------------------------------------------------------------------------
# POST /v1/onboarding
# ---------------------------------------------------------------------------
@router.post("/onboarding", response_model=OnboardingResponse)
async def onboarding(req: OnboardingRequest):
    """Register a new user and assign them to an archetype cluster."""
    user_id = str(req.user_id)
    cluster_id = assign_cluster(req.quiz_answers)

    _users[user_id] = {
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "onboarding_answers": req.quiz_answers,
        "cluster_id": cluster_id,
        "personal_model_ver": 0,
        "feedback_count": 0,
        "home_lat": req.home_lat,
        "home_lon": req.home_lon,
        "timezone": req.timezone,
    }

    return OnboardingResponse(
        assigned_cluster=cluster_id,
        initial_model_version=0,
    )


# ---------------------------------------------------------------------------
# GET /v1/profile/{user_id}
# ---------------------------------------------------------------------------
@router.get("/profile/{user_id}", response_model=UserProfileResponse)
async def get_profile(user_id: str):
    """Get a user's profile."""
    user = _users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    return UserProfileResponse(
        user_id=uuid.UUID(user_id),
        cluster_id=user.get("cluster_id"),
        model_version=user.get("personal_model_ver", 0),
        feedback_count=user.get("feedback_count", 0),
        created_at=datetime.fromisoformat(user["created_at"]),
    )


# ---------------------------------------------------------------------------
# GET /v1/insights/{user_id}
# ---------------------------------------------------------------------------
@router.get("/insights/{user_id}", response_model=InsightsResponse)
async def get_insights(user_id: str):
    """Get personalized insights and weather personality for a user."""
    user = _users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    cluster_id = user.get("cluster_id", 3)
    personality = CLUSTER_PERSONALITIES.get(cluster_id, {})
    feedback_count = user.get("feedback_count", 0)

    # Compute real sensitivity profile from feedback
    user_feedbacks = [f for f in _feedbacks if f["user_id"] == user_id]
    sensitivity = compute_user_sensitivity_profile(user_feedbacks)

    # Build personality card with real sensitivity data
    card = PersonalityCard(
        title=personality.get("title", "Weather Explorer"),
        description=personality.get("desc", "Still learning your weather personality..."),
        sensitivity_profile=SensitivityProfile(**sensitivity),
    )

    # Build accuracy trend from feedback history
    accuracy_trend = []
    for fb in user_feedbacks[-20:]:  # last 20
        error = abs(fb["predicted_score"] - fb["comfort_score"])
        accuracy_trend.append(round(1.0 - error, 3))

    # Generate discoveries based on sensitivity analysis
    discoveries = []
    if feedback_count >= 10:
        discoveries.append(
            f"Based on {feedback_count} data points, your model is becoming personalized."
        )
        # Add sensitivity-based discoveries
        if sensitivity["cold_sensitivity"] > 0.3:
            discoveries.append(
                "You consistently feel colder than average — you're cold-sensitive."
            )
        if sensitivity["pressure_sensitivity"] > 0.3:
            discoveries.append(
                "You seem sensitive to barometric pressure changes — "
                "you might notice weather shifts before they happen."
            )
        if sensitivity["wind_sensitivity"] > 0.3:
            discoveries.append(
                "Wind affects your comfort more than most people."
            )
    if feedback_count >= 20:
        avg_error = sum(
            abs(f["predicted_score"] - f["comfort_score"]) for f in user_feedbacks
        ) / len(user_feedbacks)
        discoveries.append(
            f"Your average prediction accuracy: {(1.0 - avg_error) * 100:.0f}%"
        )

    return InsightsResponse(
        personality_card=card if feedback_count >= 3 else None,
        accuracy_trend=accuracy_trend,
        discoveries=discoveries,
    )


# ---------------------------------------------------------------------------
# GET /v1/model/{user_id}/latest
# ---------------------------------------------------------------------------
@router.get("/model/{user_id}/latest", response_model=ModelDeliveryResponse)
async def get_latest_model(user_id: str):
    """Get metadata about the latest model for a user.

    In production, this would return a binary ONNX model file.
    For now, returns model metadata for the client to determine
    if it needs to update its on-device model.
    """
    user = _users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    model_ver = user.get("personal_model_ver", 0)
    model_type = "personalized" if model_ver > 0 else "base"

    return ModelDeliveryResponse(
        user_id=user_id,
        model_version=model_ver,
        model_type=model_type,
        cluster_id=user.get("cluster_id"),
        feedback_count=user.get("feedback_count", 0),
    )
