"""Tests for new API endpoints: model delivery, enhanced insights, transition alerts."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import _users, _predictions, _feedbacks


@pytest.fixture(autouse=True)
def clear_state():
    """Clear in-memory stores between tests."""
    _users.clear()
    _predictions.clear()
    _feedbacks.clear()
    yield
    _users.clear()
    _predictions.clear()
    _feedbacks.clear()


client = TestClient(app)


def _create_user(user_id: str = None) -> str:
    """Helper to onboard a test user."""
    if user_id is None:
        user_id = str(uuid.uuid4())
    resp = client.post("/v1/onboarding", json={
        "user_id": user_id,
        "quiz_answers": {"hot_cold_slider": 0, "barometric_sensitivity": "no", "climate_zone": "temperate"},
        "home_lat": 39.93,
        "home_lon": 32.86,
        "timezone": "Europe/Istanbul",
    })
    assert resp.status_code == 200
    return user_id


class TestModelDelivery:
    def test_model_not_found_for_unknown_user(self):
        resp = client.get(f"/v1/model/{uuid.uuid4()}/latest")
        assert resp.status_code == 404

    def test_base_model_for_new_user(self):
        user_id = _create_user()
        resp = client.get(f"/v1/model/{user_id}/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "base"
        assert data["model_version"] == 0
        assert data["user_id"] == user_id

    def test_model_version_increments_after_feedback(self):
        user_id = _create_user()

        # Simulate having enough feedback to trigger retrain
        # Create a fake prediction and submit 5 feedbacks
        pred_id = str(uuid.uuid4())
        _predictions[pred_id] = {
            "prediction_id": pred_id,
            "user_id": user_id,
            "feature_vector": {
                "temp_c": 20.0, "humidity_pct": 50.0, "wind_speed_ms": 2.0,
                "wind_gust_ms": 3.0, "cloud_cover_pct": 50.0,
                "solar_radiation_wm2": 200.0, "precip_mm": 0.0,
                "pressure_hpa": 1013.0, "pressure_delta_3h": 0.0,
                "dewpoint_c": 10.0, "uv_index": 3.0,
                "hour_of_day": 12, "heat_index": 0.0,
                "wind_chill": 0.0, "apparent_temp_delta": 0.0,
                "steps_last_30min": 0, "elevation_m": 0.0,
            },
            "comfort_score": 0.1,
        }

        for i in range(5):
            resp = client.post("/v1/feedback", json={
                "user_id": user_id,
                "prediction_id": pred_id,
                "comfort_score": 0.2,
            })
            assert resp.status_code == 200

        resp = client.get(f"/v1/model/{user_id}/latest")
        data = resp.json()
        assert data["model_version"] >= 1
        assert data["model_type"] == "personalized"


class TestEnhancedInsights:
    def test_sensitivity_profile_in_insights(self):
        user_id = _create_user()
        # Add some feedbacks
        _users[user_id]["feedback_count"] = 5
        for i in range(5):
            _feedbacks.append({
                "user_id": user_id,
                "feature_vector": {
                    "temp_c": 5.0, "wind_speed_ms": 5.0,
                    "humidity_pct": 50.0, "pressure_delta_3h": 0.0,
                },
                "predicted_score": -0.2,
                "comfort_score": -0.6,
                "tags": [],
            })

        resp = client.get(f"/v1/insights/{user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["personality_card"] is not None
        profile = data["personality_card"]["sensitivity_profile"]
        # Should detect cold sensitivity from the feedback pattern
        assert "cold_sensitivity" in profile

    def test_discoveries_include_sensitivity_insights(self):
        user_id = _create_user()
        _users[user_id]["feedback_count"] = 15

        # Add feedbacks showing cold sensitivity
        for i in range(15):
            _feedbacks.append({
                "user_id": user_id,
                "feature_vector": {
                    "temp_c": 3.0, "wind_speed_ms": 2.0,
                    "humidity_pct": 50.0, "pressure_delta_3h": 0.0,
                },
                "predicted_score": -0.2,
                "comfort_score": -0.8,
                "tags": ["chilled"],
            })

        resp = client.get(f"/v1/insights/{user_id}")
        data = resp.json()
        # Should have personalization discovery
        assert any("personalized" in d for d in data["discoveries"])


class TestPredictResponseIncludesAlerts:
    def test_predict_response_has_transition_alerts_field(self):
        """Verify the schema includes transition_alerts even if empty."""
        user_id = _create_user()
        # We can't easily mock weather API here, but we can verify
        # the TransitionAlertResponse schema is properly registered
        from app.models.schemas import PredictResponse
        fields = PredictResponse.model_fields
        assert "transition_alerts" in fields
