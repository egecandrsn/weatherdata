"""Tests for the FastAPI endpoints (using in-memory stores, no DB needed)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def user_id():
    return str(uuid.uuid4())


class TestHealthCheck:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestOnboarding:
    def test_onboarding_success(self, client, user_id):
        resp = client.post("/v1/onboarding", json={
            "user_id": user_id,
            "quiz_answers": {
                "hot_cold_slider": -0.5,
                "barometric_sensitivity": "sometimes",
                "climate_zone": "continental",
            },
            "home_lat": 39.93,
            "home_lon": 32.86,
            "timezone": "Europe/Istanbul",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "assigned_cluster" in data
        assert 0 <= data["assigned_cluster"] < 8


class TestProfile:
    def test_profile_not_found(self, client):
        resp = client.get(f"/v1/profile/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_profile_after_onboarding(self, client, user_id):
        client.post("/v1/onboarding", json={
            "user_id": user_id,
            "quiz_answers": {"hot_cold_slider": 0.0, "barometric_sensitivity": "no", "climate_zone": "temperate"},
            "home_lat": 40.0, "home_lon": 30.0, "timezone": "UTC",
        })
        resp = client.get(f"/v1/profile/{user_id}")
        assert resp.status_code == 200
        assert resp.json()["feedback_count"] == 0


class TestPredict:
    def test_predict_requires_onboarding(self, client, user_id):
        resp = client.post("/v1/predict", json={
            "user_id": user_id,
            "lat": 39.93,
            "lon": 32.86,
        })
        assert resp.status_code == 404


class TestFeedback:
    def test_feedback_requires_user(self, client, user_id):
        resp = client.post("/v1/feedback", json={
            "user_id": user_id,
            "prediction_id": str(uuid.uuid4()),
            "comfort_score": 0.3,
        })
        assert resp.status_code == 404


class TestInsights:
    def test_insights_not_found(self, client):
        resp = client.get(f"/v1/insights/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_insights_after_onboarding(self, client, user_id):
        client.post("/v1/onboarding", json={
            "user_id": user_id,
            "quiz_answers": {"hot_cold_slider": 0.0, "barometric_sensitivity": "no", "climate_zone": "temperate"},
            "home_lat": 40.0, "home_lon": 30.0, "timezone": "UTC",
        })
        resp = client.get(f"/v1/insights/{user_id}")
        assert resp.status_code == 200
        data = resp.json()
        # Not enough feedback for personality card
        assert data["personality_card"] is None
