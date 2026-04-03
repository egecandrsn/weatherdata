"""Tests for the retraining service."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.retraining import (
    should_retrain,
    compute_user_sensitivity_profile,
    _apply_decay_weights,
)


class TestShouldRetrain:
    def test_new_user_no_retrain(self):
        user = {"feedback_count": 0, "personal_model_ver": 0}
        assert should_retrain(user) is False

    def test_below_threshold_no_retrain(self):
        user = {"feedback_count": 3, "personal_model_ver": 0}
        assert should_retrain(user) is False

    def test_at_threshold_triggers_retrain(self):
        user = {"feedback_count": 5, "personal_model_ver": 0}
        assert should_retrain(user) is True

    def test_after_first_retrain_needs_more(self):
        user = {"feedback_count": 7, "personal_model_ver": 1}
        assert should_retrain(user) is False

    def test_second_retrain_threshold(self):
        user = {"feedback_count": 10, "personal_model_ver": 1}
        assert should_retrain(user) is True


class TestDecayWeights:
    def test_empty_list(self):
        assert _apply_decay_weights([]) == []

    def test_single_item(self):
        fb = [{"timestamp": "2024-01-01T00:00"}]
        assert len(_apply_decay_weights(fb)) == 1

    def test_recent_kept_old_dropped(self):
        feedbacks = [
            {"timestamp": f"2024-01-{i+1:02d}T00:00"} for i in range(50)
        ]
        weighted = _apply_decay_weights(feedbacks, decay_factor=0.95)
        # Old feedbacks with weight < 0.3 should be dropped
        assert len(weighted) < len(feedbacks)
        assert len(weighted) > 0


class TestSensitivityProfile:
    def test_insufficient_data_returns_zeros(self):
        profile = compute_user_sensitivity_profile([])
        assert profile["cold_sensitivity"] == 0.0
        assert profile["heat_sensitivity"] == 0.0

    def test_cold_sensitive_user(self):
        feedbacks = [
            {
                "feature_vector": {"temp_c": 5.0, "wind_speed_ms": 2.0, "humidity_pct": 50.0, "pressure_delta_3h": 0.0},
                "predicted_score": -0.3,
                "comfort_score": -0.7,  # user feels much colder than predicted
            }
            for _ in range(10)
        ]
        profile = compute_user_sensitivity_profile(feedbacks)
        assert profile["cold_sensitivity"] > 0  # should detect cold sensitivity

    def test_heat_sensitive_user(self):
        feedbacks = [
            {
                "feature_vector": {"temp_c": 32.0, "wind_speed_ms": 1.0, "humidity_pct": 50.0, "pressure_delta_3h": 0.0},
                "predicted_score": 0.3,
                "comfort_score": 0.7,  # user feels much warmer than predicted
            }
            for _ in range(10)
        ]
        profile = compute_user_sensitivity_profile(feedbacks)
        assert profile["heat_sensitivity"] > 0

    def test_pressure_sensitive_user(self):
        feedbacks = [
            {
                "feature_vector": {"temp_c": 20.0, "wind_speed_ms": 2.0, "humidity_pct": 50.0, "pressure_delta_3h": 5.0},
                "predicted_score": 0.0,
                "comfort_score": 0.3,
            }
            for _ in range(10)
        ]
        profile = compute_user_sensitivity_profile(feedbacks)
        assert profile["pressure_sensitivity"] > 0

    def test_values_clamped_to_range(self):
        feedbacks = [
            {
                "feature_vector": {"temp_c": 5.0, "wind_speed_ms": 10.0, "humidity_pct": 80.0, "pressure_delta_3h": 10.0},
                "predicted_score": -0.5,
                "comfort_score": 0.5,
            }
            for _ in range(20)
        ]
        profile = compute_user_sensitivity_profile(feedbacks)
        for key, value in profile.items():
            assert -1.0 <= value <= 1.0, f"{key} out of range: {value}"
