"""Tests for transition alert detection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.alerts import detect_transitions, TransitionAlert


class TestDetectTransitions:
    def test_no_forecasts_returns_empty(self):
        assert detect_transitions([]) == []

    def test_single_forecast_returns_empty(self):
        assert detect_transitions([{"time": "2024-01-01T12:00", "comfort_score": 0.0}]) == []

    def test_stable_forecast_returns_empty(self):
        forecasts = [
            {"time": f"2024-01-01T{h:02d}:00", "comfort_score": 0.0}
            for h in range(12, 18)
        ]
        assert detect_transitions(forecasts) == []

    def test_detects_cold_transition(self):
        forecasts = [
            {"time": "2024-01-01T12:00", "comfort_score": 0.1},
            {"time": "2024-01-01T13:00", "comfort_score": -0.4},
        ]
        alerts = detect_transitions(forecasts, threshold=0.25)
        assert len(alerts) == 1
        assert alerts[0].to_label in ("chilly", "cool")
        assert "layer" in alerts[0].message.lower() or "dropping" in alerts[0].message.lower()

    def test_detects_warm_transition(self):
        forecasts = [
            {"time": "2024-01-01T08:00", "comfort_score": -0.3},
            {"time": "2024-01-01T12:00", "comfort_score": 0.3},
        ]
        alerts = detect_transitions(forecasts, threshold=0.25)
        assert len(alerts) == 1
        assert "warming" in alerts[0].message.lower() or "shed" in alerts[0].message.lower()

    def test_multiple_transitions(self):
        forecasts = [
            {"time": "2024-01-01T08:00", "comfort_score": -0.3},
            {"time": "2024-01-01T12:00", "comfort_score": 0.3},
            {"time": "2024-01-01T18:00", "comfort_score": -0.4},
        ]
        alerts = detect_transitions(forecasts, threshold=0.25)
        assert len(alerts) == 2

    def test_clothing_change_included(self):
        forecasts = [
            {"time": "2024-01-01T12:00", "comfort_score": 0.1},
            {"time": "2024-01-01T18:00", "comfort_score": -0.5},
        ]
        alerts = detect_transitions(forecasts, threshold=0.25)
        assert len(alerts) == 1
        assert alerts[0].clothing_change is not None

    def test_small_change_below_threshold_ignored(self):
        forecasts = [
            {"time": "2024-01-01T12:00", "comfort_score": 0.0},
            {"time": "2024-01-01T13:00", "comfort_score": 0.1},
        ]
        alerts = detect_transitions(forecasts, threshold=0.25)
        assert len(alerts) == 0
