"""Tests for model inference."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.ml.inference import ComfortPredictor
from app.services.features import build_feature_vector


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "base_comfort_model.onnx")


@pytest.fixture
def predictor():
    p = ComfortPredictor()
    try:
        p.load(MODEL_PATH)
    except FileNotFoundError:
        pytest.skip("Base model not trained yet")
    return p


class TestComfortPredictor:
    def test_predict_returns_score_and_confidence(self, predictor):
        fv = build_feature_vector(temp_c=20.0, humidity_pct=50.0, wind_speed_ms=2.0)
        score, confidence = predictor.predict(fv)
        assert -1.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_hot_weather_positive_score(self, predictor):
        fv = build_feature_vector(temp_c=38.0, humidity_pct=70.0, wind_speed_ms=1.0)
        score, _ = predictor.predict(fv)
        assert score > 0.3  # should feel hot

    def test_cold_weather_negative_score(self, predictor):
        fv = build_feature_vector(temp_c=-5.0, humidity_pct=60.0, wind_speed_ms=8.0)
        score, _ = predictor.predict(fv)
        assert score < -0.3  # should feel cold

    def test_comfortable_weather_near_zero(self, predictor):
        fv = build_feature_vector(temp_c=21.0, humidity_pct=45.0, wind_speed_ms=2.0)
        score, _ = predictor.predict(fv)
        assert -0.3 < score < 0.3  # should feel comfortable

    def test_batch_predict(self, predictor):
        features = [
            build_feature_vector(temp_c=t) for t in [0.0, 10.0, 20.0, 30.0, 40.0]
        ]
        results = predictor.predict_batch(features)
        assert len(results) == 5
        # Scores should increase with temperature
        scores = [r[0] for r in results]
        assert scores[-1] > scores[0]
