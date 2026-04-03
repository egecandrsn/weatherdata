"""Model inference — load and run the comfort model."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from app.core.config import settings
from app.services.features import FeatureVector, feature_vector_to_model_input


class ComfortPredictor:
    """Loads and runs the trained comfort model for inference."""

    def __init__(self):
        self._xgb_model = None
        self._onnx_session = None

    def load(self, model_path: Optional[str] = None):
        """Load the model — tries ONNX first, falls back to XGBoost JSON."""
        if model_path is None:
            model_path = settings.base_model_path

        onnx_path = model_path
        json_path = model_path.replace(".onnx", ".json")

        if os.path.exists(onnx_path) and onnx_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(onnx_path)
                return
            except Exception:
                pass

        if os.path.exists(json_path):
            from xgboost import XGBRegressor
            self._xgb_model = XGBRegressor()
            self._xgb_model.load_model(json_path)
            return

        raise FileNotFoundError(
            f"No model found at {onnx_path} or {json_path}. Train the base model first."
        )

    def predict(self, fv: FeatureVector) -> tuple[float, float]:
        """Predict comfort score and confidence from a feature vector.

        Returns:
            (comfort_score, confidence) where score is in [-1, 1]
            and confidence is in [0, 1].
        """
        raw = feature_vector_to_model_input(fv)
        x = np.array([raw], dtype=np.float32)

        if self._onnx_session is not None:
            input_name = self._onnx_session.get_inputs()[0].name
            result = self._onnx_session.run(None, {input_name: x})
            score = float(result[0][0])
        elif self._xgb_model is not None:
            score = float(self._xgb_model.predict(x)[0])
        else:
            raise RuntimeError("Model not loaded. Call .load() first.")

        score = max(-1.0, min(1.0, score))

        # Confidence: higher when score is near 0 (common range), lower at extremes
        # This is a simple heuristic; real confidence comes from ensemble variance
        confidence = max(0.3, 1.0 - abs(score) * 0.3)

        return score, confidence

    def predict_batch(self, features: list[FeatureVector]) -> list[tuple[float, float]]:
        """Predict comfort for multiple feature vectors."""
        return [self.predict(fv) for fv in features]


# Global singleton
_predictor: Optional[ComfortPredictor] = None


def get_predictor() -> ComfortPredictor:
    """Get or initialize the global comfort predictor."""
    global _predictor
    if _predictor is None:
        _predictor = ComfortPredictor()
        try:
            _predictor.load()
        except FileNotFoundError:
            pass  # Model not trained yet; endpoints will handle gracefully
    return _predictor
