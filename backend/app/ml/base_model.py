"""Base comfort model — trains an XGBoost model on historical weather data.

Since we don't have labeled thermal comfort data (ASHRAE), we synthesize
comfort labels from the historical CSV using biophysical formulas (UTCI-
inspired), then train a gradient-boosted tree to learn the mapping.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from app.services.features import (
    compute_heat_index,
    compute_wind_chill,
    compute_standard_apparent_temp,
    FEATURE_NAMES,
)
from app.core.config import settings


# ---------------------------------------------------------------------------
# Synthetic comfort label generation (UTCI-inspired)
# ---------------------------------------------------------------------------

def _synthesize_comfort_score(
    temp_c: float,
    humidity_pct: float,
    wind_speed_ms: float,
    wind_gust_ms: float = 0.0,
    cloud_cover_pct: float = 50.0,
    solar_radiation: float = 0.0,
    precip_mm: float = 0.0,
    pressure_hpa: float = 1013.25,
) -> float:
    """Generate a synthetic comfort score in [-1, 1] using biophysical heuristics.

    Roughly maps:
      < -20 C apparent  → -1.0 (freezing)
        20 C apparent   →  0.0 (comfortable)
      > 42 C apparent   → +1.0 (sweltering)
    With modifiers for humidity, precipitation, cloud cover, and wind gusts.
    """
    apparent = compute_standard_apparent_temp(temp_c, humidity_pct, wind_speed_ms)

    # Base linear mapping: 20 C = 0, scale ~30 C range to [-1, 1]
    score = (apparent - 20.0) / 22.0

    # Humidity penalty when hot (muggy discomfort)
    if apparent > 25 and humidity_pct > 60:
        score += (humidity_pct - 60) / 200.0

    # Precipitation makes it feel worse
    if precip_mm > 0:
        score -= 0.05 * min(precip_mm, 5.0) / 5.0  # slight cold shift in rain

    # Wind gusts feel colder
    if wind_gust_ms > wind_speed_ms + 3:
        gust_excess = wind_gust_ms - wind_speed_ms
        score -= 0.02 * min(gust_excess, 10)

    # Sunny clear skies warm you up
    if cloud_cover_pct < 20 and solar_radiation > 400:
        score += 0.05

    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Data loading and feature extraction from history CSV
# ---------------------------------------------------------------------------

def load_history_csv(csv_path: str) -> pd.DataFrame:
    """Load the historical weather CSV and extract features."""
    df = pd.read_csv(csv_path)

    # Rename columns to our schema
    col_map = {
        "Temperature": "temp_c",
        "Relative Humidity": "humidity_pct",
        "Wind Speed": "wind_speed_raw",
        "Wind Gust": "wind_gust_raw",
        "Cloud Cover": "cloud_cover_pct",
        "Precipitation": "precip_mm",
        "Heat Index": "heat_index_raw",
        "Wind Chill": "wind_chill_raw",
        "Visibility": "visibility",
        "Wind Direction": "wind_dir",
        "Date time": "datetime",
        "Conditions": "conditions",
    }
    df = df.rename(columns=col_map)

    # Parse datetime and extract hour
    df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df["hour_of_day"] = df["datetime"].dt.hour

    # Convert wind from km/h to m/s
    df["wind_speed_ms"] = pd.to_numeric(df.get("wind_speed_raw", pd.Series(dtype=float)), errors="coerce").fillna(0) / 3.6
    df["wind_gust_ms"] = pd.to_numeric(df.get("wind_gust_raw", pd.Series(dtype=float)), errors="coerce").fillna(0) / 3.6

    # Numeric conversions
    for col in ["temp_c", "humidity_pct", "cloud_cover_pct", "precip_mm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["temp_c"])
    df["humidity_pct"] = df["humidity_pct"].fillna(50)
    df["cloud_cover_pct"] = df["cloud_cover_pct"].fillna(50)
    df["precip_mm"] = df["precip_mm"].fillna(0)

    # Derived features
    df["solar_radiation_wm2"] = 0.0  # Not in CSV, default
    df["pressure_hpa"] = 1013.25  # Not in CSV, default
    df["pressure_delta_3h"] = 0.0
    df["dewpoint_c"] = df["temp_c"] - ((100 - df["humidity_pct"]) / 5.0)  # approx
    df["uv_index"] = 0.0
    df["steps_last_30min"] = 0
    df["elevation_m"] = 0.0

    # Compute derived features
    df["heat_index"] = df.apply(
        lambda r: compute_heat_index(r["temp_c"], r["humidity_pct"]) or 0.0, axis=1
    )
    df["wind_chill"] = df.apply(
        lambda r: compute_wind_chill(r["temp_c"], r["wind_speed_ms"] * 3.6) or 0.0, axis=1
    )
    df["apparent_temp_delta"] = df.apply(
        lambda r: compute_standard_apparent_temp(r["temp_c"], r["humidity_pct"], r["wind_speed_ms"]) - r["temp_c"],
        axis=1,
    )

    # Synthesize comfort labels
    df["comfort_score"] = df.apply(
        lambda r: _synthesize_comfort_score(
            r["temp_c"],
            r["humidity_pct"],
            r["wind_speed_ms"],
            r["wind_gust_ms"],
            r["cloud_cover_pct"],
            r.get("solar_radiation_wm2", 0),
            r["precip_mm"],
            r.get("pressure_hpa", 1013.25),
        ),
        axis=1,
    )

    return df


def prepare_training_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from the DataFrame."""
    feature_cols = FEATURE_NAMES
    X = df[feature_cols].values.astype(np.float32)
    y = df["comfort_score"].values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_base_model(
    csv_path: str,
    output_path: Optional[str] = None,
) -> tuple[XGBRegressor, dict]:
    """Train the base comfort model on historical CSV data.

    Returns the trained model and evaluation metrics.
    """
    if output_path is None:
        output_path = settings.base_model_path

    df = load_history_csv(csv_path)
    X, y = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path.replace(".onnx", ".json"))

    # Export to ONNX for on-device inference
    _export_to_onnx(model, output_path, n_features=X.shape[1])

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
    }
    print(f"Base model trained — MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return model, metrics


def _export_to_onnx(model: XGBRegressor, output_path: str, n_features: int):
    """Export XGBoost model to ONNX format for on-device inference."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model exported to {output_path}")
    except Exception as e:
        print(f"ONNX export skipped ({e}). JSON model is available.")
