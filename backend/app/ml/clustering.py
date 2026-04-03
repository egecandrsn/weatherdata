"""User clustering for cold-start personalization.

Assigns new users to archetype clusters based on onboarding quiz answers
so they get reasonable predictions before accumulating personal feedback.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


# Pre-defined archetype clusters based on quiz dimensions:
#   [hot_cold_slider (-1=cold-natured, 1=hot-natured),
#    barometric_sensitivity (0=none, 0.5=sometimes, 1=yes),
#    climate_zone_encoded (0=tropical, 0.25=arid, 0.5=temperate, 0.75=continental, 1=polar)]

N_CLUSTERS = 8

# Seed centroids derived from expected population distribution
_SEED_CENTROIDS = np.array([
    [-0.8, 0.0, 0.5],   # 0: Very cold-natured, no baro sensitivity, temperate
    [-0.4, 0.5, 0.75],  # 1: Somewhat cold-natured, baro-sensitive, continental
    [-0.3, 0.0, 0.0],   # 2: Slightly cold-natured, tropical
    [0.0, 0.0, 0.5],    # 3: Neutral, temperate
    [0.0, 1.0, 0.75],   # 4: Neutral, very baro-sensitive, continental
    [0.3, 0.0, 0.25],   # 5: Slightly hot-natured, arid
    [0.6, 0.0, 0.0],    # 6: Hot-natured, tropical
    [0.8, 0.5, 0.5],    # 7: Very hot-natured, baro-sensitive, temperate
], dtype=np.float32)

# Cluster personality descriptions
CLUSTER_PERSONALITIES = {
    0: {"title": "The Bundler", "desc": "You chill easily and prefer to overdress. Wind is your nemesis."},
    1: {"title": "The Barometer", "desc": "You feel weather changes in your bones — literally. Pressure drops hit you hard."},
    2: {"title": "The Tropical Transplant", "desc": "You're built for warmth and feel cold creep in early."},
    3: {"title": "The Goldilocks", "desc": "You're right in the middle — sensitive to extremes but comfortable in moderate weather."},
    4: {"title": "The Storm Sensor", "desc": "You're the human weather station. Pressure changes affect your comfort significantly."},
    5: {"title": "The Desert Rose", "desc": "Dry heat is your element. You handle warmth well but wilt in humidity."},
    6: {"title": "The Sun Seeker", "desc": "You thrive in heat and need substantial cold to feel uncomfortable."},
    7: {"title": "The Radiator", "desc": "You run hot and feel overheated before others. Layers are your enemy."},
}

# Per-cluster comfort adjustments (added to base model prediction)
CLUSTER_ADJUSTMENTS = {
    0: {"cold_offset": -0.15, "heat_offset": 0.05, "wind_multiplier": 1.3},
    1: {"cold_offset": -0.10, "heat_offset": 0.0, "pressure_multiplier": 2.0},
    2: {"cold_offset": -0.12, "heat_offset": 0.08, "wind_multiplier": 1.1},
    3: {"cold_offset": 0.0, "heat_offset": 0.0, "wind_multiplier": 1.0},
    4: {"cold_offset": 0.0, "heat_offset": 0.0, "pressure_multiplier": 2.5},
    5: {"cold_offset": 0.05, "heat_offset": 0.10, "humidity_multiplier": 1.4},
    6: {"cold_offset": 0.10, "heat_offset": 0.12, "wind_multiplier": 0.8},
    7: {"cold_offset": 0.15, "heat_offset": -0.10, "wind_multiplier": 0.7},
}


CLIMATE_ZONE_MAP = {
    "tropical": 0.0,
    "arid": 0.25,
    "temperate": 0.5,
    "continental": 0.75,
    "polar": 1.0,
}

BARO_SENSITIVITY_MAP = {
    "no": 0.0,
    "sometimes": 0.5,
    "yes": 1.0,
}


def encode_quiz_answers(quiz: dict) -> np.ndarray:
    """Convert raw quiz answers to a numeric feature vector for clustering."""
    hot_cold = float(quiz.get("hot_cold_slider", 0.0))  # -1 to 1
    baro = BARO_SENSITIVITY_MAP.get(
        str(quiz.get("barometric_sensitivity", "no")).lower(), 0.0
    )
    climate = CLIMATE_ZONE_MAP.get(
        str(quiz.get("climate_zone", "temperate")).lower(), 0.5
    )
    return np.array([hot_cold, baro, climate], dtype=np.float32)


def assign_cluster(quiz_answers: dict) -> int:
    """Assign a user to the nearest archetype cluster based on quiz answers."""
    user_vec = encode_quiz_answers(quiz_answers)
    distances = np.linalg.norm(_SEED_CENTROIDS - user_vec, axis=1)
    return int(np.argmin(distances))


def apply_cluster_adjustment(
    base_score: float,
    cluster_id: int,
    temp_c: float,
    wind_speed_ms: float = 0.0,
    pressure_delta_3h: float = 0.0,
    humidity_pct: float = 50.0,
) -> float:
    """Apply cluster-specific adjustments to a base comfort score."""
    adj = CLUSTER_ADJUSTMENTS.get(cluster_id, {})

    score = base_score

    # Temperature-direction adjustments
    if base_score < 0:  # cold side
        score += adj.get("cold_offset", 0.0)
    else:  # warm side
        score += adj.get("heat_offset", 0.0)

    # Wind sensitivity
    wind_factor = adj.get("wind_multiplier", 1.0)
    if wind_speed_ms > 5:
        wind_effect = (wind_speed_ms - 5) * 0.01 * (wind_factor - 1.0)
        score -= wind_effect  # wind makes it feel colder

    # Pressure sensitivity
    pressure_factor = adj.get("pressure_multiplier", 1.0)
    if abs(pressure_delta_3h) > 2:
        pressure_effect = pressure_delta_3h * 0.005 * pressure_factor
        score -= abs(pressure_effect)  # pressure changes = discomfort

    # Humidity sensitivity
    humidity_factor = adj.get("humidity_multiplier", 1.0)
    if humidity_pct > 70 and base_score > 0:
        humidity_effect = (humidity_pct - 70) * 0.003 * humidity_factor
        score += humidity_effect  # high humidity + heat = worse

    return max(-1.0, min(1.0, score))
