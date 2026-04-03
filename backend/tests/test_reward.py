"""Tests for the reward model."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.reward_model import (
    compute_ground_truth_reward,
    encode_tags,
    prepare_reward_training_data,
    train_reward_model,
)
from app.services.features import FEATURE_NAMES


class TestRewardComputation:
    def test_perfect_prediction_high_reward(self):
        reward = compute_ground_truth_reward(0.5, 0.5)
        assert reward >= 0.0

    def test_bad_prediction_negative_reward(self):
        reward = compute_ground_truth_reward(0.8, -0.8)
        assert reward < -1.0

    def test_negative_tags_no_bonus(self):
        reward_with = compute_ground_truth_reward(0.3, 0.3, tags=["sweaty"])
        reward_without = compute_ground_truth_reward(0.3, 0.3, tags=["energized"])
        assert reward_without > reward_with

    def test_wrong_clothing_penalty(self):
        r1 = compute_ground_truth_reward(0.3, 0.3, clothing_correct=True)
        r2 = compute_ground_truth_reward(0.3, 0.3, clothing_correct=False)
        assert r1 > r2


class TestEncodeTags:
    def test_empty_tags(self):
        assert encode_tags(None) == [0.0, 0.0, 0.0, 0.0]
        assert encode_tags([]) == [0.0, 0.0, 0.0, 0.0]

    def test_single_tag(self):
        vec = encode_tags(["sweaty"])
        assert vec[0] == 1.0
        assert sum(vec) == 1.0

    def test_multiple_tags(self):
        vec = encode_tags(["sweaty", "sluggish"])
        assert vec[0] == 1.0
        assert vec[3] == 1.0


class TestRewardModelTraining:
    def test_insufficient_data(self):
        records = [
            {
                "feature_vector": {name: 0.0 for name in FEATURE_NAMES},
                "predicted_score": 0.0,
                "comfort_score": 0.1,
                "tags": [],
            }
        ]
        model, metrics = train_reward_model(records)
        assert metrics["status"] == "insufficient_data"

    def test_training_with_enough_data(self):
        records = [
            {
                "feature_vector": {name: float(i * 0.1) for name in FEATURE_NAMES},
                "predicted_score": i * 0.1,
                "comfort_score": i * 0.1 + 0.05,
                "tags": ["energized"] if i % 2 == 0 else [],
            }
            for i in range(20)
        ]
        model, metrics = train_reward_model(records, epochs=5)
        assert metrics["status"] == "trained"
        assert metrics["final_loss"] >= 0
