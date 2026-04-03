"""Reward model — learns to score prediction quality from user feedback.

Architecture: A small neural network that takes (weather features, predicted score,
actual score, tags) and outputs a scalar reward signal used to train the PPO policy.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.services.features import FEATURE_NAMES


class RewardNetwork(nn.Module):
    """3-layer MLP that predicts reward from (features, prediction, feedback)."""

    def __init__(self, n_weather_features: int = len(FEATURE_NAMES)):
        super().__init__()
        # Input: weather features + predicted_score + actual_score + tag flags (4)
        input_dim = n_weather_features + 2 + 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


TAG_INDEX = {"sweaty": 0, "chilled": 1, "energized": 2, "sluggish": 3}


def encode_tags(tags: Optional[list[str]]) -> list[float]:
    """Encode feedback tags as a binary vector."""
    vec = [0.0] * 4
    if tags:
        for tag in tags:
            idx = TAG_INDEX.get(tag.lower())
            if idx is not None:
                vec[idx] = 1.0
    return vec


def compute_ground_truth_reward(
    predicted_score: float,
    actual_score: float,
    tags: Optional[list[str]] = None,
    clothing_correct: bool = True,
) -> float:
    """Compute the ground-truth reward signal from a feedback entry.

    reward = -|predicted - actual|           # accuracy
            + bonus if qualitative match      # tags
            - penalty if clothing was wrong    # actionability
    """
    accuracy_reward = -abs(predicted_score - actual_score)

    tag_bonus = 0.0
    if tags:
        # If user didn't add negative tags, small bonus
        negative_tags = {"sweaty", "chilled", "sluggish"}
        has_negative = any(t.lower() in negative_tags for t in tags)
        if not has_negative:
            tag_bonus = 0.1

    clothing_penalty = 0.0 if clothing_correct else -0.2

    return accuracy_reward + tag_bonus + clothing_penalty


def prepare_reward_training_data(
    feedback_records: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert feedback records into tensors for reward model training.

    Each record should have: feature_vector (dict), predicted_score, comfort_score, tags
    """
    X_list = []
    y_list = []

    for rec in feedback_records:
        fv = rec["feature_vector"]
        weather_feats = [float(fv.get(name, 0.0)) for name in FEATURE_NAMES]
        predicted = rec["predicted_score"]
        actual = rec["comfort_score"]
        tags = rec.get("tags", [])
        tag_vec = encode_tags(tags)

        x = weather_feats + [predicted, actual] + tag_vec
        reward = compute_ground_truth_reward(predicted, actual, tags)

        X_list.append(x)
        y_list.append(reward)

    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(y_list, dtype=torch.float32)


def train_reward_model(
    feedback_records: list[dict],
    model: Optional[RewardNetwork] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> tuple[RewardNetwork, dict]:
    """Train or fine-tune the reward model on accumulated feedback."""
    if model is None:
        model = RewardNetwork()

    X, y = prepare_reward_training_data(feedback_records)
    if len(X) < 5:
        return model, {"status": "insufficient_data", "n_samples": len(X)}

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    metrics = {
        "status": "trained",
        "n_samples": len(X),
        "final_loss": losses[-1],
        "epochs": epochs,
    }
    return model, metrics


def save_reward_model(model: RewardNetwork, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_reward_model(path: str) -> RewardNetwork:
    model = RewardNetwork()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
