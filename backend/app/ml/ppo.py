"""PPO (Proximal Policy Optimization) for personalized comfort prediction.

The policy network maps (weather features + user embedding) → comfort prediction.
It is trained using rewards from the reward model to learn each user's preferences.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from app.services.features import FEATURE_NAMES


class ComfortPolicy(nn.Module):
    """Policy network: maps weather features + user embedding → comfort score distribution."""

    def __init__(
        self,
        n_features: int = len(FEATURE_NAMES),
        user_embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.user_embedding = nn.Linear(8, user_embed_dim)  # 8 = cluster features
        input_dim = n_features + user_embed_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Continuous action: comfort score mean and log-std
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, weather_features: torch.Tensor, user_features: torch.Tensor):
        user_embed = self.user_embedding(user_features)
        x = torch.cat([weather_features, user_embed], dim=-1)
        h = self.shared(x)
        mean = torch.tanh(self.mean_head(h).squeeze(-1))  # clamp to [-1, 1]
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, weather_features: torch.Tensor, user_features: torch.Tensor):
        mean, std = self.forward(weather_features, user_features)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action)
        return action, log_prob, mean

    def evaluate_action(
        self,
        weather_features: torch.Tensor,
        user_features: torch.Tensor,
        action: torch.Tensor,
    ):
        mean, std = self.forward(weather_features, user_features)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value function for PPO advantage estimation."""

    def __init__(self, n_features: int = len(FEATURE_NAMES), user_embed_dim: int = 16):
        super().__init__()
        self.user_embedding = nn.Linear(8, user_embed_dim)
        input_dim = n_features + user_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, weather_features: torch.Tensor, user_features: torch.Tensor):
        user_embed = self.user_embedding(user_features)
        x = torch.cat([weather_features, user_embed], dim=-1)
        return self.net(x).squeeze(-1)


class PPOTrainer:
    """PPO training loop for updating a user's personalized comfort policy."""

    def __init__(
        self,
        policy: Optional[ComfortPolicy] = None,
        value_net: Optional[ValueNetwork] = None,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        epochs_per_update: int = 4,
    ):
        self.policy = policy or ComfortPolicy()
        self.value_net = value_net or ValueNetwork()
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.epochs_per_update = epochs_per_update

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def update(self, rollout: dict) -> dict:
        """Run a PPO update step on collected experience.

        rollout keys:
            weather_features: (N, n_features)
            user_features: (N, 8)
            actions: (N,) — comfort scores predicted
            rewards: (N,) — from reward model
            old_log_probs: (N,)
        """
        weather = torch.tensor(rollout["weather_features"], dtype=torch.float32)
        user = torch.tensor(rollout["user_features"], dtype=torch.float32)
        actions = torch.tensor(rollout["actions"], dtype=torch.float32)
        rewards = torch.tensor(rollout["rewards"], dtype=torch.float32)
        old_log_probs = torch.tensor(rollout["old_log_probs"], dtype=torch.float32)

        # Compute advantages
        with torch.no_grad():
            values = self.value_net(weather, user)
            advantages = rewards - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.epochs_per_update):
            # Policy loss
            new_log_probs, entropy = self.policy.evaluate_action(weather, user, actions)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

            # Value loss
            value_pred = self.value_net(weather, user)
            value_loss = nn.functional.mse_loss(value_pred, rewards)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            metrics["policy_loss"] = policy_loss.item()
            metrics["value_loss"] = value_loss.item()
            metrics["entropy"] = entropy.mean().item()

        return metrics

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(dir_path, "policy.pt"))
        torch.save(self.value_net.state_dict(), os.path.join(dir_path, "value.pt"))

    def load(self, dir_path: str):
        self.policy.load_state_dict(
            torch.load(os.path.join(dir_path, "policy.pt"), map_location="cpu")
        )
        self.value_net.load_state_dict(
            torch.load(os.path.join(dir_path, "value.pt"), map_location="cpu")
        )
