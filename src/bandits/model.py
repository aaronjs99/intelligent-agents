"""Reusable Beta reward estimates with optional forgetting toward the prior."""

from typing import Protocol

import numpy as np


class RewardModel(Protocol):
    @property
    def mean(self) -> np.ndarray: ...
    @property
    def standard_deviation(self) -> np.ndarray: ...
    @property
    def evidence(self) -> np.ndarray: ...
    def sample(self, rng: np.random.Generator) -> np.ndarray: ...
    def update(self, action: int, reward: int) -> None: ...


class BetaRewardModel:
    def __init__(self, num_actions: int, discount: float = 1.0):
        if num_actions < 1 or not np.isfinite(discount) or not 0 < discount <= 1:
            raise ValueError("Use at least one action and a discount in (0, 1]")
        self.discount = discount
        self.successes = np.zeros(num_actions)
        self.failures = np.zeros(num_actions)

    @property
    def evidence(self):
        return self.successes + self.failures

    @property
    def mean(self):
        return (self.successes + 1) / (self.successes + self.failures + 2)

    @property
    def standard_deviation(self):
        alpha, beta = self.successes + 1, self.failures + 1
        total = alpha + beta
        return np.sqrt(alpha * beta / (total**2 * (total + 1)))

    def sample(self, rng):
        return rng.beta(self.successes + 1, self.failures + 1)

    def update(self, action: int, reward: int):
        if reward not in (0, 1) or not 0 <= action < len(self.successes):
            raise ValueError("Observe a valid action and binary reward")
        self.successes *= self.discount
        self.failures *= self.discount
        self.successes[action] += reward
        self.failures[action] += 1 - reward
