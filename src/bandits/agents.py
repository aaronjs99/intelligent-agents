"""One-step planners and a feedback agent; neither sees simulator true rates."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .environment import Hint
from .model import RewardModel


@dataclass(frozen=True)
class Decision:
    action: int
    hint_weight: float = 0.0


class Planner(Protocol):
    def choose(
        self, model: RewardModel, rng: np.random.Generator, hint: Hint | None
    ) -> Decision: ...


class Agent(Protocol):
    def act(self, hint: Hint | None = None) -> Decision: ...
    def observe(self, action: int, reward: int) -> None: ...


class ThompsonPlanner:
    """Posterior sampling with optional advice checked against reward evidence.

    Checking is a source-level heuristic, not a calibrated trust probability.
    Once a route has five effective observations, disagreement beyond two model
    standard deviations attenuates the whole source, allowing sampling to resume.
    """

    def __init__(self, advice="ignore"):
        if advice not in {"ignore", "blind", "checked"}:
            raise ValueError("Advice mode must be ignore, blind or checked")
        self.advice = advice

    def choose(self, model, rng, hint=None):
        scores = model.sample(rng)
        weights = np.zeros_like(scores)
        if hint is not None and self.advice != "ignore":
            suggested = np.asarray(hint.probabilities, dtype=float)
            if (
                suggested.shape != scores.shape
                or not np.isfinite(suggested).all()
                or np.any((suggested < 0) | (suggested > 1))
            ):
                raise ValueError(
                    "Hint probabilities must match the action set and lie in [0, 1]"
                )
            if not np.isfinite(hint.confidence) or not 0 <= hint.confidence <= 1:
                raise ValueError("Hint confidence must be between 0 and 1")
            weights[:] = hint.confidence
            if self.advice == "checked":
                disagreement = np.maximum(
                    np.abs(suggested - model.mean) - 2 * model.standard_deviation, 0
                )
                supported = model.evidence >= 5
                source_error = (
                    float(disagreement[supported].max()) if supported.any() else 0.0
                )
                weights *= np.exp(-source_error / 0.05)
            scores = (1 - weights) * scores + weights * suggested
        action = int(np.argmax(scores))
        return Decision(action, float(weights[action]))


class RewardAgent:
    """Compose an estimate, an action-selection rule, and an isolated RNG stream."""

    def __init__(self, model: RewardModel, planner: Planner, rng: np.random.Generator):
        self.model, self.planner, self.rng = model, planner, rng

    def act(self, hint=None):
        return self.planner.choose(self.model, self.rng, hint)

    def observe(self, action, reward):
        self.model.update(action, reward)
