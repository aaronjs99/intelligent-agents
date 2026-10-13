"""Changing service quality and external advice, independent of learning policies."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .validation import validate_phases


@dataclass(frozen=True)
class Hint:
    probabilities: tuple[float, ...]
    confidence: float = 0.9


@dataclass(frozen=True)
class Phase:
    name: str
    rounds: int
    probabilities: tuple[float, ...]
    hint: Hint | None = None


@dataclass(frozen=True)
class Outcome:
    reward: int
    regret: float
    optimal: bool
    phase: str


class DecisionEnvironment(Protocol):
    num_actions: int
    horizon: int

    def advice(self, step: int) -> Hint | None: ...
    def step(self, action: int, step: int) -> Outcome: ...


class ServiceRoutes:
    """Simulated independent Bernoulli successes; true rates stay out of the agent."""

    def __init__(self, phases: list[Phase], seed: int):
        validate_phases(phases)
        self.num_actions = len(phases[0].probabilities)
        self.phases = phases
        self.boundaries = np.cumsum([phase.rounds for phase in phases])
        self.horizon = int(self.boundaries[-1])
        # Counterfactual outcomes are shared across policies for the same seed.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self._uniforms = rng.random((self.horizon, self.num_actions))

    def _phase(self, step):
        if not 0 <= step < self.horizon:
            raise IndexError("Step is outside the scenario horizon")
        return self.phases[int(np.searchsorted(self.boundaries, step, side="right"))]

    def advice(self, step):
        return self._phase(step).hint

    def step(self, action, step):
        if not 0 <= action < self.num_actions:
            raise IndexError("Action is outside the route set")
        phase = self._phase(step)
        rates = phase.probabilities
        best = max(rates)
        return Outcome(
            int(self._uniforms[step, action] < rates[action]),
            best - rates[action],
            rates[action] == best,
            phase.name,
        )


def routing_scenario(horizon=1000, advice="stale"):
    """An initially strong route degrades; another route becomes the best choice."""
    if (
        isinstance(horizon, (bool, np.bool_))
        or not isinstance(horizon, int)
        or horizon < 2
    ):
        raise ValueError("The built-in scenario needs at least two rounds")
    if advice not in {"fresh", "stale", "misleading", "none"}:
        raise ValueError("Unknown advice scenario")
    before, after = (0.85, 0.55, 0.35), (0.30, 0.80, 0.55)
    if advice == "none":
        hints = [None, None]
    elif advice == "misleading":
        hints = [Hint((0.20, 0.35, 0.90))] * 2
    else:
        hints = [Hint(before), Hint(after if advice == "fresh" else before)]
    half = horizon // 2
    return [
        Phase("healthy", half, before, hints[0]),
        Phase("degraded", horizon - half, after, hints[1]),
    ]
