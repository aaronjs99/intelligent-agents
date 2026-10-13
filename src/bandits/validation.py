"""Validate scenario values without creating simulation outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .environment import Phase


def _validate_probability(value, path: str) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not 0 <= value <= 1
        or not isfinite(value)
    ):
        raise ValueError(f"{path}: expected a finite number in [0, 1]")


def _validate_probabilities(values, path: str, routes: int | None = None) -> int:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, (Sequence, np.ndarray))
        or (isinstance(values, np.ndarray) and values.ndim != 1)
    ):
        raise ValueError(f"{path}: expected a one-dimensional probability sequence")
    count = len(values)
    if routes is None and count < 2:
        raise ValueError(f"{path}: a scenario needs at least two routes")
    if routes is not None and count != routes:
        raise ValueError(f"{path}: expected {routes} probabilities, one per route")
    for index, value in enumerate(values):
        _validate_probability(value, f"{path}[{index}]")
    return count


def validate_phases(phases: Sequence[Phase]) -> None:
    """Require named, positive-length phases sharing a fixed set of routes.

    Each probability and hint confidence must be a finite real number in [0, 1].
    Rounds must be Python integers, excluding booleans. No outcomes or arrays of
    rounds are allocated, so workload limits belong to the experiment runner.
    """
    # Defer the runtime imports so environment can import this validator safely.
    from .environment import Hint, Phase

    if (
        isinstance(phases, (str, bytes))
        or not isinstance(phases, Sequence)
        or not phases
    ):
        raise ValueError("$.phases: expected a nonempty sequence of phases")
    names = set()
    routes = None
    for index, phase in enumerate(phases):
        path = f"$.phases[{index}]"
        if not isinstance(phase, Phase):
            raise ValueError(f"{path}: expected a Phase object")
        if not isinstance(phase.name, str) or not phase.name.strip():
            raise ValueError(f"{path}.name: expected a nonempty string")
        if phase.name in names:
            raise ValueError(f"{path}.name: duplicate phase name {phase.name!r}")
        names.add(phase.name)
        if (
            isinstance(phase.rounds, bool)
            or not isinstance(phase.rounds, int)
            or phase.rounds < 1
        ):
            raise ValueError(f"{path}.rounds: expected a positive Python integer")
        routes = _validate_probabilities(
            phase.probabilities, f"{path}.probabilities", routes
        )
        if phase.hint is not None:
            if not isinstance(phase.hint, Hint):
                raise ValueError(f"{path}.hint: expected a Hint object or None")
            _validate_probabilities(
                phase.hint.probabilities, f"{path}.hint.probabilities", routes
            )
            _validate_probability(phase.hint.confidence, f"{path}.hint.confidence")
