"""Reproducible policy comparisons, trace exports, and a concrete routing report."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .agents import Agent, RewardAgent, ThompsonPlanner
from .environment import DecisionEnvironment, Phase, ServiceRoutes
from .model import BetaRewardModel
from .validation import validate_phases

# Trace exports retain every decision in memory; bound work before allocation.
MAX_DECISIONS = 2_000_000
MAX_POTENTIAL_OUTCOMES = 5_000_000
MAX_ROUTE_EVALUATIONS = 20_000_000
MAX_PHASE_SUMMARIES = 10_000


def validate_comparison(phases, seeds=10, discount=0.98):
    validate_phases(phases)
    if (
        isinstance(discount, (bool, np.bool_))
        or not isinstance(discount, Real)
        or not np.isfinite(discount)
        or not 0 < discount <= 1
    ):
        raise ValueError("Discount must be a finite number in (0, 1]")
    if (
        isinstance(seeds, (bool, np.bool_))
        or not isinstance(seeds, Integral)
        or not 1 <= seeds <= 1000
    ):
        raise ValueError("Use an integer between 1 and 1000 seeds")
    horizon = sum(phase.rounds for phase in phases)
    decisions = horizon * int(seeds) * 4
    if decisions > MAX_DECISIONS:
        raise ValueError(
            f"This run needs {decisions:,} decisions; the in-memory report limit is {MAX_DECISIONS:,}. Reduce rounds or seeds."
        )
    outcomes = horizon * len(phases[0].probabilities)
    if outcomes > MAX_POTENTIAL_OUTCOMES:
        raise ValueError(
            f"This run needs {outcomes:,} potential outcomes per seed; the limit is {MAX_POTENTIAL_OUTCOMES:,}. Reduce rounds or routes."
        )
    evaluations = decisions * len(phases[0].probabilities)
    if evaluations > MAX_ROUTE_EVALUATIONS:
        raise ValueError(
            f"This run needs {evaluations:,} route evaluations; the limit is {MAX_ROUTE_EVALUATIONS:,}. Reduce rounds, routes or seeds."
        )
    summaries = 4 * (int(seeds) + 1) * (len(phases) + 1)
    if summaries > MAX_PHASE_SUMMARIES:
        raise ValueError(
            f"This run needs {summaries:,} phase/seed summaries; the limit is {MAX_PHASE_SUMMARIES:,}. Reduce phases or seeds."
        )
    return decisions


@dataclass
class Trace:
    name: str
    seed: int
    actions: np.ndarray
    rewards: np.ndarray
    regret: np.ndarray
    optimal: np.ndarray
    hint_weights: np.ndarray
    phases: list[str]


def run_episode(environment: DecisionEnvironment, agent: Agent, name: str, seed: int):
    """Only observed reward reaches the learner; oracle metrics remain evaluator data."""
    actions, rewards, regret, optimal, weights, phases = [], [], [], [], [], []
    for step in range(environment.horizon):
        decision = agent.act(environment.advice(step))
        outcome = environment.step(decision.action, step)
        agent.observe(decision.action, outcome.reward)
        actions.append(decision.action)
        rewards.append(outcome.reward)
        regret.append(outcome.regret)
        optimal.append(outcome.optimal)
        weights.append(decision.hint_weight)
        phases.append(outcome.phase)
    return Trace(
        name,
        seed,
        np.array(actions),
        np.array(rewards),
        np.array(regret),
        np.array(optimal),
        np.array(weights),
        phases,
    )


def compare(phases: list[Phase], seeds: int = 10, discount: float = 0.98):
    validate_comparison(phases, seeds, discount)
    policies = {
        "stationary-ts": (1.0, "ignore"),
        "forgetting-ts": (discount, "ignore"),
        "blind-hints": (discount, "blind"),
        "checked-hints": (discount, "checked"),
    }
    traces = []
    for seed in range(seeds):
        environment = ServiceRoutes(phases, seed)
        for name, (forgetting, advice) in policies.items():
            # The same sampling stream pairs policies; outcome randomness is separate.
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            agent = RewardAgent(
                BetaRewardModel(environment.num_actions, forgetting),
                ThompsonPlanner(advice),
                rng,
            )
            traces.append(run_episode(environment, agent, name, seed))
    return traces


def summarize(traces):
    """Compatibility entry point; report calculations live apart from simulation."""
    from .reporting import summarize as summarize_traces

    return summarize_traces(traces)


def paired_comparisons(traces, baseline="forgetting-ts"):
    from .reporting import paired_comparisons as compare_pairs

    return compare_pairs(traces, baseline)


def export_report(traces, phases, discount, output_dir):
    from .reporting import export_report as write_report

    return write_report(traces, phases, discount, output_dir)
