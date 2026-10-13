"""Phase diagnostics, paired comparisons, and complete immutable report bundles."""

import base64
import csv
import json
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _groups(traces):
    if not traces:
        raise ValueError("A report needs at least one trace")
    groups, identities = {}, set()
    reference = traces[0].phases
    if not reference:
        raise ValueError("Each trace needs at least one round")
    for trace in traces:
        identity = (trace.name, trace.seed)
        if identity in identities:
            raise ValueError(
                f"Duplicate trace for agent {trace.name}, seed {trace.seed}"
            )
        identities.add(identity)
        if trace.phases != reference:
            raise ValueError("All traces must use the same phase schedule")
        for field in ("actions", "rewards", "regret", "optimal", "hint_weights"):
            values = np.asarray(getattr(trace, field))
            if values.shape != (len(reference),) or not np.isfinite(values).all():
                raise ValueError(
                    f"Trace {trace.name}, seed {trace.seed}: invalid {field}"
                )
        if (
            not np.isin(trace.rewards, [0, 1]).all()
            or not np.isin(trace.optimal, [0, 1]).all()
        ):
            raise ValueError("Trace rewards and optimal-route labels must be binary")
        if np.any(trace.regret < 0) or np.any(
            (trace.hint_weights < 0) | (trace.hint_weights > 1)
        ):
            raise ValueError(
                "Trace regret must be nonnegative and hint weights in [0, 1]"
            )
        groups.setdefault(trace.name, []).append(trace)
    return groups


def _standard_error(values):
    return (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else None
    )


def _phase_indices(schedule):
    indices = {}
    for step, phase in enumerate(schedule):
        indices.setdefault(phase, []).append(step)
    return {phase: np.asarray(steps, dtype=np.intp) for phase, steps in indices.items()}


def _metrics(group, indices=None):
    selected = slice(None) if indices is None else indices
    regrets = [float(t.regret[selected].sum()) for t in group]
    return {
        "rounds": len(group[0].phases) if indices is None else len(indices),
        "mean_successes": float(np.mean([t.rewards[selected].sum() for t in group])),
        "mean_expected_regret": float(np.mean(regrets)),
        "regret_standard_error": _standard_error(regrets),
        "optimal_route_fraction": float(
            np.mean([t.optimal[selected].mean() for t in group])
        ),
        "mean_hint_weight": float(
            np.mean([t.hint_weights[selected].mean() for t in group])
        ),
    }


def summarize(traces):
    summary = []
    groups = _groups(traces)
    indices = _phase_indices(traces[0].phases)
    for name, group in groups.items():
        summary.append(
            {
                "agent": name,
                **_metrics(group),
                "phases": {
                    phase: _metrics(group, steps) for phase, steps in indices.items()
                },
                "per_seed": [
                    {
                        "seed": int(trace.seed),
                        **_metrics([trace]),
                        "phases": {
                            phase: _metrics([trace], steps)
                            for phase, steps in indices.items()
                        },
                    }
                    for trace in sorted(group, key=lambda t: t.seed)
                ],
            }
        )
    return summary


def paired_comparisons(traces, baseline="forgetting-ts"):
    """Positive reduction means less regret than the baseline on matching seeds.

    The uncertainty is calculated on seed-wise differences, not by treating the
    two policies' results as independent observations.
    """
    groups = _groups(traces)
    if baseline not in groups:
        raise ValueError(f"Paired comparison baseline {baseline!r} is missing")
    reference = {trace.seed: trace for trace in groups[baseline]}
    seeds = sorted(reference)
    indices = _phase_indices(traces[0].phases)
    comparisons = []
    for name, group in groups.items():
        if name == baseline:
            continue
        by_seed = {trace.seed: trace for trace in group}
        if set(by_seed) != set(reference):
            raise ValueError(f"Agent {name} must have the same seeds as {baseline}")

        def reduction(steps=None):
            values = []
            selected = slice(None) if steps is None else steps
            for seed in seeds:
                values.append(
                    float(
                        reference[seed].regret[selected].sum()
                        - by_seed[seed].regret[selected].sum()
                    )
                )
            return {
                "mean_regret_reduction": float(np.mean(values)),
                "regret_reduction_standard_error": _standard_error(values),
                "per_seed": [
                    {"seed": int(seed), "regret_reduction": value}
                    for seed, value in zip(seeds, values)
                ],
            }

        comparisons.append(
            {
                "agent": name,
                "baseline": baseline,
                "seeds": [int(seed) for seed in seeds],
                **reduction(),
                "phases": {phase: reduction(steps) for phase, steps in indices.items()},
            }
        )
    return comparisons


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def build_report(traces, phases, discount):
    from .validation import validate_phases

    validate_phases(phases)
    if not np.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError("Discount must be in (0, 1]")
    summary = summarize(traces)
    schedule = [phase.name for phase in phases for _ in range(phase.rounds)]
    if traces[0].phases != schedule:
        raise ValueError("Trace phase schedule does not match the scenario")
    for trace in traces:
        if np.any(trace.actions != np.floor(trace.actions)) or np.any(
            (trace.actions < 0) | (trace.actions >= len(phases[0].probabilities))
        ):
            raise ValueError("Trace actions must be valid route indices")
    return {
        "schema_version": 1,
        "purpose": "Choose service routes when quality changes and advice can become stale",
        "reward": "1 for success, 0 for failure",
        "expected_regret": "sum(max current success probability - chosen route probability)",
        "seeds": sorted({int(trace.seed) for trace in traces}),
        "discount": discount,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "scenario": [asdict(phase) for phase in phases],
        "summary": summary,
        "paired_comparisons": (
            paired_comparisons(traces)
            if any(t.name == "forgetting-ts" for t in traces)
            else []
        ),
        "limitations": [
            "Synthetic Bernoulli routes, not real service measurements",
            "Checked hints use a source-level heuristic evidence gate with five effective observations",
            "Forgetting Beta counts are heuristic under drift, not an exact stationary posterior",
            "Standard errors describe these seed replications, not general-world guarantees",
            "Paired regret reduction is baseline minus agent; positive values favor the agent",
        ],
    }


def _write_decisions(traces, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "agent",
                "seed",
                "round",
                "phase",
                "route",
                "success",
                "expected_regret",
                "optimal",
                "hint_weight",
            ]
        )
        for trace in traces:
            for step in range(len(trace.actions)):
                writer.writerow(
                    [
                        trace.name,
                        trace.seed,
                        step,
                        trace.phases[step],
                        int(trace.actions[step]),
                        int(trace.rewards[step]),
                        float(trace.regret[step]),
                        int(trace.optimal[step]),
                        float(trace.hint_weights[step]),
                    ]
                )


def _plot_comparison(traces, phases, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # An explicit canvas keeps headless export independent of the caller's GUI backend.
    fig = Figure(figsize=(10, 7), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    colors = ["#7e6eb1", "#2873b8", "#b67813", "#13806d"]
    styles = ["--", "-.", ":", "-"]
    for index, (name, group) in enumerate(_groups(traces).items()):
        color, style = colors[index % len(colors)], styles[index % len(styles)]
        regrets = np.array([trace.regret.cumsum() for trace in group])
        mean = regrets.mean(axis=0)
        rounds = np.arange(len(mean))
        axes[0].plot(rounds, mean, label=name, color=color, linestyle=style)
        if len(group) > 1:
            error = regrets.std(axis=0, ddof=1) / np.sqrt(len(group))
            axes[0].fill_between(
                rounds, mean - error, mean + error, alpha=0.12, color=color
            )
        window = min(50, len(mean))
        rates = np.mean(
            [
                np.convolve(
                    trace.optimal.astype(float), np.ones(window) / window, mode="valid"
                )
                for trace in group
            ],
            axis=0,
        )
        axes[1].plot(
            np.arange(window - 1, len(mean)), rates, color=color, linestyle=style
        )
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        for boundary in np.cumsum([phase.rounds for phase in phases])[:-1]:
            ax.axvline(boundary, color="#8d9694", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Cumulative expected regret")
    axes[0].legend(frameon=False, ncols=2)
    axes[1].set_ylabel("Optimal-route fraction (rolling)")
    axes[1].set_xlabel("Round (zero-based)")
    axes[1].set_ylim(0, 1.05)
    fig.savefig(path, dpi=150)


def render_html(report, png):
    """Embed data safely; user-supplied names cannot end the JSON script element."""
    data = (
        json.dumps(report, allow_nan=False, default=_json_default)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    template = (
        Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    )
    return template.replace(
        "__COMPARISON_PNG__", base64.b64encode(png).decode("ascii")
    ).replace("__REPORT_DATA__", data)


def export_report(traces, phases, discount, output_dir):
    """Publish only a fully generated bundle, refusing to replace an existing run."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Report directory already exists: {output_dir}; choose a new directory"
        )
    report = build_report(traces, phases, discount)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as directory:
        staging = Path(directory)
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        _write_decisions(traces, staging / "decisions.csv")
        _plot_comparison(traces, phases, staging / "comparison.png")
        (staging / "report.html").write_text(
            render_html(report, (staging / "comparison.png").read_bytes()),
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(f"Report directory already exists: {output_dir}")
        staging.rename(output_dir)
    return report["summary"]
