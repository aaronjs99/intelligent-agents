"""A runnable adaptive routing experiment using small composable bandit components."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

if __package__:
    from .environment import routing_scenario
    from .experiment import (
        compare,
        export_report,
        paired_comparisons,
        validate_comparison,
    )
    from .scenario import load_scenario
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bandits.environment import routing_scenario
    from bandits.experiment import (
        compare,
        export_report,
        paired_comparisons,
        validate_comparison,
    )
    from bandits.scenario import load_scenario


def add_arguments(parser):
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
        help="Rounds for the built-in two-phase scenario",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Independent replications starting at seed 0",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.98,
        help="Retention of past evidence per round, in (0, 1]",
    )
    parser.add_argument(
        "--advice", choices=["fresh", "stale", "misleading", "none"], default="stale"
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Optional JSON scenario; phase rounds override horizon",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adaptive-routing"),
        help="Parent directory for a new run folder with HTML, JSON, CSV and PNG",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check scenario and workload without simulation or writing files",
    )


def run_from_args(args):
    # Redirected Windows streams may use a legacy encoding. Preserve UTF-8 data
    # files while making unsupported console characters safe to print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    phases = (
        load_scenario(args.scenario)
        if args.scenario
        else routing_scenario(args.horizon, args.advice)
    )
    decisions = validate_comparison(phases, args.seeds, args.discount)
    print(
        f"Scenario valid: {sum(phase.rounds for phase in phases):,} rounds, {len(phases[0].probabilities)} routes, {len(phases)} phases, {args.seeds} seeds; {decisions:,} decisions."
    )
    if args.validate_only:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / f"run-{stamp}-{uuid4().hex[:8]}"
    print("Running paired simulation...", flush=True)
    traces = compare(phases, args.seeds, args.discount)
    print("Writing report...", flush=True)
    summary = export_report(traces, phases, args.discount, output_dir)
    print("Service routing: compare learning, forgetting, and uncertain advice")
    print(
        "Expected regret is evaluated against the best route in each phase; lower is better."
    )
    print(
        f"{'Agent':<20} {'Successes':>10} {'Regret':>10} {'SE':>8} {'Last phase optimal %':>20}"
    )
    last_phase = phases[-1].name
    for item in summary:
        error = item["regret_standard_error"]
        se = f"{error:.2f}" if error is not None else "n/a"
        optimal = item["phases"][last_phase]["optimal_route_fraction"] * 100
        print(
            f"{item['agent']:<20} {item['mean_successes']:>10.1f} {item['mean_expected_regret']:>10.2f} {se:>8} {optimal:>20.1f}"
        )
    checked = next(
        pair for pair in paired_comparisons(traces) if pair["agent"] == "checked-hints"
    )
    print(
        "\nChecked hints: paired regret saved vs forgetting (positive favors checking)"
    )
    for phase in phases:
        result = checked["phases"][phase.name]
        error = result["regret_reduction_standard_error"]
        se = f"{error:.2f}" if error is not None else "n/a"
        print(f"  {phase.name}: {result['mean_regret_reduction']:+.2f} (SE {se})")
    print(f"Report: {(output_dir / 'report.html').resolve()}")
    print(
        "Synthetic demonstration; existing methods and an evidence-gating heuristic, not a claimed novel algorithm."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive service routing with uncertain advice"
    )
    add_arguments(parser)
    args = parser.parse_args()
    try:
        run_from_args(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Completed runs, if any, are preserved.\n")


if __name__ == "__main__":
    main()
