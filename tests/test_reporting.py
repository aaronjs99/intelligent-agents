import tempfile
import unittest
import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.bandits import experiment
from src.bandits.environment import Phase
from src.bandits.experiment import Trace


def comparison_fixture():
    phases = [Phase("healthy", 2, (1.0, 0.8)), Phase("outage", 2, (0.5, 1.0))]
    traces = []
    for name, seed, actions, rewards, regret in [
        ("forgetting-ts", 0, [0, 1, 0, 0], [1, 0, 0, 1], [0, 0.2, 0.5, 0.5]),
        ("checked-hints", 0, [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]),
        ("forgetting-ts", 1, [0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0.5, 0]),
        ("checked-hints", 1, [1, 1, 0, 1], [0, 0, 0, 1], [0.2, 0.2, 0.5, 0]),
    ]:
        traces.append(
            Trace(
                name,
                seed,
                np.array(actions),
                np.array(rewards),
                np.array(regret),
                np.array(regret) == 0,
                np.full(4, 0.9 if name == "checked-hints" else 0),
                ["healthy", "healthy", "outage", "outage"],
            )
        )
    return phases, traces


class ReportingTests(unittest.TestCase):
    def test_template_marker_in_a_phase_name_survives_html_data_round_trip(self):
        from src.bandits.reporting import render_html

        data = {"scenario": [{"name": "__COMPARISON_PNG__"}]}
        html = render_html(data, b"example png")
        embedded = re.search(
            r'<script id="run-data" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(embedded)
        self.assertEqual(json.loads(embedded.group(1)), data)

    def test_phase_summary_exposes_successes_and_uncertainty(self):
        _, traces = comparison_fixture()
        checked = experiment.summarize(traces)[1]
        self.assertIn("mean_successes", checked["phases"]["healthy"])
        self.assertEqual(checked["phases"]["healthy"]["mean_successes"], 1)
        self.assertAlmostEqual(
            checked["phases"]["healthy"]["regret_standard_error"], 0.2
        )
        self.assertEqual(checked["per_seed"][0]["mean_expected_regret"], 0)
        self.assertAlmostEqual(checked["per_seed"][1]["mean_expected_regret"], 0.9)

    def test_paired_reduction_uses_matching_seeds_and_phase(self):
        _, traces = comparison_fixture()
        self.assertTrue(
            callable(getattr(experiment, "paired_comparisons", None)),
            "Paired comparison API is missing",
        )
        result = experiment.paired_comparisons(list(reversed(traces)))[0]
        self.assertEqual(result["agent"], "checked-hints")
        self.assertEqual(result["baseline"], "forgetting-ts")
        self.assertEqual(result["seeds"], [0, 1])
        self.assertAlmostEqual(result["mean_regret_reduction"], 0.4)
        self.assertAlmostEqual(result["regret_reduction_standard_error"], 0.8)
        self.assertAlmostEqual(
            result["phases"]["healthy"]["mean_regret_reduction"], -0.1
        )
        self.assertAlmostEqual(result["phases"]["outage"]["mean_regret_reduction"], 0.5)

    def test_single_seed_has_no_estimated_standard_error(self):
        _, traces = comparison_fixture()
        self.assertTrue(
            callable(getattr(experiment, "paired_comparisons", None)),
            "Paired comparison API is missing",
        )
        result = experiment.paired_comparisons(traces[:2])[0]
        self.assertIsNone(result["regret_reduction_standard_error"])

    def test_unpaired_seeds_are_rejected_instead_of_silently_compared(self):
        _, traces = comparison_fixture()
        self.assertTrue(
            callable(getattr(experiment, "paired_comparisons", None)),
            "Paired comparison API is missing",
        )
        with self.assertRaisesRegex(ValueError, "seed"):
            experiment.paired_comparisons(traces[:-1])

    def test_existing_report_is_preserved(self):
        phases, traces = comparison_fixture()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "completed"
            target.mkdir()
            previous = target / "summary.json"
            previous.write_text("previous run", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                experiment.export_report(traces, phases, 0.98, target)
            self.assertEqual(previous.read_text(encoding="utf-8"), "previous run")

    def test_empty_traces_do_not_publish_a_report(self):
        phases, _ = comparison_fixture()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "empty"
            with self.assertRaisesRegex(ValueError, "trace"):
                experiment.export_report([], phases, 0.98, target)
            self.assertFalse(target.exists())

    def test_failed_chart_does_not_leave_a_partial_bundle(self):
        phases, traces = comparison_fixture()
        self.assertTrue(callable(getattr(experiment, "paired_comparisons", None)))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "interrupted"
            with patch(
                "src.bandits.reporting._plot_comparison",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    experiment.export_report(traces, phases, 0.98, target)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_bundle_contains_consistent_data_and_safe_offline_html(self):
        phases, traces = comparison_fixture()
        unsafe = "</script><script>globalThis.injected=true</script>"
        phases[0] = Phase(unsafe, 2, (1.0, 0.8))
        for trace in traces:
            trace.phases[:2] = [unsafe, unsafe]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "complete"
            experiment.export_report(traces, phases, 0.98, target)
            self.assertEqual(
                {p.name for p in target.iterdir()},
                {"summary.json", "decisions.csv", "comparison.png", "report.html"},
            )
            report = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            html = (target / "report.html").read_text(encoding="utf-8")
            self.assertEqual(report["scenario"][0]["name"], unsafe)
            self.assertNotIn(unsafe, html)
            self.assertIn("data:image/png;base64,", html)
            self.assertNotIn("__REPORT_DATA__", html)
            self.assertAlmostEqual(
                report["paired_comparisons"][0]["mean_regret_reduction"], 0.4
            )
            self.assertEqual(
                len((target / "decisions.csv").read_text().splitlines()), 17
            )

    def test_mismatched_schedule_and_duplicate_seed_fail_before_writing(self):
        phases, traces = comparison_fixture()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bad"
            with self.assertRaisesRegex(ValueError, "Duplicate trace"):
                experiment.export_report(traces + traces[:1], phases, 0.98, target)
            self.assertFalse(target.exists())
            traces[0].phases[0] = "wrong"
            with self.assertRaisesRegex(ValueError, "phase schedule"):
                experiment.export_report(traces, phases, 0.98, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
