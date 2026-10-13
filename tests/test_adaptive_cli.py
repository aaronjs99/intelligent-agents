import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.bandits.environment import Phase
from src.bandits.experiment import compare, validate_comparison

ROOT = Path(__file__).resolve().parents[1]


def command(*args, cwd=ROOT, direct=False):
    entrypoint = ROOT / "src" / "bandits" / "adaptive.py" if direct else ROOT / "run.py"
    prefix = [] if direct else ["adapt"]
    return subprocess.run(
        [sys.executable, str(entrypoint), *prefix, *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=20,
    )


class AdaptiveCLITests(unittest.TestCase):
    def test_validate_only_checks_without_creating_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            result = command(
                "--horizon", 20, "--seeds", 2, "--validate-only", "--output-dir", output
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("20 rounds", result.stdout)
            self.assertIn("160 decisions", result.stdout)
            self.assertFalse(output.exists())

    def test_two_runs_preserve_both_complete_bundles_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            for _ in range(2):
                result = command(
                    "--horizon", 10, "--seeds", 1, "--output-dir", output, cwd=directory
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("report.html", result.stdout)
            runs = list(output.iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                data = json.loads((run / "summary.json").read_text())
                self.assertEqual(data["seeds"], [0])
                self.assertEqual(len(data["summary"]), 4)
                self.assertTrue((run / "report.html").is_file())

    def test_malformed_scenario_has_a_field_path_and_no_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario = Path(directory) / "bad.json"
            scenario.write_text(
                '{"phases":[{"name":"outage","rounds":true,"probabilities":[0.1,0.9]}]}'
            )
            result = command("--scenario", scenario, "--validate-only")
            self.assertEqual(result.returncode, 2)
            self.assertIn("phases[0].rounds", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_direct_entrypoint_validates_caller_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario = Path(directory) / "custom.json"
            scenario.write_text(
                '{"phases":[{"name":"outage","rounds":7,"probabilities":[0.1,0.9]}]}'
            )
            result = command(
                "--scenario",
                "custom.json",
                "--validate-only",
                cwd=directory,
                direct=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("7 rounds", result.stdout)

    def test_oversized_workload_is_rejected_before_allocating(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            result = command(
                "--horizon", 1_000_000_000, "--seeds", 1000, "--output-dir", output
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("decision", result.stderr.lower())
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_invalid_replication_count_is_rejected_as_a_value_error(self):
        phases = [Phase("normal", 2, (0.1, 0.9))]
        for seeds in [True, 1.5, 0]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                compare(phases, seeds=seeds)

    def test_route_heavy_workload_is_bounded_across_all_seeds(self):
        phases = [Phase("normal", 500, (0.5,) * 10_000)]
        with self.assertRaisesRegex(ValueError, "route"):
            validate_comparison(phases, seeds=1000)

    def test_excessive_phase_diagnostics_are_bounded_before_running(self):
        phases = [Phase(f"phase {i}", 1, (0.5, 0.7)) for i in range(1000)]
        with self.assertRaisesRegex(ValueError, "summar"):
            validate_comparison(phases, seeds=10)

    def test_unicode_phase_names_work_with_windows_redirected_output(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario = Path(directory) / "custom.json"
            scenario.write_text(
                json.dumps(
                    {
                        "phases": [
                            {
                                "name": "\u4e2d\u6587",
                                "rounds": 2,
                                "probabilities": [0.2, 0.8],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = Path(directory) / "runs"
            environment = dict(os.environ, PYTHONIOENCODING="cp1252")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "adapt",
                    "--scenario",
                    str(scenario),
                    "--seeds",
                    "1",
                    "--output-dir",
                    str(output),
                ],
                env=environment,
                capture_output=True,
                text=True,
                encoding="cp1252",
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            runs = list(output.iterdir())
            self.assertEqual(len(runs), 1)
            data = json.loads((runs[0] / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(data["scenario"][0]["name"], "\u4e2d\u6587")


if __name__ == "__main__":
    unittest.main()
