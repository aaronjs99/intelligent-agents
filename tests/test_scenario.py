"""Strict scenario loading and allocation-free domain validation."""

import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from src.bandits.environment import Hint, Phase
from src.bandits.scenario import load_scenario
from src.bandits.validation import validate_phases


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "custom scenario.json"

    def load_document(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")
        return load_scenario(self.path)

    def valid_document(self):
        return {
            "phases": [{"name": "normal", "rounds": 2, "probabilities": [0.8, 0.3]}]
        }

    def assert_invalid_document(self, document, field):
        with self.assertRaises(ValueError) as caught:
            self.load_document(document)
        message = str(caught.exception)
        self.assertIn(str(self.path), message)
        self.assertIn(field, message)

    def test_loads_phases_with_optional_advice(self):
        phases = self.load_document(
            {
                "phases": [
                    {
                        "name": "normal",
                        "rounds": 3,
                        "probabilities": [0, 0.8],
                        "hint": {"probabilities": [0.1, 1]},
                    },
                    {
                        "name": "outage",
                        "rounds": 2,
                        "probabilities": [0.9, 0.2],
                        "hint": None,
                    },
                    {
                        "name": "recovery",
                        "rounds": 4,
                        "probabilities": [0.1, 0.95],
                        "hint": {
                            "probabilities": [0.1, 0.95],
                            "confidence": 0.75,
                        },
                    },
                ]
            }
        )
        self.assertEqual(
            phases,
            [
                Phase("normal", 3, (0, 0.8), Hint((0.1, 1), 0.9)),
                Phase("outage", 2, (0.9, 0.2)),
                Phase("recovery", 4, (0.1, 0.95), Hint((0.1, 0.95), 0.75)),
            ],
        )

    def test_json_syntax_error_identifies_file_line_and_column(self):
        self.path.write_text('{\n  "phases": [\n}', encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            load_scenario(self.path)
        message = str(caught.exception)
        self.assertIn(str(self.path), message)
        self.assertRegex(message, r"line 3\b")
        self.assertRegex(message, r"column 1\b")

    def test_invalid_utf8_identifies_the_scenario_file(self):
        self.path.write_bytes(b'{"phases": [\xff]}')
        with self.assertRaises(ValueError) as caught:
            load_scenario(self.path)
        message = str(caught.exception)
        self.assertIn(str(self.path), message)
        self.assertIn("UTF-8", message)

    def test_duplicate_json_fields_are_rejected_with_their_path(self):
        for source, field in [
            ('{"phases": [], "phases": []}', "$.phases"),
            (
                '{"phases": [{"name": "normal", "rounds": 1, "rounds": 2, '
                '"probabilities": [0.8, 0.3]}]}',
                "$.phases[0].rounds",
            ),
            (
                '{"phases": [{"name": "normal", "rounds": 2, '
                '"probabilities": [0.8, 0.3], "hint": {"probabilities": [0.8, 0.3], '
                '"confidence": 0.1, "confidence": 0.9}}]}',
                "$.phases[0].hint.confidence",
            ),
        ]:
            with self.subTest(field=field):
                self.path.write_text(source, encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    load_scenario(self.path)
                message = str(caught.exception)
                self.assertIn(str(self.path), message)
                self.assertIn(field, message)
                self.assertIn("duplicate", message.lower())

    def test_root_must_be_an_object(self):
        for value in [None, True, 3, "scenario", []]:
            with self.subTest(value=value):
                self.assert_invalid_document(value, "$")

    def test_root_requires_phases(self):
        self.assert_invalid_document({}, "$.phases")

    def test_root_rejects_unknown_fields(self):
        document = self.valid_document()
        document["phase"] = []
        self.assert_invalid_document(document, "$.phase")

    def test_phases_must_be_a_nonempty_array(self):
        for value in [None, "normal", {}, 3, []]:
            with self.subTest(value=value):
                self.assert_invalid_document({"phases": value}, "$.phases")

    def test_phase_must_be_an_object(self):
        for value in [None, True, 3, "normal", []]:
            with self.subTest(value=value):
                self.assert_invalid_document({"phases": [value]}, "$.phases[0]")

    def test_phase_requires_each_domain_field(self):
        for field in ["name", "rounds", "probabilities"]:
            with self.subTest(field=field):
                document = self.valid_document()
                del document["phases"][0][field]
                self.assert_invalid_document(document, f"$.phases[0].{field}")

    def test_phase_rejects_unknown_fields(self):
        document = self.valid_document()
        document["phases"][0]["round"] = 2
        self.assert_invalid_document(document, "$.phases[0].round")

    def test_phase_names_must_be_nonempty_strings(self):
        for value in [None, True, 3, "", " \t\n"]:
            with self.subTest(value=value):
                document = self.valid_document()
                document["phases"][0]["name"] = value
                self.assert_invalid_document(document, "$.phases[0].name")

    def test_phase_names_must_be_unique(self):
        document = self.valid_document()
        document["phases"] *= 2
        self.assert_invalid_document(document, "$.phases[1].name")

    def test_rounds_require_a_positive_integer_without_coercion(self):
        for value in [None, True, False, 0, -1, 2.0, "2", []]:
            with self.subTest(value=value):
                document = self.valid_document()
                document["phases"][0]["rounds"] = value
                self.assert_invalid_document(document, "$.phases[0].rounds")

    def test_probabilities_require_an_array(self):
        for value in [None, True, 0.5, "0.8,0.3", {"0": 0.8, "1": 0.3}]:
            with self.subTest(value=value):
                document = self.valid_document()
                document["phases"][0]["probabilities"] = value
                self.assert_invalid_document(document, "$.phases[0].probabilities")

    def test_scenario_requires_at_least_two_routes(self):
        for value in [[], [0.8]]:
            with self.subTest(value=value):
                document = self.valid_document()
                document["phases"][0]["probabilities"] = value
                self.assert_invalid_document(document, "$.phases[0].probabilities")

    def test_all_phases_have_the_same_routes(self):
        document = self.valid_document()
        document["phases"].append(
            {"name": "outage", "rounds": 3, "probabilities": [0.2, 0.8, 0.4]}
        )
        self.assert_invalid_document(document, "$.phases[1].probabilities")

    def test_probabilities_reject_invalid_numbers_without_coercion(self):
        for value in [
            None,
            True,
            False,
            "0.5",
            [],
            {},
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.1,
            1.1,
            10**500,
        ]:
            with self.subTest(value=value):
                document = self.valid_document()
                document["phases"][0]["probabilities"][1] = value
                self.assert_invalid_document(document, "$.phases[0].probabilities[1]")

    def test_hint_requires_an_object_or_null(self):
        for value in [True, 3, "stale", []]:
            with self.subTest(value=value):
                document = self.valid_document()
                document["phases"][0]["hint"] = value
                self.assert_invalid_document(document, "$.phases[0].hint")

    def test_hint_requires_probabilities(self):
        document = self.valid_document()
        document["phases"][0]["hint"] = {"confidence": 0.9}
        self.assert_invalid_document(document, "$.phases[0].hint.probabilities")

    def test_hint_rejects_unknown_fields(self):
        document = self.valid_document()
        document["phases"][0]["hint"] = {
            "probabilities": [0.8, 0.3],
            "confidance": 0.9,
        }
        self.assert_invalid_document(document, "$.phases[0].hint.confidance")

    def test_hint_probabilities_require_an_array(self):
        document = self.valid_document()
        document["phases"][0]["hint"] = {"probabilities": "0.8,0.3"}
        self.assert_invalid_document(document, "$.phases[0].hint.probabilities")

    def test_hint_probabilities_match_route_count(self):
        document = self.valid_document()
        document["phases"][0]["hint"] = {"probabilities": [0.8, 0.3, 0.2]}
        self.assert_invalid_document(document, "$.phases[0].hint.probabilities")

    def test_hint_probabilities_reject_invalid_numbers(self):
        document = self.valid_document()
        document["phases"][0]["hint"] = {"probabilities": [0.8, True]}
        self.assert_invalid_document(document, "$.phases[0].hint.probabilities[1]")

    def test_hint_confidence_is_a_finite_probability_without_coercion(self):
        for value in [
            None,
            True,
            False,
            "0.9",
            [],
            {},
            float("nan"),
            float("inf"),
            -0.1,
            1.1,
            10**500,
        ]:
            with self.subTest(value=value):
                document = self.valid_document()
                document["phases"][0]["hint"] = {
                    "probabilities": [0.8, 0.3],
                    "confidence": value,
                }
                self.assert_invalid_document(document, "$.phases[0].hint.confidence")


class PhaseValidationTests(unittest.TestCase):
    def validate(self, phases):
        return validate_phases(phases)

    def test_accepts_real_numpy_numbers_without_allocating_rounds(self):
        self.assertIsNone(
            self.validate(
                [
                    Phase(
                        "large",
                        10**100,
                        (np.float32(0.8), np.int64(0)),
                        Hint(np.array([0.8, 0.3]), np.float64(1)),
                    )
                ]
            )
        )

    def test_rejects_numpy_booleans_in_probabilities_and_confidence(self):
        for phase, field in [
            (Phase("normal", 2, (0.8, np.bool_(True))), "probabilities[1]"),
            (
                Phase("normal", 2, (0.8, 0.3), Hint((0.8, np.bool_(False)))),
                "hint.probabilities[1]",
            ),
            (
                Phase("normal", 2, (0.8, 0.3), Hint((0.8, 0.3), np.bool_(True))),
                "hint.confidence",
            ),
        ]:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, re.escape(field)):
                    self.validate([phase])

    def test_rounds_require_python_integers(self):
        for value in [np.int64(2), np.bool_(True)]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "rounds"):
                    self.validate([Phase("normal", value, (0.8, 0.3))])

    def test_rejects_complex_and_scalar_array_probabilities(self):
        for value in [complex(0.5), np.complex64(0.5), np.array(0.5)]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"probabilities\[1\]"):
                    self.validate([Phase("normal", 2, (0.8, value))])

    def test_rejects_nested_probability_dimensions(self):
        with self.assertRaisesRegex(ValueError, "probabilities"):
            self.validate([Phase("normal", 2, np.array([[0.8, 0.3]]))])

    def test_rejects_non_phase_and_non_hint_objects(self):
        for phases, field in [
            ([None], "phases[0]"),
            ([{}], "phases[0]"),
            ([Phase("normal", 2, (0.8, 0.3), {})], "hint"),
        ]:
            with self.subTest(field=field, phases=phases):
                with self.assertRaisesRegex(ValueError, re.escape(field)):
                    self.validate(phases)


if __name__ == "__main__":
    unittest.main()
