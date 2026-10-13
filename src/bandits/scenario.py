"""Load custom routing phases from JSON."""

import json
from pathlib import Path

from .environment import Hint, Phase
from .validation import validate_phases


class _JSONObject(dict):
    """Retain duplicate-key information until the object's field path is known."""

    def __init__(self, pairs):
        super().__init__()
        self.duplicate_key = None
        for key, value in pairs:
            if key in self and self.duplicate_key is None:
                self.duplicate_key = key
            self[key] = value


def _require_object(value, path: str, required: set[str], optional=()) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    if isinstance(value, _JSONObject) and value.duplicate_key is not None:
        raise ValueError(f"{path}.{value.duplicate_key}: duplicate field")
    for key in sorted(required - value.keys()):
        raise ValueError(f"{path}.{key}: missing required field")
    for key in sorted(value.keys() - required - set(optional)):
        raise ValueError(f"{path}.{key}: unknown field")
    return value


def _require_array(value, path: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected an array")
    return value


def _parse_hint(value, path: str) -> Hint | None:
    if value is None:
        return None
    item = _require_object(value, path, {"probabilities"}, {"confidence"})
    probabilities = tuple(
        _require_array(item["probabilities"], f"{path}.probabilities")
    )
    if "confidence" in item:
        return Hint(probabilities, item["confidence"])
    return Hint(probabilities)


def _parse_phases(document) -> list[Phase]:
    root = _require_object(document, "$", {"phases"})
    items = _require_array(root["phases"], "$.phases")
    phases = []
    for index, value in enumerate(items):
        path = f"$.phases[{index}]"
        item = _require_object(
            value, path, {"name", "rounds", "probabilities"}, {"hint"}
        )
        phases.append(
            Phase(
                name=item["name"],
                rounds=item["rounds"],
                probabilities=tuple(
                    _require_array(item["probabilities"], f"{path}.probabilities")
                ),
                hint=_parse_hint(item.get("hint"), f"{path}.hint"),
            )
        )
    validate_phases(phases)
    return phases


def load_scenario(path: str | Path) -> list[Phase]:
    """Load strict JSON phases; report input errors with a file and field path.

    The root contains only ``phases``. Each phase requires ``name``, ``rounds``
    and ``probabilities`` and may contain a ``hint`` object or null. Hints require
    ``probabilities`` and may contain ``confidence``, defaulting to Hint's 0.9.
    """
    source = Path(path)
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_JSONObject
        )
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{source}: expected UTF-8 JSON text ({error.reason})"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{source}: invalid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    try:
        return _parse_phases(document)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
