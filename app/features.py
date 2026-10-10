"""Feature engineering for the flight-fare model.

This module is the single source of truth for how raw trip details become the
numeric feature vector the model consumes. ``train.py`` and the web app both
import ``FEATURE_ORDER`` and ``build_feature_vector`` from here, which
guarantees the features used at training time and at serving time are
identical and in the same order. That removes a whole class of silent
"training/serving skew" bugs.

It has no Flask or AWS dependencies, so it can be unit-tested in isolation
(see ``tests/test_features.py``) — this is the module to deep-dive in the
technical walkthrough.
"""
from __future__ import annotations

from datetime import datetime

# ---------------------------------------------------------------------------
# Categorical vocabularies.
#
# The model was trained with pandas.get_dummies(..., drop_first=True), which
# drops the first category alphabetically. Those dropped baselines are encoded
# implicitly as "all zeros", exactly as they were during training:
#   - Airline     -> "Air Asia" is the baseline
#   - Source      -> "Banglore" is the baseline
#   - Destination -> "Banglore" is the baseline
# ---------------------------------------------------------------------------
AIRLINES = [
    "Air India",
    "GoAir",
    "IndiGo",
    "Jet Airways",
    "Jet Airways Business",
    "Multiple carriers",
    "Multiple carriers Premium economy",
    "SpiceJet",
    "Trujet",
    "Vistara",
    "Vistara Premium economy",
]
SOURCES = ["Chennai", "Delhi", "Kolkata", "Mumbai"]
DESTINATIONS = ["Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"]

# Numeric features, in the exact order the model expects them, followed by the
# one-hot columns. This list defines the column order of the training matrix.
NUMERIC_FEATURES = [
    "Total_Stops",
    "Journey_day",
    "Journey_month",
    "Dep_hour",
    "Dep_min",
    "Arrival_hour",
    "Arrival_min",
    "Duration_hours",
    "Duration_mins",
]

FEATURE_ORDER = (
    NUMERIC_FEATURES
    + [f"Airline_{a}" for a in AIRLINES]
    + [f"Source_{s}" for s in SOURCES]
    + [f"Destination_{d}" for d in DESTINATIONS]
)

# Accept the odd value spellings the HTML form can submit and normalise them
# to the canonical vocabulary above.
_DESTINATION_ALIASES = {"New_Delhi": "New Delhi"}


class FeatureError(ValueError):
    """Raised when the incoming payload cannot be turned into features."""


def _parse_dt(value: str, field: str) -> datetime:
    """Parse an HTML ``datetime-local`` value (``YYYY-MM-DDTHH:MM``)."""
    if not value:
        raise FeatureError(f"'{field}' is required")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M")
    except ValueError as exc:  # pragma: no cover - message is the point
        raise FeatureError(
            f"'{field}' must look like 2024-06-15T22:20, got {value!r}"
        ) from exc


def _duration(dep: datetime, arr: datetime) -> tuple[int, int]:
    """Whole-hours / minutes between departure and arrival (never negative)."""
    delta = arr - dep
    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes < 0:
        # Arrival on the next day without a date component: wrap by 24h.
        total_minutes += 24 * 60
    return total_minutes // 60, total_minutes % 60


def _one_hot(value: str, vocabulary: list[str], prefix: str) -> dict[str, int]:
    return {f"{prefix}_{name}": int(name == value) for name in vocabulary}


def build_features(payload: dict) -> dict[str, int]:
    """Turn a raw form/JSON payload into a ``{feature_name: value}`` mapping.

    Expected keys: ``Dep_Time``, ``Arrival_Time`` (datetime-local strings),
    ``stops``, ``airline``, ``Source``, ``Destination``.
    """
    dep = _parse_dt(payload.get("Dep_Time", ""), "Dep_Time")
    arr = _parse_dt(payload.get("Arrival_Time", ""), "Arrival_Time")
    dur_hours, dur_mins = _duration(dep, arr)

    try:
        total_stops = int(payload.get("stops"))
    except (TypeError, ValueError) as exc:
        raise FeatureError("'stops' must be an integer 0-4") from exc
    if not 0 <= total_stops <= 4:
        raise FeatureError("'stops' must be between 0 and 4")

    airline = (payload.get("airline") or "").strip()
    source = (payload.get("Source") or "").strip()
    destination = (payload.get("Destination") or "").strip()
    destination = _DESTINATION_ALIASES.get(destination, destination)

    features: dict[str, int] = {
        "Total_Stops": total_stops,
        "Journey_day": dep.day,
        "Journey_month": dep.month,
        "Dep_hour": dep.hour,
        "Dep_min": dep.minute,
        "Arrival_hour": arr.hour,
        "Arrival_min": arr.minute,
        "Duration_hours": dur_hours,
        "Duration_mins": dur_mins,
    }
    features.update(_one_hot(airline, AIRLINES, "Airline"))
    features.update(_one_hot(source, SOURCES, "Source"))
    features.update(_one_hot(destination, DESTINATIONS, "Destination"))
    return features


def build_feature_vector(payload: dict) -> list[int]:
    """Return the feature values as a single row, ordered by ``FEATURE_ORDER``.

    This is what gets passed to ``model.predict([...])``.
    """
    features = build_features(payload)
    return [features[name] for name in FEATURE_ORDER]
