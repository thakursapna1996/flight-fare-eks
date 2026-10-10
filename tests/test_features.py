"""Unit tests for the feature-engineering layer.

These tests need no model, no network and no Flask — they pin down the exact
contract between raw form input and the model's input vector. This is the
test to walk through in the technical interview: each case states a property
of the encoding and checks it in isolation.
"""
import pytest

from app.features import (
    FEATURE_ORDER,
    FeatureError,
    build_feature_vector,
    build_features,
)

# A representative, valid payload reused across tests.
BASE_PAYLOAD = {
    "Dep_Time": "2024-06-15T22:20",
    "Arrival_Time": "2024-06-16T01:10",
    "stops": "1",
    "airline": "IndiGo",
    "Source": "Delhi",
    "Destination": "Cochin",
}


def test_feature_order_is_the_29_the_model_expects():
    assert len(FEATURE_ORDER) == 29
    # No duplicate columns — a duplicate would silently shift the vector.
    assert len(set(FEATURE_ORDER)) == 29


def test_vector_length_matches_feature_order():
    vector = build_feature_vector(BASE_PAYLOAD)
    assert len(vector) == len(FEATURE_ORDER) == 29


def test_datetime_fields_are_decomposed():
    f = build_features(BASE_PAYLOAD)
    assert (f["Journey_day"], f["Journey_month"]) == (15, 6)
    assert (f["Dep_hour"], f["Dep_min"]) == (22, 20)
    assert (f["Arrival_hour"], f["Arrival_min"]) == (1, 10)


def test_duration_spans_midnight_without_going_negative():
    # 22:20 -> 01:10 next day == 2h 50m.
    f = build_features(BASE_PAYLOAD)
    assert (f["Duration_hours"], f["Duration_mins"]) == (2, 50)


def test_airline_is_one_hot():
    f = build_features(BASE_PAYLOAD)
    assert f["Airline_IndiGo"] == 1
    others = [v for k, v in f.items() if k.startswith("Airline_") and k != "Airline_IndiGo"]
    assert sum(others) == 0


def test_dropped_baseline_category_is_all_zeros():
    # "Air Asia" was the dropped baseline at training time, so it must encode
    # as all-zero airline columns rather than raising.
    payload = {**BASE_PAYLOAD, "airline": "Air Asia"}
    f = build_features(payload)
    assert sum(v for k, v in f.items() if k.startswith("Airline_")) == 0


def test_source_and_destination_are_independent_one_hots():
    # Regression test for the original bug where Hyderabad set two columns.
    payload = {**BASE_PAYLOAD, "Source": "Mumbai", "Destination": "Hyderabad"}
    f = build_features(payload)
    assert f["Source_Mumbai"] == 1
    assert sum(v for k, v in f.items() if k.startswith("Source_")) == 1
    assert f["Destination_Hyderabad"] == 1
    assert sum(v for k, v in f.items() if k.startswith("Destination_")) == 1


def test_new_delhi_alias_is_normalised():
    # The HTML form can submit "New_Delhi"; it must map to the trained column.
    payload = {**BASE_PAYLOAD, "Destination": "New_Delhi"}
    f = build_features(payload)
    assert f["Destination_New Delhi"] == 1


@pytest.mark.parametrize("stops", ["-1", "5", "abc", None])
def test_invalid_stops_are_rejected(stops):
    with pytest.raises(FeatureError):
        build_features({**BASE_PAYLOAD, "stops": stops})


def test_missing_datetime_is_rejected():
    payload = {**BASE_PAYLOAD}
    del payload["Dep_Time"]
    with pytest.raises(FeatureError):
        build_features(payload)
