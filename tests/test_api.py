"""Integration tests through the Flask test client.

These load the real model once and exercise the HTTP surface: health/readiness
probes used by Kubernetes, and the JSON prediction endpoint end to end.
"""
VALID = {
    "Dep_Time": "2024-06-15T22:20",
    "Arrival_Time": "2024-06-16T01:10",
    "stops": "1",
    "airline": "IndiGo",
    "Source": "Delhi",
    "Destination": "Cochin",
}


def test_health_is_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_ready_reports_model_loaded(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ready"


def test_api_predict_returns_a_positive_fare(client):
    resp = client.post("/api/predict", json=VALID)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["currency"] == "INR"
    assert isinstance(body["fare"], float)
    assert body["fare"] > 0


def test_api_predict_rejects_bad_input(client):
    resp = client.post("/api/predict", json={**VALID, "stops": "nope"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_home_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<form" in resp.data


def test_html_predict_flow(client):
    resp = client.post("/predict", data=VALID)
    assert resp.status_code == 200
    assert b"predicted Flight Fare" in resp.data
