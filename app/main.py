"""Flask application entry point.

Routes:
    GET  /            -> HTML form
    POST /predict     -> HTML form + predicted fare (browser flow)
    POST /api/predict -> JSON in / JSON out (programmatic flow, used by tests)
    GET  /health      -> liveness probe (process is up)
    GET  /ready       -> readiness probe (model is loaded and can predict)

The app is built with a factory (``create_app``) so tests can create an
isolated instance and so a WSGI server (gunicorn) can import it cleanly.
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request
from flask_cors import cross_origin

from .features import FeatureError
from .model import load_model, predict_fare

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    @cross_origin()
    def home():
        return render_template("home.html")

    @app.route("/predict", methods=["GET", "POST"])
    @cross_origin()
    def predict():
        if request.method != "POST":
            return render_template("home.html")
        try:
            fare = predict_fare(request.form)
        except FeatureError as exc:
            logger.warning("bad prediction request: %s", exc)
            return render_template("home.html", prediction_text=f"Error: {exc}")
        return render_template(
            "home.html",
            prediction_text=f"The predicted Flight Fare is Rs.{fare}",
        )

    @app.route("/api/predict", methods=["POST"])
    @cross_origin()
    def api_predict():
        payload = request.get_json(silent=True) or {}
        try:
            fare = predict_fare(payload)
        except FeatureError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(fare=fare, currency="INR")

    @app.route("/health")
    def health():
        """Liveness: the process is running."""
        return jsonify(status="ok"), 200

    @app.route("/ready")
    def ready():
        """Readiness: the model is loaded and able to serve predictions."""
        try:
            load_model()
        except Exception as exc:  # noqa: BLE001 - report any load failure
            logger.error("model not ready: %s", exc)
            return jsonify(status="unavailable"), 503
        return jsonify(status="ready"), 200

    return app


# WSGI servers and `flask run` look for a module-level `app`.
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
