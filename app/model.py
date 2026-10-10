"""Model loading and prediction.

The trained estimator is loaded once at process start and reused for every
request. ``predict_fare`` is the only function the web layer needs.
"""
from __future__ import annotations

import logging
import os
import pickle
from functools import lru_cache

import numpy as np

from .features import build_feature_vector

logger = logging.getLogger(__name__)

# Allow the model path to be overridden (handy in tests and in the container,
# where the model lives at /app/model/flight_fare_rf.pkl).
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "flight_fare_rf.pkl"
)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    """Load and cache the pickled estimator."""
    path = path or MODEL_PATH
    logger.info("loading model from %s", path)
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    logger.info("model loaded: %s", type(model).__name__)
    return model


def predict_fare(payload: dict, model=None) -> float:
    """Predict a fare (in INR) for a single raw payload."""
    model = model or load_model()
    row = build_feature_vector(payload)
    prediction = model.predict(np.array([row]))
    return round(float(prediction[0]), 2)
