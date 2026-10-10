"""Train the flight-fare RandomForest model and serialise it.

Run:  python -m scripts.train  (or  make train)

Reads the raw Kaggle-style training data from ``data/Data_Train.xlsx``,
reproduces the feature engineering defined in ``app/features.py`` (so the
training matrix matches serving exactly), fits a RandomForestRegressor, prints
hold-out metrics, and writes ``model/flight_fare_rf.pkl``.

The model is pinned to the scikit-learn version in requirements.txt; retrain
whenever that version changes so the artifact and the runtime never disagree.
"""
from __future__ import annotations

import argparse
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from app.features import FEATURE_ORDER

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DATA = os.path.join(ROOT, "data", "Data_Train.xlsx")
DEFAULT_OUT = os.path.join(ROOT, "model", "flight_fare_rf.pkl")

_STOPS_MAP = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}


def _parse_duration(value: str) -> tuple[int, int]:
    """'2h 50m' -> (2, 50); '19h' -> (19, 0); '5m' -> (0, 5)."""
    hours = mins = 0
    for part in str(value).strip().split():
        if part.endswith("h"):
            hours = int(part[:-1])
        elif part.endswith("m"):
            mins = int(part[:-1])
    return hours, mins


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> model-ready matrix with columns == FEATURE_ORDER."""
    df = df.dropna().copy()

    journey = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_day"] = journey.dt.day
    df["Journey_month"] = journey.dt.month

    dep = pd.to_datetime(df["Dep_Time"])
    df["Dep_hour"] = dep.dt.hour
    df["Dep_min"] = dep.dt.minute

    arr = pd.to_datetime(df["Arrival_Time"])
    df["Arrival_hour"] = arr.dt.hour
    df["Arrival_min"] = arr.dt.minute

    dur = df["Duration"].apply(_parse_duration)
    df["Duration_hours"] = [h for h, _ in dur]
    df["Duration_mins"] = [m for _, m in dur]

    df["Total_Stops"] = df["Total_Stops"].map(_STOPS_MAP).astype(int)

    # One-hot the categoricals, then force the full, correctly ordered column
    # set from the shared spec. reindex(fill_value=0) adds any category missing
    # from this split and drops the dropped-baseline columns, guaranteeing the
    # matrix lines up with FEATURE_ORDER regardless of what the data contains.
    cats = pd.get_dummies(
        df[["Airline", "Source", "Destination"]].astype("object"),
        prefix=["Airline", "Source", "Destination"],
        prefix_sep="_",
    )
    numeric = df[
        [
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
    ]
    matrix = pd.concat([numeric, cats], axis=1)
    matrix = matrix.reindex(columns=FEATURE_ORDER, fill_value=0).astype(int)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the flight-fare model")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    raw = pd.read_excel(args.data)
    X = engineer(raw)
    y = raw.dropna()["Price"].astype(float).to_numpy()

    assert list(X.columns) == FEATURE_ORDER, "feature order drift!"
    assert X.shape[1] == len(FEATURE_ORDER) == 29

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y, test_size=0.2, random_state=args.random_state
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"features      : {X.shape[1]}")
    print(f"train / test  : {len(X_train)} / {len(X_test)}")
    print(f"R^2 (test)    : {r2_score(y_test, preds):.4f}")
    print(f"MAE (test)    : {mean_absolute_error(y_test, preds):.1f} INR")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(model, fh)
    print(f"saved model -> {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
