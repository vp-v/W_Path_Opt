from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLUMNS = [
    "order_distance",
    "avg_speed",
    "error_rate",
    "experience_years",
    "zone_match",
    "order_zone_A",
    "order_zone_B",
]


def train_picker_model(training_df: pd.DataFrame) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor to predict completion_time.
    """
    missing = set(FEATURE_COLUMNS + ["completion_time"]) - set(training_df.columns)
    if missing:
        raise ValueError(f"Training data missing columns: {missing}")

    X = training_df[FEATURE_COLUMNS]
    y = training_df["completion_time"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)
    return model


def _encode_order_zone(order_zone: str) -> Dict[str, int]:
    order_zone = str(order_zone).upper()
    return {
        "order_zone_A": 1 if order_zone == "A" else 0,
        "order_zone_B": 1 if order_zone == "B" else 0,
    }


def assign_best_picker(
    model: RandomForestRegressor,
    order_distance: float,
    order_zone: str,
    pickers_df: pd.DataFrame
) -> Tuple[str, Dict[str, float]]:
    """
    Predict completion time for each picker for a given order and choose the best.
    Returns (best_picker_id, predictions_dict).
    """
    zone_enc = _encode_order_zone(order_zone)
    predictions: Dict[str, float] = {}

    for _, p in pickers_df.iterrows():
        row = {
            "order_distance": float(order_distance),
            "avg_speed": float(p["avg_speed"]),
            "error_rate": float(p["error_rate"]),
            "experience_years": float(p["experience_years"]),
            "zone_match": int(str(order_zone).upper() == str(p["current_zone"]).upper()),
            **zone_enc
        }
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        predictions[str(p["picker_id"])] = float(model.predict(X)[0])

    best_picker_id = min(predictions, key=predictions.get)
    return best_picker_id, predictions