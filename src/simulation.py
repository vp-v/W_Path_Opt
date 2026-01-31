from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Picker:
    picker_id: str
    avg_speed: float           # higher = faster
    error_rate: float          # 0.0 - 1.0
    experience_years: float
    current_zone: str


def pick_time_estimate(distance: float, avg_speed: float, error_rate: float) -> float:
    """
    Simple time model:
    time = distance / speed, with a penalty for error_rate (rework).
    """
    base = distance / max(avg_speed, 1e-6)
    penalty = base * error_rate
    return base + penalty


def build_training_data(
    orders_df: pd.DataFrame,
    pickers_df: pd.DataFrame,
    n_synthetic_orders: int = 200,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create a training dataset of (order_features + picker_features) -> completion_time.

    For personal projects, we scale dataset size by synthetically sampling orders
    so we can evaluate ML properly.
    """
    rng = np.random.default_rng(seed)

    # Expand orders
    orders = orders_df.copy()
    if "order_zone" not in orders.columns:
        orders["order_zone"] = rng.choice(["A", "B"], size=len(orders))

    if "order_distance" not in orders.columns:
        orders["order_distance"] = rng.integers(40, 120, size=len(orders))

    # Synthetic expansion
    if len(orders) < n_synthetic_orders:
        idx = rng.integers(0, len(orders), size=n_synthetic_orders)
        orders = orders.iloc[idx].reset_index(drop=True)
        # Re-sample distances to create diversity
        orders["order_distance"] = rng.integers(40, 120, size=len(orders))
        orders["order_zone"] = rng.choice(["A", "B"], size=len(orders))

    rows: List[Dict] = []
    for _, order in orders.iterrows():
        for _, p in pickers_df.iterrows():
            zone_match = int(str(order["order_zone"]) == str(p["current_zone"]))
            completion_time = pick_time_estimate(
                float(order["order_distance"]),
                float(p["avg_speed"]),
                float(p["error_rate"])
            )
            rows.append({
                "order_distance": float(order["order_distance"]),
                "order_zone": str(order["order_zone"]),
                "avg_speed": float(p["avg_speed"]),
                "error_rate": float(p["error_rate"]),
                "experience_years": float(p["experience_years"]),
                "zone_match": zone_match,
                "completion_time": completion_time
            })

    df = pd.DataFrame(rows)

    # Convert order_zone to a simple numeric encoding for the model (keep it transparent)
    df["order_zone_A"] = (df["order_zone"] == "A").astype(int)
    df["order_zone_B"] = (df["order_zone"] == "B").astype(int)
    df = df.drop(columns=["order_zone"])

    return df