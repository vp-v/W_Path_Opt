from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import networkx as nx

from .graph_builder import build_graph
from .path_optimisation import static_pick_path, dynamic_pick_path_greedy
from .metrics import route_cost, pct_improvement
from .simulation import build_training_data
from .picker_assignment import train_picker_model, assign_best_picker


def parse_locations(locations_str: str) -> List[str]:
    return [x.strip() for x in str(locations_str).split(",") if x.strip()]


def run_end_to_end(
    layout_csv: str,
    orders_csv: str,
    pickers_csv: str,
    start_node: str = "START",
    end_node: str = "PACK"
) -> Dict:
    """
    Loads data, builds graph, runs routing + trains ML + recommends pickers.
    Returns a dict of results for easy display in Streamlit.
    """
    G = build_graph(layout_csv)
    orders = pd.read_csv(orders_csv)
    pickers = pd.read_csv(pickers_csv)

    # If you want routing distance to feed ML, you can compute it per order here.
    # For simplicity we keep a separate synthetic training generator too.
    results = []

    for _, order in orders.iterrows():
        order_id = str(order["order_id"])
        locs = parse_locations(order["locations"])
        order_zone = order.get("order_zone", "A")

        static_route = static_pick_path(G, start_node, locs, end=end_node)
        dynamic_route = dynamic_pick_path_greedy(G, start_node, locs, end=end_node)

        static_cost = route_cost(G, static_route)
        dynamic_cost = route_cost(G, dynamic_route)

        results.append({
            "order_id": order_id,
            "order_zone": order_zone,
            "static_cost": static_cost,
            "dynamic_cost": dynamic_cost,
            "improvement_pct": pct_improvement(static_cost, dynamic_cost),
        })

    results_df = pd.DataFrame(results)

    # Train ML model (synthetic expansion for robust evaluation)
    # Here we use dynamic distance as the "order_distance" feature for realism.
    orders_for_ml = orders.copy()
    orders_for_ml["order_distance"] = results_df["dynamic_cost"].values
    if "order_zone" not in orders_for_ml.columns:
        orders_for_ml["order_zone"] = "A"

    training_df = build_training_data(orders_for_ml, pickers, n_synthetic_orders=200, seed=42)
    model = train_picker_model(training_df)

    # Recommend pickers for each order based on its dynamic distance + zone
    picker_recs = []
    for _, row in results_df.iterrows():
        best_picker, scores = assign_best_picker(
            model=model,
            order_distance=float(row["dynamic_cost"]),
            order_zone=str(row["order_zone"]),
            pickers_df=pickers
        )
        picker_recs.append({
            "order_id": row["order_id"],
            "best_picker": best_picker,
            "predicted_times": scores
        })

    return {
        "graph": G,
        "orders": orders,
        "pickers": pickers,
        "routing_results": results_df,
        "picker_recommendations": picker_recs,
        "ml_training_rows": len(training_df)
    }