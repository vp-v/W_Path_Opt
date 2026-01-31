import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.pipeline import run_end_to_end


st.set_page_config(page_title="Warehouse Pick Optimisation", layout="wide")
st.title("Dynamic Pick Path Optimisation + ML Picker Assignment")

st.write(
    "This app simulates warehouse routing and recommends the best picker per order "
    "using a RandomForest model trained on synthetic + routing-derived data."
)

layout_csv = st.text_input("Warehouse layout CSV", "data/warehouse_layout.csv")
orders_csv = st.text_input("Orders CSV", "data/orders.csv")
pickers_csv = st.text_input("Pickers CSV", "data/pickers.csv")

colA, colB = st.columns(2)
with colA:
    start_node = st.text_input("Start node", "START")
with colB:
    end_node = st.text_input("End node", "PACK")

if st.button("Run optimisation"):
    try:
        out = run_end_to_end(
            layout_csv=layout_csv,
            orders_csv=orders_csv,
            pickers_csv=pickers_csv,
            start_node=start_node,
            end_node=end_node
        )

        st.success(f"Run complete. ML training rows: {out['ml_training_rows']}")

        st.subheader("Routing Results (Static vs Dynamic)")
        st.dataframe(out["routing_results"], use_container_width=True)

        st.subheader("Picker Recommendations")
        rec_rows = []
        for rec in out["picker_recommendations"]:
            # show best picker + a compact view of all predicted times
            scores = rec["predicted_times"]
            rec_rows.append({
                "order_id": rec["order_id"],
                "best_picker": rec["best_picker"],
                "all_predicted_times": ", ".join([f"{k}:{v:.2f}" for k, v in scores.items()])
            })

        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")