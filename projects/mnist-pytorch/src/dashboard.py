import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "results" / "runs"

st.set_page_config(page_title="MNIST Training Results", layout="wide")
st.title("MNIST CNN Training Results")

metrics_paths = sorted(RUNS_DIR.glob("*/metrics.json"))
if not metrics_paths:
    st.info("No training results yet. Run `python -m src.train` from the project directory.")
else:
    selected_paths = st.multiselect(
        "Runs to compare",
        metrics_paths,
        default=metrics_paths[-1:],
        format_func=lambda path: path.parent.name,
    )
    summaries = []
    for metrics_path in selected_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_name = metrics_path.parent.name
        summaries.append(
            {
                "Run": run_name,
                "Epochs": len(metrics["epochs"]),
                "Final validation accuracy": f"{metrics['val_accuracies'][-1]:.2%}",
                "Best validation accuracy": f"{max(metrics['val_accuracies']):.2%}",
            }
        )
        st.subheader(run_name)
        left, right = st.columns(2)
        with left:
            figure, axis = plt.subplots()
            axis.plot(metrics["epochs"], metrics["train_losses"], label="Train")
            axis.plot(metrics["epochs"], metrics["val_losses"], label="Validation")
            axis.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
            axis.legend()
            st.pyplot(figure)
            plt.close(figure)
        with right:
            figure, axis = plt.subplots()
            axis.plot(metrics["epochs"], metrics["train_accuracies"], label="Train")
            axis.plot(metrics["epochs"], metrics["val_accuracies"], label="Validation")
            axis.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
            axis.legend()
            st.pyplot(figure)
            plt.close(figure)
    if summaries:
        st.subheader("Run summary")
        st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)
