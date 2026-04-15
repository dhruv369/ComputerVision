import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

st.title("MNIST Model Training Dashboard")

# Path to model weights
weights_dir = "mnist/model_weights"

# Get list of metrics files
metrics_files = [f for f in os.listdir(weights_dir) if f.endswith("_metrics.json")]

if not metrics_files:
    st.error("No metrics files found. Please run training first.")
else:
    # Select models to compare
    selected_models = st.multiselect("Select models to compare", [f.replace("_metrics.json", "") for f in metrics_files], default=[f.replace("_metrics.json", "") for f in metrics_files][:1])

    if selected_models:
        fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
        fig_acc, ax_acc = plt.subplots(figsize=(10, 5))

        for model in selected_models:
            metrics_path = os.path.join(weights_dir, f"{model}_metrics.json")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            epochs = metrics["epochs"]
            train_losses = metrics["train_losses"]
            val_losses = metrics["val_losses"]
            val_accuracies = metrics["val_accuracies"]

            ax_loss.plot(epochs, train_losses, label=f"{model} Train Loss")
            ax_loss.plot(epochs, val_losses, label=f"{model} Val Loss")
            ax_acc.plot(epochs, val_accuracies, label=f"{model} Val Accuracy")

        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training and Validation Loss")
        ax_loss.legend()

        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Validation Accuracy")
        ax_acc.legend()

        st.pyplot(fig_loss)
        st.pyplot(fig_acc)

        # Display metrics table
        st.subheader("Metrics Summary")
        summary_data = []
        for model in selected_models:
            metrics_path = os.path.join(weights_dir, f"{model}_metrics.json")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            final_train_loss = metrics["train_losses"][-1]
            final_val_loss = metrics["val_losses"][-1]
            final_val_acc = metrics["val_accuracies"][-1]
            summary_data.append({
                "Model": model,
                "Final Train Loss": f"{final_train_loss:.4f}",
                "Final Val Loss": f"{final_val_loss:.4f}",
                "Final Val Accuracy": f"{final_val_acc:.4f}"
            })
        st.table(pd.DataFrame(summary_data))