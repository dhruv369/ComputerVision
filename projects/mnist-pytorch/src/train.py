from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from src.data import create_dataloaders
from src.model import MNISTModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def select_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("The data loader is empty.")
    return total_loss / total, correct / total


def save_training_figures(metrics: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = metrics["epochs"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["train_losses"], label="Train")
    plt.plot(epochs, metrics["val_losses"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["train_accuracies"], label="Train")
    plt.plot(epochs, metrics["val_accuracies"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy.png", dpi=150)
    plt.close()


def train(
    config_path: Path,
    epochs_override: int | None = None,
    device_override: str | None = None,
    run_name: str | None = None,
) -> Path:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device(device_override or training["device"])
    epochs = int(training["epochs"]) if epochs_override is None else epochs_override
    if epochs < 1:
        raise ValueError("The number of epochs must be positive.")

    data_dir = PROJECT_ROOT / config["paths"]["data"]
    artifact_dir = PROJECT_ROOT / config["paths"]["artifacts"]
    results_dir = PROJECT_ROOT / config["paths"]["results"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{run_name or training['run_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        seed=seed,
        validation_fraction=float(training["validation_fraction"]),
    )
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_name = training["optimizer"].lower()
    learning_rate = float(training["learning_rate"])
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(training["weight_decay"]),
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(training["weight_decay"]),
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(training["momentum"]),
            weight_decay=float(training["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {training['optimizer']}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(training["scheduler_patience"])
    )
    patience = int(training["early_stopping_patience"])
    best_loss = float("inf")
    no_improvement = 0
    metrics = {
        "epochs": [],
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
        "learning_rates": [],
    }

    print(f"Training on {device}; saving outputs under {run_dir.relative_to(PROJECT_ROOT)}")
    with SummaryWriter(log_dir=str(run_dir / "tensorboard")) as writer:
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer
            )
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            metrics["epochs"].append(epoch)
            metrics["train_losses"].append(train_loss)
            metrics["train_accuracies"].append(train_accuracy)
            metrics["val_losses"].append(val_loss)
            metrics["val_accuracies"].append(val_accuracy)
            metrics["learning_rates"].append(optimizer.param_groups[0]["lr"])

            for key, value in (
                ("loss/train", train_loss),
                ("loss/validation", val_loss),
                ("accuracy/train", train_accuracy),
                ("accuracy/validation", val_accuracy),
            ):
                writer.add_scalar(key, value, epoch)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }
            torch.save(checkpoint, artifact_dir / "latest_model.pth")
            if val_loss < best_loss:
                best_loss = val_loss
                no_improvement = 0
                torch.save(checkpoint, artifact_dir / "best_model.pth")
            else:
                no_improvement += 1

            print(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_accuracy={train_accuracy:.4f} val_accuracy={val_accuracy:.4f}"
            )
            if no_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_training_figures(metrics, PROJECT_ROOT / config["paths"]["figures"])
    return artifact_dir / "best_model.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MNIST CNN.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name")
    args = parser.parse_args()
    checkpoint = train(args.config, args.epochs, args.device, args.run_name)
    print(f"Best checkpoint saved to {checkpoint}")


if __name__ == "__main__":
    main()
