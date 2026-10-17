import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import PROJECT_ROOT, load_dataset
from src.model import load_model
from src.train import run_epoch, select_device


def evaluate(checkpoint_path: Path, data_dir: Path, batch_size: int, device: torch.device) -> dict:
    dataset = load_dataset(train=False, data_dir=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = load_model(str(checkpoint_path), device)
    loss, accuracy = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    return {"loss": loss, "accuracy": accuracy, "examples": len(dataset)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST model.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/best_model.pth")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    metrics = evaluate(args.checkpoint, args.data_dir, args.batch_size, device)
    print(
        f"Test loss: {metrics['loss']:.4f} | "
        f"Test accuracy: {metrics['accuracy']:.2%} | "
        f"Examples: {metrics['examples']}"
    )


if __name__ == "__main__":
    main()
