import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data import PROJECT_ROOT, build_transform
from src.model import load_model
from src.train import select_device


def predict(image_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[int, float]:
    with Image.open(image_path) as source:
        image = source.convert("L")
    tensor = build_transform()(image).unsqueeze(0).to(device)
    model = load_model(str(checkpoint_path), device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, label = probabilities.max(dim=0)
    return int(label.item()), float(confidence.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the digit in a handwritten image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/best_model.pth")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    label, confidence = predict(args.image, args.checkpoint, select_device(args.device))
    print(f"Predicted digit: {label} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
