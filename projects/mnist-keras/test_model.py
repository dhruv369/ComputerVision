import torch
import torch.nn.functional as F
from mnist_datasets import MNISTDataset
from mnist_model import MNISTModel
import os
import glob

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model(model_path):
    # Load model
    model = MNISTModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Test 1: Parameter count < 100,000
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count}")
    assert param_count < 100000, f"Model has {param_count} parameters, should be < 100,000"

    # Test 2: Input shape 28x28
    dummy_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(dummy_input)
        print("Model accepts 28x28 input successfully")
    except Exception as e:
        raise AssertionError(f"Model failed on 28x28 input: {e}")

    # Test 3: Output shape is 10
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    print("Model has 10 outputs")

    # Test 4: Accuracy > 80% on test set
    test_dataset = MNISTDataset(train=False)
    test_loader = test_dataset.test_dataloader(batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(".4f")
    assert accuracy > 0.98, ".4f"

    print("All tests passed!")

if __name__ == "__main__":
    # Find the latest model file
    model_files = glob.glob("mnist/model_weights/mnist_model_*.pth")
    if not model_files:
        raise FileNotFoundError("No model files found")
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Testing model: {latest_model}")
    test_model(latest_model)