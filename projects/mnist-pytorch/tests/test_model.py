import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.model import MNISTModel, count_parameters, load_model
from src.predict import predict


class MNISTModelTests(unittest.TestCase):
    def test_model_returns_ten_class_logits(self):
        model = MNISTModel().eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 1, 28, 28))
        self.assertEqual(output.shape, (2, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_model_has_fewer_than_one_hundred_thousand_parameters(self):
        self.assertLess(count_parameters(MNISTModel()), 100_000)

    def test_checkpoint_loads_for_inference(self):
        model = MNISTModel().eval()
        sample = torch.randn(1, 1, 28, 28)
        with torch.inference_mode():
            expected = model(sample)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pth"
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            loaded_model = load_model(str(checkpoint_path), torch.device("cpu"))
        with torch.inference_mode():
            actual = loaded_model(sample)
        torch.testing.assert_close(actual, expected)

    def test_predict_returns_digit_and_confidence(self):
        model = MNISTModel().eval()
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "digit.png"
            checkpoint_path = Path(directory) / "model.pth"
            Image.new("L", (28, 28), color=0).save(image_path)
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

            label, confidence = predict(
                image_path, checkpoint_path, torch.device("cpu")
            )

        self.assertIn(label, range(10))
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
