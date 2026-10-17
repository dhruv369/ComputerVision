import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import MNISTModel
from src.train import run_epoch


class TrainingLoopTests(unittest.TestCase):
    def test_training_epoch_returns_loss_and_accuracy(self):
        images = torch.randn(2, 1, 28, 28)
        labels = torch.tensor([1, 7])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = MNISTModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss, accuracy = run_epoch(
            model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), optimizer
        )

        self.assertGreater(loss, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_empty_loader_is_reported(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 1, 28, 28), torch.empty(0)), batch_size=2)
        with self.assertRaisesRegex(ValueError, "data loader is empty"):
            run_epoch(
                MNISTModel(),
                loader,
                torch.nn.CrossEntropyLoss(),
                torch.device("cpu"),
            )
