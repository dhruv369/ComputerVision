import unittest

from PIL import Image

from src.data import build_transform, split_indices


class MNISTDataTests(unittest.TestCase):
    def test_transform_returns_normalized_mnist_tensor(self):
        image = Image.new("L", (28, 28), color=0)
        tensor = build_transform()(image)
        self.assertEqual(tuple(tensor.shape), (1, 28, 28))
        self.assertLess(float(tensor.mean()), 0)

    def test_augmentation_preserves_image_shape(self):
        image = Image.new("L", (32, 24), color=128)
        tensor = build_transform(augment=True)(image)
        self.assertEqual(tuple(tensor.shape), (1, 28, 28))

    def test_train_and_validation_indices_are_disjoint_and_reproducible(self):
        train_indices, validation_indices = split_indices(100, seed=7)
        self.assertEqual(len(train_indices), 90)
        self.assertEqual(len(validation_indices), 10)
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(
            (train_indices, validation_indices), split_indices(100, seed=7)
        )

    def test_invalid_validation_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            split_indices(10, validation_fraction=1.0)
