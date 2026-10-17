# MNIST Digit Classification with PyTorch

A reproducible end-to-end image-classification project for recognizing handwritten digits. It includes a configurable CNN training pipeline, a held-out validation split, final test-set evaluation, single-image prediction, training visualizations, and automated tests.

## Highlights

- CNN with three convolutional blocks, batch normalization, dropout, and a 10-class output.
- Reproducible train/validation split from the official MNIST training set; the official test set is reserved for final evaluation.
- Configurable optimizer, learning rate, batch size, early stopping, and device selection.
- Checkpointing, timestamped metrics, TensorBoard logs, and generated loss/accuracy plots.
- Separate commands for training, evaluation, and image prediction.

## Project layout

```text
mnist-pytorch/
├── artifacts/                 # Local model checkpoints (not committed)
├── configs/config.yaml        # Training and output configuration
├── notebooks/                 # Dataset exploration notebook
├── results/
│   ├── figures/               # Latest generated training curves
│   └── runs/                  # Timestamped metrics and TensorBoard logs
├── src/
│   ├── data.py
│   ├── dashboard.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_train.py
├── requirements.txt
└── README.md
```

## Quick start

Run commands from this directory:

```bash
python -m pip install -r requirements.txt
python -m src.train
```

The first run downloads MNIST to `data/`. By default, training automatically selects CUDA, Apple Silicon MPS, or CPU. Adjust hyperparameters and paths in [`configs/config.yaml`](./configs/config.yaml).

### Evaluate a checkpoint

Evaluation uses the official MNIST test set, which is not used to select checkpoints:

```bash
python -m src.evaluate --checkpoint artifacts/best_model.pth
```

### Predict a digit from an image

The input image is converted to grayscale, resized to 28 × 28, and normalized using the MNIST training-set statistics:

```bash
python -m src.predict path/to/handwritten_digit.png
```

### Explore results

```bash
streamlit run src/dashboard.py
```

Training writes timestamped metrics and TensorBoard logs to `results/runs/`, saves the best and latest checkpoints to `artifacts/`, and updates `results/figures/`. Generated model files and run logs are excluded from version control. See [`results/README.md`](./results/README.md) before publishing metrics or figures.

## Tests and CI

Run the CPU-friendly unit tests without downloading MNIST:

```bash
python -m unittest discover -s tests -v
```

The manually triggered GitHub Actions workflow runs a one-epoch CPU training smoke test, runs the unit tests, and uploads the best model checkpoint.

## Data exploration

Open [`notebooks/mnist_exploration.ipynb`](./notebooks/mnist_exploration.ipynb) for a short walkthrough of example digits, class distribution, and normalized input tensors.
