# MNIST Handwritten Digit Recognition

This repository contains a PyTorch implementation for training a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

## Project Structure

- `mnist_datasets.py`: Defines the MNIST dataset class with data loading and preprocessing.
- `mnist_model.py`: Defines the CNN model architecture for MNIST classification.
- `mnist_train.py`: Contains the training loop, early stopping, and metric tracking.
- `mnist_dashboard.py`: Streamlit dashboard for visualizing training metrics and comparing models.
- `train_ci.py`: CI/CD training script that trains for 1 epoch and saves model with timestamp.
- `test_model.py`: Testing script that validates model parameters, input/output shapes, and accuracy.
- `requirements.txt`: List of Python dependencies.
- `model_weights/`: Directory containing saved model checkpoints and training metrics.
- `.github/workflows/ci-cd.yml`: GitHub Actions workflow for CI/CD pipeline.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ComputerVision/mnist
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script to train the model:

```bash
python mnist_train.py
```

This will:
- Load the MNIST dataset
- Train the CNN model
- Save the best model checkpoint to `model_weights/mnist_model_checkpoint.pth`
- Save training metrics to `model_weights/mnist_model_metrics.json`

You can modify hyperparameters in the `main()` function of `mnist_train.py`, such as:
- `epochs`: Number of training epochs
- `learn_rate`: Learning rate for the optimizer
- `model_name`: Name for the model run (for multiple experiments)

### Viewing Training Dashboard

Launch the Streamlit dashboard to visualize training metrics:

```bash
streamlit run mnist_dashboard.py
```

The dashboard allows you to:
- Select one or multiple trained models to compare
- View plots of training/validation loss and validation accuracy over epochs
- See a summary table of final metrics for each model

## CI/CD Pipeline

### Local Testing

Before pushing to GitHub, you can run the CI/CD pipeline locally:

1. Train the model:
   ```bash
   python train_ci.py
   ```
   This trains the model for 1 epoch and saves it with a timestamp suffix (e.g., `mnist_model_20260426_231352.pth`).

2. Run tests:
   ```bash
   python test_model.py
   ```
   This script checks:
   - Model has fewer than 100,000 parameters
   - Model accepts 28x28 input without issues
   - Model has exactly 10 outputs
   - Model achieves >80% accuracy on the test set

### GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/ci-cd.yml`) that:
- Installs Python and dependencies
- Trains the model for 1 epoch
- Runs all validation tests
- Uploads the trained model as an artifact if tests pass

To use:
1. Push your code to the `main` branch on GitHub
2. The workflow will automatically run
3. Check the Actions tab for results
4. Download the model artifact from successful runs

## Model Architecture

The CNN model consists of:
- 2 convolutional layers with ReLU activation and max pooling
- 2 fully connected layers
- Output layer with 10 classes (digits 0-9)
- Total parameters: ~56,714 (under 100,000)

## Dataset

The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits (28x28 grayscale).

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- Streamlit
- Matplotlib
- Pandas
- tqdm

## Contributing

Feel free to experiment with different architectures, hyperparameters, or add features like data augmentation or different optimizers. Save metrics with unique model names to compare different configurations in the dashboard.
