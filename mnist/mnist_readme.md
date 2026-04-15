# MNIST Handwritten Digit Recognition

This repository contains a PyTorch implementation for training a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

## Project Structure

- `mnist_datasets.py`: Defines the MNIST dataset class with data loading and preprocessing.
- `mnist_model.py`: Defines the CNN model architecture for MNIST classification.
- `mnist_train.py`: Contains the training loop, early stopping, and metric tracking.
- `mnist_dashboard.py`: Streamlit dashboard for visualizing training metrics and comparing models.
- `requirements.txt`: List of Python dependencies.
- `model_weights/`: Directory containing saved model checkpoints and training metrics.

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

## Model Architecture

The CNN model consists of:
- 2 convolutional layers with ReLU activation and max pooling
- 2 fully connected layers
- Output layer with 10 classes (digits 0-9)

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
