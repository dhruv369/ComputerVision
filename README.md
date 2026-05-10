# MNIST Hand Digit Recognition and Fashion Classification

This repository is a learning project for training convolutional neural networks (CNNs) on the MNIST digit recognition and Fashion-MNIST classification tasks. The goal is to experiment with different CNN concepts and training settings while building a complete training pipeline.

## Project Overview

- **Datasets**: MNIST (handwritten digits) and Fashion-MNIST (clothing items)
- **Tasks**: image classification using CNNs
- **Focus**: model design, optimizer choices, regularization, augmentation, and training visualization
- **Tools**: deep learning framework, TensorBoard for tracking training loss and accuracy

## Experiments and Model Variants

This project includes experiments with several main CNN configurations:

- `baseline_cnn_adam_lr1e-3`
- `baseline_cnn_adam_lr3e-4`
- `baseline_cnn_adamw_lr1e-3`
- `baseline_cnn_sgd_lr1e-2`
- `cnn_bn_on`
- `cnn_bn_off`
- `cnn_dropout_0.2`
- `cnn_weight_decay_1e-4`
- `cnn_aug_on`
- `cnn_aug_off`

Each experiment is designed to test a single concept or training choice, such as learning rate, optimizer type, batch normalization, dropout, weight decay, and data augmentation.

## Training Pipeline

The training pipeline typically includes:

1. loading and preprocessing the dataset
2. defining the CNN model architecture
3. choosing the optimizer and learning rate
4. applying regularization or augmentation where needed
5. training the model and validating on a held-out set
6. logging metrics for visualization

## Visualization with TensorBoard

Use TensorBoard to visualize training progress:

- training loss
- training accuracy
- validation loss
- validation accuracy

This helps compare model variants and understand the effect of each modification.

## Learning Goals

This README is intended for a learning project, so the focus is on:

- understanding how optimizer choice affects training
- observing the impact of learning rate changes
- comparing weight decay and dropout
- seeing the benefit of batch normalization
- exploring data augmentation
- tracking metrics with TensorBoard

## Notes

- Keep experiments simple and readable.
- Train the full pipeline for both MNIST and Fashion-MNIST.
- Use consistent naming for experiment versions to track results.

## Getting Started

1. Prepare the environment and install dependencies.
2. Run the training scripts for the desired experiment.
3. Start TensorBoard and inspect the logged metrics.
4. Compare model variants and update the README with results.
