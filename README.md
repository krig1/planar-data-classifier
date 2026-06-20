# Planar Data Neural Network

A shallow neural network built from scratch in NumPy that classifies non-linearly separable 2D data.
Implements the full training pipeline manually, including forward propagation, cost computation,
backpropagation, and gradient descent.

## Architecture

- **Input layer:** 2 features (x1, x2)
- **Hidden layer:** 4 units with tanh activation
- **Output layer:** 1 unit with sigmoid activation
- **Loss:** Binary cross-entropy
- **Optimizer:** Gradient descent (lr=1.2)

## How It Works

- Weights initialized randomly (scaled by 0.01), biases initialized to zero
- Forward pass: linear transform → tanh → linear transform → sigmoid
- Cost computed using binary cross-entropy over all examples
- Backward pass: gradients derived analytically using the chain rule
- Parameters updated for 10,000 iterations

## Results

- Training accuracy: ~90%
- The decision boundary plot shows the network learning a non-linear boundary
to separate the flower-shaped dataset

## Dataset

Planar "flower" dataset from Andrew Ng's Deep Learning Specialization (`planar_utils.py`).
A 2D binary classification dataset that is not linearly separable.

## Dependencies

- numpy
- matplotlib
- planar_utils 
