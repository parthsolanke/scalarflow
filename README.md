# ScalarFlow

![nn_vizorg](vizorg.png)

This project is a simple neural network library implemented from scratch in Python. It provides a framework for building and training neural networks using a custom auto-gradient engine. The library includes classes for defining neural network layers, neurons, network mode and for visualization of networks and their gradients. Supports common activation functions like sigmoid, tanh, and relu.

## Features

- Implementation of a custom auto-gradient engine for automatic differentiation.
- Definition of neural network layers with customizable activation functions.
- Support for common activation functions: sigmoid, tanh, and relu.
- Training loop using gradient descent optimization algorithm.
- Visualizations of computation graphs using Graphviz.

## Dependencies
The neural network library has the following dependencies:

- Python (>=3.6)
- Graphviz (for visualization)

## Usage

- Import the necessary modules and classes.
- Define your mathematical expressions using the value class from the autograd engine.
- Create your neural network model by composing layers and neurons from the neural network library.
- Train your neural network using the provided methods for different loss functions.

Autograd Engine
```python
# Example usage of the autograd engine
from scalarflow.engine import value

# Create value objects
a = value(2)
b = value(3)

# Perform arithmetic operations
c = a + b
d = a * b

# Compute gradients
d.backward()

# Print results
print("c =", c)  # Output: c = value(data = 5, grad = 1)
print("d =", d)  # Output: d = value(data = 6, grad = 2)
```

Neural Network Library
```python
# Example usage of the neural network library
from scalarflow.nn import network

# Create a neural network model
model = network(num_in=2, layer_sizes=[4, 2, 1], unit_activation='relu')

# Print the network architecture summary
model.summary()

# Train the model
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

model.fit(inputs, outputs, epochs=100)

# Make predictions
predictions = model.predict(inputs)
print(predictions)
```

## HTTP API

A small Go HTTP server in `server/` exposes the library over JSON. It calls the
Python package through `server/bridge.py`, so the Python dependencies are still
required.

```bash
cd server
go run .
```

The server listens on `:8080` by default. Override it with `SCALARFLOW_ADDR`
(for example `SCALARFLOW_ADDR=:9090`), the interpreter with `SCALARFLOW_PYTHON`,
and the bridge location with `SCALARFLOW_BRIDGE`.

Endpoints:

- `GET /health`
- `POST /v1/train` — train a new network or continue training a supplied model.
- `POST /v1/predict` — run inference for a supplied model.
- `POST /v1/summary` — return parameter counts for a supplied model.

Training example:

```bash
curl -s http://localhost:8080/v1/train \
  -H 'Content-Type: application/json' \
  -d '{
    "num_in": 2,
    "layer_sizes": [4, 2, 1],
    "activation": "relu",
    "inputs": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "outputs": [0, 1, 1, 0],
    "epochs": 100
  }'
```

The response contains the trained `model` (weights and biases), its
`predictions`, and the final `loss`. Pass that `model` object back to
`/v1/predict`, `/v1/summary`, or `/v1/train` to reuse or keep training it.

Prediction example:

```bash
curl -s http://localhost:8080/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"model": { ... }, "inputs": [[0, 0], [1, 1]]}'
```

