# ScalarFlow

ScalarFlow is a small neural network library implemented from scratch in Python. It provides a custom auto-gradient (`autograd`) engine and a set of neural network building blocks, so you can build and train networks without a deep learning framework.

## Features

- A `value` class in `scalarflow/engine.py` that wraps a scalar, overloads arithmetic operators, and records the computation graph used for automatic differentiation.
- Backpropagation via a topological sort of the graph (`value.backward()`).
- Activation functions on `value`: `sigmoid`, `tanh` and `relu`, plus `exp` and power operations.
- Neural network building blocks in `scalarflow/nn.py`: `module`, `neuron`, `layer` and `network`.
- Loss functions: an SVM (max-margin) loss and an MSE loss.
- Training via gradient descent with `network.fit(...)`, plus `network.predict(...)` and `network.summary()`.
- Optional computation-graph visualization with Graphviz via `value.draw_graph()`.

## Dependencies

- Python (>=3.6)
- `numpy`
- Graphviz (Python package and system binary, needed for graph visualization)
- `fastapi` and `uvicorn` (needed to run the HTTP server)

Install them with:

```bash
pip install -r requirements.txt
```

## Usage

Import the engine and network components as needed.

### Autograd engine

```python
from scalarflow.engine import value

a = value(2)
b = value(3)

c = a + b
d = a * b

d.backward()

print("c =", c)  # value(data = 5, grad = 0)
print("d =", d)  # value(data = 6, grad = 1)
```

### Neural network library

```python
from scalarflow.nn import network

# num_in = number of inputs, layer_sizes = units per layer
model = network(num_in=2, layer_sizes=[4, 2, 1], unit_activation='relu')
model.summary()

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

model.fit(inputs, outputs, epochs=100)

predictions = model.predict(inputs)
print(predictions)
```

`unit_activation` may be `'sigmoid'`, `'tanh'` or `'relu'`; when omitted, neurons default to `tanh`.

### Visualizing the graph

```python
from scalarflow.engine import value

x = value(2, label='x')
y = value(3, label='y')
z = x * y + x
z.backward()
z.draw_graph()  # returns a Graphviz Digraph
```

### Running the FastAPI server

`scalarflow/server.py` exposes a network over HTTP. The model is configured with
environment variables: `SCALARFLOW_NUM_IN` (default `2`),
`SCALARFLOW_LAYER_SIZES` (default `4,2,1`) and `SCALARFLOW_ACTIVATION` (default
`''`, which falls back to `tanh`).

```bash
uvicorn scalarflow.server:app --reload
```

Endpoints:

- `GET /health` — liveness check.
- `GET /summary` — architecture summary of the loaded network.
- `POST /predict` — body `{"inputs": [[0, 0], [1, 1]]}`, returns `predictions`.
- `POST /train` — body `{"inputs": [[0, 0]], "outputs": [0], "epochs": 100}`, trains in place and returns the training log.

### Running the tests

The test suite uses `pytest` and covers the autograd engine, the network
components and the HTTP endpoints (via `fastapi.testclient`).

```bash
pip install -r requirements.txt
pytest
```

## Project structure

```
scalarflow/__init__.py  # exposes the engine and nn modules
scalarflow/engine.py    # autograd value class and graph visualization
scalarflow/nn.py        # module, neuron, layer, network and losses
scalarflow/server.py    # FastAPI server exposing the network over HTTP
requirements.txt        # runtime, server and test dependencies
tests/                  # pytest suite for the engine, nn and server
testing_sclarflow.ipynb # example notebook: data, training and plots
```
