# ScalarFlow

![nn_vizorg](vizorg.png)

ScalarFlow is a small neural network library implemented from scratch in Python. It is built on a custom scalar autograd engine and provides the building blocks for defining, training, and visualizing neural networks. It supports the sigmoid, tanh, and relu activation functions.

## Features

- Custom automatic differentiation engine (`scalarflow.engine.value`) with operator overloading and reverse-mode backpropagation.
- Neural network building blocks: `neuron`, `layer`, and `network` (`scalarflow.nn`).
- Built-in activation functions: sigmoid, tanh, and relu.
- Loss functions: SVM-style hinge loss and mean squared error.
- Simple training loop using gradient descent.
- Computation graph visualization with Graphviz.

## Project structure

```
scalarflow/
  __init__.py     Exposes the engine and nn submodules
  engine.py       Autograd value class, activations, backprop, graph drawing
  nn.py           neuron, layer, network, losses, and training helpers
testing_sclarflow.ipynb   Notebook used to try out the library
```

## Dependencies

- Python (>=3.6)
- numpy
- Graphviz (the Python package and the system Graphviz binary, for visualization)

## Usage

### Autograd engine

```python
from scalarflow.engine import value

a = value(2)
b = value(3)

c = a + b
d = a * b

d.backward()

print("c =", c)  # value(data = 5, grad = 1)
print("d =", d)  # value(data = 6, grad = 2)
```

### Neural network

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

### Visualizing a computation graph

```python
from scalarflow.engine import value

a = value(2, label='a')
b = value(3, label='b')
c = a * b + a
c.backward()
c.draw_graph().render('graph', view=False)
```

## Notes

- Activations are selected per neuron and passed down through layers via the `unit_activation` argument of `network` (defaults to tanh when no activation is given).
- `network.fit` currently optimizes with `mse_loss`; use `svm_loss` directly for a max-margin objective.
- The `testing_sclarflow.ipynb` notebook walks through generating data, building a network, training, and visualizing results.
