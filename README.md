# ScalarFlow

![network visualization](vizorg.png)

ScalarFlow is a small neural network library written from scratch in Python. It uses a
custom scalar auto-gradient engine to build a computation graph, run backpropagation, and
then define neurons, layers, and networks on top of it. It also renders computation graphs
with Graphviz.

The package lives in `scalarflow/`:

- `scalarflow/engine.py` — the `value` scalar and its auto-gradient engine.
- `scalarflow/nn.py` — the neural network building blocks (`module`, `neuron`, `layer`, `network`).
- `scalarflow/__init__.py` — exposes the `engine` and `nn` submodules.

## Engine (`scalarflow.engine.value`)

Each `value` wraps a single scalar `data`, tracks the `value`s it was produced from, records
the `_operation` that created it, and holds a `gradient`. Supported operations are `+`, `-`,
`*`, `/`, `**`, and `exp`, plus the activations `sigmoid`, `tanh`, and `relu`.

Calling `backward()` topologically sorts the graph, seeds the output gradient to `1`, and
applies the chain rule node by node. `draw_graph()` returns a Graphviz `Digraph` (SVG) of the
graph, showing each value's data and gradient and each operation as a node.

## Neural network (`scalarflow.nn`)

- `neuron(num_in, _activation='')` — weights plus a bias, applying the chosen activation.
- `layer(num_in, num_out, _activation='')` — a list of neurons; returns a scalar when the
  layer has a single neuron, otherwise a list.
- `network(num_in, layer_sizes, unit_activation='')` — stacks layers; `layer_sizes` is a
  list of per-layer unit counts. Activations are `'sigmoid'`, `'tanh'`, `'relu'`; any other
  value falls back to `tanh`.

`network` provides `parameters()`, `summary()`, `svm_loss(x, y, batch_size=None)`,
`mse_loss(x, y)`, `fit(x, y, epochs)`, and `predict(x)`. `fit` trains with `mse_loss` and a
simple gradient-descent update with decaying step size; `svm_loss` returns a
`(loss, accuracy)` pair.

## Requirements

- Python 3
- `numpy` (used by `scalarflow.nn`)
- `graphviz` Python package plus the Graphviz system binaries (for `draw_graph`)
- To run the notebooks: `jupyter`, `matplotlib`, and `scikit-learn` (used by `make_moons`)

Install the Python packages with:

```bash
pip install numpy graphviz jupyter matplotlib scikit-learn
```

and install the Graphviz system binaries (e.g. `apt install graphviz` or `brew install graphviz`).

## Usage

Run Python from the repository root so the `scalarflow` package is importable.

Engine:

```python
from scalarflow.engine import value

a = value(2.0, label='a')
b = value(3.0, label='b')
c = a * b + a  # c = a*b + a

c.backward()
print(a.gradient, b.gradient)

# Render the computation graph (requires Graphviz)
c.draw_graph()
```

Network:

```python
from scalarflow.nn import network

model = network(num_in=2, layer_sizes=[4, 2, 1], unit_activation='relu')
model.summary()

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

model.fit(inputs, outputs, epochs=100)
print(model.predict(inputs))
```

## Notebooks

- `building_scalarflow.ipynb` builds the `value` engine step by step: numerical derivatives,
  the computation graph, manual gradients, `backward()`, and `draw_graph()` visualizations.
- `testing_sclarflow.ipynb` imports the finished library and tests it on a two-moon dataset
  from `sklearn.datasets.make_moons`, printing a `network.summary()` and training with `fit`.

Launch either notebook from the repository root:

```bash
jupyter notebook
```

## Visualization

`vizorg.png` shows an example computation graph produced by `draw_graph()`, with value nodes
(`data` / `grad`) and operation nodes connected in the graph.
