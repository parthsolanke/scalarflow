# ScalarFlow

ScalarFlow is a small neural network library implemented from scratch in
Python. It pairs a scalar automatic-differentiation engine with lightweight
`neuron`, `layer` and `network` building blocks, so you can see exactly how the
forward and backward passes work.

## Features

- A `value` autograd type supporting `+`, `-`, `*`, `/`, `**`, `exp`, `tanh`,
  `relu` and `sigmoid`, including the reverse operators
- Backpropagation via topological sort — call `.backward()` on any `value`
- Neural network building blocks (`neuron`, `layer`, `network`) with trainable
  weights and biases and a configurable activation function
- MSE and SVM (max-margin) loss helpers
- Gradient-descent training via `network.fit` and inference via
  `network.predict`
- Computation-graph visualisation with Graphviz via `value.draw_graph()`
- A dependency-free JSON HTTP server (`python -m scalarflow.server`) exposing
  `/health`, `/model`, `/train` and `/predict`
- A Rust command-line client in `rust-client/` that drives the HTTP server

## Dependencies

- Python 3
- `numpy`
- `graphviz` (the Python package and the Graphviz system binaries, for graph
  drawing)

## Usage

### Autograd engine

```python
from scalarflow.engine import value

a = value(2)
b = value(3)

c = a + b
d = a * b

d.backward()

print(c)                       # value(data = 5, grad = 0)
print(d)                       # value(data = 6, grad = 1)
print(a.gradient, b.gradient)  # 3 2
```

### Neural network

```python
from scalarflow.nn import network

model = network(num_in=2, layer_sizes=[4, 2, 1], unit_activation='relu')

# Print the network architecture summary
model.summary()

# Train the model
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]

model.fit(inputs, outputs, epochs=100)

# Make predictions
predictions = model.predict(inputs)
print(predictions)
```

### Visualising the graph

```python
from scalarflow.engine import value

x = value(2, label='x')
y = value(3, label='y')
z = x * y + x

graph = z.draw_graph()
graph.render('graph', cleanup=True)  # writes graph.svg (requires Graphviz)
```

### HTTP server

ScalarFlow ships a small JSON HTTP server built on the Python standard library,
so no extra dependency is needed.

```bash
python -m scalarflow.server --host 127.0.0.1 --port 8000 \
  --num-in 2 --layer-sizes 4 2 1 --activation tanh
```

The server keeps one in-memory network. Endpoints:

- `GET /health` — server status and current model configuration
- `POST /model` — replace the network with
  `{"num_in": 2, "layer_sizes": [4, 2, 1], "activation": "tanh"}`
- `POST /train` — train on
  `{"inputs": [[0, 0], [1, 1]], "outputs": [0, 1], "epochs": 100}` and return
  the final MSE loss plus predictions
- `POST /predict` — run a forward pass on
  `{"inputs": [[0, 0], [1, 0]]}`

Training requires the final layer size to be `1`. Example:

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[0, 0], [0, 1], [1, 0], [1, 1]],
       "outputs": [0, 1, 1, 0], "epochs": 100}'
```

### Rust client

A small Rust command-line client for the same HTTP API lives in
[`rust-client/`](rust-client). It has no Python dependency and prints each JSON
response pretty-printed.

```bash
cd rust-client
cargo run -- health
cargo run -- train \
  --inputs '[[0,0],[0,1],[1,0],[1,1]]' \
  --outputs '[0,1,1,0]' --epochs 100
cargo run -- predict --inputs '[[0,0],[1,0]]'
```

The client defaults to `http://127.0.0.1:8000`; pass `--url` or set
`SCALARFLOW_URL` to point it at another host. See `rust-client/README.md` for
the full command list.

## Tests

The Python test suite uses the standard-library `unittest` runner and covers the
autograd engine, the network building blocks and the HTTP endpoints. Run it from
the repository root:

```bash
python -m unittest discover -s tests -v
```

The Rust client has unit tests for its argument parsing and request-body
building. Run them from `rust-client/`:

```bash
cd rust-client
cargo test
```

## Package layout

- `scalarflow/engine.py` — the `value` autograd type, backpropagation and graph
  tracing
- `scalarflow/nn.py` — `module`, `neuron`, `layer` and `network`
- `scalarflow/server.py` — standard-library HTTP server exposing predict/train
  endpoints over JSON
- `rust-client/` — Rust command-line client for the HTTP server
- `scalarflow/__init__.py` — re-exports the `engine` and `nn` modules
- `building_scalarflow.ipynb` — step-by-step notebook that builds the autograd
  engine and network from scratch, with plots and graph renders
- `testing_sclarflow.ipynb` — notebook used to sanity-test the library
- `vizorg.png` — example computation-graph image produced while building the
  library
