# ScalarFlow

![nn_vizorg](vizorg.png)

This project is a simple neural network library implemented from scratch in Python. It provides a framework for building and training neural networks using a custom auto-gradient engine. The library includes classes for defining neural network layers, neurons, network mode and for visualization of networks and their gradients. Supports common activation functions like sigmoid, tanh, and relu.

## Features

- Implementation of a custom auto-gradient engine for automatic differentiation.
- Definition of neural network layers with customizable activation functions.
- Support for common activation functions: sigmoid, tanh, and relu.
- Training loop using gradient descent optimization algorithm.
- Visualizations of computation graphs using Graphviz.

## Go client
A native Go port of the library lives in `scalarflow-go/` (module `scalarflow`, standard library only, Go >= 1.20). It mirrors the Python API using methods instead of operator overloading: `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Exp`, `Tanh`, `Relu` and `Sigmoid` on `Value`, plus the `Neuron`, `Layer` and `Network` types with `NewNetwork`, `Summary`, `Fit` and `Predict` (and the `SVMLoss`/`MSELoss` helpers).

```go
package main

import (
	"fmt"

	scalarflow "scalarflow"
)

func main() {
	model := scalarflow.NewNetwork(2, []int{4, 2, 1}, "relu")
	model.Summary()

	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs := []float64{0, 1, 1, 0}

	model.Fit(inputs, outputs, 100)
	fmt.Println(model.Predict(inputs))
}
```

Run the Go tests with:

```bash
cd scalarflow-go
go test ./...
```

## Ping/pong connectivity
ScalarFlow can talk to the Namergence service over HTTP. ScalarFlow serves a `/pong` endpoint and its Go client can call a peer's `/ping` endpoint, so the two projects can confirm they are reachable from each other.

Run the ScalarFlow pong server:

```bash
cd scalarflow-go
go run ./cmd/pong
```

It listens on `http://127.0.0.1:8001` and answers `GET /pong` with:

```json
{"message": "pong", "service": "scalarflow"}
```

Check that the Namergence server (whose `GET /ping` at `http://127.0.0.1:8000` answers `{"message": "pong", "service": "namergence"}`) is reachable:

```go
client := scalarflow.NewPingClient("http://127.0.0.1:8000")
pong, err := client.Ping(context.Background())
if err != nil {
	log.Fatal(err)
}
fmt.Println(pong.Service, pong.Message)
```

The connectivity API is `NewPingClient`/`Ping` on the client side and `NewPongServer`/`PongHandler` (with the `Pong` handler) on the server side. `NewPongServer("")` and `NewPingClient("")` fall back to the defaults `127.0.0.1:8001` and `http://127.0.0.1:8000`. `NewPongServer` and `PongHandler` can also be mounted in your own Go HTTP server instead of running `cmd/pong`.

## Dependencies
The neural network library has the following dependencies:

- Python (>=3.6)
- Graphviz (for visualization)
- Go (>=1.20) for the Go client in `scalarflow-go/`

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
