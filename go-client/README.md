# scalarflow (Go client)

A Go port of the ScalarFlow Python package. It exposes the same public behavior
as `scalarflow/engine.py` and `scalarflow/nn.py`, and mirrors the JavaScript
client in `js-client/`.

## Install

```bash
go get /path/to/this/repo/go-client
```

Or add a `replace` directive to your `go.mod` while working locally:

```
require scalarflow v0.0.0
replace scalarflow => /path/to/this/repo/go-client
```

## Usage

### Autograd engine

Python overloads arithmetic operators; Go callers use the matching methods
(`Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`).

```go
package main

import (
	"fmt"

	scalarflow "scalarflow"
)

func main() {
	a := scalarflow.NewValue(2)
	b := scalarflow.NewValue(3)

	c := a.Add(b)
	d := a.Mul(b)

	d.Backward()

	fmt.Println(c)            // value(data = 5, grad = 0)
	fmt.Println(d)            // value(data = 6, grad = 1)
	fmt.Println(a.Gradient)   // 3
}
```

`Value` also provides `Exp`, `Tanh`, `Relu`, `Sigmoid`, `Backward`, `Trace`,
`ToDot`, `DrawGraph`, `Format`, `AsValue` and `NextId`. `DrawGraph` returns
Graphviz DOT source as a string.

### Neural networks

```go
package main

import (
	"fmt"

	scalarflow "scalarflow"
)

func main() {
	model := scalarflow.NewNetwork(2, []int{4, 2, 1}, scalarflow.ActivationRelu)
	model.Summary()

	inputs := [][]scalarflow.Scalar{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}
	outputs := []scalarflow.Scalar{0.0, 1.0, 1.0, 0.0}

	model.Fit(inputs, outputs, 100)

	fmt.Println(model.Predict(inputs))
}
```

`Module` is implemented by `Neuron`, `Layer` and `Network`. `Network` supports
`Forward`/`Call`, `Parameters`, `ZeroGrad`, `Summary`, `SvmLoss`, `MseLoss`,
`Fit` and `Predict`. `SvmLoss` returns `(loss *Value, accuracy float64)`.

The `Scalarflow` aggregate exposes constructors through `scalarflow.Default`:

```go
x := scalarflow.Default.NewValue(1.5)
model := scalarflow.Default.NewNetwork(1, []int{1})
```
