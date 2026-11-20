package scalarflow

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Scalar is a numeric operand accepted by the engine and network operations.
// It is either a float64/int (or another numeric type) or a *Value.
type Scalar = interface{}

// Activation is the name of a supported activation function.
type Activation string

const (
	ActivationNone    Activation = ""
	ActivationSigmoid Activation = "sigmoid"
	ActivationTanh    Activation = "tanh"
	ActivationRelu    Activation = "relu"
)

var nextID int

// NextId returns the next engine node id. It mirrors the module-level nextId
// counter in the JavaScript client.
func NextId() int {
	id := nextID
	nextID++
	return id
}

// Format renders a scalar with four decimal places, matching the JavaScript
// format helper.
func Format(v float64) string {
	return strconv.FormatFloat(v, 'f', 4, 64)
}

func toFloat(other Scalar) (float64, bool) {
	switch v := other.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint8:
		return float64(v), true
	case uint16:
		return float64(v), true
	case uint32:
		return float64(v), true
	case uint64:
		return float64(v), true
	default:
		return 0, false
	}
}

// AsValue converts a scalar operand into a *Value. Existing values are
// returned unchanged, mirroring asValue in the JavaScript client.
func AsValue(other Scalar) *Value {
	switch v := other.(type) {
	case *Value:
		return v
	case Value:
		return &v
	}
	if n, ok := toFloat(other); ok {
		return NewValue(n)
	}
	panic(fmt.Sprintf("scalarflow: unsupported scalar type %T", other))
}

// Value is the scalar autograd node. It builds a computation graph and
// propagates gradients through Backward.
type Value struct {
	Data       float64
	Gradient   float64
	Label      string
	Prev       []*Value
	Operation  string
	BackwardFn func()
	ID         int
}

// NewValue creates a leaf Value with the given data.
func NewValue(data float64) *Value {
	return NewValueWith(data, nil, "", "")
}

// NewValueWith creates a Value with explicit parents, operation and label,
// mirroring the JavaScript value constructor.
func NewValueWith(data float64, parents []*Value, operation, label string) *Value {
	return &Value{
		Data:       data,
		Label:      label,
		Prev:       parents,
		Operation:  operation,
		BackwardFn: func() {},
		ID:         NextId(),
	}
}

func (v *Value) String() string {
	return fmt.Sprintf("value(data = %v, grad = %v)", v.Data, v.Gradient)
}

// Add returns v + other.
func (v *Value) Add(other Scalar) *Value {
	o := AsValue(other)
	var sum *Value
	sum = NewValueWith(v.Data+o.Data, []*Value{v, o}, "+", "")
	sum.BackwardFn = func() {
		v.Gradient += sum.Gradient
		o.Gradient += sum.Gradient
	}
	return sum
}

// Radd returns other + v.
func (v *Value) Radd(other Scalar) *Value {
	return AsValue(other).Add(v)
}

// Neg returns -v.
func (v *Value) Neg() *Value {
	return v.Mul(-1.0)
}

// Sub returns v - other.
func (v *Value) Sub(other Scalar) *Value {
	return v.Add(AsValue(other).Neg())
}

// Rsub returns other - v.
func (v *Value) Rsub(other Scalar) *Value {
	return AsValue(other).Add(v.Neg())
}

// Mul returns v * other.
func (v *Value) Mul(other Scalar) *Value {
	o := AsValue(other)
	var product *Value
	product = NewValueWith(v.Data*o.Data, []*Value{v, o}, "*", "")
	product.BackwardFn = func() {
		v.Gradient += o.Data * product.Gradient
		o.Gradient += v.Data * product.Gradient
	}
	return product
}

// Rmul returns other * v.
func (v *Value) Rmul(other Scalar) *Value {
	return AsValue(other).Mul(v)
}

// Pow returns v raised to a numeric power.
func (v *Value) Pow(other float64) *Value {
	out := NewValueWith(math.Pow(v.Data, other), []*Value{v}, fmt.Sprintf("**%v", other), "")
	out.BackwardFn = func() {
		v.Gradient += other * math.Pow(v.Data, other-1) * out.Gradient
	}
	return out
}

// Truediv returns v / other.
func (v *Value) Truediv(other Scalar) *Value {
	return v.Mul(AsValue(other).Pow(-1))
}

// Div is an alias for Truediv.
func (v *Value) Div(other Scalar) *Value {
	return v.Truediv(other)
}

// Rtruediv returns other / v.
func (v *Value) Rtruediv(other Scalar) *Value {
	return AsValue(other).Mul(v.Pow(-1))
}

// Rdiv is an alias for Rtruediv.
func (v *Value) Rdiv(other Scalar) *Value {
	return v.Rtruediv(other)
}

// Exp returns e^v.
func (v *Value) Exp() *Value {
	e := NewValueWith(math.Exp(v.Data), []*Value{v}, "exp", "")
	e.BackwardFn = func() {
		v.Gradient += e.Data * e.Gradient
	}
	return e
}

// Tanh returns the hyperbolic tangent of v.
func (v *Value) Tanh() *Value {
	n := math.Tanh(v.Data)
	t := NewValueWith(n, []*Value{v}, "tanh", "")
	t.BackwardFn = func() {
		v.Gradient += (1 - n*n) * t.Gradient
	}
	return t
}

// Relu returns the rectified linear unit of v.
func (v *Value) Relu() *Value {
	data := v.Data
	if data < 0 {
		data = 0
	}
	out := NewValueWith(data, []*Value{v}, "relu", "")
	out.BackwardFn = func() {
		if out.Data > 0 {
			v.Gradient += out.Gradient
		}
	}
	return out
}

// Sigmoid returns the logistic sigmoid of v.
func (v *Value) Sigmoid() *Value {
	s := NewValueWith(1/(1+math.Exp(-v.Data)), []*Value{v}, "sigmoid", "")
	s.BackwardFn = func() {
		v.Gradient += s.Data * (1 - s.Data) * s.Gradient
	}
	return s
}

// Backward runs reverse-mode differentiation from v.
func (v *Value) Backward() {
	var topo []*Value
	visited := map[*Value]bool{}

	var build func(*Value)
	build = func(node *Value) {
		if !visited[node] {
			visited[node] = true
			for _, child := range node.Prev {
				build(child)
			}
			topo = append(topo, node)
		}
	}
	build(v)

	v.Gradient = 1
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].BackwardFn()
	}
}

// Trace returns all nodes and edges reachable from v.
func (v *Value) Trace() ([]*Value, [][2]*Value) {
	var nodes []*Value
	var edges [][2]*Value
	seen := map[*Value]bool{}

	var build func(*Value)
	build = func(node *Value) {
		if !seen[node] {
			seen[node] = true
			nodes = append(nodes, node)
			for _, parent := range node.Prev {
				edges = append(edges, [2]*Value{parent, node})
				build(parent)
			}
		}
	}
	build(v)

	return nodes, edges
}

// ToDot renders the computation graph as Graphviz DOT source.
func (v *Value) ToDot() string {
	nodes, edges := v.Trace()
	lines := []string{"digraph G {", "  rankdir=LR;"}

	for _, n := range nodes {
		uid := strconv.Quote(fmt.Sprintf("v%d", n.ID))
		lines = append(lines, fmt.Sprintf(
			`  %s [label="{ %s | data: %s | grad: %s }", shape=record, style=filled, fillcolor=lightblue];`,
			uid, n.Label, Format(n.Data), Format(n.Gradient),
		))
		if n.Operation != "" {
			opID := strconv.Quote(fmt.Sprintf("v%d_%s", n.ID, n.Operation))
			lines = append(lines, fmt.Sprintf(
				`  %s [label=%s, shape=circle, style=filled, fillcolor=orange];`,
				opID, strconv.Quote(n.Operation),
			))
			lines = append(lines, fmt.Sprintf("  %s -> %s [style=dashed];", opID, uid))
		}
	}

	for _, edge := range edges {
		from := strconv.Quote(fmt.Sprintf("v%d", edge[0].ID))
		to := strconv.Quote(fmt.Sprintf("v%d_%s", edge[1].ID, edge[1].Operation))
		lines = append(lines, fmt.Sprintf("  %s -> %s [style=dashed];", from, to))
	}

	lines = append(lines, "}")
	return strings.Join(lines, "\n")
}

// DrawGraph returns the Graphviz DOT source for the computation graph.
func (v *Value) DrawGraph() string {
	return v.ToDot()
}
