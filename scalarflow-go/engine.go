// Package scalarflow is a from-scratch scalar autodiff and neural network
// library. It is a native Go port of the Python implementation under
// ../scalarflow.
package scalarflow

import (
	"fmt"
	"math"
	"os"
	"strings"
)

// Value is a scalar node in a computation graph. It stores the data, the
// accumulated gradient and the parents that produced it.
type Value struct {
	Data  float64
	Grad  float64
	Label string

	prev     []*Value
	op       string
	backward func()
}

// NewValue creates a leaf Value.
func NewValue(data float64) *Value {
	return &Value{Data: data, backward: func() {}}
}

// NewLabeledValue creates a leaf Value with a display label.
func NewLabeledValue(data float64, label string) *Value {
	return &Value{Data: data, Label: label, backward: func() {}}
}

func newOp(data float64, prev []*Value, op string) *Value {
	return &Value{Data: data, prev: prev, op: op, backward: func() {}}
}

func (v *Value) String() string {
	return fmt.Sprintf("value(data = %g, grad = %g)", v.Data, v.Grad)
}

// Add returns v + other.
func (v *Value) Add(other *Value) *Value {
	sum := newOp(v.Data+other.Data, []*Value{v, other}, "+")
	sum.backward = func() {
		v.Grad += sum.Grad
		other.Grad += sum.Grad
	}
	return sum
}

// Neg returns -v.
func (v *Value) Neg() *Value {
	return v.Mul(NewValue(-1))
}

// Sub returns v - other.
func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

// Mul returns v * other.
func (v *Value) Mul(other *Value) *Value {
	product := newOp(v.Data*other.Data, []*Value{v, other}, "*")
	product.backward = func() {
		v.Grad += other.Data * product.Grad
		other.Grad += v.Data * product.Grad
	}
	return product
}

// Pow returns v raised to a constant power.
func (v *Value) Pow(power float64) *Value {
	out := newOp(math.Pow(v.Data, power), []*Value{v}, fmt.Sprintf("**%g", power))
	out.backward = func() {
		v.Grad += power * math.Pow(v.Data, power-1) * out.Grad
	}
	return out
}

// Div returns v / other.
func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

// Exp returns e raised to v.
func (v *Value) Exp() *Value {
	e := newOp(math.Exp(v.Data), []*Value{v}, "exp")
	e.backward = func() {
		v.Grad += e.Data * e.Grad
	}
	return e
}

// Tanh returns the hyperbolic tangent of v.
func (v *Value) Tanh() *Value {
	x := v.Data
	n := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	t := newOp(n, []*Value{v}, "tanh")
	t.backward = func() {
		v.Grad += (1 - n*n) * t.Grad
	}
	return t
}

// Relu returns the rectified linear unit of v.
func (v *Value) Relu() *Value {
	data := v.Data
	if data < 0 {
		data = 0
	}
	out := newOp(data, []*Value{v}, "relu")
	out.backward = func() {
		if out.Data > 0 {
			v.Grad += out.Grad
		}
	}
	return out
}

// Sigmoid returns the logistic sigmoid of v.
func (v *Value) Sigmoid() *Value {
	x := v.Data
	s := newOp(1/(1+math.Exp(-x)), []*Value{v}, "sigmoid")
	s.backward = func() {
		v.Grad += (s.Data * (1 - s.Data)) * s.Grad
	}
	return s
}

// Backward runs backpropagation over the graph rooted at v.
func (v *Value) Backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}
	var build func(*Value)
	build = func(node *Value) {
		if !visited[node] {
			visited[node] = true
			for _, child := range node.prev {
				build(child)
			}
			topo = append(topo, node)
		}
	}
	build(v)

	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backward()
	}
}

func (v *Value) trace() ([]*Value, [][2]*Value) {
	nodes := []*Value{}
	seen := map[*Value]bool{}
	edges := [][2]*Value{}
	var build func(*Value)
	build = func(node *Value) {
		if !seen[node] {
			seen[node] = true
			nodes = append(nodes, node)
			for _, parent := range node.prev {
				edges = append(edges, [2]*Value{parent, node})
				build(parent)
			}
		}
	}
	build(v)
	return nodes, edges
}

// DOT renders the computation graph in Graphviz DOT format.
func (v *Value) DOT() string {
	nodes, edges := v.trace()
	nodeID := func(n *Value) string { return fmt.Sprintf("n%p", n) }

	var b strings.Builder
	b.WriteString("digraph G {\n")
	b.WriteString("  rankdir=LR;\n")
	for _, n := range nodes {
		id := nodeID(n)
		b.WriteString(fmt.Sprintf(
			"  %s [label=\"%s | data: %.4f | grad: %.4f\", shape=record, style=filled, fillcolor=lightblue];\n",
			id, n.Label, n.Data, n.Grad,
		))
		if n.op != "" {
			b.WriteString(fmt.Sprintf("  %s_op [label=\"%s\", shape=circle, style=filled, fillcolor=orange];\n", id, n.op))
			b.WriteString(fmt.Sprintf("  %s_op -> %s [style=dashed];\n", id, id))
		}
	}
	for _, e := range edges {
		b.WriteString(fmt.Sprintf("  %s -> %s_op [style=dashed];\n", nodeID(e[0]), nodeID(e[1])))
	}
	b.WriteString("}\n")
	return b.String()
}

// DrawGraph writes the computation graph as a DOT file that Graphviz can render.
func (v *Value) DrawGraph(filename string) error {
	return os.WriteFile(filename, []byte(v.DOT()), 0o644)
}
