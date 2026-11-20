package scalarflow

import (
	"fmt"
	"math/rand"
)

// Module is implemented by neurons, layers and networks.
type Module interface {
	Parameters() []*Value
	ZeroGrad()
}

func zeroGrad(params []*Value) {
	for _, p := range params {
		p.Grad = 0
	}
}

func applyActivation(raw *Value, activation string) *Value {
	switch activation {
	case "sigmoid":
		return raw.Sigmoid()
	case "tanh":
		return raw.Tanh()
	case "relu":
		return raw.Relu()
	default:
		return raw.Tanh()
	}
}

func toValues(row []float64) []*Value {
	vals := make([]*Value, len(row))
	for i, x := range row {
		vals[i] = NewValue(x)
	}
	return vals
}

func sumValues(vals []*Value) *Value {
	total := NewValue(0)
	for _, v := range vals {
		total = total.Add(v)
	}
	return total
}

// Neuron computes a weighted sum of its inputs plus a bias, then applies an
// activation function.
type Neuron struct {
	Weights    []*Value
	Bias       *Value
	Activation string
}

// NewNeuron creates a neuron with random weights in [-1, 1) and a zero bias.
func NewNeuron(numIn int, activation string) *Neuron {
	weights := make([]*Value, numIn)
	for i := range weights {
		weights[i] = NewLabeledValue(rand.Float64()*2-1, fmt.Sprintf("w%d", i))
	}
	return &Neuron{
		Weights:    weights,
		Bias:       NewLabeledValue(0, "b"),
		Activation: activation,
	}
}

// Forward evaluates the neuron on the given inputs.
func (n *Neuron) Forward(x []*Value) *Value {
	raw := n.Bias
	for i, w := range n.Weights {
		raw = raw.Add(w.Mul(x[i]))
	}
	return applyActivation(raw, n.Activation)
}

// Parameters returns the weights followed by the bias.
func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, 0, len(n.Weights)+1)
	params = append(params, n.Weights...)
	params = append(params, n.Bias)
	return params
}

// ZeroGrad resets the gradient of every parameter.
func (n *Neuron) ZeroGrad() { zeroGrad(n.Parameters()) }

func (n *Neuron) String() string {
	return fmt.Sprintf("\nneuron(weights: %d) - activation: '%s'", len(n.Weights), n.Activation)
}

// Layer is a dense layer of neurons.
type Layer struct {
	Neurons []*Neuron
}

// NewLayer creates numOut neurons, each accepting numIn inputs.
func NewLayer(numIn, numOut int, activation string) *Layer {
	neurons := make([]*Neuron, numOut)
	for i := range neurons {
		neurons[i] = NewNeuron(numIn, activation)
	}
	return &Layer{Neurons: neurons}
}

// Forward evaluates every neuron in the layer.
func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Forward(x)
	}
	return out
}

// Parameters returns the parameters of every neuron.
func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

// ZeroGrad resets the gradient of every parameter.
func (l *Layer) ZeroGrad() { zeroGrad(l.Parameters()) }

func (l *Layer) String() string {
	return fmt.Sprintf("\nlayer of %d neurons:", len(l.Neurons))
}

// Network is a stack of dense layers.
type Network struct {
	UnitActivation string
	Layers         []*Layer
}

// NewNetwork builds a network with the given input size and per-layer sizes.
func NewNetwork(numIn int, layerSizes []int, activation string) *Network {
	sizes := append([]int{numIn}, layerSizes...)
	layers := make([]*Layer, len(layerSizes))
	for i := range layerSizes {
		layers[i] = NewLayer(sizes[i], sizes[i+1], activation)
	}
	return &Network{UnitActivation: activation, Layers: layers}
}

// Forward runs the input through every layer and returns the final outputs.
func (n *Network) Forward(x []*Value) []*Value {
	out := x
	for _, layer := range n.Layers {
		out = layer.Forward(out)
	}
	return out
}

// Parameters returns the parameters of every layer.
func (n *Network) Parameters() []*Value {
	var params []*Value
	for _, layer := range n.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

// ZeroGrad resets the gradient of every parameter.
func (n *Network) ZeroGrad() { zeroGrad(n.Parameters()) }

func (n *Network) String() string {
	return fmt.Sprintf("neural network of %d layers", len(n.Layers))
}

// Summary prints the network architecture and parameter counts.
func (n *Network) Summary() {
	weights := 0
	neurons := 0
	for _, layer := range n.Layers {
		neurons += len(layer.Neurons)
		if len(layer.Neurons) > 0 {
			weights += len(layer.Neurons[0].Weights)
		}
	}

	fmt.Println("Network architecture:")
	fmt.Println("--------------------------------------")
	fmt.Println(n)
	fmt.Println("--------------------------------------")
	fmt.Printf("Total number of parameters: %d\n", len(n.Parameters()))
	fmt.Printf("Total number of dense layers: %d\n", len(n.Layers))
	fmt.Printf("Total number of neurons: %d\n", neurons)
	fmt.Printf("Total number of weights: %d\n", weights)
	fmt.Printf("Total number of biases: %d\n", neurons)
}

// SVMLoss computes the max-margin (SVM) loss and accuracy for a single output
// network. A batchSize <= 0 or >= len(x) uses the full dataset.
func (n *Network) SVMLoss(x [][]float64, y []float64, batchSize int) (*Value, float64) {
	Xb, yb := x, y
	if batchSize > 0 && batchSize < len(x) {
		perm := rand.Perm(len(x))[:batchSize]
		Xb = make([][]float64, len(perm))
		yb = make([]float64, len(perm))
		for i, idx := range perm {
			Xb[i] = x[idx]
			yb[i] = y[idx]
		}
	}

	losses := make([]*Value, len(Xb))
	scores := make([]*Value, len(Xb))
	for i := range Xb {
		out := n.Forward(toValues(Xb[i]))
		score := out[0]
		scores[i] = score
		losses[i] = NewValue(1).Add(score.Mul(NewValue(-yb[i]))).Relu()
	}

	dataLoss := sumValues(losses).Mul(NewValue(1.0 / float64(len(losses))))

	reg := NewValue(0)
	for _, p := range n.Parameters() {
		reg = reg.Add(p.Pow(2))
	}
	totalLoss := dataLoss.Add(reg.Mul(NewValue(0.0001)))

	correct := 0
	for i, score := range scores {
		if (yb[i] > 0) == (score.Data > 0) {
			correct++
		}
	}
	return totalLoss, float64(correct) / float64(len(scores))
}

// MSELoss computes mean squared error with L2 regularization for a single
// output network.
func (n *Network) MSELoss(x [][]float64, y []float64) *Value {
	loss := NewValue(0)
	for i := range x {
		out := n.Forward(toValues(x[i]))
		diff := out[0].Add(NewValue(-y[i]))
		loss = loss.Add(diff.Pow(2))
	}

	reg := NewValue(0)
	for _, p := range n.Parameters() {
		reg = reg.Add(p.Pow(2))
	}

	return loss.Add(reg.Mul(NewValue(0.00001))).Mul(NewValue(1.0 / (2 * float64(len(x)))))
}

// Fit trains the network with gradient descent for the given number of epochs.
func (n *Network) Fit(x [][]float64, y []float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		loss := n.MSELoss(x, y)

		n.ZeroGrad()
		loss.Backward()

		alpha := 1.0 - 0.9*float64(epoch)/100
		for _, param := range n.Parameters() {
			param.Data -= alpha * param.Grad
		}

		if epoch%10 == 0 {
			fmt.Printf("Epoch: %d/%d - Loss: %.4f\n", epoch+1, epochs, loss.Data)
		}
	}
}

// Predict returns the network output for each input row.
func (n *Network) Predict(x [][]float64) []float64 {
	out := make([]float64, len(x))
	for i, row := range x {
		out[i] = n.Forward(toValues(row))[0].Data
	}
	return out
}
