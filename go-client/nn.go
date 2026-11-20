package scalarflow

import (
	"fmt"
	"math/rand"
	"strings"
)

// Output is the result of a module forward pass. It is either a single
// *Value or a slice of *Value, matching the JavaScript forward return type.
type Output = interface{}

// Module is the shared interface implemented by Neuron, Layer and Network.
type Module interface {
	ZeroGrad()
	Parameters() []*Value
	Forward(x []Scalar) Output
	Call(x []Scalar) Output
}

// ToNumber converts a Scalar (or a slice whose first element is a Scalar)
// into a float64, mirroring toNumber in the JavaScript client.
func ToNumber(v Scalar) float64 {
	switch x := v.(type) {
	case *Value:
		return x.Data
	case Value:
		return x.Data
	case []*Value:
		if len(x) == 0 {
			return 0
		}
		return x[0].Data
	case []Scalar:
		if len(x) == 0 {
			return 0
		}
		return ToNumber(x[0])
	}
	if n, ok := toFloat(v); ok {
		return n
	}
	return 0
}

// ScoreOf returns the first Value from a scalar or a slice of scalars,
// mirroring scoreOf in the JavaScript client.
func ScoreOf(v Scalar) *Value {
	switch x := v.(type) {
	case *Value:
		return x
	case Value:
		return &x
	case []*Value:
		if len(x) == 0 {
			return NewValue(0)
		}
		return x[0]
	case []Scalar:
		if len(x) == 0 {
			return NewValue(0)
		}
		return ScoreOf(x[0])
	}
	return AsValue(v)
}

func toScalars(v Output) []Scalar {
	switch x := v.(type) {
	case []Scalar:
		return x
	case []*Value:
		out := make([]Scalar, len(x))
		for i := range x {
			out[i] = x[i]
		}
		return out
	case []float64:
		out := make([]Scalar, len(x))
		for i := range x {
			out[i] = x[i]
		}
		return out
	case nil:
		return nil
	default:
		return []Scalar{x}
	}
}

func toValues(v Output) []*Value {
	scalars := toScalars(v)
	out := make([]*Value, len(scalars))
	for i, s := range scalars {
		out[i] = ScoreOf(s)
	}
	return out
}

func activationName(a Activation) string {
	s := string(a)
	if s == "" {
		return ""
	}
	return strings.ToUpper(s[:1]) + strings.ToLower(s[1:])
}

var (
	_ Module = (*Neuron)(nil)
	_ Module = (*Layer)(nil)
	_ Module = (*Network)(nil)
)

// Neuron is a single dense unit with a bias and one weight per input.
type Neuron struct {
	Weights    []*Value
	Bias       *Value
	Activation Activation
}

// NewNeuron creates a neuron with numIn inputs and an optional activation.
func NewNeuron(numIn int, activation ...Activation) *Neuron {
	act := ActivationNone
	if len(activation) > 0 {
		act = activation[0]
	}
	weights := make([]*Value, numIn)
	for i := range weights {
		weights[i] = NewValueWith(rand.Float64()*2-1, nil, "", fmt.Sprintf("w%d", i))
	}
	return &Neuron{
		Weights:    weights,
		Bias:       NewValueWith(0, nil, "", "b"),
		Activation: act,
	}
}

// Forward computes the neuron output for x.
func (n *Neuron) Forward(x []Scalar) Output {
	raw := n.Bias
	size := len(n.Weights)
	if len(x) < size {
		size = len(x)
	}
	for i := 0; i < size; i++ {
		raw = raw.Add(n.Weights[i].Mul(x[i]))
	}

	switch n.Activation {
	case ActivationSigmoid:
		return raw.Sigmoid()
	case ActivationTanh:
		return raw.Tanh()
	case ActivationRelu:
		return raw.Relu()
	default:
		return raw.Tanh()
	}
}

// Call is an alias for Forward.
func (n *Neuron) Call(x []Scalar) Output {
	return n.Forward(x)
}

// Parameters returns the neuron weights followed by its bias.
func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, 0, len(n.Weights)+1)
	params = append(params, n.Weights...)
	params = append(params, n.Bias)
	return params
}

// ZeroGrad resets the gradient of every parameter.
func (n *Neuron) ZeroGrad() {
	for _, p := range n.Parameters() {
		p.Gradient = 0
	}
}

func (n *Neuron) String() string {
	return fmt.Sprintf("\nneuron(weights: %d) - activation: '%s'", len(n.Weights), activationName(n.Activation))
}

// Layer is a dense layer of neurons.
type Layer struct {
	Neurons []*Neuron
}

// NewLayer creates a layer with numOut neurons, each accepting numIn inputs.
func NewLayer(numIn, numOut int, activation ...Activation) *Layer {
	neurons := make([]*Neuron, numOut)
	for i := range neurons {
		neurons[i] = NewNeuron(numIn, activation...)
	}
	return &Layer{Neurons: neurons}
}

// Forward computes the layer output. A single-neuron layer returns one Value.
func (l *Layer) Forward(x []Scalar) Output {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = ScoreOf(n.Forward(x))
	}
	if len(out) == 1 {
		return out[0]
	}
	return out
}

// Call is an alias for Forward.
func (l *Layer) Call(x []Scalar) Output {
	return l.Forward(x)
}

// Parameters returns the parameters of every neuron in the layer.
func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

// ZeroGrad resets the gradient of every parameter.
func (l *Layer) ZeroGrad() {
	for _, p := range l.Parameters() {
		p.Gradient = 0
	}
}

func (l *Layer) String() string {
	parts := make([]string, len(l.Neurons))
	for i, n := range l.Neurons {
		parts[i] = n.String()
	}
	return fmt.Sprintf("\nlayer of %d neurons: [%s]", len(l.Neurons), strings.Join(parts, ", "))
}

// Network is a feed-forward stack of dense layers.
type Network struct {
	UnitActivation Activation
	Layers         []*Layer
}

// NewNetwork creates a network with numIn inputs, the given layer sizes and an
// optional activation.
func NewNetwork(numIn int, layerSizes []int, unitActivation ...Activation) *Network {
	act := ActivationNone
	if len(unitActivation) > 0 {
		act = unitActivation[0]
	}
	size := make([]int, 0, len(layerSizes)+1)
	size = append(size, numIn)
	size = append(size, layerSizes...)

	layers := make([]*Layer, len(layerSizes))
	for i := range layerSizes {
		layers[i] = NewLayer(size[i], size[i+1], act)
	}
	return &Network{UnitActivation: act, Layers: layers}
}

// Forward runs x through every layer of the network.
func (n *Network) Forward(x []Scalar) Output {
	out := toScalars(x)
	for _, l := range n.Layers {
		out = toScalars(l.Forward(out))
	}
	values := make([]*Value, len(out))
	for i, s := range out {
		values[i] = ScoreOf(s)
	}
	if len(values) == 1 {
		return values[0]
	}
	return values
}

// Call is an alias for Forward.
func (n *Network) Call(x []Scalar) Output {
	return n.Forward(x)
}

// Parameters returns the parameters of every layer in the network.
func (n *Network) Parameters() []*Value {
	var params []*Value
	for _, l := range n.Layers {
		params = append(params, l.Parameters()...)
	}
	return params
}

// ZeroGrad resets the gradient of every parameter.
func (n *Network) ZeroGrad() {
	for _, p := range n.Parameters() {
		p.Gradient = 0
	}
}

func (n *Network) String() string {
	parts := make([]string, len(n.Layers))
	for i, l := range n.Layers {
		parts[i] = l.String()
	}
	return fmt.Sprintf("neural network of %d layers: [%s\n]", len(n.Layers), strings.Join(parts, ", "))
}

// Summary prints a human-readable network summary.
func (n *Network) Summary() {
	fmt.Println("Network architecture:")
	fmt.Println("--------------------------------------")
	fmt.Println(n.String())
	fmt.Println("--------------------------------------")
	fmt.Printf("Total number of parameters: %d\n", len(n.Parameters()))
	fmt.Printf("Total number of dense layers: %d\n", len(n.Layers))

	neurons := 0
	weights := 0
	biases := 0
	for _, l := range n.Layers {
		neurons += len(l.Neurons)
		biases += len(l.Neurons)
		if len(l.Neurons) > 0 {
			weights += len(l.Neurons[0].Weights)
		}
	}
	fmt.Printf("Total number of neurons: %d\n", neurons)
	fmt.Printf("Total number of weights: %d\n", weights)
	fmt.Printf("Total number of biases: %d\n", biases)
}

// SvmLoss returns the SVM max-margin loss and accuracy. When batchSize is
// provided a random batch of that size is used.
func (n *Network) SvmLoss(x [][]Scalar, y []Scalar, batchSize ...int) (*Value, float64) {
	Xb := x
	yb := y

	if len(batchSize) > 0 {
		size := batchSize[0]
		indices := rand.Perm(len(x))
		if size < len(indices) {
			indices = indices[:size]
		}
		Xb = make([][]Scalar, len(indices))
		yb = make([]Scalar, len(indices))
		for i, idx := range indices {
			Xb[i] = x[idx]
			yb[i] = y[idx]
		}
	}

	inputs := make([][]Scalar, len(Xb))
	for i, xrow := range Xb {
		row := make([]Scalar, len(xrow))
		for j, v := range xrow {
			row[j] = AsValue(v)
		}
		inputs[i] = row
	}

	scores := make([]*Value, len(inputs))
	for i, input := range inputs {
		scores[i] = ScoreOf(n.Forward(input))
	}

	losses := make([]*Value, len(scores))
	for i, score := range scores {
		losses[i] = score.Mul(-ToNumber(yb[i])).Add(1).Relu()
	}
	dataLoss := NewValue(0)
	for _, loss := range losses {
		dataLoss = dataLoss.Add(loss)
	}
	dataLoss = dataLoss.Mul(1.0 / float64(len(losses)))

	lam := 0.0001
	regLoss := NewValue(0)
	for _, p := range n.Parameters() {
		regLoss = regLoss.Add(p.Mul(p))
	}
	regLoss = regLoss.Mul(lam)

	totalLoss := dataLoss.Add(regLoss)

	correct := 0
	for i, score := range scores {
		if (ToNumber(yb[i]) > 0) == (score.Data > 0) {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(scores))

	return totalLoss, accuracy
}

// MseLoss returns the mean squared error plus L2 regularization.
func (n *Network) MseLoss(x [][]Scalar, y []Scalar) *Value {
	scores := make([]*Value, len(x))
	for i, xi := range x {
		scores[i] = ScoreOf(n.Forward(xi))
	}

	loss := NewValue(0)
	for i, score := range scores {
		error := score.Sub(ToNumber(y[i]))
		loss = loss.Add(error.Mul(error))
	}

	lam := 0.00001
	regLoss := NewValue(0)
	for _, p := range n.Parameters() {
		regLoss = regLoss.Add(p.Mul(p))
	}
	regLoss = regLoss.Mul(lam)

	return loss.Add(regLoss).Mul(1.0 / float64(2*len(scores)))
}

// Fit trains the network with gradient descent for the given number of epochs.
func (n *Network) Fit(x [][]Scalar, y []Scalar, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		loss := n.MseLoss(x, y)

		n.ZeroGrad()
		loss.Backward()

		alpha := 1.0 - 0.9*float64(epoch)/100
		for _, param := range n.Parameters() {
			param.Data -= alpha * param.Gradient
		}

		if epoch%10 == 0 {
			fmt.Printf("Epoch: %d/%d - Loss: %s\n", epoch+1, epochs, Format(loss.Data))
		}
	}
}

// Predict returns the network output for every input row.
func (n *Network) Predict(x [][]Scalar) []float64 {
	out := make([]float64, len(x))
	for i, xi := range x {
		out[i] = ScoreOf(n.Forward(xi)).Data
	}
	return out
}
