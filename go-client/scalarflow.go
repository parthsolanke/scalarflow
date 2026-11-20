package scalarflow

// Scalarflow is the aggregate namespace for the client. It mirrors the default
// scalarflow object exported by the JavaScript client.
type Scalarflow struct{}

// Default is the shared Scalarflow namespace instance.
var Default = Scalarflow{}

// NewValue creates a leaf Value.
func (Scalarflow) NewValue(data float64) *Value {
	return NewValue(data)
}

// NewValueWith creates a Value with explicit parents, operation and label.
func (Scalarflow) NewValueWith(data float64, parents []*Value, operation, label string) *Value {
	return NewValueWith(data, parents, operation, label)
}

// NewNeuron creates a neuron with an optional activation.
func (Scalarflow) NewNeuron(numIn int, activation ...Activation) *Neuron {
	return NewNeuron(numIn, activation...)
}

// NewLayer creates a dense layer with an optional activation.
func (Scalarflow) NewLayer(numIn, numOut int, activation ...Activation) *Layer {
	return NewLayer(numIn, numOut, activation...)
}

// NewNetwork creates a feed-forward network with an optional activation.
func (Scalarflow) NewNetwork(numIn int, layerSizes []int, unitActivation ...Activation) *Network {
	return NewNetwork(numIn, layerSizes, unitActivation...)
}
