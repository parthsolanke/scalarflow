import random
from scalarflow.engine import value

# building a neural network library 

class neuron:
    
    def __init__(self, num_in , _activation = ''):
        self.weights = [value(random.uniform(-1, 1), label=f'w{i}') for i in range(num_in)]
        self.bias = value(0, label='b')
        self._activation = _activation
        
    def __call__(self, x):
        raw = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        act = raw.sigmoid() if self._activation == 'sigmoid' \
            else raw.tanh() if self._activation == 'tanh' \
            else raw.relu() if self._activation == 'relu' else raw
        return act
    
    def parameters(self):
        return self.weights + [self.bias]
    
class layer:
        
        # num_out = number of units in the layer, num_in = number of inputs to each unit
        def __init__(self, num_in, num_out, _activation = ''):
            self.neurons = [neuron(num_in, _activation) for _ in range(num_out)]
            
        def __call__(self, x):
            out = [n(x) for n in self.neurons]
            return out[0] if len(out) == 1 else out
        
class network:
    
    # num_out = list which defines size of each layer, num_in = number of inputs
    def __init__(self, num_in, layer_sizes, unit_activation = ''):
        self.unit_activation = unit_activation
        size  = [num_in] + layer_sizes
        self.layers = [layer(size[i], size[i+1], unit_activation) for i in range(len(size)-1)]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x