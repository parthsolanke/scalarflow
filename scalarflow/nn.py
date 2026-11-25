"""Neural-network building blocks built on top of the ScalarFlow autograd engine.

The hierarchy is ``module`` (shared parameter/gradient helpers) -> ``neuron``
(a single weighted sum plus activation) -> ``layer`` (a list of neurons) ->
``network`` (a stack of layers with training and loss helpers). All maths is
expressed with ``value`` so gradients are tracked automatically.
"""

import random
import numpy as np
from scalarflow.engine import value

class module:
    """Base class giving every component a uniform parameter/gradient interface."""
    
    def zero_grad(self):
        # Reset accumulated gradients before the next backward pass.
        for p in self.parameters():
            p.gradient = 0
            
    def parameters(self):
        # Subclasses override this with their trainable ``value`` objects.
        return []

class neuron(module):
    """A single unit: a weighted sum of inputs plus a bias, passed through an activation."""
    
    def __init__(self, num_in , _activation = ''):
        # One weight per input, randomly initialised in [-1, 1].
        self.weights = [value(random.uniform(-1, 1), label=f'w{i}') for i in range(num_in)]
        self.bias = value(0, label='b')
        self._activation = _activation
        
    def __call__(self, x):
        # Weighted sum; the bias is the initial value so it is always included.
        raw = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        # Apply the configured activation, falling back to tanh when unset/unknown.
        act = raw.sigmoid() if self._activation == 'sigmoid' \
            else raw.tanh() if self._activation == 'tanh' \
            else raw.relu() if self._activation == 'relu' else raw.tanh()
        return act
    
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f"\nneuron(weights: {len(self.weights)}) - activation: '{self._activation.capitalize()}'"
    
class layer(module):
        """A dense layer made of ``num_out`` neurons sharing the same input width."""
        
        # num_out = number of units in the layer, num_in = number of inputs to each unit
        def __init__(self, num_in, num_out, _activation = ''):
            self.neurons = [neuron(num_in, _activation) for _ in range(num_out)]
            
        def __call__(self, x):
            out = [n(x) for n in self.neurons]
            # Collapse a single-unit layer to a scalar so later layers get the shape they expect.
            return out[0] if len(out) == 1 else out
        
        def parameters(self):
            #return [p for n in self.neurons for p in n.parameters()]
            params = []
            for n in self.neurons:
                p_unit = n.parameters()
                params.extend(p_unit)
            return params
        
        def __repr__(self):
            return f"\nlayer of {len(self.neurons)} neurons: [{', '.join(str(n) for n in self.neurons)}]"   
        
class network(module):
    """A feed-forward stack of ``layer`` objects with training helpers."""
    
    # num_out = list which defines size of each layer, num_in = number of inputs
    def __init__(self, num_in, layer_sizes, unit_activation = ''):
        self.unit_activation = unit_activation
        # Prepend the input width so consecutive entries describe each layer's in/out sizes.
        size  = [num_in] + layer_sizes
        self.layers = [layer(size[i], size[i+1], unit_activation) for i in range(len(layer_sizes))]
        
    def __call__(self, x):
        # Forward pass: feed the input through every layer in order.
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """Flatten every layer's parameters into one list."""
        #return [p for layer in self.layers for p in layer.parameters()]
        params = []
        for layer in self.layers:
            p_layer = layer.parameters()
            params.extend(p_layer)
        return params
    
    def __repr__(self):
        return f"neural network of {len(self.layers)} layers: [{', '.join(str(layer) for layer in self.layers)}\n]"
    
    def summary(self):
        """Print a human-readable description of the architecture and parameter count."""
        print(f"Network architecture:")
        print("--------------------------------------")
        print(self)
        print("--------------------------------------")
        print(f"Total number of parameters: {len(self.parameters())}")
        print(f"Total number of dense layers: {len(self.layers)}")
        print(f"Total number of neurons: {sum(len(layer.neurons) for layer in self.layers)}")
        print(f"Total number of weights: {sum(len(layer.neurons[0].weights) for layer in self.layers)}")
        print(f"Total number of biases: {sum(len(layer.neurons) for layer in self.layers)}")
        
    def svm_loss(self, x, y, batch_size=None):
        """Hinge/max-margin loss with L2 regularization; returns (loss, accuracy)."""
        
        # inline dataloder: either use the full batch or sample a random subset
        if batch_size is None:
            Xb, yb = x, y
        else:
            ri = np.random.permutation(x.shape[0])[:batch_size]
            Xb, yb = x[ri], y[ri]
        # Wrap each input feature in a ``value`` so gradients can flow back through it.
        inputs = [list(map(value, xrow)) for xrow in Xb]
        
        scores = list(map(self, inputs))
        
        # svm "max-margin" loss: max(0, 1 - y * score), averaged over the batch
        losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        
        # regularization
        lam = 0.0001
        reg_loss  = lam * sum((p*p for p in self.parameters()))
        total_loss = data_loss + reg_loss
        
        # A prediction counts as correct when its sign matches the label's sign.
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)
    
    def mse_loss(self, x, y):
        """Mean squared error plus a small L2 penalty on the parameters."""
        scores = list(map(self, x))
        # mse
        loss  = sum((s-y_pred)**2 for s,y_pred in zip(scores, y))
        # regularization
        lam = 0.00001
        reg_loss  = lam * sum(p**2 for p in self.parameters())
        
        total_loss = (loss + reg_loss) * (1.0 / (2*len(scores)))
        return total_loss
        
    def fit(self, x, y, epochs):
        """Train with plain gradient descent, decaying the learning rate each epoch."""
        for epoch in range(epochs):
            # forward pass: compute the current loss
            loss = self.mse_loss(x, y)
            
            # setting the gradient to zero for each parameter
            self.zero_grad()
            # backward pass: accumulate new gradients
            loss.backward()
            
            # update parameters with a learning rate that shrinks over time
            alpha = 1.0 - 0.9*epoch/100
            for param in self.parameters():
                param.data -= alpha * param.gradient
               
            # periodic progress log
            if epoch % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs} - Loss: {loss.data:.4f}')
                
    def predict(self, x):
        """Return the raw numeric outputs for a list of inputs."""
        return [self(xi).data for xi in x]