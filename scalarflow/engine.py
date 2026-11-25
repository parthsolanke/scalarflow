"""Core automatic-differentiation engine for ScalarFlow.

Every scalar in a computation is represented by a ``value`` object. The object
stores its numeric ``data``, the set of parent values that produced it, the
operation that created it, and a ``_backward`` closure that knows how to push
the gradient from this node to its parents. Calling ``backward`` on an output
walks the recorded graph in reverse topological order and applies the chain rule.
"""

from graphviz import Digraph
import math

class value:
    """A single scalar node in a dynamically built computation graph."""

    def __init__(self, data, _parents=(), _operation='', label=''):
        # Numeric payload of this node.
        self.data = data
        # Parent nodes that were combined to produce this value.
        self._prev = set(_parents)
        # Human-readable operation name used by the graph drawing helpers.
        self._operation = _operation
        # Optional label shown when rendering the graph.
        self.label = label
        # Gradient of the final output with respect to this node, accumulated in place.
        self.gradient = 0
        # No-op closure; overwritten by each operation that builds a node.
        self._backward = lambda: None
        
    def __repr__(self):
        return f'value(data = {self.data}, grad = {self.gradient})'
    
    '''<overloading operators>'''
    def __add__(self, other):
        # Wrap plain numbers so the rest of the method can treat them as values.
        other = other if isinstance(other, value) else value(other)
        sum = value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Addition distributes the incoming gradient equally to both operands.
            self.gradient += sum.gradient
            other.gradient += sum.gradient
        sum._backward = _backward
        
        return sum
    
    def __radd__(self, other): # reverse addition (other + self)
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other): # reverse subtraction (other - self)
        return other + (-self)
    
    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        product = value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Multiplication: each operand receives the other operand's value times the gradient.
            self.gradient += other.data * product.gradient
            other.gradient += self.data * product.gradient
        product._backward = _backward
        
        return product
    
    def __rmul__(self, other): # reverse multiplication (other * self)
        return self * other
    
    def __pow__(self, other):
        # Only constant numeric exponents are supported.
        assert isinstance(other, (int, float)), 'only supporting int and float powers'
        out = value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            # Derivative of x**n is n * x**(n-1).
            self.gradient += (other * self.data**(other-1)) * out.gradient    
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other): # reverse division (other / self)
        return other * self**-1
    
    def exp(self):
        x = self.data
        e = value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            # Derivative of exp(x) is exp(x) itself.
            self.gradient += e.data * e.gradient
        e._backward = _backward
        
        return e
    '''</overloading operators>'''
    
    '''<activation functions>'''
    def tanh(self):
        # tanh(x) computed from exponentials: (e^2x - 1) / (e^2x + 1).
        x = self.data
        n = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        t = value(n, (self, ), 'tanh')
        
        def _backward():
            # Derivative of tanh(x) is 1 - tanh(x)^2.
            self.gradient += (1 - n**2) * t.gradient
        t._backward = _backward
        
        return t
    
    def relu(self):
        # Rectified linear unit: max(0, x).
        out = value(0 if self.data < 0 else self.data, (self, ), 'relu')
        
        def _backward():
            # Gradient flows only where the output is positive.
            self.gradient += (out.data > 0) * out.gradient 
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        x = self.data
        s = value(1/(1 + math.exp(-x)), (self, ), 'sigmoid')
        
        def _backward():
            # Derivative of sigmoid(x) is s * (1 - s).
            self.gradient += (s.data * (1 - s.data)) * s.gradient
        s._backward = _backward
        
        return s
    '''</activation functions>'''
    
    '''<backpropagation>'''
    def backward(self):
        """Backpropagate from this node (assumed to be the output) to all ancestors."""
        # topological sort in a graph; children are appended after their parents
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        
        # go one variable at a time and apply chain rule to it
        self.gradient = 1
        for node in reversed(topo):
            node._backward()
    '''</backpropagation>'''
    
    '''<visualization methods>'''
    def _trace(_root):
        """Collect all nodes reachable from ``_root`` plus the edges between them."""
        # builds a set of all nodes and edges in a graph
        _nodes, _edges = set(), set()
        # recursively builds the graph by looking at all the parents of each node
        def _build(v):
            if v not in _nodes:
                _nodes.add(v)
                for parent in v._prev:
                    _edges.add((parent, v))
                    _build(parent)
        _build(_root)
        return _nodes, _edges
    
    def draw_graph(self):
        """Render the computation graph as a Graphviz ``Digraph`` (left to right)."""
        _dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
        nodes, edges = self._trace()
        
        for n in nodes:
            uid = str(id(n))
            # for any value in the graph, create a rectangular ('record') node for it
            _dot.node(name = uid, label = "{ %s | data: %.4f | grad: %.4f }" % (n.label, n.data, n.gradient), shape='record', style='filled', fillcolor='lightblue')
            # operation nodes are pseudo nodes, just expressed as nodes as node for connecting to actual nodes
            if n._operation:
                # if this value is a result of some operation, create an operation node for it
                _dot.node(name = uid + n._operation, label = n._operation, shape='circle', style='filled', fillcolor='orange')
                # and connect this node to it
                _dot.edge(uid + n._operation, uid, style='dashed')

        for n1, n2 in edges:
            # connect n1 to the operation node of n2
            _dot.edge(str(id(n1)), str(id(n2)) + n2._operation, style='dashed')

        return _dot
    '''</visualization methods>'''