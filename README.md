# scalarflow
andre -> part
Value = value
_children = _parents
_op = _operation
grad  = gradient
draw_dot() = draw_graph()

# vizorg
<pre>
# now we need a way to visualize the data structure

from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  # recursively builds the graph by looking at all the parents of each node
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for parent in v._prev:
        edges.add((parent, v))
        build(parent)
  build(root)
  return nodes, edges

def draw_graph(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  nodes, edges = trace(root)
  
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data: %.4f | grad: %.4f }" % (n.label, n.data, n.gradient), shape='record', style='filled', fillcolor='lightblue')
    # operation nodes are pseudo nodes, just expressed as nodes as node for connecting to actual nodes
    if n._operation:
      # if this value is a result of some operation, create an operation node for it
      dot.node(name = uid + n._operation, label = n._operation, shape='circle', style='filled', fillcolor='orange')
      # and connect this node to it
      dot.edge(uid + n._operation, uid, style='dashed', color='orange')

  for n1, n2 in edges:
    # connect n1 to the operation node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._operation, style='dashed', color='orange')

  return dot
</pre>