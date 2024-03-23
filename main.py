from positron_builder import Value, label
from graphviz import Digraph
from typing import Tuple, Set

import os

def main():
  a = Value(2.0, 'a')
  b = Value(-3.0, 'b')
  c = Value(10.0, 'c')
  e = label(a * b, 'e')
  d = label(e + c, 'd')
  f = Value(-2.0, 'f')
  L = label(d*f, 'L')

  graph = draw_dot(L, "TB")

  generate_computation_graph(graph)

def trace(root: Value):
  """
  Returns the nodes and edges of neighbor values used to form a graph
  """
  nodes = set()
  edges = set()
  
  def build(value: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    if value not in nodes:
      nodes.add(value)

      for child in value._prev:
        edges.add((child, value))
        build(child)

  build(root)
  
  return nodes, edges

def draw_dot(root, drawing_direction='TB'):
  """
  Draw the computation graph from the given node
  """
  dot = Digraph(format='svg', graph_attr={'rankdir': drawing_direction}) # TB top to bottom

  nodes, edges = trace(root)

  for node in nodes:
    uid = str(id(node))

    dot.node(
      name = uid,
      label = "{ %s | d: %.4f | g: %.4f }" % (node.label, node.data, node.gradient),
      shape = 'record',
    )

    if node._operator:
      dot.node(name = uid + node._operator, label = node._operator)
      dot.edge(uid + node._operator, uid)

  for neighbor1, neighbor2, in edges:
    dot.edge(str(id(neighbor1)), str(id(neighbor2)) + neighbor2._operator)

  return dot

def generate_computation_graph(dot, output_filename="./visualization/computation_graph", view=True):
  unique_filename = output_filename
  counter = 1

  while os.path.exists(f'{unique_filename}.svg'):
      unique_filename = f"{output_filename}_{counter}"
      counter += 1

  dot.render(unique_filename, view=True)

if __name__ == "__main__":
  main()
